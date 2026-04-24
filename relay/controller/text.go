package controller

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"

	"github.com/gin-gonic/gin"

	"github.com/songquanpeng/one-api/common/config"
	"github.com/songquanpeng/one-api/common/conv"
	"github.com/songquanpeng/one-api/common/ctxkey"
	"github.com/songquanpeng/one-api/common/logger"
	"github.com/songquanpeng/one-api/relay"
	"github.com/songquanpeng/one-api/relay/adaptor"
	"github.com/songquanpeng/one-api/relay/adaptor/openai"
	"github.com/songquanpeng/one-api/relay/apitype"
	"github.com/songquanpeng/one-api/relay/billing"
	billingratio "github.com/songquanpeng/one-api/relay/billing/ratio"
	"github.com/songquanpeng/one-api/relay/channeltype"
	"github.com/songquanpeng/one-api/relay/meta"
	"github.com/songquanpeng/one-api/relay/model"
	"github.com/songquanpeng/one-api/relay/reasoning"
)

func RelayTextHelper(c *gin.Context) *model.ErrorWithStatusCode {
	ctx := c.Request.Context()
	meta := meta.GetByContext(c)
	// get & validate textRequest
	textRequest, err := getAndValidateTextRequest(c, meta.Mode)
	if err != nil {
		logger.Errorf(ctx, "getAndValidateTextRequest failed: %s", err.Error())
		return openai.ErrorWrapper(err, "invalid_text_request", http.StatusBadRequest)
	}
	meta.IsStream = textRequest.Stream

	// map model name
	meta.OriginModelName = textRequest.Model
	textRequest.Model, _ = getMappedModelName(textRequest.Model, meta.ModelMapping)
	meta.ActualModelName = textRequest.Model

	// Log incoming client request (headers + body)
	{
		var headerBuf strings.Builder
		for k, vs := range c.Request.Header {
			headerBuf.WriteString(k + ": " + strings.Join(vs, ", ") + "\n")
		}
		bodyStr := ""
		if rawBody, ok := c.Get(ctxkey.KeyRequestBody); ok && rawBody != nil {
			bodyStr = string(rawBody.([]byte))
		}
		logger.Infof(ctx, "[relay] client request model=%s stream=%v\nheaders:\n%s\nbody:\n%s",
			meta.ActualModelName, textRequest.Stream, headerBuf.String(), bodyStr)
	}

	// For deepseek models: inject stored reasoning_content from previous turn into the
	// last assistant message so multi-turn tool-call chains satisfy DeepSeek's API requirement.
	var copilotSessionID string
	if strings.HasPrefix(meta.ActualModelName, "deepseek") {
		copilotSessionID = c.GetHeader("X-Interaction-Id")
		if copilotSessionID == "" {
			copilotSessionID = c.GetHeader("X-Agent-Task-Id")
		}
		if copilotSessionID != "" {
			injectReasoningContent(c, textRequest, copilotSessionID)
		}
	}
	// set system prompt if not empty
	systemPromptReset := setSystemPrompt(ctx, textRequest, meta.ForcedSystemPrompt)
	// get model ratio & group ratio
	modelRatio := billingratio.GetModelRatio(textRequest.Model, meta.ChannelType)
	groupRatio := billingratio.GetGroupRatio(meta.Group)
	ratio := modelRatio * groupRatio
	// pre-consume quota
	promptTokens := getPromptTokens(textRequest, meta.Mode)
	meta.PromptTokens = promptTokens
	preConsumedQuota, bizErr := preConsumeQuota(ctx, textRequest, promptTokens, ratio, meta)
	if bizErr != nil {
		logger.Warnf(ctx, "preConsumeQuota failed: %+v", *bizErr)
		return bizErr
	}

	adaptor := relay.GetAdaptor(meta.APIType)
	if adaptor == nil {
		return openai.ErrorWrapper(fmt.Errorf("invalid api type: %d", meta.APIType), "invalid_api_type", http.StatusBadRequest)
	}
	adaptor.Init(meta)

	// get request body
	requestBody, err := getRequestBody(c, meta, textRequest, adaptor)
	if err != nil {
		return openai.ErrorWrapper(err, "convert_request_failed", http.StatusInternalServerError)
	}

	// Log the body that will actually be sent to the upstream API
	outBytes, _ := io.ReadAll(requestBody)
	logger.Infof(ctx, "[relay] outgoing to upstream API body:\n%s", string(outBytes))
	requestBody = bytes.NewBuffer(outBytes)

	// do request
	resp, err := adaptor.DoRequest(c, meta, requestBody)
	if err != nil {
		logger.Errorf(ctx, "DoRequest failed: %s", err.Error())
		return openai.ErrorWrapper(err, "do_request_failed", http.StatusInternalServerError)
	}
	if isErrorHappened(meta, resp) {
		billing.ReturnPreConsumedQuota(ctx, preConsumedQuota, meta.TokenId)
		return RelayErrorHandler(resp)
	}

	// do response
	usage, respErr := adaptor.DoResponse(c, resp, meta)
	if respErr != nil {
		logger.Errorf(ctx, "respErr is not nil: %+v", respErr)
		billing.ReturnPreConsumedQuota(ctx, preConsumedQuota, meta.TokenId)
		return respErr
	}

	// Persist reasoning_content returned by the API for the next turn.
	if copilotSessionID != "" {
		if rc, exists := c.Get("reasoning_content"); exists {
			if rcStr, ok := rc.(string); ok && rcStr != "" {
				reasoning.SetReasoningContent(copilotSessionID, rcStr)
			}
		}
	}
	// post-consume quota
	go postConsumeQuota(ctx, usage, meta, textRequest, ratio, preConsumedQuota, modelRatio, groupRatio, systemPromptReset)
	return nil
}

func getRequestBody(c *gin.Context, meta *meta.Meta, textRequest *model.GeneralOpenAIRequest, adaptor adaptor.Adaptor) (io.Reader, error) {
	if !config.EnforceIncludeUsage &&
		meta.APIType == apitype.OpenAI &&
		meta.OriginModelName == meta.ActualModelName &&
		meta.ChannelType != channeltype.Baichuan &&
		meta.ForcedSystemPrompt == "" {
		// no need to convert request for openai
		return c.Request.Body, nil
	}

	// get request body
	var requestBody io.Reader
	convertedRequest, err := adaptor.ConvertRequest(c, meta.Mode, textRequest)
	if err != nil {
		logger.Debugf(c.Request.Context(), "converted request failed: %s\n", err.Error())
		return nil, err
	}
	jsonData, err := json.Marshal(convertedRequest)
	if err != nil {
		logger.Debugf(c.Request.Context(), "converted request json_marshal_failed: %s\n", err.Error())
		return nil, err
	}
	logger.Debugf(c.Request.Context(), "converted request: \n%s", string(jsonData))
	requestBody = bytes.NewBuffer(jsonData)
	return requestBody, nil
}

// injectReasoningContent looks up the stored reasoning_content for the given session and,
// if found, sets it on the last assistant message in the request that has no reasoning_content
// yet. It also refreshes c.Request.Body and the cached body so that getRequestBody's
// early-return path (which bypasses ConvertRequest) still forwards the updated payload.
func injectReasoningContent(c *gin.Context, req *model.GeneralOpenAIRequest, sessionID string) {
	rc, ok := reasoning.GetReasoningContent(sessionID)
	if !ok || rc == "" {
		return
	}
	// Find the last assistant message that has no reasoning_content yet.
	for i := len(req.Messages) - 1; i >= 0; i-- {
		if req.Messages[i].Role != "assistant" {
			continue
		}
		if conv.AsString(req.Messages[i].ReasoningContent) != "" {
			// Already has reasoning_content; nothing to do.
			break
		}
		req.Messages[i].ReasoningContent = rc
		// Rebuild the cached body so all read paths see the updated request.
		jsonData, err := json.Marshal(req)
		if err != nil {
			return
		}
		c.Set(ctxkey.KeyRequestBody, jsonData)
		c.Request.Body = io.NopCloser(bytes.NewBuffer(jsonData))
		break
	}
}
