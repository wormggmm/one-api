package reasoning

import "sync"

// globalRecords stores reasoning_content keyed by session ID (e.g. mcp-session-id).
// It is used to propagate reasoning_content across turns for deepseek models,
// so that multi-turn tool-call conversations satisfy DeepSeek's requirement of
// including reasoning_content in every assistant message of the history.
var globalRecords = &store{m: make(map[string]string)}

type store struct {
	mu sync.RWMutex
	m  map[string]string
}

// GetReasoningContent returns the stored reasoning content for the given session ID.
// The second return value is false when no entry exists for that session.
func GetReasoningContent(sessionID string) (string, bool) {
	globalRecords.mu.RLock()
	defer globalRecords.mu.RUnlock()
	v, ok := globalRecords.m[sessionID]
	return v, ok
}

// SetReasoningContent stores (or replaces) the reasoning content for the given session ID.
func SetReasoningContent(sessionID, content string) {
	globalRecords.mu.Lock()
	defer globalRecords.mu.Unlock()
	globalRecords.m[sessionID] = content
}
