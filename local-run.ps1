docker run -d `
    --name one-api `
    --restart always `
    -p 3001:3000 `
    -e TZ=Asia/Shanghai `
    -v one-api:/data `
    one-api:latest