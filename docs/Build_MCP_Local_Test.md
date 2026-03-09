# Build MCP 本地可运行测试与参数说明

## 1) 一键本地测试（不依赖真实 kernel 源码）

仓库已提供脚本：

```bash
bash scripts/test_build_mcp.sh
```

该脚本会：
1. 启动一个 mock build 容器；
2. 启动 Build MCP 服务（`127.0.0.1:7335`）；
3. 通过 `POST /tools/call` 调用：
   - `kernel_build_trigger`
   - `kernel_sign_trigger`（即 package/sign 流程）
4. 校验两个调用的 `returncode == 0`。

---

## 2) 真实环境如何启动 Build MCP

```bash
# 推荐 docker compose
bash scripts/docker_oneclick.sh build-mcp
```

关键环境变量（`.env.docker`）：

```bash
HMOPT_BUILD_MCP_MODE=exec
HMOPT_BUILD_MCP_RUNNER_CONTAINER=<你的build容器名，必须是 docker ps 可见的容器名，不是镜像名>
HMOPT_BUILD_MCP_PROJECT_PATH=<build容器内kernel工程路径>
HMOPT_BUILD_MCP_SIGN_WORKSPACE=<build容器内hm-CI路径>
```



### 2.1 exec 模式 vs run 模式（避免 No such container）

- `exec` 模式：`HMOPT_BUILD_MCP_RUNNER_CONTAINER` 必须是**正在运行的容器名/ID**。
  - 先检查：`docker ps --format "{{.Names}}" | grep <container_name>`
- 如果你手里只有镜像（例如 `ryan-hione-img` 出现在 `docker images`），请改用 `run` 模式：

```bash
HMOPT_BUILD_MCP_MODE=run
HMOPT_BUILD_MCP_RUNNER_IMAGE=ryan-hione-img:latest
HMOPT_BUILD_MCP_PROJECT_PATH=/work/trunk_new
HMOPT_BUILD_MCP_PROJECT_MOUNT=/home/ryan/code/scratch/tongkun/hione/work/trunk_new
```

`run` 模式会自动 `docker run --rm ...` 启动临时容器执行 build。


### 2.2 你这个路径模型（host /work 映射 + 代码子路径）

按你的描述可直接配置为：

```bash
HMOPT_BUILD_MCP_MODE=run
HMOPT_BUILD_MCP_RUNNER_IMAGE=ryan-hione-img:latest

# host 路径（可配置）
HMOPT_BUILD_MCP_HOST_WORKDIR=/home/ryan/code/scratch/tongkun/hione/work
# container 映射根路径（可配置）
HMOPT_BUILD_MCP_CONTAINER_WORKDIR=/work
# 进入 docker 后 cd /work/<代码path> （可配置）
HMOPT_BUILD_MCP_CODE_SUBPATH=trunk_new

# lz4 安装策略：
# auto: run 模式自动安装；exec 模式默认不装（认为容器已准备好）
# always: 总是尝试安装；never: 从不安装
HMOPT_BUILD_MCP_INSTALL_LZ4=auto
```

触发 build 时可不再传 `project_path`，Build MCP 会自动解析到：
- `project_path = /work/trunk_new`

并执行：
1. `cd /work/trunk_new`
2. 根据策略安装 `lz4`（`auto` 且 `run` 模式会安装）
3. 运行 `kernel/hongmeng/build/hm_scripts/build.sh ...`

---

## 3) 如何传参触发 build

### 3.1 release + charlotte

```bash
curl -X POST http://127.0.0.1:7335/tools/call \
  -H 'Content-Type: application/json' \
  -d '{
    "tool": "kernel_build_trigger",
    "arguments": {
      "device": "charlotte",
      "profile": "release",
      "project_path": "/home/xxx/hione/work/trunk_new/kernel/hongmeng",
      "user": "delivery",
      "modem": "full",
      "target_perf": true,
      "toolchain": "",
      "timeout_s": 7200
    }
  }'
```

### 3.2 debug + nashville

```bash
curl -X POST http://127.0.0.1:7335/tools/call \
  -H 'Content-Type: application/json' \
  -d '{
    "tool": "kernel_build_trigger",
    "arguments": {
      "device": "nashville",
      "profile": "debug",
      "project_path": "/home/xxx/hione/work/trunk_new/kernel/hongmeng"
    }
  }'
```

参数说明：
- `device`：`charlotte|nashville|changsha|bootimage-xxx`
- `profile`：`release|debug`（自动映射默认 defconfig）
- `defconfig`：可选，传入可覆盖默认映射
- `project_path`：可不传；不传时读取 `HMOPT_BUILD_MCP_PROJECT_PATH`
- `timeout_s`：执行超时秒数

---

## 3.3 OpenCode `McpError -322001 request timeout` 的根因与配置

你传给 tool 的 `timeout_s=1800` 只是 **Build MCP 服务端执行超时**，不是 OpenCode 的 MCP 客户端等待时长。

- 服务端：`subprocess.run(..., timeout=timeout_s)`，默认 build `7200s` / sign `1800s`。
- 客户端（OpenCode）：看 `opencode.json` 中该 MCP 项的 `timeout`（单位毫秒）。

如果 OpenCode 的 `timeout` 只有 120000（2 分钟）或 30000（30 秒），会先于服务端超时并报：
`McpError -322001 : request timeout`。

建议：给 build MCP 单独配置更大的 timeout（例如 30 分钟）。

```json
{
  "mcp": {
    "hmopt_build_mcp": {
      "type": "remote",
      "enabled": true,
      "url": "http://127.0.0.1:7335/mcp",
      "headers": {
        "Authorization": "Bearer {env:HMOPT_MCP_SERVER_API_KEY}"
      },
      "timeout": 1800000
    }
  }
}
```

并确保 tool 参数也匹配：
- `kernel_build_trigger.timeout_s >= opencode_timeout_ms/1000`
- 常见组合：`timeout_s=3600`，`timeout=3700000`

如果 OpenCode 已把该 tool 标成 invalid，通常重启 OpenCode 或重新加载 MCP 配置后可恢复。

## 3.4 Build MCP 异步任务模式（避免 OpenCode 长请求超时）

如果客户端仍存在 2~5 分钟级别的超时限制，建议改为异步调用：

1) 提交任务（立即返回 `task_id`）

```bash
curl -X POST http://127.0.0.1:7335/tools/call   -H 'Content-Type: application/json'   -d '{
    "tool": "kernel_build_trigger_async",
    "arguments": {
      "device": "charlotte",
      "profile": "release",
      "timeout_s": 7200
    }
  }'
```

返回示例：

```json
{
  "result": {
    "tool": "kernel_build_trigger_async",
    "content": {
      "task_id": "2a8d...",
      "status": "pending",
      "kind": "kernel_build_trigger"
    }
  }
}
```

2) 轮询任务状态

```bash
curl -X POST http://127.0.0.1:7335/tools/call   -H 'Content-Type: application/json'   -d '{
    "tool": "kernel_build_status",
    "arguments": {"task_id": "2a8d..."}
  }'
```

状态字段：
- `pending`：已提交
- `running`：执行中
- `succeeded`：完成，`result` 内含 returncode/stdout/stderr
- `failed`：失败，`error` 给出原因

这样 OpenCode 侧只需处理短请求（提交+轮询），不需要维持一个超长阻塞调用。

## 4) 如何传参触发 package/sign

```bash
curl -X POST http://127.0.0.1:7335/tools/call \
  -H 'Content-Type: application/json' \
  -d '{
    "tool": "kernel_sign_trigger",
    "arguments": {
      "device": "charlotte",
      "timeout_s": 1800
    }
  }'
```

设备与签名用户映射：
- `charlotte -> aln_user`
- `nashville -> nsv_user`
- `changsha -> changsha_user`

会执行：

```bash
bash do_pack_hione_trunk.sh <mapped_user> <bootimage-target>
```
