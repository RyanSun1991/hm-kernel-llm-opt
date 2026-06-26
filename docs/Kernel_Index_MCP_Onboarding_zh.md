# HMOPT Kernel Index MCP 服务接入指南（中文版）

本文档面向需要把 **HMOPT 内核代码索引 MCP 服务** 接入到自己 AI 设计文档工具（Claude Code / Cursor / Continue.dev / 其他支持 MCP 的客户端）的开发者。完成本文档后，你的 AI 在写鸿蒙内核相关的设计文档时，能直接调用真实的 clangd + 向量 + Neo4j 索引来验证函数签名、追溯调用链、批量取函数体——而不是凭训练数据"猜"。

---

## 一、能力总览

启动后你会得到 5 个可被 AI 调用的 MCP 工具。前 3 个是「一次性"问一个具体问题"」工具，后 2 个是「先看图再取代码」的两步式工具——后者深度可达 6 层、调用站点带 `file:line`，是写依赖链 / 影响范围 / 优化分析类设计文档的主力。

| 工具名 | 适用场景 | 是否带函数体 |
|---|---|---|
| `kernel_index_code` | 通用语义检索，"实现了什么 / 在哪里" | ✅ |
| `kernel_symbol_graph` | 单次问 caller/callee 图（深度 ≤ 4） | ✅ |
| `kernel_hotspot_context` | 与运行时热点关联的代码上下文 | ✅ |
| `kernel_call_chain` | **结构化调用链**（深度 ≤ 6，带 call-site `file:line`） | ❌（只返图） |
| `kernel_get_snippets` | **批量取函数体**（带 budget / truncation 报告） | ✅ |

完整参数与使用范式见第 [八](#八mcp-工具能力清单) / [九](#九设计文档写作推荐查询协议两步法) 章。

---

## 二、端口拓扑（必读，避免接入时踩坑）

`scripts/docker_oneclick.sh` 里启动容器时的端口映射如下（`-p 宿主端口:容器端口`）：

| 服务 | 容器端口 | **宿主机访问端口** | 备注 |
|---|---|---|---|
| 主 Kernel Index MCP | 7331 | **7332** ⚠️ | 容器内是 7331，**外部 AI 客户端要连 7332** |
| Sequential Thinking MCP | 7333 | 7333 | 顺序思考 |
| Build MCP | 7335 | 7335 | 触发 kernel build |
| Auto-Test MCP | 7336 | 7336 | hdc / 自动化测试 |
| Neo4j HTTP | 7474 | 7475 | 图数据库 UI |
| Neo4j Bolt | 7687 | 7688 | 图数据库 Bolt 协议 |
| REST API | 8000 | 8001 | `/runs` / `/health` 等 |

> **最常见的接入错误：** 在你的 AI 客户端里把 MCP URL 配成 `http://<host>:7331/mcp/`。容器内是 7331，但**从你笔记本/IDE 访问时一定是 7332**。如果你在容器里 `docker exec` 进去手测，那才是 7331。

---

## 三、前置条件

1. 一台能跑 Docker 的 Linux 机器（建议 16 GB 内存以上，索引大型仓库时高内存有帮助）。
2. 待索引的内核源码目录可访问。
3. 一份 `compile_commands.json`（无则按本指南第 [六](#六生成-compile_commandsjson) 章用 Yocto 编译日志生成）。
4. LLM API 端点（用于 embedding / 部分检索增强）；只要兼容 OpenAI Chat Completions 协议即可。
5. 你的 AI 客户端支持 **MCP streamable-http** 协议（Claude Code / Cursor / Continue.dev / Cline 等都支持）。

---

## 四、克隆仓库

```bash
git clone ssh://git@rnd-gitlab-ca-y.huawei.com:2222/hongmeng/hm-kernel-llm-opt.git -b br_opencode
cd hm-kernel-llm-opt
```

---

## 五、配置环境变量

```bash
cp .env.docker.example .env.docker
vim .env.docker
```

按下表填关键变量。**没注释「可选」的都必须填**：

```bash
# === 路径 ===
PROJECT_REPO_PATH=/path/to/your/project/trunk/
# 例：/home/ryan/code/scratch/tongkun/hione/work/trunk

KERNEL_REPO_PATH=/path/to/your/kernel/source
# 例：/home/ryan/code/scratch/tongkun/hione/work/trunk/kernel/hongmeng/hm-verif-kernel

# === LLM ===
HMOPT_LLM_BASE_URL=http://<your-llm-host>:<port>/v1
HMOPT_LLM_API_KEY=<your-api-key>

# === MCP 服务鉴权（可选，强烈建议生产场景启用）===
# 留空 = 任何来源都能调用 MCP；填了之后所有 /tools/call 与 /mcp/* 请求需带
# Authorization: Bearer <key>
HMOPT_MCP_SERVER_API_KEY=

# === 以下仅在你需要 build / 自动测试 MCP 时配，纯查询索引可全部留空 ===
HMOPT_BUILD_MCP_RUNNER_CONTAINER=<你的 build 容器 ID，例 f291c3fb187f>
HMOPT_BUILD_MCP_HOST_WORKDIR=<build 容器内的工作目录，例 /home/ryan/code/scratch/tongkun/hione/work/>
HMOPT_BUILD_MCP_CODE_SUBPATH=<build 容器子路径，例 trunk/>
HMOPT_BUILD_MCP_SIGN_WORKSPACE=<内核打包目录，例 hm-CI/>

HMOPT_BUILD_PASSWD=<域账号密码>
HMOPT_BUILD_USERNAME=<域账号用户名>
HMOPT_BUILD_TARGET_DEVICE=bootimage-nashvilleoh

HMOPT_AUTO_TEST_TARGET=<hdc 设备 ID，例 2LQ0223A31014882>
```

---

## 六、生成 `compile_commands.json`

clangd 索引必须有 `compile_commands.json`。两种来源：

### 6.1 已经有 → 跳过本节

确保 `compile_commands.json` 存在于内核仓库某一目录下（通常是子模块根，比如 `<kernel_repo>/sysmgr/compile_commands.json`），记下这个目录路径，下一章会用到。

### 6.2 从 Yocto 编译日志生成

仓库自带 `scripts/parse_compilelog.py`，可以从 Yocto 编译日志（`do_compile.NNNN`）反推 `compile_commands.json`：

```bash
python scripts/parse_compilelog.py \
  -i <yocto_build_log> \
  --host-prefix <宿主机源码前缀> \
  --docker-trunk <容器内源码前缀> \
  --map <build_module>=<原始代码绝对路径>
```

参数说明：
- `-i` Yocto 编译日志（`temp/log.do_compile.<jobid>`）
- `--host-prefix` 宿主机看到的源码根（你做修改的目录）
- `--docker-trunk` 容器内看到的源码根（与 `PROJECT_REPO_PATH` 的容器侧映射对齐）
- `--map`（**关键**）模块名到**真实源码路径**的映射。Yocto 编译日志里的路径是构建中转目录（如 `git-r0/`），不能直接用，必须通过 `--map` 把它指回你日常修改的源码目录，否则 clangd 找不到文件。可重复多次。

完整示例：

```bash
python scripts/parse_compilelog.py \
  -i ~/code/scratch/tongkun/hione/work/trunk/kernel/hongmeng/build_tools/yocto/ng/build/tmp/work/aarch64-euler-elf/hm-sysmgr-nashvilleoh/git-r0/temp/log.do_compile.230241 \
  --host-prefix /home/ryan/code/scratch/tongkun/hione/ \
  --docker-trunk /work/trunk \
  --map hm-sysmgr-nashvilleoh=/home/ryan/code/scratch/tongkun/hione/work/trunk/kernel/hongmeng/hm-verif-kernel
```

执行成功后会在当前目录或 `--map` 指定的源码路径下生成 `compile_commands.json`。

---

## 七、构建并启动 Docker

### 7.1 选项 A：本地构建镜像

```bash
bash scripts/docker_oneclick.sh up
```

### 7.2 选项 B：使用预编译镜像包（推荐，省 30+ 分钟构建时间）

```bash
# 把镜像包拷到本机（dist/ 目录下，或当前目录）
scp ryan@10.123.104.98:/home/ryan/hmopt_bundle.tar.gz ./

# 加载镜像 + 启动
bash scripts/docker_oneclick.sh load-images
bash scripts/docker_oneclick.sh up-prebuilt
```

### 7.3 验证容器在跑

```bash
docker ps | grep hmopt
bash scripts/docker_oneclick.sh shell    # 进入容器交互
```

---

## 八、构建代码索引

### 8.1 在容器内构建（推荐）

```bash
# 进入容器
bash scripts/docker_oneclick.sh shell

# 在容器里执行
python -m hmopt.cli index-kernel \
  --repo-path <内核源码目录> \
  --compile-commands-dir <compile_commands.json 所在目录>
```

参数说明：
- `--repo-path` 索引的代码根（一般指向你关心的子系统，如 `…/hm-verif-kernel/sysmgr/`，缩小范围能显著加速）
- `--compile-commands-dir` `compile_commands.json` 文件所在的目录（不是文件本身的路径）

示例：

```bash
python -m hmopt.cli index-kernel \
  --repo-path /home/ryan/code/scratch/tongkun/hione/work/trunk/kernel/hongmeng/hm-verif-kernel/sysmgr/ \
  --compile-commands-dir /home/ryan/code/scratch/tongkun/hione/work/trunk/kernel/hongmeng/hm-verif-kernel/sysmgr/
```

### 8.2 容器外触发（可选）

```bash
bash scripts/docker_oneclick.sh index
```
此命令会把 `.env.docker` 里 `KERNEL_REPO_PATH` 透传给容器内的 `index-kernel`。

### 8.3 构建耗时与重建时机

- 中小子系统（几千个符号）：约 5–15 分钟
- 整个 `hm-verif-kernel`：30 分钟到数小时，取决于机器
- **何时重建索引：**
  - 内核源码有变更（增量重建会基于内容哈希跳过未改变的节点，所以重跑就行）
  - 升级到引入了新边字段的版本（如 `call_site_path` / `call_site_line` 的 schema 变更——见 commit history）
  - 切换到不同的子系统/不同的 `compile_commands.json`

---

## 九、启动 MCP 服务

### 9.1 容器外一键启动所有 MCP（最常用）

```bash
bash scripts/docker_oneclick.sh mcp-all
```

后台启动主 MCP（7331/容器 → 7332/宿主）、顺序思考（7333）、build MCP（7335）、auto-test MCP（7336）。

### 9.2 仅启主 MCP

```bash
bash scripts/docker_oneclick.sh mcp
```

### 9.3 容器内手动启动（调试用）

```bash
# 进容器
bash scripts/docker_oneclick.sh shell

# 启所有
bash scripts/run_all_mcp_servers.sh

# 或仅启主 MCP
bash scripts/run_mcp_server.sh
```

### 9.4 健康检查

从你的开发机：

```bash
curl http://<docker-host>:7332/health
```

期望返回（关键字段）：

```json
{
  "status": "ok",
  "tool_names": {
    "general":  "kernel_index_code",
    "graph":    "kernel_symbol_graph",
    "hotspot":  "kernel_hotspot_context",
    "call_chain": "kernel_call_chain",
    "snippets": "kernel_get_snippets"
  },
  "mcp_mount_path": "/mcp",
  "mcp_api_key_required": false,
  "mcp_protocol_enabled": true
}
```

`mcp_protocol_enabled: false` 表示容器里没装 `mcp[cli]` 包，需检查镜像版本；`tool_names` 里少 `call_chain` / `snippets` 表示用的是旧版本镜像，需更新。

---

## 十、在你的 AI 工具里接入 MCP

主 Kernel Index MCP 暴露两套接口：

- **MCP 标准协议（streamable-http）：** `http://<docker-host>:7332/mcp/`（推荐，现代 MCP 客户端都用这个）
- **遗留 HTTP 接口：** `POST http://<docker-host>:7332/tools/call`，body：`{"tool": "<工具名>", "arguments": {...}}`（不支持 MCP 协议的客户端 / 想直接 curl 测试时用）

### 10.1 Claude Code

```bash
# 无鉴权
claude mcp add hmopt-kernel-index --transport http http://<docker-host>:7332/mcp/

# 有鉴权（HMOPT_MCP_SERVER_API_KEY 已设置）
claude mcp add hmopt-kernel-index --transport http http://<docker-host>:7332/mcp/ \
  --header "Authorization: Bearer <你的 key>"
```

### 10.2 Cursor / Continue.dev / 通用 JSON 配置

在客户端的 MCP 配置文件里（路径因客户端而异，常见有 `~/.cursor/mcp.json` / `.cursor/mcp.json` / Continue 的 `~/.continue/config.json`）加：

```json
{
  "mcpServers": {
    "hmopt-kernel-index": {
      "type": "streamable-http",
      "url": "http://<docker-host>:7332/mcp/",
      "headers": {
        "Authorization": "Bearer <你的 key，未启用鉴权可省略此字段>"
      }
    }
  }
}
```

重启客户端后，AI 应能在工具列表里看到 `kernel_index_code` / `kernel_symbol_graph` / `kernel_hotspot_context` / `kernel_call_chain` / `kernel_get_snippets` 五个工具。

### 10.3 直接 curl 测试（不接入客户端，只是验证）

```bash
curl -X POST http://<docker-host>:7332/tools/call \
  -H "Content-Type: application/json" \
  -d '{"tool":"kernel_call_chain","arguments":{"symbols":["try_to_free_pages"],"direction":"callees","depth":3}}'
```

---

## 十一、MCP 工具能力清单

下面每个工具的参数都来自 `src/hmopt/api/mcp_service.py`，与你 AI 调用时实际收到的 schema 一致。

### 11.1 `kernel_index_code` — 通用检索

参数：
- `query` (str) — 查询文本
- `scenario` (str, 可选) — `general` / `implementation` / `call_graph` / `impact_analysis` / `hotspot_debug` / `patch_planning`
- `symbols` (list[str], 可选) — focus 符号
- `runtime_hints` (str, 可选) — 运行时提示（perf / flamegraph 输出）
- `top_k` / `max_snippets` / `max_chars` / `graph_depth` (int, 可选)
- `response_format` — `markdown`（默认）/ `json`

适用于："X 函数的实现是什么"、"哪些符号和 reclaim 路径相关"。

### 11.2 `kernel_symbol_graph` — 单次取调用图

参数：`symbols`（必填）+ `query` / `top_k` / `graph_depth` (≤4) / `max_snippets` / `max_chars` / `response_format`

> ⚠️ 这个工具会**同时**返回图边和函数体，受 LLM 上下文窗口限制 `graph_depth` 被夹在 4，且尾部 snippet 会被截断。**深度 ≥ 3 的调用链分析请改用 `kernel_call_chain` + `kernel_get_snippets` 组合（见下）。**

### 11.3 `kernel_hotspot_context` — 热点关联检索

参数：`symbols` / `query` / `runtime_hints` / `top_k` / `graph_depth` / `response_format`

适用于：你已经有 hiperf / flamegraph 输出，想问"这个热点函数周围还涉及哪些代码"。

### 11.4 `kernel_call_chain` — 纯结构化调用链 ⭐

签名：

```
kernel_call_chain(
  symbols: list[str],          # 必填，根符号
  direction: "both"|"callers"|"callees" = "both",
  depth: int = 3,              # 1..6（深度 6 不再被截断）
  per_hop_limit: int = 100,    # 每层 Cypher LIMIT
  frontier_cap: int = 50,      # 每层最多展开多少符号
  edge_kinds: list[str] | None = None,  # 过滤 ["calls","uses_type","uses_macro"]
  response_format: "markdown"|"json" = "markdown"
)
```

返回结构（json 模式）：

```json
{
  "roots": ["..."],
  "direction": "callees",
  "depth_requested": 3,
  "edges": [
    {
      "src": "...", "dst": "...", "rel": "calls",
      "depth": 1, "direction": "callee",
      "call_site_path": "/k/x.c", "call_site_line": 234
    }
  ],
  "nodes": {
    "<qualname>": {
      "symbol_name": "...", "path": "...",
      "start_line": 10, "end_line": 80, "kind": "function", "depth": 1
    }
  },
  "stats": {
    "total_edges": N, "total_nodes": M,
    "depth_reached": D,
    "hops_truncated_at": [4]   // 哪些层被 per_hop_limit / frontier_cap 截断
  }
}
```

**关键能力：** 只返图、不返函数体，因此 `depth=6` 不会触发上下文截断；每条边带 `call_site_path:line`，可直接做静态 HOT/SLOW 分类的 file:line 引用。`hops_truncated_at` 非空时表明 BFS 在某层被截，需要 narrow `symbols` 后重发。

### 11.5 `kernel_get_snippets` — 批量函数体 ⭐

签名：

```
kernel_get_snippets(
  symbols: list[str],                  # 必填
  per_symbol_max_chars: int = 4000,    # 单个 snippet 上限
  total_max_chars: int | None = None,  # 总预算（可选）
  response_format: "markdown"|"json" = "markdown"
)
```

返回结构（json 模式）：

```json
{
  "snippets": [
    {
      "symbol": "...", "symbol_id": "...",
      "path": "...", "start_line": 10, "end_line": 80,
      "truncated": false, "char_count": 1234,
      "text": "..."
    }
  ],
  "missing": [
    {"symbol": "x", "reason": "not_found"},
    {"symbol": "y", "reason": "budget_hit"}
  ],
  "stats": {
    "returned": N, "missing": M,
    "truncated_count": K, "total_chars": C, "budget_hit": false
  }
}
```

**关键能力：** 批量取，不再一个个查；超 budget 的符号显式列在 `missing` 而不是静默丢失，方便客户端分批补齐。

---

## 十二、设计文档写作推荐查询协议（两步法）

写设计文档时，最常见的需求是"画出 X 函数的调用链 / 影响范围 / 热点路径"。**不要直接调 `kernel_symbol_graph` 一锅端**——它在深度 ≥ 3 时会截断函数体，得到的是不完整的图 + 不完整的代码。

正确流程：

**第 1 步：用 `kernel_call_chain` 看图的形状**

```jsonc
{
  "symbols": ["try_to_free_pages"],
  "direction": "callees",
  "depth": 3,
  "edge_kinds": ["calls"]
}
```

返回的 `nodes`（path / start_line / end_line / kind）与 `edges`（call_site_path:line）足以画出调用图骨架，并初步给每个节点打 `[HOT] / [SLOW] / [COLD] / [UNKNOWN]` 静态分类。

**第 2 步：用 `kernel_get_snippets` 批量取代码**

挑出图里需要看实现的节点（root + 每个非 COLD 的关键 callee），一次性取：

```jsonc
{
  "symbols": ["try_to_free_pages", "shrink_zone", "shrink_lruvec", "..."],
  "per_symbol_max_chars": 6000
}
```

返回的 `snippets[].text` 直接贴进设计文档的 Implementation Walkthrough / Concurrency / Error Paths 章节，每段都带 path:line。

**第 3 步：处理截断（如果发生）**

`kernel_call_chain` 的 `stats.hops_truncated_at` 如果非空：
- 单层超 `per_hop_limit` → 把 `symbols` 缩到该层父节点，提高 `per_hop_limit`（最大 500）后再发一次
- 单层超 `frontier_cap` → 把 frontier 拆成多批，多次调用 `kernel_call_chain`
- 都到了 6 层还在截 → 函数 fan-out 太大（如 syscall dispatch），降深度并在文档 Open Questions 里记一笔

`kernel_get_snippets` 的 `missing[reason=budget_hit]` 类似处理：增大 `total_max_chars`，或对剩余 symbols 单独再发一次。

---

## 十三、常见问题

**Q1：客户端连不上 `http://<host>:7331/mcp/`？**
A：宿主端口是 **7332**，不是 7331（见第二章）。

**Q2：AI 调工具时报 "tool not found: kernel_call_chain"？**
A：镜像版本太旧，没把 Phase 1-4 提交（参考 commit `428987d`）打进去。`/health` 返回的 `tool_names` 缺这两个就是该原因。

**Q3：`kernel_call_chain` 返回的 `call_site_path` 全是 null？**
A：索引是用旧 schema 建的，没有 `call_site_path` / `call_site_line` 边属性。重新跑 `index-kernel` 即可；schema 是向后兼容的，旧索引能查、只是这两个字段为空。

**Q4：索引特别慢？**
A：先缩 `--repo-path` 到具体子系统（如 `…/sysmgr/`）而不是整个 kernel；clangd 第一次冷启动慢是正常的，第二次会快很多。

**Q5：返回的 `snippets[].text` 看起来是截断的？**
A：检查每个 snippet 的 `truncated` flag 和 `stats.truncated_count`。截断了就调高 `per_symbol_max_chars`（上限 20000），或缩短 `symbols` 列表。

**Q6：AI 客户端启用了鉴权但调用 401？**
A：确认 `HMOPT_MCP_SERVER_API_KEY` 在 `.env.docker` 里和客户端 `Authorization: Bearer <key>` 一致；`/health` 的 `mcp_api_key_required: true` 表示服务端启用了鉴权。

**Q7：能不能从 Windows / Mac 笔记本连容器跑在另一台 Linux 机器上的 MCP？**
A：可以。把 `<docker-host>` 换成 Linux 机器的 IP / 域名，确保 7332 / 7333 / 7335 / 7336 等端口在宿主机防火墙放通即可。

---

## 十四、参考

- 仓库根 `CLAUDE.md` — 总览与命令清单
- `docs/Docker_OneClick_Delivery.md` — Docker 一键交付细节
- `docs/architecture.md` — 整体架构
- `src/hmopt/api/mcp_service.py` — MCP 工具实现的权威源（参数若与本文档不一致以代码为准）
- `src/hmopt/indexing/llamaindex_pipeline.py` — `retrieve_call_chain` / `fetch_code_snippets` 的实现
