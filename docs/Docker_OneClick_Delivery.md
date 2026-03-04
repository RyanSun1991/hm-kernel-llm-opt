# HMOPT Docker 一键交付方案 / HMOPT Docker One-Click Delivery

## 关键变更 / Key Change

现在是**单镜像单容器**方案：
- 仅使用 `kernel.dockerhub.rnd.huawei.com/hmci-docker-image:v3-4.2` 作为基础镜像。
- Neo4j 直接安装并运行在 hmopt 容器内，不再依赖单独 `NEO4J_IMAGE`。

Single-container mode is used now:
- Base image: `kernel.dockerhub.rnd.huawei.com/hmci-docker-image:v3-4.2`
- Neo4j is installed and started inside hmopt container.

---

## 1) 一次性配置 / One-time setup

```bash
cp .env.docker.example .env.docker
```

至少配置：
- `KERNEL_REPO_PATH`
- `HMOPT_LLM_BASE_URL`
- `HMOPT_LLM_API_KEY`

---

## 2) 一键启动 / Start

```bash
bash scripts/docker_oneclick.sh up
```

这会：
1. 构建 hmopt 镜像（内部基础镜像 + pip 华为源）
2. 启动单个 hmopt 容器
3. 容器内自动启动 Neo4j（容器监听地址已设置为 `0.0.0.0`；容器端口 7474/7687，对外映射宿主 7475/7688）

---

## 3) 索引与服务 / Index + services

MCP/API 端口映射：宿主 7332 -> 容器 7331；宿主 7334 -> Git MCP 容器 7334；宿主 8001 -> 容器 8000。
MCP/API 端口映射：宿主 7332 -> 容器 7331；顺序思考 MCP 宿主 7334 -> 容器 7333；宿主 8001 -> 容器 8000。
Neo4j 端口映射：宿主 7475/7688 -> 容器 7474/7687。
默认密码：`@huawei2026`（首次初始化数据目录时生效）。
Git MCP 默认仓库可由 `HMOPT_GIT_MCP_REPOSITORY` 配置（默认 `/workspace/kernel`）。
Neo4j 数据默认挂载到 host：`./data/neo4j/{data,logs,plugins}`，`down` + `up-prebuilt` 后数据会保留。
APOC 优先从 `/app/libs` 安装；若仓库挂载覆盖导致找不到，则回退使用镜像内 `/opt/hmopt-libs`。

```bash
bash scripts/docker_oneclick.sh index
bash scripts/docker_oneclick.sh mcp

# 启动独立 Git MCP（streamable-http）
bash scripts/docker_oneclick.sh git-mcp
bash scripts/docker_oneclick.sh seq-mcp
bash scripts/docker_oneclick.sh build-mcp
bash scripts/docker_oneclick.sh oneclick
bash scripts/docker_oneclick.sh api
```

如需等价于你原来的：
`python -m hmopt.cli index-kernel --repo-path ... --compile-commands-dir ...`
有两种方式：

1) 固定路径（写 `.env.docker`）：
```bash
INDEX_REPO_PATH=/workspace/kernel
INDEX_COMPILE_COMMANDS_DIR=/workspace/kernel
```

2) 每次临时传参（不用改配置文件）：
```bash
bash scripts/docker_oneclick.sh index --repo-path /workspace/kernel --compile-commands-dir /workspace/kernel
```

脚本也支持把常见 host 路径自动映射到容器路径：
- `${KERNEL_REPO_PATH}/...` -> `/workspace/kernel/...`
- `${PWD}/...` -> `/app/...`

> 说明：容器只能读取已挂载的路径；未挂载的任意 host 目录无法直接被容器读取。


如果你传的是 host 绝对路径（如 `/home/...`），它必须位于已挂载目录内；否则容器内不可见。

`up` 启动时默认会挂载：
- `KERNEL_REPO_PATH -> /workspace/kernel`
- `KERNEL_REPO_PATH -> 容器内同名绝对路径`
- `KERNEL_WORKSPACE_PATH -> 容器内同名绝对路径`（建议指向同时包含 `hm-verif-kernel` 与 `build_tools` 的根目录）

对于“compile_commands 在另一台机器生成，前缀不同”的场景（如 `/home/ryan/...` vs `/home/irtos/...`），
可在 `.env.docker` 设置路径别名（无需重生成 compile_commands）：

```bash
KERNEL_REPO_PATH=/home/irtos/.../hm-verif-kernel
KERNEL_WORKSPACE_PATH=/home/irtos/.../hongmeng
HMOPT_PATH_ALIAS=/home/ryan/.../hongmeng:/home/irtos/.../hongmeng
```

然后重启容器使挂载与别名生效：
```bash
bash scripts/docker_oneclick.sh down && bash scripts/docker_oneclick.sh up-prebuilt
```

---

## 4) 离线交付（不让同事 build） / Offline handoff (no teammate build)

发布机：
```bash
bash scripts/docker_oneclick.sh package-images
```

目标机：
```bash
cp .env.docker.example .env.docker
bash scripts/docker_oneclick.sh load-images
bash scripts/docker_oneclick.sh up-prebuilt
```

> 若目标机是 podman 的 docker 兼容层，`docker load` 可能把镜像名显示成 `localhost/local:latest`。
> `load-images` 已自动尝试重打标签到 `${HMOPT_IMAGE:-hmopt:local}`，可直接继续 `up-prebuilt`。

---

## 5) 常见问题 / Common issues


### Host可访问但Docker不可访问 Neo4j 源 / Host can access but Docker build cannot

如果 host 上 `curl https://debian.neo4j.com/...` 正常，但 Docker build 超时，推荐使用**host预下载 + Docker离线安装**：

```bash
# 1) 在host下载 neo4j deb 到仓库目录
bash scripts/docker_oneclick.sh prepare-neo4j-offline

# 2) 确认离线模式
echo "NEO4J_INSTALL_MODE=offline" >> .env.docker

# 3) 再执行构建
bash scripts/docker_oneclick.sh up
```

离线模式会优先使用 `docker/neo4j-offline/` 下 host 下载的 `neo4j*.deb` + `cypher-shell*.deb`，避免 Docker build 阶段访问 `debian.neo4j.com`。



### Neo4j key 下载证书报错（curl: (60) self-signed certificate）

`bash scripts/docker_oneclick.sh prepare-neo4j-offline` 已内置自动回退：
- 先尝试正常 TLS 校验下载 key
- 失败后自动 `curl -k` 重试

因此在新机器企业代理/自签证书环境下无需手工改命令。

### Neo4j apt 网络问题（你遇到的 timeout）/ Neo4j apt timeout

当前 Dockerfile 已按你主机命令实现以下步骤：
- `curl ... neotechnology.gpg.key`
- 添加 `debian.neo4j.com` apt 源
- 写入 `Acquire::https::debian.neo4j.com::Verify-Peer "false";`

并支持通过 `.env.docker` 传参：

```bash
NEO4J_REPO_URL=https://debian.neo4j.com
NEO4J_VERIFY_PEER=false
```

If your network still cannot connect to `debian.neo4j.com:443`, this is a connectivity route/firewall issue rather than cert verification only.


### MCP 返回 421 Invalid Host header

这是 FastMCP 的 Host 头校验触发（DNS rebinding protection）。
在 `.env.docker` 中设置允许的 Host：

```bash
HMOPT_MCP_ALLOWED_HOSTS=10.123.104.145:7331,10.123.104.145:*,localhost:*,127.0.0.1:*
```

如需临时“放开全部 Host 校验”（不推荐生产环境）：

```bash
HMOPT_MCP_DISABLE_HOST_CHECK=1
```

修改后重启容器：

```bash
bash scripts/docker_oneclick.sh down
bash scripts/docker_oneclick.sh up-prebuilt
```

### BuildKit/buildx missing
脚本已自动回退到经典 builder。若需安装：
```bash
sudo apt-get update
sudo apt-get install -y docker-buildx-plugin
```

### x509 certificate
可用：
```bash
bash scripts/docker_oneclick.sh insecure-registry
```

### mirror 返回 HTML / manifest 异常
在 `.env.docker` 设置：
```bash
HMOPT_BASE_IMAGE=kernel.dockerhub.rnd.huawei.com/hmci-docker-image:v3-4.2
HMOPT_BASE_IMAGE_CANDIDATES=镜像A,镜像B,python:3.10-slim
```



### Build MCP（跨容器触发内核构建）

新增 `hmopt-build-mcp` 服务，默认端口 `7335`，用于在 MCP 容器内调用宿主 Docker，再到另一个 build 容器执行命令。

推荐模式：
- `HMOPT_BUILD_MCP_MODE=exec`（默认）
- `HMOPT_BUILD_MCP_RUNNER_CONTAINER=<你的构建容器名>`
- `HMOPT_BUILD_MCP_PROJECT_PATH=<构建容器内项目路径>`
- `HMOPT_BUILD_MCP_SIGN_WORKSPACE=<构建容器内 hm-CI 路径>`

该服务提供两个 MCP tool：
- `kernel_build_trigger`：执行 `kernel/hongmeng/build/hm_scripts/build.sh ...`
- `kernel_sign_trigger`：执行 `do_pack_hione_trunk.sh`

调用样例（逻辑映射）：
- `device=charlotte,nashville,changsha` -> `bootimage-*`
- `profile=release|debug` -> 默认 defconfig 自动映射（可手动覆盖）

注意：`hmopt-build-mcp` 通过挂载 `/var/run/docker.sock` 控制宿主 Docker，请仅在受信环境使用。
