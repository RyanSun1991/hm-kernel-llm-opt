# HM-VERIF Kernel Code & Performance Optimization Platform (LLM-Driven)

This repository contains a **code + performance analysis and optimization platform** targeting the `hm-verif-kernel` codebase.

The platform is designed as an **agentic, closed-loop pipeline**:

1) Ingest **source code** + **kernel configs** + **hardware specs**
2) Ingest **performance artifacts** (e.g., hitrace/hiperf/flamegraph/proc/klog/perf)
3) Build **static structure graphs** + **runtime call graphs**
4) Correlate profiling bottlenecks with code locations
5) Use LLM agents to propose patches, run verification, re-profile, and iterate
6) Persist all artifacts into a local database to support:
   - retrieval-augmented analysis (RAG)
   - reproducible optimization experiments
   - dataset generation for later fine-tuning / training

## Quickstart

- Configure the internal LLM API in `configs/model_server.yaml` (or set `HMOPT_LLM_API_KEY` / `HMOPT_LLM_BASE_URL`).
- Point the platform at the `hm-verif-kernel` repo path in `configs/app.yaml`.
- Run an end-to-end loop (dummy adapters by default, safe for local testing):

```bash
python3 -m hmopt.cli run --config configs/app.yaml
```

- Launch the REST API (serves `/health`, `/runs`, `/runs/{id}/metrics`, `/runs/{id}/report`):

```bash
bash scripts/run_api.sh
```

- Launch the MCP server for OpenCode/other MCP clients (`/mcp` + legacy `/tools/call`):

```bash
bash scripts/run_mcp_server.sh
```

- Launch the Git MCP server (standalone streamable-http endpoint):

```bash
bash scripts/run_git_mcp_server.sh
```

默认可通过 `HMOPT_GIT_MCP_REPOSITORY` 设置仓库根路径，这样调用 Git MCP 工具时可不显式传 `repo_path`；也可继续在每次 tool 调用中传入 `repo_path` 覆盖。

- Launch the Build MCP server (default `0.0.0.0:7335`):

```bash
bash scripts/run_build_mcp_server.sh
```

Build MCP can trigger kernel build/sign commands in another Docker container via `docker exec/run`. Configure with `HMOPT_BUILD_MCP_*` environment variables.

- Launch the Sequential Thinking MCP server (default `0.0.0.0:7333`):

```bash
bash scripts/run_seq_mcp_server.sh
```

- Launch the Auto-Test MCP server for running phone test scripts through `hdc` (default `0.0.0.0:7336`):

```bash
bash scripts/run_auto_test_mcp_server.sh
```

Auto-Test MCP exposes tool `phone_test_run` by default, supports per-test parameters from MCP client, runs `hdc shell`, and pulls result files via `hdc file recv`. By default it attempts a connect step first (`hdc connect`, with legacy `hdc tconn` fallback).
For environments that do not need explicit connect, set `connect_before_shell=false` and optionally `use_target_flag=false` to run plain `hdc shell` / `hdc file recv`.
If the device tunnel endpoint is on host (e.g. `ssh -R 8710:localhost:8710 ...`), pass `target=host.docker.internal:8710` in MCP tool arguments when calling from containerized server.

Outputs (DB + artifacts + reports) are stored under `data/`.

For OpenCode MCP integration details, see `docs/OpenCode_MCP_Integration_Guide.md`.

For Docker one-click local indexing + OpenCode integration (works with docker compose and docker-only fallback), see `docs/Docker_OneClick_Delivery.md`.

For local runnable Build MCP test and parameter examples, see `docs/Build_MCP_Local_Test.md`.

## Repository Layout

See `docs/architecture.md` for the full framework design.
