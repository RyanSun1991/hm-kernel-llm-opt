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

### Evidence-driven evolution loop (V2 local integration)

The new `hmopt.evolution` module implements the shared workflow for historical mining,
pattern curation, candidate screening, owner confirmation, agent handoffs, review gates,
paired A/B or correctness report evaluation, and journal/staging/hub knowledge.
V2 adds typed workspace evidence, owner review sheets, task-specific OpenCode
dispatch artifacts, strict IC conversion, quality overlays, checkpointed discovery,
12 research templates, and sanitized knowledge review bundles. It runs independently
of the legacy analysis stack. The included demo creates a new example Git repository
and uses explicitly synthetic measurements; it does not run a kernel or a device.

```bash
python -m pip install -r requirements-evolution.txt
PYTHONPATH=src python -m hmopt.evolution.cli demo-v2 --output /tmp/hmopt-evolution-v2-example
```

The output directory must not already exist. On PowerShell, first set
`$env:PYTHONPATH = "src"`, then run the Python command with a new local output path.
Git must be on `PATH`, or supplied via `--git-bin` before the subcommand.
With a full package installation, use `hmopt evolve` or `hmopt-evolve`.

Start with the [V2 design and delivery plan](docs/EVOLUTION_V2_IMPLEMENTATION_CN.md)
and [V2 operations guide](docs/EVOLUTION_V2_OPERATIONS_CN.md).
For a first installation and one real candidate, follow the
[step-by-step setup guide](docs/EVOLUTION_V2_FIRST_RUN_CN.md), including the existing
team delegate runtime prerequisite and a CLI-only execution path.
The [baseline design](docs/EVOLUTION_PLATFORM_DESIGN_CN.md),
[gate contract guide](docs/EVOLUTION_QUICKSTART_CN.md), and
[production roadmap](docs/EVOLUTION_ROADMAP_CN.md) explain the shared foundations.
Actual model execution, authenticated identities, trusted device collection and
native team Skill Hub publication still require deployment integrations.

### Existing analysis and device services

The legacy `run`/`index` path in this checkout references a missing trace preprocessing
module. The new evolution commands and lightweight pipeline entrypoints do not import
that path. See [the repository assessment](docs/PROJECT_UNDERSTANDING.md) for details.

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

Auto-Test MCP exposes tool `phone_test_run` by default, supports per-test parameters from MCP client, runs `hdc shell`, and pulls result files via `hdc file recv`. Recommended Docker setup: configure `HMOPT_AUTO_TEST_TARGET` in `.env.docker` and use direct `hdc -t <target> ...` without connect.
Built-in case: set `test_case=basic_swipe` to auto-push and run `scripts/phone_tests/basic_swipe.sh` on device (`remote_script` may be empty). The script performs swipe workload and aligns with legacy flow by running `hiperf record/report`; optional `extra_args=[duration_s, swipe_count]`; default result path is `/data/local/tmp/basic_swipe.result`.
Connect is disabled by default (`connect_before_shell=false`, `HMOPT_AUTO_TEST_HDC_CONNECT_MODE=none`). If needed, you can still enable connect/tconn explicitly.
If the device tunnel endpoint is on host (e.g. `ssh -R 8710:localhost:8710 ...`), pass `target=host.docker.internal:8710` in MCP tool arguments when calling from containerized server.

For direct local debugging without starting MCP service, run:
`PYTHONPATH=src python -m hmopt.api.auto_test_mcp_service --test-case basic_swipe --remote-script ""` (uses `HMOPT_AUTO_TEST_TARGET` by default).

Outputs (DB + artifacts + reports) are stored under `data/`.

For OpenCode MCP integration details, see `docs/OpenCode_MCP_Integration_Guide.md`.

For the OpenCode multi-agent workflow and the repo-backed `.opencode/` workspace, see `docs/OpenCode_JKernel_Multi_Agent_Optimization_Workflow.md` and `.opencode/README.md`.

For the one-click OpenCode pipeline entry flow, use `python3 -m hmopt.cli start-pipeline ...`, `python3 -m hmopt.cli list-pipeline-profiles`, or `bash scripts/run_opencode_pipeline.sh ...`.

For the staged one-click workflow guide, see `docs/OpenCode_One_Click_Pipeline_Guide.md`.

For Docker one-click local indexing + OpenCode integration (works with docker compose and docker-only fallback), see `docs/Docker_OneClick_Delivery.md`.

For local runnable Build MCP test and parameter examples, see `docs/Build_MCP_Local_Test.md`.

## Repository Layout

See `docs/architecture.md` for the full framework design.
