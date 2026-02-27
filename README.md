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

Outputs (DB + artifacts + reports) are stored under `data/`.

For OpenCode MCP integration details, see `docs/OpenCode_MCP_Integration_Guide.md`.

For Docker one-click local indexing + OpenCode integration (works with docker compose and docker-only fallback), see `docs/Docker_OneClick_Delivery.md`.

## Repository Layout

See `docs/architecture.md` for the full framework design.
