# AGENTS.md

This file provides guidance to Codex (Codex.ai/code) when working with code in this repository.

## Project Overview

HM-VERIF Kernel Code & Performance Optimization Platform — an LLM-driven, agentic closed-loop system for analyzing and optimizing the `hm-verif-kernel` codebase. The pipeline: ingest code/traces → build structure graphs → detect hotspots → propose patches → verify → re-profile → iterate.

## Build & Install

```bash
pip install -e .                 # editable install (creates `hmopt` CLI)
pip install -e ".[dev]"          # if dev extras are added later
```

Source layout: `src/hmopt/` (configured via `pyproject.toml` `package-dir`).

## Key Commands

### CLI (all via `hmopt` or `python -m hmopt.cli`)

```bash
# Full optimization pipeline
hmopt run --config configs/app.yaml

# Optimize with iteration budget
hmopt optimize --config configs/app.yaml --iterations 3

# Analyze existing artifacts (no live profiling)
hmopt analyze-artifacts --artifact flamegraph:path.json --artifact hiperf:path.json

# Build code index (clangd + LlamaIndex)
hmopt index-kernel --repo-path /path/to/kernel --compile-commands-dir /path

# Build runtime index for a run
hmopt index-runtime <run_id>

# Query indexes (modes: auto|code|runtime|runtime_code|graph)
hmopt query "Which function is hot?" --mode runtime

# List available pipeline profiles
hmopt list-pipeline-profiles

# Start OpenCode pipeline session
hmopt start-pipeline --profile generic_full --target <symbol>

# Resume pipeline session
hmopt resume-pipeline
```

### MCP Servers

```bash
bash scripts/run_mcp_server.sh           # Main MCP (port 7331)
bash scripts/run_git_mcp_server.sh        # Git MCP (port 7331)
bash scripts/run_build_mcp_server.sh      # Build MCP (port 7335)
bash scripts/run_seq_mcp_server.sh        # Sequential Thinking (port 7333)
bash scripts/run_auto_test_mcp_server.sh  # Auto-Test / hdc (port 7336)
bash scripts/run_all_mcp_servers.sh       # All servers
```

### REST API

```bash
bash scripts/run_api.sh   # Serves /health, /runs, /runs/{id}/metrics, /runs/{id}/report
```

### Tests

```bash
pytest tests/                       # all tests
pytest tests/test_opencode_pipeline.py  # single test file
PYTHONPATH=src pytest tests/        # if not installed editable
```

## Linting

Ruff with 100-char line length (configured in `pyproject.toml`).

```bash
ruff check src/ tests/
ruff format src/ tests/
```

## Architecture

### Three Layers

1. **Ingestion** — Code indexing (clangd LSP + LlamaIndex vectors), performance artifact parsing (hitrace/hiperf/perf/flamegraph), config/hardware capture.
2. **Analysis** — Static structure graphs (PSG/CFG/call graphs), runtime hotspot detection and ranking, code-to-trace correlation.
3. **Agentic Optimization** — LangGraph multi-agent orchestration loop.

### Source Package (`src/hmopt/`)

| Module | Purpose |
|---|---|
| `cli.py` | Typer CLI entrypoint |
| `core/` | Config loading (`AppConfig` from YAML), shared models |
| `agents/` | LangGraph agent implementations (Conductor, TraceAnalyst, Coder, Reviewer, Verifier, Profiler, SafetyGuard) |
| `orchestration/` | Pipeline runner — wires agents into the closed loop |
| `analysis/` | Hotspot detection, bottleneck classification, artifact parsers |
| `indexing/` | LlamaIndex + clangd index builders, query routing |
| `storage/` | SQLAlchemy ORM (SQLite default), artifact file store, vector embeddings |
| `api/` | FastAPI REST + MCP server endpoints (main, git, build, auto-test, sequential-thinking) |
| `mcp_server_git/` | Git MCP tool implementations |
| `opencode/` | Pipeline session init, profile loading, OpenCode integration |
| `sequential_thinking/` | Step-by-step reasoning MCP service |
| `tools/` | Shared tool utilities |

### Database Models (`storage/db/models.py`)

Run, Artifact, Metric, Hotspot, Graph, Patch, Evaluation, AgentMessage, VectorEmbedding — every optimization run is an immutable experiment with versioned artifacts.

### Configuration

- `configs/app.yaml` — main config (LLM endpoints, storage, indexing, profiling)
- `configs/model_server.yaml` — LLM server endpoints (included by app.yaml)
- `configs/workloads.yaml` — test workload definitions (included by app.yaml)
- `configs/pipeline_profiles.yaml` — named pipeline presets (generic_full, hyperhold_full, memmgr_reclaim_full, sync_review, workqueue_full, etc.)
- `.env.example` / `.env.docker.example` — environment variable templates

Config supports `includes:` for composing YAML files. LLM keys come from env vars (`HMOPT_LLM_API_KEY`, `HMOPT_LLM_BASE_URL`).

### OpenCode Agent Workbench (`.opencode/`) — the thin constitution

`.opencode/` hosts two lanes (design: `docs/Agent_Workbench_Design_EN.md`):

**Workbench lane (default).** Everyday unit of work = one role + a small selected
skill set + a lightweight task workspace.

- Default entry is `assistant`; an ordinary prompt NEVER implicitly starts a pipeline.
- The user owns routing — roles suggest handoff/consult/fork with forwardable briefs;
  only the user triggers them.
- 7 domain-free roles in `.opencode/agents/` (assistant · researcher · architect ·
  implementer · reviewer · validator · coordinator); domain knowledge lives in
  `.opencode/skills/scenario/` packs; profiles in `agents/profiles/` are preloaded
  compositions. Every role loads `skills/infra/agent-core/SKILL.md` (the base
  contract).
- Skills are discovered via `skills/_registry.yaml` (suggest ≤3 with reasons; full
  text loads only after the user confirms; ≤4 active non-core skills).
- Task truth lives in `.opencode/local/workspaces/<slug>/` (git-ignored; template in
  `.opencode/templates/workspace/`); the capsule is the only handoff/resume carrier.
- Permission ceilings are runtime-enforced by role frontmatter as pattern-scoped
  `edit` maps: each role writes only its own artifact directories, source is denied
  for every role except implementer (whose every edit asks; destructive ops denied);
  read-only bash runs freely, mutating bash asks; device/R3 MCP operations are
  contract-gated per-action. Only coordinator delegates. Skills never widen
  permissions. Execution rights ≠ claim rights: artifact status promotions
  (approved / ready-to-land / validated) have role-owned conditions.

**Pipeline lane (explicit recipes only).** `/optimize_*` commands run the strict
staged pipeline with mandatory gates:

```
intake → routing → research → plan review (GATE) → implementation → code review (GATE) → test → decision
```

Key rule: no implementation without plan review approval, no acceptance without code
review, no test verdict without stock-vs-feature A/B. Stage gates live in
`skills/infra/pipeline/` and apply only inside recipe runs. See `.opencode/CLAUDE.md`
(the `.opencode`-local mirror of this constitution) and
`.opencode/docs/harness_engineer_system.md` for enforcement details.

Golden rule: create a role only when responsibility/authority changes; a skill when
domain/method changes; a profile when a composition repeats; a workflow only for
repeatable coordination. Task truth lives in the workspace; reusable truth in Team
Memory / the Skill Hub.

## Docker

Docker-based deployment via `docker-compose`. Build MCP can trigger kernel builds in separate containers via `docker exec/run`. See `docs/Docker_OneClick_Delivery.md` for one-click setup.

## Important Docs

- `docs/architecture.md` — framework design
- `docs/pipeline.md` — pipeline flow details
- `docs/data_model.md` — database schema
- `docs/OpenCode_MCP_Integration_Guide.md` — MCP integration
- `docs/OpenCode_One_Click_Pipeline_Guide.md` — staged pipeline guide
