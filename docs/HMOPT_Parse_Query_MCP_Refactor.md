# HMOPT Parse / Query / MCP Refactor Notes

This document summarizes the recent refactor across runtime parsing, query routing, and MCP-assisted retrieval.

## Scope

Changes covered here focus on:

- Runtime artifact ingestion (flamegraph / hitrace / hiperf / sysfs)
- Query routing (runtime -> code -> graph + MCP)
- Call stack preservation and formatting
- MCP tool integration and retrieval behavior
- Config and CLI updates

## 1) Runtime Parse: analyze-artifacts (lightweight ingest)

### Entry

```
python3 -m hmopt.cli analyze-artifacts \
  --artifact flamegraph:outputs/framegraph.json
```

### Inputs

- `--artifact kind:path` (repeatable)
  - `flamegraph`: .json / .html / directory
  - `hitrace`, `hiperf`, `sysfs`
- `--legacy-pipeline` (optional)
  - When enabled (or `--with-patch`), uses original PSG/LLM path

### Outputs

- DB records: `runs`, `metrics`, `hotspots`, `artifacts`
- Run outputs: `data/runs/<run_id>/dataset.json`
- Artifacts stored under `data/artifacts/`

### Logic Summary

1. Register run and run directory.
2. Parse artifacts (no LLM).
3. Build metrics + hotspots.
4. Store full flamegraph maps (counts, call stacks, name maps, pcg).
5. Align hotspots to kernel index (lookup by symbol name).
6. Persist metrics, hotspots, evidence_pack, dataset.json.

### Notes

- The lightweight flow **does not require repo_path**.
- Repo snapshot is only attempted if `project.repo_path` exists and path is valid.

## 2) Flamegraph Parse Preservation

The parser stores more than hotspots. For each flamegraph parse run, the following artifacts are retained:

- `flamegraph_call_stacks`
- `flamegraph_symbol_counts`
- `flamegraph_symbol_counts_raw`
- `flamegraph_symbol_counts_per_thread`
- `flamegraph_name_map`
- `pcg_flamegraph`

These enable later query of non-hotspot symbols.

## 3) Query Pipeline (runtime_code)

### Entry

```
python3 -m hmopt.cli query "analyze hotspots" \
  --mode runtime_code --run-id <run_id>

python3 -m hmopt.cli query @queries/runtime.md \
  --mode runtime_code --hotspot-top-k 5

python3 -m hmopt.cli query "..." \
  --mode runtime_code --symbols foo,bar
```

### Inputs

- `query_str`: direct text or `@/path/to/file`
- `--run-id`: selects runtime data from DB
- `--symbols`: optional list of symbols
- `--hotspot-top-k`: optional top-k selection
- `--mode`: `auto | code | runtime | runtime_code | graph`

### Outputs

- LLM response text per symbol
- If multiple symbols: one block per symbol (`## Hotspot <symbol>`)

### Query Flow (runtime_code)

1. Load runtime nodes from DB when run_id is provided.
2. Build `symbol_queue`:
   - Explicit symbols > hotspot_focus_symbol > top-k hotspots.
3. For each symbol:
   - Build runtime_context:
     - from hotspots if available
     - otherwise from flamegraph call_stacks + symbol_counts
   - Build code_context:
     - vector snippets (snippets or hybrid mode)
     - MCP retrieval (mcp or hybrid mode)
   - Construct final prompt and send to LLM.

### Runtime Context Format (example)

```
Top runtime hotspots:
- reclaim_services score=1248931.0 path=kernel/mm/reclaim.c lines=321-321
  Call paths:
    (call) do_fork (self=123, sub=87432) -> copy_process (self=88, sub=87432) -> reclaim_services (self=51200, sub=87432) (total_events=87432, thread_id=1234)
    (called) reclaim_services (self=51200, sub=51200) -> shrink_node (self=23000, sub=51200) -> shrink_lruvec (self=18000, sub=51200) (total_events=51200, thread_id=1234)
  ...and 3 more call paths (top 20 by total_events)

Runtime metrics:
- instruction_count_total value=60411932298.0 unit=instructions
- flamegraph_thread_count value=412.0 unit=
```

If symbol is not a hotspot, fallback uses flamegraph artifacts:

```
Flamegraph symbol stats: reclaim_services score=87432.0
Flamegraph call paths:
    (call) do_fork (self=123, sub=87432) -> ... -> reclaim_services (self=51200, sub=87432) (total_events=87432)
```

## 4) MCP and Kernel Index Retrieval

### MCP Flow (hybrid: forced + tool-call)

- MCP is invoked via `MCPToolAgent` when enabled and query_code_context_mode is `mcp` or `hybrid`.
- `MCPToolAgent` **always performs an initial forced MCP retrieval**, then starts an LLM tool-call loop.
- The MCP tool name is `kernel_index_code`.
- The MCP server responds by calling `retrieve_code_context()`.
- `retrieve_code_context()` returns: code implementation + graph expansion (multi-hop) + reranked symbols.

### retrieve_code_context

Retrieval combines:

1) **Vector**: LlamaIndex code index (top_k chunks).
2) **Graph**: Neo4j relationship expansion for matched symbols.

The returned context includes code snippets and graph relations.

## 5) Configuration

Key config additions (configs/app.yaml):

```
indexing:
  query_hotspot_top_k: 10
  query_code_context_mode: snippets  # snippets|mcp|hybrid
  mcp:
    enabled: false
    base_url: http://10.90.56.33:20010/v1
    api_key_env: HMOPT_LLM_API_KEY
    model: glm-4.7
    timeout_sec: 30
    tool_name: kernel_index_code
    top_k: 6
    mcp_base_url: http://localhost:8000
    mcp_api_key_env: HMOPT_MCP_API_KEY
```

## 6) CLI Updates

- `analyze-artifacts`
  - default = lightweight ingest
  - `--legacy-pipeline` restores old flow

- `query`
  - `--symbols` to force symbol list
  - `--hotspot-top-k` to control hotspot iteration
  - `@file` query input supported

## 7) Key Files Touched

- `src/hmopt/orchestration/graph.py`
- `src/hmopt/indexing/llamaindex_pipeline.py`
- `src/hmopt/cli.py`
- `src/hmopt/core/config.py`
- `configs/app.yaml`
- `src/hmopt/api/main.py`

## 8) Suggested Commit Message

```
refactor: simplify artifact ingest and unify runtime->code query with MCP

- add lightweight runtime ingest path for analyze-artifacts (no PSG/LLM)
- preserve full flamegraph artifacts (counts, call stacks, name maps, pcg)
- align hotspots to kernel index via code symbol lookup
- enhance query pipeline with symbol list/top-k routing and @file prompts
- add MCP/hybrid code context retrieval (vector + graph)
- extend runtime call stack formatting with per-frame self/sub events
- update config/CLI to support new pipeline and MCP settings
```

