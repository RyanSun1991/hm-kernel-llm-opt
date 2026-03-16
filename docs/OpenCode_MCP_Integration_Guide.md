# OpenCode MCP Integration Guide (HMOPT Kernel Index)

## Goal

Expose the existing HMOPT kernel index (LlamaIndex + Neo4j) through standard MCP so OpenCode can retrieve deeper, dependency-aware kernel context while coding and analysis.

## Capability Assessment

### Before this change

- `src/hmopt/api/mcp_server.py` only exposed a custom `POST /tools/call` endpoint.
- That endpoint was not a standard MCP lifecycle endpoint.
- Result: OpenCode `remote` MCP could not connect directly.

### After this change

- Standard MCP `streamable-http` endpoint is available at `/mcp`.
- Standard MCP `stdio` server is available via `python -m hmopt.api.mcp_stdio`.
- Legacy `POST /tools/call` is still supported for internal compatibility.

This now supports OpenCode in both:

- `local` MCP mode (stdio)
- `remote` MCP mode (streamable-http)

## Why this is better than direct source reading

OpenCode reading raw files is limited to lexical context. The MCP tools now provide:

1. Vector retrieval over indexed kernel chunks (semantic relevance).
2. Neo4j graph expansion (caller/callee and dependency propagation).
3. Relation-aware symbol reranking (`calls`, `uses_type`, `uses_macro`, etc.).
4. Symbol location metadata (path + line range) aligned to indexed entities.
5. Scenario-specific retrieval strategies for implementation lookup, call graph analysis, hotspot analysis, and patch planning.

This yields more complete and accurate context for change impact analysis and performance-oriented kernel edits.

## MCP Tools Exposed

Tool names are configurable via environment variables:

- `HMOPT_MCP_TOOL_NAME` (default: `kernel_index_code`)
- `HMOPT_MCP_GRAPH_TOOL_NAME` (default: `kernel_symbol_graph`)
- `HMOPT_MCP_HOTSPOT_TOOL_NAME` (default: `kernel_hotspot_context`)

### 1) `kernel_index_code` (general, scenario-aware)

Use for broad retrieval with scenario tuning.

Primary args:

- `query: str`
- `scenario: str`  
  Supported: `general`, `implementation`, `call_graph`, `impact_analysis`, `hotspot_debug`, `patch_planning`
- `symbols: list[str] | comma-separated string` (optional)
- `runtime_hints: str` (optional)
- `top_k`, `max_snippets`, `max_chars`, `graph_depth`
- `response_format`: `markdown` or `json`

### 2) `kernel_symbol_graph` (graph-centric)

Use for caller/callee and transitive dependency analysis.

Primary args:

- `symbols: list[str]`
- `query: str` (optional)
- `graph_depth`, `top_k`, `max_snippets`, `max_chars`
- `response_format`

### 3) `kernel_hotspot_context` (hotspot-centric)

Use when runtime hotspots are known and you need optimization-oriented code context.

Primary args:

- `symbols: list[str]` (optional but recommended)
- `query: str` (optional)
- `runtime_hints: str` (optional)
- `graph_depth`, `top_k`
- `response_format`

## Scenario Coverage Matrix (OpenCode)

Use this matrix to choose the most accurate MCP path by task intent.

| OpenCode task intent | Recommended MCP tool | Required/important args | Why this is stronger than direct file reading |
| --- | --- | --- | --- |
| Understand one function implementation quickly | `kernel_index_code` | `scenario=implementation`, `query`, optional `symbols` | Returns semantic matches + exact symbol snippets + file/line location. |
| Trace caller/callee chains | `kernel_symbol_graph` | `symbols`, optional `graph_depth` | Uses Neo4j relationships to expand transitive dependencies. |
| Estimate change impact radius | `kernel_index_code` or `kernel_symbol_graph` | `scenario=impact_analysis`, `symbols`, `graph_depth` | Prioritizes upstream/downstream graph neighbors with relation-aware ranking. |
| Runtime hotspot optimization | `kernel_hotspot_context` | `symbols`, `runtime_hints` | Prioritizes performance-critical paths and related implementation context. |
| Plan a safe multi-file patch | `kernel_index_code` | `scenario=patch_planning`, `symbols`, `graph_depth` | Surfaces coupled symbols/files that should be modified together. |

## Neo4j Vector + Graph Fusion Details

The MCP retrieval pipeline combines:

1. **Vector retrieval** from indexed kernel code chunks (semantic relevance).
2. **Graph expansion** from focus symbols (incoming + outgoing neighbors up to configured depth).
3. **Scenario-aware relation weighting**:
   - `call_graph`: strongly boosts `calls`.
   - `impact_analysis`: boosts `calls` + type/containment relations.
   - `hotspot_debug`: boosts performance-relevant dependency relations.
4. **Relation-aware reranking**:
   - Final score uses vector score + focus/depth bonus + graph relation bonus + relation degree bonus.
5. **Evidence-rich output**:
   - ranked symbols with `vector_score`, `relation_bonus`, `graph_degree`, `relation_breakdown`.
   - snippets with path/line metadata.
   - graph edges + relation summary.

This is the core reason OpenCode gets deeper and more accurate context than plain source browsing.

## Integration Paths

### Option A (recommended): OpenCode local MCP (`stdio`)

1. Install project dependencies:

```bash
pip install -e .
```

2. Add MCP config to OpenCode (see `examples/opencode.mcp.local.jsonc`):

```json
{
  "$schema": "https://opencode.ai/config.json",
  "mcp": {
    "hmopt_kernel_index": {
      "type": "local",
      "command": ["python3", "-m", "hmopt.api.mcp_stdio"],
      "enabled": true
    }
  }
}
```

### Option B: OpenCode remote MCP (`streamable-http`)

1. Start MCP server:

```bash
bash scripts/run_mcp_server.sh
```

2. Configure OpenCode remote MCP (see `examples/opencode.mcp.remote.jsonc`):

```json
{
  "$schema": "https://opencode.ai/config.json",
  "mcp": {
    "hmopt_kernel_index_remote": {
      "type": "remote",
      "url": "http://127.0.0.1:7331/mcp",
      "enabled": true
    }
  }
}
```

3. Optional API key hardening:

- Set server env: `HMOPT_MCP_SERVER_API_KEY=...`
- Set OpenCode header: `Authorization: Bearer ...`


## Auto-Test MCP (Phone via HDC)

If you need OpenCode to trigger test execution on a phone, start the Auto-Test MCP server and configure it as a remote MCP endpoint:

```bash
bash scripts/run_auto_test_mcp_server.sh
```

OpenCode MCP sample (`examples/opencode.mcp.remote.jsonc`) includes `hmopt_auto_test_remote` at `http://127.0.0.1:7336/mcp`.

Default tool: `phone_test_run`

Typical tool arguments:

- `target`: optional connect key / endpoint (for `hdc connect`/`tconn` and optional `-t`)
- `test_case`: test identifier passed to remote script
- `remote_script`: script path on phone
- `remote_result_path`: file path on phone to retrieve
- `extra_args`: optional argument list for script
- `local_result_dir`: optional local output directory

Execution flow:

1. Optional connect step (`connect_before_shell=true`): try `hdc connect <target>`, auto-fallback to `hdc tconn <target>` for legacy hdc.
2. Run shell: `hdc [-t <target>] shell <remote_script> <test_case> ...`
3. Pull artifact: `hdc [-t <target>] file recv <remote_result_path> <local_result_path>`

### SSH reverse tunnel + Docker bridge (your 8710 scenario)

If your phone is reachable on your **build PC** via a reverse tunnel such as:

```bash
ssh -R 8710:localhost:8710 damon@10.123.104.91
```

and Auto-Test MCP runs inside Docker on that server, make container networking able to access host tunnel endpoint:

1. Ensure container can resolve host gateway: `host.docker.internal` (this repo now adds it in compose/one-click).
2. Use MCP tool argument `target=host.docker.internal:8710` (instead of `127.0.0.1:8710` inside container).
3. Keep tunnel alive before running MCP test call.

Example MCP tool call arguments:

```json
{
  "target": "host.docker.internal:8710",
  "test_case": "boot_smoke",
  "remote_script": "/data/local/tmp/run_test.sh",
  "remote_result_path": "/data/local/tmp/results/boot_smoke.xml",
  "connect_before_shell": false,
  "use_target_flag": false
}
```

## Legacy Compatibility

`POST /tools/call` still works and now supports all configured tool names.

Example:

```json
{
  "tool": "kernel_index_code",
  "arguments": {
    "query": "Analyze reclaim_services implementation and call graph",
    "scenario": "call_graph",
    "symbols": ["reclaim_services"],
    "response_format": "markdown"
  }
}
```

## Validation Checklist

1. Health:

```bash
curl http://127.0.0.1:7331/health
```

2. Legacy endpoint:

```bash
curl -X POST http://127.0.0.1:7331/tools/call \
  -H 'Content-Type: application/json' \
  -d '{"tool":"kernel_index_code","arguments":{"query":"memcpy implementation","scenario":"implementation"}}'
```

3. In OpenCode, run a prompt that explicitly asks for MCP tool usage and verify:

- symbol ranking exists
- snippet path/line metadata exists
- graph edges / relation summary exists

## Key Files

- `src/hmopt/api/mcp_service.py`
- `src/hmopt/api/mcp_server.py`
- `src/hmopt/api/mcp_stdio.py`
- `src/hmopt/indexing/llamaindex_pipeline.py`
- `examples/opencode.mcp.local.jsonc`
- `examples/opencode.mcp.remote.jsonc`
