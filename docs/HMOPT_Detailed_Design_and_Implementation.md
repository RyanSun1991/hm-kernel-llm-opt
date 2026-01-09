# HM-Verif-Kernel Code & Performance Optimization Platform (HMOPT)
## Detailed Design Document + Implementation Playbook (Human- and Codex-Executable)

**Version:** 1.0  
**Date:** 2025-12-31 (Europe/Berlin)  
**Target repo:** `hm-verif-kernel` (integration adapter required)  
**Primary goal:** Build a local-first platform that *ingests code + performance artifacts*, correlates them, and runs an *LLM-controlled iterative optimization loop* with reproducible builds/tests/profiling and long-term storage for context learning and future training.

---

## 0. References (must-read inputs)

- **PRAGMA**: Profiling-guided multi-agent optimization loop (Conductor/Coder/Verifier/Profiler + best-history retention). fileciteturn0file0  
- **POLO**: Project-level optimization with PCG (runtime call graph) + PSG (static structure graph) + multi-round generator/decision agents. fileciteturn0file1  
- **Model gateway**: OpenAI-compatible internal API base, model list, security note (internal vs external proxy). fileciteturn0file2  
- **HM kernel pain points & goal**: dynamic/nonlinear perf issues; massive trace data; self-evolution objective. fileciteturn0file3  
- LangChain Agents & LangGraph orchestration concepts (for engineering patterns). citeturn0search2turn0search1  

> Note: Gemini shared links were provided, but require interactive login in this environment and were not accessible during document drafting.

---

## 1. Executive Summary

We will implement **HMOPT**, a platform that unifies:
1) **Data plane:** code snapshots + perf artifacts + derived metrics + graphs + patches, stored locally and reproducibly.  
2) **Reasoning plane:** an explicit **LangGraph** state machine orchestrating a PRAGMA-like multi-agent loop for analysis and optimization. citeturn0search1  
3) **Control plane:** strict experiment tracking, safety policies (internal models by default), stop conditions, regression detection, and reporting.

The system is designed to:
- Reduce manual analysis of huge traces (hitrace/hiperf/flamegraph/klog/proc). fileciteturn0file3  
- Support project-level changes safely via static and runtime graph context (POLO). fileciteturn0file1  
- Improve kernel code/config iteratively with profiling evidence and historical best retention (PRAGMA). fileciteturn0file0  
- Accumulate a local dataset for future fine-tuning/training and “self-evolution.” fileciteturn0file3  

---

## 2. Goals, Non-Goals, Constraints

### 2.1 Goals
- **G1: Reproducible experiments**: each run has immutable identifiers; artifacts hashed; metrics comparable across iterations.
- **G2: Performance artifact ingestion**: flamegraph, hitrace, hiperf (minimum), with a plugin interface for new formats.
- **G3: Correlation and evidence packs**: align symptoms (frame drops, scheduler latency spikes) to code regions with structured evidence.
- **G4: Agentic optimization loop**:
  - Coder proposes diffs
  - Verifier builds/tests (non-LLM)
  - Profiler collects traces (non-LLM)
  - Conductor reasons with profiling feedback and best-history comparison
- **G5: Local-first storage**: relational DB + artifact store + vector index to enable RAG and dataset generation.
- **G6: Security-by-default model routing**: internal models only unless explicitly enabled by policy. fileciteturn0file2  

### 2.2 Non-Goals (v1)
- Building a full GUI; v1 is CLI + API (optional) + reports.
- Supporting all possible perf tools; we implement the minimum set first and expand.
- Guaranteeing improvements for all workloads: the loop is evidence-driven but may plateau.

### 2.3 Constraints
- Kernel builds/tests/profiling are environment- and device-dependent ⇒ implement **adapters**.
- LLM prompts must not leak sensitive information to external proxies; default to internal gateway. fileciteturn0file2  

---

## 3. System Architecture

### 3.1 Layered view
**A) Data Plane**
- Repo snapshots (commit + file hashes)
- Perf artifacts (raw files)
- Derived metrics tables + time series
- Graphs: PSG (static), PCG (runtime)
- Patch diffs and iteration outcomes

**B) Reasoning Plane**
- LangGraph state machine (explicit nodes + transitions) citeturn0search1  
- Agents: Conductor, Trace Analyst, Coder, Safety; Tools: DB query, code retrieval, patch apply, build/test, profile

**C) Control Plane**
- Experiment manager (IDs, hashing, lineage)
- Safety/policy gate (model selection, redaction)
- Stop conditions + regression detection + budget controls
- Reporting + dataset export

### 3.2 Component diagram (logical)
1) Ingestion: repo + artifacts  
2) Static analysis: symbol index + PSG builder  
3) Runtime analysis: parsers + metrics + PCG  
4) Correlation: align perf hotspots/symptoms ↔ PSG nodes  
5) Orchestration: LangGraph loop across agents and tools  
6) Storage: DB + artifacts + vector store  
7) Evaluation: compare runs, compute deltas, detect regressions  
8) Dataset export: create training examples from successful iterations

---

## 4. Data Model (DB + Artifacts + Vectors)

### 4.1 Storage strategy
- **Relational DB (SQLite default, Postgres optional):** metadata, metrics, graphs, run lineage.
- **Artifact store (filesystem):** raw traces/logs/binaries, content-addressed by sha256.
- **Vector store (local):** embeddings for code chunks, trace summaries, best-fix rationales, patches.

### 4.2 Canonical entities (tables)
> Minimal schema for v1; can be normalized further later.

#### `runs`
- `run_id` (UUID)
- `parent_run_id` (nullable) – for iterations / lineage
- `repo_uri`, `repo_rev`, `repo_dirty`  
- `workload_id`, `device_id`, `toolchain_id`
- `status` (created/running/succeeded/failed/stopped)
- timestamps: `created_at`, `finished_at`

#### `artifacts`
- `artifact_id` (UUID)
- `run_id`
- `kind` (build_log, test_log, hitrace, hiperf, flamegraph, report, patch_diff, binary, …)
- `sha256`, `path`, `bytes`, `mime`
- `metadata_json`

#### `metrics`
- `run_id`
- `scope` (system/process/thread/frame)
- `metric_name` (fps, frame_drop_rate, sched_latency_p99, ipc, cache_miss_rate, …)
- `value`, `unit`
- `tags_json` (e.g., process name, thread name, cpu cluster)

#### `hotspots`
- `run_id`
- `symbol` (function)
- `file_path`, `line_start`, `line_end`
- `score` (FuncRank-like, or tool-specific)
- `evidence_artifact_ids` (JSON list)

#### `graphs`
- `run_id`
- `kind` (PSG/PCG)
- `format` (json/adjlist)
- `payload_artifact_id` (graph stored as artifact)
- `metadata_json` (node/edge counts, generation tool)

#### `patches`
- `run_id`
- `iteration`
- `diff_artifact_id`
- `apply_status` (applied/rejected/failed)
- `files_changed_json`

#### `evaluations`
- `run_id`
- `baseline_run_id`
- `delta_metrics_json`
- `correctness_passed` (bool)
- `perf_improved` (bool)
- `notes`

#### `agent_messages`
- `run_id`, `iteration`
- `agent_name`
- `model_id`
- `prompt_artifact_id`, `output_artifact_id`
- `summary_artifact_id` (optional for compact recall)

### 4.3 Artifact store layout
- `data/artifacts/<sha256_prefix>/<sha256>.<ext>`  
- DB references artifact path + metadata.

### 4.4 Vector store objects
- `CodeChunk`: `{repo_rev, path, symbol, text, embedding, tags}`
- `TraceSummary`: `{run_id, kind, summary_text, embedding, time_range}`
- `FixCase`: `{baseline_run_id, improved_run_id, hotspot, diff, delta_metrics, embedding}`

---

## 5. Performance Artifact Normalization (Schema)

### 5.1 Why normalize
Raw traces are too large and heterogeneous to feed into an LLM directly; normalization enables:
- structured metric extraction
- consistent evidence packs
- robust cross-run comparisons
- dataset creation

### 5.2 Normalized event model (conceptual)
- `Event(ts, category, name, dur, pid, tid, cpu, args_json)`
- `Sample(ts, stack, weight, pid, tid, cpu, symbol_map_ref)`

### 5.3 Derived KPIs (HM-oriented)
Based on HM pain points and typical targets: FPS, frame drops, sched latency, CPU idle/sleep residency, IRQ latency/balance, memory efficiency. fileciteturn0file3  

Minimum v1 KPIs:
- `fps_avg`, `frame_drop_rate`, `jank_p95_ms`
- `sched_latency_p50/p95/p99`
- `cpu_util_avg`, `cpu_idle_ratio`
- `top_hotspots` (time/sample-based)
- `regression_flags` (binary indicators)

---

## 6. Static Analysis (PSG) Design

### 6.1 PSG definition (POLO-inspired)
PSG is a directed attributed graph capturing project structure. fileciteturn0file1  

Nodes:
- functions, structs/classes, global variables, config knobs

Edges:
- `call_static`, `ref`, `hasmember/ismember`, `inheritance`, `config_guard`

Attributes:
- file path, namespace, signature, body hash, etc.

### 6.2 Implementation approach
v1 options (choose one first):
- **Option A (fast):** ctags + regex heuristics for function boundaries (low fidelity, quick start)
- **Option B (recommended):** clang/libtooling for C/C++ and a dedicated parser for kernel-specific constructs (higher fidelity, cost)

We will start with A to make pipeline runnable, then upgrade to B.

---

## 7. Runtime Analysis (PCG) + Hotspot Ranking

### 7.1 PCG definition (POLO-inspired)
PCG represents runtime call relationships and costs. fileciteturn0file1  

Nodes:
- runtime functions (custom + library)

Edges:
- call edges with count and cost contribution

### 7.2 Hotspot ranking
Implement two rankers:
1) Baseline: sort by self-time or samples
2) FuncRank-like: PageRank-inspired using execution cost + propagation (POLO) fileciteturn0file1  

Output:
- ranked hotspot candidates with evidence references

---

## 8. Correlation Pack (Evidence Pack) Design

### 8.1 Purpose
The Evidence Pack is the **single source of truth** for agent reasoning:
- compact
- structured
- reproducible
- anchored to artifact IDs and source locations

### 8.2 Contents
For each selected hotspot:
- Hotspot identity: symbol, file, lines
- PSG neighborhood: callers/callees, owning struct, referenced globals/config
- PCG neighborhood: top callers/callees and their costs
- Symptom windows (frame drops, latency spikes) with trace excerpts
- Derived metric deltas vs baseline and vs best historical run
- Risk notes: locking, concurrency, memory safety boundaries
- “Suggested optimization strategies” (template-level, not yet code)

### 8.3 Token budgeting
Evidence Pack must carry:
- `max_prompt_tokens`
- `priority_fields`
- per-section summaries (short + medium)
So the Conductor can assemble stable prompts over many iterations.

---

## 9. Agent Orchestration (PRAGMA + POLO fusion)

### 9.1 Why LangGraph
LangGraph is designed for **long-running, stateful agent orchestration** with explicit nodes and transitions. citeturn0search1  
Agents in LangChain run tool loops to reach a goal and stop when stop conditions trigger. citeturn0search2  
We need explicit workflow control, retries, and deterministic stop policies ⇒ graph-based orchestration.

### 9.2 Agent roles (v1)
- **Conductor (LLM):** orchestrates, diagnoses bottlenecks, sets next steps; compares to historical best (PRAGMA). fileciteturn0file0  
- **Trace Analyst (LLM):** interprets normalized metrics and trace windows into bottleneck classes (HM taxonomy).
- **Coder (LLM):** generates minimal diffs; adheres to patch constraints and style.
- **Safety/Policy (deterministic + optional LLM):** model routing and prompt redaction, internal-only by default. fileciteturn0file2  
- **Verifier (non-LLM):** build + test + triage logs.
- **Profiler (non-LLM):** run workload + collect artifacts.

### 9.3 Orchestration state (`RunState`)
Fields:
- `run_id`, `iteration`
- `baseline_run_id`, `best_run_id`
- `current_patch_id`
- `selected_hotspots`
- `evidence_pack_id`
- `stop_reason`, `budget_remaining`

### 9.4 Graph nodes (recommended v1)
1) `init_run`
2) `snapshot_repo`
3) `static_analysis_psg`
4) `baseline_build_test`
5) `baseline_profile`
6) `runtime_analysis_pcg_hotspots`
7) `build_evidence_pack`
8) `conductor_decide`
9) `coder_generate_patch`
10) `apply_patch`
11) `verify_build_test`
12) `profile_candidate`
13) `evaluate_delta_and_update_best`
14) `stop_or_continue`
15) `generate_report`
16) `export_dataset`

Stop conditions:
- regression detected N times
- no improvement for K iterations
- budget/time limit
- Conductor explicitly ends with justification

### 9.5 “Best history retention” (PRAGMA)
Store best run ID + its profiling summary; Conductor must compare current candidate to best historical each iteration. fileciteturn0file0  

---

## 10. LLM Gateway Integration (Internal models by default)

### 10.1 Endpoint and models
Use OpenAI-compatible API base and API key from the internal gateway doc. fileciteturn0file2  

- `api_base`: `http://10.90.56.33:20010/v1`
- internal models include `qwen3-coder-30b`, `gpt-oss-120b`, `glm-4.7`, etc. fileciteturn0file2  

### 10.2 Model routing policy
- Default: `allow_external_proxy_models = false` fileciteturn0file2  
- Use:
  - `coding_fast`: qwen3-coder-30b
  - `reasoning_strong`: gpt-oss-120b
  - `general`: glm-4.7
- Embeddings: `qwen3-embedding-8b`

### 10.3 Prompt safety
- Redaction rules for logs/traces (PII, secrets, internal hostnames if needed)
- Hard block of external proxied models unless whitelisted for a given run
- Record prompt + output artifacts for auditability

---

## 11. Interfaces: CLI, API, Tools, Adapters

### 11.1 CLI (must-have)
Commands:
- `hmopt run --config configs/app.yaml`
- `hmopt ingest repo --path ... --rev ...`
- `hmopt ingest artifacts --run-id ... --path ...`
- `hmopt analyze --run-id ...`
- `hmopt optimize --baseline-run-id ... --workload ...`
- `hmopt report --run-id ...`

### 11.2 API (optional but recommended)
- `GET /health`
- `POST /runs`
- `GET /runs/{run_id}`
- `GET /runs/{run_id}/metrics`
- `GET /runs/{run_id}/artifacts`
- `POST /runs/{run_id}/optimize`

### 11.3 Adapter interfaces (critical for hm-verif-kernel)
We must not hard-code kernel build/test/profile commands.
Define interfaces:

**BuildAdapter**
- `build(repo_path, build_config) -> BuildResult(artifact_ids, success, logs)`

**TestAdapter**
- `test(repo_path, test_plan) -> TestResult(success, logs, coverage?)`

**WorkloadAdapter**
- `run(workload_id, duration, options) -> WorkloadResult(artifacts)`

**ProfilerAdapter**
- `collect(workload_result) -> PerfArtifactBundle(hitrace, hiperf, flamegraph, ...)`

Adapters are configured by YAML (device/workload-specific).

---

## 12. Observability & Reporting

### 12.1 Logging
- structured JSON logs with `run_id`, `iteration`, `node_name`
- store logs as artifacts (not only stdout)

### 12.2 Metrics
- time spent per pipeline node
- build/test pass rate
- perf improvement distribution
- agent token usage per iteration (for cost control)

### 12.3 Reports (markdown + optional HTML)
Report sections:
- summary: baseline vs best
- KPIs table and charts references
- hotspots + code locations
- patch diff summary
- trace evidence windows
- risk assessment and rollback plan

---

## 13. Security & Governance

- Default internal-only models. fileciteturn0file2  
- Explicit allow-list for external proxy models (off by default).
- Prompt redaction stage for logs/traces.
- Database permission boundary: DB connection used by agent tools must be scoped (read-only tools vs write tools).
- Tool execution sandboxing: patch apply/build/profile restricted to a working directory.

---

## 14. Testing Strategy

### 14.1 Unit tests
- parsers: flamegraph/hitrace/hiperf parsing with golden files
- artifact store: hashing and retrieval
- DB: schema migrations and CRUD

### 14.2 Integration tests
- “dummy adapter” that simulates build/test/profile with canned artifacts
- run the full pipeline end-to-end in CI

### 14.3 Regression tests
- ensure evidence pack schema is backward compatible
- ensure stop conditions function

---

## 15. Implementation Plan (Phased Roadmap)

This roadmap is designed to be executable by humans and by coding agents.

### Phase A — Core plumbing (Week 1–2)
**Goal:** create RunIDs, store artifacts/metrics, call LLM gateway.
- A1: Config loader + typed config
- A2: DB engine + migrations + artifact store
- A3: LLM client wrapper + model router (internal-only)
- A4: CLI scaffolding + `hmopt run` creates a run record

Acceptance:
- `hmopt run` creates a run directory and DB rows; writes one dummy artifact and metric.

### Phase B — Repo ingestion + static indexing (Week 2–4)
- B1: repo snapshot (commit hash, file hashes, manifests)
- B2: symbol index v1 (ctags)
- B3: PSG v1 (calls/ref heuristics)

Acceptance:
- `hmopt ingest repo` persists snapshot; `hmopt analyze --static` can output PSG stats.

### Phase C — Perf ingestion + runtime analysis (Week 4–6)
- C1: artifact ingestion (hitrace/hiperf/flamegraph files)
- C2: flamegraph parser → fps, drop windows
- C3: hiperf parser → sample stacks → PCG
- C4: hotspot ranking (baseline + FuncRank-like)

Acceptance:
- `hmopt analyze --runtime` produces ranked hotspots and saves them to DB.

### Phase D — Evidence pack + reporting (Week 6–8)
- D1: correlation pack builder
- D2: report generator v1 (markdown)
- D3: vector store ingestion for code chunks + summaries

Acceptance:
- `hmopt report --run-id` generates a report artifact with hotspots + trace windows.

### Phase E — LangGraph agent loop (Week 8–12)
- E1: implement LangGraph nodes for the loop
- E2: implement Coder patch generation (diff format, constraints)
- E3: implement Verifier & Profiler adapters (real + dummy)
- E4: best-history retention + comparison logic

Acceptance:
- `hmopt optimize` runs at least 2 iterations end-to-end with dummy adapters and stores all artifacts.

### Phase F — HM domain expansion + dataset export (Week 12+)
- F1: HM bottleneck taxonomy + strategy templates (scheduler/memory/IRQ/flamegraph)
- F2: stronger redaction policy for sensitive logs
- F3: dataset exporter for successful runs

Acceptance:
- Export dataset entries with `(evidence_pack, diff, delta_metrics)`.

---

## 16. Codex-Executable Implementation Backlog (Structured Tasks)

The following tasks are designed to be copy-pasted into an AI coding agent (Codex-like) or used as a human checklist.
Each task includes:
- **Files** to modify/create
- **Steps**
- **Acceptance tests**

### Task T-A2: DB + Artifact Store
**Files**
- `src/hmopt/storage/db/engine.py`
- `src/hmopt/storage/db/schema.sql`
- `src/hmopt/storage/artifact_store.py`

**Steps**
1) Implement SQLite connection manager with context handling.
2) Implement schema bootstrap from `schema.sql`.
3) Implement artifact store:
   - compute sha256
   - store under `data/artifacts/<prefix>/<sha256>.<ext>`
   - return artifact metadata

**Acceptance**
- Run `python -c "from hmopt.storage... import ..."` without error
- Store a test file and confirm `sha256` path exists on disk
- Insert artifact row and query it back

### Task T-C2: Flamegraph Parser v1
**Files**
- `src/hmopt/analysis/runtime/traces/flamegraph_parser.py`
- `src/hmopt/analysis/runtime/metrics.py`

**Steps**
1) Define expected input formats (JSON/CSV) as configuration.
2) Parse frames timeline into:
   - fps_avg
   - frame_drop_rate
   - jank_p95_ms
   - top N jank windows (time ranges)

**Acceptance**
- Unit test with a small synthetic frame timeline.
- Metrics inserted into DB.

### Task T-E1: LangGraph Loop Skeleton
**Files**
- `src/hmopt/orchestration/state.py`
- `src/hmopt/orchestration/graph.py`
- `src/hmopt/agents/conductor.py`
- `src/hmopt/agents/coder.py`
- `src/hmopt/agents/verifier.py`
- `src/hmopt/agents/profiler.py`

**Steps**
1) Define `RunState` dataclass.
2) Build StateGraph with nodes:
   `init -> analyze -> decide -> patch -> verify -> profile -> eval -> decide`
3) Implement dummy verifier/profiler that reads canned artifacts.
4) Persist each iteration output as artifacts.

**Acceptance**
- A dry run completes 2 iterations and produces:
  - 2 patch artifacts
  - 2 evaluation rows
  - best_run_id updated

---

## 17. How to Use the Skeleton Repository (Current Status)

### 17.1 Install
```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
```

### 17.2 Configure model gateway
Use `.env` or edit `configs/model_server.yaml` with:
- `api_base: http://10.90.56.33:20010/v1` fileciteturn0file2  
- `api_key: <your_token>`

Windows proxy workaround (if needed): set `no_proxy=10.90.56.33`. fileciteturn0file2  

### 17.3 Run API service
```bash
bash scripts/run_api.sh
curl http://127.0.0.1:8000/health
```

### 17.4 Run pipeline
Pipeline runner is intentionally stubbed in the skeleton; implement phases A→E to enable full runs.

---

## 18. Appendices

### Appendix A: HM Performance Taxonomy (starter)
Based on the HM pain points doc: fileciteturn0file3  
- Frame stability: fps, drops, jank clusters
- Scheduler: runnable latency, wakeup latency, priority inversion indicators
- CPU: utilization, idle residency, freq transitions (energy/perf)
- IRQ: rate, affinity imbalance, latency
- Memory: reclaim pressure, alloc hotspots, fragmentation signals

### Appendix B: Why this design matches the papers
- PRAGMA: explicit roles + profiling loop + historical best. fileciteturn0file0  
- POLO: PSG+PCG context + iterative generator/decision. fileciteturn0file1  
- HM doc: local-first, workload-driven iterative self-evolution. fileciteturn0file3  
