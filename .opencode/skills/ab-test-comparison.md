# A/B Instruction-Count Test + Compare Protocol

This skill defines the Auto-Test MCP workflow the tester runs **after** stock / feature images have been flashed and settled.  It covers only the instruction-count test execution and the compare step — nothing about build/sign/flash/settle.

## Scope

- Auto-Test MCP only.
- Assumes `build-and-sign.md` produced a signed feature image *and* `flash-device-operations.md` flashed both stock and feature (each followed by a ~10 min settle).

## Tools

| Tool | Purpose |
|---|---|
| Auto-Test MCP `auto_test_relay_health` | Confirm the Windows relay used for instruction tests is reachable |
| Auto-Test MCP `run_instruction_test_async` | Submit modelCase `main.py` as an async task; returns `task_id` |
| Auto-Test MCP `instruction_test_status` | Poll a submitted task by `task_id` |
| Auto-Test MCP `compare_reports` | Fallback — re-compare two existing report dirs without re-running the test |

**Only `run_instruction_test_async` is acceptable** for the test step.  A single instruction-count run is 30–120 minutes on device; the sync `run_instruction_test` will tear the MCP session down before it finishes.  Always go async and poll.

## Preconditions

Before invoking any tool in this skill:

1. `auto_test_relay_health()` returns `relay_reachable=True`.  If not: verdict = **skipped** (infrastructure failure), return.
2. For Phase A: stock image already flashed and settled.
3. For Phase B: feature image already flashed and settled, and the `baseline_report` path from Phase A is in hand.

If any precondition fails, do not invoke the test — return the failing precondition to the manager.

## Long-Running Reality — Hard Rules for the Polling Loop

A full A/B cycle is 1–4 hours of device time.  The tester must stay alive:

1. After `run_instruction_test_async` returns a `task_id`, keep polling `instruction_test_status(task_id)` until `status["status"]` is `succeeded` or `failed`.  Progression: `pending` → `running` → `succeeded | failed`.
2. Do NOT return to the manager, delegate elsewhere, or declare a verdict while either async task is `pending` or `running`.
3. Poll interval: **60 seconds**.  Use Bash `sleep 60` if the model has no native sleep primitive.
4. Ceiling: **180 minutes per phase**.  Past that, mark **inconclusive** and report.
5. Emit a progress line every 5–10 polls so the user can see liveness: elapsed minutes, current status, task_id.

## Mandatory Sequence

### Phase A — Stock Instruction-Count Test

```
# Precondition: stock image is already flashed + settled (flash-device-operations.md).

# A1. Submit the async test.
task_stock = run_instruction_test_async(
    compare=False,        # stock has no baseline yet
    # test_dir / main_script / pipeline_script default to the Windows host config.
)
task_id_stock = task_stock["task_id"]

# A2. Poll until terminal.
#   Loop: status = instruction_test_status(task_id_stock)
#         if status["status"] in ("succeeded", "failed"): break
#         sleep 60s
#         (every 5–10 iterations, emit a progress line)

# A3. Capture baseline_report.
baseline_report = status["result"]["report_path"]
#   e.g. r"D:\modelCase_OH_single\reports\report_20260414114948"
# Pass this into Phase B.
```

On `failed` / wait-ceiling / `result.success=False`: report the phase, `status["error"]`, and `run_result.stderr_tail`.  Verdict = **skipped** (stock environment issues are almost never patch-related).

### Phase B — Feature Instruction-Count Test + Compare

```
# Precondition: feature image is already flashed + settled AND baseline_report from Phase A is known.

# B1. Submit async test with compare=True.
task_feature = run_instruction_test_async(
    compare=True,
    baseline_report=baseline_report,
    # Granularity — pick the metric the plan actually targets (default "total"):
    compare_level="total",   # or "process" / "thread" / "lib" / "function"
    # For anything deeper than "total", provide the names at and above that level:
    #   compare_level="process"  → compare_process="<processName>"
    #   compare_level="thread"   → compare_process=... compare_thread=...
    #   compare_level="lib"      → compare_process=... compare_thread=... compare_lib=...
    #   compare_level="function" → all four: compare_process / _thread / _lib / _function
)
task_id_feature = task_feature["task_id"]

# B2. Poll using the same loop as Phase A.
```

### Phase C — Extract Comparison

When `task_feature` reaches `succeeded`, the pipeline has already invoked `report_compare.py` on Windows and embedded the result:

```
status["result"]["compare"] = {
    "ok": true,
    "returncode": 0 | 2,
    "baseline_report": "D:\\...\\report_20260414114948",
    "candidate_report": "D:\\...\\report_20260414120950",
    "result": {
        "success": true,
        "level": "total",       # or process / thread / lib / function
        "target": {...},        # names at the chosen level
        "aggregate": {
            "baseline": 101886000000,
            "candidate": 98500000000,
            "delta": -3386000000,
            "delta_pct": -3.32,
            "baseline_found": true,
            "candidate_found": true,
            "pairs_compared": 3,
            "pairs_missing_baseline": 0,
            "pairs_missing_candidate": 0
        },
        "reports": [ /* per (case, round, step) pair */ ]
    }
}
```

If the embedded compare is missing or errored, fall back to an explicit call:

```
compare_reports(
    baseline_report=baseline_report,
    candidate_report=status["result"]["report_path"],
    level=<same level as B1>,
    process=..., thread=..., lib=..., function=...,
)
```

## Decision Criteria

All judgments use the `aggregate` section — the sum across every matched `(case, round, step)` pair.

### PASS

ALL of the following must hold:

- `aggregate.delta <= 0` (feature uses no more instructions than stock at the chosen level)
- `aggregate.pairs_missing_baseline == 0` AND `aggregate.pairs_missing_candidate == 0` (every pair matched — no dropped test case)
- `aggregate.baseline_found` AND `aggregate.candidate_found` are both True when a named target was specified
- No correctness regressions in `run_result.stderr_tail` of either phase
- Both async tasks reached `status == "succeeded"`

### FAIL

ANY of the following:

- `aggregate.delta > 0` (regression at the chosen level)
- A previously-present process/thread/lib/function disappears on the feature side when the plan's metric depends on it
- Crash / hang / test-failure strings in stderr tail

### INCONCLUSIVE

- `abs(aggregate.delta_pct)` within noise margin (< 1 %)
- `pairs_missing_*` > 0
- 180-minute wait ceiling hit on either phase

### SKIPPED

- Infrastructure failure (relay unreachable, device not flashed/settled, build/sign failed upstream)

## Output Format (Validation Report Section)

The tester's validation artifact `.opencode/bench/<artifact>_validation.md` includes for this skill's scope:

```markdown
## A/B Instruction-Count Test

### Stock (Phase A)
- Async task_id: {task_id}
- Wait time: {elapsed_minutes}m
- Terminal status: succeeded | failed
- report_path: {D:\modelCase_OH_single\reports\report_YYYYMMDDHHMMSS}
- Notable stderr lines: {one or two diagnostic lines}

### Feature (Phase B)
- Async task_id: {task_id}
- Wait time: {elapsed_minutes}m
- Terminal status: succeeded | failed
- report_path: {D:\modelCase_OH_single\reports\report_YYYYMMDDHHMMSS}
- Notable stderr lines: {one or two diagnostic lines}

### Delta (Phase C)
- Compare level: total | process | thread | lib | function
- Target: {names at and above the level, or "—" for total}
- Pairs compared: {N}
- Aggregate baseline: {int} instructions
- Aggregate candidate: {int} instructions
- Aggregate delta: {signed int} ({signed pct}%)
- Per-pair table: case/round/step → baseline / candidate / delta / delta_pct
- Missing targets: {list or "none"}

### Verdict: pass | fail | inconclusive | skipped
### Confidence: high | medium | low
### Recommended next route: accept | iterate | reject
### Rationale: {one-paragraph; cite aggregate.delta_pct and notable per-pair entries}
```

## Hard Rules (recap)

1. NEVER use `run_instruction_test` (sync).  Only `run_instruction_test_async` + `instruction_test_status` polling.
2. NEVER return to the manager, delegate, or end the session while an async test is still `pending` or `running`.
3. NEVER start the test before the image is flashed AND settled.  This skill assumes `flash-device-operations.md` already ran.
4. Both phases MUST use identical test workspace + parameters — the only legitimate delta is the flashed image.
5. If stock phase fails, report infrastructure failure — not a patch failure.
6. The `compare_level` in Phase B MUST match the metric named in the plan.
