# A/B Test Comparison Protocol

This skill defines the mandatory A/B (stock vs feature) test comparison protocol. When the tester agent validates a patch, it MUST run both a stock baseline instruction-count test and a feature candidate instruction-count test, then compare the two reports.

## Tools Used

| Tool | Purpose |
|---|---|
| Flash MCP `flash_stock` / `flash_feature` | Flash the baseline / patched image to the device |
| Auto-Test MCP `run_instruction_test_async` | Submit the modelCase `main.py` instruction-count run on Windows; returns a `task_id` immediately |
| Auto-Test MCP `instruction_test_status` | Poll the async task by `task_id` |
| Auto-Test MCP `compare_reports` | (Optional) Re-run `report_compare.py` against two existing report dirs without re-running the test |
| Auto-Test MCP `auto_test_relay_health` | Verify the Windows relay is reachable before anything else |

**Only `run_instruction_test_async` is acceptable for the test step.** The synchronous `run_instruction_test` is NOT permitted: a single test run is 30 minutes to 2 hours, and a sync call will either block the MCP transport until it times out or tear the session down mid-run. Always go async and poll.

## Long-Running Reality

A typical stock or feature run takes **30–120 minutes** on real hardware. Across a full A/B cycle the tester is waiting for ~1–4 hours of device time. This is fine — the close-loop (research → plan → code → review → test → compare → iterate) only works if the tester stays alive through the wait.

### Hard Rules for the Polling Loop

1. After `run_instruction_test_async` returns a `task_id`, the tester MUST keep polling `instruction_test_status(task_id)` until the task's `status` is `succeeded` or `failed`. The `status` progression is `pending` → `running` → `succeeded | failed`.
2. The tester MUST NOT return to the manager, delegate elsewhere, or declare a verdict until **both** the stock and the feature task have reached a terminal status.
3. A failed `run_instruction_test_async` payload parse or a `failed` task in either phase → report as infrastructure failure (verdict: **skipped** / **fail**), do NOT fabricate a delta.
4. Poll interval: **60 seconds** between `instruction_test_status` calls. Faster polling wastes tokens; slower polling delays the close-loop. Use Bash `sleep 60` between status calls if a native sleep primitive isn't available.
5. Overall wait ceiling: **180 minutes (3 hours)** per phase. If the task is still `running` after that, escalate as an infrastructure issue.
6. Keep a running note in the tester's conversation so the user can see progress: log every 5–10 polls with `elapsed`, `status`, and the test_dir being run in.

## Mandatory A/B Sequence

### Phase 0 — Infrastructure Check

```
auto_test_relay_health()
```

If `relay_reachable` is False, verdict = **skipped** (infrastructure failure). Do NOT continue.

### Phase A — Stock Baseline

```
# A1. Flash the STOCK image
flash_stock(device_serial="<serial>")    # synchronous, minutes, fine to await
# Confirm flash.success is True before proceeding.

# A2. Submit the stock instruction-count test (async — required)
task_stock = run_instruction_test_async(
    compare=False,          # stock has no baseline yet
    # test_dir / main_script / pipeline_script default to the workspace
    # configured on the Windows host (D:\modelCase_OH_single by default).
)
task_id_stock = task_stock["task_id"]

# A3. Poll until terminal.
#     Loop:
#         status = instruction_test_status(task_id_stock)
#         if status["status"] in ("succeeded", "failed"): break
#         sleep 60s
#         continue
#
#     Every 5–10 iterations, emit a progress line to the conversation so
#     the user can see liveness: elapsed minutes, current status, task_id.

# A4. Extract baseline_report.
baseline_report = status["result"]["report_path"]
#   e.g. r"D:\modelCase_OH_single\reports\report_20260414114948"
# Save this path — Phase B needs it.
```

If `status["status"] == "failed"` or `status["result"]["success"] is False`, report the phase, the recorded `error`, and the tail from `run_result.stderr_tail`. Verdict = **skipped** (test infrastructure / stock image) or **fail** (stock failure is almost always environmental, not patch-related).

### Phase B — Feature Candidate

```
# B1. Flash the FEATURE image.
flash_feature(device_serial="<serial>")
# Confirm flash.success is True.

# B2. Submit the feature test — this time with compare=True and the
#     baseline_report from Phase A.
task_feature = run_instruction_test_async(
    compare=True,
    baseline_report=baseline_report,

    # Pick the granularity the plan cares about.  Defaults to "total".
    compare_level="total",         # or "process" / "thread" / "lib" / "function"
    # For anything deeper than total, pass the names at and above that level:
    #   compare_level="process"  → compare_process="<processName>"
    #   compare_level="thread"   → compare_process=... compare_thread=...
    #   compare_level="lib"      → compare_process=... compare_thread=... compare_lib=...
    #   compare_level="function" → all four: compare_process / _thread / _lib / _function
)
task_id_feature = task_feature["task_id"]

# B3. Same polling loop as A3, now on task_id_feature.
#     Same ceiling and cadence.
```

### Phase C — Extract Comparison

The feature task's terminal `status["result"]` already contains the comparison output from `report_compare.py` (the pipeline ran it in-process on Windows after the test finished):

```
status["result"]["compare"] = {
    "ok": true,
    "returncode": 0 | 2,
    "command": "...",
    "baseline_report": "D:\\...\\report_20260414114948",
    "candidate_report": "D:\\...\\report_20260414120950",
    "result": {
        "success": true,
        "level": "total",            # or process / thread / lib / function
        "target": {...},             # names at the chosen level
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
        "reports": [
            {"case": "...", "round": 0, "step": 1,
             "baseline": ..., "candidate": ..., "delta": ..., "delta_pct": ...,
             "baseline_found": true, "candidate_found": true,
             "baseline_path": "...xlsx", "candidate_path": "...xlsx"}
            ...
        ]
    }
}
```

If the pipeline somehow produced a report pair but the embedded compare didn't run (e.g. it errored or was skipped), you can still invoke the comparison directly against the two report dirs:

```
compare_reports(
    baseline_report=baseline_report,          # stock report_path from Phase A
    candidate_report=status["result"]["report_path"],  # feature report_path
    level="total",
    # …plus process / thread / lib / function names if needed
)
```

## Decision Criteria

All judgments use the **aggregate** section of the compare result — that's the sum across every matched `(case, round, step)` pair.

### PASS

ALL of the following must hold:

- `aggregate.delta <= 0` (feature uses no more instructions than stock at the chosen level)
- `aggregate.pairs_missing_baseline == 0` and `aggregate.pairs_missing_candidate == 0` (every pair matched — no dropped test case)
- `aggregate.baseline_found` and `aggregate.candidate_found` are both True when a named target was specified (target exists on both sides)
- Test correctness maintained (no crashes, no functional failures in `run_result.stderr_tail`)
- Both flash operations succeeded
- Both async tasks reached `status == "succeeded"`

### FAIL

ANY of the following triggers a fail:

- `aggregate.delta > 0` (feature uses more instructions than stock — regression at the chosen level)
- A previously-present process/thread/lib/function disappears on the feature side (`candidate_found == False` when the baseline had it) if the target is the metric the plan chose
- Functional regression detected (crash, hang, test failure in stderr tail)
- Feature image flash failed but stock succeeded (patch likely broke the image)

### INCONCLUSIVE

- `abs(aggregate.delta_pct)` is within noise margin (< 1 %)
- One or both phases produced `pairs_missing_*` > 0 — report dirs didn't line up
- One or both tasks hit the 180-minute wait ceiling
- Relay / device intermittence corrupted runs

## Comparison Output Format

The validation artifact `.opencode/bench/*_validation.md` MUST include:

```markdown
## A/B Comparison

### Infrastructure
- Relay reachable: yes | no
- Device serial: {serial}
- Windows test workspace: {test_dir}

### Stock Baseline (Phase A)
- Flash result: success | fail
- Async task_id: {task_id}
- Wait time: {elapsed_minutes}m
- Terminal status: succeeded | failed
- report_path: {D:\modelCase_OH_single\reports\report_YYYYMMDDHHMMSS}
- Stdout tail notable lines: {one or two diagnostic lines}

### Feature Candidate (Phase B)
- Flash result: success | fail
- Async task_id: {task_id}
- Wait time: {elapsed_minutes}m
- Terminal status: succeeded | failed
- report_path: {D:\modelCase_OH_single\reports\report_YYYYMMDDHHMMSS}
- Stdout tail notable lines: {one or two diagnostic lines}

### Delta Analysis
- Compare level: total | process | thread | lib | function
- Target: {names at and above the level, or "—" for total}
- Pairs compared: {N}
- Aggregate baseline: {int} instructions
- Aggregate candidate: {int} instructions
- Aggregate delta: {signed int} ({signed pct}%)
- Per-pair breakdown: {table of case/round/step → baseline / candidate / delta / delta_pct}
- New missing targets: {list or "none"}

### Verdict
- Decision: pass | fail | inconclusive | skipped
- Confidence: high | medium | low
- Recommended next route: accept | iterate | reject
- Rationale: {one-paragraph explanation that cites the aggregate.delta_pct and any notable per-pair entries}
```

## Hard Rules (recap)

1. NEVER use `run_instruction_test` (sync). Always use `run_instruction_test_async` + `instruction_test_status` polling.
2. NEVER return to the manager, delegate, or end the session while an async test is still `pending` or `running`.
3. NEVER report a verdict based on only the feature image without a stock baseline.
4. NEVER fabricate comparison numbers if either phase fails. Report the failure and the phase that failed.
5. Both phases MUST use identical test workspace and parameters (same `test_dir`, same `main_script`, same device). The only legitimate delta between phases is the flashed image.
6. If the stock phase fails, report as infrastructure failure — not a patch failure.
7. If the feature phase flashes fine but the test errors, inspect `run_result.stderr_tail`: patch-introduced crashes show up there.
8. The comparison level chosen in Phase B MUST match the metric named in the plan. If the plan targets a specific function, use `compare_level="function"` with the exact names.
