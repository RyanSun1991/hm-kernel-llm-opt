---
name: ab-test-comparison-lmbench
description: lmbench full-suite A/B validation protocol (the default test method). Flash stock, run the lmbench suite, flash feature, run it again; the second run's vs-previous delta is the patch A/B result, with HM-vs-Linux as competitive context. Verdict uses the benchmark delta with a noise floor, not instruction count.
---

# A/B Test Comparison — lmbench full suite (default test method)

This is the **default** tester method (`test_method: lmbench-suite`). It measures
real microbenchmark performance (memory bandwidth, latency, syscall, IPC, fio…)
on the device via the Windows relay, instead of static instruction count. Use the
`instruction-count` method (`ab-test-comparison/SKILL.md`) when you specifically
want fast per-function IC A/B for a micro-optimization.

## Why a different verdict than instruction-count

Instruction count is a static proxy; lmbench is a dynamic, noisy measurement of
actual throughput/latency. So: the PASS threshold has a **noise floor (~2%)**, the
comparison is **per-benchmark-group** (not per-function), and "better" is
**direction-aware** (bandwidth higher-is-better, latency lower-is-better — the
digest already normalizes this).

## Long run, do NOT block

The full suite runs **2-5 hours per pass**. The Auto-Test MCP launches it detached
on Windows and polls; you drive it through the async tools and **poll on a slow
cadence** (every few minutes), never a tight loop.

| MCP tool (Auto-Test MCP) | use |
|---|---|
| `run_lmbench_test_async(...)` | start a suite run; returns `{task_id, kind:"run_lmbench_test"}` |
| `lmbench_test_status(task_id)` | poll; when `result.status == "done"`, `result.digest` holds the analysis |

`run_lmbench_test_async` accepts `test_dir` (default `D:\LmbenchAutoTest`),
`run_cmd` (default `main.py`), `top_n`, and timeout knobs; defaults are fine.

## A/B protocol (strict)

The framework keeps **timestamped** result files, so two back-to-back runs give a
clean A/B: the second run's `digest.vs_previous` compares it against the first.

1. **Flash stock** (Flash MCP `flash_stock` / `flash_stock_async`) + settle (~10 min).
2. **Run lmbench on stock** — `run_lmbench_test_async()` → poll `lmbench_test_status`
   until `result.status == "done"`. This stock run becomes the **previous** result.
   (Discard its `vs_previous`; only its absolute numbers + HM-vs-Linux matter here.)
3. **Flash feature** (Flash MCP `flash_feature`) + settle (~10 min).
4. **Run lmbench on feature** — `run_lmbench_test_async()` → poll until done. Now
   `result.digest.vs_previous` = **stock → feature delta = the patch A/B result**.
5. **Verdict** from `result.digest`:
   - `vs_previous.regressed` / `vs_previous.improved` counts + `top_regressions` /
     `top_improvements` (direction-normalized `improvement_pct`).
   - `hm_vs_linux.overall_weighted_gap_pct` + `top_regressions` = competitive context.
   - `anomalies` = high-dispersion metrics — discount changes on those.

## Verdict criteria

- **pass** — no benchmark group regressed beyond the **2% noise floor** (i.e. every
  `vs_previous.top_regressions[*].improvement_pct >= -2%`, ignoring `anomalies`),
  and the overall picture is flat-or-better.
- **fail** — a non-anomalous benchmark group regressed by **> 2%** (real slowdown
  the patch caused), or a run crashed / produced no result.
- **inconclusive** — only anomalous (high-dispersion) metrics moved, or one run
  failed to produce a `total_result` xlsx, or all deltas are within ±2%.
- **skipped** — infra failure (relay unreachable, flash failed, device offline);
  report to the manager, not a patch/plan problem.

## Tester → Manager report

Write `.opencode/bench/<artifact>_validation.md` with the SAME contract as the
instruction-count method (verdict, confidence, recommended_next_route) so the
manager routes uniformly — only the table differs:

- header cites `test_method: lmbench-suite` and the 2% noise floor.
- **per-benchmark-group table**: `group | stock | feature | delta% | improvement% | direction`
  (top regressions + top improvements from `digest.vs_previous`).
- **HM-vs-Linux line**: overall weighted gap + worst vs-Linux items (from `digest.hm_vs_linux`).
- **anomalies** discounted, listed explicitly.
- paste the compact `digest` (not raw xlsx) for the manager/human.

Under iterative mode, do not switch `test_method` mid-iteration (keep it in
`current_task.json`); stock-flash in pass K+1 already includes prior-pass patches,
so the baseline shift is correct.
