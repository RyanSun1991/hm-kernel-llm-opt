# A/B Test Comparison Protocol

This skill defines the mandatory A/B (stock vs feature) test comparison protocol. When the tester agent validates a patch, it MUST run both a stock baseline test and a feature candidate test, then compare results.

## Mandatory A/B Sequence

The tester MUST execute these phases in order. Skipping any phase is FORBIDDEN unless the phase explicitly fails and cannot proceed.

### Phase A: Stock Baseline

1. Flash the STOCK image to the device using Flash MCP `flash_and_boot`.
2. Wait for the device to complete boot.
3. Run the test suite using Auto-Test MCP `phone_test_run`.
4. Retrieve and store results as `stock` baseline artifacts.
5. Label all collected metrics with prefix `baseline_`.

### Phase B: Feature Candidate

1. Flash the FEATURE image (with optimization patch) using Flash MCP `flash_and_boot`.
2. Wait for the device to complete boot.
3. Run the SAME test suite using Auto-Test MCP `phone_test_run` with identical parameters.
4. Retrieve and store results as `feature` candidate artifacts.
5. Label all collected metrics with prefix `candidate_`.

### Phase C: Comparison

1. Parse baseline and candidate result artifacts.
2. Compute instruction-count delta (or approved proxy metric delta).
3. If flamegraph/hitrace/hiperf artifacts are available, perform differential analysis.
4. Determine the verdict based on Decision Criteria below.

## Decision Criteria

### PASS

ALL of the following must hold:

- Feature instruction count on hot path <= Stock instruction count on hot path
- No new top-20 hotspot introduced by the patch
- Test correctness maintained (no crashes, no functional failures)
- Both flash operations succeeded
- Both test runs completed successfully

### FAIL

ANY of the following triggers a fail:

- Feature instruction count on hot path > Stock instruction count (regression)
- New high-rank hotspot introduced by the patch
- Functional regression detected (crash, hang, test failure)
- Feature image flash failed but stock succeeded (patch may have broken the image)

### INCONCLUSIVE

- One or both tests produced noisy or unreliable data
- Delta is within noise margin (less than 1% difference)
- Missing artifacts prevent meaningful comparison
- Both flash operations failed (infrastructure issue, not patch issue)

## Comparison Output Format

The validation artifact `.opencode/bench/*_validation.md` MUST include:

```markdown
## A/B Comparison

### Stock Baseline
- Flash result: success | fail
- Test result: pass | fail
- Key metrics: {instruction count, top-5 functions, duration}
- Artifact paths: {local paths to stock results}

### Feature Candidate
- Flash result: success | fail
- Test result: pass | fail
- Key metrics: {instruction count, top-5 functions, duration}
- Artifact paths: {local paths to feature results}

### Delta Analysis
- Instruction count delta: -X% (improvement) or +X% (regression)
- Hot path changes: {function-level summary}
- New hotspots: {list or "none"}
- Flamegraph diff: {path to diff artifact or "not available"}

### Verdict
- Decision: pass | fail | inconclusive
- Confidence: high | medium | low
- Recommended next route: accept | iterate | reject
- Rationale: {one-paragraph explanation}
```

## Hard Rules

1. It is FORBIDDEN to report a test result based on only the feature image without a stock baseline.
2. It is FORBIDDEN to fabricate comparison data if either flash or test fails.
3. If the stock flash fails, report infrastructure failure — not a patch failure.
4. If the feature flash fails but stock succeeded, this MAY indicate a patch-introduced build issue.
5. Both test runs MUST use identical parameters (same test case, same duration, same device).
6. The tester MUST NOT modify the test parameters between stock and feature runs.
