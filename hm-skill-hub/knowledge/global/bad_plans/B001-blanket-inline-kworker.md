# Global Bad Plans

Globally rejected mechanism × scope combinations. Stable id prefix `B`. Append-only.

---

### B001 — blanket inline of kworker hot entries
- **mechanism**: inline-callee
- **aliases**: ["force-inline kworker entry", "inline process_one_work"]
- **target_pattern**: kworker entry functions on cache-constrained devices
- **scope**: function
- **applies_to**:
    subsystems: [workqueue-threadpool]
    platforms: [phone-X-class]
- **reason**: Blanket inline of kworker entry hot-path functions blows up
    i-cache on cache-constrained devices; net function-level regression
    (+1.4% i-cache miss observed).
- **evidence**:
    - {kind: bench, ref: bench/wq_threadpool__iter2_validation.md}
    - {kind: review, ref: reviews/wq_threadpool__iter2_code_review.md}
- **rejected_on**: 2026-04-22
- **rejected_by**: kernel-code-reviewer (run r_2026_0422_wq3)
- **status**: active
