---
id: B001
title: "blanket inline of kworker hot entries"
mechanism: inline-callee
aliases: ["force-inline kworker entry", "inline process_one_work"]
target_pattern: "kworker entry functions on cache-constrained devices"
scope: function
applies_to:
  subsystems: ["*"]
  platforms: [phone-X-class]
reason: "Blanket inline of kworker entry hot-path functions blows up i-cache on cache-constrained devices; net function-level regression (+1.4% i-cache miss observed)."
evidence:
  - {kind: bench, ref: bench/wq_threadpool__iter2_validation.md}
  - {kind: review, ref: reviews/wq_threadpool__iter2_code_review.md}
rejected_on: 2026-04-22
rejected_by: kernel-code-reviewer (run r_2026_0422_wq3)
status: active
---

# B001 — blanket inline of kworker hot entries

Blanket inlining of kworker entry hot-path functions (e.g. `process_one_work`)
blows up the instruction cache on cache-constrained devices. The measured net
effect was a function-level regression of +1.4% i-cache miss rate.

This is a *global* bad plan (`applies_to.subsystems: ['*']`): the i-cache
blow-up is device-class driven, not specific to the workqueue subsystem.
Selective inlining of a single proven-hot callee may still be valid — what is
rejected is the *blanket* mechanism.
