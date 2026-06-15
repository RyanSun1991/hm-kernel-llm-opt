# best_skill — instruction-count-first (v0.1.0, SEED)

Match the hot pattern to a mechanism, apply, re-measure. Seeded incomplete: it
covers some mechanisms but not all the suite's patterns, so the optimizer has
room to improve it under the eval gate.

## Mechanism playbook

- **Coarse lock on the hot path**: split the lock into per-domain / per-shard
  finer locks (mechanism: lock-split). Re-measure contention at process level.
- **Many small operations / round-trips**: coalesce them into one batch round-trip
  (mechanism: batch-coalesce). Watch the batch size vs latency trade.

## Discipline

- Always state a hot-path delta thesis before spending an iteration.
- Re-measure at the right `compare_level` (total / process / function); never
  compare across levels.
