---
name: domain-hyperhold-io
description: >-
  Domain pack for hyperhold / swap I/O — hpio, iotab, eid mapping, inflight state,
  serialization/wait paths, and compression branches. Pure domain knowledge extracted
  from the legacy hyperhold-io-opt specialist: any role loads it. No process rules.
---

# Domain Pack — hyperhold / swap I/O

## Scope (path anchors)

- `sysmgr/memmgr/mem/swap/hyperhold/**` — the whole subsystem
- `hpio` — the per-request I/O descriptor and its pooling/lifecycle
- `iotab` — the I/O table: slot allocation, lookup, inflight tracking
- `eid` — entry-id mapping between swap slots and backing storage extents
- compression vs non-compression branches on the read/write paths

## Read these before exploring from scratch

1. `.opencode/docs/hyperhold_io_design.md` if present (keep it living)
2. `.opencode/docs/memmgr-reclaim_bootstrap.md` — reclaim is the main producer of
   swap-out traffic; the two subsystems meet at the swap entry points
3. `.opencode/memory/subsystems/` + `.opencode/memory/targets/<target>.md` as named
4. Reject ledgers: `.opencode/state/bad_plans.md` and
   `.opencode/state/hyperhold-io-opt_bad_plans.md` if present

## The system model a claim must rest on

- **Request lifecycle**: how an hpio is allocated, filled, submitted, completed, and
  recycled — and which fields are valid at each stage
- **iotab discipline**: slot alloc/free, lookup on completion, inflight accounting —
  what the table lock actually protects at each site
- **eid mapping**: slot ↔ extent translation, batching behavior, where translation
  cost lands (submit vs completion)
- **Serialization and wait paths**: who waits on inflight I/O, wakeup ordering, and
  which waits are per-request vs global
- **Compression branches**: when data takes the compressed path, where the
  compress/decompress cost sits relative to the I/O, and which branches are dead in
  the current product configuration

## Optimization-sensitive spots (where cost usually hides)

- inflight-state bookkeeping updated more than once per request
- iotab lookups or lock round-trips repeated within one request's path
- serialization points where per-request waits could batch
- compression-path branches evaluated per-page where the answer is per-request
  constant
- eid translations recomputed instead of carried through the request

## Domain-specific review cautions

- Completion runs asynchronously to submission: any field read on the completion
  path must be published before submit — "removing a redundant store" here can be
  removing the publish.
- Inflight accounting drives throttling and shutdown correctness; an accounting
  optimization that can miss a decrement leaks a wait forever.
- Dead-looking compression branches may be product-config-dependent — confirm via
  the product config before excising, and record the config evidence.
