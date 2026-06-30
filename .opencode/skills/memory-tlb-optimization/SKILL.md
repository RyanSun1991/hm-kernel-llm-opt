---
name: memory-tlb-optimization
description: Playbook for memory-management syscalls (mprotect / madvise / munmap / mmap / mlock / brk). Hot path is VMA lookup/split-merge → page-table update → TLB maintenance, where TLB (esp. SMP shootdown) is usually dominant. Primary metric is TLB/page-walk counters + measured latency, not instruction count. Selected by the perf-bottleneck classification gate.
---

# Memory / TLB Optimization Playbook (`memory-tlb-bound`)

The `memory-tlb-bound` entry in `perf-bottleneck-playbooks`. Applies when the
classification gate finds the hot path dominated by VMA + page-table + TLB work —
the lmbench memory-syscall group (`lat_syscall mprotect`, `lat_syscall madvise`,
plus `munmap`, `mmap`, `mlock`, `brk`). It carries only the domain-specific 10%;
pipeline, gates, handoff, and the lmbench A/B run come from the shared skills.

## Step 0 — confirm advice / prot FIRST (decides everything)

Cost spans ~10-50× depending on the argument. Before ideation, confirm what the
benchmark actually issues (grep the `lat_syscall` source for `MADV_*` / `PROT_*`):

| `madvise` advice | Kernel work | Cost |
|---|---|---|
| `MADV_NORMAL/RANDOM/SEQUENTIAL` | set VMA readahead flag only | cheap — VMA lookup dominates |
| `MADV_DONTNEED` | tear down PTEs + free pages + **TLB flush** | expensive, scales with len |
| `MADV_FREE` | mark lazily reclaimable | medium |

| `mprotect` | Kernel work | Cost |
|---|---|---|
| single-page `R ↔ RW` | VMA split + 1 PTE perm change + **TLB flush** | medium, TLB-dominated |
| large / multi-VMA range | repeated split/merge + bulk PTE + TLB | scales with range |

A cheap-flag path is really a VMA-lookup micro-op (closer to `compute-bound`); the
teardown/perm paths are TLB-dominated. Classify accordingly.

## Hot-path taxonomy

```
mprotect/madvise → lsyscall hook
  ① VMA lookup (addr → region)
  ② VMA split / merge (changing the middle of a region → split into ≤3)
  ③ page-table walk + PTE update (perm bits / clear PTE)
  ④ TLB maintenance            ← usually the dominant cost
  ⑤ capability check (caller's cap to the memory object)
  ⑥ memmgr/sysmgr IPC, if VMA metadata is owned in userspace (secondary, ipc-bound)
```

## Primary metric & IC's partial role

Primary metric = **TLB / page-walk performance counters + measured lmbench
latency**, NOT instruction count. Split honestly:

- **IC-trackable** (use whole-path IC as a secondary signal): ② VMA split/merge,
  ⑤ capability check, ⑥ IPC marshalling.
- **IC-blind** (must read counters / measured latency): ④ TLB flush — a single
  `TLBI` is one instruction but drains the pipeline and stalls remote cores; ③
  page-table walk is a few instructions but several cache-line misses.
- **Invisible to initiating-core IC**: SMP **TLB shootdown** — the IPI'd remote
  cores' stall never appears in the caller's instruction count.

Rule: never declare a win on IC alone for this class. Confirm with TLB counters
and/or the lmbench memory group beyond the 2% noise floor.

## Optimization moves (ranked by leverage)

1. **TLB maintenance — the biggest lever (esp. SMP).**
   - prefer ARM broadcast `TLBI` over IPI-driven per-core shootdown;
   - **ASID/VMID-tagged** TLB so address-space changes don't force a full flush;
   - **range-based `TLBI`** — flush only the changed range, not the whole ASID;
   - **batch / defer** — coalesce multiple PTE edits in one syscall into a single
     flush; defer where correctness allows.
2. **VMA data structure** — lookup + split/merge complexity (rbtree / maple-tree
   style); avoid needless split/merge; cache-friendly node layout.
3. **Page-table walk** — cut cache misses; bulk PTE updates; larger page
   granularity to shorten walks.
4. **memmgr IPC (if ⑥ applies)** — same structural round-trip as `ipc-bound`:
   cache/batch VMA metadata client-side, short-circuit the memmgr round-trip.
5. **Capability check** — cap-lookup cost on the hot path.

## HongMeng source anchors (confirm on the tree; inferred from the `fcntl/` layout)

| Layer | Likely location |
|---|---|
| lsyscall hook | `kernel/extensions/lsyscall/mprotect/mprotect.c`, `.../madvise/madvise.c` (peer of `fcntl/`) |
| address space / VMA | `kernel/mem/...` or `aspace`/`vspace` (VMA split/merge) |
| arch page table + **TLB** | `arch/aarch64/.../pgtable`, `.../tlb` (`TLBI`) |
| userspace memory mgr | `sysmgr/memmgr/` (the hyperhold/memmgr profile target) |

Decisive greps: `ls kernel/extensions/lsyscall/ | grep -E 'madvise|mprotect'`;
`grep -rniE 'tlbi|flush_tlb|tlb_flush|shootdown' arch/aarch64`; check whether the
hook sets `info->server`/`rpc_info` (⇒ memmgr round-trip, like `fcntl`) or does the
work inline.

## Validation tie-in

Validate via `ab-test-comparison-lmbench` on the **memory group** specifically,
plus TLB counters where available. Report in `<target>_validation.md` with the
classification (`memory-tlb-bound`) and the primary metric used, so the verdict is
judged on TLB/latency — not on an IC number that can't see the flush.
