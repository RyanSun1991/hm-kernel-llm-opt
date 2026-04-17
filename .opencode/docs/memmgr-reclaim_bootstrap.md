# Memmgr Reclaim Bootstrap

This file is the reusable starting context for memmgr reclaim work in this repository.

## Primary Scope

Key directories and files:

- `sysmgr/memmgr/include/reclaim/`
- `sysmgr/memmgr/mem/reclaim/`
- `sysmgr/memmgr/mem/stat/`
- `sysmgr/memmgr/mem/swap/`
- `sysmgr/memmgr/page/`
- `sysmgr/memmgr/psi/`
- `sysmgr/memmgr/mem/vmpressure.c`

## Reclaim Entry

The working assumption from the existing repo notes is:

- page allocation slow path reaches reclaim from `page/palloc.c`
- sync reclaim is triggered directly from allocation pressure
- async reclaim is handled by reclaim threads

Future sessions should verify exact symbols with Kernel Index MCP before relying on this summary.

## Core Questions To Answer

For any reclaim task, establish:

- where reclaim is triggered
- how sync and async reclaim differ
- what reclaim instances exist
- how watermark logic and pressure signals interact
- how page allocator slow paths depend on reclaim behavior

## Pressure Signals To Check

- reclaim watermarks
- `vmpressure`
- `psi`
- swap or zswap pressure
- any external reporting or procfs state

## Expected Outputs

The reclaim specialist should maintain:

- `memmgr-reclaim_design.md`
- `memmgr-reclaim_trace.md`

When new stable knowledge is discovered, add it back here.
