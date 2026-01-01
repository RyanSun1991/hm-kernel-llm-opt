# Pipeline

## Default pipeline: optimize_kernel

1. `ingest_repo`
2. `ingest_perf_artifacts`
3. `static_analysis`
4. `runtime_analysis`
5. `correlate_code_and_perf`
6. `agent_loop` (iterate until convergence)
7. `persist_results`
8. `export_training_dataset`
