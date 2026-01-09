# Architecture

This document describes the target architecture for the HM-VERIF optimization platform.

## Design Principles

- **Closed-loop optimization**: propose → build/test → profile → analyze → iterate.
- **Profiler-guided reasoning**: convert low-level metrics to actionable optimization hints.
- **Project-level context**: reason about call graphs and cross-file dependencies.
- **Local-first storage**: all artifacts and metadata stored in a local database.
- **Reproducibility**: every run is an immutable experiment with versioned artifacts.
- **Security-by-default**: internal models only unless explicitly allowed.

## High-level Components

- Ingestion
  - Repo snapshotting (git)
  - Static index (AST/CFG/callgraph)
  - Config + hardware spec capture

- Profiling
  - Trace collection (hitrace/hiperf/perf/flamegraph)
  - Normalization to a common schema

- Analysis
  - Hotspot detection & ranking
  - Bottleneck classification
  - Code↔trace correlation

- Agentic Optimization (LangGraph)
  - Conductor (planner/orchestrator)
  - Trace Analyst (performance reasoning)
  - Coder (patch generation)
  - Verifier (build/test)
  - Profiler (run + collect metrics)
  - Safety/Policy (data leakage prevention)

- Storage
  - Relational DB (runs, artifacts, metrics, patches)
  - Vector store (embeddings for RAG)

- Interfaces
  - CLI for batch workflows
  - REST API for UI + automation

