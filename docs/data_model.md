# Data Model (Draft)

Entities (initial draft):

- Project
- RepoSnapshot (commit, diff)
- Workload
- Run (experiment instance)
- Artifact (trace/log/report/binary)
- MetricSeries / MetricPoint
- CodeElement (file/function/struct)
- GraphEdge (static/runtime)
- Hotspot
- Patch (diff + rationale)
- AgentMessage (prompt/response/tool calls)

All entities must be versioned and linked to a `Run` for reproducibility.
