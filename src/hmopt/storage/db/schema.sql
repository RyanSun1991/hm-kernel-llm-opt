-- Canonical SQLite schema for HMOPT. ORM creates tables too; this file
-- is used as a bootstrap reference and for environments without Alembic.

CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    parent_run_id TEXT,
    repo_uri TEXT,
    repo_rev TEXT,
    repo_dirty INTEGER DEFAULT 0,
    workload_id TEXT,
    device_id TEXT,
    toolchain_id TEXT,
    status TEXT,
    created_at TEXT,
    finished_at TEXT
);

CREATE TABLE IF NOT EXISTS artifacts (
    artifact_id TEXT PRIMARY KEY,
    run_id TEXT,
    kind TEXT,
    sha256 TEXT,
    path TEXT,
    bytes INTEGER,
    mime TEXT,
    metadata_json TEXT
);

CREATE TABLE IF NOT EXISTS metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT,
    scope TEXT,
    metric_name TEXT,
    value REAL,
    unit TEXT,
    tags_json TEXT
);

CREATE TABLE IF NOT EXISTS hotspots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT,
    symbol TEXT,
    file_path TEXT,
    line_start INTEGER,
    line_end INTEGER,
    score REAL,
    evidence_artifact_ids TEXT,
    call_stacks_json TEXT
);

CREATE TABLE IF NOT EXISTS graphs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT,
    kind TEXT,
    format TEXT,
    payload_artifact_id TEXT,
    metadata_json TEXT
);

CREATE TABLE IF NOT EXISTS patches (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT,
    iteration INTEGER,
    diff_artifact_id TEXT,
    apply_status TEXT,
    files_changed_json TEXT
);

CREATE TABLE IF NOT EXISTS evaluations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT,
    baseline_run_id TEXT,
    delta_metrics_json TEXT,
    correctness_passed INTEGER,
    perf_improved INTEGER,
    notes TEXT
);

CREATE TABLE IF NOT EXISTS agent_messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT,
    iteration INTEGER,
    agent_name TEXT,
    model_id TEXT,
    prompt_artifact_id TEXT,
    output_artifact_id TEXT,
    summary_artifact_id TEXT
);

CREATE TABLE IF NOT EXISTS vector_embeddings (
    id TEXT PRIMARY KEY,
    kind TEXT,
    ref_id TEXT,
    run_id TEXT,
    embedding_json TEXT,
    metadata_json TEXT,
    created_at TEXT
);
