"""Transactional, restartable workflow records and content-addressed evidence."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator


class ConflictError(ValueError):
    """A stale version or reused request ID would change an existing decision."""


def canonical_json(value: Any) -> str:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    )


def digest(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


class EvolutionStore:
    def __init__(self, root: str | Path):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.db_path = self.root / "evolution.sqlite3"
        with self.transaction() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS records (
                    kind TEXT NOT NULL, id TEXT NOT NULL, payload TEXT NOT NULL,
                    version INTEGER NOT NULL DEFAULT 1,
                    PRIMARY KEY(kind, id)
                );
                CREATE TABLE IF NOT EXISTS evidence (
                    sha256 TEXT PRIMARY KEY, content TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS events (
                    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
                    entity_id TEXT NOT NULL, action TEXT NOT NULL, actor TEXT NOT NULL,
                    details TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS requests (
                    request_id TEXT PRIMARY KEY, input_digest TEXT NOT NULL, result TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS cursors (
                    repo_path TEXT PRIMARY KEY, revision TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS outcomes (
                    fingerprint TEXT PRIMARY KEY, candidate_id TEXT NOT NULL, outcome TEXT NOT NULL
                );
            """)

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        db = sqlite3.connect(self.db_path, timeout=30, isolation_level=None)
        db.row_factory = sqlite3.Row
        try:
            db.execute("PRAGMA journal_mode=WAL")
            db.execute("BEGIN IMMEDIATE")
            yield db
            db.commit()
        except BaseException:
            db.rollback()
            raise
        finally:
            db.close()

    @staticmethod
    def get(db: sqlite3.Connection, kind: str, record_id: str) -> dict:
        row = db.execute(
            "SELECT payload,version FROM records WHERE kind=? AND id=?", (kind, record_id)
        ).fetchone()
        if row is None:
            raise ValueError(f"Unknown {kind}: {record_id}")
        return {"id": record_id, "version": row["version"], "data": json.loads(row["payload"])}

    @staticmethod
    def put(
        db: sqlite3.Connection,
        kind: str,
        record_id: str,
        data: dict,
        expected_version: int | None = None,
    ) -> dict:
        payload = canonical_json(data)
        row = db.execute(
            "SELECT version,payload FROM records WHERE kind=? AND id=?", (kind, record_id)
        ).fetchone()
        if row is None:
            if expected_version is not None:
                raise ConflictError("Record does not exist at the expected version")
            db.execute(
                "INSERT INTO records(kind,id,payload) VALUES(?,?,?)", (kind, record_id, payload)
            )
        elif expected_version is None:
            if row["payload"] != payload:
                raise ConflictError(f"Immutable record already exists: {kind}/{record_id}")
        elif row["version"] != expected_version:
            raise ConflictError(
                f"Stale version: expected {expected_version}, current {row['version']}"
            )
        else:
            db.execute(
                "UPDATE records SET payload=?,version=version+1 WHERE kind=? AND id=?",
                (payload, kind, record_id),
            )
        return EvolutionStore.get(db, kind, record_id)

    @staticmethod
    def evidence(db: sqlite3.Connection, content: Any) -> str:
        text = canonical_json(content)
        sha = hashlib.sha256(text.encode("utf-8")).hexdigest()
        db.execute("INSERT OR IGNORE INTO evidence(sha256,content) VALUES(?,?)", (sha, text))
        return sha

    @staticmethod
    def event(db: sqlite3.Connection, entity: str, action: str, actor: str, details: dict) -> None:
        db.execute(
            "INSERT INTO events(entity_id,action,actor,details) VALUES(?,?,?,?)",
            (entity, action, actor, canonical_json(details)),
        )

    @staticmethod
    def replay(db: sqlite3.Connection, request_id: str, request: dict) -> dict | None:
        if not request_id.strip():
            raise ValueError("A nonempty idempotency request_id is required")
        row = db.execute("SELECT * FROM requests WHERE request_id=?", (request_id,)).fetchone()
        if row:
            if row["input_digest"] != digest(request):
                raise ConflictError("request_id already used with different arguments")
            return json.loads(row["result"])
        return None

    @staticmethod
    def remember(db: sqlite3.Connection, request_id: str, request: dict, result: dict) -> None:
        db.execute(
            "INSERT INTO requests VALUES(?,?,?)",
            (request_id, digest(request), canonical_json(result)),
        )

    def list(self, kind: str, *, limit: int = 100, offset: int = 0) -> list[dict]:
        if not 1 <= limit <= 1000 or offset < 0:
            raise ValueError("limit must be 1..1000 and offset nonnegative")
        with self.transaction() as db:
            rows = db.execute(
                "SELECT id FROM records WHERE kind=? ORDER BY rowid LIMIT ? OFFSET ?",
                (kind, limit, offset),
            ).fetchall()
            return [self.get(db, kind, row["id"]) for row in rows]

    def read(self, kind: str, record_id: str) -> dict:
        with self.transaction() as db:
            return self.get(db, kind, record_id)

    def read_evidence(self, sha256: str) -> Any:
        with self.transaction() as db:
            row = db.execute("SELECT content FROM evidence WHERE sha256=?", (sha256,)).fetchone()
            if row is None:
                raise ValueError("Unknown evidence digest")
            content = json.loads(row["content"])
            if digest(content) != sha256:
                raise ConflictError("Evidence integrity check failed")
            return content

    def audit(self, entity_id: str, *, limit: int = 1000) -> list[dict]:
        if not 1 <= limit <= 1000:
            raise ValueError("limit must be 1..1000")
        with self.transaction() as db:
            rows = db.execute(
                "SELECT * FROM events WHERE entity_id=? ORDER BY sequence DESC LIMIT ?",
                (entity_id, limit),
            ).fetchall()
            return [{**dict(row), "details": json.loads(row["details"])} for row in reversed(rows)]
