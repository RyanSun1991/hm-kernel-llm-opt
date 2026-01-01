"""Git integration helpers."""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path
from typing import Dict, List


def _run_git(repo: Path, args: list[str]) -> str | None:
    try:
        out = subprocess.check_output(["git", *args], cwd=repo, text=True)
        return out.strip()
    except Exception:
        return None


def get_repo_state(repo_path: str | Path) -> dict:
    repo = Path(repo_path)
    commit = _run_git(repo, ["rev-parse", "HEAD"])
    dirty = bool(_run_git(repo, ["status", "--porcelain"]))
    remote = _run_git(repo, ["config", "--get", "remote.origin.url"])
    return {"commit": commit, "dirty": dirty, "remote": remote}


def snapshot_files(repo_path: str | Path) -> List[dict]:
    """Produce a deterministic snapshot of file hashes."""
    repo = Path(repo_path)
    snapshots: list[dict] = []
    for path in repo.rglob("*"):
        if path.is_file():
            rel = path.relative_to(repo)
            try:
                data = path.read_bytes()
            except Exception:
                continue
            sha = hashlib.sha256(data).hexdigest()
            snapshots.append({"path": str(rel), "sha256": sha, "bytes": len(data)})
    snapshots.sort(key=lambda x: x["path"])
    return snapshots
