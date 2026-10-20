"""Registered Skill snapshots and immutable, bounded source evidence for Agent methods."""

from __future__ import annotations

import hashlib
from pathlib import Path

import yaml

from .mining import _Git, _limit, _path
from .store import ConflictError, digest


def skill_snapshot(service, names: list[str], workspace_root=None) -> dict:
    root = Path(workspace_root or service.workspace_root or "")
    if not root.is_absolute() or root.parts[-3:] != (".opencode", "local", "workspaces"):
        raise ValueError("Configure workspace_root before running Skill-based methods")
    workbench = root.parents[2].resolve()
    skills = workbench / ".opencode" / "skills"
    registry_path = skills / "_registry.yaml"
    if (
        not registry_path.is_file()
        or registry_path.resolve() != registry_path
        or registry_path.stat().st_size > 1024 * 1024
    ):
        raise ValueError("Skill registry is missing or too large")
    registry = yaml.safe_load(registry_path.read_text(encoding="utf-8"))["skills"]
    entries = {entry["name"]: entry for entry in registry}
    if len(entries) != len(registry) or not 1 <= len(names) <= 4 or len(names) != len(set(names)):
        raise ValueError("Expected one to four distinct registered Skills")
    snapshots = []
    for name in names:
        if name not in entries:
            raise ValueError(f"Unknown registered Skill: {name}")
        entry = entries[name]
        relative = _path(f"{entry['tier']}/{name}/SKILL.md")
        path = skills / relative
        if not path.is_file() or path.resolve() != path or path.stat().st_size > 65536:
            raise ValueError("Skill must be a bounded regular file inside the configured workbench")
        content = path.read_text(encoding="utf-8")
        if not content.strip():
            raise ValueError("Skill content is empty")
        snapshots.append(
            {
                "name": name,
                "path": str(path),
                "content": content,
                "sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
            }
        )
    bundle = {
        "schema_version": 1,
        "skills": snapshots,
        "authority": "Method instructions only; role and service gates remain authoritative.",
    }
    with service.store.transaction() as db:
        sha = service.store.evidence(db, bundle)
    return {"sha256": sha, **bundle}


def code_context(service, repo, revision, path, *, start_line=1, line_count=200) -> dict:
    """Read source from Git objects, never execute code or substitute working-tree contents."""
    _limit(start_line, "start_line", 10_000_000)
    _limit(line_count, "line_count", 400)
    path = _path(path)
    git = _Git(Path(repo), service.git_bin)
    git.repo_id = service.repo_identity(git.root)
    commit = git.revision(revision)
    raw, cut = git.read(["ls-tree", "-z", commit, "--", ":(literal)" + path], 16384)
    if cut or not raw:
        raise ValueError("Context path must identify one existing source file")
    metadata, actual = raw.rstrip(b"\x00").split(b"\t", 1)
    mode, kind, oid = metadata.decode("ascii").split()
    if mode not in {"100644", "100755"} or kind != "blob" or actual.decode("utf-8") != path:
        raise ValueError("Only regular source blobs can be read as code context")
    raw, cut = git.read(["cat-file", "blob", oid], 4 * 1024 * 1024)
    if cut or b"\x00" in raw:
        raise ValueError("Context source is binary or exceeds 4 MiB; partition its investigation")
    lines = raw.decode("utf-8", errors="strict").splitlines()
    if start_line > max(1, len(lines)):
        raise ValueError("start_line is outside the source file")
    selected = lines[start_line - 1 : start_line - 1 + line_count]
    # A pathological single line must not defeat response budgeting.
    if sum(len(line.encode("utf-8")) for line in selected) > 65536:
        raise ValueError("Context window exceeds 64 KiB; request fewer lines")
    context = {
        "schema_version": 1,
        "kind": "code_context",
        "repo_id": git.repo_id,
        "revision": commit,
        "path": path,
        "object_id": oid,
        "source_sha256": hashlib.sha256(raw).hexdigest(),
        "start_line": start_line,
        "lines": selected,
        "total_lines": len(lines),
        "next_line": start_line + len(selected)
        if start_line + len(selected) <= len(lines)
        else None,
    }
    with service.store.transaction() as db:
        sha = service.store.evidence(db, context)
    return {"sha256": sha, **context}


def verify_citation(service, citation, *, repo_id, revision) -> None:
    context = service.store.read_evidence(citation["context_sha256"])
    if (
        context.get("kind") != "code_context"
        or context["repo_id"] != repo_id
        or context["revision"] != revision
    ):
        raise ConflictError("Citation belongs to a different repository or revision")
    index = citation["line_start"] - context["start_line"]
    quote = citation["quote"].splitlines()
    if index < 0 or not quote or context["lines"][index : index + len(quote)] != quote:
        raise ValueError("Citation does not match exact immutable source lines")


def verify_snapshot(service, sha):
    value = service.store.read_evidence(sha)
    if digest(value) != sha or not value.get("skills"):
        raise ConflictError("Invalid Skill snapshot")
    return value
