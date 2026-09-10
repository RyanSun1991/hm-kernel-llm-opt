"""Bounded, read-only history mining and literal candidate discovery.

All Git reads use a resolved commit and raw tree/blob objects, never worktree
contents. Draft distillation is deliberately a removed-line heuristic: it does
not infer semantic equivalence or claim that a historical change improved a
metric. Human curation and activation are required before scanning a pattern.

Globs are anchored at the repository root. ``*.c`` matches root files and
``**/*.c`` also matches nested files. Only ``*``, ``?`` and whole-segment ``**``
wildcards are accepted. Owner rules use most literal characters, then fewest
wildcards, then lexical glob order. A pipeline lane is a prioritization hint,
never authorization to change code or dispatch a job.
"""

from __future__ import annotations

import fnmatch
import hashlib
import heapq
import json
import os
import subprocess
import threading
import time
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

_Id = Annotated[str, Field(min_length=1, max_length=160)]
_Text = Annotated[str, Field(min_length=1, max_length=4000)]
_Short = Annotated[str, Field(min_length=1, max_length=256)]
_MAX_HISTORY_BYTES = 24 * 1024 * 1024
_MAX_SCAN_BYTES = 64 * 1024 * 1024
_HEURISTIC = "HEURISTIC:"
_TRUNCATED = "TRUNCATED:"
_SOURCE_SUFFIXES = {".c", ".h", ".cc", ".cpp", ".cxx", ".hpp", ".rs", ".py", ".go"}


def _plain(value: str) -> str:
    if not value.strip() or any(ord(c) < 32 or ord(c) == 127 for c in value):
        raise ValueError("Expected nonblank text without control characters")
    return value


def _path(value: str, *, glob: bool = False) -> str:
    if not isinstance(value, str) or not value or len(value) > 4096:
        raise ValueError("Expected a bounded repository-relative POSIX path")
    _plain(value)
    if value.startswith("/") or "\\" in value or ":" in value:
        raise ValueError("Absolute paths, drive paths and backslashes are not accepted")
    parts = value.split("/")
    if len(parts) > 128:
        raise ValueError("Paths may contain at most 128 segments")
    if any(part in {"", ".", ".."} for part in parts):
        raise ValueError("Empty or traversal path segments are not accepted")
    if glob:
        if len(value) > 512 or any(c in value for c in "[]{}"):
            raise ValueError("Only bounded *, ?, and whole-segment ** globs are accepted")
        if any("**" in part and part != "**" for part in parts):
            raise ValueError("** must occupy an entire path segment")
    elif any(c in value for c in "*?[]"):
        raise ValueError("A concrete path cannot contain glob metacharacters")
    return value


def _sha(value: str) -> str:
    if len(value) not in {40, 64} or any(c not in "0123456789abcdef" for c in value):
        raise ValueError("Revision must be a full lowercase Git object ID")
    return value


def _digest(value: object, prefix: str) -> str:
    payload = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return prefix + hashlib.sha256(payload.encode("utf-8")).hexdigest()


class _Record(BaseModel):
    model_config = ConfigDict(strict=True, frozen=True, extra="forbid", validate_default=True)


class ChangeRecord(_Record):
    source_id: _Id
    repo_id: _Id
    revision: str
    parent_revision: str | None = None
    subject: Annotated[str, Field(max_length=1024)]
    message: Annotated[str, Field(max_length=16384)]
    paths: Annotated[list[str], Field(max_length=1024)]
    patch: Annotated[str, Field(max_length=1048576)]
    review_notes: Annotated[list[_Text], Field(max_length=64)] = Field(default_factory=list)
    truncated: bool = False

    _revision = field_validator("revision")(_sha)

    @field_validator("parent_revision")
    @classmethod
    def valid_parent(cls, value: str | None) -> str | None:
        return _sha(value) if value is not None else None

    @field_validator("paths")
    @classmethod
    def valid_paths(cls, values: list[str]) -> list[str]:
        return sorted({_path(value) for value in values})


class Matcher(_Record):
    file_globs: Annotated[list[str], Field(min_length=1, max_length=32)]
    all_of: Annotated[list[_Text], Field(min_length=1, max_length=32)]
    any_of: Annotated[list[_Text], Field(max_length=32)] = Field(default_factory=list)
    none_of: Annotated[list[_Text], Field(max_length=32)] = Field(default_factory=list)

    @field_validator("file_globs")
    @classmethod
    def valid_globs(cls, values: list[str]) -> list[str]:
        return sorted({_path(value, glob=True) for value in values})

    @field_validator("all_of", "any_of", "none_of")
    @classmethod
    def valid_literals(cls, values: list[str]) -> list[str]:
        if any(not value.strip() or "\x00" in value for value in values):
            raise ValueError("Literal predicates cannot be blank or contain NUL")
        return sorted(set(values))

    @model_validator(mode="after")
    def consistent(self) -> Matcher:
        if set(self.all_of) & set(self.none_of):
            raise ValueError("A required literal cannot also be forbidden")
        return self


class Pattern(_Record):
    pattern_id: _Id
    version: Annotated[int, Field(ge=1, le=1000000)] = 1
    title: _Short
    kind: Literal["optimization", "diagnostic", "anti_pattern"]
    problem: _Text
    diagnosis: _Text
    remedy: _Text
    matcher: Matcher
    primary_metric: _Short
    direction: Literal["minimize", "maximize"] = "minimize"
    unit: _Short = "count"
    preconditions: Annotated[list[_Text], Field(max_length=64)]
    risks: Annotated[list[_Text], Field(max_length=64)]
    source_ids: Annotated[list[_Id], Field(max_length=1024)]
    status: Literal["draft", "active", "retired"] = "draft"

    @field_validator("source_ids")
    @classmethod
    def sorted_sources(cls, values: list[str]) -> list[str]:
        return sorted(set(values))


class Hotspot(_Record):
    path: str
    symbol: _Short | None = None
    weight: Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]
    revision: str

    _revision = field_validator("revision")(_sha)
    _path = field_validator("path")(_path)


class Candidate(_Record):
    candidate_id: _Id
    pattern_id: _Id
    pattern_version: Annotated[int, Field(ge=1)]
    repo_path: _Text
    repo_revision: str
    path: str
    line_start: Annotated[int, Field(ge=1)]
    source_sha256: str
    excerpt: Annotated[str, Field(min_length=1, max_length=4096)]
    owner: _Short | None = None
    score: Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]
    score_breakdown: dict[str, Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]]
    reasons: Annotated[list[_Text], Field(max_length=64)]
    source_ids: Annotated[list[_Id], Field(max_length=1024)]
    lane: Literal["workbench", "pipeline"]

    _revision = field_validator("repo_revision")(_sha)
    _path = field_validator("path")(_path)

    @field_validator("source_sha256")
    @classmethod
    def valid_hash(cls, value: str) -> str:
        if len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
            raise ValueError("source_sha256 must be a SHA-256 digest")
        return value


def _limit(value: int, name: str, maximum: int) -> int:
    if type(value) is not int or not 1 <= value <= maximum:
        raise ValueError(f"{name} must be an integer between 1 and {maximum}")
    return value


class _Git:
    """Bounded stdout, discarded stderr, per-command and operation deadlines."""

    def __init__(self, repo_path: Path, git_bin: str) -> None:
        if not isinstance(repo_path, Path) or not repo_path.is_dir():
            raise ValueError("repo_path must be an existing directory Path")
        if not isinstance(git_bin, str) or not git_bin or "\x00" in git_bin:
            raise ValueError("git_bin must name an executable")
        self.root = repo_path.resolve()
        self.git_bin = git_bin
        self.deadline = time.monotonic() + 120
        raw, cut = self.read(["rev-parse", "--show-toplevel"], 16384)
        if cut:
            raise ValueError("Repository root path exceeds the supported limit")
        self.root = Path(raw.decode("utf-8", errors="strict").strip()).resolve()
        self.repo_id = _digest(str(self.root), "repo_")

    def read(self, arguments: list[str], cap: int) -> tuple[bytes, bool]:
        remaining = self.deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("Git mining/scanning exceeded its 120 second budget")
        env = dict(os.environ)
        # Prevent caller-inherited Git location overrides and replacement objects.
        for key in (
            "GIT_DIR",
            "GIT_WORK_TREE",
            "GIT_INDEX_FILE",
            "GIT_OBJECT_DIRECTORY",
            "GIT_ALTERNATE_OBJECT_DIRECTORIES",
            "GIT_COMMON_DIR",
        ):
            env.pop(key, None)
        env.update(GIT_NO_REPLACE_OBJECTS="1", GIT_OPTIONAL_LOCKS="0", GIT_TERMINAL_PROMPT="0")
        argv = [
            self.git_bin,
            "--no-pager",
            "-c",
            "core.fsmonitor=false",
            "-c",
            "core.quotePath=false",
            "-c",
            "color.ui=false",
            "-C",
            str(self.root),
            *arguments,
        ]
        try:
            proc = subprocess.Popen(
                argv,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                stdin=subprocess.DEVNULL,
                env=env,
                shell=False,
            )
        except OSError as exc:
            raise ValueError("Unable to run the configured Git executable") from exc
        timed_out = threading.Event()

        def kill() -> None:
            timed_out.set()
            try:
                proc.kill()
            except OSError:
                pass

        timer = threading.Timer(min(30.0, remaining), kill)
        timer.daemon = True
        timer.start()
        try:
            assert proc.stdout is not None
            data = proc.stdout.read(cap + 1)
            truncated = len(data) > cap
            if truncated:
                proc.kill()
            code = proc.wait()
            if timed_out.is_set():
                raise TimeoutError("Git read exceeded its time budget")
            if code and not truncated:
                raise ValueError(f"Git {arguments[0]} failed; verify repository and revision")
            return data[:cap], truncated
        finally:
            timer.cancel()
            if proc.poll() is None:
                proc.kill()
                proc.wait()
            if proc.stdout:
                proc.stdout.close()

    def revision(self, value: str) -> str:
        if (
            type(value) is not str
            or not value
            or len(value) > 256
            or value.startswith("-")
            or any(c.isspace() or ord(c) < 32 for c in value)
            or "\x00" in value
        ):
            raise ValueError("Invalid revision expression")
        raw, cut = self.read(
            ["rev-parse", "--verify", "--end-of-options", value + "^{commit}"], 128
        )
        if cut:
            raise ValueError("Invalid resolved revision")
        return _sha(raw.decode("ascii").strip())


def mine_git_history(
    repo_path: Path,
    *,
    revision: str = "HEAD",
    after_revision: str | None = None,
    max_commits: int = 200,
    max_patch_bytes: int = 64000,
    git_bin: str = "git",
) -> list[ChangeRecord]:
    """Read bounded first-parent commits/diffs, oldest first, for cursor paging.

    ``after_revision`` is excluded, must be an ancestor, and is resolved before
    walking. Per-record truncation includes clipped metadata/path lists. Total
    returned material is additionally bounded by a 24 MiB operation budget.
    Commit IDs, rather than patch size or retrieval time, identify sources.
    Side-branch changes are represented by their merge's first-parent diff.
    A bounded page is not itself evidence truncation; callers track pagination
    separately from each commit's clipped-content flag.
    """
    _limit(max_commits, "max_commits", 2000)
    _limit(max_patch_bytes, "max_patch_bytes", 1048576)
    git = _Git(repo_path, git_bin)
    tip = git.revision(revision)
    history_range = tip
    if after_revision is not None:
        after = git.revision(after_revision)
        git.read(["merge-base", "--is-ancestor", after, tip], 1)
        history_range = f"{after}..{tip}"
    count_raw, count_cut = git.read(
        ["rev-list", "--first-parent", "--count", history_range, "--"], 32
    )
    if count_cut:
        raise ValueError("Unexpected oversized history count")
    total = int(count_raw.decode("ascii").strip())
    skip = max(0, total - max_commits)
    raw, cut = git.read(
        [
            "rev-list",
            "--first-parent",
            "--reverse",
            f"--skip={skip}",
            f"--max-count={max_commits}",
            history_range,
            "--",
        ],
        max_commits * 65,
    )
    if cut:
        raise ValueError("Unexpected oversized revision listing")
    records: list[ChangeRecord] = []
    remaining = _MAX_HISTORY_BYTES
    for revision_id in raw.decode("ascii").splitlines():
        _sha(revision_id)
        if remaining < max_patch_bytes + 16384 + 131072:
            break
        metadata, meta_cut = git.read(
            ["show", "-s", "--format=%P%x00%s%x00%B", revision_id, "--"], 16384
        )
        fields = metadata.decode("utf-8", errors="replace").split("\x00", 2)
        if len(fields) < 3:
            raise ValueError("Commit metadata exceeded its budget; history cursor must not advance")
        parents, subject, message = fields
        parent = parents.split()[0] if parents else None
        commit_pair = [parent, revision_id] if parent else ["--root", revision_id]
        path_data, paths_cut = git.read(
            [
                "diff-tree",
                "--no-commit-id",
                "--name-only",
                "--no-renames",
                "-z",
                "-r",
                *commit_pair,
                "--",
            ],
            131072,
        )
        complete = path_data.split(b"\x00")[:-1]
        paths = sorted({_path(item.decode("utf-8", errors="strict")) for item in complete})
        paths_cut = paths_cut or len(paths) > 1024
        paths = paths[:1024]
        patch, patch_cut = git.read(
            [
                "diff-tree",
                "--no-commit-id",
                "--patch",
                "--no-ext-diff",
                "--no-textconv",
                "--no-renames",
                "--unified=3",
                "-r",
                *commit_pair,
                "--",
            ],
            max_patch_bytes,
        )
        records.append(
            ChangeRecord(
                source_id=_digest([git.repo_id, revision_id], "source_"),
                repo_id=git.repo_id,
                revision=revision_id,
                parent_revision=parent,
                subject=subject[:1024],
                message=message.rstrip(),
                paths=paths,
                patch=patch.decode("utf-8", errors="ignore"),
                truncated=meta_cut or paths_cut or patch_cut or len(subject) > 1024,
            )
        )
        remaining -= len(metadata) + len(path_data) + len(patch)
    return records


def read_git(
    repo_path: Path,
    arguments: list[str],
    *,
    git_bin: str = "git",
    max_bytes: int = 5_000_000,
) -> bytes:
    """Read a bounded Git command result for trusted in-process callers.

    Only inspection commands are accepted. Arguments are individual argv
    elements, never a shell string. External diff/textconv and output-file
    switches are disallowed; overflow raises rather than returning partial
    evidence. Callers must still bind revision/path arguments to their contract.
    """
    _limit(max_bytes, "max_bytes", 64 * 1024 * 1024)
    verbs = {
        "rev-parse",
        "rev-list",
        "show",
        "cat-file",
        "ls-tree",
        "diff",
        "diff-tree",
        "merge-base",
        "status",
        "ls-files",
        "log",
        "for-each-ref",
    }
    if (
        type(arguments) is not list
        or not 1 <= len(arguments) <= 128
        or any(type(arg) is not str or len(arg) > 4096 or "\x00" in arg for arg in arguments)
        or arguments[0] not in verbs
    ):
        raise ValueError("Expected bounded argv for a supported Git inspection command")
    unsafe = ("--output", "--ext-diff", "--textconv", "--filters", "--path", "--no-index")
    if any(
        any(arg == flag or arg.startswith(flag + "=") for flag in unsafe) for arg in arguments[1:]
    ):
        raise ValueError("Git inspection cannot run filters or write output files")
    args = list(arguments)
    if args[0] in {"diff", "diff-tree", "show", "log"}:
        args[1:1] = ["--no-ext-diff", "--no-textconv"]
    raw, cut = _Git(repo_path, git_bin).read(args, max_bytes)
    if cut:
        raise ValueError("Git output exceeded max_bytes; partial evidence was discarded")
    return raw


def distill_patterns(changes: list[ChangeRecord]) -> list[Pattern]:
    """Turn explicit performance/fix clues into provenance-bearing draft leads.

    At most three substantive removed lines per change and 256 distinct drafts
    are retained. Repeated historical literals merge their source IDs. Mined
    heuristics stay in the workbench until their marker/preconditions are
    explicitly curated, even if their status is subsequently activated.
    """
    if (
        type(changes) is not list
        or len(changes) > 2000
        or any(not isinstance(c, ChangeRecord) for c in changes)
    ):
        raise ValueError("changes must contain at most 2000 ChangeRecord instances")
    perf = ("perf", "optimiz", "instruction", "latency", "redundant", "faster", "speed up")
    fixes = ("fix", "race", "deadlock", "overflow", "use-after-free", "leak", "bounds")
    drafts: dict[str, Pattern] = {}
    consumed = 0
    for original in sorted(changes, key=lambda item: item.source_id):
        change = ChangeRecord.model_validate(original.model_dump())
        consumed += len(change.patch.encode("utf-8"))
        if consumed > _MAX_HISTORY_BYTES:
            break
        clue = " ".join([change.subject, change.message, *change.review_notes]).casefold()
        kind = (
            "optimization"
            if any(word in clue for word in perf)
            else ("anti_pattern" if any(word in clue for word in fixes) else None)
        )
        if kind is None:
            continue
        suffixes = sorted({Path(path).suffix for path in change.paths} & _SOURCE_SUFFIXES)
        if not suffixes:
            continue
        literals = []
        for line in change.patch.splitlines():
            if not line.startswith("-") or line.startswith("---"):
                continue
            literal = line[1:].strip()
            if (
                8 <= len(literal) <= 160
                and not literal.startswith(("//", "/*", "*", "#"))
                and any(c.isalpha() for c in literal)
                and literal not in literals
            ):
                literals.append(literal)
            if len(literals) >= 3:
                break
        for literal in literals:
            matcher = Matcher(file_globs=["**/*" + suffix for suffix in suffixes], all_of=[literal])
            pattern_id = _digest([kind, matcher.model_dump()], "pattern_")
            prior = drafts.get(pattern_id)
            if prior:
                payload = prior.model_dump()
                payload["source_ids"] = sorted(set(prior.source_ids + [change.source_id]))[:1024]
                if change.truncated and not any(
                    p.startswith(_TRUNCATED) for p in prior.preconditions
                ):
                    payload["preconditions"].append(
                        _TRUNCATED + " historical patch or metadata was clipped."
                    )
                drafts[pattern_id] = Pattern.model_validate(payload)
                continue
            if len(drafts) >= 256:
                continue
            preconditions = [
                _HEURISTIC + " removed-line matching does not establish semantic equivalence.",
                "Inspect the full historical diff and verify applicability to this repository revision.",
                "Curate the literal matcher, metric, constraints and remedy before activation.",
            ]
            if change.truncated:
                preconditions.append(_TRUNCATED + " historical patch or metadata was clipped.")
            drafts[pattern_id] = Pattern(
                pattern_id=pattern_id,
                title="Historical removed-code lead: " + literal,
                kind=kind,
                problem="A change with performance or correctness wording removed this code literal.",
                diagnosis="This is a textual historical clue, not evidence that the same defect or cost exists here.",
                remedy="Review the originating patch and reviews; specify an applicable remedy before changing code.",
                matcher=matcher,
                primary_metric="instruction_count"
                if kind == "optimization"
                else "correctness_failure_count",
                preconditions=preconditions,
                risks=[
                    "The literal can occur in unrelated contexts.",
                    "The historical change may have regressed behavior or had no measured improvement.",
                ],
                source_ids=[change.source_id],
                status="draft",
            )
    return [drafts[key] for key in sorted(drafts)]


def _matches(path: str, glob: str) -> bool:
    parts, rules = tuple(path.split("/")), tuple(glob.split("/"))

    @lru_cache(maxsize=65536)
    def match(i: int, j: int) -> bool:
        if j == len(rules):
            return i == len(parts)
        if rules[j] == "**":
            return match(i, j + 1) or (i < len(parts) and match(i + 1, j))
        return i < len(parts) and fnmatch.fnmatchcase(parts[i], rules[j]) and match(i + 1, j + 1)

    return match(0, 0)


def _owners(values: dict[str, str] | None) -> list[tuple[str, str]]:
    if values is None:
        return []
    if type(values) is not dict or len(values) > 1024:
        raise ValueError("owners must be a mapping of at most 1024 glob rules")
    for glob, owner in values.items():
        _path(glob, glob=True)
        if not isinstance(owner, str) or len(owner) > 256:
            raise ValueError("Owner names must be bounded strings")
        _plain(owner)
    return sorted(
        values.items(),
        key=lambda item: (
            -sum(c not in "*?" for c in item[0]),
            sum(c in "*?" for c in item[0]),
            item[0],
        ),
    )


def scan_candidates(
    repo_path: Path,
    patterns: list[Pattern],
    *,
    revision: str = "HEAD",
    hotspots: list[Hotspot] | None = None,
    owners: dict[str, str] | None = None,
    top_k: int = 20,
    max_files: int = 5000,
    max_file_bytes: int = 1000000,
    git_bin: str = "git",
    telemetry: dict | None = None,
) -> list[Candidate]:
    """Discover literal matches in an immutable tree, retaining only the top K.

    Examines at most ``max_files`` tree entries and 64 MiB of eligible blobs.
    Oversize, symlink, non-UTF-8 and binary blobs are skipped. Enumeration or
    byte-budget truncation forces all results to the workbench. Scoring is
    explicit: literal match .4, historical sources up to .2, current hotspot
    weight up to .3, owner .1. No regular expressions or source execution occur.
    """
    _limit(top_k, "top_k", 1000)
    _limit(max_files, "max_files", 50000)
    _limit(max_file_bytes, "max_file_bytes", 10485760)
    if (
        type(patterns) is not list
        or len(patterns) > 256
        or any(not isinstance(p, Pattern) for p in patterns)
    ):
        raise ValueError("patterns must contain at most 256 Pattern instances")
    hotspot_list = [] if hotspots is None else hotspots
    if (
        type(hotspot_list) is not list
        or len(hotspot_list) > 10000
        or any(not isinstance(h, Hotspot) for h in hotspot_list)
    ):
        raise ValueError("hotspots must contain at most 10000 Hotspot instances")
    owner_rules = _owners(owners)
    git = _Git(repo_path, git_bin)
    commit = git.revision(revision)
    if any(h.revision != commit for h in hotspot_list):
        raise ValueError("Stale hotspot evidence: every hotspot must identify the scanned commit")
    unique: dict[tuple[str, int], Pattern] = {}
    for pattern in patterns:
        # Revalidate lists: frozen Pydantic models do not freeze nested containers.
        checked = Pattern.model_validate(pattern.model_dump())
        key = (checked.pattern_id, checked.version)
        if key in unique and unique[key] != checked:
            raise ValueError("Conflicting pattern content for the same ID/version")
        unique[key] = checked
    active = sorted(
        (p for p in unique.values() if p.status == "active"),
        key=lambda p: (p.pattern_id, p.version),
    )
    stats = {
        "repo_revision": commit,
        "active_patterns": len(active),
        "tree_entries_seen": 0,
        "eligible_files": 0,
        "text_files_checked": 0,
        "blob_bytes": 0,
        "skipped_large": 0,
        "skipped_binary_or_encoding": 0,
        "matches": 0,
        "retained": 0,
        "coverage_complete": False,
        "results_capped": False,
        "not_scanned_reason": None if active else "no_active_patterns",
    }
    if telemetry is not None:
        telemetry.update(stats)
    if not active:
        return []
    raw, clipped = git.read(
        ["ls-tree", "-r", "-z", "-l", commit, "--"], min(16 * 1024 * 1024, (max_files + 1) * 4250)
    )
    rows = raw.split(b"\x00")[:-1]
    incomplete = clipped or len(rows) > max_files
    consumed = 0
    retained: list[tuple[float, int, Candidate]] = []
    by_path: dict[str, list[Hotspot]] = {}
    for hotspot in hotspot_list:
        checked = Hotspot.model_validate(hotspot.model_dump())
        by_path.setdefault(checked.path, []).append(checked)
    for row in rows[:max_files]:
        stats["tree_entries_seen"] += 1
        info, raw_path = row.split(b"\t", 1)
        mode, object_type, object_id, size_raw = info.split()
        path = _path(raw_path.decode("utf-8", errors="strict"))
        if object_type != b"blob" or mode not in {b"100644", b"100755"}:
            continue
        applicable = [
            p for p in active if any(_matches(path, glob) for glob in p.matcher.file_globs)
        ]
        if not applicable:
            continue
        stats["eligible_files"] += 1
        size = int(size_raw)
        if size > max_file_bytes:
            stats["skipped_large"] += 1
            incomplete = True
            continue
        if consumed + size > _MAX_SCAN_BYTES:
            incomplete = True
            break
        data, cut = git.read(["cat-file", "blob", _sha(object_id.decode("ascii"))], max_file_bytes)
        consumed += len(data)
        if cut or len(data) != size:
            raise ValueError("Git blob did not match its bounded tree metadata")
        if b"\x00" in data:
            stats["skipped_binary_or_encoding"] += 1
            incomplete = True
            continue
        try:
            content = data.decode("utf-8")
        except UnicodeDecodeError:
            stats["skipped_binary_or_encoding"] += 1
            incomplete = True
            continue
        stats["text_files_checked"] += 1
        for pattern in applicable:
            matcher = pattern.matcher
            if (
                not all(literal in content for literal in matcher.all_of)
                or (matcher.any_of and not any(literal in content for literal in matcher.any_of))
                or any(literal in content for literal in matcher.none_of)
            ):
                continue
            stats["matches"] += 1
            first = min(content.index(literal) for literal in matcher.all_of)
            line = content.count("\n", 0, first) + 1
            start = content.rfind("\n", 0, first) + 1
            excerpt = content[start : start + 4096]
            owner = next((owner for glob, owner in owner_rules if _matches(path, glob)), None)
            weight = max(
                (
                    h.weight
                    for h in by_path.get(path, [])
                    if h.symbol is None or h.symbol in excerpt
                ),
                default=0.0,
            )
            breakdown = {
                "literal_match": 0.4,
                "history": min(0.2, 0.05 * len(pattern.source_ids)),
                "hotspot": 0.3 * weight,
                "ownership": 0.1 if owner else 0.0,
            }
            score = round(sum(breakdown.values()), 8)
            unresolved = any(
                value.startswith((_HEURISTIC, _TRUNCATED))
                for value in [*pattern.preconditions, *pattern.risks]
            )
            excerpt_complete = all(literal in excerpt for literal in matcher.all_of) and (
                not matcher.any_of or any(literal in excerpt for literal in matcher.any_of)
            )
            lane = (
                "pipeline"
                if (
                    pattern.kind == "optimization"
                    and owner
                    and weight > 0
                    and pattern.source_ids
                    and pattern.preconditions
                    and not unresolved
                    and excerpt_complete
                )
                else "workbench"
            )
            reasons = [
                "All required literal predicates matched the immutable Git blob.",
                f"Historical sources: {len(pattern.source_ids)}; current hotspot weight: {weight:.4f}.",
                "Literal matches require semantic review; lane is not execution approval.",
            ]
            if unresolved:
                reasons.append(
                    "Unresolved historical heuristic or truncated provenance requires workbench review."
                )
            if not excerpt_complete:
                reasons.append("Not all required literals fit in the bounded excerpt.")
            if owner:
                reasons.append("Owner selected by most-specific literal glob precedence.")
            source_hash = hashlib.sha256(data).hexdigest()
            candidate_id = _digest(
                [git.repo_id, commit, pattern.model_dump(), path, source_hash, line], "candidate_"
            )
            candidate = Candidate(
                candidate_id=candidate_id,
                pattern_id=pattern.pattern_id,
                pattern_version=pattern.version,
                repo_path=str(git.root),
                repo_revision=commit,
                path=path,
                line_start=line,
                source_sha256=source_hash,
                excerpt=excerpt,
                owner=owner,
                score=score,
                score_breakdown=breakdown,
                reasons=reasons,
                source_ids=list(pattern.source_ids),
                lane=lane,
            )
            entry = (score, -int(candidate_id.removeprefix("candidate_"), 16), candidate)
            if len(retained) < top_k:
                heapq.heappush(retained, entry)
            else:
                heapq.heappushpop(retained, entry)
    results = [entry[2] for entry in retained]
    stats.update(
        blob_bytes=consumed,
        retained=len(results),
        coverage_complete=not incomplete,
        results_capped=stats["matches"] > len(results),
    )
    if telemetry is not None:
        telemetry.update(stats)
    if incomplete:
        results = [
            Candidate.model_validate(
                {
                    **item.model_dump(),
                    "lane": "workbench",
                    "reasons": [
                        *item.reasons,
                        "Scan enumeration or byte budget was truncated; coverage is incomplete.",
                    ],
                }
            )
            for item in results
        ]
    return sorted(results, key=lambda item: (-item.score, item.candidate_id))


__all__ = [
    "ChangeRecord",
    "Matcher",
    "Pattern",
    "Hotspot",
    "Candidate",
    "mine_git_history",
    "distill_patterns",
    "scan_candidates",
    "read_git",
]
