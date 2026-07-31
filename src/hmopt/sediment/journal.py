"""Per-contributor journal store for Team Memory (Team Memory design §4).

The journal is the *personal work layer* of the three-tier memory:

    journal (personal, unreviewed)  ->  knowledge (team, curated)  ->  skills

Entries are distilled **at capture time** by the conversing LLM + user
("当场蒸馏、只存蒸馏物") — the system never stores raw transcripts. One entry
= one markdown file with YAML frontmatter, laid out per contributor:

    $HMOPT_MEMBER_MEMORY_ROOT/            # default /data/team-memory
      <contributor>/
        <project>/journal/<YYYY-MM>/J-<ULID>.md
        <project>/.last_sediment          # marker for `pending` in memory_status
        feedback.jsonl                    # memory_feedback append-only log
        inbox/                            # P2 plugin drop-box (reserved)

Contracts carried over from the sediment package (design §2):
  * non-blocking — readers collect errors, never raise to the caller;
  * redact-on-write — an entry containing a secret pattern is REJECTED with
    the reason (never silently dropped, never silently stored);
  * outcome gate — entries with outcome attempted/unknown are withheld from
    hub candidates so model optimism can't masquerade as validated fact;
  * physical delete is allowed here (memory_forget) because the journal is a
    personal layer, NOT team truth — hub records still only supersede.
"""

from __future__ import annotations

import json
import logging
import os
import re
import stat
import threading
import time
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:  # pragma: no cover
    raise ImportError("PyYAML required for sediment.journal") from None

from hmopt.skillhub.embeddings import tokenize as _hub_tokenize

from .extractors import Candidate, _now, _slugify, _today

logger = logging.getLogger(__name__)

DEFAULT_MEMORY_ROOT = "/data/team-memory"

ENTRY_TYPES = ("fact", "heuristic", "anti_pattern", "validation_pitfall", "bad_plan", "idea")
OUTCOMES = ("validated", "accepted", "attempted", "failed", "reverted", "unknown")
#: outcome gate (design §4.6): these never become hub candidates.
TENTATIVE_OUTCOMES = ("attempted", "unknown")
FEEDBACK_VERDICTS = ("helpful", "harmful", "stale", "inapplicable")
CONFIDENCES = ("high", "medium", "low")

# CRLF-tolerant, same as skillhub/records.py (the house frontmatter regex).
_FRONTMATTER_RE = re.compile(r"^---\r?\n(?P<fm>.*?)\r?\n---\r?\n?(?P<body>.*)$", re.DOTALL)

_ULID_ALPHABET = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"  # Crockford base32
_ENTRY_ID_RE = re.compile(r"^J-[0-9A-HJKMNP-TV-Z]{26}$")
_HUB_ID_RE = re.compile(r"^[A-Z][0-9]{3,}$")
_NAMESPACE_RE = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9._-]{0,126}[A-Za-z0-9])?$")
_TARGET_SLUG_RE = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
_CJK_RUN_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]+")

MAX_TITLE_CHARS = 200
MAX_BODY_CHARS = 4000
MAX_BODY_LINES = 10

_feedback_lock = threading.Lock()


# --------------------------------------------------------------------------- #
# ids / paths
# --------------------------------------------------------------------------- #
_ulid_lock = threading.Lock()
_ulid_last: list[int] = [-1, 0]  # [ms, rand] of the last id handed out


def new_ulid(now_ms: int | None = None) -> str:
    """26-char Crockford-base32 ULID (10 time chars + 16 random chars).

    Hand-rolled because the platform has no ulid dependency. Monotonic within
    the process (same-ms collisions increment the random part), so entry ids
    are strictly ordered by creation — the sediment marker relies on that.
    """
    if now_ms is None:
        ms = int(time.time() * 1000)
        with _ulid_lock:
            if ms <= _ulid_last[0]:
                ms = _ulid_last[0]
                rand = (_ulid_last[1] + 1) % (1 << 80)
            else:
                rand = int.from_bytes(os.urandom(10), "big")
            _ulid_last[0], _ulid_last[1] = ms, rand
    else:
        ms = int(now_ms)
        rand = int.from_bytes(os.urandom(10), "big")
    chars: list[str] = []
    for _ in range(10):
        chars.append(_ULID_ALPHABET[ms & 31])
        ms >>= 5
    head = "".join(reversed(chars))
    chars = []
    for _ in range(16):
        chars.append(_ULID_ALPHABET[rand & 31])
        rand >>= 5
    return head + "".join(reversed(chars))


def is_journal_id(entry_id: str) -> bool:
    return bool(_ENTRY_ID_RE.match(str(entry_id or "").strip()))


def resolve_memory_root(memory_root: str | Path | None = None) -> Path:
    env = os.getenv("HMOPT_MEMBER_MEMORY_ROOT", "").strip()
    return Path(memory_root or env or DEFAULT_MEMORY_ROOT).expanduser()


def safe_name(value: str | None, *, fallback: str) -> str:
    """Sanitize contributor/project for use as a path component (no traversal)."""
    cleaned = re.sub(r"[^a-zA-Z0-9._-]+", "-", (value or "").strip()).strip("-.")
    if not cleaned or set(cleaned) <= {"."}:
        return fallback
    return cleaned


#: Names that collide with fixed children of a contributor directory — a
#: project named like these would be invisible to recall/status ("inbox") or
#: shadow a data file ("feedback.jsonl"). Compared case-insensitively.
RESERVED_PROJECT_NAMES = frozenset({"inbox", "feedback.jsonl", "sediment_staging"})


def validate_namespace(value: str, *, field_name: str) -> str | None:
    """Validate a contributor/project id without silently merging namespaces.

    ``safe_name`` remains available for trusted/internal path construction, but
    user-supplied namespace ids are rejected unless already canonical. Silent
    normalization (especially of non-ASCII names) could otherwise map unrelated
    members to the same ``anonymous`` directory.
    """
    if not value:
        return f"{field_name} is required"
    if not _NAMESPACE_RE.fullmatch(value) or ".." in value:
        return (
            f"{field_name} must be 1-128 ASCII letters/digits plus ._- "
            "(must start/end with a letter or digit and cannot contain '..')"
        )
    return None


def validate_project(value: str) -> str | None:
    """Namespace rules plus the reserved-name check for project ids."""
    if error := validate_namespace(value, field_name="project"):
        return error
    if value.lower() in RESERVED_PROJECT_NAMES:
        return (
            f"project {value!r} is reserved for journal internals "
            f"({'/'.join(sorted(RESERVED_PROJECT_NAMES))}); choose another project id"
        )
    return None


def lexical_tokens(text: str) -> list[str]:
    """Tokenize code/English plus CJK runs and bigrams for P1 lexical recall.

    The shared Skill Hub tokenizer intentionally targets code identifiers and is
    ASCII-only. Team Memory is primarily used in Chinese conversations, so a
    small dependency-free CJK layer is required here. Bigrams let a query such
    as ``设备重连`` match a longer phrase such as ``设备重连失败`` without
    introducing a vector/search dependency.
    """
    tokens = list(_hub_tokenize(text))
    for run in _CJK_RUN_RE.findall(text):
        tokens.append(run)
        if len(run) > 1:
            tokens.extend(run[i : i + 2] for i in range(len(run) - 1))
    return tokens


def _ensure_private_dir(path: Path) -> None:
    """Create a journal directory and keep it owner-only (0700)."""
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    # chmod is deliberate even for an existing path: an earlier version used
    # the process umask and commonly left contributor data world-readable.
    path.chmod(0o700)


def _write_private_text(path: Path, text: str, *, exclusive: bool = False) -> None:
    """Write a private (0600) UTF-8 file, optionally with O_EXCL."""
    flags = os.O_WRONLY | os.O_CREAT | (os.O_EXCL if exclusive else os.O_TRUNC)
    fd = os.open(path, flags, 0o600)
    try:
        payload = text.encode("utf-8")
        with os.fdopen(fd, "wb", closefd=True) as f:
            fd = -1
            f.write(payload)
            f.flush()
            os.fsync(f.fileno())
    finally:
        if fd >= 0:
            os.close(fd)
    path.chmod(0o600)


def contributor_dir(root: str | Path, contributor: str) -> Path:
    return Path(root) / safe_name(contributor, fallback="anonymous")


def journal_dir(root: str | Path, contributor: str, project: str) -> Path:
    return contributor_dir(root, contributor) / safe_name(project, fallback="general") / "journal"


# --------------------------------------------------------------------------- #
# redact-on-write (design §4.7: reuse hub rules, builtin fallback)
# --------------------------------------------------------------------------- #
_FALLBACK_REDACT_VERSION = "builtin-2026.07"
# Vendored copy of hm-skill-hub/tools/redact.py PATTERNS — the hub's tools/ dir
# is not an importable package, so the live rules are loaded from the hub
# checkout when one is reachable and this copy only covers the degraded case.
_FALLBACK_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("aws-akid", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    ("ssh-priv", re.compile(r"-----BEGIN (?:RSA|OPENSSH|DSA|EC|PGP) PRIVATE KEY-----")),
    ("generic-hex-key", re.compile(r"\b[0-9a-fA-F]{40,}\b")),
    ("device-serial", re.compile(r"\b(?:serial|SN|imei|IMEI)\s*[:=]\s*[A-Za-z0-9]{8,}\b")),
    ("dev-serial-path", re.compile(r"/dev/(?:ttyUSB|ttyACM|serial/by-id/)[A-Za-z0-9_./-]+")),
    ("github-pat", re.compile(r"\bghp_[A-Za-z0-9]{30,}\b")),
    ("slack-token", re.compile(r"\bxox[abps]-[A-Za-z0-9-]{10,}\b")),
]
_FALLBACK_ALLOW_TAG = re.compile(r"\[REDACTED\]|\[FAKE\]|<!--\s*allow-secret\s*-->")


def load_redact_rules(
    hub_root: str | Path | None = None,
) -> tuple[list[tuple[str, re.Pattern[str]]], re.Pattern[str], str]:
    """Return (patterns, allow_tag, rules_version), preferring the hub's redact.py."""
    if hub_root:
        mod_path = Path(hub_root) / "tools" / "redact.py"
        if mod_path.is_file():
            try:
                import importlib.util

                spec = importlib.util.spec_from_file_location("_hmopt_hub_redact", mod_path)
                if spec is not None and spec.loader is not None:
                    mod = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(mod)
                    patterns = list(mod.PATTERNS)
                    allow = getattr(mod, "ALLOW_TAG", _FALLBACK_ALLOW_TAG)
                    return patterns, allow, f"hub:{mod_path}"
            except Exception:
                logger.warning("failed to load hub redact rules from %s", mod_path, exc_info=True)
    return list(_FALLBACK_PATTERNS), _FALLBACK_ALLOW_TAG, _FALLBACK_REDACT_VERSION


def redact_scan(
    text: str,
    *,
    hub_root: str | Path | None = None,
    allow_markers: bool = False,
) -> list[tuple[int, str, str]]:
    """Scan text for secret patterns; returns (lineno, pattern_name, snippet) hits.

    Journal-facing calls are strict by default: a caller cannot smuggle a real
    secret through with the hub curator's trusted ``<!-- allow-secret -->`` or
    ``[FAKE]`` marker. ``allow_markers=True`` exists only for trusted hub tooling.
    """
    patterns, allow, _ = load_redact_rules(hub_root)
    hits: list[tuple[int, str, str]] = []
    for lineno, line in enumerate(text.splitlines(), 1):
        if allow_markers and allow.search(line):
            continue
        for name, pat in patterns:
            if pat.search(line):
                hits.append((lineno, name, line.strip()[:160]))
    return hits


# --------------------------------------------------------------------------- #
# entry model + file I/O
# --------------------------------------------------------------------------- #
@dataclass
class JournalEntry:
    """One distilled experience (design §4.3 — a lightweight episode)."""

    id: str
    type: str
    title: str
    body: str = ""
    project: str = "general"
    contributor: str = "anonymous"
    target_slug: str = ""
    tags: list[str] = field(default_factory=list)
    outcome: str = "unknown"
    evidence: list[str] = field(default_factory=list)
    applies_when: list[str] = field(default_factory=list)
    invalidated_by: list[str] = field(default_factory=list)
    confidence: str = ""
    ts: str = ""
    path: str = ""

    def search_text(self) -> str:
        parts = [
            self.id,
            self.title,
            self.target_slug,
            " ".join(self.tags),
            " ".join(self.applies_when),
            self.body,
            " ".join(self.evidence),
        ]
        return "\n".join(p for p in parts if p)


def _stringify_dates(value: Any) -> Any:
    """yaml.safe_load coerces unquoted ISO dates to date/datetime; undo that so
    entries stay json-serializable (same hazard hub tools/parse_memory.py fixes)."""
    if isinstance(value, datetime | date):
        return value.isoformat()
    if isinstance(value, list):
        return [_stringify_dates(v) for v in value]
    if isinstance(value, dict):
        return {k: _stringify_dates(v) for k, v in value.items()}
    return value


def _as_str_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value.strip() else []
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    return [str(value)]


def render_entry(entry: JournalEntry) -> str:
    """Serialize to `---\\n<yaml>\\n---\\n\\n<body>\\n` with only-meaningful fields."""
    fm: dict[str, Any] = {
        "id": entry.id,
        "type": entry.type,
        "title": entry.title,
        "project": entry.project,
    }
    if entry.target_slug:
        fm["target_slug"] = entry.target_slug
    if entry.tags:
        fm["tags"] = entry.tags
    fm["outcome"] = entry.outcome
    if entry.evidence:
        fm["evidence"] = entry.evidence
    if entry.applies_when:
        fm["applies_when"] = entry.applies_when
    if entry.invalidated_by:
        fm["invalidated_by"] = entry.invalidated_by
    if entry.confidence:
        fm["confidence"] = entry.confidence
    fm["contributor"] = entry.contributor
    fm["ts"] = entry.ts
    fm_text = yaml.safe_dump(fm, sort_keys=False, allow_unicode=True).strip()
    return f"---\n{fm_text}\n---\n\n{entry.body.strip()}\n"


def parse_entry_file(path: Path) -> JournalEntry:
    """Parse one journal file. Raises ValueError on malformed content."""
    m = _FRONTMATTER_RE.match(path.read_text(encoding="utf-8"))
    if not m:
        raise ValueError(f"{path}: missing YAML frontmatter")
    fm = yaml.safe_load(m.group("fm"))
    if not isinstance(fm, dict):
        raise TypeError(f"{path}: frontmatter is not a mapping")
    fm = _stringify_dates(fm)
    entry_id = str(fm.get("id") or "").strip()
    if not is_journal_id(entry_id):
        raise ValueError(f"{path}: invalid journal id {entry_id!r}")
    if path.stem != entry_id:
        raise ValueError(f"{path}: filename/id mismatch ({path.stem!r} != {entry_id!r})")
    entry = JournalEntry(
        id=entry_id,
        type=str(fm.get("type") or "fact"),
        title=str(fm.get("title") or "").strip(),
        body=m.group("body").strip(),
        project=str(fm.get("project") or "general"),
        contributor=str(fm.get("contributor") or "anonymous"),
        target_slug=str(fm.get("target_slug") or "").strip(),
        tags=_as_str_list(fm.get("tags")),
        outcome=str(fm.get("outcome") or "unknown"),
        evidence=_as_str_list(fm.get("evidence")),
        applies_when=_as_str_list(fm.get("applies_when")),
        invalidated_by=_as_str_list(fm.get("invalidated_by")),
        confidence=str(fm.get("confidence") or ""),
        ts=str(fm.get("ts") or ""),
        path=str(path),
    )
    errors = validate_entry_input(
        type=entry.type,
        title=entry.title,
        body=entry.body,
        outcome=entry.outcome,
        confidence=entry.confidence,
        contributor=entry.contributor,
        project=entry.project,
        target_slug=entry.target_slug,
    )
    if errors:
        raise ValueError(f"{path}: invalid journal entry: {'; '.join(errors)}")
    return entry


def validate_entry_input(
    *,
    type: str,
    title: str,
    body: str,
    outcome: str,
    confidence: str = "",
    contributor: str = "anonymous",
    project: str = "general",
    target_slug: str = "",
) -> list[str]:
    """Return a list of rejection reasons ([] == acceptable)."""
    errs: list[str] = []
    if type not in ENTRY_TYPES:
        errs.append(f"type must be one of {'/'.join(ENTRY_TYPES)} (got {type!r})")
    clean_title = (title or "").strip()
    clean_body = (body or "").strip()
    if not clean_title:
        errs.append("title is required")
    elif "\n" in clean_title or "\r" in clean_title:
        errs.append("title must be a single line")
    elif len(clean_title) > MAX_TITLE_CHARS:
        errs.append(f"title must be <= {MAX_TITLE_CHARS} characters")
    if not clean_body:
        errs.append("body is required (a few lines: what happened + why it generalizes)")
    else:
        if len(clean_body) > MAX_BODY_CHARS:
            errs.append(f"body must be <= {MAX_BODY_CHARS} characters")
        if len(clean_body.splitlines()) > MAX_BODY_LINES:
            errs.append(
                f"body must be <= {MAX_BODY_LINES} lines (store a distilled item, not a transcript)"
            )
    if outcome not in OUTCOMES:
        errs.append(f"outcome must be one of {'/'.join(OUTCOMES)} (got {outcome!r})")
    if confidence and confidence not in CONFIDENCES:
        errs.append(f"confidence must be one of {'/'.join(CONFIDENCES)} (got {confidence!r})")
    if error := validate_namespace(contributor, field_name="contributor"):
        errs.append(error)
    if error := validate_project(project):
        errs.append(error)
    if target_slug and (len(target_slug) > 128 or not _TARGET_SLUG_RE.fullmatch(target_slug)):
        errs.append(
            "target_slug must be a canonical lowercase kebab slug "
            "(letters/digits separated by single hyphens)"
        )
    return errs


def write_entry(
    root: str | Path,
    *,
    contributor: str,
    project: str,
    type: str,
    title: str,
    body: str,
    tags: list[str] | None = None,
    target_slug: str = "",
    outcome: str = "unknown",
    evidence: list[str] | None = None,
    applies_when: list[str] | None = None,
    invalidated_by: list[str] | None = None,
    confidence: str = "",
    hub_root: str | Path | None = None,
) -> tuple[JournalEntry | None, list[str]]:
    """Validate + redact-check + persist one entry. Returns (entry, errors).

    A non-empty errors list means the entry was REJECTED and nothing was
    written — the caller must surface the reasons (never silently drop).
    """
    normalized_contributor = (contributor or "").strip() or "anonymous"
    normalized_project = (project or "").strip() or "general"
    normalized_target = (target_slug or "").strip()
    errs = validate_entry_input(
        type=type,
        title=title,
        body=body,
        outcome=outcome,
        confidence=confidence,
        contributor=normalized_contributor,
        project=normalized_project,
        target_slug=normalized_target,
    )
    if errs:
        return None, errs
    entry = JournalEntry(
        id=f"J-{new_ulid()}",
        type=type,
        title=title.strip(),
        body=body.strip(),
        project=normalized_project,
        contributor=normalized_contributor,
        target_slug=normalized_target,
        tags=_as_str_list(tags),
        outcome=outcome,
        evidence=_as_str_list(evidence),
        applies_when=_as_str_list(applies_when),
        invalidated_by=_as_str_list(invalidated_by),
        confidence=confidence,
        ts=_now(),
    )
    text = render_entry(entry)
    hits = redact_scan(text, hub_root=hub_root)
    if hits:
        return None, [
            f"redact: secret-pattern={name} at line {lineno}" for lineno, name, _snippet in hits
        ] + ["rewrite without the secret (or mask it as [REDACTED]) and log again"]
    month = entry.ts[:7] or datetime.now(timezone.utc).strftime("%Y-%m")
    cdir = contributor_dir(root, entry.contributor)
    project_dir = cdir / entry.project
    out_dir = project_dir / "journal" / month
    for private_dir in (cdir, project_dir, project_dir / "journal", out_dir):
        _ensure_private_dir(private_dir)
    out_path = out_dir / f"{entry.id}.md"
    _write_private_text(out_path, text, exclusive=True)
    entry.path = str(out_path)
    return entry, []


def iter_entries(
    root: str | Path,
    contributor: str,
    *,
    project: str | None = None,
) -> tuple[list[JournalEntry], list[str]]:
    """Load a contributor's entries (all projects unless one is named).

    Never raises: unreadable/malformed files are reported in the errors list.
    """
    namespace_error = validate_namespace((contributor or "").strip(), field_name="contributor")
    if namespace_error:
        return [], [namespace_error]
    cdir = contributor_dir(root, contributor)
    entries: list[JournalEntry] = []
    errors: list[str] = []
    try:
        cstat = cdir.stat()
    except FileNotFoundError:
        return entries, errors
    except OSError as exc:
        return entries, [f"{cdir}: {exc}"]
    if not stat.S_ISDIR(cstat.st_mode):
        return entries, [f"{cdir}: contributor path is not a directory"]
    if project is not None:
        clean_project = (project or "").strip()
        if error := validate_project(clean_project):
            return entries, [error]
        proj_dirs = [cdir / clean_project]
    else:
        try:
            proj_dirs = sorted(
                p
                for p in cdir.iterdir()
                if p.is_dir() and p.name.lower() not in RESERVED_PROJECT_NAMES
            )
        except OSError as exc:
            return entries, [f"{cdir}: {exc}"]
    for pdir in proj_dirs:
        jdir = pdir / "journal"
        if not jdir.is_dir():
            continue
        for f in sorted(jdir.rglob("J-*.md")):
            try:
                entry = parse_entry_file(f)
                if entry.contributor != contributor:
                    raise ValueError(
                        f"{f}: frontmatter contributor {entry.contributor!r} "
                        f"does not match directory {contributor!r}"
                    )
                if entry.project != pdir.name:
                    raise ValueError(
                        f"{f}: frontmatter project {entry.project!r} "
                        f"does not match directory {pdir.name!r}"
                    )
                entries.append(entry)
            except (OSError, TypeError, ValueError, yaml.YAMLError) as exc:
                errors.append(str(exc))
    return entries, errors


def find_entry(root: str | Path, contributor: str, entry_id: str) -> JournalEntry | None:
    entry_id = (entry_id or "").strip()
    if not is_journal_id(entry_id) or validate_namespace(
        (contributor or "").strip(), field_name="contributor"
    ):
        return None
    cdir = contributor_dir(root, contributor)
    if not cdir.is_dir():
        return None
    for f in sorted(cdir.rglob(f"{entry_id}.md")):
        try:
            entry = parse_entry_file(f)
            if entry.contributor == contributor:
                return entry
        except (OSError, TypeError, ValueError, yaml.YAMLError):
            continue
    return None


def forget_entry(root: str | Path, contributor: str, entry_id: str) -> bool:
    """Physically delete the contributor's OWN entry (design §4.7).

    The journal is a personal work layer, so a real delete is allowed here —
    unlike hub knowledge, which only ever supersedes/tombstones.
    """
    entry = find_entry(root, contributor, entry_id)
    if entry is None or not entry.path:
        return False
    Path(entry.path).unlink()
    return True


# --------------------------------------------------------------------------- #
# feedback loop (design §4.5)
# --------------------------------------------------------------------------- #
def append_feedback(
    root: str | Path,
    *,
    contributor: str,
    entry_id: str,
    verdict: str,
    note: str = "",
    hub_root: str | Path | None = None,
) -> tuple[Path | None, list[str]]:
    if verdict not in FEEDBACK_VERDICTS:
        return None, [f"verdict must be one of {'/'.join(FEEDBACK_VERDICTS)} (got {verdict!r})"]
    clean_id = (entry_id or "").strip()
    if not (is_journal_id(clean_id) or _HUB_ID_RE.fullmatch(clean_id)):
        return None, ["id must be a journal id (J-…) or curated hub id (for example F031)"]
    clean_contributor = (contributor or "").strip()
    if error := validate_namespace(clean_contributor, field_name="contributor"):
        return None, [error]
    if len(note) > 500:
        return None, ["note must be <= 500 characters"]
    if "\x00" in note:
        return None, ["note cannot contain NUL"]
    cdir = contributor_dir(root, contributor)
    rec = {
        "ts": _now(),
        "id": clean_id,
        "verdict": verdict,
        "contributor": clean_contributor,
    }
    if note.strip():
        rec["note"] = note.strip()
    line = json.dumps(rec, ensure_ascii=False) + "\n"
    hits = redact_scan(line, hub_root=hub_root)
    if hits:
        return None, [
            f"redact: secret-pattern={name} in feedback" for _lineno, name, _snippet in hits
        ] + ["rewrite the feedback without the secret (or mask it as [REDACTED])"]

    _ensure_private_dir(cdir)
    fpath = cdir / "feedback.jsonl"
    # O_APPEND plus one os.write keeps each JSONL record intact under concurrent
    # FastMCP calls within this process; the lock also covers non-POSIX filesystems.
    with _feedback_lock:
        fd = os.open(fpath, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        try:
            os.write(fd, line.encode("utf-8"))
            os.fsync(fd)
        finally:
            os.close(fd)
        fpath.chmod(0o600)
    return fpath, []


# --------------------------------------------------------------------------- #
# recall scoring (design §4.4: lexical — token overlap + tag/target + decay)
# --------------------------------------------------------------------------- #
def score_text(
    query_tokens: set[str],
    text: str,
    *,
    tags: list[str] | None = None,
    target_slug: str = "",
) -> tuple[float, list[str]]:
    """Lexical relevance of `text` to a tokenized query. Returns (score, matched).

    score = query-token overlap ratio + 0.15/tag hit + 0.2 target-slug hit.
    Deliberately simple and self-explaining: `matched` feeds the recall
    output's `matched:` annotation (design §2.4 — recall must explain itself).
    """
    if not query_tokens:
        return 0.0, []
    doc_tokens = set(lexical_tokens(text))
    overlap = query_tokens & doc_tokens
    score = len(overlap) / len(query_tokens)
    matched = sorted(overlap)
    tag_tokens = set(lexical_tokens(" ".join(tags or [])))
    tag_hits = query_tokens & tag_tokens
    score += 0.15 * len(tag_hits)
    if target_slug:
        slug_tokens = set(lexical_tokens(target_slug))
        if slug_tokens and slug_tokens & query_tokens:
            score += 0.2
            matched = sorted(set(matched) | (slug_tokens & query_tokens))
    return score, matched[:6]


def time_decay(ts: str, *, now: datetime | None = None, half_life_days: float = 30.0) -> float:
    """Recency weight for journal entries (curated hub records are not decayed).

    Floors at 0.3 so an old-but-relevant personal note still surfaces.
    """
    try:
        normalized = ts[:-1] + "+00:00" if ts.endswith("Z") else ts
        then = datetime.fromisoformat(normalized)
        if then.tzinfo is None:
            then = then.replace(tzinfo=timezone.utc)
    except (ValueError, TypeError):
        return 1.0
    now = now or datetime.now(timezone.utc)
    age_days = max(0.0, (now - then).total_seconds() / 86400.0)
    return max(0.3, 0.5 ** (age_days / half_life_days))


def recall_entries(
    query: str,
    entries: list[JournalEntry],
    *,
    k: int = 5,
    now: datetime | None = None,
) -> list[tuple[JournalEntry, float, list[str]]]:
    """Rank journal entries against a query; drops zero-signal entries."""
    qtok = set(lexical_tokens(query))
    scored: list[tuple[JournalEntry, float, list[str]]] = []
    for e in entries:
        base, matched = score_text(qtok, e.search_text(), tags=e.tags, target_slug=e.target_slug)
        if base <= 0:
            continue
        scored.append((e, base * time_decay(e.ts, now=now), matched))
    scored.sort(key=lambda t: (t[1], t[0].ts), reverse=True)  # score desc, newest on ties
    return scored[:k]


# --------------------------------------------------------------------------- #
# sediment marker (feeds `pending` in memory_status)
# --------------------------------------------------------------------------- #
def _marker_path(root: str | Path, contributor: str, project: str) -> Path:
    return (
        contributor_dir(root, contributor)
        / safe_name(project, fallback="general")
        / ".last_sediment"
    )


def write_sediment_marker(
    root: str | Path,
    contributor: str,
    project: str,
    *,
    last_id: str = "",
    covered_ids: list[str] | None = None,
) -> None:
    """Record exactly which entries a sediment run emitted for this project.

    Older markers only stored a high-water ULID. Version 2 stores the explicit
    emitted-id set so outcome-gated, redacted, or schema-invalid entries remain
    visibly pending even when a newer entry was emitted.
    """
    p = _marker_path(root, contributor, project)
    _ensure_private_dir(p.parent)
    if covered_ids is None:
        _write_private_text(p, f"{_now()} {last_id}".strip() + "\n")
        return
    payload = {
        "version": 2,
        "sedimented_at": _now(),
        "last_id": last_id,
        "covered_ids": sorted(set(covered_ids)),
    }
    _write_private_text(
        p,
        json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n",
    )


def read_sediment_marker(root: str | Path, contributor: str, project: str) -> str:
    """Return the legacy high-water id (kept for compatibility/tests)."""
    _covered, last_id = _read_sediment_coverage(root, contributor, project)
    return last_id


def _read_sediment_coverage(
    root: str | Path, contributor: str, project: str
) -> tuple[set[str] | None, str]:
    """Return (explicit covered ids, legacy last id).

    ``covered is None`` denotes a legacy high-water marker; an empty set is a
    valid v2 marker meaning the run emitted no entries.
    """
    p = _marker_path(root, contributor, project)
    try:
        text = p.read_text(encoding="utf-8").strip()
    except OSError:
        return None, ""
    try:
        doc = json.loads(text)
    except json.JSONDecodeError:
        parts = text.split()
        return None, parts[1] if len(parts) > 1 else ""
    if not isinstance(doc, dict) or doc.get("version") != 2:
        return None, ""
    covered = {
        str(entry_id) for entry_id in doc.get("covered_ids") or [] if is_journal_id(str(entry_id))
    }
    return covered, str(doc.get("last_id") or "")


def _entry_is_pending(
    entry: JournalEntry,
    marker_last_id: str,
    covered_ids: set[str] | None = None,
) -> bool:
    """Pending == not successfully emitted by a sediment run."""
    if covered_ids is not None:
        return entry.id not in covered_ids
    # Legacy high-water markers predate exact coverage. Tentative entries were
    # never emitted and must remain pending even if their ULID is below it.
    if entry.outcome in TENTATIVE_OUTCOMES:
        return True
    return not marker_last_id or entry.id > marker_last_id


def journal_status(
    root: str | Path,
    contributor: str,
    *,
    project: str | None = None,
    hub_root: str | Path | None = None,
) -> dict[str, Any]:
    """Summary for memory_status: counts / latest / pending / feedback / redact rules.

    Disk problems are reported in `errors` (design §7: never silently lose)."""
    entries, errors = iter_entries(root, contributor, project=project)
    per_project: dict[str, int] = {}
    markers: dict[str, tuple[set[str] | None, str]] = {}
    pending = 0
    for e in entries:
        per_project[e.project] = per_project.get(e.project, 0) + 1
        if e.project not in markers:
            markers[e.project] = _read_sediment_coverage(root, contributor, e.project)
        covered_ids, last_id = markers[e.project]
        if _entry_is_pending(e, last_id, covered_ids):
            pending += 1
    latest = max(entries, key=lambda e: e.ts) if entries else None
    feedback_count = 0
    fpath = contributor_dir(root, contributor) / "feedback.jsonl"
    if fpath.is_file():
        try:
            lines = fpath.read_text(encoding="utf-8").splitlines()
            feedback_count = sum(1 for ln in lines if ln.strip())
        except OSError as exc:
            errors.append(f"{fpath}: {exc}")
    _, _, redact_version = load_redact_rules(hub_root)
    resolved_root = resolve_memory_root(root)
    writable_target = resolved_root
    while not writable_target.exists() and writable_target.parent != writable_target:
        writable_target = writable_target.parent
    try:
        writable = writable_target.is_dir() and os.access(writable_target, os.W_OK | os.X_OK)
    except OSError as exc:
        writable = False
        errors.append(f"{writable_target}: {exc}")
    if resolved_root.exists() and not resolved_root.is_dir():
        writable = False
        errors.append(f"{resolved_root}: memory_root is not a directory")
    elif not writable:
        errors.append(f"{resolved_root}: memory_root is not writable")
    return {
        "memory_root": str(resolved_root),
        "memory_root_writable": writable,
        "contributor": contributor,
        "entries": len(entries),
        "per_project": per_project,
        "latest": {"id": latest.id, "title": latest.title, "ts": latest.ts} if latest else None,
        "pending_sediment": pending,
        "feedback": feedback_count,
        "redact_rules": redact_version,
        "errors": errors,
    }


# --------------------------------------------------------------------------- #
# journal -> hub-candidate deterministic mapping (design §4.6)
# --------------------------------------------------------------------------- #
def _date_of(ts: str) -> str:
    return ts[:10] if len(ts) >= 10 else _today()


def _first_line(text: str) -> str:
    for line in text.splitlines():
        if line.strip():
            return line.strip()
    return ""


def _journal_ref(entry: JournalEntry) -> str:
    return f"journal:{entry.contributor}/{entry.project}/{entry.id}"


def _evidence_refs(entry: JournalEntry) -> list[dict[str, str]]:
    refs = [{"kind": "doc", "ref": _journal_ref(entry)}]
    refs.extend({"kind": "doc", "ref": s[:300]} for s in entry.evidence)
    return refs


def _mechanism_of(entry: JournalEntry) -> str:
    return _slugify(entry.tags[0] if entry.tags else entry.title)[:80]


def candidate_eligible_entries(entries: list[JournalEntry]) -> list[JournalEntry]:
    """Entries expected to map one-to-one to candidates before schema/redact gates."""
    idea_outcomes = {"validated", "accepted", "failed", "reverted"}
    return [
        entry
        for entry in entries
        if entry.outcome in OUTCOMES
        and entry.outcome not in TENTATIVE_OUTCOMES
        and entry.type in ENTRY_TYPES
        and (entry.type != "idea" or entry.outcome in idea_outcomes)
    ]


def journal_to_candidates(
    entries: list[JournalEntry],
    *,
    contributor: str,
) -> tuple[list[Candidate], list[JournalEntry], list[str]]:
    """Map entries onto the four hub schemas — deterministically, no LLM.

    Returns (candidates, gated, errors). `gated` holds entries withheld by the
    outcome gate (attempted/unknown — design §4.6 item 2). Contributor is
    written as the *bare* stable name so the hub's (target_slug, contributor)
    promotion distinctness counts members, not runs.
    """
    candidates: list[Candidate] = []
    gated: list[JournalEntry] = []
    errors: list[str] = []
    normalized_contributor = safe_name(contributor, fallback="anonymous")
    seq = {"F": 0, "H": 0, "A": 0, "V": 0, "B": 0, "L": 0}
    lesson_prefix = {"heuristic": "H", "anti_pattern": "A", "validation_pitfall": "V"}

    for e in entries:
        if e.outcome not in OUTCOMES:
            errors.append(f"{e.id}: unknown outcome {e.outcome!r}; skipped")
            continue
        if e.outcome in TENTATIVE_OUTCOMES:
            gated.append(e)
            continue
        if e.type not in ENTRY_TYPES:
            errors.append(f"{e.id}: unknown type {e.type!r}; skipped")
            continue
        if e.type == "idea" and e.outcome not in {"validated", "accepted", "failed", "reverted"}:
            errors.append(f"{e.id}: outcome {e.outcome!r} cannot map to an idea verdict; skipped")
            continue
        ts = e.ts or _now()
        target_slug = e.target_slug or _slugify(e.project)

        if e.type == "fact":
            seq["F"] += 1
            body = e.body
            if e.evidence:
                body += "\n\nEvidence:\n" + "\n".join(f"- {s}" for s in e.evidence)
            # Hub materialization has a target path but no global fact path.
            # target_slug remains optional at capture time; a project-level fact
            # deterministically falls back to an architectural project target.
            scope: dict[str, Any] = {
                "level": "function" if e.target_slug else "architectural",
                "target_slug": target_slug,
            }
            record: dict[str, Any] = {
                "id": f"F{900 + seq['F']}",
                "type": "fact",
                "title": e.title[:200],
                "body": body,
                "scope": scope,
                "source": _evidence_refs(e),
                "maturity": "L1",
                "status": "active",
                "contributor": normalized_contributor,
                "created_at": ts,
            }
            if e.applies_when:
                record["applies_when"] = "; ".join(e.applies_when)[:300]
            if e.invalidated_by:
                record["invalidation"] = "; ".join(e.invalidated_by)[:300]
            candidates.append({"schema": "memory_item", "record": record})

        elif e.type in lesson_prefix:
            prefix = lesson_prefix[e.type]
            seq[prefix] += 1
            record = {
                "id": f"{prefix}{950 + seq[prefix]}",
                "lesson": e.title[:300],
                "kind": e.type,
                "applies_when": ("; ".join(e.applies_when) or e.target_slug or "general")[:300],
                "do_or_dont": (" ".join(e.body.split()) or e.title)[:300],
                "tags": (e.tags or ["journal"])[:6],
                "evidence": _evidence_refs(e),
                "confidence": "tentative",
                "added_on": _date_of(ts),
                "added_by": normalized_contributor,
                "status": "active",
            }
            candidates.append({"schema": "global_lesson", "record": record})

        elif e.type == "bad_plan":
            seq["B"] += 1
            record = {
                "id": f"B{900 + seq['B']}",
                "title": e.title[:200],
                "mechanism": _mechanism_of(e),
                "target_pattern": e.target_slug or "*",
                "scope": "function",
                "applies_to": {"subsystems": ["*"]},
                "reason": (e.body or e.title)[:1500],
                "evidence": _evidence_refs(e),
                "rejected_on": _date_of(ts),
                "rejected_by": normalized_contributor,
                "status": "active",
            }
            candidates.append({"schema": "bad_plan", "record": record})

        elif e.type == "idea":
            status_map = {
                "validated": "approved",
                "accepted": "approved",
                "failed": "rejected",
                "reverted": "reverted",
            }
            seq["L"] += 1
            rationale_parts = [e.body or e.title, f"Source: {_journal_ref(e)}"]
            if e.evidence:
                rationale_parts.append("Evidence: " + "; ".join(e.evidence))
            record = {
                "id": f"L{900 + seq['L']}",
                "mechanism": _mechanism_of(e),
                "target_slug": target_slug,
                "scope": e.target_slug or f"project:{e.project}",
                "status": status_map[e.outcome],
                "verdicted_by": normalized_contributor,
                "verdicted_at": ts,
                "rationale": "\n".join(rationale_parts)[:1500],
            }
            if record["status"] == "approved":
                record["approved_on"] = ts
            elif record["status"] == "rejected":
                record["rejected_on"] = ts
            candidates.append({"schema": "idea", "record": record})

    return candidates, gated, errors
