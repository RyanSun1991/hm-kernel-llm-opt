"""Read a member's `.opencode/memory/` markdown stores into sediment candidates.

This is the source adapter the real workflow needs: members do not run
`hmopt run`; they work through the `.opencode/` multi-agent harness, and their
durable findings accumulate as markdown under `.opencode/memory/` (idea ledgers,
global lessons, per-target / per-subsystem notes). This module parses those
markdown formats into the same `{"schema", "record"}` candidates the rest of the
sediment pipeline (validate -> staging -> hub dedup/curate/promote) already
consumes — so the existing downstream is reused unchanged; only the front-end
source is pointed at `.opencode/memory/` instead of `hmopt run` artifacts.

Format mapping (design §8, member-side stores):
    idea_ledger/<target>.md  rows (### L00x ...)  -> hub idea
    global_lessons.md        entries              -> hub global_lesson
    targets/<slug>.md        "Known Bad Plans"     -> hub bad_plan
    targets/<slug>.md        "Stable Structural Facts" / "Good Optimization
                             Directions"           -> hub memory_item fact
    subsystems/<name>.md     facts                 -> hub memory_item fact

Parsing is resilient and non-blocking (design §8): malformed rows are collected,
not raised. HTML-comment example blocks (`<!-- ... -->`) in the templates are
stripped first so template scaffolding never sediments as real data.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .extractors import (
    BenchResult,
    Candidate,
    LedgerChange,
    ReviewVerdict,
    _slugify,
    bench_to_fact,
    ledger_to_idea,
    review_to_anti_pattern,
    review_to_bad_plan,
)

_COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)
_BULLET_RE = re.compile(r"^\s*[-*]\s+\*\*(?P<key>[^*]+)\*\*\s*:\s*(?P<val>.+?)\s*$", re.MULTILINE)
_PLAIN_BULLET_RE = re.compile(r"^\s*[-*]\s+(?P<val>(?!\*\*).+?)\s*$", re.MULTILINE)
_LEDGER_ROW_RE = re.compile(r"^###\s+(?P<id>L\d+)\b[ \t]*(?P<desc>.*)$", re.MULTILINE)
_TARGET_SLUG_RE = re.compile(r"Target\s+slug\s*:\s*`?(?P<slug>[a-zA-Z0-9_./:-]+)`?", re.IGNORECASE)

_LEDGER_STATUS = ("approved", "landed", "reverted", "rejected", "deferred")
_COMPARE_LEVELS = {"total", "process", "thread", "lib", "function"}


@dataclass
class OpenCodeMemory:
    """Candidates parsed from a member's `.opencode/` tree, plus the non-fatal
    problems hit while parsing (design §8: never gate). `free_text` carries the
    research design/plan documents for the optional LLM salience pass."""

    candidates: list[Candidate] = field(default_factory=list)
    parse_errors: list[str] = field(default_factory=list)
    free_text: list[str] = field(default_factory=list)
    # files examined per source — lets the CLI explain a 0-candidate result
    # ("scanned 3 ledgers but no ### L00x rows matched") instead of leaving the
    # member guessing whether the tool even looked at their data.
    scanned: dict[str, int] = field(default_factory=dict)


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _today() -> str:
    return datetime.now(timezone.utc).date().isoformat()


def _strip_comments(text: str) -> str:
    return _COMMENT_RE.sub("", text)


def _bullets(block: str) -> dict[str, str]:
    """`- **key**: value` lines -> {key.lower(): value}."""
    return {m.group("key").strip().lower(): m.group("val").strip()
            for m in _BULLET_RE.finditer(block)}


def _plain_bullets(block: str) -> list[str]:
    """`- text` lines that are NOT `- **key**: ...` field bullets."""
    return [m.group("val").strip() for m in _PLAIN_BULLET_RE.finditer(block)]


def _split_h2(text: str) -> list[tuple[str, str]]:
    """Split markdown into (h2_heading, body) sections."""
    parts = re.split(r"^##[ \t]+(.+)$", text, flags=re.MULTILINE)
    # parts[0] is the preamble before the first h2; then [heading, body]*.
    out: list[tuple[str, str]] = []
    for i in range(1, len(parts), 2):
        out.append((parts[i].strip(), parts[i + 1] if i + 1 < len(parts) else ""))
    return out


def _float_or_none(s: str | None) -> float | None:
    if not s:
        return None
    m = re.search(r"-?\d+(?:\.\d+)?", s)
    return float(m.group(0)) if m else None


# --- idea ledger ------------------------------------------------------------

def _ledger_target_slug(text: str, path: Path) -> str:
    m = _TARGET_SLUG_RE.search(text)
    if m:
        return _slugify(m.group("slug"))
    return _slugify(path.stem)


def parse_idea_ledger(path: Path, *, contributor: str, errors: list[str]) -> list[Candidate]:
    """Parse `idea_ledger/<target>.md` rows (### L00x ...) into idea candidates."""
    raw = _strip_comments(path.read_text(encoding="utf-8"))
    target_slug = _ledger_target_slug(raw, path)
    out: list[Candidate] = []
    seq = 0
    for heading, body in _split_h2(raw):
        section_status = next((s for s in _LEDGER_STATUS if s in heading.lower()), None)
        rows = list(_LEDGER_ROW_RE.finditer(body))
        for i, m in enumerate(rows):
            start = m.end()
            end = rows[i + 1].start() if i + 1 < len(rows) else len(body)
            block = body[start:end]
            fields = _bullets(block)
            status = (fields.get("status") or section_status or "").lower()
            if status not in _LEDGER_STATUS:
                errors.append(f"{path.name}:{m.group('id')}: unknown/missing status "
                              f"{status!r} — skipped")
                continue
            mechanism = (m.group("desc").strip() or fields.get("mechanism")
                         or fields.get("scope") or m.group("id"))
            compare = (fields.get("compare_level") or "").lower() or None
            if compare and compare not in _COMPARE_LEVELS:
                compare = None
            seq += 1
            change = LedgerChange(
                mechanism=mechanism,
                target_slug=fields.get("target_slug") or target_slug,
                scope=fields.get("scope") or "function",
                status=status,  # type: ignore[arg-type]
                rationale=fields.get("rationale") or mechanism,
                verdicted_by=fields.get("verdicted_by") or contributor,
                verdicted_at=fields.get("verdicted_at") or "",
                delta_pct=_float_or_none(fields.get("delta_pct")),
                compare_level=compare,
                validation_path=fields.get("validation_path"),
                reopen_trigger=fields.get("reopen_trigger"),
            )
            cand = ledger_to_idea(change, run_id=f"opencode:{path.name}", seq=seq)
            # preserve the member's stable ledger id as provenance, but renumber to
            # the provisional 9xx band so hub id allocation stays with the Curator.
            cand["record"]["related_ids"] = [m.group("id")]
            out.append(cand)
    return out


# --- global lessons ---------------------------------------------------------

_GL_KINDS = {"heuristic", "anti_pattern", "validation_pitfall"}


def parse_global_lessons(path: Path, *, contributor: str, errors: list[str]) -> list[Candidate]:
    """Parse `global_lessons.md` entries into global_lesson candidates.

    Recommended per-entry shape (mirrors the idea-ledger bullet style)::

        ### <one-line lesson>
        - **kind**: heuristic | anti_pattern | validation_pitfall   (optional)
        - **applies_when**: <trigger>
        - **do_or_dont**: do: ... / don't: ...
        - **tags**: a, b

    Fallback: a `### heading` with a paragraph body becomes a heuristic whose
    `do_or_dont` is the paragraph.
    """
    raw = _strip_comments(path.read_text(encoding="utf-8"))
    out: list[Candidate] = []
    # The schema ties id prefix to kind (H=heuristic, A=anti_pattern,
    # V=validation_pitfall), so number each family independently.
    seqs = {"heuristic": 0, "anti_pattern": 0, "validation_pitfall": 0}
    prefix = {"heuristic": "H", "anti_pattern": "A", "validation_pitfall": "V"}
    for m in re.finditer(r"^###[ \t]+(?P<title>.+)$", raw, flags=re.MULTILINE):
        start = m.end()
        nxt = raw.find("\n### ", start)
        block = raw[start: nxt if nxt != -1 else len(raw)]
        lesson = m.group("title").strip()
        if not lesson:
            continue
        fields = _bullets(block)
        kind = (fields.get("kind") or "heuristic").lower()
        if kind not in _GL_KINDS:
            kind = "heuristic"
        body_text = "\n".join(_plain_bullets(block)).strip()
        do_or_dont = fields.get("do_or_dont") or body_text or f"consider: {lesson}"
        tags = [t.strip() for t in re.split(r"[,;]", fields.get("tags", "")) if t.strip()]
        seqs[kind] += 1
        out.append({"schema": "global_lesson", "record": {
            "id": f"{prefix[kind]}{950 + seqs[kind]}",
            "lesson": lesson[:300],
            "kind": kind,
            "applies_when": (fields.get("applies_when") or "unspecified")[:300],
            "do_or_dont": do_or_dont[:300],
            "tags": tags or ["opencode-memory"],
            "evidence": [{"kind": "run_id", "ref": f"opencode:{path.name}"}],
            "confidence": "tentative",
            "added_on": _today(),
            "added_by": f"{contributor} (.opencode/memory)",
            "status": "active",
        }})
    return out


# --- target / subsystem memory ----------------------------------------------

_BAD_PLAN_SECTIONS = ("known bad plans", "bad plans", "rejected")
_FACT_SECTIONS = ("stable structural facts", "good optimization directions",
                  "hot paths", "structural facts")


def parse_target_memory(path: Path, *, contributor: str, errors: list[str],
                        subsystem: str | None = None) -> list[Candidate]:
    """Parse `targets/<slug>.md` (or `subsystems/<name>.md`) sections into
    bad_plan + memory_item-fact candidates."""
    raw = _strip_comments(path.read_text(encoding="utf-8"))
    slug = _slugify(path.stem)
    out: list[Candidate] = []
    bad_seq = fact_seq = 0
    for heading, body in _split_h2(raw):
        h = heading.lower()
        bullets = _plain_bullets(body)
        if any(k in h for k in _BAD_PLAN_SECTIONS):
            for text in bullets:
                bad_seq += 1
                out.append(_bad_plan(text, slug, subsystem, contributor, bad_seq, path))
        elif any(k in h for k in _FACT_SECTIONS):
            level = "subsystem" if subsystem else "function"
            for text in bullets:
                fact_seq += 1
                out.append(_fact(text, slug, subsystem, level, contributor, fact_seq, path))
    return out


def _bad_plan(text: str, slug: str, subsystem: str | None, contributor: str,
              seq: int, path: Path) -> Candidate:
    mech = _slugify(" ".join(text.split()[:6]))
    return {"schema": "bad_plan", "record": {
        "id": f"B{900 + seq}",
        "title": text[:120],
        "mechanism": mech,
        "target_pattern": slug,
        "scope": "subsystem" if subsystem else "function",
        "applies_to": {"subsystems": [subsystem] if subsystem else ["*"]},
        "reason": text[:1500],
        "evidence": [{"kind": "review", "ref": f"opencode:{path.name}"}],
        "rejected_on": _today(),
        "rejected_by": f"{contributor} (.opencode/memory)",
        "status": "active",
    }}


def _fact(text: str, slug: str, subsystem: str | None, level: str, contributor: str,
          seq: int, path: Path) -> Candidate:
    scope: dict[str, Any] = {"level": level}
    if subsystem:
        scope["subsystem"] = subsystem
    else:
        scope["target_slug"] = slug
    return {"schema": "memory_item", "record": {
        "id": f"F{900 + seq}",
        "type": "fact",
        "title": text[:120],
        "body": text,
        "scope": scope,
        "source": [{"kind": "run_id", "ref": f"opencode:{path.name}"}],
        "maturity": "L1",
        "status": "active",
        "contributor": contributor,
        "created_at": _now(),
    }}


# --- run artifacts: reviews / bench / state bad-plans -----------------------
#
# The harness write contract (.opencode/CLAUDE.md):
#   plan-reviewer  -> reviews/<artifact>_plan_review.md   (## Decision approve|reject)
#   code-reviewer  -> reviews/<artifact>_code_review.md
#   tester         -> bench/<artifact>_validation.md      (verdict + aggregate delta)
#   manager        -> state/*bad_plans*.md                (globally rejected patterns)
# memory/ is the *curated* layer the agents are told to distill into; these raw
# Tier-0 artifacts are parsed as a safety net so a verdict still sediments even
# when an agent forgot the memory-accumulation step. Duplicates across sources
# (e.g. a landed ledger row AND its bench report) are reconciled by hub dedup
# (merge => confirmations++, which is exactly the corroboration we want).

_DECISION_SECTION_RE = re.compile(
    r"^##[ \t]+Decision[ \t]*\n+(?P<body>.*?)(?=^##[ \t]|\Z)", re.MULTILINE | re.DOTALL)
_VERDICT_RE = re.compile(r"\bverdict\b[^a-zA-Z]{0,10}(pass|fail|inconclusive|skipped)", re.I)
_DELTA_RE = re.compile(r"delta[_\s-]*pct\b[^-\d]{0,20}(-?\d+(?:\.\d+)?)", re.IGNORECASE)
_COMPARE_RE = re.compile(r"compare[_\s-]*level\b[^a-z]{0,10}(total|process|thread|lib|function)", re.I)


def _artifact_slug(path: Path) -> str:
    """`<artifact>_plan_review.md` / `<artifact>_validation.md` -> artifact slug."""
    stem = path.stem
    for suffix in ("_plan_review", "_code_review", "_validation", "_review"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    return _slugify(stem)


def _section(text: str, heading: str) -> str:
    m = re.search(rf"^##[ \t]+{re.escape(heading)}[ \t]*\n+(.*?)(?=^##[ \t]|\Z)",
                  text, re.MULTILINE | re.DOTALL)
    return m.group(1).strip() if m else ""


def parse_review(path: Path, *, contributor: str, errors: list[str]) -> list[Candidate]:
    """A reviews/*_plan_review.md / *_code_review.md REJECT verdict ->
    anti_pattern + bad_plan (the same mapping the run-dir extractors use).
    Approvals yield nothing — approval itself is not knowledge; the landed
    outcome is, and that arrives via the ledger / bench report."""
    raw = _strip_comments(path.read_text(encoding="utf-8"))
    m = _DECISION_SECTION_RE.search(raw)
    if not m:
        return []
    decision = m.group("body").strip().splitlines()[0].strip().lower() if m.group("body").strip() else ""
    if "reject" not in decision:
        return []
    slug = _artifact_slug(path)
    reason = (_section(raw, "Findings") or _section(raw, "Risk Summary")
              or "rejected in review")[:1500]
    verdict = ReviewVerdict(
        decision="reject",
        mechanism=slug,
        target_pattern=slug,
        reason=reason,
        review_ref=f"opencode:{path.name}",
    )
    run_id = f"opencode:{path.name}"
    return [
        review_to_anti_pattern(verdict, run_id=run_id, contributor=contributor, seq=1),
        review_to_bad_plan(verdict, run_id=run_id, seq=1),
    ]


def parse_bench_validation(path: Path, *, contributor: str, errors: list[str]
                           ) -> list[Candidate]:
    """A bench/*_validation.md with a PASS verdict and a measured delta ->
    memory_item fact. Fail/inconclusive/skipped yield nothing (the rejection
    knowledge comes from the review / ledger, with its reason attached)."""
    raw = path.read_text(encoding="utf-8")
    vm = _VERDICT_RE.search(raw)
    if not vm or vm.group(1).lower() != "pass":
        return []
    dm = _DELTA_RE.search(raw)
    if not dm:
        errors.append(f"{path.name}: pass verdict but no delta_pct found — skipped")
        return []
    cm = _COMPARE_RE.search(raw)
    slug = _artifact_slug(path)
    bench = BenchResult(
        mechanism=slug,
        target=slug,
        delta_pct=float(dm.group(1)),
        compare_level=(cm.group(1).lower() if cm else "total"),  # type: ignore[arg-type]
        validation_path=f".opencode/bench/{path.name}",
        verdict="landed",
    )
    return [bench_to_fact(bench, run_id=f"opencode:{path.name}",
                          contributor=contributor, seq=1)]


def parse_state_bad_plans(path: Path, *, contributor: str, errors: list[str]
                          ) -> list[Candidate]:
    """state/*bad_plans*.md bullet entries -> global bad_plan candidates."""
    raw = _strip_comments(path.read_text(encoding="utf-8"))
    out: list[Candidate] = []
    for seq, text in enumerate(_plain_bullets(raw), start=1):
        cand = _bad_plan(text, "*", None, contributor, seq, path)
        cand["record"]["target_pattern"] = "*"  # global pattern, not a slug
        out.append(cand)
    return out


def _collect_free_text(root: Path) -> list[str]:
    """research design/plan docs -> raw text for the optional LLM salience pass.
    Templates are skipped; deterministic extraction is not attempted here."""
    texts: list[str] = []
    for sub in ("docs", "plans"):
        d = root / sub
        if not d.exists():
            continue
        for p in sorted(d.glob("*.md")):
            if "template" in p.name.lower() or p.name == "README.md":
                continue
            try:
                texts.append(p.read_text(encoding="utf-8"))
            except OSError:
                continue
    return texts


# --- orchestrator -----------------------------------------------------------

def read_opencode_memory(opencode_root: str | Path, *, contributor: str = "opencode"
                         ) -> OpenCodeMemory:
    """Walk a member's `.opencode/memory/` tree into validated-shape candidates."""
    root = Path(opencode_root)
    mem = root / "memory" if (root / "memory").exists() else root
    result = OpenCodeMemory()

    def _safe(fn, p: Path, source: str, **kw):
        result.scanned[source] = result.scanned.get(source, 0) + 1
        try:
            result.candidates.extend(fn(p, contributor=contributor,
                                        errors=result.parse_errors, **kw))
        except OSError as e:
            result.parse_errors.append(f"{p}: {e}")

    ledger_dir = mem / "idea_ledger"
    if ledger_dir.exists():
        for p in sorted(ledger_dir.glob("*.md")):
            if p.name in {"README.md", "template.md"}:
                continue
            _safe(parse_idea_ledger, p, "memory/idea_ledger")

    gl = mem / "global_lessons.md"
    if gl.exists():
        _safe(parse_global_lessons, gl, "memory/global_lessons")

    for p in sorted((mem / "targets").glob("*.md")) if (mem / "targets").exists() else []:
        if p.name in {"README.md", "target_memory_template.md"}:
            continue
        _safe(parse_target_memory, p, "memory/targets")

    for p in sorted((mem / "subsystems").glob("*.md")) if (mem / "subsystems").exists() else []:
        if p.name in {"README.md"}:
            continue
        _safe(parse_target_memory, p, "memory/subsystems", subsystem=_slugify(p.stem))

    # Tier-0 run artifacts (safety net beside the curated memory/ layer).
    # Skipped entirely when the caller points at a bare memory/ tree.
    if (root / "memory").exists():
        reviews = root / "reviews"
        if reviews.exists():
            for p in sorted(reviews.glob("*.md")):
                if "template" in p.name.lower():
                    continue
                _safe(parse_review, p, "reviews")

        bench = root / "bench"
        if bench.exists():
            for p in sorted(bench.glob("*_validation.md")):
                if "template" in p.name.lower():
                    continue
                _safe(parse_bench_validation, p, "bench")

        state = root / "state"
        if state.exists():
            for p in sorted(state.glob("*bad_plans*.md")):
                _safe(parse_state_bad_plans, p, "state/bad_plans")

        result.free_text = _collect_free_text(root)
        if result.free_text:
            result.scanned["docs+plans free_text"] = len(result.free_text)

    return result
