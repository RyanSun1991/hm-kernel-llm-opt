"""Runtime composition: resolve (target, stage) into skills + knowledge.

Implements design §12.2: selector matches a domain skill -> pull `requires`
(core + technique) -> retrieve target-anchored + mechanism-anchored knowledge
from hub (and local in-flight memory) -> merge/dedup (hub precedence) -> trim to
a per-stage context budget -> log `retrieval.jsonl` (§12.4).

Hub vs local are not a flat overlay — they contribute different facets: hub
gives curated L2/L3 knowledge, local adds the member's not-yet-promoted ideas.
"""
from __future__ import annotations

import fnmatch
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:  # pragma: no cover
    raise ImportError("PyYAML required for skillhub.resolver") from None

from .embeddings import Embedder
from .records import load_records
from .retrieval import HybridRetriever, RetrievalHit, ScopeFilter


def _slugify(value: str) -> str:
    """Same slug convention as hmopt.opencode.pipeline so target_slug aligns
    across resolver / runs / sediment."""
    import re

    normalized = re.sub(r"[^a-zA-Z0-9]+", "-", value.strip().lower()).strip("-")
    return normalized or "task"


# (knowledge top_k, token cap) per pipeline stage — design §12.2 table.
STAGE_BUDGET: dict[str, tuple[int, int]] = {
    "research": (8, 3000),
    "plan": (5, 2000),
    "plan-review": (5, 2000),
    "implement": (3, 1500),
    "code-review": (5, 2000),
    "test": (3, 1000),
    "decision": (3, 1000),
}
_DEFAULT_BUDGET = (5, 2000)
_MATURITY_RANK = {"L0": 0, "L1": 1, "L2": 2, "L3": 3}


@dataclass
class Skill:
    name: str
    kind: str
    path: str
    applies_to: dict[str, Any] = field(default_factory=dict)
    requires: list[str] = field(default_factory=list)
    fields: dict[str, Any] = field(default_factory=dict)

    @property
    def ref(self) -> str:
        return f"{self.kind}/{self.name}"


@dataclass
class ResolvedContext:
    target: str
    stage: str
    target_slug: str
    subsystems: list[str]
    skills: list[Skill]
    knowledge: list[RetrievalHit]
    queries: dict[str, str]
    token_budget: int
    token_used: int
    latency_ms: float = 0.0

    def to_log_record(self) -> dict[str, Any]:
        return {
            "target": self.target,
            "stage": self.stage,
            "scope": {"target_slug": self.target_slug, "subsystems": self.subsystems},
            "queries": self.queries,
            "returned_ids": [h.record.id for h in self.knowledge],
            "skills": [s.ref for s in self.skills],
            "token_budget": self.token_budget,
            "token_used": self.token_used,
            "latency_ms": round(self.latency_ms, 2),
        }


def find_hub_root(repo_root: str | Path) -> Path | None:
    root = Path(repo_root)
    for cand in (root / ".opencode" / "hub", root / "hm-skill-hub"):
        if (cand / "knowledge").exists() or (cand / "skills").exists():
            return cand
    return None


_FRONTMATTER_RE = __import__("re").compile(r"^---\n(.*?)\n---\n", __import__("re").DOTALL)


def load_skills(hub_root: str | Path) -> dict[str, Skill]:
    root = Path(hub_root) / "skills"
    out: dict[str, Skill] = {}
    if not root.exists():
        return out
    for sk in sorted(root.rglob("SKILL.md")):
        m = _FRONTMATTER_RE.match(sk.read_text(encoding="utf-8"))
        if not m:
            continue
        fm = yaml.safe_load(m.group(1)) or {}
        name = fm.get("name")
        kind = fm.get("kind")
        if not name or not kind:
            continue
        skill = Skill(
            name=str(name),
            kind=str(kind),
            path=str(sk),
            applies_to=fm.get("applies_to") or {},
            requires=list(fm.get("requires") or []),
            fields=fm,
        )
        out[skill.ref] = skill
    return out


def load_selectors(hub_root: str | Path) -> dict[str, dict[str, list[str]]]:
    path = Path(hub_root) / "_registry" / "subsystem_selectors.yaml"
    if not path.exists():
        return {}
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    out: dict[str, dict[str, list[str]]] = {}
    for entry in raw.get("subsystems", []) or []:
        name = entry.get("name")
        if not name:
            continue
        out[name] = {
            "path_globs": list(entry.get("path_globs") or []),
            "symbol_selectors": list(entry.get("symbol_selectors") or []),
        }
    return out


def split_target(target: str) -> tuple[str, str | None]:
    """`mm/vmscan.c::shrink_node` -> ('mm/vmscan.c', 'shrink_node')."""
    if "::" in target:
        path, sym = target.split("::", 1)
        return path.strip(), sym.strip() or None
    return target.strip(), None


def match_subsystems(target: str, selectors: dict[str, dict[str, list[str]]]) -> list[str]:
    path, symbol = split_target(target)
    matched: list[str] = []
    for sub, sel in selectors.items():
        path_hit = any(fnmatch.fnmatch(path, g) for g in sel.get("path_globs", []))
        sym_hit = bool(symbol) and any(
            fnmatch.fnmatch(symbol, g) for g in sel.get("symbol_selectors", [])
        )
        if path_hit or sym_hit:
            matched.append(sub)
    return matched


class Resolver:
    def __init__(
        self,
        repo_root: str | Path,
        hub_root: str | Path | None = None,
        embedder: Embedder | None = None,
        local_memory_root: str | Path | None = None,
    ):
        self.repo_root = Path(repo_root)
        self.hub_root = Path(hub_root) if hub_root else find_hub_root(repo_root)
        if self.hub_root is None:
            raise FileNotFoundError("no skill hub found (.opencode/hub or hm-skill-hub)")
        self.skills = load_skills(self.hub_root)
        self.selectors = load_selectors(self.hub_root)
        self.hub_records = load_records(self.hub_root / "knowledge", origin="hub")
        self.hub_retriever = HybridRetriever(self.hub_records, embedder=embedder)
        self.local_retriever: HybridRetriever | None = None
        if local_memory_root and Path(local_memory_root).exists():
            local_records = load_records(local_memory_root, origin="local")
            if local_records:
                self.local_retriever = HybridRetriever(local_records, embedder=embedder)

    # --- skills -------------------------------------------------------------

    def _resolve_skill_closure(self, subsystems: list[str]) -> list[Skill]:
        """Domain skills matching the subsystems + transitive `requires`."""
        seen: dict[str, Skill] = {}
        frontier: list[str] = []
        for skill in self.skills.values():
            if skill.kind == "domain":
                subs = set(skill.applies_to.get("subsystems") or [])
                if subs & set(subsystems):
                    seen[skill.ref] = skill
                    frontier.extend(skill.requires)
        while frontier:
            ref = frontier.pop()
            if ref in seen:
                continue
            sk = self.skills.get(ref)
            if sk is None:
                continue
            seen[ref] = sk
            frontier.extend(sk.requires)
        # stable order: core, technique, domain
        order = {"core": 0, "technique": 1, "domain": 2}
        return sorted(seen.values(), key=lambda s: (order.get(s.kind, 3), s.name))

    # --- knowledge ----------------------------------------------------------

    def _retrieve(self, queries: dict[str, str], scope: ScopeFilter, top_k: int,
                  mechanism: str | None) -> list[RetrievalHit]:
        merged: dict[str, RetrievalHit] = {}

        def absorb(hits: list[RetrievalHit], prefer: bool) -> None:
            for h in hits:
                cur = merged.get(h.record.id)
                if cur is None:
                    merged[h.record.id] = h
                elif prefer and h.record.origin == "hub" and cur.record.origin == "local":
                    merged[h.record.id] = h  # hub precedence on same stable id

        ta = self.hub_retriever.retrieve(queries["target_anchored"], k=top_k, scope=scope,
                                         maturity_floor="L2", mechanism=mechanism)
        absorb(ta, prefer=True)
        if "mechanism_anchored" in queries:
            ma = self.hub_retriever.retrieve(queries["mechanism_anchored"], k=top_k, scope=scope,
                                             maturity_floor="L2")
            absorb(ma, prefer=True)
        if self.local_retriever is not None:
            la = self.local_retriever.retrieve(queries["target_anchored"], k=top_k, scope=scope,
                                               maturity_floor="L0")
            absorb(la, prefer=False)  # local only fills ids hub didn't supply

        hits = sorted(merged.values(), key=lambda h: h.score, reverse=True)
        return hits

    @staticmethod
    def _est_tokens(text: str) -> int:
        return max(1, len(text) // 4)

    def _trim_to_budget(self, hits: list[RetrievalHit], top_k: int, token_cap: int
                        ) -> tuple[list[RetrievalHit], int]:
        hits = hits[:top_k]
        # if over budget, drop weakest first: low maturity, low score, weak evidence
        def weakness(h: RetrievalHit):
            r = h.record
            ev = (r.fields.get("evidence") or {})
            conf = ev.get("confirmations", 0) if isinstance(ev, dict) else 0
            return (_MATURITY_RANK.get(r.maturity, 0), r.score, conf, h.score)

        kept = list(hits)
        used = sum(self._est_tokens(h.record.search_text()) for h in kept)
        # Drop weakest first, but never below one record: a stage should still get
        # the single strongest hit even if it alone exceeds the cap (truncate-at-
        # consume-time rather than return nothing).
        while len(kept) > 1 and used > token_cap:
            victim = min(kept, key=weakness)
            kept.remove(victim)
            used -= self._est_tokens(victim.record.search_text())
        kept.sort(key=lambda h: h.score, reverse=True)
        return kept, used

    # --- main ---------------------------------------------------------------

    def resolve(self, target: str, stage: str, mechanism: str | None = None,
                run_dir: str | Path | None = None) -> ResolvedContext:
        t0 = time.perf_counter()
        slug = _slugify(target)
        _, symbol = split_target(target)
        subsystems = match_subsystems(target, self.selectors)
        skills = self._resolve_skill_closure(subsystems)

        top_k, token_cap = STAGE_BUDGET.get(stage, _DEFAULT_BUDGET)
        queries = {"target_anchored": f"{target} {symbol or ''}".strip()}
        if mechanism:
            queries["mechanism_anchored"] = mechanism

        scope = ScopeFilter(
            subsystem=subsystems[0] if subsystems else None,
            target_slug=slug,
        )
        hits = self._retrieve(queries, scope, top_k, mechanism)
        kept, used = self._trim_to_budget(hits, top_k, token_cap)

        ctx = ResolvedContext(
            target=target, stage=stage, target_slug=slug, subsystems=subsystems,
            skills=skills, knowledge=kept, queries=queries,
            token_budget=token_cap, token_used=used,
            latency_ms=(time.perf_counter() - t0) * 1000.0,
        )
        if run_dir:
            self._log_retrieval(Path(run_dir), ctx)
        return ctx

    @staticmethod
    def _log_retrieval(run_dir: Path, ctx: ResolvedContext) -> None:
        run_dir.mkdir(parents=True, exist_ok=True)
        with open(run_dir / "retrieval.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(ctx.to_log_record(), ensure_ascii=False) + "\n")
