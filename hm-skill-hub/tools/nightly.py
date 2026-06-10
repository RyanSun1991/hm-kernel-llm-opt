"""Nightly closed-loop optimization job (design §11, plan P4-1).

Composes the engine-A and engine-B tools into the 7-step loop:

    (1) Collect    gather staging/*.jsonl candidates
    (2) Normalize  schema lint + secret-scan (redact)
    (3) Cluster    central_curate: dedup / conflict / subsumption + promotion signals
    (4) Optimize   skill_optimizer (bounded edit under the eval gate, engine B)
    (5) Validate   eval_gate: reject any skill regression
    (6) Promote    release: semver bump + registry + notes
    (7) Broadcast  regenerate skill-memory.lock for consumers

Default is **dry-run / half-automatic** (design §11 early-safety: auto-PR, human
merge). `--apply` writes the artifacts (optimizer edits, scorecards, registry,
release notes, lock, dashboard) — reserved for the trusted/approved path.

CLI:
    python tools/nightly.py [--apply] [--min-trust=3]
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import broadcast  # type: ignore
import dashboard  # type: ignore
import lint  # type: ignore
import redact  # type: ignore
import release  # type: ignore
import skill_optimizer as so  # type: ignore
from auto_merge_gate import decide  # type: ignore
from central_curate import curate_batch  # type: ignore
from hub_records import load_hub_knowledge  # type: ignore

HUB = Path(__file__).resolve().parent.parent


def validate_candidates(candidates: list[dict]) -> tuple[list[dict], list[str]]:
    """Schema-validate staged candidates against the hub JSON-Schemas.

    `_collect` reads `staging/*.jsonl` as raw dicts that previously flowed
    straight into the curator with no schema check — the "Normalize" step only
    linted `knowledge/`, never the inbound candidates. A malformed candidate
    (wrong id pattern, missing required fields) could enter central curation and
    be "added". We validate here and drop invalid candidates before clustering.
    """
    import jsonschema  # provided via lint's dependency

    valid: list[dict] = []
    invalid: list[str] = []
    for c in candidates:
        schema_name = c.get("schema")
        record = c.get("record")
        if not schema_name or not isinstance(record, dict):
            invalid.append(f"{c.get('schema', '?')}: missing 'schema' or 'record'")
            continue
        try:
            schema = lint.load_schema(str(schema_name))
        except OSError:
            invalid.append(f"{schema_name}: unknown schema")
            continue
        errs = sorted(
            jsonschema.Draft7Validator(schema).iter_errors(record),
            key=lambda e: list(e.absolute_path),
        )
        if errs:
            rid = record.get("id", "?")
            invalid.append(f"{schema_name}/{rid}: {errs[0].message}")
        else:
            valid.append(c)
    return valid, invalid


@dataclass
class NightlyReport:
    collected: int = 0
    schema_invalid: list[str] = field(default_factory=list)
    normalize_ok: bool = False
    cluster: dict = field(default_factory=dict)
    optimize: list = field(default_factory=list)
    validate_ok: bool = True
    validate_msgs: list = field(default_factory=list)
    release: dict = field(default_factory=dict)
    lock: str = ""
    auto_merge: dict = field(default_factory=dict)
    applied: bool = False

    def summary(self) -> str:
        out = [
            "# Nightly report",
            f"- collected candidates: {self.collected}",
            f"- normalize (lint+redact): {'ok' if self.normalize_ok else 'FAILED'}",
            f"- cluster decisions: {self.cluster.get('counts', {})}; "
            f"promotion signals: {self.cluster.get('promotion_signals', [])}",
            f"- optimize: {sum(len(o['accepted']) for o in self.optimize)} accepted edit(s) across "
            f"{len(self.optimize)} skill(s)",
            f"- validate (eval-gate): {'ok' if self.validate_ok else 'REGRESSION'}",
            f"- promote: {self.release.get('old_version')} → {self.release.get('new_version')} "
            f"({self.release.get('bump')})",
            f"- auto-merge: {self.auto_merge}",
            f"- applied: {self.applied}",
        ]
        return "\n".join(out)


def _collect(hub_root: Path) -> list[dict]:
    out: list[dict] = []
    staging = hub_root / "staging"
    if staging.exists():
        for p in sorted(staging.rglob("*.jsonl")):
            for line in p.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line:
                    out.append(json.loads(line))
    return out


def run_nightly(hub_root: str | Path = HUB, *, apply: bool = False,
                min_trust: int = 3) -> NightlyReport:
    hub_root = Path(hub_root)
    rep = NightlyReport(applied=apply)

    # (1) Collect
    candidates = _collect(hub_root)
    rep.collected = len(candidates)

    # (2) Normalize — schema + secrets. Validate BOTH the existing knowledge
    # (lint) AND the inbound staged candidates (which lint does not cover), so a
    # malformed candidate cannot slip into curation.
    candidates, rep.schema_invalid = validate_candidates(candidates)
    lint_ok = lint.main([str(hub_root)]) == 0
    redact_ok = redact.main(["--check", str(hub_root)]) == 0
    rep.normalize_ok = lint_ok and redact_ok and not rep.schema_invalid

    # (3) Cluster — engine A (only schema-valid candidates)
    cr = curate_batch(candidates, load_hub_knowledge(hub_root))
    rep.cluster = {"counts": cr.counts(), "promotion_signals": cr.promotion_signals,
                   "promotions": [c.proposed_skill for c in cr.promotions]}

    # (4) Optimize — engine B (bounded edit under the gate). Each result carries
    # its PRE-optimize baseline (captured before any write). The optimizer only
    # WRITES when Normalize (schema + secrets) passed — a hub failing lint/redact
    # must never be mutated by the trusted path; it still runs dry for the report.
    opt_apply = apply and rep.normalize_ok
    skill_dirs = [p.parent for p in sorted((hub_root / "skills").rglob("SKILL.md"))
                  if (p.parent / "best_skill.md").exists()]
    results: dict = {}
    accepted_any = False
    for d in skill_dirs:
        res = so.optimize(d, apply=opt_apply)
        results[d] = res
        accepted_any = accepted_any or bool(res.accepted)
        rep.optimize.append({"skill": d.name,
                             "baseline": round(res.baseline.pass_rate, 3),
                             "final": round(res.final.pass_rate, 3),
                             "accepted": [e.mechanism for e in res.accepted]})

    # (5) Validate — the optimizer's `final` must not regress vs the PRE-optimize
    # baseline (independent of the scorecard it just wrote; this is the real
    # monotone gate in the loop — eval_gate.py is the separate cross-PR tool, run
    # in CI for hand edits).
    for d, res in results.items():
        reg = res.final.regression_rate_vs(res.baseline)
        ok = reg == 0.0 and res.final.pass_rate >= res.baseline.pass_rate
        rep.validate_msgs.append(
            f"{d.name}: pass_rate {res.baseline.pass_rate:.2f}→{res.final.pass_rate:.2f}, "
            f"regression={reg:.2f}")
        rep.validate_ok = rep.validate_ok and ok

    # (6) Promote — release bump (escalate to minor if a skill improved)
    rel = release.compute_release(hub_root, bump="auto")
    if accepted_any and rel["bump"] == "patch":
        rel = release.compute_release(hub_root, bump="minor")
    rep.release = {k: v for k, v in rel.items() if k not in ("inventory", "prev_releases")}

    # (7) Broadcast — lock for consumers
    rep.lock = broadcast.lock_content(rel["new_version"])

    # auto-merge decision per skill (half-automatic until trusted)
    for d in skill_dirs:
        cards = [json.loads(p.read_text(encoding="utf-8"))
                 for p in sorted((d / "scorecards").glob("*.json"))] if (d / "scorecards").exists() else []
        if cards:
            mode, reason = decide(cards, min_trust)
            rep.auto_merge[d.name] = mode

    if apply and rep.normalize_ok and rep.validate_ok:
        # operate on the hub we were handed (NOT the module-level HUB) so a temp /
        # alternate hub is never bypassed and the real hub isn't mutated.
        release.apply_release(hub_root, bump=rep.release["bump"])
        broadcast.write_lock(hub_root.parent / ".opencode" / "skill-memory.lock",
                             rel["new_version"])
        dashboard.write_dashboard(hub_root)
    return rep


def main(argv: list[str]) -> int:
    apply = "--apply" in argv
    min_trust = int(next((a.split("=", 1)[1] for a in argv if a.startswith("--min-trust=")), "3"))
    rep = run_nightly(HUB, apply=apply, min_trust=min_trust)
    print(rep.summary())
    if not rep.normalize_ok:
        sys.stderr.write("\nnightly: normalize gate failed — aborting promote.\n")
        return 1
    if not rep.validate_ok:
        sys.stderr.write("\nnightly: eval-gate regression — aborting promote.\n")
        return 1
    if not apply:
        print("\n(dry-run; pass --apply to write edits/registry/lock/dashboard — "
              "default is half-automatic per design §11)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
