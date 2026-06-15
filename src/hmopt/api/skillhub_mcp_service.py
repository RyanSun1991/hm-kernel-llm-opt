"""MCP tool service for the team Skill Hub read/write bridge.

OpenCode agents run inside a *kernel* repo and reach this platform only through
MCP — exactly like the git / build / auto-test servers. They cannot run the
`hmopt` CLI directly (it is not installed in their environment). This service
exposes the hub bridge as MCP tools instead:

  skillhub_resolve   — READ  (wraps `hmopt resolve`): mount team skills + knowledge
                       for a target/stage; returns a ready-to-paste `## Hub context`
                       block and appends an audit line to `<opencode>/state/retrieval.jsonl`.
  skillhub_sediment  — WRITE (wraps `hmopt sediment-opencode`): distill the member's
                       `.opencode/memory` into hub candidates + a `_bundle.jsonl`.
  skillhub_status    — report the pinned hub version and whether the hub is reachable.

The kernel repo's `.opencode/` is volume-mounted into this container (see
docker-compose `hmopt-skillhub-mcp`), so the tools take an absolute `opencode_dir`
the server can read/write. Every tool degrades gracefully — a missing hub never
raises; it returns a `hub: unavailable` note so the pipeline is never blocked.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_SERVER_NAME = "hmopt-skillhub-mcp"


# --------------------------------------------------------------------------- #
# Path / hub discovery
# --------------------------------------------------------------------------- #
def _default_opencode_dir() -> str:
    """Filesystem path to the member's `.opencode/`.

    NOTE: never derive this from ``HMOPT_SKILLHUB_MCP_MOUNT_PATH`` — that env is the
    HTTP streamable-http path (e.g. ``/mcp``), not a filesystem location. Prefer the
    explicit env, then the canonical docker kernel mount, then the cwd; pick the
    first that actually exists so a misconfigured env does not shadow a real tree.
    """
    env = os.getenv("HMOPT_SKILLHUB_OPENCODE_DIR", "").strip()
    candidates = [c for c in (env, "/workspace/kernel/.opencode", str(Path.cwd() / ".opencode")) if c]
    for c in candidates:
        if Path(c).is_dir():
            return c
    return env or "/workspace/kernel/.opencode"


def _resolve_opencode_dir(opencode_dir: str | None) -> Path:
    return Path(opencode_dir or _default_opencode_dir()).expanduser()


def _opencode_missing_msg(oc: Path) -> str:
    return (
        f"hub: unavailable (opencode_dir not found: {oc}). Pass opencode_dir="
        "<abs path to the member's .opencode>, or set HMOPT_SKILLHUB_OPENCODE_DIR on "
        "the skill-hub MCP server (and mount that path into the container)."
    )


def _resolve_hub(opencode_dir: Path) -> Path | None:
    """Find the hub: kernel-repo `.opencode/hub` or `<repo>/hm-skill-hub`, then the
    platform install's `hm-skill-hub`, then `HMOPT_SKILLHUB_HUB_ROOT`."""
    from hmopt.skillhub.resolver import find_hub_root

    hub = find_hub_root(opencode_dir.parent)
    if hub is not None:
        return hub
    here = Path(__file__).resolve()
    for up in here.parents:
        cand = up / "hm-skill-hub"
        if (cand / "knowledge").exists() or (cand / "skills").exists():
            return cand
    env = os.getenv("HMOPT_SKILLHUB_HUB_ROOT", "").strip()
    if env and Path(env).exists():
        return Path(env)
    return None


def _hub_version(opencode_dir: Path, hub_root: Path | None) -> str:
    lock = opencode_dir / "skill-memory.lock"
    if lock.exists():
        for line in lock.read_text(encoding="utf-8").splitlines():
            s = line.split("#", 1)[0].strip()
            if s.startswith("hub_version:"):
                return s.split(":", 1)[1].strip() or "unknown"
    if hub_root is not None:
        reg = hub_root / "registry.yaml"
        if reg.exists():
            for line in reg.read_text(encoding="utf-8").splitlines():
                if line.strip().startswith("version:"):
                    return line.split(":", 1)[1].strip() or "unknown"
    return "unknown"


# --------------------------------------------------------------------------- #
# Read path
# --------------------------------------------------------------------------- #
def _format_hub_context(ctx: Any, hub_version: str) -> str:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    lines = [f"## Hub context (resolved {now} @ hub {hub_version}, stage={ctx.stage})"]
    if ctx.skills:
        lines.append("Skills (read-only guidance): " + ", ".join(s.ref for s in ctx.skills))
    else:
        lines.append("Skills (read-only guidance): (none)")
    if ctx.knowledge:
        lines.append("Team knowledge (cite by id; do NOT re-derive; dedup against these):")
        for hit in ctx.knowledge:
            r = hit.record
            kind = str(r.fields.get("kind") or r.kind)
            mark = " — DO NOT propose" if kind == "bad_plan" or r.kind == "bad_plan" else ""
            lines.append(f"- [{r.id}·{r.maturity}·{kind}] {r.title}{mark}")
    else:
        lines.append("Team knowledge: (none — no hub coverage for this target yet)")
    ids = ", ".join(h.record.id for h in ctx.knowledge)
    lines.append(f"Audit: <opencode>/state/retrieval.jsonl (returned_ids=[{ids}])")
    return "\n".join(lines)


def skillhub_resolve(
    target: str,
    *,
    stage: str = "research",
    opencode_dir: str | None = None,
    mechanism: str | None = None,
) -> str:
    """READ path. Returns a `## Hub context` block ready to paste into a design doc
    or a delegation handoff. Non-blocking: returns a `hub: unavailable` note instead
    of raising when the hub or its dependencies are missing."""
    oc = _resolve_opencode_dir(opencode_dir)
    if not oc.is_dir():
        return _opencode_missing_msg(oc)
    try:
        from hmopt.skillhub.resolver import Resolver
    except Exception as exc:  # pragma: no cover - import guard
        return f"hub: unavailable (skillhub import failed: {exc})"

    hub_root = _resolve_hub(oc)
    if hub_root is None:
        return (
            "hub: unavailable (no .opencode/hub or hm-skill-hub found). "
            "Proceed without hub context; mount the hub to enable it."
        )
    local_mem = oc / "memory"
    run_dir = oc / "state"
    try:
        run_dir.mkdir(parents=True, exist_ok=True)
        r = Resolver(
            oc.parent,
            hub_root=hub_root,
            local_memory_root=str(local_mem) if local_mem.exists() else None,
        )
        ctx = r.resolve(target, stage=stage, mechanism=mechanism, run_dir=str(run_dir))
    except Exception as exc:
        logger.warning("skillhub_resolve failed (opencode_dir=%s): %s", oc, exc, exc_info=True)
        return f"hub: unavailable (resolve error at {oc}: {exc})"
    return _format_hub_context(ctx, _hub_version(oc, hub_root))


# --------------------------------------------------------------------------- #
# Write path
# --------------------------------------------------------------------------- #
def skillhub_sediment(
    *,
    opencode_dir: str | None = None,
    contributor: str = "opencode",
    bundle: bool = True,
) -> str:
    """WRITE path. Distill `<opencode>/memory` (+ tier-0 reviews/bench/state) into
    hub candidates under `<opencode>/local/sediment_staging`, optionally bundling
    them into `_bundle.jsonl`. Non-blocking: 0 candidates is a valid result."""
    oc = _resolve_opencode_dir(opencode_dir)
    if not oc.is_dir():
        return _opencode_missing_msg(oc)
    try:
        from hmopt.sediment.pipeline import bundle_staging, sediment_opencode
    except Exception as exc:  # pragma: no cover - import guard
        return f"hub: unavailable (sediment import failed: {exc})"

    hub_root = _resolve_hub(oc)
    out_dir = oc / "local" / "sediment_staging"
    try:
        res = sediment_opencode(
            oc, out_dir=out_dir, contributor=contributor, hub_root=hub_root
        )
    except Exception as exc:
        logger.warning("skillhub_sediment failed (opencode_dir=%s): %s", oc, exc, exc_info=True)
        return f"hub: unavailable (sediment error at {oc}: {exc})"

    # Read summary fields defensively: the MCP server and the hmopt sediment module
    # may drift across container/version rebuilds (e.g. an older SedimentResult with
    # no `scanned`). A cosmetic field must never crash the WRITE path.
    run_id = getattr(res, "run_id", "?")
    out_path = getattr(res, "out_path", None)
    n_valid = getattr(res, "n_valid", None)
    if n_valid is None:
        n_valid = len(getattr(res, "candidates", []) or [])
    scanned = getattr(res, "scanned", None)
    invalid = getattr(res, "invalid", None) or []
    parse_errors = getattr(res, "parse_errors", None) or []
    lines = [f"sediment-opencode[{run_id}]: {n_valid} valid candidate(s) -> {out_path}"]
    if invalid:
        lines.append(f"  invalid: {len(invalid)} candidate(s) failed schema validation")
    if scanned:
        lines.append(f"  scanned: {dict(scanned)}")
    if parse_errors:
        lines.append(f"  notes: {parse_errors[0]}")
    if bundle:
        try:
            bundle_path, n = bundle_staging(out_dir, out_dir / "_bundle.jsonl")
            lines.append(f"bundle: {n} record(s) -> {bundle_path}")
            lines.append(
                "To share: copy the bundle to <hub>/staging/<member>/<date>.jsonl and open a PR "
                "(the member decides what to share — not auto-pushed)."
            )
        except Exception as exc:
            lines.append(f"bundle: skipped ({exc})")
    return "\n".join(lines)


def skillhub_status(*, opencode_dir: str | None = None) -> str:
    """Report the pinned hub version and reachability for this `.opencode/` tree."""
    oc = _resolve_opencode_dir(opencode_dir)
    hub_root = _resolve_hub(oc)
    reachable = hub_root is not None and (hub_root / "knowledge").exists()
    return (
        f"opencode_dir: {oc}\n"
        f"hub_root: {hub_root if hub_root else 'unavailable'}\n"
        f"hub_reachable: {reachable}\n"
        f"hub_version: {_hub_version(oc, hub_root)}"
    )


# --------------------------------------------------------------------------- #
# FastMCP server
# --------------------------------------------------------------------------- #
def build_skillhub_fastmcp_server() -> Any | None:
    try:
        from mcp.server.fastmcp import FastMCP  # type: ignore
    except ImportError:
        logger.warning(
            "mcp package is not installed; skill-hub MCP endpoint unavailable. "
            "Install with: pip install 'mcp[cli]'"
        )
        return None

    server_name = os.getenv("HMOPT_SKILLHUB_MCP_SERVER_NAME", DEFAULT_SERVER_NAME).strip() or DEFAULT_SERVER_NAME
    disable_host_check = os.getenv("HMOPT_SKILLHUB_MCP_DISABLE_HOST_CHECK", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    transport_security = None
    if disable_host_check:
        from mcp.server.transport_security import TransportSecuritySettings  # type: ignore

        transport_security = TransportSecuritySettings(enable_dns_rebinding_protection=False)

    mcp = FastMCP(
        server_name,
        stateless_http=True,
        json_response=True,
        transport_security=transport_security,
    )

    @mcp.tool(
        name="skillhub_resolve",
        description=(
            "READ the team Skill Hub for a kernel target. Returns a `## Hub context` "
            "block (team skills + curated facts/heuristics + rejected bad-plans) to "
            "paste into the research design doc or a plan-review handoff. Call at the "
            "start of research and before plan-review. Degrades to a `hub: unavailable` "
            "note if the hub is not mounted — never blocks."
        ),
    )
    def _resolve(
        target: str,
        stage: str = "research",
        opencode_dir: str | None = None,
        mechanism: str | None = None,
    ) -> str:
        return skillhub_resolve(
            target, stage=stage, opencode_dir=opencode_dir, mechanism=mechanism
        )

    @mcp.tool(
        name="skillhub_sediment",
        description=(
            "WRITE path: distill this member's .opencode/memory into team-hub "
            "candidates and a _bundle.jsonl, at the decision stage of a clean pass. "
            "Returns a summary + the human copy-to-staging + open-PR instruction. "
            "Non-blocking; 0 candidates is valid."
        ),
    )
    def _sediment(
        opencode_dir: str | None = None,
        contributor: str = "opencode",
        bundle: bool = True,
    ) -> str:
        return skillhub_sediment(
            opencode_dir=opencode_dir, contributor=contributor, bundle=bundle
        )

    @mcp.tool(
        name="skillhub_status",
        description="Report the pinned hub version and whether the hub is reachable for this .opencode/ tree.",
    )
    def _status(opencode_dir: str | None = None) -> str:
        return skillhub_status(opencode_dir=opencode_dir)

    return mcp
