"""Read-only checks of local discovery prerequisites, never an execution verdict."""

from __future__ import annotations

import asyncio
import importlib
import importlib.util
import json
import os
import shutil
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from pathlib import Path

_EXPECTED_ERRORS = (
    OSError,
    ValueError,
    TypeError,
    RuntimeError,
    ImportError,
    sqlite3.Error,
    KeyError,
)
_OVERRIDES = (
    "HMOPT_EVOLUTION_ROOT",
    "HMOPT_EVOLUTION_GIT_BIN",
    "HMOPT_EVOLUTION_WORKSPACE_ROOT",
    "HMOPT_EVOLUTION_ARTIFACTS_ROOT",
    "HMOPT_EVOLUTION_DISCOVERY_PROFILES",
)
_TOOLS = {
    "evolution_workspace",
    "evolution_scan",
    "evolution_experiment",
    "evolution_production_status",
    "evolution_start_mining",
    "evolution_mining_control",
    "evolution_suggest_groups",
    "evolution_request_approval",
    "evolution_evaluate",
    "evolution_code_context",
    "evolution_prepare_research",
    "evolution_submit_research",
    "evolution_candidates",
    "evolution_dossier",
    "evolution_create_batch",
    "evolution_batch_next",
    "evolution_batch_block",
    "evolution_discovery_profiles",
    "evolution_run_discovery",
    "evolution_discovery_step",
    "evolution_history_analysis",
    "evolution_submit_history_analysis",
    "evolution_read",
    "evolution_digest",
    "evolution_list",
    "evolution_show",
    "evolution_handoff",
    "evolution_submit",
    "evolution_validate",
    "evolution_recall",
    "evolution_audit",
    "evolution_evidence",
    "evolution_capture",
    "evolution_quality",
    "evolution_convert_ic",
    "evolution_dispatch",
    "evolution_materialize_workspace",
    "evolution_convert_lmbench",
}


def _message(error: Exception) -> str:
    # Pydantic's default formatting echoes input values. Configuration diagnostics
    # need locations and reasons, never arbitrary pasted input or credentials.
    from pydantic import ValidationError

    if isinstance(error, ValidationError):
        return "; ".join(
            f"{'.'.join(map(str, item['loc']))}: {item['msg']}"
            for item in error.errors(include_input=False, include_url=False)[:10]
        )
    return str(error)[:1000] or type(error).__name__


def _registry(options: dict) -> int:
    from hmopt.api.evolution_mcp_service import build_evolution_fastmcp_server

    # Import the unified transport without invoking its globally cached factory
    # or modifying this process's environment/configuration.
    importlib.import_module("hmopt.api.mcp_stdio")
    server = build_evolution_fastmcp_server(**options)

    def collect():
        return asyncio.run(server.list_tools())

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        registered = collect()
    else:
        with ThreadPoolExecutor(max_workers=1) as executor:
            registered = executor.submit(collect).result()
    names = {tool.name for tool in registered}
    if names != _TOOLS or len(registered) != len(_TOOLS):
        raise ValueError("Evolution MCP registry does not expose exactly the expected tools")
    if any(tool.inputSchema.get("additionalProperties") is not False for tool in registered):
        raise ValueError("Evolution MCP registry does not reject extra tool arguments")
    return len(registered)


def _store_snapshot(root: Path) -> tuple[str, str, dict]:
    database = root / "evolution.sqlite3"
    if not database.exists():
        for location in (root, *root.parents):
            if location.exists():
                if not location.is_dir():
                    return "blocked", "Configured store root or its parent is not a directory.", {}
                break
        return "not_configured", "No workflow database exists; doctor did not create it.", {}
    if not database.is_file():
        return "blocked", "Configured workflow database is not a regular file.", {}
    wal = database.with_name(database.name + "-wal")
    if wal.exists() and wal.stat().st_size:
        return (
            "not_verified",
            "A WAL is present; obtain a stable snapshot before counting records.",
            {},
        )
    # immutable mode never creates journal/SHM files. A live WAL cannot be read
    # this way, so it was explicitly excluded above rather than silently ignored.
    with closing(
        sqlite3.connect(database.resolve().as_uri() + "?mode=ro&immutable=1", uri=True)
    ) as db:
        counts = dict(db.execute("SELECT kind,COUNT(*) FROM records GROUP BY kind"))
        patterns = db.execute(
            "SELECT payload FROM records WHERE kind='pattern' LIMIT 10001"
        ).fetchall()
    if wal.exists() and wal.stat().st_size:
        return "not_verified", "A WAL appeared while reading; stored counts were discarded.", {}
    if len(patterns) > 10000:
        return (
            "not_verified",
            "Pattern snapshot exceeds the bounded diagnostic limit.",
            {"records": counts},
        )
    active = sum(json.loads(payload)["status"] == "active" for (payload,) in patterns)
    return (
        "passed" if active else "warning",
        "Read-only stored record counts; active versions are not a candidate approval verdict.",
        {"records": counts, "active_pattern_versions": active},
    )


def doctor(config_path: Path) -> dict:
    """Inspect effective operator configuration without creating state or running work."""
    result = {
        "config_path": str(Path(config_path).absolute()),
        "scope": "offline_configuration_and_discovery_prerequisites",
        "execution_prerequisites_scope": "local_assets_and_clean_matching_checkout_only",
        "ready_for_discovery": False,
        "execution_prerequisites": False,
        "production_validated": False,
        "model_executed": False,
        "device_executed": False,
        "opencode_connected": False,
        "environment_overrides": [name for name in _OVERRIDES if os.getenv(name, "").strip()],
        "discovery_profiles": [],
        "checks": [],
        "next_steps": [],
    }

    def check(name, status, message, **details):
        result["checks"].append({"name": name, "status": status, "message": message, **details})

    try:
        from hmopt.evolution.mcp_discovery import prepare_profiles
        from hmopt.evolution.mining import _owners, read_git
        from hmopt.evolution.setup import _ASSETS

        from .configuration import read_environment_options

        options = read_environment_options(Path(config_path))
        profiles = prepare_profiles(options["discovery_profiles"])
        if options.get("source_workspace"):
            from .workspace import freeze_workspace

            manifest = freeze_workspace(options["source_workspace"], git_bin=options["git_bin"])
            check(
                "source_workspace",
                "passed",
                "Selected Git roots and build-only dependencies resolve independently.",
                workspace_id=manifest["workspace_id"],
                projects=[p["project_id"] for p in manifest["projects"]],
                dependencies=[p["project_id"] for p in manifest["dependencies"]],
            )
        result["discovery_profiles"] = list(profiles)
        root = Path(options["root"]).resolve()
        workspace = options.get("workspace_root")
        workspace = Path(workspace).resolve() if workspace is not None else None
        check(
            "configuration", "passed", "Configuration parsed with explicit environment overrides."
        )
    except _EXPECTED_ERRORS as error:
        check("configuration", "blocked", _message(error))
        result["next_steps"].append(
            "Correct the reported configuration error and run doctor again."
        )
        return result

    discovery_ok = True
    execution_ok = True
    if workspace is None or workspace.parts[-3:] != (".opencode", "local", "workspaces"):
        check(
            "workbench_layout", "blocked", "workspace_root must end in .opencode/local/workspaces."
        )
        discovery_ok = False
        execution_ok = False
    else:
        workbench = workspace.parents[2]
        required = [workbench / ".opencode" / asset for asset in _ASSETS]
        try:
            missing = [
                str(path.relative_to(workbench))
                for path in required
                if not path.is_file() or not path.read_text(encoding="utf-8-sig").strip()
            ]
            if missing:
                check(
                    "workbench_assets",
                    "blocked",
                    "Required nonempty Workbench files are missing.",
                    missing=missing,
                )
                discovery_ok = False
            else:
                check(
                    "workbench_assets",
                    "passed",
                    "Generic roles, commands, execution skill and their required supporting assets exist.",
                )
            if workspace.is_dir():
                check(
                    "workspace_directory", "passed", "Configured task workspace directory exists."
                )
            else:
                check(
                    "workspace_directory",
                    "warning",
                    "Task workspace directory is absent; initialize it before dispatch.",
                )
                execution_ok = False
        except _EXPECTED_ERRORS as error:
            check("workbench_assets", "blocked", _message(error))
            discovery_ok = False

    if not profiles:
        check(
            "discovery_profiles",
            "blocked",
            "At least one registered discovery profile is required.",
        )
        discovery_ok = False
    for name, profile in profiles.items():
        try:
            if not profile.owners or any(not owner.strip() for owner in profile.owners.values()):
                raise ValueError("Profile must configure nonempty owner routing identifiers")
            _owners(profile.owners)
            repo = Path(profile.repo_path)

            def git(*arguments, repo=repo):
                return (
                    read_git(repo, list(arguments), git_bin=options["git_bin"], max_bytes=16384)
                    .decode("utf-8")
                    .strip()
                )

            top = Path(git("rev-parse", "--show-toplevel")).resolve()
            if top != repo.resolve():
                raise ValueError("Profile repo_path must name the Git top-level directory")
            head = git("rev-parse", "--verify", "HEAD^{commit}")
            revision = git(
                "rev-parse", "--verify", "--end-of-options", profile.revision + "^{commit}"
            )
            dirty = bool(
                git("status", "--porcelain=v1", "--untracked-files=no", "--ignore-submodules=all")
            )
            check(
                "profile:" + name,
                "passed",
                "Git repository, HEAD, revision and owner routing are configured.",
                head=head,
                revision=revision,
            )
            check(
                "tracked_worktree:" + name,
                "warning" if dirty else "passed",
                "Tracked changes block execution preparation; discovery reads committed Git snapshots."
                if dirty
                else "Tracked worktree is clean; untracked files were not assessed.",
                dirty=dirty,
            )
            execution_ok = execution_ok and not dirty
            if revision != head:
                check(
                    "execution_revision:" + name,
                    "warning",
                    "Profile revision differs from target HEAD; history discovery is valid, but execution requires the candidate baseline checkout.",
                )
                execution_ok = False
            if profile.workspace:
                check(
                    "sources:" + name,
                    "passed",
                    "Configured source directory passed profile path validation.",
                )
            else:
                check(
                    "sources:" + name,
                    "not_configured",
                    "No workspace record source configured; Git history remains available.",
                )
        except _EXPECTED_ERRORS as error:
            check("profile:" + name, "blocked", _message(error))
            discovery_ok = False

    if options.get("production"):
        from .production import check_production

        production = check_production(options["production"])
        check(
            "production",
            "passed" if not production["missing"] else "blocked",
            "Optional production settings checked offline; no external services contacted.",
            **production,
        )

    try:
        count = _registry(options)
        check(
            "mcp_registry",
            "passed",
            "Listed the real Evolution registry and imported the unified API entrypoint; no transport was started.",
            evolution_tool_count=count,
            unified_transport_started=False,
        )
    except _EXPECTED_ERRORS as error:
        check("mcp_registry", "blocked", _message(error))
        discovery_ok = False

    try:
        status, message, details = _store_snapshot(root)
        check("workflow_store", status, message, **details)
        if status == "blocked":
            discovery_ok = False
        if status == "not_verified":
            execution_ok = False
    except _EXPECTED_ERRORS as error:
        check("workflow_store", "blocked", _message(error))
        discovery_ok = False

    try:
        available = importlib.util.find_spec("openpyxl") is not None
        check(
            "lmbench_dependency",
            "passed" if available else "not_configured",
            "openpyxl is available; raw measurement provenance remains unverified."
            if available
            else "Install the validation extra before converting lmbench workbooks.",
        )
    except _EXPECTED_ERRORS as error:
        check("lmbench_dependency", "not_verified", _message(error))
    artifacts = options.get("artifacts_root")
    artifacts_ready = artifacts is not None and Path(artifacts).is_dir()
    check(
        "lmbench_raw_evidence",
        "not_verified" if artifacts_ready else "not_configured",
        "No raw A/B measurements, frozen profile, hardware identity or benefit were verified.",
    )
    check(
        "opencode_binary",
        "passed" if shutil.which("opencode") else "not_configured",
        "Executable discovery only; no OpenCode session or MCP connection was attempted.",
    )
    check(
        "production_execution",
        "not_verified",
        "Model, device, owner approvals and production validation were not executed or verified.",
    )
    result["ready_for_discovery"] = discovery_ok
    result["execution_prerequisites"] = discovery_ok and execution_ok
    if result["environment_overrides"]:
        result["next_steps"].append(
            "Review the listed environment override names; they take precedence over the config file."
        )
    if not discovery_ok:
        result["next_steps"].append(
            "Resolve blocked prerequisite checks before starting discovery."
        )
    else:
        result["next_steps"].append(
            "Connect this configuration in OpenCode, then select one bounded pilot command: "
            + "; ".join(f"/evolve-discover {name}" for name in profiles)
        )
    if not execution_ok:
        result["next_steps"].append(
            "Prepare a clean target checkout and task workspace before dispatching implementation."
        )
    result["next_steps"].extend(
        [
            "Review mined patterns and candidates with their assigned owners; doctor grants no approval.",
            "Run the selected recipe and independent reviews, then collect real paired A/B evidence before claiming validation.",
        ]
    )
    return result
