"""Prepare a small local pilot configuration without starting any workflow."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import sys
import tempfile
from pathlib import Path

from .mining import read_git

_ROLES = (
    "assistant",
    "researcher",
    "architect",
    "implementer",
    "reviewer",
    "validator",
    "coordinator",
)
_ASSETS = (
    *(f"agents/{role}.md" for role in _ROLES),
    "commands/evolve-discover.md",
    "commands/evolve-candidate.md",
    "commands/evolve-research.md",
    "commands/evolve-queue.md",
    "commands/evolve-batch.md",
    "commands/evolve-production.md",
    "commands/evolve-workspace.md",
    "config.yaml",
    "skills/_registry.yaml",
    "skills/infra/agent-core/SKILL.md",
    "skills/infra/language-config/SKILL.md",
    "skills/infra/pipeline/evolution-execution/SKILL.md",
    "skills/infra/pipeline/evolution-mining/SKILL.md",
    "skills/infra/pipeline/pattern-synthesis/SKILL.md",
    "skills/infra/pipeline/candidate-assessment/SKILL.md",
    "skills/infra/pipeline/evolution-batch/SKILL.md",
    *(
        f"skills/role/{skill}/SKILL.md"
        for skill in (
            "research-discipline",
            "plan-funnel",
            "review-checklists",
            "implementation-guardrails",
            "validation-flight-check",
        )
    ),
)


def _executable(value: str, name: str) -> str:
    if not isinstance(value, str) or not value.strip() or "\x00" in value:
        raise ValueError(f"{name} must name an executable; supply its full filesystem path")
    located = shutil.which(value)
    if located is None and Path(value).expanduser().is_file():
        located = str(Path(value).expanduser())
    if located is None:
        raise ValueError(
            f"Cannot find {name}: {value}. Install it or supply its full executable path"
        )
    # Preserve the launcher path: POSIX venv/bin/python is commonly a symlink.
    # Resolving it to the system interpreter drops the environment's installed hmopt.
    return str(Path(located).absolute())


def _workbench(path: Path) -> Path:
    root = Path(path).expanduser().resolve()
    if not root.is_dir():
        raise ValueError(
            "Workbench must be an existing HMOPT checkout; pass --workbench <checkout>"
        )
    missing = []
    for asset in _ASSETS:
        item = root / ".opencode" / asset
        if not item.is_file() or not item.read_bytes().strip():
            missing.append(f".opencode/{asset}")
    if missing:
        raise ValueError(
            "Workbench is missing installed Evolution roles/commands/skills: "
            + ", ".join(missing)
            + ". Use the HMOPT opencode checkout containing this feature; setup does not install roles"
        )
    return root


def _local_path(path: Path) -> None:
    # Reject redirected .opencode/local directories (including Windows junctions)
    # so setup cannot silently place operator configuration in another checkout.
    if path.resolve() != path or path.is_symlink():
        raise ValueError(f"Local setup path is redirected: {path}. Use a normal local directory")
    if path.exists() and not path.is_dir():
        raise ValueError(f"Local setup path must be a directory: {path}. Review the existing file")


def _matches_existing(path: Path, expected: dict) -> bool:
    if not path.exists() and not path.is_symlink():
        return False
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 1_000_000:
        raise ValueError(f"Existing setup artifact is not a regular bounded JSON file: {path}")
    try:
        actual = json.loads(path.read_text(encoding="utf-8-sig"))
        actual_json = json.dumps(actual, sort_keys=True, allow_nan=False)
    except (UnicodeError, ValueError) as exc:
        raise ValueError(
            f"Existing setup JSON is invalid: {path}. Review or back it up before configuring again"
        ) from exc
    # JSON booleans/numbers must not compare equal through Python's True == 1.
    if actual_json != json.dumps(expected, sort_keys=True, allow_nan=False):
        raise ValueError(
            f"Existing setup configuration differs: {path}. Refusing to overwrite; "
            "review the existing configuration or back it up before configuring another pilot"
        )
    return True


def _publish_json(path: Path, value: dict) -> bool:
    if _matches_existing(path, value):
        return False
    raw = (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=".evolution-setup-", dir=path.parent, delete=False
        ) as f:
            temporary = Path(f.name)
            f.write(raw)
            f.flush()
            os.fsync(f.fileno())
        try:
            # Same-directory hard links publish all bytes atomically without replacing
            # an existing configuration, including when another setup won the race.
            os.link(temporary, path)
        except FileExistsError:
            _matches_existing(path, value)
            return False
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return True


def _mcp_fragment(python_executable: str, environment: dict[str, str]) -> dict:
    """One connection shape for shared YAML and legacy JSON setup."""
    return {
        "mcp": {
            "hmopt_kernel_index": {
                "type": "local",
                "command": [python_executable, "-m", "hmopt.api.mcp_stdio"],
                "environment": environment,
                # Client wait covers a scheduling window plus a bounded Git call.
                "timeout": 180000,
                "enabled": True,
            }
        }
    }


def setup_workspace(
    workbench: Path,
    source_root: Path | None,
    workspace_id: str,
    projects: dict[str, str | None],
    owner: str,
    *,
    git_bin="git",
    python_executable=sys.executable,
    platform_config: Path | None = None,
) -> dict:
    from .workspace import WorkspaceConfig, freeze_workspace

    workbench = _workbench(workbench)
    from .platform_setup import find_platform, setup_platform

    platform = find_platform(workbench, platform_config)
    if platform is not None:
        return setup_platform(
            workbench,
            platform,
            source_root,
            workspace_id,
            projects,
            owner,
            git_bin=git_bin,
            python_executable=python_executable,
        )
    from .configuration import source_root as inherited_source_root

    source_root = Path(inherited_source_root(workbench, source_root))
    projects = dict(projects)
    if "kernel" in projects and projects["kernel"] is None:
        from .configuration import kernel_path

        try:
            projects["kernel"] = (
                Path(kernel_path({}, workbench)).relative_to(source_root).as_posix()
            )
        except ValueError:
            raise ValueError(
                "Configure KERNEL_REPO_PATH inside the business workspace or use --project kernel=relative/path"
            ) from None
    git_bin = _executable(git_bin, "Git")
    python_executable = _executable(python_executable, "Python")
    workspace = WorkspaceConfig.model_validate(
        {
            "workspace_id": workspace_id,
            "root": str(source_root.expanduser().resolve()),
            "projects": {
                name: {"path": path, "owners": {"**": owner}} for name, path in projects.items()
            },
            "selected": list(projects),
        }
    ).model_dump(mode="json")
    manifest = freeze_workspace(workspace, git_bin=git_bin)
    directory = workbench / ".opencode/local/evolution" / workspace_id
    workspace_root = workbench / ".opencode/local/workspaces"
    for path in (directory, workspace_root, directory / "state", directory / "artifacts"):
        _local_path(path)
    directory.mkdir(parents=True, exist_ok=True)
    workspace_root.mkdir(parents=True, exist_ok=True)
    (directory / "artifacts").mkdir(exist_ok=True)
    config_path = directory / "config.json"
    config = {
        "schema_version": 1,
        "root": str(directory / "state"),
        "git_bin": git_bin,
        "workspace_root": str(workspace_root),
        "artifacts_root": str(directory / "artifacts"),
        "source_workspace": workspace,
    }
    fragment = _mcp_fragment(python_executable, {"HMOPT_EVOLUTION_CONFIG": str(config_path)})
    fragment_path = directory / "opencode.fragment.json"
    _matches_existing(config_path, config)
    _matches_existing(fragment_path, fragment)
    _publish_json(config_path, config)
    _publish_json(fragment_path, fragment)
    return {
        "config_path": str(config_path),
        "fragment_path": str(fragment_path),
        "manifest": manifest,
        "next_steps": [
            "Merge the Evolution environment setting into the existing main MCP. For a remote main MCP, configure the server-side environment and visible paths; do not add a second local connection.",
            "Configure production.worker with the existing OpenCode server/provider/model.",
            f'python -m hmopt evolve --config "{config_path}" doctor',
            f'python -m hmopt evolve --config "{config_path}" serve',
            "In OpenCode: /evolve-workspace start. Use returned run IDs to inspect or resume.",
        ],
    }


def setup(
    workbench: Path,
    repo: Path,
    owner: str,
    *,
    git_bin: str = "git",
    python_executable: str = sys.executable,
    platform_config: Path | None = None,
) -> dict:
    """Create operator config and a mergeable MCP fragment, never run discovery or agents."""
    if not isinstance(owner, str) or not owner.strip() or len(owner.strip()) > 200:
        raise ValueError("Owner must be a nonempty responsibility label (at most 200 characters)")
    owner = owner.strip()
    if any(ord(character) < 32 or ord(character) == 127 for character in owner):
        raise ValueError("Owner must be a single responsibility label without control characters")
    workbench = _workbench(workbench)
    from .platform_setup import find_platform, setup_platform, single_project_inputs

    platform = find_platform(workbench, platform_config)
    if platform is not None:
        root, projects = single_project_inputs(platform, repo)
        return setup_platform(
            workbench,
            platform,
            root,
            "business",
            projects,
            owner,
            git_bin=git_bin,
            python_executable=python_executable,
        )
    if repo is None:
        from .configuration import kernel_path

        repo = Path(kernel_path({}, workbench))
    repository = Path(repo).expanduser().resolve()
    if not repository.is_dir():
        raise ValueError("Target repository must exist; pass --repo <existing Git checkout>")
    git_bin = _executable(git_bin, "Git")
    python_executable = _executable(python_executable, "Python")
    try:
        top = read_git(
            repository, ["rev-parse", "--show-toplevel"], git_bin=git_bin, max_bytes=16384
        )
        repository = Path(top.decode("utf-8").strip()).resolve()
        head = (
            read_git(
                repository,
                ["rev-parse", "--verify", "HEAD^{commit}"],
                git_bin=git_bin,
                max_bytes=128,
            )
            .decode("ascii")
            .strip()
        )
    except (ValueError, OSError, UnicodeError, TimeoutError) as exc:
        raise ValueError(
            "Cannot resolve the target Git checkout and HEAD commit. "
            "Use a working checkout with at least one commit and verify --git-bin"
        ) from exc
    if not re.fullmatch(r"[0-9a-f]{40,64}", head):
        raise ValueError("Git HEAD did not resolve to a commit; inspect the target checkout")
    directory = workbench / ".opencode" / "local" / "evolution"
    state_root = directory / "state"
    workspace_root = workbench / ".opencode" / "local" / "workspaces"
    artifacts_root = directory / "artifacts"
    for path in (directory, state_root, workspace_root, artifacts_root):
        _local_path(path)
    repo_id = (
        "local-" + hashlib.sha256(os.path.normcase(str(repository)).encode("utf-8")).hexdigest()
    )
    config = {
        "schema_version": 1,
        "root": str(state_root),
        "git_bin": git_bin,
        "workspace_root": str(workspace_root),
        "artifacts_root": str(artifacts_root),
        "discovery_profiles": {
            "pilot": {
                "repo_path": str(repository),
                "repo_id": repo_id,
                "revision": "HEAD",
                "workspace": str(workbench),
                "owners": {"**": owner},
                "max_pages": 1,
                "page_size": 50,
                "source_page_size": 50,
                "scheduling_budget_s": 30,
            }
        },
    }
    config_path = directory / "config.json"
    fragment_path = directory / "opencode.fragment.json"
    fragment = _mcp_fragment(python_executable, {"HMOPT_EVOLUTION_CONFIG": str(config_path)})
    # Check both artifacts before creating either, including reruns with a changed
    # Python executable or owner. A failed second publication is repairable on rerun.
    _matches_existing(config_path, config)
    _matches_existing(fragment_path, fragment)
    for path in (directory, workspace_root, artifacts_root):
        path.mkdir(parents=True, exist_ok=True)
        _local_path(path)
    config_created = _publish_json(config_path, config)
    fragment_created = _publish_json(fragment_path, fragment)
    return {
        "status": "setup_ready",
        "config_path": str(config_path),
        "fragment_path": str(fragment_path),
        "workbench": str(workbench),
        "repo_path": str(repository),
        "profile": "pilot",
        "target_revision": head,
        "owner": owner,
        "config_created": config_created,
        "fragment_created": fragment_created,
        "workflow_started": False,
        "next_steps": [
            (
                "连接现有 HMOPT 主 MCP：本地模式将片段合并到现有连接，保留原有配置；"
                "仅在尚无主 MCP 连接时添加整个片段。已有 remote 主 MCP 时，只在服务器或容器设置 "
                "HMOPT_EVOLUTION_CONFIG 为服务器可见的配置路径并重启原 MCP，不新增并行本地连接；"
                "配置里的仓库、工作台和数据路径也必须在服务端可见。"
            ),
            (
                "重新加载 OpenCode 的原主 MCP，调用 evolution_discovery_profiles()，"
                "确认 pilot 的仓库、责任人和目录与本次配置一致；setup_ready 仅表示本地配置已生成。"
            ),
            (
                "在工作台执行 /evolve-discover pilot，运行一个有边界的挖掘批次并查看实际证据。"
                "草稿 pattern 需人工激活，候选需责任人确认；需要续跑时使用返回的 batch-id，"
                "确认后的候选可执行 /evolve-candidate <candidate-id> full 或指定步骤。"
            ),
        ],
    }
