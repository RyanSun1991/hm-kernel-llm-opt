"""FastMCP service wrapper for triggering kernel build workflows in Docker."""

from __future__ import annotations

import logging
import os
import shlex
import subprocess
import threading
import time
import uuid
from copy import deepcopy
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger(__name__)
DEFAULT_SERVER_NAME = "hmopt-build-mcp"

_TARGET_MAP = {
    "charlotte": "bootimage-charlotteoh",
    "nashville": "bootimage-nashvilleoh",
    "changsha": "bootimage-changshaoh",
}

_PROFILE_MAP = {
    "release": "release_aarch64le_defconfig",
    "debug": "aarch64le_defconfig",
}

_BUILD_TASKS: dict[str, dict[str, Any]] = {}
_BUILD_TASKS_LOCK = threading.Lock()


def _env(name: str, default: str = "") -> str:
    return os.getenv(name, default).strip()


def _resolve_target(device: str) -> str:
    normalized = device.strip().lower()
    if normalized in _TARGET_MAP:
        return _TARGET_MAP[normalized]
    if normalized.startswith("bootimage-"):
        return normalized
    raise ValueError(f"unsupported device/target: {device}")


def _resolve_defconfig(profile: str, defconfig: str | None = None) -> str:
    if defconfig and defconfig.strip():
        return defconfig.strip()

    normalized = profile.strip().lower()
    if normalized in _PROFILE_MAP:
        return _PROFILE_MAP[normalized]
    raise ValueError(f"unsupported profile: {profile}")


def _build_hione_command(
    *,
    project_path: str,
    profile: str,
    device: str,
    user: str,
    modem: str,
    target_perf: bool,
    toolchain: str,
    defconfig: str | None,
) -> str:
    target = _resolve_target(device)
    resolved_defconfig = _resolve_defconfig(profile, defconfig)
    resolved_profile = profile.strip().lower()

    return " ".join(
        [
            "kernel/hongmeng/build/hm_scripts/build.sh",
            f"--user={shlex.quote(user)}",
            f"-w={shlex.quote(project_path)}",
            f"--defconfig={shlex.quote(resolved_defconfig)}",
            f"--profile={shlex.quote(resolved_profile)}",
            f"--target={shlex.quote(target)}",
            f"--toolchain={shlex.quote(toolchain)}",
            f"--modem={shlex.quote(modem)}",
            f"--target_perf={str(target_perf).lower()}",
        ]
    )


def _run_cmd(argv: list[str], timeout_s: int) -> dict[str, Any]:
    started = time.time()
    proc = subprocess.run(
        argv,
        capture_output=True,
        text=True,
        timeout=timeout_s,
        check=False,
    )
    duration = round(time.time() - started, 3)
    return {
        "command": " ".join(shlex.quote(item) for item in argv),
        "returncode": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
        "duration_s": duration,
    }


def _container_is_running(docker_bin: str, container_name: str) -> bool:
    proc = subprocess.run(
        [docker_bin, "inspect", "-f", "{{.State.Running}}", container_name],
        capture_output=True,
        text=True,
        check=False,
    )
    return proc.returncode == 0 and proc.stdout.strip().lower() == "true"


def _image_exists(docker_bin: str, image: str) -> bool:
    proc = subprocess.run(
        [docker_bin, "image", "inspect", image],
        capture_output=True,
        text=True,
        check=False,
    )
    return proc.returncode == 0


def _resolve_project_mount(project_path: str) -> str:
    host_workspace = _env("HMOPT_BUILD_MCP_HOST_WORKDIR")
    container_workspace = _env("HMOPT_BUILD_MCP_CONTAINER_WORKDIR")
    code_subpath = _env("HMOPT_BUILD_MCP_CODE_SUBPATH")

    if host_workspace and container_workspace and code_subpath:
        return host_workspace

    explicit = _env("HMOPT_BUILD_MCP_PROJECT_MOUNT")
    if explicit:
        return explicit
    path = Path(project_path).expanduser()
    return str(path)


def _resolve_project_path(project_path: str | None = None) -> str:
    if project_path and project_path.strip():
        return project_path.strip()

    direct = _env("HMOPT_BUILD_MCP_PROJECT_PATH")
    if direct:
        return direct

    container_workspace = _env("HMOPT_BUILD_MCP_CONTAINER_WORKDIR")
    code_subpath = _env("HMOPT_BUILD_MCP_CODE_SUBPATH")
    if container_workspace and code_subpath:
        return str(Path(container_workspace) / code_subpath)

    raise ValueError(
        "project_path is required (or set HMOPT_BUILD_MCP_PROJECT_PATH, "
        "or set HMOPT_BUILD_MCP_CONTAINER_WORKDIR + HMOPT_BUILD_MCP_CODE_SUBPATH)"
    )


def _resolve_lz4_install_policy(mode: str) -> str:
    policy = _env("HMOPT_BUILD_MCP_INSTALL_LZ4", "auto").lower()
    if policy not in {"auto", "always", "never"}:
        raise ValueError("HMOPT_BUILD_MCP_INSTALL_LZ4 only supports auto/always/never")
    if policy == "always":
        return "(command -v lz4 >/dev/null 2>&1) || ((apt-get update && apt-get install -y lz4) || (sudo apt-get update && sudo apt-get install -y lz4))"
    if policy == "never":
        return ""
    if mode == "run":
        return "(command -v lz4 >/dev/null 2>&1) || ((apt-get update && apt-get install -y lz4) || (sudo apt-get update && sudo apt-get install -y lz4))"
    return ""


def _run_in_build_docker(inner_cmd: str, *, timeout_s: int, workdir: str | None = None) -> dict[str, Any]:
    docker_bin = _env("HMOPT_BUILD_MCP_DOCKER_BIN", "docker")
    mode = _env("HMOPT_BUILD_MCP_MODE", "exec").lower()
    runtime_workdir = (workdir or _env("HMOPT_BUILD_MCP_WORKDIR")).strip()

    if mode == "exec":
        container = _env("HMOPT_BUILD_MCP_RUNNER_CONTAINER")
        if not container:
            raise ValueError("HMOPT_BUILD_MCP_RUNNER_CONTAINER is required in exec mode")

        if not _container_is_running(docker_bin, container):
            if _image_exists(docker_bin, container):
                raise ValueError(
                    f"HMOPT_BUILD_MCP_RUNNER_CONTAINER={container!r} looks like an image, not a running container. "
                    "Use a running container name/id in exec mode, or switch to run mode with "
                    "HMOPT_BUILD_MCP_MODE=run and HMOPT_BUILD_MCP_RUNNER_IMAGE set."
                )
            raise ValueError(
                f"runner container {container!r} is not running. Start it first or switch to run mode."
            )

        cmd = [docker_bin, "exec"]
        if runtime_workdir:
            cmd.extend(["-w", runtime_workdir])
        cmd.extend([container, "bash", "-lc", inner_cmd])
        return _run_cmd(cmd, timeout_s)

    if mode == "run":
        image = _env("HMOPT_BUILD_MCP_RUNNER_IMAGE")
        if not image:
            raise ValueError("HMOPT_BUILD_MCP_RUNNER_IMAGE is required in run mode")

        container_project_path = _resolve_project_path()

        host_workspace = _env("HMOPT_BUILD_MCP_HOST_WORKDIR")
        container_workspace = _env("HMOPT_BUILD_MCP_CONTAINER_WORKDIR")
        code_subpath = _env("HMOPT_BUILD_MCP_CODE_SUBPATH")
        volume_target = container_project_path
        if host_workspace and container_workspace and code_subpath:
            container_project_path = str(Path(container_workspace) / code_subpath)
            volume_target = container_workspace

        project_mount = _resolve_project_mount(container_project_path)
        target_workdir = runtime_workdir or container_project_path

        cmd = [
            docker_bin,
            "run",
            "--rm",
            "--network=host",
            "-v",
            f"{project_mount}:{volume_target}:rw",
            "-w",
            target_workdir,
            image,
            "bash",
            "-lc",
            inner_cmd,
        ]
        return _run_cmd(cmd, timeout_s)

    raise ValueError("HMOPT_BUILD_MCP_MODE only supports exec/run")


def trigger_hione_build(
    device: str,
    profile: str = "release",
    *,
    project_path: str | None = None,
    user: str = "delivery",
    modem: str = "full",
    target_perf: bool = True,
    toolchain: str = "",
    defconfig: str | None = None,
    timeout_s: int = 7200,
) -> dict[str, Any]:
    resolved_project_path = _resolve_project_path(project_path)

    build_cmd = _build_hione_command(
        project_path=resolved_project_path,
        profile=profile,
        device=device,
        user=user,
        modem=modem,
        target_perf=target_perf,
        toolchain=toolchain,
        defconfig=defconfig,
    )

    mode = _env("HMOPT_BUILD_MCP_MODE", "exec").lower()
    lz4_prepare_cmd = _resolve_lz4_install_policy(mode)
    full_cmd = f"cd {shlex.quote(resolved_project_path)} && {build_cmd}"
    if lz4_prepare_cmd:
        full_cmd = f"cd {shlex.quote(resolved_project_path)} && {lz4_prepare_cmd} && {build_cmd}"

    return _run_in_build_docker(
        full_cmd,
        timeout_s=timeout_s,
        workdir=resolved_project_path,
    )


def trigger_hione_sign(device: str, *, timeout_s: int = 1800) -> dict[str, Any]:
    target = _resolve_target(device)
    sign_workspace = _env("HMOPT_BUILD_MCP_SIGN_WORKSPACE")
    if not sign_workspace:
        raise ValueError("HMOPT_BUILD_MCP_SIGN_WORKSPACE is required for signing")

    user_map = {
        "bootimage-charlotteoh": "aln_user",
        "bootimage-nashvilleoh": "nsv_user",
        "bootimage-changshaoh": "changsha_user",
    }
    sign_user = user_map[target]
    sign_cmd = f"cd {shlex.quote(sign_workspace)} && bash do_pack_hione_trunk.sh {sign_user} {target}"
    return _run_in_build_docker(sign_cmd, timeout_s=timeout_s, workdir=sign_workspace)


def _register_async_task(kind: str, payload: dict[str, Any]) -> str:
    task_id = str(uuid.uuid4())
    now = time.time()
    with _BUILD_TASKS_LOCK:
        _BUILD_TASKS[task_id] = {
            "task_id": task_id,
            "kind": kind,
            "status": "pending",
            "created_at": now,
            "updated_at": now,
            "payload": payload,
            "result": None,
            "error": None,
        }
    return task_id


def _set_task_running(task_id: str) -> None:
    with _BUILD_TASKS_LOCK:
        task = _BUILD_TASKS.get(task_id)
        if task is None:
            return
        task["status"] = "running"
        task["updated_at"] = time.time()


def _set_task_result(task_id: str, *, result: dict[str, Any] | None = None, error: str | None = None) -> None:
    with _BUILD_TASKS_LOCK:
        task = _BUILD_TASKS.get(task_id)
        if task is None:
            return
        task["status"] = "failed" if error else "succeeded"
        task["result"] = result
        task["error"] = error
        task["updated_at"] = time.time()


def _run_async_task(task_id: str, runner: Callable[[], dict[str, Any]]) -> None:
    _set_task_running(task_id)
    try:
        result = runner()
        _set_task_result(task_id, result=result)
    except Exception as exc:  # pylint: disable=broad-except
        logger.exception("build mcp async task failed: task_id=%s", task_id)
        _set_task_result(task_id, error=str(exc))


def _submit_async_task(kind: str, payload: dict[str, Any], runner: Callable[[], dict[str, Any]]) -> dict[str, Any]:
    task_id = _register_async_task(kind, payload)
    thread = threading.Thread(target=_run_async_task, args=(task_id, runner), daemon=True)
    thread.start()
    return {"task_id": task_id, "status": "pending", "kind": kind}


def trigger_hione_build_async(
    device: str,
    profile: str = "release",
    *,
    project_path: str | None = None,
    user: str = "delivery",
    modem: str = "full",
    target_perf: bool = True,
    toolchain: str = "",
    defconfig: str | None = None,
    timeout_s: int = 7200,
) -> dict[str, Any]:
    payload = {
        "device": device,
        "profile": profile,
        "project_path": project_path,
        "user": user,
        "modem": modem,
        "target_perf": target_perf,
        "toolchain": toolchain,
        "defconfig": defconfig,
        "timeout_s": timeout_s,
    }

    return _submit_async_task(
        "kernel_build_trigger",
        payload,
        lambda: trigger_hione_build(
            device=device,
            profile=profile,
            project_path=project_path,
            user=user,
            modem=modem,
            target_perf=target_perf,
            toolchain=toolchain,
            defconfig=defconfig,
            timeout_s=timeout_s,
        ),
    )


def get_build_task_status(task_id: str) -> dict[str, Any]:
    with _BUILD_TASKS_LOCK:
        task = _BUILD_TASKS.get(task_id)
        if task is None:
            raise ValueError(f"task not found: {task_id}")
        snapshot = deepcopy(task)

    snapshot["duration_s"] = round(snapshot["updated_at"] - snapshot["created_at"], 3)
    return snapshot


@lru_cache(maxsize=1)
def build_build_fastmcp_server() -> Any | None:
    try:
        from mcp.server.fastmcp import FastMCP  # type: ignore
    except ImportError:
        logger.warning(
            "mcp package is not installed; build MCP protocol endpoint is unavailable. "
            "Install with: pip install 'mcp[cli]'"
        )
        return None

    server_name = _env("HMOPT_BUILD_MCP_SERVER_NAME", DEFAULT_SERVER_NAME) or DEFAULT_SERVER_NAME
    mcp = FastMCP(server_name, stateless_http=True, json_response=True)

    @mcp.tool(
        name="kernel_build_trigger",
        description="Trigger HM kernel build inside another Docker container via docker exec/run.",
    )
    def mcp_kernel_build_trigger(
        device: str,
        profile: str = "release",
        project_path: str | None = None,
        user: str = "delivery",
        modem: str = "full",
        target_perf: bool = True,
        toolchain: str = "",
        defconfig: str | None = None,
        timeout_s: int = 7200,
    ) -> dict[str, Any]:
        return trigger_hione_build(
            device=device,
            profile=profile,
            project_path=project_path,
            user=user,
            modem=modem,
            target_perf=target_perf,
            toolchain=toolchain,
            defconfig=defconfig,
            timeout_s=timeout_s,
        )

    @mcp.tool(
        name="kernel_build_trigger_async",
        description="Submit kernel build async task and return task_id immediately.",
    )
    def mcp_kernel_build_trigger_async(
        device: str,
        profile: str = "release",
        project_path: str | None = None,
        user: str = "delivery",
        modem: str = "full",
        target_perf: bool = True,
        toolchain: str = "",
        defconfig: str | None = None,
        timeout_s: int = 7200,
    ) -> dict[str, Any]:
        return trigger_hione_build_async(
            device=device,
            profile=profile,
            project_path=project_path,
            user=user,
            modem=modem,
            target_perf=target_perf,
            toolchain=toolchain,
            defconfig=defconfig,
            timeout_s=timeout_s,
        )

    @mcp.tool(
        name="kernel_build_status",
        description="Query async build task status by task_id.",
    )
    def mcp_kernel_build_status(task_id: str) -> dict[str, Any]:
        return get_build_task_status(task_id=task_id)

    @mcp.tool(
        name="kernel_sign_trigger",
        description="Trigger hm-CI do_pack_hione_trunk.sh signing inside build Docker container.",
    )
    def mcp_kernel_sign_trigger(device: str, timeout_s: int = 1800) -> dict[str, Any]:
        return trigger_hione_sign(device=device, timeout_s=timeout_s)

    return mcp
