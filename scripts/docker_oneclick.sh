#!/usr/bin/env bash
set -euo pipefail

ACTION="${1:-help}"
ACTION_ARGS=("${@:2}")
CONFIG_PATH="${HMOPT_DOCKER_CONFIG:-configs/app.yaml}"
ENV_FILE=".env.docker"
HMOPT_CONTAINER="hmopt-app"
HMOPT_IMAGE="${HMOPT_IMAGE:-hmopt:local}"
HMOPT_BASE_IMAGE="${HMOPT_BASE_IMAGE:-kernel.dockerhub.rnd.huawei.com/hmci-docker-image:v3-7.0}"
HMOPT_BASE_IMAGE_CANDIDATES="${HMOPT_BASE_IMAGE_CANDIDATES:-}"
PIP_INDEX_URL="${PIP_INDEX_URL:-https://mirrors.tools.huawei.com/pypi/simple}"
PIP_TRUSTED_HOST="${PIP_TRUSTED_HOST:-mirrors.tools.huawei.com}"
REGISTRY_HOST="${REGISTRY_HOST:-kernel.dockerhub.rnd.huawei.com}"
IMAGE_BUNDLE_TAR="${IMAGE_BUNDLE_TAR:-hmopt_bundle.tar.gz}"
RUNTIME_IDENTITY_ARGS=()
HOST_EXEC_IDENTITY_ARGS=()

usage() {
  cat <<USAGE
Usage / 用法: $0 <action>

Actions / 动作:
  up              Build image and start hmopt container (with embedded neo4j) / 构建并启动单容器（内置neo4j）
  up-prebuilt     Start with preloaded image (no local build) / 使用预置镜像启动（不本地构建）
  index           Build kernel index / 执行内核索引
  mcp             Start MCP server (port 7331) / 启动 MCP 服务（7331）
  seq-mcp         Start sequential thinking MCP server (host 7334 -> container 7333) / 启动顺序思考 MCP 服务（宿主7334->容器7333）
  git-mcp         Start Git MCP server (host port 7334 -> container 7334) / 启动 Git MCP 服务（宿主7334->容器7334）
  build-mcp       Start Build MCP server (host 7335 -> container 7335) / 启动 Build MCP 服务（宿主7335->容器7335）
  oneclick        Start MCP + sequential MCP servers in background / 一键后台启动 MCP+顺序思考 MCP 服务
  api             Start REST API (port 8000) / 启动 REST API（8000）
  clone           Clone kernel repo to KERNEL_REPO_PATH / 克隆代码到 KERNEL_REPO_PATH
  insecure-registry  Enable docker insecure-registry bypass / 配置 docker 跳过证书校验
  package-images  Export deliverable image bundle / 导出可交付镜像包
  load-images     Import image bundle on target machine / 在目标机器导入镜像包
  doctor          Check registry and image readiness / 检查镜像仓库与镜像可用性
  down            Stop and remove container / 停止并删除容器
  logs            Tail logs / 查看日志
  shell           Enter hmopt container shell / 进入 hmopt 容器
USAGE
}

require_env_file() {
  if [[ ! -f "$ENV_FILE" ]]; then
    echo "Missing $ENV_FILE. Please run: cp .env.docker.example .env.docker"
    exit 1
  fi
}

load_env() {
  require_env_file
  set -a
  # shellcheck source=/dev/null
  source "$ENV_FILE"
  set +a
}

has_compose() {
  docker compose version >/dev/null 2>&1
}

compose_run() {
  docker compose --env-file "$ENV_FILE" "$@"
}

ensure_image_available() {
  local image="$1"
  docker image inspect "$image" >/dev/null 2>&1 || docker pull "$image"
}

docker_build_once() {
  local base_image="$1"
  local use_buildkit="$2"

  DOCKER_BUILDKIT="$use_buildkit" docker build -t "$HMOPT_IMAGE" -f Dockerfile \
    --build-arg BASE_IMAGE="$base_image" \
    --build-arg PIP_INDEX_URL="$PIP_INDEX_URL" \
    --build-arg PIP_TRUSTED_HOST="$PIP_TRUSTED_HOST" \
    .
}

docker_build_with_fallback() {
  local base_image="$1"
  docker_build_once "$base_image" 1 || {
    echo "[build] BuildKit/buildx unavailable, fallback to classic docker builder"
    docker_build_once "$base_image" 0
  }
}

build_hmopt_image() {
  local requested_base_image="$1"
  local candidates_raw="$requested_base_image"
  local candidate
  local tried=""

  [[ -n "$HMOPT_BASE_IMAGE_CANDIDATES" ]] && candidates_raw="$candidates_raw,$HMOPT_BASE_IMAGE_CANDIDATES"
  candidates_raw="$candidates_raw"

  IFS=',' read -r -a _candidates <<< "$candidates_raw"
  for candidate in "${_candidates[@]}"; do
    candidate="$(echo "$candidate" | xargs)"
    [[ -z "$candidate" ]] && continue
    [[ ",$tried," == *",$candidate,"* ]] && continue
    tried="$tried,$candidate"

    echo "[build] trying BASE_IMAGE: $candidate"
    if docker_build_with_fallback "$candidate"; then
      return 0
    fi
  done

  echo "[build] all candidate BASE_IMAGE values failed: ${tried#,}"
  return 1
}

doctor() {
  load_env
  echo "[doctor] checking base image: $HMOPT_BASE_IMAGE"
  ensure_image_available "$HMOPT_BASE_IMAGE"
  echo "[doctor] OK"
}


resolve_runtime_identity_args() {
  RUNTIME_IDENTITY_ARGS=()
  local run_as_host_user="${HMOPT_RUN_AS_HOST_USER:-1}"
  local host_uid="${HMOPT_HOST_UID:-}"
  local host_gid="${HMOPT_HOST_GID:-}"
  local start_neo4j="${HMOPT_START_NEO4J:-1}"
  local force_host_user_with_neo4j="${HMOPT_FORCE_HOST_USER_WITH_NEO4J:-0}"

  if [[ "$run_as_host_user" != "1" ]]; then
    return 0
  fi

  if [[ "$start_neo4j" == "1" && "$force_host_user_with_neo4j" != "1" ]]; then
    echo "[docker] HMOPT_START_NEO4J=1 detected; fallback to root runtime for neo4j compatibility."
    echo "[docker] Set HMOPT_FORCE_HOST_USER_WITH_NEO4J=1 if you want to force host UID/GID mode."
    return 0
  fi

  if [[ -z "$host_uid" ]]; then
    host_uid="$(id -u)"
  fi
  if [[ -z "$host_gid" ]]; then
    host_gid="$(id -g)"
  fi

  RUNTIME_IDENTITY_ARGS=(
    --user "${host_uid}:${host_gid}"
    -e "HOME=/tmp/hmopt-home"
    -e "USER=${USER:-hmopt}"
    -e "LOGNAME=${USER:-hmopt}"
    -v /etc/passwd:/etc/passwd:ro
    -v /etc/group:/etc/group:ro
  )
}

resolve_host_exec_identity_args() {
  HOST_EXEC_IDENTITY_ARGS=()
  local run_as_host_user="${HMOPT_RUN_AS_HOST_USER:-1}"
  local host_uid="${HMOPT_HOST_UID:-}"
  local host_gid="${HMOPT_HOST_GID:-}"

  if [[ "$run_as_host_user" != "1" ]]; then
    return 0
  fi

  if [[ -z "$host_uid" ]]; then
    host_uid="$(id -u)"
  fi
  if [[ -z "$host_gid" ]]; then
    host_gid="$(id -g)"
  fi

  HOST_EXEC_IDENTITY_ARGS=(
    --user "${host_uid}:${host_gid}"
    -e "HOME=/tmp/hmopt-home"
    -e "USER=${USER:-hmopt}"
    -e "LOGNAME=${USER:-hmopt}"
  )
}

configure_git_safe_directories() {
  local -a safe_dirs=()
  local raw_extra="${HMOPT_GIT_SAFE_DIRECTORIES:-}"
  local entry

  resolve_host_exec_identity_args

  [[ -n "${PROJECT_REPO_PATH:-}" ]] && safe_dirs+=("${PROJECT_REPO_PATH}")
  [[ -n "${KERNEL_REPO_PATH:-}" ]] && safe_dirs+=("${KERNEL_REPO_PATH}")
  safe_dirs+=("/workspace/project" "/workspace/kernel")

  if [[ -n "$raw_extra" ]]; then
    IFS=',' read -r -a _extra <<< "$raw_extra"
    for entry in "${_extra[@]}"; do
      entry="$(echo "$entry" | xargs)"
      [[ -n "$entry" ]] && safe_dirs+=("$entry")
    done
  fi

  for entry in "${safe_dirs[@]}"; do
    docker exec "$HMOPT_CONTAINER" bash -lc "if [ -d '$entry' ]; then git config --global --add safe.directory '$entry'; fi" >/dev/null
    if [[ ${#HOST_EXEC_IDENTITY_ARGS[@]} -gt 0 ]]; then
      docker exec "${HOST_EXEC_IDENTITY_ARGS[@]}" "$HMOPT_CONTAINER" bash -lc "if [ -d '$entry' ]; then git config --global --add safe.directory '$entry'; fi" >/dev/null || true
    fi
  done
}

run_hmopt_container() {
  local project_repo_path="${PROJECT_REPO_PATH:-}"
  local kernel_repo_path="${KERNEL_REPO_PATH:-$(pwd)/data/sample-kernel}"
  local inherit_host_gitconfig="${HMOPT_INHERIT_HOST_GITCONFIG:-1}"
  local host_gitconfig_source="${HMOPT_GITCONFIG_SOURCE:-$HOME/.gitconfig}"
  local -a volume_args

  resolve_runtime_identity_args

  volume_args=(
    -v "$(pwd):/app"
    -v "$(pwd)/data/neo4j/data:/var/lib/neo4j"
    -v "$(pwd)/data/neo4j/logs:/var/log/neo4j"
    -v "$(pwd)/data/neo4j/plugins:/var/lib/neo4j/plugins"
    -v /var/run/docker.sock:/var/run/docker.sock
    -v /usr/bin/docker:/usr/bin/docker:ro
  )

  if [[ -n "$project_repo_path" ]]; then
    # Mount project root as-is so all sub paths (including kernel and parent .git metadata) are available.
    volume_args+=(
      -v "${project_repo_path}:${project_repo_path}:rw"
      -v "${project_repo_path}:/workspace/project:rw"
    )
  else
    # Backward-compatible fallback when only kernel repo path is provided.
    volume_args+=(
      -v "${kernel_repo_path}:/workspace/kernel:rw"
      -v "${kernel_repo_path}:${KERNEL_REPO_PATH:-/workspace/kernel}:rw"
    )
  fi
  if [[ "$inherit_host_gitconfig" == "1" ]]; then
    if [[ -f "$host_gitconfig_source" ]]; then
      # Reuse host git global config so commit name/email are available in container.
      volume_args+=(
        -v "${host_gitconfig_source}:/root/.gitconfig:ro"
        -v "${host_gitconfig_source}:/tmp/hmopt-home/.gitconfig:ro"
      )
    else
      echo "[docker] HMOPT_INHERIT_HOST_GITCONFIG=1 but gitconfig not found: $host_gitconfig_source"
      echo "[docker] set HMOPT_GITCONFIG_SOURCE to your host gitconfig path if needed."
    fi
  fi

  docker rm -f "$HMOPT_CONTAINER" >/dev/null 2>&1 || true
  docker run -d \
    --name "$HMOPT_CONTAINER" \
    --env-file .env.docker \
    -p 7475:7474 -p 7688:7687 -p 7330:7330 -p 7332:7331 -p 7333:7333 -p 7334:7334 -p 7335:7335  -p 8001:8000 \
    -e HMOPT_LLM_BASE_URL="${HMOPT_LLM_BASE_URL:-http://host.docker.internal:20010/v1}" \
    -e HMOPT_LLM_API_KEY="${HMOPT_LLM_API_KEY:-}" \
    -e HMOPT_MCP_SERVER_API_KEY="${HMOPT_MCP_SERVER_API_KEY:-}" \
    -e HMOPT_MCP_HOST='0.0.0.0' \
    -e HMOPT_MCP_PORT='7331' \
    -e HMOPT_SEQ_MCP_HOST='0.0.0.0' \
    -e HMOPT_SEQ_MCP_PORT='7333' \
    -e HMOPT_START_NEO4J="${HMOPT_START_NEO4J:-1}" \
    -e NEO4J_USER="${NEO4J_USER:-neo4j}" \
    -e NEO4J_PASSWORD="${NEO4J_PASSWORD:-@huawei2026}" \
    -w /app \
    "${RUNTIME_IDENTITY_ARGS[@]}" \
    "${volume_args[@]}" \
    "$HMOPT_IMAGE" \
    bash -lc "tail -f /dev/null" >/dev/null

  docker exec "$HMOPT_CONTAINER" bash -lc 'mkdir -p "${HOME:-/tmp/hmopt-home}"' >/dev/null
  resolve_host_exec_identity_args
  if [[ ${#HOST_EXEC_IDENTITY_ARGS[@]} -gt 0 ]]; then
    docker exec "${HOST_EXEC_IDENTITY_ARGS[@]}" "$HMOPT_CONTAINER" bash -lc 'mkdir -p "${HOME:-/tmp/hmopt-home}"' >/dev/null || true
  fi
  configure_git_safe_directories
}

up_docker_native() {
#  prepare_neo4j_offline
  load_env
  if ! build_hmopt_image "$HMOPT_BASE_IMAGE"; then
    echo "[build] failed. Please set HMOPT_BASE_IMAGE / HMOPT_BASE_IMAGE_CANDIDATES in .env.docker"
    exit 1
  fi
  run_hmopt_container
  echo "Started hmopt single-container mode (neo4j embedded)."
}

up_prebuilt() {
  load_env
  ensure_image_available "$HMOPT_IMAGE"
  run_hmopt_container
  echo "Started with prebuilt hmopt image (neo4j embedded)."
}


build_index_args() {
  local args=(--config "$CONFIG_PATH")
  local i=0

  if [[ -n "${INDEX_REPO_PATH:-}" ]]; then
    args+=(--repo-path "$(map_host_path_to_container "$INDEX_REPO_PATH")")
  fi
  if [[ -n "${INDEX_COMPILE_COMMANDS_DIR:-}" ]]; then
    args+=(--compile-commands-dir "$(map_host_path_to_container "$INDEX_COMPILE_COMMANDS_DIR")")
  fi

  while [[ $i -lt ${#ACTION_ARGS[@]} ]]; do
    case "${ACTION_ARGS[$i]}" in
      --repo-path|--compile-commands-dir)
        if [[ $((i+1)) -ge ${#ACTION_ARGS[@]} ]]; then
          echo "missing value for ${ACTION_ARGS[$i]}" >&2
          exit 1
        fi
        args+=("${ACTION_ARGS[$i]}" "$(map_host_path_to_container "${ACTION_ARGS[$((i+1))]}")")
        i=$((i+2))
        ;;
      *)
        args+=("${ACTION_ARGS[$i]}")
        i=$((i+1))
        ;;
    esac
  done

  printf "%s\n" "${args[@]}"
}

map_host_path_to_container() {
  local raw_path="$1"
  local pwd_path
  pwd_path="$(pwd)"

  if [[ -n "${PROJECT_REPO_PATH:-}" && "$raw_path" == "$PROJECT_REPO_PATH"* ]]; then
    printf "%s\n" "$raw_path"
    return 0
  fi

  if [[ "$raw_path" == /workspace/* || "$raw_path" == /app/* ]]; then
    printf "%s\n" "$raw_path"
    return 0
  fi

  if [[ -n "${KERNEL_REPO_PATH:-}" && "$raw_path" == "$KERNEL_REPO_PATH"* ]]; then
    if [[ -n "${PROJECT_REPO_PATH:-}" ]]; then
      printf "%s\n" "$raw_path"
    else
      printf "/workspace/kernel/%s\n" "${raw_path#${KERNEL_REPO_PATH}}"
    fi
    return 0
  fi

  if [[ "$raw_path" == "$pwd_path"* ]]; then
    printf "/app%s\n" "${raw_path#${pwd_path}}"
    return 0
  fi

  printf "%s\n" "$raw_path"
}

index_docker_native() {
  mapfile -t _index_args < <(build_index_args)
  docker exec "$HMOPT_CONTAINER" python -m hmopt.cli index-kernel "${_index_args[@]}"
}


mcp_docker_native() { docker exec "$HMOPT_CONTAINER" bash -lc 'python -m hmopt.cli serve-mcp --host 0.0.0.0 --port 7331'; }
seq_mcp_docker_native() { docker exec "$HMOPT_CONTAINER" bash -lc 'bash scripts/run_seq_mcp_server.sh'; }
oneclick_docker_native() {
  docker exec -d "$HMOPT_CONTAINER" bash -lc 'nohup bash scripts/run_all_mcp_servers.sh >/tmp/all_mcp_servers.log 2>&1 &'
  echo 'Started MCP (7331) and sequential thinking MCP (7333) in background.'
}
api_docker_native() { docker exec "$HMOPT_CONTAINER" bash -lc 'uvicorn hmopt.api.main:app --host 0.0.0.0 --port 8000'; }
git_mcp_docker_native() {
  resolve_host_exec_identity_args
  if [[ ${#HOST_EXEC_IDENTITY_ARGS[@]} -gt 0 ]]; then
    docker exec "${HOST_EXEC_IDENTITY_ARGS[@]}" "$HMOPT_CONTAINER" bash -lc 'bash scripts/run_git_mcp_server.sh'
  else
    docker exec "$HMOPT_CONTAINER" bash -lc 'bash scripts/run_git_mcp_server.sh'
  fi
}
build_mcp_docker_native() { docker exec "$HMOPT_CONTAINER" bash -lc 'bash scripts/run_build_mcp_server.sh'; }
down_docker_native() { docker rm -f "$HMOPT_CONTAINER" >/dev/null 2>&1 || true; }
logs_docker_native() { docker logs -f "$HMOPT_CONTAINER"; }
shell_docker_native() { docker exec -it "$HMOPT_CONTAINER" bash; }

package_images() {
  load_env
  mkdir -p "$(dirname "$IMAGE_BUNDLE_TAR")"
  if ! docker image inspect "$HMOPT_IMAGE" >/dev/null 2>&1; then
    echo "[package-images] HMOPT image not found locally: $HMOPT_IMAGE"
    echo "[package-images] building it first..."
    build_hmopt_image "$HMOPT_BASE_IMAGE" || { echo "[package-images] build failed"; exit 1; }
  fi
  local tmp_tar="${IMAGE_BUNDLE_TAR%.gz}"
  docker save "$HMOPT_IMAGE" -o "$tmp_tar"
  gzip -f "$tmp_tar"
  echo "[package-images] exported: $IMAGE_BUNDLE_TAR"
}

load_images() {
  load_env
  [[ -f "$IMAGE_BUNDLE_TAR" ]] || { echo "[load-images] bundle not found: $IMAGE_BUNDLE_TAR"; exit 1; }
  gunzip -c "$IMAGE_BUNDLE_TAR" | docker load
  echo "[load-images] import complete"
}

enable_insecure_registry() {
  load_env
  command -v sudo >/dev/null 2>&1 || { echo "sudo is required for insecure-registry action"; exit 1; }
  echo "[insecure-registry] target registry: ${REGISTRY_HOST}"
  sudo python3 - <<PY
import json
from pathlib import Path
registry = "${REGISTRY_HOST}"
path = Path('/etc/docker/daemon.json')
if path.exists():
    try:
        data = json.loads(path.read_text(encoding='utf-8'))
    except Exception:
        data = {}
else:
    data = {}
arr = data.get('insecure-registries', [])
if not isinstance(arr, list):
    arr = []
if registry not in arr:
    arr.append(registry)
data['insecure-registries'] = arr
path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding='utf-8')
print('updated', path)
PY
  if command -v systemctl >/dev/null 2>&1; then sudo systemctl restart docker; else sudo service docker restart; fi
  echo "[insecure-registry] done. Docker will skip TLS verify for ${REGISTRY_HOST}."
}


prepare_neo4j_offline() {
  bash scripts/prepare_neo4j_offline.sh
}

clone_repo() {
  load_env
  [[ -n "${KERNEL_REPO_GIT_URL:-}" ]] || { echo "KERNEL_REPO_GIT_URL is empty in $ENV_FILE"; exit 1; }
  mkdir -p "$KERNEL_REPO_PATH"
  if [[ -d "$KERNEL_REPO_PATH/.git" ]]; then
    echo "Repo already exists: $KERNEL_REPO_PATH"
  else
    git clone "$KERNEL_REPO_GIT_URL" "$KERNEL_REPO_PATH"
  fi
}

case "$ACTION" in
  help|-h|--help) usage; exit 0 ;;
  insecure-registry) enable_insecure_registry ;;
  package-images) package_images ;;
  load-images) load_images ;;
  doctor) doctor ;;
  up) up_docker_native ;;
  up-prebuilt)  up_prebuilt ;;
  index) load_env; index_docker_native ;;
  mcp) mcp_docker_native ;;
  seq-mcp) up_docker_native; seq_mcp_docker_native ;;
  git-mcp) up_docker_native; git_mcp_docker_native ;;
  build-mcp) up_docker_native; build_mcp_docker_native ;;
  mcp-all) up_docker_native; oneclick_docker_native ;;
  down) down_docker_native ;;
  logs) logs_docker_native ;;
  shell)shell_docker_native ;;
  *) usage; exit 1 ;;
esac
