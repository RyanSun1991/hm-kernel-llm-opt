#!/usr/bin/env bash
set -euo pipefail

ACTION="${1:-help}"
ACTION_ARGS=("${@:2}")
CONFIG_PATH="${HMOPT_DOCKER_CONFIG:-configs/app.docker.yaml}"
ENV_FILE=".env.docker"
HMOPT_CONTAINER="hmopt-app"
HMOPT_IMAGE="${HMOPT_IMAGE:-hmopt:local}"
HMOPT_BASE_IMAGE="${HMOPT_BASE_IMAGE:-kernel.dockerhub.rnd.huawei.com/hmci-docker-image:v3-4.2}"
HMOPT_BASE_IMAGE_CANDIDATES="${HMOPT_BASE_IMAGE_CANDIDATES:-}"
PIP_INDEX_URL="${PIP_INDEX_URL:-https://mirrors.tools.huawei.com/pypi/simple}"
PIP_TRUSTED_HOST="${PIP_TRUSTED_HOST:-mirrors.tools.huawei.com}"
REGISTRY_HOST="${REGISTRY_HOST:-kernel.dockerhub.rnd.huawei.com}"
IMAGE_BUNDLE_TAR="${IMAGE_BUNDLE_TAR:-dist/hmopt_bundle.tar.gz}"

usage() {
  cat <<USAGE
Usage / 用法: $0 <action>

Actions / 动作:
  up              Build image and start hmopt container (with embedded neo4j) / 构建并启动单容器（内置neo4j）
  up-prebuilt     Start with preloaded image (no local build) / 使用预置镜像启动（不本地构建）
  index [args]    Build kernel index (supports extra CLI args) / 执行内核索引（支持附加参数）
  mcp             Start MCP server (host port 7332 -> container 7331) / 启动 MCP 服务（宿主7332->容器7331）
  git-mcp         Start Git MCP server (host port 7334 -> container 7334) / 启动 Git MCP 服务（宿主7334->容器7334）
  seq-mcp         Start sequential thinking MCP server (host 7334 -> container 7333) / 启动顺序思考 MCP 服务（宿主7334->容器7333）
  oneclick        Start both MCP servers in background / 一键后台启动两个 MCP 服务
  api             Start REST API (host port 8001 -> container 8000) / 启动 REST API（宿主8001->容器8000）
  clone           Clone kernel repo to KERNEL_REPO_PATH / 克隆代码到 KERNEL_REPO_PATH
  prepare-neo4j-offline  Download neo4j deb on host for offline build / 在host下载neo4j包供离线构建
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
  candidates_raw="$candidates_raw,python:3.10-slim"

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

run_hmopt_container() {
  docker rm -f "$HMOPT_CONTAINER" >/dev/null 2>&1 || true
  docker run -d \
    --name "$HMOPT_CONTAINER" \
    -p 7475:7474 -p 7688:7687 -p 7332:7331 -p 7334:7333 -p 8001:8000 \
    -e HMOPT_LLM_BASE_URL="${HMOPT_LLM_BASE_URL:-http://host.docker.internal:20010/v1}" \
    -e HMOPT_LLM_API_KEY="${HMOPT_LLM_API_KEY:-}" \
    -e HMOPT_MCP_SERVER_API_KEY="${HMOPT_MCP_SERVER_API_KEY:-}" \
    -e HMOPT_MCP_ALLOWED_HOSTS="${HMOPT_MCP_ALLOWED_HOSTS:-}" \
    -e HMOPT_MCP_DISABLE_HOST_CHECK="${HMOPT_MCP_DISABLE_HOST_CHECK:-}" \
    -e HMOPT_MCP_HOST='0.0.0.0' \
    -e HMOPT_MCP_PORT='7331' \
    -e HMOPT_SEQ_MCP_HOST='0.0.0.0' \
    -e HMOPT_SEQ_MCP_PORT='7333' \
    -e HMOPT_START_NEO4J="${HMOPT_START_NEO4J:-1}" \
    -e NEO4J_USER="${NEO4J_USER:-neo4j}" \
    -e NEO4J_PASSWORD="${NEO4J_PASSWORD:-@huawei2026}" \
    -e HMOPT_PATH_ALIAS="${HMOPT_PATH_ALIAS:-}" \
    -v "$(pwd):/app" \
    -v "${KERNEL_REPO_PATH:-$(pwd)/data/sample-kernel}:/workspace/kernel:rw" \
    -v "${KERNEL_REPO_PATH:-$(pwd)/data/sample-kernel}:${KERNEL_REPO_PATH:-/workspace/kernel}:rw" \
    -v "${KERNEL_WORKSPACE_PATH:-$(dirname "${KERNEL_REPO_PATH:-$(pwd)/data/sample-kernel}")}:${KERNEL_WORKSPACE_PATH:-$(dirname "${KERNEL_REPO_PATH:-$(pwd)/data/sample-kernel}")}:rw" \
    -v "$(pwd)/data/neo4j/data:/var/lib/neo4j" \
    -v "$(pwd)/data/neo4j/logs:/var/log/neo4j" \
    -v "$(pwd)/data/neo4j/plugins:/var/lib/neo4j/plugins" \
    -w /app \
    "$HMOPT_IMAGE" \
    bash -lc "tail -f /dev/null" >/dev/null
}

up_docker_native() {
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

  if [[ "$raw_path" == /workspace/* || "$raw_path" == /app/* ]]; then
    printf "%s\n" "$raw_path"
    return 0
  fi

  if [[ -n "${KERNEL_REPO_PATH:-}" && "$raw_path" == "$KERNEL_REPO_PATH"* ]]; then
    printf "/workspace/kernel%s\n" "${raw_path#${KERNEL_REPO_PATH}}"
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
git_mcp_docker_native() { docker exec "$HMOPT_CONTAINER" bash -lc 'bash scripts/run_git_mcp_server.sh'; }
seq_mcp_docker_native() { docker exec "$HMOPT_CONTAINER" bash -lc 'bash scripts/run_seq_mcp_server.sh'; }
oneclick_docker_native() {
  docker exec -d "$HMOPT_CONTAINER" bash -lc 'nohup bash scripts/run_all_mcp_servers.sh >/tmp/all_mcp_servers.log 2>&1 &'
  echo 'Started MCP (7331) and sequential thinking MCP (7333) in background.'
}
api_docker_native() { docker exec "$HMOPT_CONTAINER" bash -lc 'uvicorn hmopt.api.main:app --host 0.0.0.0 --port 8000'; }
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

  local load_output=""
  load_output="$(gunzip -c "$IMAGE_BUNDLE_TAR" | docker load)"
  echo "$load_output"

  if ! docker image inspect "$HMOPT_IMAGE" >/dev/null 2>&1; then
    local loaded_ref=""
    loaded_ref="$(printf '%s\n' "$load_output" | sed -n -E 's/^Loaded image\(s\):[[:space:]]*//p' | head -n1 | cut -d',' -f1 | xargs)"
    if [[ -n "$loaded_ref" ]]; then
      docker tag "$loaded_ref" "$HMOPT_IMAGE"
      echo "[load-images] retagged $loaded_ref -> $HMOPT_IMAGE"
    fi
  fi

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
  up) if has_compose; then compose_run up -d --build; else up_docker_native; fi ;;
  up-prebuilt) if has_compose; then compose_run up -d --no-build; else up_prebuilt; fi ;;
  index) load_env; if has_compose; then compose_run up -d hmopt; mapfile -t _index_args < <(build_index_args); compose_run exec hmopt python -m hmopt.cli index-kernel "${_index_args[@]}"; else up_docker_native; index_docker_native; fi ;;
  mcp) if has_compose; then compose_run up -d hmopt; compose_run exec hmopt bash -lc 'python -m hmopt.cli serve-mcp --host 0.0.0.0 --port 7331'; else up_docker_native; mcp_docker_native; fi ;;
  git-mcp) if has_compose; then compose_run up -d hmopt-git-mcp; else up_docker_native; git_mcp_docker_native; fi ;;
  seq-mcp) if has_compose; then compose_run up -d hmopt; compose_run exec hmopt bash -lc 'bash scripts/run_seq_mcp_server.sh'; else up_docker_native; seq_mcp_docker_native; fi ;;
  oneclick) if has_compose; then compose_run up -d hmopt; compose_run exec -d hmopt bash -lc 'nohup bash scripts/run_all_mcp_servers.sh >/tmp/all_mcp_servers.log 2>&1 &'; echo 'Started MCP (7331) and sequential thinking MCP (7333) in background.'; else up_docker_native; oneclick_docker_native; fi ;;
  api) if has_compose; then compose_run up -d hmopt; compose_run exec hmopt bash -lc 'uvicorn hmopt.api.main:app --host 0.0.0.0 --port 8000'; else up_docker_native; api_docker_native; fi ;;
  clone) clone_repo ;;
  prepare-neo4j-offline) prepare_neo4j_offline ;;
  down) if has_compose; then compose_run down; else down_docker_native; fi ;;
  logs) if has_compose; then compose_run logs -f; else logs_docker_native; fi ;;
  shell) if has_compose; then compose_run up -d hmopt; compose_run exec hmopt bash; else up_docker_native; shell_docker_native; fi ;;
  *) usage; exit 1 ;;
esac
