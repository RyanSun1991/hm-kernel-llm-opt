#!/usr/bin/env bash
set -euo pipefail

CONFIG="configs/app.yaml"
PROFILES_PATH="configs/pipeline_profiles.yaml"
PROFILE=""
TARGET=""
OBJECTIVE=""
OPEN_CMD=""
REPO_PATH=""
COMPILE_COMMANDS_DIR=""
START_MCP=0
INDEX_KERNEL=0
LAUNCH_OPENCODE=0
ARTIFACTS=()

usage() {
  cat <<'EOF'
Usage:
  bash scripts/run_opencode_pipeline.sh --profile <name> --target <path-or-symbol> [options]

Options:
  --profile <name>                Pipeline profile name
  --target <value>                Target path, symbol, or subsystem
  --objective <text>              Override objective text
  --artifact <kind:path>          Repeatable artifact spec
  --config <path>                 HMOPT config path
  --profiles-path <path>          Pipeline profiles yaml path
  --repo-path <path>              Repo path for index-kernel
  --compile-commands-dir <path>   compile_commands directory for index-kernel
  --start-mcp                     Start MCP services if ports are not listening
  --index-kernel                  Run kernel indexing before staging prompt
  --launch-opencode               Launch opencode CLI after staging prompt if installed
  --open-cmd <command>            Custom command to launch OpenCode or a wrapper
  --help                          Show this message
EOF
}

port_listening() {
  local port="$1"
  if command -v ss >/dev/null 2>&1; then
    ss -ltn | awk '{print $4}' | grep -q ":${port}$"
    return $?
  fi
  return 1
}

start_service() {
  local name="$1"
  local port="$2"
  local script="$3"
  local log_dir="outputs/opencode_pipeline"
  mkdir -p "$log_dir"
  if port_listening "$port"; then
    echo "[skip] ${name} already listening on :${port}"
    return 0
  fi
  echo "[start] ${name} via ${script}"
  nohup bash "$script" > "${log_dir}/${name}.log" 2>&1 &
  sleep 1
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --profile)
      PROFILE="$2"
      shift 2
      ;;
    --target)
      TARGET="$2"
      shift 2
      ;;
    --objective)
      OBJECTIVE="$2"
      shift 2
      ;;
    --artifact)
      ARTIFACTS+=("$2")
      shift 2
      ;;
    --config)
      CONFIG="$2"
      shift 2
      ;;
    --profiles-path)
      PROFILES_PATH="$2"
      shift 2
      ;;
    --repo-path)
      REPO_PATH="$2"
      shift 2
      ;;
    --compile-commands-dir)
      COMPILE_COMMANDS_DIR="$2"
      shift 2
      ;;
    --start-mcp)
      START_MCP=1
      shift
      ;;
    --index-kernel)
      INDEX_KERNEL=1
      shift
      ;;
    --launch-opencode)
      LAUNCH_OPENCODE=1
      shift
      ;;
    --open-cmd)
      OPEN_CMD="$2"
      shift 2
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ -z "$PROFILE" || -z "$TARGET" ]]; then
  usage
  exit 1
fi

if [[ "$START_MCP" -eq 1 ]]; then
  start_service "kernel_index_mcp" 7331 "scripts/run_mcp_server.sh"
  start_service "seq_mcp" 7333 "scripts/run_seq_mcp_server.sh"
  start_service "git_mcp" 7334 "scripts/run_git_mcp_server.sh"
  start_service "build_mcp" 7335 "scripts/run_build_mcp_server.sh"
  start_service "auto_test_mcp" 7336 "scripts/run_auto_test_mcp_server.sh"
fi

if [[ "$INDEX_KERNEL" -eq 1 ]]; then
  if [[ -z "$REPO_PATH" || -z "$COMPILE_COMMANDS_DIR" ]]; then
    echo "--index-kernel requires --repo-path and --compile-commands-dir" >&2
    exit 1
  fi
  PYTHONPATH=src python3 -m hmopt.cli index-kernel \
    --config "$CONFIG" \
    --repo-path "$REPO_PATH" \
    --compile-commands-dir "$COMPILE_COMMANDS_DIR"
fi

CMD=(python3 -m hmopt.cli start-pipeline --config "$CONFIG" --profiles-path "$PROFILES_PATH" --profile "$PROFILE" --target "$TARGET")
if [[ -n "$OBJECTIVE" ]]; then
  CMD+=(--objective "$OBJECTIVE")
fi
for artifact in "${ARTIFACTS[@]}"; do
  CMD+=(--artifact "$artifact")
done
if [[ "$LAUNCH_OPENCODE" -eq 1 ]]; then
  CMD+=(--launch-opencode)
fi
if [[ -n "$OPEN_CMD" ]]; then
  CMD+=(--open-cmd "$OPEN_CMD")
fi

PYTHONPATH=src "${CMD[@]}"
