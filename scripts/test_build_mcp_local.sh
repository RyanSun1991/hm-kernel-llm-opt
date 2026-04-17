#!/usr/bin/env bash
set -euo pipefail

PORT="${HMOPT_BUILD_MCP_TEST_PORT:-7335}"
RUNNER_IMAGE="${HMOPT_BUILD_MCP_TEST_RUNNER_IMAGE:-bash:5.2}"
WORKDIR="$(mktemp -d)"
RUNNER_NAME="hmopt-build-runner-test-$$"
LOG_FILE="$WORKDIR/build_mcp_server.log"

cleanup() {
  set +e
  [[ -n "${SERVER_PID:-}" ]] && kill "$SERVER_PID" >/dev/null 2>&1 || true
  docker rm -f "$RUNNER_NAME" >/dev/null 2>&1 || true
  rm -rf "$WORKDIR"
}
trap cleanup EXIT

mkdir -p "$WORKDIR/project/kernel/hongmeng/build/hm_scripts"
mkdir -p "$WORKDIR/hm-CI"

cat > "$WORKDIR/project/kernel/hongmeng/build/hm_scripts/build.sh" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
printf 'build.sh called with: %s\n' "$*"
EOF
chmod +x "$WORKDIR/project/kernel/hongmeng/build/hm_scripts/build.sh"

cat > "$WORKDIR/hm-CI/do_pack_hione_trunk.sh" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
printf 'do_pack_hione_trunk.sh called with: %s\n' "$*"
EOF
chmod +x "$WORKDIR/hm-CI/do_pack_hione_trunk.sh"

echo "[1/4] starting mock build runner container: $RUNNER_NAME"
docker run -d --rm --name "$RUNNER_NAME" -v "$WORKDIR:/workspace" "$RUNNER_IMAGE" bash -lc 'tail -f /dev/null' >/dev/null

echo "[2/4] starting Build MCP server on port $PORT"
HMOPT_BUILD_MCP_HOST=127.0.0.1 \
HMOPT_BUILD_MCP_PORT="$PORT" \
HMOPT_BUILD_MCP_MODE=exec \
HMOPT_BUILD_MCP_RUNNER_CONTAINER="$RUNNER_NAME" \
HMOPT_BUILD_MCP_PROJECT_PATH=/workspace/project \
HMOPT_BUILD_MCP_SIGN_WORKSPACE=/workspace/hm-CI \
PYTHONPATH=src \
python -m uvicorn hmopt.api.build_mcp_server:app --host 127.0.0.1 --port "$PORT" >"$LOG_FILE" 2>&1 &
SERVER_PID=$!

for _ in {1..30}; do
  if curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null; then
    break
  fi
  sleep 1
done
curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null

echo "[3/4] calling kernel_build_trigger"
BUILD_RES="$(curl -sf -X POST "http://127.0.0.1:${PORT}/tools/call" \
  -H 'Content-Type: application/json' \
  -d '{
    "tool": "kernel_build_trigger",
    "arguments": {
      "device": "charlotte",
      "profile": "release",
      "project_path": "/workspace/project"
    }
  }')"
python - <<'PY' "$BUILD_RES"
import json,sys
obj=json.loads(sys.argv[1])
content=obj["result"]["content"]
print("build returncode:", content["returncode"])
print("build stdout:", content["stdout"].strip())
assert content["returncode"] == 0, content
PY

echo "[4/4] calling kernel_sign_trigger (package/sign)"
SIGN_RES="$(curl -sf -X POST "http://127.0.0.1:${PORT}/tools/call" \
  -H 'Content-Type: application/json' \
  -d '{
    "tool": "kernel_sign_trigger",
    "arguments": {
      "device": "nashville"
    }
  }')"
python - <<'PY' "$SIGN_RES"
import json,sys
obj=json.loads(sys.argv[1])
content=obj["result"]["content"]
print("sign returncode:", content["returncode"])
print("sign stdout:", content["stdout"].strip())
assert content["returncode"] == 0, content
PY

echo "PASS: Build MCP local test completed"
