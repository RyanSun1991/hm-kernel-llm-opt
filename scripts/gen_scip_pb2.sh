#!/usr/bin/env bash
# Generate src/hmopt/indexing/_generated/scip_pb2.py from third_party/scip/scip.proto
#
# Two ways to get protoc:
#   1. System install (preferred on shared dev boxes):
#        macOS:  brew install protobuf
#        Linux:  apt-get install protobuf-compiler   (or equivalent)
#   2. Python-wheel install (no system privileges needed; bundles protoc):
#        pip install grpcio-tools
#      The script auto-detects this and uses `python -m grpc_tools.protoc`.
#
# Usage:
#   bash scripts/gen_scip_pb2.sh           # generate
#   bash scripts/gen_scip_pb2.sh --check   # check protoc availability only
#
# Notes:
#   - The generated scip_pb2.py is checked into the repo, so end users and
#     CI do NOT need protoc — only `protobuf` (Python runtime, in pyproject).
#   - Requires protoc >= 3.19 to emit modern descriptor-pool-based code that
#     works with protobuf>=4.25. Older protoc emits descriptor-direct calls
#     that fail at import time under modern protobuf runtimes.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROTO_DIR="$REPO_ROOT/third_party/scip"
PROTO_FILE="$PROTO_DIR/scip.proto"
OUT_DIR="$REPO_ROOT/src/hmopt/indexing/_generated"

# Detect a usable protoc. Prefer system protoc; fall back to grpcio-tools.
detect_protoc() {
  if command -v protoc >/dev/null 2>&1; then
    PROTOC_CMD=(protoc)
    PROTOC_VERSION="$(protoc --version)"
    return 0
  fi
  for py in python3 python; do
    if command -v "$py" >/dev/null 2>&1 \
        && "$py" -c "import grpc_tools.protoc" >/dev/null 2>&1; then
      PROTOC_CMD=("$py" -m grpc_tools.protoc)
      PROTOC_VERSION="$("$py" -m grpc_tools.protoc --version 2>&1 | head -n1)"
      return 0
    fi
  done
  return 1
}

if [[ "${1:-}" == "--check" ]]; then
  if detect_protoc; then
    echo "[gen_scip_pb2] using: ${PROTOC_CMD[*]} ($PROTOC_VERSION)"
    exit 0
  fi
  echo "[gen_scip_pb2] no protoc available" >&2
  echo "  Install either:" >&2
  echo "    macOS: brew install protobuf" >&2
  echo "    Linux: apt-get install protobuf-compiler" >&2
  echo "  Or pip install: pip install grpcio-tools" >&2
  exit 1
fi

if [[ ! -f "$PROTO_FILE" ]]; then
  echo "[gen_scip_pb2] scip.proto not found at $PROTO_FILE" >&2
  exit 1
fi

if ! detect_protoc; then
  echo "[gen_scip_pb2] no protoc available — see --check for install hints" >&2
  exit 1
fi

echo "[gen_scip_pb2] using: ${PROTOC_CMD[*]} ($PROTOC_VERSION)"

mkdir -p "$OUT_DIR"
[[ -f "$OUT_DIR/__init__.py" ]] || cat > "$OUT_DIR/__init__.py" <<'EOF'
"""Auto-generated protobuf bindings for the SCIP protocol.

Regenerate from third_party/scip/scip.proto via:
    bash scripts/gen_scip_pb2.sh
"""
EOF

"${PROTOC_CMD[@]}" \
  --proto_path="$PROTO_DIR" \
  --python_out="$OUT_DIR" \
  "$PROTO_FILE"

echo "[gen_scip_pb2] wrote $OUT_DIR/scip_pb2.py"
