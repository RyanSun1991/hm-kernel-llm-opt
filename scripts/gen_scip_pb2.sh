#!/usr/bin/env bash
# Generate src/hmopt/indexing/_generated/scip_pb2.py from third_party/scip/scip.proto
#
# Prerequisites:
#   - protoc (Protocol Buffers compiler), version >= 3.20
#     macOS:  brew install protobuf
#     Linux:  apt-get install protobuf-compiler   (or equivalent)
#
# Usage:
#   bash scripts/gen_scip_pb2.sh           # generate
#   bash scripts/gen_scip_pb2.sh --check   # check protoc availability only
#
# The generated file is checked into the repo so installs without protoc work.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROTO_DIR="$REPO_ROOT/third_party/scip"
PROTO_FILE="$PROTO_DIR/scip.proto"
OUT_DIR="$REPO_ROOT/src/hmopt/indexing/_generated"

check_protoc() {
  if ! command -v protoc >/dev/null 2>&1; then
    echo "[gen_scip_pb2] protoc not found on PATH" >&2
    echo "  Install via:" >&2
    echo "    macOS: brew install protobuf" >&2
    echo "    Linux: apt-get install protobuf-compiler" >&2
    return 1
  fi
  echo "[gen_scip_pb2] protoc: $(protoc --version)"
  return 0
}

if [[ "${1:-}" == "--check" ]]; then
  check_protoc
  exit $?
fi

if [[ ! -f "$PROTO_FILE" ]]; then
  echo "[gen_scip_pb2] scip.proto not found at $PROTO_FILE" >&2
  exit 1
fi

check_protoc || exit 1

mkdir -p "$OUT_DIR"
touch "$OUT_DIR/__init__.py"

protoc \
  --proto_path="$PROTO_DIR" \
  --python_out="$OUT_DIR" \
  "$PROTO_FILE"

# protoc emits scip_pb2.py at OUT_DIR root; nothing else to do.
echo "[gen_scip_pb2] wrote $OUT_DIR/scip_pb2.py"
