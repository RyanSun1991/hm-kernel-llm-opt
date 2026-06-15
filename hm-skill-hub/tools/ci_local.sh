#!/usr/bin/env bash
# Local mirror of the hub CI (.github/workflows/ci.yml / skill-hub-ci.yml) — run
# the exact same blocking gates WITHOUT GitHub, on a laptop or inside the hmopt
# Docker image. Run from anywhere; it cd's to the hub root.
#
#   bash tools/ci_local.sh                 # full hub gate (tests+lint+redact+dedup+eval)
#   PYTHON=/path/to/python bash tools/ci_local.sh
#
# In Docker (everything is already installed in the hmopt image):
#   docker compose run --rm hmopt bash -lc "bash hm-skill-hub/tools/ci_local.sh"
#   # or, against a running container:
#   docker exec hmopt-app bash -lc "bash hm-skill-hub/tools/ci_local.sh"

set -uo pipefail
PY="${PYTHON:-python3}"
cd "$(dirname "$0")/.."   # -> hub root
fail=0

step() { echo; echo "== $* =="; }
chk()  { if "$@"; then :; else echo "  ^ FAILED"; fail=1; fi; }

step "1/5 toolchain tests (lint/scope/curator/dedup/subsumption/promotion/skillopt)"
chk "$PY" -m pytest tools/tests/ -q

step "2/5 schema lint + path<->scope consistency + id uniqueness"
chk "$PY" tools/lint.py

step "3/5 secret scan (redact)"
chk "$PY" tools/redact.py --check

step "4/5 dedup gate over staging/ (engine A — fail on unresolved conflict)"
staged=()
while IFS= read -r f; do staged+=("$f"); done < <(find staging -type f -name '*.jsonl' 2>/dev/null | sort)
if [ ${#staged[@]} -eq 0 ]; then
  echo "  (no staged *.jsonl — nothing to dedup)"
else
  for f in "${staged[@]}"; do echo "  dedup $f"; chk "$PY" tools/dedup.py "$f" --check; done
fi

step "5/5 eval-gate (engine B — reject skill regressions)"
chk "$PY" tools/eval_gate.py

echo
if [ "$fail" -eq 0 ]; then echo "✅ local CI passed"; else echo "❌ local CI FAILED"; exit 1; fi
