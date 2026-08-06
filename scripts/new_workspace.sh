#!/usr/bin/env bash
# Create a workbench task workspace from the tracked template.
#
#   bash scripts/new_workspace.sh <task-slug>            # new workspace
#   bash scripts/new_workspace.sh <task-slug> --fork <src-slug>   # fork: copy an
#                                                        # existing workspace instead
#
# Workspaces live in .opencode/local/workspaces/ (git-ignored runtime state — never
# committed). Forks copy the whole directory so branches can be compared; they never
# overwrite the original. See skills/infra/agent-core/SKILL.md §5.

set -euo pipefail

usage() { grep '^#' "$0" | sed 's/^# \{0,1\}//' | head -9; exit 1; }

SLUG="${1:-}"
[ -n "$SLUG" ] || usage
case "$SLUG" in
  *[!a-z0-9-]*) echo "error: slug must be lowercase kebab-case (got: $SLUG)" >&2; exit 1 ;;
esac

REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
WS_ROOT="$REPO_ROOT/.opencode/local/workspaces"
TEMPLATE="$REPO_ROOT/.opencode/templates/workspace"
DEST="$WS_ROOT/$SLUG"

if [ -e "$DEST" ]; then
  echo "error: workspace already exists: $DEST (forks must use a new slug)" >&2
  exit 1
fi

mkdir -p "$WS_ROOT"

if [ "${2:-}" = "--fork" ]; then
  SRC="$WS_ROOT/${3:-}"
  [ -d "$SRC" ] || { echo "error: fork source not found: $SRC" >&2; exit 1; }
  cp -r "$SRC" "$DEST"
  printf '\n> forked from %s on %s\n' "${3}" "$(date +%F)" >> "$DEST/task.md"
  echo "forked workspace: $DEST (from ${3})"
else
  [ -d "$TEMPLATE" ] || { echo "error: template missing: $TEMPLATE" >&2; exit 1; }
  cp -r "$TEMPLATE" "$DEST"
  echo "new workspace: $DEST"
fi

echo "next: fill in task.md + capsule.md; the active role updates capsule.md every turn"
