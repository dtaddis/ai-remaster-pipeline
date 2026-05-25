#!/usr/bin/env sh
set -eu
ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
VERSION=$(cat "$ROOT/VERSION" 2>/dev/null || printf '0.0.0')
COMMIT=$(git -C "$ROOT" rev-parse --short HEAD 2>/dev/null || printf 'unknown')
printf 'ARP %s\n' "$VERSION"
printf 'Commit %s\n' "$COMMIT"
if [ -x "$ROOT/.venv/bin/python" ]; then
  PY="$ROOT/.venv/bin/python"
else
  PY="${PYTHON:-python3}"
fi
exec "$PY" -m ai_remaster_gui "$@"
