#!/usr/bin/env sh
set -eu
ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
VERSION=$(cat "$ROOT/VERSION" 2>/dev/null || printf '0.0.0')
git_head_commit() {
  git_path="$ROOT/.git"
  [ -e "$git_path" ] || return 1
  git_dir="$git_path"
  if [ -f "$git_path" ]; then
    git_dir=$(sed -n 's/^gitdir: //p' "$git_path" | head -n 1)
    [ -n "$git_dir" ] || return 1
    case "$git_dir" in
      /*) ;;
      *) git_dir="$ROOT/$git_dir" ;;
    esac
  fi
  [ -f "$git_dir/HEAD" ] || return 1
  head_value=$(sed -n '1p' "$git_dir/HEAD")
  case "$head_value" in
    "ref: "*)
      ref_name=${head_value#ref: }
      if [ -f "$git_dir/$ref_name" ]; then
        head_value=$(sed -n '1p' "$git_dir/$ref_name")
      elif [ -f "$git_dir/packed-refs" ]; then
        head_value=$(awk -v ref="$ref_name" '$2 == ref { print $1; exit }' "$git_dir/packed-refs")
        [ -n "$head_value" ] || return 1
      else
        return 1
      fi
      ;;
  esac
  printf '%.7s' "$head_value"
}
COMMIT=$(git -C "$ROOT" rev-parse --short HEAD 2>/dev/null || git_head_commit || printf 'unknown')
printf 'ARP %s\n' "$VERSION"
printf 'Commit %s\n' "$COMMIT"
if [ -x "$ROOT/.venv/bin/python" ]; then
  PY="$ROOT/.venv/bin/python"
else
  PY="${PYTHON:-python3}"
fi
exec "$PY" -m ai_remaster_gui "$@"
