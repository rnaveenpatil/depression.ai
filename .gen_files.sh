#!/usr/bin/env bash
set -euo pipefail
cd /home/patil/depression.ai

# Directories to prune (never glob-expanded thanks to -prune form)
prune=(
  -name .git -o
  -name __pycache__ -o
  -name .mypy_cache -o
  -name .pytest_cache -o
  -name .ruff_cache -o
  -path ./deprssion -o
  -path ./dist -o
  -path ./.opencode/node_modules -o
  -path ./src/agent/depression -o
  -path ./FILES.md -o
  -path ./.gen_files.sh
)

list_files() { find . \( "${prune[@]}" \) -prune -o -type f "$@" ; }

# Single source of truth: the pruned file list (relative paths).
tmp=$(mktemp)
trap 'rm -f "$tmp"' EXIT
list_files -printf '%P\n' | sort > "$tmp"
total=$(wc -l < "$tmp")

{
  echo "# depression.ai — Project File Listing"
  echo
  echo "Generated: $(date '+%Y-%m-%d %H:%M:%S')"
  echo "Repo: $(git rev-parse --short HEAD 2>/dev/null)  Branch: $(git rev-parse --abbrev-ref HEAD 2>/dev/null)"
  echo
  echo "## Summary"
  echo
  echo "| Area | Files | Size |"
  echo "|---|---:|---:|"
  dirs=(src/agent src/agent/depression tests docs assets .agents .opencode .github .githooks .agent)
  counted=0
  for d in "${dirs[@]}"; do
    n=$(grep -c "^$d/" "$tmp" || true)
    counted=$((counted + n))
    [ "$n" -eq 0 ] && continue
    s=$(du -sh "$d" 2>/dev/null | cut -f1)
    echo "| \`$d\` | $n | $s |"
  done
  echo "| (root files) | $((total - counted)) | — |"
  echo "| **total** | **$total** | |"
  echo
  echo "Excluded: .git/, deprssion/ (venv), dist/, .opencode/node_modules/,"
  echo "src/agent/depression/ (nested venv), .mypy_cache/, .pytest_cache/,"
  echo ".ruff_cache/, __pycache__/"
  echo
  echo "## Files with sizes (bytes)"
  echo
  echo '```'
  list_files -printf '%10s  %P\n' | sort -k2
  echo '```'
} > FILES.md

echo "WROTE FILES.md ($(wc -l < FILES.md) lines, $(stat -c%s FILES.md) bytes)"