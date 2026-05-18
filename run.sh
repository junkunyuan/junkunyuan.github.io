#!/usr/bin/env bash
# Validate YAML, then build both the home page and the reading-list pages.
# Pass --watch to keep rebuilding every few seconds.

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"

build() {
    echo "=== $(date '+%Y-%m-%d %H:%M:%S') ==="
    python -m build.scripts.validate
    python -m build.index
    python -m build.publications
    python -m build.reading_list
}

if [[ "${1:-}" == "--watch" ]]; then
    trap 'echo "exiting"; exit 0' INT TERM
    while true; do
        build || echo "(build failed; will retry)"
        sleep 3
    done
else
    build
fi
