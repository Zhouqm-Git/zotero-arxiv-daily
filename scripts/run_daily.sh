#!/usr/bin/env bash
# Daily runner for zotero-arxiv-daily, invoked by launchd.
#
# launchd gives a minimal PATH, so we set an explicit one. We use `uv run`
# against the project so the venv is always in sync before each run.
set -euo pipefail

PROJECT_DIR="/Users/zhouqm/projects/zotero-arxiv-daily"
LOG_DIR="$HOME/Library/Logs/arxiv-daily"
mkdir -p "$LOG_DIR"

# launchd provides a sparse PATH; uv + git live in /opt/homebrew/bin.
export PATH="/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin"
export UV_PYTHON="$PROJECT_DIR/.venv/bin/python"

cd "$PROJECT_DIR"

# uv sync is cheap when nothing changed; guarantees deps are ready.
uv sync --quiet 2>>"$LOG_DIR/err.log"

# Run the pipeline. Output is also streamed to a dated log for debugging.
uv run python src/zotero_arxiv_daily/main.py >>"$LOG_DIR/out.log" 2>>"$LOG_DIR/err.log"
