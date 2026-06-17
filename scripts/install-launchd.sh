#!/usr/bin/env bash
# Install (or update) the launchd agent for zotero-arxiv-daily.
#
# Usage:
#   scripts/install-launchd.sh          # install / re-install
#   scripts/install-launchd.sh remove   # uninstall
#
# Copies the plist into ~/Library/LaunchAgents/ (resolving the repo-relative
# paths to absolute, since launchd needs those) and bootstraps it.
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PLIST_SRC="$PROJECT_DIR/scripts/com.zhouqm.arxiv-daily.plist"
PLIST_DST="$HOME/Library/LaunchAgents/com.zhouqm.arxiv-daily.plist"
LABEL="com.zhouqm.arxiv-daily"

action="${1:-install}"

if [ "$action" = "remove" ] || [ "$action" = "uninstall" ]; then
    echo "Uninstalling $LABEL..."
    launchctl bootout "gui/$UID/$LABEL" 2>/dev/null || echo "  (was not loaded)"
    rm -f "$PLIST_DST"
    echo "Removed $PLIST_DST"
    exit 0
fi

# Ensure log dir exists (the plist writes here).
mkdir -p "$HOME/Library/Logs/arxiv-daily"

# The plist in the repo already has absolute paths; copy it verbatim.
cp "$PLIST_SRC" "$PLIST_DST"
echo "Installed plist -> $PLIST_DST"

# Bootstrap (replaces any previously loaded copy).
launchctl bootout "gui/$UID/$LABEL" 2>/dev/null || true
launchctl bootstrap "gui/$UID" "$PLIST_DST"
echo "Loaded $LABEL."

# Show the next firing time so the user can confirm scheduling.
echo
echo "Scheduled (next fire will be at the configured calendar interval)."
echo "Verify with:  launchctl print gui/$UID/$LABEL | grep -A6 'calendar'"
echo "Run now:      scripts/install-launchd.sh run"
echo "Logs:         tail -f ~/Library/Logs/arxiv-daily/out.log"

if [ "${2:-}" = "run" ]; then
    echo
    echo "Triggering an immediate run..."
    launchctl kickstart -k "gui/$UID/$LABEL"
fi
