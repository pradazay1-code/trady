#!/usr/bin/env bash
# Schedule Trady to run unattended on your own machine.
#
#   ./scripts/schedule.sh install    add cron entries
#   ./scripts/schedule.sh remove     take them out again
#   ./scripts/schedule.sh show       print what is scheduled
#   ./scripts/schedule.sh test       run each job once, now
#
# Three jobs get installed:
#   validate  02:00 weekdays  — measure the strategy on real data, push a verdict
#   watch     09:55 weekdays  — start the live alert loop for the session
#   eod       16:05 weekdays  — close out, report, self-review
#
# Times are your machine's local time, so set the clock to US/Eastern or adjust
# the hours below if you are elsewhere.

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="$(command -v python3)"
LOGS="$REPO/logs"
TAG="# trady-scheduled"

usage() { sed -n '2,20p' "$0" | sed 's/^# \{0,1\}//'; exit 1; }

jobs() {
  cat <<EOF
0 2 * * 1-5 cd $REPO && $PY -m trady validate --notify --channels ntfy >> $LOGS/validate.log 2>&1 $TAG
55 9 * * 1-5 cd $REPO && $PY -m trady watch --channels ntfy --interval 300 >> $LOGS/watch.log 2>&1 $TAG
5 16 * * 1-5 cd $REPO && $PY -m trady eod >> $LOGS/eod.log 2>&1 $TAG
EOF
}

case "${1:-}" in
  install)
    mkdir -p "$LOGS"
    current="$(crontab -l 2>/dev/null | grep -v "$TAG" || true)"
    { [ -n "$current" ] && echo "$current"; jobs; } | crontab -
    echo "installed:"; jobs | sed 's/^/  /'
    echo
    echo "Set TRADY_NTFY_TOPIC in your shell profile so cron can see it, e.g."
    echo "  echo 'export TRADY_NTFY_TOPIC=your-topic' >> ~/.profile"
    echo
    echo "Logs: $LOGS/"
    ;;
  remove)
    crontab -l 2>/dev/null | grep -v "$TAG" | crontab - || true
    echo "removed"
    ;;
  show)
    crontab -l 2>/dev/null | grep "$TAG" || echo "nothing scheduled"
    ;;
  test)
    mkdir -p "$LOGS"
    echo "--- validate (bundled real data, quick) ---"
    (cd "$REPO" && "$PY" -m trady validate AAPL MSFT IBM GOOG --real --min-trades 20) || true
    echo
    echo "--- watch, single pass ---"
    (cd "$REPO" && "$PY" -m trady watch --once --dry-run --ignore-clock) || true
    ;;
  *) usage ;;
esac
