#!/usr/bin/env bash
# run_analysis.sh — launcher for the cross-asset price-discovery stack.
#
# The legacy before/after tables (paper-style estimators next to the upgraded ones) default to ON
# here. Disable with:  LEGACY=false ./run_analysis.sh ...
#
# Examples:
#   ./run_analysis.sh                                              # demo run, legacy on (no data needed)
#   ./run_analysis.sh --source extract --date-range 2025-03-25:2025-04-12 --interval 1s
#   LEGACY=false ./run_analysis.sh --source extract --dates 2025-04-03,2025-04-09
set -euo pipefail
cd "$(dirname "$0")"

LEGACY="${LEGACY:-true}"                       # before/after legacy tables default ON
if [ "$LEGACY" = "true" ]; then LEGACY_FLAG="--legacy"; else LEGACY_FLAG="--no-legacy"; fi

PY="${PYTHON:-python}"                          # override interpreter with PYTHON=python3 if needed

# no user args -> a self-contained demo run; otherwise forward the user's args verbatim
if [ "$#" -eq 0 ]; then set -- --source demo --quick; fi

echo "+ $PY run_analysis.py $LEGACY_FLAG $*"
exec "$PY" run_analysis.py "$LEGACY_FLAG" "$@"
