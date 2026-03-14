#!/usr/bin/env bash
# Profile deit_tiny (or another example) using einlang's built-in profile tools.
# Usage: ./scripts/profile_deit_tiny.sh [example_dir]
#   example_dir: examples/deit_tiny (default), or e.g. examples/whisper_tiny
# Einlang CLI profile options (add to the einlang command as needed):
#   --profile-functions   per-function total (default in this script)
#   --profile-statements  per top-level statement + line buckets
#   --profile-reductions  reduction path (matmul / vectorized / scalar) per sum/max/min
#   --profile-lines N     bucket by source lines (e.g. 10 → L0-L10, L10-L20, ...)
#   --cprofile           run under Python cProfile; --cprofile-out FILE for snakeviz

set -e
cd "$(dirname "$0")/.."
EXAMPLE="${1:-examples/deit_tiny}"
MAIN="${EXAMPLE}/main.ein"
if [[ ! -f "$MAIN" ]]; then
  echo "Not found: $MAIN" >&2
  exit 1
fi
export PYTHONPATH="${PWD}/src"
echo "Running: python3 -m einlang $MAIN --profile-functions (from $EXAMPLE)"
python3 -m einlang "$MAIN" --profile-functions
