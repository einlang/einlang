#!/usr/bin/env bash
# Run pytest with cProfile; results in prof/combined.prof.
# Usage: ./scripts/profile_tests.sh [path]
#   path: optional test path (default: tests/unit)
# View: python3 -m pstats prof/combined.prof

set -e
cd "$(dirname "$0")/.."
PATH="${1:-tests/unit}"
python3 -m pytest "$PATH" -n 0 --profile -q --tb=short
echo "Profile data: prof/combined.prof (view with: python3 -m pstats prof/combined.prof)"
