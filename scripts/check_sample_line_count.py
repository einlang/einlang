#!/usr/bin/env python3
"""
Check that all sample .ein files have > MIN_LINES code lines.

Excluded from count:
- Comments and blank lines
- print(...) statements
- Duplicated patterns: if a line differs only by numbers/identifiers from another
  (e.g. balance_year1, balance_year2), only the first occurrence of that pattern counts.
  So manual expansion or repetition of the same pattern does not add to the count.

Samples = .ein files under examples/ in: applications, basics, demos, finance, job_search,
value_iteration, optimization, time_series, ode, pde_1d, wave_2d, brusselator, recurrence,
numerics; plus run_numerics*.ein and mnist/main.ein, mnist_quantized/main.ein, deit_tiny/main.ein.

Excludes: examples/units/, examples/whisper_tiny/test_patterns/, examples/whisper_tiny/profile/,
and examples/hello.ein (minimal entry point).

Usage: python3 scripts/check_sample_line_count.py [--min N]
Exit: 0 if all pass, 1 if any sample has <= MIN_LINES code lines.
"""

import argparse
import re
import sys
from pathlib import Path
from typing import List


MIN_LINES_DEFAULT = 50

SAMPLE_DIRS = (
    "applications",
    "basics",
    "demos",
    "finance",
    "job_search",
    "value_iteration",
    "optimization",
    "time_series",
    "ode",
    "pde_1d",
    "wave_2d",
    "brusselator",
    "recurrence",
    "numerics",
)

SAMPLE_ROOT_FILES = (
    "run_numerics.ein",
)

SAMPLE_MAIN_FILES = (
    "mnist/main.ein",
    "mnist_quantized/main.ein",
    "deit_tiny/main.ein",
)

EXCLUDE_SUBDIRS = ("units", "test_patterns", "profile")
EXCLUDE_FILES = ("hello.ein",)


def _normalize_pattern(line: str) -> str:
    """Replace runs of digits with 0 so duplicate patterns (year1/year2, etc.) collapse."""
    return re.sub(r"\d+", "0", line)


def count_code_lines(path: Path) -> int:
    text = path.read_text()
    lines = text.splitlines()
    seen_patterns = set()
    n = 0
    in_block = False
    for line in lines:
        s = line.strip()
        if "/*" in s:
            in_block = True
        if in_block:
            if "*/" in s:
                in_block = False
            continue
        if not s or s.startswith("//"):
            continue
        if s.startswith("print("):
            continue
        pattern = _normalize_pattern(s)
        if pattern in seen_patterns:
            continue
        seen_patterns.add(pattern)
        n += 1
    return n


def collect_sample_paths(examples_dir: Path) -> List[Path]:
    paths = []
    for d in SAMPLE_DIRS:
        sub = examples_dir / d
        if not sub.is_dir():
            continue
        for ein in sub.rglob("*.ein"):
            if any(ex in ein.parts for ex in EXCLUDE_SUBDIRS):
                continue
            paths.append(ein)
    for name in SAMPLE_ROOT_FILES:
        p = examples_dir / name
        if p.exists():
            paths.append(p)
    for name in SAMPLE_MAIN_FILES:
        p = examples_dir / name
        if p.exists():
            paths.append(p)
    paths = [p for p in paths if p.name not in EXCLUDE_FILES]
    return sorted(set(paths))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--min", type=int, default=MIN_LINES_DEFAULT, help="Minimum code lines (default %s)" % MIN_LINES_DEFAULT)
    parser.add_argument("--examples", type=Path, default=Path("examples"), help="Path to examples dir")
    args = parser.parse_args()
    examples_dir = args.examples.resolve()
    if not examples_dir.is_dir():
        print("Not a directory:", examples_dir, file=sys.stderr)
        return 1
    paths = collect_sample_paths(examples_dir)
    failed = []
    for p in paths:
        try:
            n = count_code_lines(p)
        except Exception as e:
            print("Error reading %s: %s" % (p, e), file=sys.stderr)
            failed.append((p, None))
            continue
        try:
            rel = p.relative_to(examples_dir)
        except ValueError:
            rel = p
        if n <= args.min:
            failed.append((rel, n))
    if failed:
        for rel, n in failed:
            print("%s: %s code lines (need > %s)" % (rel, n if n is not None else "?", args.min))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
