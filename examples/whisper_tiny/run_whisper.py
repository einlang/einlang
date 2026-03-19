#!/usr/bin/env python3
"""
Run Whisper-tiny (main.ein) and print transcript.

From repo root:
  python3 examples/whisper_tiny/run_whisper.py
  python3 examples/whisper_tiny/run_whisper.py --profile-einlang   # print vectorized/hybrid/scalar + per-statement time
"""

import os
import sys
from pathlib import Path

# Set before any backend code runs (per VECTORIZATION_DESIGN; whisper has long recurrence)
os.environ.setdefault("EINLANG_EINSTEIN_LOOP_MAX", "10000")

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from einlang.compiler.driver import CompilerDriver
from einlang.runtime.runtime import EinlangRuntime

EXAMPLE_DIR = Path(__file__).resolve().parent
MAIN_EIN = EXAMPLE_DIR / "main.ein"


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Run Whisper-tiny and print transcript")
    ap.add_argument("--profile-einlang", action="store_true", help="Print backend path (vectorized/hybrid/scalar) and per-statement profile")
    ap.add_argument("--profile-verbose", action="store_true", help="Full debug: statements, functions, blocks, reductions, vectorize summary")
    args = ap.parse_args()

    os.chdir(EXAMPLE_DIR)

    if args.profile_verbose:
        os.environ["EINLANG_PROFILE_STATEMENTS"] = "1"
        os.environ["EINLANG_PROFILE_FUNCTIONS"] = "1"
        os.environ["EINLANG_PROFILE_BLOCKS"] = "1"
        os.environ["EINLANG_PROFILE_REDUCTIONS"] = "1"
        os.environ["EINLANG_DEBUG_VECTORIZE"] = "1"
        os.environ.setdefault("EINLANG_PROFILE_LINES", "20")
    elif args.profile_einlang:
        os.environ["EINLANG_DEBUG_VECTORIZE"] = "1"
        os.environ["EINLANG_PROFILE_STATEMENTS"] = "1"

    source = MAIN_EIN.read_text(encoding="utf-8")
    compiler = CompilerDriver()
    runtime = EinlangRuntime()
    result = compiler.compile(source, str(MAIN_EIN), root_path=EXAMPLE_DIR)
    if not result.success:
        err = result.tcx
        if err and err.reporter:
            print(err.reporter.format_all_errors(), file=sys.stderr)
        sys.exit(1)
    exec_result = runtime.execute(result)
    if exec_result.error is not None:
        print(exec_result.error, file=sys.stderr)
        sys.exit(1)
    out = exec_result.value
    if out is not None:
        print(out)


if __name__ == "__main__":
    main()
