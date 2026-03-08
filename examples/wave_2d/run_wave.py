#!/usr/bin/env python3
"""
Run 2D wave equation (main.ein) and write HTML animation.

From repo root:
  python3 examples/wave_2d/run_wave.py
  python3 examples/wave_2d/run_wave.py --html wave.html
  python3 examples/wave_2d/run_wave.py --profile-einlang   # per-clause time + vectorized/hybrid/scalar summary
"""

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np
from einlang.compiler.driver import CompilerDriver
from einlang.runtime.runtime import EinlangRuntime
from einlang.utils.html_wave import write_wave_html

EXAMPLE_DIR = Path(__file__).resolve().parent
MAIN_EIN = EXAMPLE_DIR / "main.ein"


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Run 2D wave equation and write HTML animation")
    ap.add_argument("--html", type=str, default="wave.html", help="Output HTML path")
    ap.add_argument("--profile", action="store_true", help="Print per-clause profile (L12, L13, recurrence)")
    ap.add_argument("--profile-einlang", action="store_true", help="Per-clause time + vectorized/hybrid/scalar summary")
    args = ap.parse_args()

    if args.profile or args.profile_einlang:
        os.environ["EINLANG_PROFILE_STATEMENTS"] = "1"
    if args.profile_einlang:
        os.environ["EINLANG_DEBUG_VECTORIZE"] = "1"

    source = MAIN_EIN.read_text(encoding="utf-8")
    compiler = CompilerDriver()
    runtime = EinlangRuntime()
    result = compiler.compile(source, str(MAIN_EIN), root_path=EXAMPLE_DIR)
    if not result.success:
        err = getattr(result, "tcx", None)
        if err and getattr(err, "reporter", None):
            print(err.reporter.format_all_errors(), file=sys.stderr)
        sys.exit(1)
    exec_result = runtime.execute(result)
    if exec_result.error:
        print(exec_result.error, file=sys.stderr)
        sys.exit(1)

    h = np.asarray(exec_result.outputs.get("h"))
    if h is None or h.ndim != 3:
        print("Expected 3D array h[t,i,j]", file=sys.stderr)
        sys.exit(1)

    write_wave_html(h, args.html)
    print("Wave equation result: shape h =", h.shape)
    print("Open in browser: file://%s" % Path(args.html).resolve())


if __name__ == "__main__":
    main()
