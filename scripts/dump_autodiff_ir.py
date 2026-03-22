#!/usr/bin/env python3
"""Dump IR as an S-expression string after a chosen compiler pass.

Default is immediately after AutodiffPass (high-level Einstein / @ rules still present;
EinsteinLoweringPass has not run yet).

Other useful stops:
  EinsteinLoweringPass   — lowered Einstein clauses (close to NumPy execution)
  IRValidationPass       — full pipeline before codegen use

Per-pass dumps (before/after every pass):
  EINLANG_DUMP_IR_PER_PASS=1 python3 -m einlang your.ein --dump-ir /dev/null
  → writes ir_dumps/NN_before_<Pass>.sexpr and NN_after_<Pass>.sexpr

Examples:
  PYTHONPATH=src python3 scripts/dump_autodiff_ir.py -c 'let x=1.0; let y=x*x; print(@y);' \\
    --stop-after AutodiffPass -o /tmp/ad.sexpr

  PYTHONPATH=src python3 scripts/dump_autodiff_ir.py path/to/main.ein --stop-after EinsteinLoweringPass
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> int:
    repo = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(repo / "src"))

    from einlang.compiler.driver import CompilerDriver
    from einlang.ir.serialization import serialize_ir

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("file", nargs="?", help=".ein file (omit with -c)")
    parser.add_argument("-c", "--code", metavar="SRC", help="Einlang source string")
    parser.add_argument(
        "--stop-after",
        default="AutodiffPass",
        metavar="PASS",
        help="Pass class name to stop after (default: AutodiffPass)",
    )
    parser.add_argument("-o", "--output", type=Path, default=None, help="Write S-expr here (default: stdout)")
    parser.add_argument("--root", type=Path, default=repo, help="Project root for modules (default: repo root)")
    args = parser.parse_args()

    if args.code is not None:
        source = args.code
        source_file = "<dump_autodiff_ir>"
    elif args.file is not None:
        p = Path(args.file)
        source = p.read_text(encoding="utf-8")
        source_file = str(p.resolve())
    else:
        parser.error("need a .ein file or -c SRC")

    compiler = CompilerDriver()
    result = compiler.compile(
        source.strip(),
        source_file=source_file,
        root_path=args.root.resolve(),
        stop_after_pass=args.stop_after,
    )
    if not result.success:
        sys.stderr.write(result.get_errors() or "compile failed\n")
        return 1
    if result.ir is None:
        sys.stderr.write("compile returned no IR\n")
        return 1

    text = serialize_ir(result.ir)
    if args.output is not None:
        out = args.output.resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
        sys.stderr.write(f"wrote {len(text)} chars to {out}\n")
    else:
        sys.stdout.write(text)
        if not text.endswith("\n"):
            sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
