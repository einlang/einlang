#!/usr/bin/env python3
"""Compile an .ein file and print S-expression IR to stdout (for debugging).

Examples:
  PYTHONPATH=src python3 scripts/dump_compiled_ir.py examples/run_numerics.ein
  PYTHONPATH=src python3 scripts/dump_compiled_ir.py examples/run_numerics.ein -o ir_dumps/run_numerics.sexpr
  PYTHONPATH=src python3 scripts/dump_compiled_ir.py examples/run_numerics.ein --head 120
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent


def main() -> None:
    p = argparse.ArgumentParser(description="Dump compiled program IR as S-expressions.")
    p.add_argument("ein_file", type=Path, help="Path to .ein source (relative to repo ok)")
    p.add_argument(
        "--head",
        type=int,
        metavar="N",
        help="Print only first N lines (default: full output)",
    )
    p.add_argument("--types", action="store_true", help="Include type_info in output")
    p.add_argument("--loc", action="store_true", help="Include source locations")
    p.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        metavar="PATH",
        help="Write full S-expression to this file (default: stdout)",
    )
    args = p.parse_args()

    src_path = args.ein_file.resolve()
    if not src_path.is_file():
        print(f"error: not a file: {src_path}", file=sys.stderr)
        sys.exit(1)

    root_path = src_path.parent
    source = src_path.read_text(encoding="utf-8")

    sys.path.insert(0, str(_REPO / "src"))
    from einlang.compiler.driver import CompilerDriver
    from einlang.ir.serialization import serialize_ir

    r = CompilerDriver().compile(source, source_file=str(src_path), root_path=root_path)
    if not r.success:
        for e in r.get_errors() or []:
            print(e, file=sys.stderr)
        sys.exit(1)

    out = serialize_ir(
        r.ir,
        include_location=args.loc,
        include_type_info=args.types,
        pretty=True,
    )
    lines = out.splitlines()
    if args.head is not None:
        lines = lines[: args.head]
        if len(out.splitlines()) > args.head:
            lines.append(f"... ({len(out.splitlines()) - args.head} more lines)")
    text = "\n".join(lines)
    if args.output is not None:
        out_path = args.output.resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text, encoding="utf-8")
        print(f"wrote {len(text)} chars to {out_path}", file=sys.stderr)
    else:
        print(text)


if __name__ == "__main__":
    main()
