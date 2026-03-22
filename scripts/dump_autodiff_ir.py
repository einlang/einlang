#!/usr/bin/env python3
"""Dump IR as an S-expression string after a chosen compiler pass.

**Autodiff IR (default stop):** ``--stop-after AutodiffPass`` (default) freezes the program right after
``AutodiffPass``: ``@`` / ``_@`` tangents and Jacobian-style bindings are still high-level Einstein IR;
``EinsteinLoweringPass`` has not run.

**Slice to autodiff-only bindings:** ``--autodiff-only`` keeps only bindings that look compiler- or
Jacobian-generated (``_@…``, user ``@…`` tangent names, or ``d*_*`` / ``d*_d_*`` names). Primal ``let``s
are dropped so the sexpr is smaller for reviewing AD output.

Other useful stops:
  EinsteinLoweringPass   — lowered Einstein clauses (close to NumPy execution)
  IRValidationPass       — full pipeline before codegen use

Per-pass dumps (before/after every pass):
  EINLANG_DUMP_IR_PER_PASS=1 python3 -m einlang your.ein --dump-ir /dev/null
  → writes ir_dumps/NN_before_<Pass>.sexpr and NN_after_<Pass>.sexpr

Examples:
  python3 scripts/dump_autodiff_ir.py -c 'let x=1.0; let y=x*x; print(@y);' -o /tmp/ad_full.sexpr

  python3 scripts/dump_autodiff_ir.py -c 'let x=1.0; let y=x*x; print(@y);' \\
    --autodiff-only -o /tmp/ad_slice.sexpr

  python3 scripts/dump_autodiff_ir.py path/to/main.ein --stop-after EinsteinLoweringPass

**Debug std::ml::matmul (2D):** compare callee body after autodiff vs a minimal working program::

  python3 scripts/dump_autodiff_ir.py -c 'use std::ml; let A=[[1.0,2.0],[3.0,4.0]]; let B=[[5.0,6.0],[7.0,8.0]]; let C=std::ml::matmul(A,B); print(@C);' \\
    --autodiff-only -o /tmp/matmul_ad.sexpr

``matmul`` is 2D-only in ``stdlib/ml/linalg_ops.ein``; batched use ``batch_matmul``. Debug batched
``print(@C)`` tangents, e.g.:
``python3 scripts/dump_autodiff_ir.py -c 'use std::ml; let A=[[[1.0,2.0]]]; let B=[[[1.0,1.0]]]; let C=std::ml::batch_matmul(A,B); print(@C);' --autodiff-only``
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> int:
    repo = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(repo / "src"))

    from einlang.compiler.driver import CompilerDriver
    from einlang.ir.nodes import BindingIR, ProgramIR
    from einlang.ir.serialization import serialize_ir
    from einlang.passes.autodiff import _is_diff_name

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("file", nargs="?", help=".ein file (omit with -c)")
    parser.add_argument("-c", "--code", metavar="SRC", help="Einlang source string")
    parser.add_argument(
        "--stop-after",
        default="AutodiffPass",
        metavar="PASS",
        help="Pass class name to stop after (default: AutodiffPass)",
    )
    parser.add_argument(
        "--autodiff-only",
        action="store_true",
        help="Serialize only autodiff-related bindings (_@…, @ tangent names, d*_d_* / d*_* Jacobian names)",
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

    program = result.ir
    if args.autodiff_only:

        def _binding_is_autodiff_slice(b: BindingIR) -> bool:
            n = (b.name or "").strip()
            if not n:
                return False
            if _is_diff_name(n):
                return True
            # Jacobian / partial names from @y/@x lowering (e.g. dC_dA, dy_dx)
            if n.startswith("d") and "_" in n:
                return True
            return False

        kept = [b for b in (program.bindings or []) if isinstance(b, BindingIR) and _binding_is_autodiff_slice(b)]
        if not kept:
            sys.stderr.write(
                "warning: --autodiff-only matched no bindings; output is an empty (program :bindings ())\n"
            )
        program = ProgramIR(
            statements=list(kept),
            source_files=program.source_files,
            modules=program.modules,
        )

    text = serialize_ir(program)
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
