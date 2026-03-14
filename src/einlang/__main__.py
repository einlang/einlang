"""CLI entry point: run `einlang file.ein`, `einlang -c "code"`, or `einlang -` (stdin)."""

import sys
from pathlib import Path


def main() -> int:
    import argparse
    from .compiler.driver import CompilerDriver
    from .runtime.runtime import EinlangRuntime

    parser = argparse.ArgumentParser(
        prog="einlang",
        description="Run an Einlang (.ein) file, inline code (-c), or stdin.",
    )
    parser.add_argument(
        "file",
        nargs="?",
        default=None,
        help="Path to .ein file, or '-' to read from stdin",
    )
    parser.add_argument(
        "-c",
        dest="code",
        metavar="CODE",
        help="Run CODE as Einlang source (inline, like python -c)",
    )
    parser.add_argument("--backend", default="numpy", help="Execution backend (default: numpy)")
    parser.add_argument("--root", type=Path, default=None, metavar="DIR", help="Project root for module resolution (default: directory of file or cwd)")
    parser.add_argument("--stdlib", type=Path, default=None, metavar="DIR", help="Stdlib root (default: search up from --root). EINLANG_STDLIB env overrides if set.")
    parser.add_argument("--profile-lines", type=int, default=0, metavar="N", help="Profile by source line buckets (e.g. 10 for L0-L10, L10-L20, ...)")
    parser.add_argument("--profile-statements", action="store_true", help="Profile each top-level statement separately (reset buckets per statement)")
    parser.add_argument("--debug-vectorize", action="store_true", help="Print [vectorized] or [scalar] per Einstein clause")
    parser.add_argument("--profile-functions", action="store_true", help="Print runtime per Einlang function (e.g. encode, encoder_block)")
    parser.add_argument("--profile-blocks", action="store_true", help="Print runtime per block expression (e.g. LSTM gate body)")
    parser.add_argument("--profile-reductions", action="store_true", help="Print reduction path per sum/max/min: matmul, vectorized, or scalar (with source line)")
    parser.add_argument("--cprofile", action="store_true", help="Run execution under cProfile and print stats")
    parser.add_argument("--cprofile-out", type=Path, default=None, metavar="FILE", help="Write cProfile stats to FILE (for snakeviz, etc.)")
    parser.add_argument("--dump-ir", type=Path, default=None, metavar="FILE", help="After compile, dump IR S-expr to FILE (default: <source_dir>/ir_dump.sexpr)")
    args = parser.parse_args()
    import os
    if args.profile_lines > 0:
        os.environ["EINLANG_PROFILE_LINES"] = str(args.profile_lines)
    if args.profile_statements:
        os.environ["EINLANG_PROFILE_STATEMENTS"] = "1"
    if args.profile_functions:
        os.environ["EINLANG_PROFILE_FUNCTIONS"] = "1"
    if args.profile_reductions:
        os.environ["EINLANG_PROFILE_REDUCTIONS"] = "1"
    if args.debug_vectorize:
        os.environ["EINLANG_DEBUG_VECTORIZE"] = "1"

    if args.profile_functions or args.profile_statements or args.profile_blocks or args.profile_lines or args.profile_reductions:
        sys.stdout.flush()
        sys.stderr.flush()

    # Resolve source: -c CODE (inline), - or stdin, or file path
    if args.code is not None:
        source = args.code
        source_file = "<inline>"
        root_path = Path.cwd()
    elif args.file == "-" or (args.file is None and not sys.stdin.isatty()):
        try:
            source = sys.stdin.read()
        except Exception as e:
            sys.stderr.write(f"einlang: error: could not read stdin: {e}\n")
            return 1
        source_file = "<stdin>"
        root_path = Path.cwd()
    elif args.file is None:
        sys.stderr.write("einlang: error: need a file path, '-c CODE', or pipe source to stdin (e.g. einlang -)\n")
        parser.print_help(sys.stderr)
        return 1
    else:
        path = Path(args.file).resolve()
        if not path.exists():
            sys.stderr.write(f"einlang: error: file not found: {path}\n")
            return 1
        if not path.is_file():
            sys.stderr.write(f"einlang: error: not a file: {path}\n")
            return 1
        try:
            source = path.read_text(encoding="utf-8")
        except Exception as e:
            sys.stderr.write(f"einlang: error: could not read file: {e}\n")
            return 1
        source_file = str(path)
        root_path = path.parent

    if args.root is not None:
        root_path = args.root.resolve()
    stdlib_root = None
    if os.environ.get("EINLANG_STDLIB"):
        stdlib_root = Path(os.environ["EINLANG_STDLIB"]).resolve()
    elif args.stdlib is not None:
        stdlib_root = args.stdlib.resolve()

    if str(root_path) not in sys.path:
        sys.path.insert(0, str(root_path))
    compiler = CompilerDriver()
    result = compiler.compile(source, source_file, root_path=root_path, stdlib_root=stdlib_root)

    if not result.success:
        if result.tcx and result.tcx.reporter.has_errors():
            sys.stderr.write(result.tcx.reporter.format_all_errors())
        else:
            sys.stderr.write("einlang: compilation failed\n")
        return 1

    if args.dump_ir is not None:
        from .ir.serialization import serialize_ir
        out_path = args.dump_ir if args.dump_ir.suffix else Path(str(args.dump_ir) + ".sexpr")
        out_path = out_path.resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        ir = result.ir
        if ir is not None:
            out_path.write_text(serialize_ir(ir), encoding="utf-8")
            sys.stderr.write(f"IR dumped to {out_path}\n")
        else:
            sys.stderr.write("einlang: no IR on result (compile did not return ir)\n")
            return 1

    runtime = EinlangRuntime(backend=args.backend)

    if args.cprofile:
        import cProfile
        import pstats
        prof = cProfile.Profile()
        prof.enable()
        exec_result = runtime.execute(result, inputs={})
        prof.disable()
        if args.cprofile_out:
            prof.dump(str(args.cprofile_out))
        stats = pstats.Stats(prof)
        stats.sort_stats(pstats.SortKey.CUMULATIVE)
        stats.print_stats(60)
        if not args.cprofile_out:
            stats.print_callers(20)
    else:
        exec_result = runtime.execute(result, inputs={})

    if exec_result.error is not None:
        sys.stderr.write(f"einlang: runtime error: {exec_result.error}\n")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
