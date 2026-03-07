"""CLI entry point: run `einlang file.ein` or `python -m einlang file.ein`."""

import sys
from pathlib import Path


def main() -> int:
    import argparse
    from .compiler.driver import CompilerDriver
    from .runtime.runtime import EinlangRuntime

    parser = argparse.ArgumentParser(prog="einlang", description="Run an Einlang (.ein) file.")
    parser.add_argument("file", type=Path, help="Path to .ein source file")
    parser.add_argument("--backend", default="numpy", help="Execution backend (default: numpy)")
    parser.add_argument("--profile-lines", type=int, default=0, metavar="N", help="Profile by source line buckets (e.g. 10 for L0-L10, L10-L20, ...)")
    parser.add_argument("--profile-statements", action="store_true", help="Profile each top-level statement separately (reset buckets per statement)")
    parser.add_argument("--debug-vectorize", action="store_true", help="Print [vectorized] or [scalar] per Einstein clause")
    parser.add_argument("--profile-functions", action="store_true", help="Print runtime per Einlang function (e.g. encode, encoder_block)")
    parser.add_argument("--profile-reductions", action="store_true", help="Print reduction path per sum/max/min: matmul, vectorized, or scalar (with source line)")
    parser.add_argument("--cprofile", action="store_true", help="Run execution under cProfile and print stats")
    parser.add_argument("--cprofile-out", type=Path, default=None, metavar="FILE", help="Write cProfile stats to FILE (for snakeviz, etc.)")
    parser.add_argument("--parallel-clauses", action="store_true", help="Run independent Einstein clauses in parallel (BLAS/OpenMP set to 1 thread per worker to avoid oversubscription)")
    parser.add_argument("--parallel-clauses-workers", type=int, default=4, metavar="N", help="Max workers for --parallel-clauses (default: 4)")
    args = parser.parse_args()
    import os
    if args.parallel_clauses:
        os.environ["EINLANG_PARALLEL_CLAUSES"] = "1"
        os.environ["EINLANG_PARALLEL_CLAUSES_WORKERS"] = str(args.parallel_clauses_workers)
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

    path = args.file.resolve()
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

    root_path = path.parent
    if str(root_path) not in sys.path:
        sys.path.insert(0, str(root_path))
    compiler = CompilerDriver()
    result = compiler.compile(source, str(path), root_path=root_path)

    if not result.success:
        if result.tcx and getattr(result.tcx, "reporter", None) and result.tcx.reporter.has_errors():
            sys.stderr.write(result.tcx.reporter.format_all_errors())
        else:
            sys.stderr.write("einlang: compilation failed\n")
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
