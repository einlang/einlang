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
    args = parser.parse_args()

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
    exec_result = runtime.execute(result, inputs={})

    if exec_result.error is not None:
        sys.stderr.write(f"einlang: runtime error: {exec_result.error}\n")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
