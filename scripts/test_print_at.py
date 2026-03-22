"""CLI for ``print(@…)`` diagnostics (goldens live in ``tests/print_at_fixtures.py``).

**New golden:** Run ``compile_exec_capture_print_at`` on the program, compare stdout to calculus /
``GOLDEN_CALCULUS``, optionally ``python3 scripts/dump_autodiff_ir.py -c '…' --autodiff-only``, then add
``(label, source, expected)`` to ``GOLDEN_PRINT_CASES`` and an entry to ``GOLDEN_CALCULUS`` in
``tests/print_at_fixtures.py``.

Usage:
  python3 -m pytest tests/unit/test_print_at.py -q   # golden stdout checks
  python3 scripts/test_print_at.py                    # same goldens (CLI)
  python3 scripts/test_print_at.py --report         # markdown table vs math refs
  python3 scripts/test_print_at.py --help

``--report`` alone prints the table and exits (status 1 if any golden mismatch).

Report focus: **compile → exec ok →** compare printed tangents to the math reference (structural fit,
not symbolic proof). ``?`` inside shape prints like ``[f32; ?, ?]`` is ignored for the stray-``?`` check.
"""
from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(_REPO))

from tests.print_at_fixtures import (  # noqa: E402
    GOLDEN_PRINT_CASES,
    compile_exec_capture_print_at,
    run_report_to_stdout,
)

failures: list[tuple[str, str]] = []


def run(label: str, source: str, expected: str) -> None:
    c_ok, e_ok, printed, err = compile_exec_capture_print_at(source)
    if not c_ok:
        failures.append((label, f"COMPILE FAIL: {err}"))
        print(f"  FAIL  {label}: COMPILE FAIL")
        return
    if not e_ok:
        failures.append((label, f"EXEC FAIL: {err}"))
        print(f"  FAIL  {label}: EXEC FAIL")
        return
    if printed == expected:
        print(f"  OK    {label}: {printed}")
    else:
        failures.append((label, f"got '{printed}', expected '{expected}'"))
        print(f"  FAIL  {label}: got '{printed}', expected '{expected}'")


def golden_tests() -> None:
    failures.clear()
    for label, source, expected in GOLDEN_PRINT_CASES:
        run(label, source, expected)


def main() -> None:
    argv = sys.argv[1:]
    if "-h" in argv or "--help" in argv:
        print(__doc__)
        sys.exit(0)

    report_code = 0
    argv_set = set(argv)
    if "--report" in argv_set:
        report_code = run_report_to_stdout()
        print()
        if argv_set <= {"--report"}:
            sys.exit(report_code)

    golden_tests()
    print()
    if failures:
        print(f"{len(failures)} FAILED:")
        for label, msg in failures:
            print(f"  {label}: {msg}")
        sys.exit(1)
    print("All passed.")
    sys.exit(report_code)


if __name__ == "__main__":
    main()
