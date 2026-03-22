"""CLI for ``print(@…)`` diagnostics (goldens in ``tests/unit/test_print_at_golden.py``).

**New golden:** Run ``compile_exec_capture_print_at`` on the program, optionally
``python3 scripts/dump_autodiff_ir.py -c '…' --autodiff-only``, then add
``(label, source, expected)`` to ``GOLDEN_PRINT_CASES`` in ``tests/unit/test_print_at_golden.py``.

Usage:
  python3 -m pytest tests/unit/test_print_at_golden.py -q   # golden stdout checks
  python3 -m pytest tests/unit/test_print_at_ml_smoke.py -q # std::ml activation goldens
  python3 scripts/test_print_at.py                    # same goldens (CLI)
  python3 scripts/test_print_at.py --help
"""
from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(_REPO))

from tests.print_at_fixtures import compile_exec_capture_print_at  # noqa: E402
from tests.unit.test_print_at_golden import GOLDEN_PRINT_CASES  # noqa: E402

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
    if argv:
        print(f"Unknown arguments: {argv!r}", file=sys.stderr)
        sys.exit(2)

    golden_tests()
    print()
    if failures:
        print(f"{len(failures)} FAILED:")
        for label, msg in failures:
            print(f"  {label}: {msg}")
        sys.exit(1)
    print("All passed.")
    sys.exit(0)


if __name__ == "__main__":
    main()
