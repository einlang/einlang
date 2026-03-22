"""CLI for ``print(@…)`` diagnostics (goldens in ``tests/unit/test_print_at_golden.py``).

**New golden:** Run ``compile_exec_capture_print_at`` on the program, optionally
``python3 scripts/dump_autodiff_ir.py -c '…' --autodiff-only``, then add
``(label, source, expected)`` to ``GOLDEN_PRINT_CASES`` in ``tests/unit/test_print_at_golden.py``.

Calculus reference strings live in ``tests/print_at_calculus_catalog.py`` (``GOLDEN_CALCULUS``),
re-exported from ``tests/print_at_fixtures.py``.

Usage:
  python3 -m pytest tests/unit/test_print_at_golden.py -q   # golden stdout checks
  python3 -m pytest tests/unit/test_print_at_ml_smoke.py -q # std::ml activation goldens
  python3 scripts/test_print_at.py                    # same goldens (CLI)
  python3 scripts/test_print_at.py --report           # markdown: calculus vs captured stdout
  python3 scripts/test_print_at.py --print-audit      # calculus line + expected print(@…) stdout (all cases)
  python3 scripts/test_print_at.py --write-formatted-tables  # tests/print_at_formatted_table*.md from goldens
  python3 scripts/test_print_at.py --help
"""
from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(_REPO))

from tests.print_at_fixtures import (  # noqa: E402
    GOLDEN_CALCULUS,
    compile_exec_capture_print_at,
)
from tests.unit.test_print_at_golden import GOLDEN_PRINT_CASES  # noqa: E402
from tests.unit.test_print_at_ml_smoke import (  # noqa: E402
    ML_ACTIVATION_PRINT_AT_GOLDEN_CASES,
)

failures: list[tuple[str, str]] = []


def _md_cell(s: str) -> str:
    return s.replace("|", "\\|").replace("\n", " ").strip()


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


def report_markdown() -> None:
    def row(label: str, source: str, expected: str, section: str) -> None:
        calc = GOLDEN_CALCULUS.get(label, "")
        c_ok, e_ok, printed, err = compile_exec_capture_print_at(source)
        if not c_ok:
            ok = "compile fail"
            cap = _md_cell(err or "")
        elif not e_ok:
            ok = "exec fail"
            cap = _md_cell(err or "")
        elif printed == expected:
            ok = "yes"
            cap = _md_cell(printed[:200] + ("…" if len(printed) > 200 else ""))
        else:
            ok = "no"
            cap = _md_cell(printed[:120] + ("…" if len(printed) > 120 else ""))
        print(
            f"| {section} | {_md_cell(label)} | {_md_cell(ok)} | {_md_cell(calc)} | {_md_cell(cap)} |"
        )

    print("# print(@…) vs calculus catalog\n")
    print("| suite | label | matches golden | calculus (reference) | captured (truncated) |")
    print("| --- | --- | --- | --- | --- |")
    for label, source, expected in GOLDEN_PRINT_CASES:
        row(label, source, expected, "golden")
    for label, source, expected in ML_ACTIVATION_PRINT_AT_GOLDEN_CASES:
        row(label, source, expected, "ml_activation")


def print_audit_table() -> None:
    def block(label: str, calculus: str, printed: str) -> None:
        print("=== %s ===" % label)
        print(calculus)
        print(printed)
        print()

    for label, source, expected in GOLDEN_PRINT_CASES:
        block(label, GOLDEN_CALCULUS.get(label, ""), expected)
    for label, source, expected in ML_ACTIVATION_PRINT_AT_GOLDEN_CASES:
        block(label, GOLDEN_CALCULUS.get(label, ""), expected)


def write_formatted_tables() -> None:
    """Regenerate ``tests/print_at_formatted_table.md`` and ``tests/print_at_formatted_table_full.md``."""

    def esc(s: str) -> str:
        return s.replace("|", "\\|")

    rows = list(GOLDEN_PRINT_CASES) + list(ML_ACTIVATION_PRINT_AT_GOLDEN_CASES)
    trunc_limit = 120

    lines_full: list[str] = [
        "| label | calculus | expected_einlang (full, one line) |",
        "| :--- | :--- | :--- |",
    ]
    lines_trunc: list[str] = [
        "# Expected print(@…) goldens vs calculus reference",
        "",
        "Full multiline expected: `python3 scripts/test_print_at.py --print-audit`",
        "",
        "| label | calculus | expected_einlang (one line, truncated) |",
        "| :--- | :--- | :--- |",
    ]
    for label, _source, expected in rows:
        calc = GOLDEN_CALCULUS.get(label, "")
        one_line = expected.replace("\n", " ").strip()
        exp_full = esc(one_line)
        if len(one_line) > trunc_limit:
            exp_trunc = esc(one_line[: trunc_limit - 3] + "...")
        else:
            exp_trunc = exp_full
        row_full = f"| {esc(label)} | {esc(calc)} | {exp_full} |"
        row_trunc = f"| {esc(label)} | {esc(calc)} | {exp_trunc} |"
        lines_full.append(row_full)
        lines_trunc.append(row_trunc)

    out_full = _REPO / "tests" / "print_at_formatted_table_full.md"
    out_trunc = _REPO / "tests" / "print_at_formatted_table.md"
    out_full.write_text("\n".join(lines_full) + "\n", encoding="utf-8")
    out_trunc.write_text("\n".join(lines_trunc) + "\n", encoding="utf-8")
    print(f"Wrote {out_full.relative_to(_REPO)}")
    print(f"Wrote {out_trunc.relative_to(_REPO)}")


def main() -> None:
    argv = sys.argv[1:]
    if "-h" in argv or "--help" in argv:
        print(__doc__)
        sys.exit(0)
    if argv == ["--report"]:
        report_markdown()
        sys.exit(0)
    if argv == ["--print-audit"]:
        print_audit_table()
        sys.exit(0)
    if argv == ["--write-formatted-tables"]:
        write_formatted_tables()
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
