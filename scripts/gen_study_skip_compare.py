#!/usr/bin/env python3
"""Regenerate docs/TEST_PRINT_AT_STUDY_SKIP_COMPARE.md from live compile/exec of STUDY_SKIP_CASES."""
from __future__ import annotations

import contextlib
import importlib.util
import sys
from io import StringIO
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from einlang.compiler.driver import CompilerDriver
from einlang.runtime.runtime import EinlangRuntime


def _load_test_print_at():
    spec = importlib.util.spec_from_file_location("test_print_at", REPO / "scripts" / "test_print_at.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def run_case(source: str) -> tuple[str, str, str]:
    """Returns (status, printed, detail). status is compile_fail | exec_fail | ok."""
    cd = CompilerDriver()
    r = cd.compile(source.strip(), source_file="<test>", root_path=REPO)
    if not r.success:
        err = str(r.get_errors() or "")
        first = err.split("\n")[0][:120]
        return "compile_fail", "", first
    buf = StringIO()
    with contextlib.redirect_stdout(buf):
        ex = EinlangRuntime(backend="numpy").execute(r)
    out = buf.getvalue().strip()
    if not ex.success:
        err = str(getattr(ex, "error", None) or ex.errors or "")
        first = err.split("\n")[0][:140]
        return "exec_fail", out, first
    return "ok", out, ""


def main() -> None:
    tpa = _load_test_print_at()
    cases = tpa.STUDY_SKIP_CASES
    math = tpa.STUDY_MATH_REFERENCE

    rows: list[tuple[str, str, str, str, str, str]] = []
    n_compile = n_exec = n_ok = 0
    for label, source, skip_reason in cases:
        status, printed, detail = run_case(source)
        if status == "compile_fail":
            n_compile += 1
        elif status == "exec_fail":
            n_exec += 1
        else:
            n_ok += 1
        note = detail
        if printed and status == "exec_fail":
            p1 = printed.replace("\n", "\\n")
            if len(p1) > 80:
                p1 = p1[:77] + "…"
            note = f"stdout: {p1!r} | {detail}"
        rows.append((label, skip_reason, math.get(label, ""), status, printed, note))

    lines: list[str] = [
        "# Study-skip cases: expected math vs actual result",
        "",
        "Programs from [`scripts/test_print_at.py`](../scripts/test_print_at.py) (`STUDY_SKIP_CASES`).",
        "",
        "**Regenerate:** `python3 scripts/gen_study_skip_compare.py`",
        "",
        "## Summary",
        "",
        f"- **COMPILE FAIL:** {n_compile}",
        f"- **COMPILE OK, EXEC FAIL:** {n_exec}",
        f"- **EXEC OK (got symbolic print):** {n_ok}",
        "",
        "Symbolic `print(@y)` output is only comparable to the math reference when compile and exec both succeed.",
        "",
        "## Comparison table",
        "",
        "| Case | Pytest skip reason | Expected math (reference) | Outcome | Actual (error or print) |",
        "|------|--------------------|-----------------------------|---------|-------------------------|",
    ]

    def esc(cell: str, max_len: int) -> str:
        s = cell.replace("|", "\\|").replace("\n", " ")
        if len(s) > max_len:
            return s[: max_len - 1] + "…"
        return s

    for label, skip_reason, mref, status, _printed, note in rows:
        if status == "compile_fail":
            outcome = "**COMPILE FAIL**"
        elif status == "exec_fail":
            outcome = "**EXEC FAIL**"
        else:
            outcome = "**OK**"
        lines.append(
            f"| `{label}` | {esc(skip_reason, 48)} | {esc(mref, 72)} | {outcome} | {esc(note, 88)} |"
        )

    lines.extend(
        [
            "",
            "## Full expected math (same as `STUDY_MATH_REFERENCE`)",
            "",
        ]
    )
    for label, _, mref, _, _, _ in rows:
        if mref:
            lines.append(f"- **`{label}`:** {mref}")
        else:
            lines.append(f"- **`{label}`:** (no reference)")

    lines.extend(
        [
            "",
            "## See also",
            "",
            "- [TEST_PRINT_AT_STUDY_SKIP_DUMP.md](TEST_PRINT_AT_STUDY_SKIP_DUMP.md)",
            "- [PRINT_DIFFERENTIAL.md](PRINT_DIFFERENTIAL.md)",
            "",
        ]
    )

    out_path = REPO / "docs" / "TEST_PRINT_AT_STUDY_SKIP_COMPARE.md"
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("Wrote", out_path)


if __name__ == "__main__":
    main()
