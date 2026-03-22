"""Golden stdout checks for ``print(@…)`` symbolic tangents."""

from __future__ import annotations

import pytest

from tests.print_at_fixtures import GOLDEN_PRINT_CASES, compile_exec_capture_print_at


@pytest.mark.parametrize(
    "label,source,expected",
    [(row[0], row[1], row[2]) for row in GOLDEN_PRINT_CASES],
    ids=[row[0] for row in GOLDEN_PRINT_CASES],
)
def test_golden_print_at_stdout(label: str, source: str, expected: str) -> None:
    """Each ``GOLDEN_PRINT_CASES`` row: compile, run, ``print(@…)`` stdout must match exactly."""
    c_ok, e_ok, out, err = compile_exec_capture_print_at(source)
    assert c_ok, "%s: compile failed: %s" % (label, err)
    assert e_ok, "%s: exec failed: %s" % (label, err)
    assert out == expected, "%s: got %r, expected %r" % (label, out, expected)
