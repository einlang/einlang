"""Minimal compile helper for ``print(@…)`` tests (pytest imports this only)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from einlang.compiler.driver import CompilerDriver
from einlang.ir.nodes import BuiltinCallIR, LiteralIR

from tests.print_at_calculus_catalog import GOLDEN_CALCULUS

REPO_ROOT = Path(__file__).resolve().parent.parent


def short_err_print_at(obj: object, limit: int = 600) -> str:
    if obj is None:
        return ""
    s = str(obj)
    if len(s) > limit:
        return s[:limit] + "..."
    return s


def compile_capture_rewritten_print_at(
    source: str,
    compiler: Optional[CompilerDriver] = None,
) -> Tuple[bool, str, str]:
    """Compile ``print(@...)`` source and return the rewritten literal print payload.

    Returns ``(compile_ok, printed_text, err)``.
    """
    comp = compiler if compiler is not None else CompilerDriver()
    result = comp.compile(source.strip(), source_file="<test>", root_path=REPO_ROOT)
    if not result.success:
        return False, "", short_err_print_at(result.get_errors())

    statements = getattr(result.ir, "statements", ()) or ()
    for stmt in reversed(statements):
        if not isinstance(stmt, BuiltinCallIR):
            continue
        if stmt.builtin_name != "print":
            continue
        args = stmt.args or ()
        if len(args) != 1 or not isinstance(args[0], LiteralIR) or not isinstance(args[0].value, str):
            return False, "", short_err_print_at("expected rewritten print(@...) literal string")
        return True, args[0].value, ""

    return False, "", short_err_print_at("rewritten print(@...) call not found")
