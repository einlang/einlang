"""Minimal compile/exec helper for ``print(@…)`` tests (pytest imports this only)."""

from __future__ import annotations

from io import StringIO
from pathlib import Path
import sys
from typing import Tuple

from einlang.compiler.driver import CompilerDriver
from einlang.runtime.runtime import EinlangRuntime

REPO_ROOT = Path(__file__).resolve().parent.parent


def short_err_print_at(obj: object, limit: int = 600) -> str:
    if obj is None:
        return ""
    s = str(obj)
    if len(s) > limit:
        return s[:limit] + "..."
    return s


def compile_exec_capture_print_at(source: str) -> Tuple[bool, bool, str, str]:
    """Compile ``source``, run with numpy backend, return ``(compile_ok, exec_ok, stdout.strip(), err)``."""
    compiler = CompilerDriver()
    result = compiler.compile(source.strip(), source_file="<test>", root_path=REPO_ROOT)
    if not result.success:
        return False, False, "", short_err_print_at(result.get_errors())
    runtime = EinlangRuntime(backend="numpy")
    buf = StringIO()
    old_stdout = sys.stdout
    sys.stdout = buf
    try:
        exec_result = runtime.execute(result)
    finally:
        sys.stdout = old_stdout
    if not exec_result.success:
        err = getattr(exec_result, "error", None) or exec_result.errors or "exec failed"
        return True, False, "", short_err_print_at(err)
    return True, True, buf.getvalue().strip(), ""
