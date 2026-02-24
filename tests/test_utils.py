"""
Test utilities for Einlang test suite.

Provides helpers for the compile-then-execute pattern required
by the architecture where runtime executes IR (IR-only execution).
"""

import math
import numpy as np
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from einlang.compiler.driver import CompilerDriver as EinlangCompiler
from einlang.runtime.runtime import EinlangRuntime, ExecutionResult


@dataclass
class ExecutionResult:
    """Unified execution result for tests (unified test execution result)."""
    value: Any = None
    outputs: Dict[str, Any] = field(default_factory=dict)
    success: bool = False
    error: Optional[str] = None
    errors: List[str] = field(default_factory=list)

    def get_errors(self) -> list:
        return self.errors


def _ir_data_equal(a: Any, b: Any, visited: Optional[set] = None) -> bool:
    """Recursively compare two IR values by data (type + attributes + children)."""
    diff = _ir_data_equal_diff(a, b, "", visited)
    return diff is None


def _ir_data_equal_diff(
    a: Any, b: Any, path: str, visited: Optional[set] = None
) -> Optional[str]:
    """Return None if equal, else a string describing the first difference."""
    if visited is None:
        visited = set()
    if a is b:
        return None
    if a is None or b is None:
        if a != b:
            return "%s: None vs %r" % (path, b if a is None else a)
        return None
    if isinstance(a, (bool, str)):
        if a != b:
            return "%s: %r != %r" % (path, a, b)
        return None
    if isinstance(a, (int, float)):
        if isinstance(b, (int, float)):
            if a != b:
                return "%s: %r != %r" % (path, a, b)
            return None
        return "%s: type mismatch %r vs %r" % (path, type(a).__name__, type(b).__name__)
    if hasattr(a, "item") and hasattr(b, "item") and hasattr(np, "ndarray"):
        if isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
            if a.shape != b.shape or not np.array_equal(a, b):
                return "%s: array shape/content diff" % path
            return None
        if isinstance(a, (np.integer, np.floating)) and isinstance(b, (int, float, np.integer, np.floating)):
            if float(a) != float(b):
                return "%s: %r != %r (numeric)" % (path, a, b)
            return None
    if isinstance(b, (int, float)) and hasattr(a, "item"):
        try:
            if float(a) != float(b):
                return "%s: %r != %r" % (path, a, b)
            return None
        except (TypeError, ValueError):
            return "%s: type %r vs %r" % (path, type(a).__name__, type(b).__name__)
    if type(a) != type(b):
        return "%s: type %s != %s" % (path, type(a).__name__, type(b).__name__)
    if hasattr(a, "krate") and hasattr(a, "index") and hasattr(b, "krate") and hasattr(b, "index"):
        if a.krate != b.krate or a.index != b.index:
            return "%s: DefId %s:%s != %s:%s" % (path, a.krate, a.index, b.krate, b.index)
        return None
    if hasattr(a, "file") and hasattr(a, "line") and hasattr(a, "column"):
        if a.file != b.file or a.line != b.line or a.column != b.column:
            return "%s: Location diff" % path
        return None
    if isinstance(a, (list, tuple)):
        if len(a) != len(b):
            return "%s: len %d != %d" % (path, len(a), len(b))
        for i, (ax, bx) in enumerate(zip(a, b)):
            d = _ir_data_equal_diff(ax, bx, "%s[%d]" % (path, i), visited)
            if d is not None:
                return d
        return None
    if isinstance(a, dict):
        if len(a) != len(b):
            return "%s: dict len %d != %d" % (path, len(a), len(b))
        for k in a:
            bk = next((bkey for bkey in b if _ir_data_equal(k, bkey, visited)), None)
            if bk is None:
                return "%s: key %r only in orig" % (path, k)
            d = _ir_data_equal_diff(a[k], b[bk], "%s[%r]" % (path, k), visited)
            if d is not None:
                return d
        return None
    key = (id(a), id(b))
    if key in visited:
        return None
    visited.add(key)
    try:
        slots = set()
        for cls in type(a).__mro__:
            slots.update(getattr(cls, "__slots__", ()))
        if slots:
            for slot in slots:
                va = getattr(a, slot, None)
                vb = getattr(b, slot, None)
                d = _ir_data_equal_diff(va, vb, "%s.%s" % (path or type(a).__name__, slot), visited)
                if d is not None:
                    return d
            return None
        if hasattr(a, "__dict__") and hasattr(b, "__dict__"):
            return _ir_data_equal_diff(a.__dict__, b.__dict__, "%s.__dict__" % path, visited)
        if a != b:
            return "%s: %r != %r" % (path, a, b)
        return None
    finally:
        visited.discard(key)


def apply_ir_round_trip(compilation_result: Any) -> Any:
    """Replace IR with serialize->deserialize round-trip. Preserves all info used by runtime. Modifies in place."""
    if not getattr(compilation_result, "success", True):
        return compilation_result
    ir = getattr(compilation_result, "ir", None) or getattr(compilation_result, "ir_program", None)
    if ir is None:
        return compilation_result
    from einlang.ir.serialization import serialize_ir, deserialize_ir
    _ser_opts = {"pretty": False, "include_type_info": True, "include_location": True}
    sexpr_str = serialize_ir(ir, **_ser_opts)
    round_tripped = deserialize_ir(sexpr_str)
    if getattr(round_tripped, "defid_to_name", None) is None or not getattr(round_tripped, "defid_to_name", {}):
        d2n = getattr(ir, "defid_to_name", None)
        if d2n:
            round_tripped.defid_to_name = d2n
    if getattr(round_tripped, "source_files", None) is None or not getattr(round_tripped, "source_files", {}):
        sf = getattr(ir, "source_files", None)
        if sf:
            round_tripped.source_files = sf
    _diff = _ir_data_equal_diff(round_tripped, ir, "ProgramIR")
    if _diff is not None:
        import sys
        sys.stderr.write("=== first difference: %s\n" % _diff)
        _orig_r = repr(ir)
        _rt_r = repr(round_tripped)
        _max = 30000
        sys.stderr.write("=== original (repr, first %d chars) ===\n" % _max)
        sys.stderr.write(_orig_r[:_max] + ("\n... truncated\n" if len(_orig_r) > _max else "\n"))
        sys.stderr.write("=== round-tripped (repr, first %d chars) ===\n" % _max)
        sys.stderr.write(_rt_r[:_max] + ("\n... truncated\n" if len(_rt_r) > _max else "\n"))
        _dir = Path("/tmp/einlang_ir_dump")
        _dir.mkdir(parents=True, exist_ok=True)
        (_dir / "ir_round_trip_orig.txt").write_text(_orig_r, encoding="utf-8")
        (_dir / "ir_round_trip_rt.txt").write_text(_rt_r, encoding="utf-8")
        (_dir / "ir_round_trip_sexpr.txt").write_text(sexpr_str, encoding="utf-8")
        _pretty = serialize_ir(ir, **{**_ser_opts, "pretty": True})
        (_dir / "ir_round_trip_sexpr_pretty.txt").write_text(_pretty, encoding="utf-8")
        sys.stderr.write("Full dumps: %s/ir_round_trip_orig.txt and ir_round_trip_rt.txt\n" % _dir)
        sys.stderr.write("IR sexpr: %s/ir_round_trip_sexpr.txt (pretty: ir_round_trip_sexpr_pretty.txt)\n" % _dir)
        assert False, "round-trip must preserve data: %s" % _diff
    orig_stmts = getattr(ir, "statements", None) or []
    rt_stmts = getattr(round_tripped, "statements", None) or []
    for i in range(min(len(orig_stmts), len(rt_stmts))):
        orig, rt = orig_stmts[i], rt_stmts[i]
        orig_defid = getattr(orig, "defid", None)
        rt_binding = getattr(rt, "_binding", None)
        rt_defid = getattr(rt_binding, "defid", None) if rt_binding else getattr(rt, "defid", None)
        if orig_defid is not None:
            assert rt_defid is not None, (
                f"round-trip lost defid for statement {i} (orig defid={orig_defid})"
            )
            assert rt_defid.krate == orig_defid.krate and rt_defid.index == orig_defid.index, (
                f"round-trip defid mismatch for statement {i}: orig={orig_defid} rt={rt_defid}"
            )
    if hasattr(compilation_result, "ir"):
        compilation_result.ir = round_tripped
    if hasattr(compilation_result, "ir_program"):
        compilation_result.ir_program = round_tripped
    return compilation_result


def compile_and_execute(
    source_code: str,
    compiler: EinlangCompiler,
    runtime: EinlangRuntime,
    inputs: Optional[Dict[str, Any]] = None,
    source_file: Optional[str] = None,
    source_overlay: Optional[Dict[tuple, str]] = None,
) -> ExecutionResult:
    """
    Helper to compile source code and execute the resulting IR.

    1. Compiler: source -> AST -> IR
    2. Runtime: IR -> result
    """
    if inputs is None:
        inputs = {}

    source_file_path = source_file if source_file and source_file != "<test>" else "<test>"
    root_path = Path(source_file_path).parent if source_file_path != "<test>" else Path.cwd()
    try:
        result = compiler.compile(
            source_code, source_file_path, root_path=root_path,
            source_overlay=source_overlay or {},
        )
    except TypeError as e:
        err = str(e)
        if "source_overlay" in err or "root_path" in err:
            try:
                result = compiler.compile(source_code, source_file_path, root_path=root_path)
            except TypeError:
                result = compiler.compile(source_code, source_file_path)
        else:
            raise

    if not result.success:
        error_messages = []
        if result.tcx and result.tcx.reporter.has_errors():
            formatted_errors = result.tcx.reporter.format_all_errors()
            error_messages = [formatted_errors] if formatted_errors else []
        return ExecutionResult(
            value=None,
            outputs={},
            success=False,
            error=None,
            errors=error_messages,
        )
    apply_ir_round_trip(result)
    exec_result = runtime.execute(result, inputs=inputs or {})
    error_str = str(exec_result.error) if exec_result.error else None
    return ExecutionResult(
        value=exec_result.value,
        outputs=exec_result.outputs or {},
        success=exec_result.error is None,
        error=error_str,
        errors=[error_str] if error_str else [],
    )


def assert_float_close(
    actual: Union[float, np.ndarray],
    expected: Union[float, np.ndarray],
    rel_tol: float = 1e-6,
    abs_tol: float = 1e-7,
    msg: Optional[str] = None
) -> None:
    """
    Assert that two floats or arrays are approximately equal.

    Uses relative and absolute tolerance for robust float comparison.
    """
    if isinstance(actual, np.ndarray) or isinstance(expected, np.ndarray):
        np.testing.assert_allclose(
            np.asarray(actual),
            np.asarray(expected),
            rtol=rel_tol,
            atol=abs_tol,
            err_msg=msg
        )
    else:
        if not math.isclose(actual, expected, rel_tol=rel_tol, abs_tol=abs_tol):
            error_msg = msg or f"Values not close: actual={actual}, expected={expected}"
            raise AssertionError(error_msg)
