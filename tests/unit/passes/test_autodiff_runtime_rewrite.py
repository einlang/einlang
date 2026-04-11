from __future__ import annotations

import numpy as np

from einlang.compiler.driver import CompilerDriver
from einlang.ir.nodes import BindingIR, BuiltinCallIR, DifferentialIR, FunctionValueIR, IRNode
from einlang.passes.autodiff import AutodiffPass
from einlang.shared.autodiff_intrinsics import AutodiffBuiltinKind
from einlang.runtime.runtime import EinlangRuntime


def _compile_analysis(source: str):
    result = CompilerDriver().compile(
        source.strip(),
        source_file="<autodiff_runtime_rewrite>",
        stop_after_pass="AutodiffPass",
    )
    assert result.success, result.get_errors()
    assert result.tcx is not None
    return result.tcx.get_analysis(AutodiffPass)


def _binding_by_name(binding_map, name: str) -> BindingIR:
    for binding in (binding_map or {}).values():
        if isinstance(binding, BindingIR) and binding.name == name:
            return binding
    raise AssertionError(f"binding {name!r} not found")


def _contains_node_type(node: object, needle: type) -> bool:
    seen = set()
    stack = [node]
    while stack:
        cur = stack.pop()
        if cur is None:
            continue
        oid = id(cur)
        if oid in seen:
            continue
        seen.add(oid)
        if isinstance(cur, needle):
            return True
        if isinstance(cur, dict):
            stack.extend(cur.keys())
            stack.extend(cur.values())
            continue
        if isinstance(cur, (list, tuple)):
            stack.extend(cur)
            continue
        if isinstance(cur, IRNode):
            for cls in type(cur).__mro__:
                for slot in getattr(cls, "__slots__", ()):
                    stack.append(getattr(cur, slot, None))
    return False


def _builtin_names(node: object) -> set[str]:
    out: set[str] = set()
    seen = set()
    stack = [node]
    while stack:
        cur = stack.pop()
        if cur is None:
            continue
        oid = id(cur)
        if oid in seen:
            continue
        seen.add(oid)
        if isinstance(cur, BuiltinCallIR):
            out.add(cur.builtin_name)
        if isinstance(cur, dict):
            stack.extend(cur.keys())
            stack.extend(cur.values())
            continue
        if isinstance(cur, (list, tuple)):
            stack.extend(cur)
            continue
        if isinstance(cur, IRNode):
            for cls in type(cur).__mro__:
                for slot in getattr(cls, "__slots__", ()):
                    stack.append(getattr(cur, slot, None))
    return out


def test_autodiff_analysis_graph_rewrites_direct_source_quotients_for_runtime():
    analysis = _compile_analysis(
        """
        let x = 3.0;
        let y = x * x;
        let main = @y / @x;
        """
    )

    main_binding = _binding_by_name(analysis["graph_binding_by_defid"], "main")
    request = analysis["graph_builtin_requests_by_expr_id"][id(main_binding.expr)]

    assert isinstance(main_binding.expr, BuiltinCallIR)
    assert main_binding.expr.builtin_name == "__autodiff_jacobian"
    assert request.kind is AutodiffBuiltinKind.JACOBIAN
    assert not _contains_node_type(main_binding.expr, DifferentialIR)


def test_autodiff_pass_skips_graph_snapshot_when_program_has_no_autodiff_requests():
    analysis = _compile_analysis(
        """
        let x = 3.0;
        let y = x * x;
        let main = y;
        """
    )

    assert analysis["graph_program"] is None
    assert analysis["graph_binding_by_defid"] == {}
    assert analysis["graph_builtin_requests_by_expr_id"] == {}


def test_autodiff_pass_ignores_dormant_custom_diff_rules_without_runtime_requests():
    analysis = _compile_analysis(
        """
        use std::math::exp;
        let x = exp(2.0);
        let main = x;
        """
    )

    assert analysis["graph_program"] is None
    assert analysis["graph_binding_by_defid"] == {}
    assert analysis["graph_builtin_requests_by_expr_id"] == {}


def test_autodiff_analysis_rewrites_custom_diff_body_for_runtime():
    analysis = _compile_analysis(
        """
        fn ratio(x, y) { x / y }
        @fn ratio(x, y) { @x / @y }
        let a = 6.0;
        let b = 3.0;
        let y = ratio(a, b);
        let main = @y / @a;
        """
    )

    ratio_binding = _binding_by_name(analysis["graph_function_ir_map"], "ratio")

    assert isinstance(ratio_binding.expr, FunctionValueIR)
    assert ratio_binding.expr.custom_diff_body is not None
    requests = analysis["graph_builtin_requests_by_expr_id"]
    assert any(req.kind is AutodiffBuiltinKind.JACOBIAN for req in requests.values())
    assert not _contains_node_type(ratio_binding.expr.custom_diff_body, DifferentialIR)
    assert "__autodiff_jacobian" in _builtin_names(ratio_binding.expr.custom_diff_body)


def test_runtime_executes_rewritten_autodiff_builtins_end_to_end():
    compiler = CompilerDriver()
    runtime = EinlangRuntime(backend="numpy")
    result = compiler.compile(
        """
        let x = 3.0;
        let y = x * x;
        let main = @y / @x;
        """,
        source_file="<autodiff_runtime_exec>",
    )
    assert result.success, result.get_errors()

    exec_result = runtime.execute(result, inputs={})

    assert exec_result.success, exec_result.error
    out = exec_result.value if exec_result.value is not None else exec_result.outputs.get("main")
    assert np.allclose(np.asarray(out, dtype=np.float64), np.array(6.0, dtype=np.float64))


def test_custom_diff_rewritten_tangent_uses_incoming_chain_tangent():
    compiler = CompilerDriver()
    runtime = EinlangRuntime(backend="numpy")
    result = compiler.compile(
        """
        use std::math::exp;
        let k = 0.05;
        let t = 0.2;
        let y = exp(-k * t);
        let main = @y / @k;
        """,
        source_file="<autodiff_custom_diff_chain>",
    )
    assert result.success, result.get_errors()

    exec_result = runtime.execute(result, inputs={})

    assert exec_result.success, exec_result.error
    out = exec_result.value if exec_result.value is not None else exec_result.outputs.get("main")
    expected = -0.2 * np.exp(-0.05 * 0.2)
    assert np.allclose(np.asarray(out, dtype=np.float64), np.array(expected, dtype=np.float64))
