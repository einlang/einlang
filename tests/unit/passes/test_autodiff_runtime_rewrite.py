from __future__ import annotations

import numpy as np

from einlang.compiler.driver import CompilerDriver
from einlang.ir.nodes import BindingIR, BuiltinCallIR, DifferentialIR, IRNode, LiteralIR
from einlang.passes.autodiff import AutodiffPass
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


def _compile_after_autodiff(source: str):
    result = CompilerDriver().compile(
        source.strip(),
        source_file="<autodiff_compiletime>",
        stop_after_pass="AutodiffPass",
    )
    assert result.success, result.get_errors()
    return result


def _binding_by_name(binding_map, name: str) -> BindingIR:
    items = (binding_map or {}).values() if hasattr(binding_map, "values") else (binding_map or [])
    for binding in items:
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


def _compiled_print_arg(source: str):
    result = CompilerDriver().compile(
        source.strip(),
        source_file="<autodiff_print_rewrite>",
    )
    assert result.success, result.get_errors()
    statements = getattr(result.ir, "statements", ()) or ()
    for stmt in reversed(statements):
        if isinstance(stmt, BuiltinCallIR) and stmt.builtin_name == "print":
            args = stmt.args or ()
            assert len(args) == 1
            return args[0]
    raise AssertionError("rewritten print call not found")


def test_autodiff_rewrites_direct_source_quotients_at_compile_time():
    result = _compile_after_autodiff(
        """
        let x = 3.0;
        let y = x * x;
        let main = @y / @x;
        """
    )

    main_binding = _binding_by_name(result.ir.bindings, "main")

    assert str(main_binding.expr) == "2.0 * x"
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


def test_autodiff_rewrites_custom_diff_calls_at_compile_time():
    result = _compile_after_autodiff(
        """
        fn ratio(x, y) { x / y }
        @fn ratio(x, y) { (y * @x - x * @y) / y ** 2.0 }
        let a = 6.0;
        let b = 3.0;
        let y = ratio(a, b);
        let main = @y / @a;
        """
    )

    main_binding = _binding_by_name(result.ir.bindings, "main")

    assert str(main_binding.expr) == "b / b ** 2.0"
    assert not _contains_node_type(result.ir, DifferentialIR)


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


def test_direct_print_of_identifier_differential_rewrites_to_symbolic_literal():
    arg = _compiled_print_arg(
        """
        let xxx = 3.0;
        print(@xxx);
        """
    )

    assert isinstance(arg, LiteralIR)
    assert arg.value == "let @xxx = @xxx;"


def test_direct_print_of_binding_differential_rewrites_to_compile_time_tangent_program():
    arg = _compiled_print_arg(
        """
        let x = 3.0;
        let y = x * x;
        print(@y);
        """
    )

    assert isinstance(arg, LiteralIR)
    assert arg.value == "let @y = 2.0 * x * @x;"


def test_runtime_keeps_direct_print_symbolic_but_bound_tangent_numeric(capsys):
    compiler = CompilerDriver()
    runtime = EinlangRuntime(backend="numpy")
    result = compiler.compile(
        """
        let xxx = 3.0;
        print(@xxx);
        let dxxx = @xxx;
        print(dxxx);
        """,
        source_file="<autodiff_print_symbolic_vs_numeric>",
    )
    assert result.success, result.get_errors()

    exec_result = runtime.execute(result, inputs={})

    assert exec_result.success, exec_result.error
    lines = [line.strip() for line in capsys.readouterr().out.splitlines() if line.strip()]
    assert lines == ["let @xxx = @xxx;", "1.0"]
