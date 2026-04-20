from __future__ import annotations

import numpy as np
import pytest
from pathlib import Path

from einlang.compiler.driver import CompilerDriver
from einlang.ir import dump_ir
from einlang.ir.nodes import (
    BindingIR,
    BlockExpressionIR,
    BuiltinCallIR,
    DifferentialIR,
    FunctionValueIR,
    IRNode,
    IdentifierIR,
    JvpIR,
    LazyJacobianIR,
    LiteralIR,
    RectangularAccessIR,
    VjpIR,
)
from einlang.passes.autodiff import AutodiffPass
from einlang.shared.autodiff_intrinsics import autodiff_builtin_kind
from einlang.runtime.runtime import EinlangRuntime

PROJECT_ROOT = Path(__file__).resolve().parents[3]


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


def _compile_after_request_lowering(source: str):
    result = CompilerDriver().compile(
        source.strip(),
        source_file="<autodiff_runtime_request_lowering>",
        stop_after_pass="AutodiffRequestLoweringPass",
    )
    assert result.success, result.get_errors()
    return result


def _assert_no_runtime_autodiff_ir(node: object) -> None:
    assert not _contains_node_type(node, LazyJacobianIR)
    assert not _contains_node_type(node, JvpIR)
    assert not _contains_node_type(node, VjpIR)
    assert not _contains_autodiff_builtin(node)


def _compile_and_run_main(source: str, source_file: str):
    compiler = CompilerDriver()
    runtime = EinlangRuntime(backend="numpy")
    result = compiler.compile(source, source_file=source_file)
    assert result.success, result.get_errors()
    _assert_no_runtime_autodiff_ir(result.ir)
    exec_result = runtime.execute(result, inputs={})
    assert exec_result.success, exec_result.error
    return exec_result.value if exec_result.value is not None else exec_result.outputs.get("main"), result


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


def _contains_autodiff_builtin(node: object) -> bool:
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
        if isinstance(cur, BuiltinCallIR) and autodiff_builtin_kind(getattr(cur, "defid", None)) is not None:
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


def _identifier_nodes(node: object):
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
        if isinstance(cur, IdentifierIR):
            yield cur
            continue
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


def _walk_nodes(node: object):
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
        if isinstance(cur, IRNode):
            yield cur
            for cls in type(cur).__mro__:
                for slot in getattr(cls, "__slots__", ()):
                    stack.append(getattr(cur, slot, None))
            continue
        if isinstance(cur, dict):
            stack.extend(cur.keys())
            stack.extend(cur.values())
            continue
        if isinstance(cur, (list, tuple)):
            stack.extend(cur)


def test_rewrites_direct_quotients():
    result = _compile_after_autodiff(
        """
        let x = 3.0;
        let y = x * x;
        let main = @y / @x;
        """
    )

    main_binding = _binding_by_name(result.ir.bindings, "main")

    assert isinstance(main_binding.expr, LazyJacobianIR)
    assert not _contains_node_type(main_binding.expr, DifferentialIR)


def test_recurrence_slice_no_nested_hoists():
    source = """
    let epochs = 1;

    let x[n in 0..2, k in 0..4] = (1 + n + k) as f32;

    let theta0[p in 0..5, j in 0..3] =
        if p == 4 { 0.5 * (1 + j) as f32 } else { 0.1 * (1 + p + j) as f32 };

    let theta[step in 0..epochs + 1] = if step == 0 {
        theta0
    } else {
        let prev_theta = theta[step - 1] as [f32; 5, 3];
        let logits_before[n in 0..2, j in 0..3] =
            sum[k in 0..4](x[n, k] * prev_theta[k, j]) + prev_theta[4, j];
        let diff_before[n in 0..2, j in 0..3] = logits_before[n, j] - (n == j) as f32;
        let loss_before = sum[n in 0..2, j in 0..3](diff_before[n, j] * diff_before[n, j]);
        let d_theta = @loss_before / @prev_theta;
        let next_theta[p in 0..5, j in 0..3] = prev_theta[p, j] - 0.01 * d_theta[p, j];
        next_theta
    };

    let final_theta = theta[epochs] as [f32; 5, 3];
    let logits[n in 0..2, j in 0..3] =
        sum[k in 0..4](x[n, k] * final_theta[k, j]) + final_theta[4, j];
    let diff[n in 0..2, j in 0..3] = logits[n, j] - (n == j) as f32;
    let loss = sum[n in 0..2, j in 0..3](diff[n, j] * diff[n, j]);

    print(logits);
    print(loss);
    """
    result = CompilerDriver().compile(
        source,
        source_file="<autodiff_recurrence_slice_hoist>",
    )

    assert result.success, result.get_errors()
    ir_text = dump_ir(result.ir)

    assert "lowered-recurrence" in ir_text
    assert "__ad_hoist_" not in ir_text


@pytest.mark.parametrize(
    "source",
    [
        """
        let x = 3.0;
        let y = x * x;
        let main = y;
        """,
        """
        use std::math::exp;
        let x = exp(2.0);
        let main = x;
        """,
    ],
    ids=["no-autodiff-requests", "dormant-custom-diff"],
)
def test_no_graph(source):
    analysis = _compile_analysis(source)
    assert analysis["graph_program"] is None
    assert analysis["graph_binding_by_defid"] != {}
    assert analysis["graph_builtin_requests_by_expr_id"] == {}


def test_rewrites_custom_diff_calls():
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

    assert isinstance(main_binding.expr, LazyJacobianIR)
    assert not _contains_node_type(result.ir, DifferentialIR)


def test_imported_custom_diff_param_defids():
    result = CompilerDriver().compile(
        """
        use std::ml;
        let y = std::ml::asin(0.5);
        """,
        source_file="<autodiff_imported_custom_diff>",
        stop_after_pass="ASTToIRLoweringPass",
    )
    assert result.success, result.get_errors()

    asin_binding = _binding_by_name(result.ir.bindings, "asin")
    assert isinstance(asin_binding.expr, FunctionValueIR)
    assert isinstance(asin_binding.expr.custom_diff_body, BlockExpressionIR)

    param_defid = asin_binding.expr.parameters[0].defid
    assert param_defid is not None

    custom_diff_identifiers = [
        ident for ident in _identifier_nodes(asin_binding.expr.custom_diff_body) if ident.name == "x"
    ]
    assert custom_diff_identifiers
    assert all(ident.defid == param_defid for ident in custom_diff_identifiers)


def test_scalar_quotients_lower_to_plain_ir():
    result = _compile_after_request_lowering(
        """
        let x = 3.0;
        let y = x * x;
        let main = @y / @x;
        """
    )

    main_binding = _binding_by_name(result.ir.bindings, "main")

    assert not _contains_node_type(main_binding.expr, LazyJacobianIR)
    assert not _contains_node_type(main_binding.expr, JvpIR)
    assert not _contains_node_type(main_binding.expr, VjpIR)
    assert not _contains_autodiff_builtin(main_binding.expr)


def test_runtime_executes_scalar_plain_ir():
    out, _ = _compile_and_run_main(
        """
        let x = 3.0;
        let y = x * x;
        let main = @y / @x;
        """,
        "<autodiff_runtime_exec>",
    )
    assert np.allclose(np.asarray(out, dtype=np.float64), np.array(6.0, dtype=np.float64))


def test_runtime_executes_tensor_jacobian():
    out, _ = _compile_and_run_main(
        """
        let A = [[1.0, 2.0], [3.0, 4.0]];
        let B = [[5.0, 6.0], [7.0, 8.0]];
        let C[i, j] = sum[k](A[i, k] * B[k, j]);
        let main = @C / @A;
        """,
        "<autodiff_tensor_jacobian_plain_ir>",
    )
    actual = np.asarray(out, dtype=np.float64)
    expected = np.zeros((2, 2, 2, 2), dtype=np.float64)
    B = np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float64)
    for i in range(2):
        for j in range(2):
            for k in range(2):
                for l in range(2):
                    expected[i, j, k, l] = B[l, j] if i == k else 0.0
    assert actual.shape == expected.shape
    np.testing.assert_allclose(actual, expected)


def test_local_recurrence_quotients_no_lazy_cycle():
    result = CompilerDriver().compile(
        """
        let alpha = 0.25;
        let x[0] = 0.0;
        let x[k in 1..5] = {
            let prev = x[k - 1];
            let loss = prev * prev;
            let g = @loss / @prev;
            prev - alpha * g
        };
        let main = x;
        """,
        source_file="<autodiff_recurrence_local_quotient>",
        stop_after_pass="AutodiffPass",
    )

    assert result.success, result.get_errors()
    assert not _contains_node_type(result.ir, DifferentialIR)


def test_custom_diff_uses_chain_tangent():
    out, _ = _compile_and_run_main(
        """
        use std::math::exp;
        let k = 0.05;
        let t = 0.2;
        let y = exp(-k * t);
        let main = @y / @k;
        """,
        "<autodiff_custom_diff_chain>",
    )
    expected = -0.2 * np.exp(-0.05 * 0.2)
    assert np.allclose(np.asarray(out, dtype=np.float64), np.array(expected, dtype=np.float64))


def test_print_identifier_diff_rewrites():
    arg = _compiled_print_arg(
        """
        let xxx = 3.0;
        print(@xxx);
        """
    )

    assert isinstance(arg, LiteralIR)
    assert arg.value == "let @xxx = @xxx;"


def test_print_binding_diff_rewrites():
    arg = _compiled_print_arg(
        """
        let x = 3.0;
        let y = x * x;
        print(@y);
        """
    )

    assert isinstance(arg, LiteralIR)
    assert arg.value == "let @y = 2.0 * x * @x;"


def test_runtime_print_symbolic_then_numeric(capsys):
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


def test_nested_max_pool_pullback():
    out, _ = _compile_and_run_main(
        """
        use std::ml::{max_pool, relu};

        let x[n in 0..1, c in 0..1, h in 0..4, w in 0..4] = (h * 4 + w + 1) as f32;
        let p1 = max_pool(relu(x), [2, 2], [2, 2], [0, 0]);
        let p2 = max_pool(relu(p1), [2, 2], [2, 2], [0, 0]);
        let loss = p2[0, 0, 0, 0];
        let main = @loss / @x;
        """,
        "<autodiff_nested_max_pool_pullback>",
    )
    actual = np.asarray(out, dtype=np.float64)
    expected = np.zeros((1, 1, 4, 4), dtype=np.float64)
    expected[0, 0, 3, 3] = 1.0
    np.testing.assert_allclose(actual, expected)


@pytest.mark.skip(reason="overlapping grad not supported yet")
def test_average_pool_pullback():
    # Test average pool gradient computation with overlapping windows
    out, _ = _compile_and_run_main(
        """
        use std::ml::average_pool;

        let x = [[[[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0], [13.0, 14.0, 15.0, 16.0]]]];
        let pooled = average_pool(x, [2, 2], [1, 1], [0, 0]);
        let loss = sum[n in 0..1, c in 0..1, h in 0..3, w in 0..3](pooled[n, c, h, w]);
        let main = @loss / @x;
        """,
        "<autodiff_average_pool_pullback>",
    )
    actual = np.asarray(out, dtype=np.float64)
    # With 2x2 kernel and stride [1,1], each input element appears in multiple windows:
    # Corner elements (like [0,0]): 1 window
    # Edge elements (like [0,1], [1,0]): 2 windows  
    # Center elements (like [1,1]): 4 windows
    # Each window contributes 1.0/4 = 0.25 to each element in it
    expected = np.array([[[[
        [0.25, 0.50, 0.50, 0.25],  # row 0: corner=1*0.25, edges=2*0.25
        [0.50, 1.00, 1.00, 0.50],  # row 1: edges=2*0.25, center=4*0.25
        [0.50, 1.00, 1.00, 0.50],  # row 2: edges=2*0.25, center=4*0.25
        [0.25, 0.50, 0.50, 0.25]   # row 3: corner=1*0.25, edges=2*0.25
    ]]]], dtype=np.float64)
    np.testing.assert_allclose(actual, expected)


def test_intermediate_max_pool_pullback():
    out, _ = _compile_and_run_main(
        """
        use std::ml::{max_pool, relu};

        let x[n in 0..1, c in 0..1, h in 0..4, w in 0..4] = (h * 4 + w + 1) as f32;
        let p1 = max_pool(relu(x), [2, 2], [2, 2], [0, 0]);
        let p2 = max_pool(relu(p1), [2, 2], [2, 2], [0, 0]);
        let loss = p2[0, 0, 0, 0];
        let main = @loss / @p1;
        """,
        "<autodiff_nested_max_pool_intermediate_pullback>",
    )
    actual = np.asarray(out, dtype=np.float64)
    expected = np.zeros((1, 1, 2, 2), dtype=np.float64)
    expected[0, 0, 1, 1] = 1.0
    np.testing.assert_allclose(actual, expected)


def test_req_lower_clears():
    result = _compile_after_request_lowering(
        """
        let x = [[1.0, 2.0], [3.0, 4.0]];
        let y = [[1.0, 0.0], [0.0, 1.0]];
        let w0[i in 0..2, j in 0..2] = 0.1 * (1.0 + (i + j) as f32);
        let b0[j in 0..2] = 0.0;
        let logits0[n in 0..2, j in 0..2] = sum[d in 0..2](x[n, d] * w0[d, j]) + b0[j];
        let diff0[n in 0..2, j in 0..2] = logits0[n, j] - y[n, j];
        let loss0 = sum[n in 0..2, j in 0..2](diff0[n, j] * diff0[n, j]);
        let d_w0 = @loss0 / @w0;
        let d_b0 = @loss0 / @b0;
        let w1[i in 0..2, j in 0..2] = w0[i, j] - 0.1 * d_w0[i, j];
        let b1[j in 0..2] = b0[j] - 0.1 * d_b0[j];
        let logits1[n in 0..2, j in 0..2] = sum[d in 0..2](x[n, d] * w1[d, j]) + b1[j];
        let diff1[n in 0..2, j in 0..2] = logits1[n, j] - y[n, j];
        let loss1 = sum[n in 0..2, j in 0..2](diff1[n, j] * diff1[n, j]);
        let d_w1 = @loss1 / @w1;
        let d_b1 = @loss1 / @b1;
        let main = d_b1[0];
        """
    )

    assert not _contains_node_type(result.ir, DifferentialIR)
    assert not _contains_node_type(result.ir, LazyJacobianIR)
    assert not _contains_node_type(result.ir, JvpIR)
    assert not _contains_node_type(result.ir, VjpIR)


def test_explicit_matmul_bias_grad():
    compiler = CompilerDriver()
    runtime = EinlangRuntime(backend="numpy")
    result = compiler.compile(
        """
        let x[n in 0..2, d in 0..4] = (1 + n + d) as f32;
        let y[n in 0..2, cls in 0..3] = (n == cls) as f32;
        let theta[p in 0..5, cls in 0..3] =
            if p == 4 { 0.0 } else { 1e-2 * (1 + p + cls) as f32 };
        let logits[n in 0..2, cls in 0..3] =
            sum[d in 0..4](x[n, d] * theta[d, cls]) + theta[4, cls];
        let diff[n in 0..2, cls in 0..3] = logits[n, cls] - y[n, cls];
        let loss = sum[n in 0..2, cls in 0..3](diff[n, cls] * diff[n, cls]) / 6.0;
        let grad = @loss / @theta;
        let main = grad[0, 0];
        """,
        source_file="<autodiff_explicit_matmul_bias_gradient>",
    )
    assert result.success, result.get_errors()

    grad_binding = _binding_by_name(result.ir.bindings, "grad")
    theta_accesses = [
        node
        for node in _walk_nodes(grad_binding.expr)
        if isinstance(node, RectangularAccessIR)
        and isinstance(getattr(node, "array", None), IdentifierIR)
        and getattr(node.array, "defid", None) == _binding_by_name(result.ir.bindings, "theta").defid
    ]
    assert theta_accesses, "expected explicit theta gradient to preserve accesses to the wrt tensor"

    exec_result = runtime.execute(result, inputs={})

    assert exec_result.success, exec_result.error
    main = exec_result.value if exec_result.value is not None else exec_result.outputs.get("main")
    assert np.isfinite(float(np.asarray(main, dtype=np.float64)))
