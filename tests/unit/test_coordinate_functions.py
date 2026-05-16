import numpy as np

from einlang.compiler.driver import CompilerDriver
from einlang.ir.nodes import (
    BindingIR,
    EinsteinIR,
    FunctionCallIR,
    FunctionValueIR,
    IdentifierIR,
    ReductionExpressionIR,
    TupleExpressionIR,
)
from einlang.ir.serialization import deserialize_ir, serialize_ir
from einlang.runtime.runtime import EinlangRuntime
from einlang.shared.types import I32, ReductionOp
from tests.test_utils import compile_and_execute


def _compile(source: str, stop_after_pass: str = "ASTToIRLoweringPass"):
    result = CompilerDriver().compile(
        source,
        source_file="<test>",
        stop_after_pass=stop_after_pass,
    )
    assert result.success, result.get_errors()
    assert result.ir is not None
    return result.ir


def _binding(program, name: str) -> BindingIR:
    for stmt in program.statements:
        if isinstance(stmt, BindingIR) and stmt.name == name:
            return stmt
    raise AssertionError(f"missing binding {name!r}")


def test_coordinate_function_signature_and_call_lower_to_ir_metadata():
    program = _compile(
        """
        fn softmax[j](x: [f32; ..left, j, ..right])
            -> [f32; ..left, j, ..right]
        {
            x
        }

        let logits[class in 0..2] = class as f32;
        let p = softmax[class](logits);
        """
    )

    softmax = _binding(program, "softmax")
    assert isinstance(softmax.expr, FunctionValueIR)
    assert softmax.expr.coordinate_params == ("j",)
    assert str(softmax.expr.return_type) == "[f32; ..left, j, ..right]"

    p = _binding(program, "p")
    assert isinstance(p.expr, FunctionCallIR)
    assert [arg.name for arg in p.expr.coordinate_args] == ["class"]


def test_custom_diff_rule_preserves_coordinate_params_on_merged_function():
    program = _compile(
        """
        fn soft_surrogate_tangent[j](
            p: [f32; ..left, j, ..right],
            dp: [f32; ..left, j, ..right]
        ) -> [f32; ..left, j, ..right]
        {
            p
        }

        fn ste_top1[j](p: [f32; ..left, j, ..right]) -> [i32; ..left, ..right]
        {
            argmax[j](p[..left, j, ..right])
        }

        @fn ste_top1[j](p: [f32; ..left, j, ..right]) {
            soft_surrogate_tangent[j](p, @p)
        }
        """
    )

    ste_top1 = _binding(program, "ste_top1")
    assert isinstance(ste_top1.expr, FunctionValueIR)
    assert ste_top1.expr.coordinate_params == ("j",)
    assert ste_top1.expr.custom_diff_body is not None


def test_argmax_identifier_shorthand_parses_as_selection_reduction():
    program = _compile(
        """
        let logits[class in 0..3] = class as f32;
        let pred = argmax[class](logits);
        """
    )

    pred = _binding(program, "pred")
    assert isinstance(pred.expr, ReductionExpressionIR)
    assert pred.expr.operation == ReductionOp.ARGMAX
    assert [var.name for var in pred.expr.loop_vars] == ["class"]
    assert isinstance(pred.expr.body, IdentifierIR)
    assert pred.expr.body.name == "logits"


def test_argmax_reduction_infers_plain_i32_result_type():
    program = _compile(
        """
        let logits = [1.0, 5.0, 3.0];
        let pred = argmax[class](logits[class]);
        """,
        stop_after_pass="TypeInferencePass",
    )

    pred = _binding(program, "pred")
    assert isinstance(pred.expr, ReductionExpressionIR)
    assert pred.expr.type_info == I32


def test_coordinate_call_and_signature_round_trip_through_ir_serialization():
    program = _compile(
        """
        fn id_axis[j](x: [f32; ..left, j, ..right])
            -> [f32; ..left, j, ..right]
        {
            x
        }

        let x[class in 0..2] = class as f32;
        let y = id_axis[class](x);
        """
    )

    round_tripped = deserialize_ir(
        serialize_ir(program, include_location=True, include_type_info=True, pretty=False)
    )

    id_axis = _binding(round_tripped, "id_axis")
    assert isinstance(id_axis.expr, FunctionValueIR)
    assert id_axis.expr.coordinate_params == ("j",)

    y = _binding(round_tripped, "y")
    assert isinstance(y.expr, FunctionCallIR)
    assert [arg.name for arg in y.expr.coordinate_args] == ["class"]


def test_coordinate_analysis_stamps_ir_nodes_and_round_trips_metadata():
    program = _compile(
        """
        fn top1[j](x: [f32; ..left, j, ..right]) -> [i32; ..left, ..right] {
            argmax[j](x[..left, j, ..right])
        }

        let logits[b in 0..2, class in 0..3] =
            if class == b + 1 { 10.0 } else { class as f32 };
        let pred = top1[class](logits);
        let direct[b] = argmax[class](logits);
        """,
        stop_after_pass="CoordinateGroundingPass",
    )

    pred = _binding(program, "pred")
    direct = _binding(program, "direct")
    assert isinstance(pred.expr, FunctionCallIR)
    assert isinstance(direct.expr, EinsteinIR)
    direct_reduction = direct.expr.clauses[0].value
    assert isinstance(direct_reduction, ReductionExpressionIR)
    assert pred.expr.coordinate_layout == ("b",)
    assert direct.expr.coordinate_layout == ("b",)
    assert direct_reduction.coordinate_address_domain == "class"

    round_tripped = deserialize_ir(
        serialize_ir(program, include_location=True, include_type_info=True, pretty=False)
    )
    rt_pred = _binding(round_tripped, "pred")
    rt_direct = _binding(round_tripped, "direct")
    assert rt_pred.expr.coordinate_layout == ("b",)
    assert rt_direct.expr.coordinate_layout == ("b",)
    assert rt_direct.expr.clauses[0].value.coordinate_address_domain == "class"


def test_coordinate_function_selection_with_rest_packs_executes():
    result = compile_and_execute(
        """
        fn top1[j](x: [f32; ..left, j, ..right]) -> [i32; ..left, ..right] {
            argmax[j](x[..left, j, ..right])
        }

        let logits[b in 0..2, class in 0..3, t in 0..2] =
            if class == b + t { 10.0 } else { class as f32 };
        let pred = top1[class](logits);
        let probe0 = pred[0, 0];
        let probe1 = pred[1, 1];
        (probe0, probe1);
        """,
        CompilerDriver(),
        EinlangRuntime(backend="numpy"),
    )

    assert result.success, result.errors
    assert int(np.asarray(result.outputs["probe0"]).item()) == 0
    assert int(np.asarray(result.outputs["probe1"]).item()) == 2


def test_softmax_coordinate_call_rejects_ungrounded_coordinate():
    result = compile_and_execute(
        """
        fn softmax[j](x) { x }

        let raw = [1.0, 2.0];
        let p = softmax[class](raw);
        p;
        """,
        CompilerDriver(),
        EinlangRuntime(backend="numpy"),
    )

    assert not result.success


def test_argmax_identifier_shorthand_executes_after_coordinate_grounding():
    result = compile_and_execute(
        """
        let logits[class in 0..3] = if class == 1 { 5.0 } else { class as f32 };
        let pred = argmax[class](logits);
        pred;
        """,
        CompilerDriver(),
        EinlangRuntime(backend="numpy"),
    )

    assert result.success, result.errors
    assert int(np.asarray(result.outputs["pred"]).item()) == 1


def test_argmax_coordinate_expression_shorthand_expands_pointwise_body():
    result = compile_and_execute(
        """
        let x[class in 0..3] = if class == 1 { -5.0 } else { class as f32 };
        let pred = argmax[class](x ** 2.0);
        pred;
        """,
        CompilerDriver(),
        EinlangRuntime(backend="numpy"),
    )

    assert result.success, result.errors
    assert int(np.asarray(result.outputs["pred"]).item()) == 1


def test_argmax_coordinate_expression_accepts_explicit_indexed_body():
    result = compile_and_execute(
        """
        let x = [-1.0, 3.0, -5.0];
        let pred = argmax[class](x[class] ** 2.0);
        pred;
        """,
        CompilerDriver(),
        EinlangRuntime(backend="numpy"),
    )

    assert result.success, result.errors
    assert int(np.asarray(result.outputs["pred"]).item()) == 2


def test_named_reduction_shorthand_expands_pointwise_body():
    result = compile_and_execute(
        """
        let x[class in 0..3] = class as f32 + 1.0;
        let s = sum[class](x ** 2.0);
        let m = max[class](x ** 2.0);
        (s, m);
        """,
        CompilerDriver(),
        EinlangRuntime(backend="numpy"),
    )

    assert result.success, result.errors
    assert float(np.asarray(result.outputs["s"]).item()) == 14.0
    assert float(np.asarray(result.outputs["m"]).item()) == 9.0


def test_named_reduction_shorthand_expands_multiple_axes_and_preserves_context():
    result = compile_and_execute(
        """
        let x[b in 0..2, row in 0..2, col in 0..3] =
            (b + row + col) as f32;
        let s[b] = sum[row, col](x ** 2.0);
        s;
        """,
        CompilerDriver(),
        EinlangRuntime(backend="numpy"),
    )

    assert result.success, result.errors
    np.testing.assert_allclose(np.asarray(result.outputs["s"]), np.array([19.0, 43.0]))


def test_nested_reduction_shorthand_uses_outer_reduction_context():
    result = compile_and_execute(
        """
        let A[k in 0..2, n in 0..3] = (10 * k + n) as f32;
        let y = sum[k](max[n](A));
        y;
        """,
        CompilerDriver(),
        EinlangRuntime(backend="numpy"),
    )

    assert result.success, result.errors
    assert float(np.asarray(result.outputs["y"]).item()) == 14.0


def test_coordinate_function_signature_instantiates_return_layout():
    result = compile_and_execute(
        """
        fn top1[j](x: [f32; b, j]) -> [i32; b] {
            argmax[j](x[b, j])
        }

        let logits[b in 0..2, class in 0..3] =
            if class == b + 1 { 10.0 } else { class as f32 };
        let pred = top1[class](logits);
        let total = sum[b](pred);
        total;
        """,
        CompilerDriver(),
        EinlangRuntime(backend="numpy"),
    )

    assert result.success, result.errors
    assert int(np.asarray(result.outputs["total"]).item()) == 3


def test_coordinate_function_rejects_return_only_dimension():
    result = CompilerDriver().compile(
        """
        fn bad[j](x: [f32; j]) -> [i32; b] {
            argmax[j](x[j])
        }
        """,
        source_file="<test>",
    )

    assert not result.success
    assert "return-only symbolic dimensions are not allowed" in "\n".join(result.get_errors())


def test_coordinate_function_signature_instantiates_return_rank():
    program = _compile(
        """
        fn drop_axis[j](x: [f32; ..left, j, ..right])
            -> [f32; ..left, ..right]
        {
            0.0
        }

        let logits[b in 0..2, class in 0..3, t in 0..4] =
            (b + class + t) as f32;
        let y = drop_axis[class](logits);
        """,
        stop_after_pass="UnifiedShapeAnalysisPass",
    )

    y = _binding(program, "y")
    assert isinstance(y.expr, FunctionCallIR)
    assert y.expr.shape_info == (2, 4)


def test_coordinate_function_signature_instantiates_return_type_shape():
    program = _compile(
        """
        fn drop_axis[j](x: [f32; ..left, j, ..right])
            -> [f32; ..left, ..right]
        {
            0.0
        }

        let logits[b in 0..2, class in 0..3, t in 0..4] =
            (b + class + t) as f32;
        let y = drop_axis[class](logits);
        """,
        stop_after_pass="TypeInferencePass",
    )

    y = _binding(program, "y")
    assert isinstance(y.expr, FunctionCallIR)
    assert str(y.expr.type_info) == "[f32; 2, 4]"


def test_coordinate_function_specialization_expands_multiple_rest_packs():
    result = compile_and_execute(
        """
        fn swap[j, k](x: [f32; ..left, j, ..middle, k, ..right])
            -> [f32; ..left, k, ..middle, j, ..right]
        {
            let y[..left, k, ..middle, j, ..right] =
                x[..left, j, ..middle, k, ..right];
            y
        }

        let x[a in 0..2, b in 0..3, c in 0..4, d in 0..5] =
            (1000 * a + 100 * b + 10 * c + d) as f32;
        let y = swap[b, d](x);
        let probe = y[1, 4, 2, 1];
        probe;
        """,
        CompilerDriver(),
        EinlangRuntime(backend="numpy"),
    )

    assert result.success, result.errors
    assert float(np.asarray(result.outputs["probe"]).item()) == 1124.0


def test_coordinate_function_multiple_rest_packs_stamp_layout_shape_and_type():
    program = _compile(
        """
        fn swap[j, k](x: [f32; ..left, j, ..middle, k, ..right])
            -> [f32; ..left, k, ..middle, j, ..right]
        {
            let y[..left, k, ..middle, j, ..right] =
                x[..left, j, ..middle, k, ..right];
            y
        }

        let x[a in 0..2, b in 0..3, c in 0..4, d in 0..5] =
            (1000 * a + 100 * b + 10 * c + d) as f32;
        let y = swap[b, d](x);
        """,
        stop_after_pass="TypeInferencePass",
    )

    y = _binding(program, "y")
    assert isinstance(y.expr, FunctionCallIR)
    assert y.expr.coordinate_layout == ("a", "d", "c", "b")
    assert y.expr.shape_info == (2, 5, 4, 3)
    assert str(y.expr.type_info) == "[f32; 2, 5, 4, 3]"


def test_coordinate_parameter_pack_uses_parenthesized_call_group():
    program = _compile(
        """
        fn id_axes[..axes](x: [f32; ..axes]) -> [f32; ..axes] {
            x
        }

        let x[h in 0..2, w in 0..3] = (10 * h + w) as f32;
        let y = id_axes[(h, w)](x);
        """,
        stop_after_pass="TypeInferencePass",
    )

    y = _binding(program, "y")
    assert isinstance(y.expr, FunctionCallIR)
    assert len(y.expr.coordinate_args) == 1
    assert isinstance(y.expr.coordinate_args[0], TupleExpressionIR)
    assert y.expr.coordinate_layout == ("h", "w")
    assert y.expr.coordinate_pack_bindings == {"axes": ("h", "w")}
    assert y.expr.shape_info == (2, 3)
    assert str(y.expr.type_info) == "[f32; 2, 3]"


def test_coordinate_parameter_pack_rejects_bare_multiple_call_args():
    result = CompilerDriver().compile(
        """
        fn id_axes[..axes](x: [f32; ..axes]) -> [f32; ..axes] {
            x
        }

        let x[h in 0..2, w in 0..3] = (10 * h + w) as f32;
        let y = id_axes[h, w](x);
        """,
        source_file="<test>",
        stop_after_pass="CoordinateGroundingPass",
    )

    assert not result.success
    assert "expects 1 coordinate argument at most, got 2" in "\n".join(result.get_errors())


def test_one_pack_resolves_adjacent_pack_by_elimination():
    """When only one pack is a coordinate param, the other resolves by elimination."""
    from tests.test_utils import compile_and_execute

    source = """
    fn pool[..spatial](x: [f32; ..batch, ..spatial]) -> [f32; ..batch] {
        max[..spatial](x[..batch, ..spatial])
    }

    let x[b in 0..2, h in 0..3, w in 0..4] = (100 * b + 10 * h + w) as f32;
    let y = pool[(h, w)](x);
    let probe0 = y[0];
    let probe1 = y[1];
    (probe0, probe1);
    """

    result = compile_and_execute(
        source,
        CompilerDriver(),
        EinlangRuntime(),
        source_file="<test>",
    )
    assert result.success, f"Compilation or execution failed: {result.errors}"
    import numpy as np
    assert float(np.asarray(result.outputs["probe0"]).item()) == 23.0
    assert float(np.asarray(result.outputs["probe1"]).item()) == 123.0


def test_two_named_coordinate_params_with_rests_between():
    """Two named coord params serve as anchors for 3 surrounding rest packs."""
    result = compile_and_execute(
        """
        fn max_over[j, k](x: [f32; ..left, j, ..right, k, ..rightmost])
            -> [f32; ..left, ..right, ..rightmost]
        {
            max[j, k](x[..left, j, ..right, k, ..rightmost])
        }

        let x[a in 0..2, h in 0..2, w in 0..2] = (a + h + w) as f32;
        let y = max_over[h, w](x);
        let probe0 = y[0];
        let probe1 = y[1];
        (probe0, probe1);
        """,
        CompilerDriver(),
        EinlangRuntime(),
        source_file="<test>",
    )
    assert result.success, f"Compilation or execution failed: {result.errors}"
    import numpy as np
    assert float(np.asarray(result.outputs["probe0"]).item()) == 2.0
    assert float(np.asarray(result.outputs["probe1"]).item()) == 3.0


def test_two_adjacent_named_coord_params_no_rest_between():
    """Two adjacent named coord params resolve by position: each is its own anchor."""
    result = compile_and_execute(
        """
        fn twin_sum[j, k](x: [f32; ..left, j, k, ..right])
            -> [f32; ..left, ..right]
        {
            sum[j, k](x[..left, j, k, ..right])
        }

        let x[a in 0..2, h in 0..2, w in 0..2] = (a + h + w) as f32;
        let y = twin_sum[h, w](x);
        let probe0 = y[0];
        let probe1 = y[1];
        (probe0, probe1);
        """,
        CompilerDriver(),
        EinlangRuntime(),
        source_file="<test>",
    )
    assert result.success, f"Compilation or execution failed: {result.errors}"
    import numpy as np
    assert float(np.asarray(result.outputs["probe0"]).item()) == 4.0
    assert float(np.asarray(result.outputs["probe1"]).item()) == 8.0


def test_multiple_named_coords_in_brackets_packs_only_in_value_params():
    """Two named coords in brackets; rest packs only in value param shape — anchor-based."""
    result = compile_and_execute(
        """
        fn max_over[j, k](x: [f32; ..left, j, ..right, k, ..rightmost])
            -> [f32; ..left, ..right, ..rightmost]
        {
            max[j, k](x[..left, j, ..right, k, ..rightmost])
        }

        let x[a in 0..2, h in 0..2, w in 0..2, b in 0..3] = (a + h + w + b) as f32;
        let y = max_over[h, w](x);
        let probe0 = y[0, 0];
        let probe1 = y[0, 2];
        (probe0, probe1);
        """,
        CompilerDriver(),
        EinlangRuntime(),
        source_file="<test>",
    )
    assert result.success, f"Compilation or execution failed: {result.errors}"
    import numpy as np
    assert float(np.asarray(result.outputs["probe0"]).item()) == 2.0
    assert float(np.asarray(result.outputs["probe1"]).item()) == 4.0


def test_coordinate_parameter_pack_can_be_inferred_after_scalar_coordinate():
    result = compile_and_execute(
        """
        fn move_channel[channel, ..spatial](x: [f32; channel, ..spatial])
            -> [f32; ..spatial, channel]
        {
            let y[..spatial, channel] = x[channel, ..spatial];
            y
        }

        let x[c in 0..2, h in 0..3, w in 0..4] =
            (100 * c + 10 * h + w) as f32;
        let y = move_channel[c](x);
        let probe = y[2, 3, 1];
        probe;
        """,
        CompilerDriver(),
        EinlangRuntime(backend="numpy"),
    )

    assert result.success, result.errors
    assert float(np.asarray(result.outputs["probe"]).item()) == 123.0


def test_coordinate_parameter_pack_infers_layout_shape_and_type_after_scalar_coordinate():
    program = _compile(
        """
        fn move_channel[channel, ..spatial](x: [f32; channel, ..spatial])
            -> [f32; ..spatial, channel]
        {
            let y[..spatial, channel] = x[channel, ..spatial];
            y
        }

        let x[c in 0..2, h in 0..3, w in 0..4] =
            (100 * c + 10 * h + w) as f32;
        let y = move_channel[c](x);
        """,
        stop_after_pass="TypeInferencePass",
    )

    y = _binding(program, "y")
    assert isinstance(y.expr, FunctionCallIR)
    assert [arg.name for arg in y.expr.coordinate_args] == ["c"]
    assert y.expr.coordinate_layout == ("h", "w", "c")
    assert y.expr.coordinate_axis_bindings == {"channel": "c"}
    assert y.expr.coordinate_pack_bindings == {"spatial": ("h", "w")}
    assert y.expr.shape_info == (3, 4, 2)
    assert str(y.expr.type_info) == "[f32; 3, 4, 2]"
