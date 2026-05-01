import numpy as np

from einlang.compiler.driver import CompilerDriver
from einlang.ir.nodes import (
    BindingIR,
    FunctionCallIR,
    FunctionValueIR,
    IdentifierIR,
    ReductionExpressionIR,
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
