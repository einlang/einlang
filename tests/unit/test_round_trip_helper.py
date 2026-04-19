from types import SimpleNamespace

from einlang.ir.nodes import BindingIR, FunctionValueIR, LiteralIR, ParameterIR, ProgramIR
from einlang.shared.defid import DefId
from einlang.shared.source_location import SourceLocation
from einlang.shared.types import PrimitiveType
from tests.test_utils import apply_ir_round_trip


def test_apply_ir_round_trip_preserves_function_value_return_type():
    loc = SourceLocation("<roundtrip-helper>", 1, 1)
    f32 = PrimitiveType("f32")
    fn = FunctionValueIR(
        parameters=[ParameterIR("x", loc, param_type=f32, defid=DefId(1, 2))],
        body=LiteralIR(1.0, loc, type_info=f32),
        location=loc,
        return_type=f32,
    )
    program = ProgramIR(
        statements=[BindingIR("identity", fn, location=loc, defid=DefId(1, 1))],
        source_files={0: "<roundtrip-helper>"},
    )
    compilation_result = SimpleNamespace(success=True, ir=program)

    apply_ir_round_trip(compilation_result)

    round_tripped_fn = compilation_result.ir.statements[0].expr
    assert isinstance(round_tripped_fn, FunctionValueIR)
    assert round_tripped_fn.return_type == f32
