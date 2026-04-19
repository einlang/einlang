from __future__ import annotations

import pytest

from einlang.ir.nodes import FunctionValueIR, IdentifierIR, JvpIR, LiteralIR, ParameterIR, VjpIR
from einlang.ir.serialization import deserialize_ir, serialize_ir
from einlang.shared.defid import DefId
from einlang.shared.source_location import SourceLocation
from einlang.shared.types import PrimitiveType


def _identifier(name: str, loc: SourceLocation, index: int) -> IdentifierIR:
    return IdentifierIR(
        name,
        loc,
        defid=DefId(1, index),
        type_info=PrimitiveType("f32"),
        shape_info=(2, None),
    )


def _round_trip(node, *, include_type_info: bool):
    wire = serialize_ir(node, include_location=True, include_type_info=include_type_info, pretty=False)
    return deserialize_ir(wire)


@pytest.mark.parametrize(
    ("factory", "seed_attr"),
    (
        (lambda target, wrt, loc: JvpIR(target=target, wrt=wrt, location=loc, type_info=PrimitiveType("f32"), shape_info=(2, None)), "tangent"),
        (lambda target, wrt, loc: VjpIR(target=target, wrt=wrt, location=loc, type_info=PrimitiveType("f32"), shape_info=(2, None)), "cotangent"),
    ),
)
def test_round_trip_preserves_autodiff_request_metadata_without_explicit_seed(factory, seed_attr):
    loc = SourceLocation("<autodiff_serialization>", 3, 7)
    target = _identifier("y", loc, 2)
    wrt = _identifier("x", loc, 3)

    round_tripped = _round_trip(factory(target, wrt, loc), include_type_info=True)

    assert type(round_tripped).__name__ in {"JvpIR", "VjpIR"}
    assert round_tripped.location == loc
    assert round_tripped.type_info == PrimitiveType("f32")
    assert round_tripped.shape_info == (2, None)
    assert getattr(round_tripped, seed_attr) is None


@pytest.mark.parametrize(
    ("factory", "seed_attr"),
    (
        (lambda target, wrt, loc: JvpIR(target=target, wrt=wrt, location=loc), "tangent"),
        (lambda target, wrt, loc: VjpIR(target=target, wrt=wrt, location=loc), "cotangent"),
    ),
)
def test_round_trip_preserves_autodiff_request_location_without_type_metadata(factory, seed_attr):
    loc = SourceLocation("<autodiff_serialization>", 9, 11)
    target = _identifier("y", loc, 4)
    wrt = _identifier("x", loc, 5)

    round_tripped = _round_trip(factory(target, wrt, loc), include_type_info=False)

    assert round_tripped.location == loc
    assert getattr(round_tripped, seed_attr) is None


def test_round_trip_preserves_function_value_generic_specialization_metadata():
    loc = SourceLocation("<autodiff_serialization>", 12, 5)
    f32 = PrimitiveType("f32")
    fn = FunctionValueIR(
        parameters=[ParameterIR("x", loc, param_type=f32, defid=DefId(1, 7))],
        body=LiteralIR(1.0, loc, type_info=f32),
        location=loc,
        return_type=f32,
        _is_partially_specialized=True,
        _generic_defid=DefId(0, 2905),
    )

    round_tripped = _round_trip(fn, include_type_info=True)

    assert round_tripped.location == loc
    assert round_tripped.return_type == f32
    assert round_tripped._is_partially_specialized is True
    assert round_tripped._generic_defid == DefId(0, 2905)
