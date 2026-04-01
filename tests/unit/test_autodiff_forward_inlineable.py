from einlang.ir.nodes import (
    BinaryOpIR,
    BindingIR,
    BlockExpressionIR,
    IdentifierIR,
    IfExpressionIR,
    LiteralIR,
)
from einlang.passes.autodiff._forward import _pretty_callee_tangent_inlineable
from einlang.shared.source_location import SourceLocation
from einlang.shared.types import BinaryOp


def _loc() -> SourceLocation:
    return SourceLocation("<test>", 1, 1)


def test_pretty_callee_tangent_inlineable_accepts_simple_expression_tree() -> None:
    loc = _loc()
    expr = BinaryOpIR(BinaryOp.ADD, IdentifierIR("x", loc), LiteralIR(1, loc), loc)
    assert _pretty_callee_tangent_inlineable(expr) is True


def test_pretty_callee_tangent_inlineable_rejects_blocks_with_bindings() -> None:
    loc = _loc()
    expr = BlockExpressionIR(
        [BindingIR(name="tmp", expr=LiteralIR(1, loc), location=loc)],
        loc,
        final_expr=IdentifierIR("tmp", loc),
    )
    assert _pretty_callee_tangent_inlineable(expr) is False


def test_pretty_callee_tangent_inlineable_rejects_if_expression() -> None:
    loc = _loc()
    expr = IfExpressionIR(
        IdentifierIR("cond", loc),
        LiteralIR(1, loc),
        loc,
        else_expr=LiteralIR(0, loc),
    )
    assert _pretty_callee_tangent_inlineable(expr) is False
