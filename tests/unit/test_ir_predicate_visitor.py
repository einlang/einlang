from einlang.ir.nodes import (
    BinaryOpIR,
    BindingIR,
    IdentifierIR,
    IfExpressionIR,
    LiteralIR,
    LoweredEinsteinClauseIR,
    LoweredEinsteinIR,
    LoweredReductionIR,
)
from einlang.ir.predicate_visitor import RecursivePredicateVisitor
from einlang.shared.source_location import SourceLocation
from einlang.shared.types import BinaryOp, ReductionOp


def _loc() -> SourceLocation:
    return SourceLocation("<test>", 1, 1)


class _ContainsNodeTypeVisitor(RecursivePredicateVisitor):
    def __init__(self, *node_types: type) -> None:
        self._node_types = node_types

    def matches(self, node):
        return isinstance(node, self._node_types)


def test_recursive_predicate_visitor_finds_nested_if_in_lowered_reduction() -> None:
    loc = _loc()
    cond = IdentifierIR("cond", loc)
    then_expr = LiteralIR(1, loc)
    else_expr = LiteralIR(0, loc)
    nested_if = IfExpressionIR(cond, then_expr, loc, else_expr=else_expr)
    reduction = LoweredReductionIR(
        body=nested_if,
        operation=ReductionOp.SUM,
        location=loc,
    )

    assert _ContainsNodeTypeVisitor(IfExpressionIR).visit(reduction) is True


def test_recursive_predicate_visitor_finds_lowered_einstein_inside_binding_tree() -> None:
    loc = _loc()
    lowered = LoweredEinsteinIR(
        items=[
            LoweredEinsteinClauseIR(
                body=LiteralIR(0, loc),
                location=loc,
            )
        ],
        location=loc,
    )
    tree = BindingIR(
        name="x",
        expr=BinaryOpIR(BinaryOp.ADD, lowered, LiteralIR(1, loc), loc),
        location=loc,
    )

    assert _ContainsNodeTypeVisitor(LoweredEinsteinIR).visit(tree) is True
