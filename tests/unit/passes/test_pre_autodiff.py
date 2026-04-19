from einlang.ir.nodes import (
    ArrayLiteralIR,
    BinaryOpIR,
    BlockExpressionIR,
    BuiltinCallIR,
    BindingIR,
    EinsteinClauseIR,
    EinsteinIR,
    IdentifierIR,
    IfExpressionIR,
    LiteralIR,
    ProgramIR,
)
from einlang.passes.base import TyCtxt
from einlang.passes.pre_autodiff_pruning import PreAutodiffPruningPass
from einlang.shared.source_location import SourceLocation
from einlang.shared.types import BinaryOp


def _loc() -> SourceLocation:
    return SourceLocation(file="<test>", line=1, column=1)


def test_pre_autodiff_pruning_prunes_constant_if_in_binding() -> None:
    loc = _loc()
    expr = IfExpressionIR(
        condition=LiteralIR(True, loc),
        then_expr=LiteralIR(10, loc),
        else_expr=LiteralIR(20, loc),
        location=loc,
    )
    binding = BindingIR(name="x", expr=expr, location=loc)
    program = ProgramIR(statements=[binding], location=loc)

    tcx = TyCtxt()
    PreAutodiffPruningPass().run(program, tcx)

    assert isinstance(binding.expr, LiteralIR)
    assert binding.expr.value == 10
    stats = tcx.get_analysis(PreAutodiffPruningPass)
    assert stats["implemented"] is True
    assert stats["pruned_if_count"] >= 1


def test_pre_autodiff_pruning_prunes_constant_if_in_einstein_clause() -> None:
    loc = _loc()
    clause = EinsteinClauseIR(
        indices=[],
        value=IfExpressionIR(
            condition=LiteralIR(0, loc),
            then_expr=LiteralIR(1, loc),
            else_expr=LiteralIR(2, loc),
            location=loc,
        ),
        location=loc,
    )
    binding = BindingIR(
        name="y",
        expr=EinsteinIR(clauses=[clause], location=loc),
        location=loc,
    )
    program = ProgramIR(statements=[binding], location=loc)

    tcx = TyCtxt()
    PreAutodiffPruningPass().run(program, tcx)

    assert isinstance(clause.value, LiteralIR)
    assert clause.value.value == 2


def test_pre_autodiff_pruning_prunes_rank_branch_from_local_len_binding() -> None:
    loc = _loc()
    rank_binding = BindingIR(
        name="rank",
        expr=BuiltinCallIR("len", [ArrayLiteralIR([LiteralIR(1, loc)], loc)], loc),
        location=loc,
    )
    branch = IfExpressionIR(
        condition=BinaryOpIR(
            BinaryOp.EQ,
            IdentifierIR("rank", loc),
            LiteralIR(1, loc),
            loc,
        ),
        then_expr=LiteralIR(11, loc),
        else_expr=LiteralIR(22, loc),
        location=loc,
    )
    block = BlockExpressionIR(statements=[rank_binding], final_expr=branch, location=loc)
    binding = BindingIR(name="z", expr=block, location=loc)
    program = ProgramIR(statements=[binding], location=loc)

    tcx = TyCtxt()
    PreAutodiffPruningPass().run(program, tcx)

    assert isinstance(block.final_expr, LiteralIR)
    assert block.final_expr.value == 11


def test_pre_autodiff_pruning_removes_dead_constant_binding_from_block() -> None:
    loc = _loc()
    block = BlockExpressionIR(
        statements=[BindingIR(name="rank", expr=LiteralIR(1, loc), location=loc)],
        final_expr=LiteralIR(7, loc),
        location=loc,
    )
    binding = BindingIR(name="z", expr=block, location=loc)
    program = ProgramIR(statements=[binding], location=loc)

    tcx = TyCtxt()
    PreAutodiffPruningPass().run(program, tcx)

    assert isinstance(binding.expr, LiteralIR)
    assert binding.expr.value == 7


def test_pre_autodiff_pruning_does_not_fold_general_arithmetic_binding() -> None:
    loc = _loc()
    expr = BinaryOpIR(
        BinaryOp.ADD,
        LiteralIR(1, loc),
        LiteralIR(2, loc),
        loc,
    )
    binding = BindingIR(name="x", expr=expr, location=loc)
    program = ProgramIR(statements=[binding], location=loc)

    tcx = TyCtxt()
    PreAutodiffPruningPass().run(program, tcx)

    assert isinstance(binding.expr, BinaryOpIR)
    assert binding.expr.operator == BinaryOp.ADD


def test_pre_autodiff_pruning_does_not_prune_non_metadata_local_constant_branch() -> None:
    loc = _loc()
    x_binding = BindingIR(
        name="x",
        expr=LiteralIR(1, loc),
        location=loc,
    )
    branch = IfExpressionIR(
        condition=BinaryOpIR(
            BinaryOp.EQ,
            IdentifierIR("x", loc),
            LiteralIR(1, loc),
            loc,
        ),
        then_expr=LiteralIR(11, loc),
        else_expr=LiteralIR(22, loc),
        location=loc,
    )
    block = BlockExpressionIR(statements=[x_binding], final_expr=branch, location=loc)
    binding = BindingIR(name="z", expr=block, location=loc)
    program = ProgramIR(statements=[binding], location=loc)

    tcx = TyCtxt()
    PreAutodiffPruningPass().run(program, tcx)

    assert isinstance(block.final_expr, IfExpressionIR)


def test_pre_autodiff_pruning_preserves_if_branch_block_wrappers() -> None:
    loc = _loc()
    expr = IfExpressionIR(
        condition=BinaryOpIR(
            BinaryOp.GT,
            IdentifierIR("x", loc),
            LiteralIR(0, loc),
            loc,
        ),
        then_expr=BlockExpressionIR(statements=[], final_expr=IdentifierIR("dx", loc), location=loc),
        else_expr=BlockExpressionIR(statements=[], final_expr=LiteralIR(0, loc), location=loc),
        location=loc,
    )
    binding = BindingIR(name="y", expr=expr, location=loc)
    program = ProgramIR(statements=[binding], location=loc)

    tcx = TyCtxt()
    PreAutodiffPruningPass().run(program, tcx)

    assert isinstance(binding.expr, IfExpressionIR)
    assert isinstance(binding.expr.then_expr, BlockExpressionIR)
    assert isinstance(binding.expr.else_expr, BlockExpressionIR)
