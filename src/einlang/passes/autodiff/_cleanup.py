from __future__ import annotations

from typing import Any

from ._graph import _DefIdCollector
from ...ir.nodes import (
    ArrayComprehensionIR,
    ArrayLiteralIR,
    BindingIR,
    BinaryOpIR,
    BlockExpressionIR,
    BuiltinCallIR,
    CastExpressionIR,
    DifferentialIR,
    DiffRuleIR,
    EinsteinClauseIR,
    EinsteinIR,
    ExpressionIR,
    FunctionCallIR,
    FunctionValueIR,
    IdentifierIR,
    IfExpressionIR,
    IndexRestIR,
    IndexVarIR,
    InterpolatedStringIR,
    JaggedAccessIR,
    LambdaIR,
    LiteralIR,
    MatchExpressionIR,
    MemberAccessIR,
    PipelineExpressionIR,
    ProgramIR,
    RangeIR,
    RectangularAccessIR,
    ReductionExpressionIR,
    SelectAtArgmaxIR,
    TryExpressionIR,
    TupleAccessIR,
    TupleExpressionIR,
    UnaryOpIR,
    WhereExpressionIR,
)
from ...shared.types import Type, strip_differential_types_deep


class _TypeStripper(_DefIdCollector):
    """Walk IR in-place: strip DifferentialType from type_info."""

    @staticmethod
    def _st(ty: Any) -> Any:
        return strip_differential_types_deep(ty)

    def _se(self, n: ExpressionIR) -> None:
        if n.type_info is not None:
            n.type_info = self._st(n.type_info)

    def visit_literal(self, n: LiteralIR) -> None:
        self._se(n)

    def visit_identifier(self, n: IdentifierIR) -> None:
        self._se(n)
        if n.defid is not None:
            self.defids.add(n.defid)

    def visit_binary_op(self, n: BinaryOpIR) -> None:
        self._se(n)
        n.left.accept(self)
        n.right.accept(self)

    def visit_unary_op(self, n: UnaryOpIR) -> None:
        self._se(n)
        n.operand.accept(self)

    def visit_builtin_call(self, n: BuiltinCallIR) -> None:
        self._se(n)
        for a in n.args or []:
            a.accept(self)

    def visit_function_call(self, n: FunctionCallIR) -> None:
        self._se(n)
        if n.callee_expr is not None:
            n.callee_expr.accept(self)
        for a in n.arguments or []:
            a.accept(self)

    def visit_rectangular_access(self, n: RectangularAccessIR) -> None:
        self._se(n)
        n.array.accept(self)
        for i in n.indices or []:
            i.accept(self)

    def visit_jagged_access(self, n: JaggedAccessIR) -> None:
        self._se(n)
        if n.base is not None:
            n.base.accept(self)
        for idx in n.index_chain or []:
            idx.accept(self)

    def visit_block_expression(self, n: BlockExpressionIR) -> None:
        self._se(n)
        for s in n.statements or []:
            if isinstance(s, (BindingIR, ExpressionIR)):
                s.accept(self)
        if n.final_expr is not None:
            n.final_expr.accept(self)

    def visit_if_expression(self, n: IfExpressionIR) -> None:
        self._se(n)
        n.condition.accept(self)
        n.then_expr.accept(self)
        if n.else_expr is not None:
            n.else_expr.accept(self)

    def visit_cast_expression(self, n: CastExpressionIR) -> None:
        self._se(n)
        tt = n.target_type
        if tt is not None and isinstance(tt, Type):
            n.target_type = self._st(tt)
        n.expr.accept(self)

    def visit_differential(self, n: DifferentialIR) -> None:
        self._se(n)
        n.operand.accept(self)

    def visit_lambda(self, n: LambdaIR) -> None:
        self._se(n)
        for p in n.parameters or []:
            if p.param_type is not None:
                p.param_type = self._st(p.param_type)
        n.body.accept(self)

    def visit_range(self, n: RangeIR) -> None:
        self._se(n)
        n.start.accept(self)
        n.end.accept(self)

    def visit_reduction_expression(self, n: ReductionExpressionIR) -> None:
        self._se(n)
        for lv in n.loop_vars or []:
            lv.accept(self)
        n.body.accept(self)
        if n.where_clause is not None:
            for c in n.where_clause.constraints or []:
                c.accept(self)

    def visit_where_expression(self, n: WhereExpressionIR) -> None:
        self._se(n)
        n.expr.accept(self)
        for c in n.constraints or []:
            c.accept(self)

    def visit_pipeline_expression(self, n: PipelineExpressionIR) -> None:
        self._se(n)
        n.left.accept(self)
        n.right.accept(self)

    def visit_array_comprehension(self, n: ArrayComprehensionIR) -> None:
        self._se(n)
        for v in n.loop_vars or []:
            v.accept(self)
        for r in n.ranges or []:
            r.accept(self)
        for c in n.constraints or []:
            c.accept(self)
        n.body.accept(self)

    def visit_array_literal(self, n: ArrayLiteralIR) -> None:
        self._se(n)
        for e in n.elements or []:
            e.accept(self)

    def visit_tuple_expression(self, n: TupleExpressionIR) -> None:
        self._se(n)
        for e in n.elements or []:
            e.accept(self)

    def visit_tuple_access(self, n: TupleAccessIR) -> None:
        self._se(n)
        n.tuple_expr.accept(self)

    def visit_member_access(self, n: MemberAccessIR) -> None:
        self._se(n)
        n.object.accept(self)

    def visit_function_value(self, n: FunctionValueIR) -> None:
        self._se(n)
        if n.return_type is not None:
            object.__setattr__(n, "return_type", self._st(n.return_type))
        for p in n.parameters or []:
            if p.param_type is not None:
                p.param_type = self._st(p.param_type)
        if n.body is not None:
            n.body.accept(self)

    def visit_try_expression(self, n: TryExpressionIR) -> None:
        self._se(n)
        n.operand.accept(self)

    def visit_match_expression(self, n: MatchExpressionIR) -> None:
        self._se(n)
        n.scrutinee.accept(self)
        for arm in n.arms or []:
            if getattr(arm, "body", None) is not None:
                arm.body.accept(self)

    def visit_interpolated_string(self, n: InterpolatedStringIR) -> None:
        self._se(n)
        for p in n.parts or []:
            if isinstance(p, ExpressionIR):
                p.accept(self)

    def visit_binding(self, n: BindingIR) -> None:
        if n.type_info is not None:
            n.type_info = self._st(n.type_info)
        if n.expr is not None:
            n.expr.accept(self)

    def visit_einstein(self, n: EinsteinIR) -> None:
        self._se(n)
        et = n.element_type
        if et is not None and isinstance(et, Type):
            n.element_type = self._st(et)
        for c in n.clauses or []:
            if isinstance(c, EinsteinClauseIR):
                c.accept(self)

    def visit_einstein_clause(self, n: EinsteinClauseIR) -> None:
        for idx in n.indices or []:
            if isinstance(idx, ExpressionIR):
                idx.accept(self)
        if n.value is not None:
            n.value.accept(self)
        if n.where_clause is not None:
            for c in n.where_clause.constraints or []:
                c.accept(self)

    def visit_select_at_argmax(self, n: SelectAtArgmaxIR) -> None:
        self._se(n)
        if n.primal_body is not None:
            n.primal_body.accept(self)
        if n.diff_body is not None:
            n.diff_body.accept(self)

    def visit_index_var(self, n: IndexVarIR) -> None:
        self._se(n)
        if n.range_ir is not None:
            n.range_ir.accept(self)

    def visit_index_rest(self, n: IndexRestIR) -> None:
        self._se(n)
        if getattr(n, "defid", None) is not None:
            self.defids.add(n.defid)


class _CleanupVisitor(_TypeStripper):
    """Post-pass: clear custom_diff_body, drop DiffRuleIR, strip DifferentialType."""

    def visit_function_value(self, n: FunctionValueIR) -> None:
        cdb = getattr(n, "custom_diff_body", None)
        if cdb is not None:
            cdb.accept(self)
            object.__setattr__(n, "custom_diff_body", None)
        super().visit_function_value(n)

    def visit_program(self, n: ProgramIR) -> None:
        n.statements = [s for s in (n.statements or []) if not isinstance(s, DiffRuleIR)]
        n.bindings = [s for s in n.statements if isinstance(s, BindingIR)]
        for s in n.statements or []:
            if isinstance(s, (BindingIR, ExpressionIR)):
                s.accept(self)
        for mod in n.modules or []:
            mod.accept(self)

    def visit_module(self, n: Any) -> None:
        for b in n.functions or []:
            b.accept(self)
        for b in n.constants or []:
            b.accept(self)
        for sub in n.submodules or []:
            sub.accept(self)

    def visit_diff_rule(self, n: DiffRuleIR) -> None:
        if n.body is not None:
            n.body.accept(self)


def clear_custom_diff_body_everywhere(program: ProgramIR) -> None:
    """Public API: clear autodiff-only IR artefacts."""
    program.accept(_CleanupVisitor())


__all__ = ["_CleanupVisitor", "_TypeStripper", "clear_custom_diff_body_everywhere"]
