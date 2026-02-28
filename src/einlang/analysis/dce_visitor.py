"""
Dead Code Elimination visitor for specialized function bodies.

Prunes unreachable if-else branches by evaluating conditions that become
compile-time constants after monomorphization (e.g. len(data.shape) → 2).

Only used on already-cloned specialized copies — never mutates generic IR.
"""

from typing import Optional, Any
from ..ir.scoped_visitor import ScopedIRVisitor
from ..ir.nodes import (
    ExpressionIR,
    LiteralIR, IdentifierIR, BinaryOpIR, UnaryOpIR,
    FunctionCallIR, BindingIR,
    RectangularAccessIR, JaggedAccessIR,
    BlockExpressionIR, IfExpressionIR, LambdaIR,
    RangeIR, ArrayComprehensionIR, ArrayLiteralIR,
    TupleExpressionIR, TupleAccessIR, InterpolatedStringIR,
    CastExpressionIR, MemberAccessIR,
    TryExpressionIR, MatchExpressionIR,
    ReductionExpressionIR, WhereExpressionIR,
    ArrowExpressionIR, PipelineExpressionIR,
    BuiltinCallIR,
    is_function_binding, is_einstein_binding,
    LiteralPatternIR, IdentifierPatternIR, WildcardPatternIR,
    TuplePatternIR, ArrayPatternIR, RestPatternIR, GuardPatternIR,
    ProgramIR,
)
from ..shared.types import BinaryOp, RectangularType


class DCEVisitor(ScopedIRVisitor[Any]):
    """Visitor that prunes dead if-else branches on specialized function bodies.

    Extends ScopedIRVisitor for DefId-keyed scope management.  Derives
    parameter ranks from param_type on ParameterIR nodes (set during
    monomorphization) — no external precomputed map needed.
    """

    # ------------------------------------------------------------------
    # Compile-time evaluation helpers
    # ------------------------------------------------------------------

    def _rank_of(self, defid) -> Optional[int]:
        val = self.get_var(defid)
        if isinstance(val, RectangularType) and val.shape:
            return len(val.shape)
        return None

    def _eval_len_shape(self, node) -> Optional[int]:
        """Evaluate ``len(x.shape)`` → int when *x* has a known RectangularType."""
        if not isinstance(node, (BuiltinCallIR, FunctionCallIR)):
            return None
        fname = getattr(node, 'builtin_name', None) or getattr(node, 'function_name', None)
        if fname != 'len':
            return None
        args = getattr(node, 'args', None) or getattr(node, 'arguments', None) or []
        if len(args) != 1:
            return None
        arg = args[0]
        if isinstance(arg, MemberAccessIR) and getattr(arg, 'member', None) == 'shape':
            did = getattr(arg.object, 'defid', None)
            if did is not None:
                rank = self._rank_of(did)
                if rank is not None:
                    return rank
        return None

    def _try_eval(self, node) -> Optional[Any]:
        """Try to evaluate *node* to a Python constant."""
        if isinstance(node, LiteralIR):
            return node.value
        did = getattr(node, 'defid', None)
        if did is not None:
            val = self.get_var(did)
            if isinstance(val, (int, float, bool)):
                return val
        v = self._eval_len_shape(node)
        if v is not None:
            return v
        if isinstance(node, BinaryOpIR):
            lv = self._try_eval(node.left)
            rv = self._try_eval(node.right)
            if lv is not None and rv is not None:
                op = node.operator
                try:
                    if op == BinaryOp.ADD: return lv + rv
                    if op == BinaryOp.SUB: return lv - rv
                    if op == BinaryOp.EQ: return lv == rv
                    if op == BinaryOp.NE: return lv != rv
                    if op == BinaryOp.LT: return lv < rv
                    if op == BinaryOp.GT: return lv > rv
                    if op == BinaryOp.LE: return lv <= rv
                    if op == BinaryOp.GE: return lv >= rv
                    if op == BinaryOp.AND: return bool(lv) and bool(rv)
                    if op == BinaryOp.OR: return bool(lv) or bool(rv)
                except Exception:
                    pass
        return None

    # ------------------------------------------------------------------
    # Core transforming visitors
    # ------------------------------------------------------------------

    def visit_binding(self, node: BindingIR):
        if is_function_binding(node):
            with self.scope():
                for p in node.parameters:
                    pt = getattr(p, 'param_type', None)
                    if pt is not None and p.defid is not None:
                        self.set_var(p.defid, pt)
                if node.body:
                    new_body = node.body.accept(self)
                    if new_body is not None and new_body is not node.body:
                        object.__setattr__(node.expr, 'body', new_body)
            return node
        if is_einstein_binding(node):
            return node
        if node.value:
            new_val = node.value.accept(self)
            did = getattr(node, 'defid', None)
            if did is not None:
                cv = self._try_eval(new_val)
                if cv is not None:
                    self.set_var(did, cv)
            if new_val is not node.value:
                object.__setattr__(node, 'expr', new_val)
        return node

    def visit_if_expression(self, node: IfExpressionIR):
        cond_val = self._try_eval(node.condition)
        if cond_val is not None:
            if bool(cond_val):
                return node.then_expr.accept(self)
            elif node.else_expr is not None:
                return node.else_expr.accept(self)
        return IfExpressionIR(
            condition=node.condition.accept(self),
            then_expr=node.then_expr.accept(self) if node.then_expr else None,
            else_expr=node.else_expr.accept(self) if node.else_expr else None,
            location=node.location,
            type_info=getattr(node, 'type_info', None),
            shape_info=getattr(node, 'shape_info', None),
        )

    def visit_block_expression(self, node: BlockExpressionIR):
        with self.scope():
            new_stmts = [s.accept(self) for s in (node.statements or [])]
            new_final = node.final_expr.accept(self) if node.final_expr else None
        return BlockExpressionIR(
            statements=new_stmts,
            final_expr=new_final,
            location=node.location,
            type_info=getattr(node, 'type_info', None),
            shape_info=getattr(node, 'shape_info', None),
        )

    # ------------------------------------------------------------------
    # Identity visitors — return node unchanged, required by IRVisitor ABC
    # ------------------------------------------------------------------

    def visit_literal(self, node: LiteralIR): return node
    def visit_identifier(self, node: IdentifierIR): return node
    def visit_binary_op(self, node: BinaryOpIR): return node
    def visit_unary_op(self, node: UnaryOpIR): return node
    def visit_function_call(self, node: FunctionCallIR): return node
    def visit_rectangular_access(self, node: RectangularAccessIR): return node
    def visit_jagged_access(self, node: JaggedAccessIR): return node
    def visit_lambda(self, node: LambdaIR): return node
    def visit_range(self, node: RangeIR): return node
    def visit_array_comprehension(self, node: ArrayComprehensionIR): return node
    def visit_module(self, node): return node
    def visit_array_literal(self, node: ArrayLiteralIR): return node
    def visit_tuple_expression(self, node: TupleExpressionIR): return node
    def visit_tuple_access(self, node: TupleAccessIR): return node
    def visit_interpolated_string(self, node: InterpolatedStringIR): return node
    def visit_cast_expression(self, node: CastExpressionIR): return node
    def visit_member_access(self, node: MemberAccessIR): return node
    def visit_try_expression(self, node: TryExpressionIR): return node
    def visit_match_expression(self, node: MatchExpressionIR): return node
    def visit_reduction_expression(self, node: ReductionExpressionIR): return node
    def visit_where_expression(self, node: WhereExpressionIR): return node
    def visit_arrow_expression(self, node: ArrowExpressionIR): return node
    def visit_pipeline_expression(self, node: PipelineExpressionIR): return node
    def visit_builtin_call(self, node: BuiltinCallIR): return node
    def visit_literal_pattern(self, node: LiteralPatternIR): return node
    def visit_identifier_pattern(self, node: IdentifierPatternIR): return node
    def visit_wildcard_pattern(self, node: WildcardPatternIR): return node
    def visit_tuple_pattern(self, node: TuplePatternIR): return node
    def visit_array_pattern(self, node: ArrayPatternIR): return node
    def visit_rest_pattern(self, node: RestPatternIR): return node
    def visit_guard_pattern(self, node: GuardPatternIR): return node
    def visit_program(self, node: ProgramIR): return node
