"""
Autodiff pass v2: expand @ and @y/@x into plain IR (forward diff).

Forward diff only: from dx to dy. For each binding y = f(x1, x2, ...)
we emit d_y = (df/dx1)*d_x1 + (df/dx2)*d_x2 + ... in execution order.

- DifferentialIR (@expr): @x -> d_x.
- @y/@x: derivative (dy/dx). Direct dep -> symbolic Jacobian; transitive -> quotient form.
"""

from typing import Any, Dict, List, Optional, Set, Tuple, Union

from .base import BasePass, TyCtxt
from .type_inference import TypeInferencePass
from .shape_analysis import UnifiedShapeAnalysisPass
from ..ir.nodes import (
    ArrayLiteralIR,
    ProgramIR,
    BindingIR,
    BlockExpressionIR,
    CastExpressionIR,
    DifferentialIR,
    BinaryOpIR,
    UnaryOpIR,
    BuiltinCallIR,
    LiteralIR,
    IdentifierIR,
    ExpressionIR,
    IRVisitor,
    IfExpressionIR,
    InterpolatedStringIR,
    LambdaIR,
    RectangularAccessIR,
    IndexVarIR,
    RangeIR,
    MemberAccessIR,
    FunctionCallIR,
    FunctionValueIR,
    EinsteinIR,
    EinsteinClauseIR,
    ReductionExpressionIR,
    SelectAtArgmaxIR,
    WhereExpressionIR,
    WhereClauseIR,
    IndexRestIR,
    DiffRuleIR,
    TupleExpressionIR,
    TupleAccessIR,
    MatchExpressionIR,
    PipelineExpressionIR,
    ArrayComprehensionIR,
    TryExpressionIR,
    JaggedAccessIR,
    is_function_binding,
)
from ..shared.types import BinaryOp, UnaryOp, PrimitiveType, UNKNOWN, F32, STR, BOOL, Type
from ..shared.types import strip_differential_types_deep
from ..shared.types import ReductionOp
from ..shared.defid import DefId
from ..shared.source_location import SourceLocation


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _float_lit(value: Any, loc: SourceLocation) -> LiteralIR:
    return LiteralIR(value, loc, type_info=F32)


def _simplify(expr: ExpressionIR, loc: SourceLocation) -> ExpressionIR:
    """Algebraic simplification: 0+x->x, 0*x->0, 1*x->x, x**1->x, constant fold."""
    if not isinstance(expr, BinaryOpIR):
        if isinstance(expr, UnaryOpIR):
            return UnaryOpIR(expr.operator, _simplify(expr.operand, loc), expr.location,
                             type_info=getattr(expr, "type_info", None),
                             shape_info=getattr(expr, "shape_info", None))
        return expr
    left = _simplify(expr.left, loc)
    right = _simplify(expr.right, loc)
    op = expr.operator
    if op == BinaryOp.ADD:
        if isinstance(right, LiteralIR) and right.value == 0:
            return left
        if isinstance(left, LiteralIR) and left.value == 0:
            return right
        if isinstance(left, LiteralIR) and isinstance(right, LiteralIR):
            try:
                return LiteralIR(left.value + right.value, loc,
                                 type_info=getattr(left, "type_info", None))
            except TypeError:
                pass
    if op == BinaryOp.SUB:
        if isinstance(right, LiteralIR) and right.value == 0:
            return left
        if isinstance(left, LiteralIR) and isinstance(right, LiteralIR):
            try:
                return LiteralIR(left.value - right.value, loc,
                                 type_info=getattr(left, "type_info", None))
            except TypeError:
                pass
    if op == BinaryOp.MUL:
        if isinstance(left, LiteralIR) and left.value == 0:
            return _float_lit(0, loc)
        if isinstance(right, LiteralIR) and right.value == 0:
            return _float_lit(0, loc)
        if isinstance(left, LiteralIR) and left.value == 1:
            return right
        if isinstance(right, LiteralIR) and right.value == 1:
            return left
    if op == BinaryOp.POW and isinstance(right, LiteralIR) and right.value == 1:
        return left
    return BinaryOpIR(op, left, right, expr.location,
                      type_info=getattr(expr, "type_info", None),
                      shape_info=getattr(expr, "shape_info", None))


def _differential_target_from_operand(operand: ExpressionIR) -> Optional[Tuple[DefId, str]]:
    if isinstance(operand, IdentifierIR) and operand.defid is not None:
        return (operand.defid, operand.name or "")
    return None


def _expr_to_diff_source(
    expr: ExpressionIR,
    d_defid_to_at_name: Dict[DefId, str],
    scope_binding_by_defid: Dict[DefId, Any],
) -> str:
    if isinstance(expr, IdentifierIR) and expr.defid is not None:
        if expr.defid in d_defid_to_at_name:
            return d_defid_to_at_name[expr.defid]
        b = scope_binding_by_defid.get(expr.defid)
        return (b.name or "?") if b else "?"
    if isinstance(expr, LiteralIR):
        return str(expr.value)
    if isinstance(expr, BinaryOpIR):
        op_str = getattr(expr.operator, "value", str(expr.operator))
        left_s = _expr_to_diff_source(expr.left, d_defid_to_at_name, scope_binding_by_defid)
        right_s = _expr_to_diff_source(expr.right, d_defid_to_at_name, scope_binding_by_defid)
        if expr.operator == BinaryOp.POW and not isinstance(expr.right, (LiteralIR, IdentifierIR)):
            right_s = "(" + right_s + ")"
        return left_s + op_str + right_s
    if isinstance(expr, UnaryOpIR):
        op_str = getattr(expr.operator, "value", str(expr.operator))
        inner = _expr_to_diff_source(expr.operand, d_defid_to_at_name, scope_binding_by_defid)
        return op_str + inner
    return "?"


def _flatten_product(expr: ExpressionIR) -> Optional[List[Tuple[ExpressionIR, List[Any]]]]:
    """Extract (array_ident, [indices]) factors from a product of RectangularAccessIR."""
    if isinstance(expr, RectangularAccessIR):
        arr = expr.array
        indices = list(expr.indices or [])
        if isinstance(arr, IdentifierIR) and arr.defid is not None:
            return [(arr, indices)]
        return None
    if isinstance(expr, BinaryOpIR) and expr.operator == BinaryOp.MUL:
        left = _flatten_product(expr.left)
        right = _flatten_product(expr.right)
        if left is not None and right is not None:
            return left + right
    return None


def _is_reachable(source_defid: DefId, target_defid: DefId,
                  binding_by_defid: Dict[DefId, BindingIR]) -> bool:
    """BFS: is target_defid reachable from source_defid via binding dependencies?"""
    visited: Set[DefId] = set()
    queue = [source_defid]
    while queue:
        cur = queue.pop()
        if cur == target_defid:
            return True
        if cur in visited:
            continue
        visited.add(cur)
        b = binding_by_defid.get(cur)
        if b is not None and b.expr is not None:
            for dep in _collect_defids(b.expr):
                if dep not in visited:
                    queue.append(dep)
    return False


def _path_has_einstein(source_defid: DefId, target_defid: DefId,
                       binding_by_defid: Dict[DefId, BindingIR]) -> bool:
    """BFS: does any binding on the path from source to target have an EinsteinIR expr?"""
    visited: Set[DefId] = set()
    queue = [source_defid]
    while queue:
        cur = queue.pop()
        if cur == target_defid:
            continue
        if cur in visited:
            continue
        visited.add(cur)
        b = binding_by_defid.get(cur)
        if b is not None and b.expr is not None:
            if isinstance(b.expr, EinsteinIR):
                return True
            for dep in _collect_defids(b.expr):
                if dep not in visited:
                    queue.append(dep)
    return False


def _set_type_info(expr: Any, type_info: Any, shape_info: Any) -> None:
    """Propagate type_info/shape_info onto expr tree (in-place, only fills None slots)."""
    if expr is None:
        return
    if hasattr(expr, "type_info") and getattr(expr, "type_info", None) is None and type_info is not None:
        expr.type_info = type_info
    if hasattr(expr, "shape_info") and getattr(expr, "shape_info", None) is None and shape_info is not None:
        expr.shape_info = shape_info
    if isinstance(expr, BinaryOpIR):
        _set_type_info(expr.left, type_info, shape_info)
        _set_type_info(expr.right, type_info, shape_info)
    elif isinstance(expr, UnaryOpIR):
        _set_type_info(expr.operand, type_info, shape_info)
    elif isinstance(expr, BlockExpressionIR):
        for s in expr.statements or []:
            if isinstance(s, ExpressionIR):
                _set_type_info(s, type_info, shape_info)
        _set_type_info(expr.final_expr, type_info, shape_info)
    elif isinstance(expr, EinsteinIR):
        for c in expr.clauses or []:
            for idx in getattr(c, "indices", None) or []:
                _set_type_info(idx, type_info, shape_info)
            if getattr(c, "value", None) is not None:
                _set_type_info(c.value, type_info, shape_info)
    elif isinstance(expr, ReductionExpressionIR):
        _set_type_info(expr.body, type_info, shape_info)
        if getattr(expr, "where_clause", None) is not None:
            _set_type_info(expr.where_clause, type_info, shape_info)
    elif isinstance(expr, SelectAtArgmaxIR):
        _set_type_info(expr.primal_body, type_info, shape_info)
        _set_type_info(expr.diff_body, type_info, shape_info)
    elif isinstance(expr, RectangularAccessIR):
        _set_type_info(expr.array, type_info, shape_info)
        for idx in expr.indices or []:
            if isinstance(idx, ExpressionIR):
                _set_type_info(idx, type_info, shape_info)
    elif isinstance(expr, WhereClauseIR):
        for c in getattr(expr, "constraints", None) or []:
            _set_type_info(c, type_info, shape_info)


# ---------------------------------------------------------------------------
# Visitor 1: _DefIdCollector
# ---------------------------------------------------------------------------

class _DefIdCollector(IRVisitor[None]):
    """Collect all DefIds referenced in an expression tree."""

    def __init__(self) -> None:
        self.defids: Set[DefId] = set()

    def visit_identifier(self, node: IdentifierIR) -> None:
        if node.defid is not None:
            self.defids.add(node.defid)

    def visit_literal(self, node: LiteralIR) -> None:
        pass

    def visit_binary_op(self, node: BinaryOpIR) -> None:
        node.left.accept(self)
        node.right.accept(self)

    def visit_unary_op(self, node: UnaryOpIR) -> None:
        node.operand.accept(self)

    def visit_builtin_call(self, node: BuiltinCallIR) -> None:
        for a in node.args or []:
            a.accept(self)

    def visit_function_call(self, node: FunctionCallIR) -> None:
        if node.callee_expr is not None:
            node.callee_expr.accept(self)
        for a in node.arguments or []:
            a.accept(self)

    def visit_rectangular_access(self, node: RectangularAccessIR) -> None:
        node.array.accept(self)
        for i in node.indices or []:
            i.accept(self)

    def visit_jagged_access(self, node: Any) -> None:
        if node.base is not None:
            node.base.accept(self)

    def visit_block_expression(self, node: BlockExpressionIR) -> None:
        for s in node.statements or []:
            if hasattr(s, "accept"):
                s.accept(self)
        if node.final_expr is not None:
            node.final_expr.accept(self)

    def visit_if_expression(self, node: IfExpressionIR) -> None:
        node.condition.accept(self)
        node.then_expr.accept(self)
        if node.else_expr is not None:
            node.else_expr.accept(self)

    def visit_cast_expression(self, node: CastExpressionIR) -> None:
        node.expr.accept(self)

    def visit_differential(self, node: DifferentialIR) -> None:
        node.operand.accept(self)

    def visit_lambda(self, node: Any) -> None:
        node.body.accept(self)

    def visit_range(self, node: RangeIR) -> None:
        node.start.accept(self)
        node.end.accept(self)

    def visit_reduction_expression(self, node: ReductionExpressionIR) -> None:
        node.body.accept(self)

    def visit_where_expression(self, node: Any) -> None:
        node.expr.accept(self)
        for c in node.constraints or []:
            c.accept(self)

    def visit_pipeline_expression(self, node: Any) -> None:
        node.left.accept(self)
        node.right.accept(self)

    def visit_array_comprehension(self, node: Any) -> None:
        node.body.accept(self)

    def visit_array_literal(self, node: ArrayLiteralIR) -> None:
        for e in node.elements or []:
            e.accept(self)

    def visit_tuple_expression(self, node: Any) -> None:
        for e in node.elements or []:
            e.accept(self)

    def visit_tuple_access(self, node: Any) -> None:
        node.tuple_expr.accept(self)

    def visit_member_access(self, node: MemberAccessIR) -> None:
        node.object.accept(self)

    def visit_function_value(self, node: FunctionValueIR) -> None:
        if node.body is not None:
            node.body.accept(self)

    def visit_try_expression(self, node: Any) -> None:
        node.operand.accept(self)

    def visit_match_expression(self, node: Any) -> None:
        node.scrutinee.accept(self)
        for arm in node.arms or []:
            if getattr(arm, "body", None) is not None:
                arm.body.accept(self)

    def visit_interpolated_string(self, node: Any) -> None:
        pass

    def visit_binding(self, node: BindingIR) -> None:
        if node.expr is not None:
            node.expr.accept(self)

    def visit_program(self, node: ProgramIR) -> None:
        for b in node.bindings or []:
            b.accept(self)

    def visit_einstein(self, node: EinsteinIR) -> None:
        for c in node.clauses or []:
            if hasattr(c, "accept"):
                c.accept(self)

    def visit_einstein_clause(self, node: EinsteinClauseIR) -> None:
        if node.value is not None:
            node.value.accept(self)

    def visit_select_at_argmax(self, node: SelectAtArgmaxIR) -> None:
        if node.primal_body is not None:
            node.primal_body.accept(self)
        if node.diff_body is not None:
            node.diff_body.accept(self)

    def visit_module(self, node: Any) -> None:
        pass

    def visit_index_var(self, node: Any) -> None:
        pass

    def visit_index_rest(self, node: Any) -> None:
        pass

    def visit_identifier_pattern(self, node: Any) -> None:
        pass

    def visit_wildcard_pattern(self, node: Any) -> None:
        pass

    def visit_literal_pattern(self, node: Any) -> None:
        pass

    def visit_tuple_pattern(self, node: Any) -> None:
        pass

    def visit_array_pattern(self, node: Any) -> None:
        pass

    def visit_rest_pattern(self, node: Any) -> None:
        pass

    def visit_guard_pattern(self, node: Any) -> None:
        pass

    def visit_or_pattern(self, node: Any) -> None:
        pass

    def visit_constructor_pattern(self, node: Any) -> None:
        pass

    def visit_binding_pattern(self, node: Any) -> None:
        pass

    def visit_range_pattern(self, node: Any) -> None:
        pass


class _StripDiffTypesWalker(_DefIdCollector):
    """Strip DifferentialType from type_info and type-bearing fields while walking the IR."""

    @staticmethod
    def _strip_ty(ty: Any) -> Any:
        return strip_differential_types_deep(ty)

    def _strip_expr(self, node: ExpressionIR) -> None:
        if node.type_info is not None:
            node.type_info = self._strip_ty(node.type_info)

    def visit_literal(self, node: LiteralIR) -> None:
        self._strip_expr(node)

    def visit_identifier(self, node: IdentifierIR) -> None:
        self._strip_expr(node)
        if node.defid is not None:
            self.defids.add(node.defid)

    def visit_binary_op(self, node: BinaryOpIR) -> None:
        self._strip_expr(node)
        node.left.accept(self)
        node.right.accept(self)

    def visit_unary_op(self, node: UnaryOpIR) -> None:
        self._strip_expr(node)
        node.operand.accept(self)

    def visit_builtin_call(self, node: BuiltinCallIR) -> None:
        self._strip_expr(node)
        for a in node.args or []:
            a.accept(self)

    def visit_function_call(self, node: FunctionCallIR) -> None:
        self._strip_expr(node)
        if node.callee_expr is not None:
            node.callee_expr.accept(self)
        for a in node.arguments or []:
            a.accept(self)

    def visit_rectangular_access(self, node: RectangularAccessIR) -> None:
        self._strip_expr(node)
        node.array.accept(self)
        for i in node.indices or []:
            i.accept(self)

    def visit_jagged_access(self, node: JaggedAccessIR) -> None:
        self._strip_expr(node)
        if node.base is not None:
            node.base.accept(self)
        for idx in node.index_chain or []:
            idx.accept(self)

    def visit_block_expression(self, node: BlockExpressionIR) -> None:
        self._strip_expr(node)
        for s in node.statements or []:
            if hasattr(s, "accept"):
                s.accept(self)
        if node.final_expr is not None:
            node.final_expr.accept(self)

    def visit_if_expression(self, node: IfExpressionIR) -> None:
        self._strip_expr(node)
        node.condition.accept(self)
        node.then_expr.accept(self)
        if node.else_expr is not None:
            node.else_expr.accept(self)

    def visit_cast_expression(self, node: CastExpressionIR) -> None:
        self._strip_expr(node)
        tt = node.target_type
        if tt is not None and isinstance(tt, Type):
            node.target_type = self._strip_ty(tt)
        node.expr.accept(self)

    def visit_differential(self, node: DifferentialIR) -> None:
        self._strip_expr(node)
        node.operand.accept(self)

    def visit_lambda(self, node: LambdaIR) -> None:
        self._strip_expr(node)
        for p in node.parameters or []:
            if p.param_type is not None:
                p.param_type = self._strip_ty(p.param_type)
        node.body.accept(self)

    def visit_range(self, node: RangeIR) -> None:
        self._strip_expr(node)
        node.start.accept(self)
        node.end.accept(self)

    def visit_reduction_expression(self, node: ReductionExpressionIR) -> None:
        self._strip_expr(node)
        for lv in node.loop_vars or []:
            lv.accept(self)
        node.body.accept(self)
        if node.where_clause is not None:
            for c in node.where_clause.constraints or []:
                c.accept(self)

    def visit_where_expression(self, node: WhereExpressionIR) -> None:
        self._strip_expr(node)
        node.expr.accept(self)
        for c in node.constraints or []:
            c.accept(self)

    def visit_pipeline_expression(self, node: PipelineExpressionIR) -> None:
        self._strip_expr(node)
        node.left.accept(self)
        node.right.accept(self)

    def visit_array_comprehension(self, node: ArrayComprehensionIR) -> None:
        self._strip_expr(node)
        for v in node.loop_vars or []:
            v.accept(self)
        for r in node.ranges or []:
            r.accept(self)
        for c in node.constraints or []:
            c.accept(self)
        node.body.accept(self)

    def visit_array_literal(self, node: ArrayLiteralIR) -> None:
        self._strip_expr(node)
        for e in node.elements or []:
            e.accept(self)

    def visit_tuple_expression(self, node: TupleExpressionIR) -> None:
        self._strip_expr(node)
        for e in node.elements or []:
            e.accept(self)

    def visit_tuple_access(self, node: TupleAccessIR) -> None:
        self._strip_expr(node)
        node.tuple_expr.accept(self)

    def visit_member_access(self, node: MemberAccessIR) -> None:
        self._strip_expr(node)
        node.object.accept(self)

    def visit_function_value(self, node: FunctionValueIR) -> None:
        self._strip_expr(node)
        if node.return_type is not None:
            object.__setattr__(node, "return_type", self._strip_ty(node.return_type))
        for p in node.parameters or []:
            if p.param_type is not None:
                p.param_type = self._strip_ty(p.param_type)
        if node.body is not None:
            node.body.accept(self)

    def visit_try_expression(self, node: TryExpressionIR) -> None:
        self._strip_expr(node)
        node.operand.accept(self)

    def visit_match_expression(self, node: MatchExpressionIR) -> None:
        self._strip_expr(node)
        node.scrutinee.accept(self)
        for arm in node.arms or []:
            if getattr(arm, "body", None) is not None:
                arm.body.accept(self)

    def visit_interpolated_string(self, node: InterpolatedStringIR) -> None:
        self._strip_expr(node)
        for p in node.parts or []:
            if isinstance(p, ExpressionIR):
                p.accept(self)

    def visit_binding(self, node: BindingIR) -> None:
        if node.type_info is not None:
            node.type_info = self._strip_ty(node.type_info)
        if node.expr is not None:
            node.expr.accept(self)

    def visit_einstein(self, node: EinsteinIR) -> None:
        self._strip_expr(node)
        et = node.element_type
        if et is not None and isinstance(et, Type):
            node.element_type = self._strip_ty(et)
        for c in node.clauses or []:
            if hasattr(c, "accept"):
                c.accept(self)

    def visit_einstein_clause(self, node: EinsteinClauseIR) -> None:
        for idx in node.indices or []:
            if isinstance(idx, ExpressionIR):
                idx.accept(self)
            elif isinstance(idx, (list, tuple)):
                for sub in idx:
                    if isinstance(sub, ExpressionIR):
                        sub.accept(self)
        if node.value is not None:
            node.value.accept(self)
        if node.where_clause is not None:
            for c in node.where_clause.constraints or []:
                c.accept(self)

    def visit_select_at_argmax(self, node: SelectAtArgmaxIR) -> None:
        self._strip_expr(node)
        if node.primal_body is not None:
            node.primal_body.accept(self)
        if node.diff_body is not None:
            node.diff_body.accept(self)

    def visit_index_var(self, node: IndexVarIR) -> None:
        self._strip_expr(node)
        if node.range_ir is not None:
            node.range_ir.accept(self)

    def visit_index_rest(self, node: IndexRestIR) -> None:
        self._strip_expr(node)
        if node.defid is not None:
            self.defids.add(node.defid)


class _ClearAutodiffArtifactsVisitor(_StripDiffTypesWalker):
    """
    After AutodiffPass: clear custom_diff bodies, drop DiffRuleIR statements,
    and strip DifferentialType from all IR type annotations.
    """

    def visit_function_value(self, node: FunctionValueIR) -> None:
        cdb = getattr(node, "custom_diff_body", None)
        if cdb is not None:
            cdb.accept(self)
            object.__setattr__(node, "custom_diff_body", None)
        super().visit_function_value(node)

    def visit_program(self, node: ProgramIR) -> None:
        node.statements = [s for s in (node.statements or []) if not isinstance(s, DiffRuleIR)]
        node.bindings = [s for s in node.statements if isinstance(s, BindingIR)]
        for s in node.statements or []:
            if isinstance(s, BindingIR):
                s.accept(self)
            elif hasattr(s, "accept"):
                s.accept(self)
        for mod in node.modules or []:
            mod.accept(self)

    def visit_module(self, node: Any) -> None:
        for b in node.functions or []:
            b.accept(self)
        for b in node.constants or []:
            b.accept(self)
        for sub in node.submodules or []:
            sub.accept(self)

    def visit_diff_rule(self, node: DiffRuleIR) -> None:
        if node.body is not None:
            node.body.accept(self)


def clear_custom_diff_body_everywhere(program: ProgramIR) -> None:
    """Clear autodiff-only IR: custom_diff_body, DiffRuleIR stmts, DifferentialType annotations."""
    program.accept(_ClearAutodiffArtifactsVisitor())


def _collect_defids(expr: Optional[ExpressionIR]) -> Set[DefId]:
    if expr is None:
        return set()
    c = _DefIdCollector()
    expr.accept(c)
    return c.defids


# ---------------------------------------------------------------------------
# Visitor 2: _SymbolicDiffVisitor  (d(expr)/d(wrt))
# ---------------------------------------------------------------------------

class _SymbolicDiffVisitor(IRVisitor[ExpressionIR]):
    """Compute symbolic partial derivative d(expr)/d(wrt_defid).

    Each visit_* method returns the derivative expression (IR).
    """

    def __init__(self, wrt_defid: DefId, loc: SourceLocation,
                 binding_by_defid: Dict[DefId, BindingIR],
                 resolver: Any) -> None:
        self._wrt = wrt_defid
        self._loc = loc
        self._bindings = binding_by_defid
        self._resolver = resolver

    # -- atoms --

    def visit_identifier(self, node: IdentifierIR) -> ExpressionIR:
        if node.defid == self._wrt:
            return _float_lit(1, self._loc)
        if node.defid is not None:
            b = self._bindings.get(node.defid)
            if b is not None and b.expr is not None:
                return b.expr.accept(self)
        return _float_lit(0, self._loc)

    def visit_literal(self, node: LiteralIR) -> ExpressionIR:
        return _float_lit(0, self._loc)

    def visit_array_literal(self, node: ArrayLiteralIR) -> ExpressionIR:
        return _float_lit(0, self._loc)

    # -- binary --

    def visit_binary_op(self, node: BinaryOpIR) -> ExpressionIR:
        loc = node.location or self._loc
        d_left = node.left.accept(self)
        d_right = node.right.accept(self)
        op = node.operator
        if op == BinaryOp.ADD:
            return BinaryOpIR(BinaryOp.ADD, d_left, d_right, loc)
        if op == BinaryOp.SUB:
            return BinaryOpIR(BinaryOp.SUB, d_left, d_right, loc)
        if op == BinaryOp.MUL:
            return BinaryOpIR(
                BinaryOp.ADD,
                BinaryOpIR(BinaryOp.MUL, d_left, node.right, loc),
                BinaryOpIR(BinaryOp.MUL, node.left, d_right, loc),
                loc,
            )
        if op == BinaryOp.DIV:
            num = BinaryOpIR(
                BinaryOp.SUB,
                BinaryOpIR(BinaryOp.MUL, d_left, node.right, loc),
                BinaryOpIR(BinaryOp.MUL, node.left, d_right, loc),
                loc,
            )
            den = BinaryOpIR(BinaryOp.POW, node.right, _float_lit(2, loc), loc)
            return BinaryOpIR(BinaryOp.DIV, num, den, loc)
        if op == BinaryOp.POW and isinstance(node.right, LiteralIR):
            c = node.right.value
            if isinstance(c, (int, float)) and c != 0:
                bm1 = _float_lit(c - 1, loc)
                a_bm1 = BinaryOpIR(BinaryOp.POW, node.left, bm1, loc)
                return BinaryOpIR(
                    BinaryOp.MUL,
                    BinaryOpIR(BinaryOp.MUL, _float_lit(c, loc), a_bm1, loc),
                    d_left,
                    loc,
                )
        if op == BinaryOp.MOD:
            return d_left
        raise ValueError(f"Autodiff: unsupported binary op in d/d(wrt): {op}")

    # -- unary --

    def visit_unary_op(self, node: UnaryOpIR) -> ExpressionIR:
        d_op = node.operand.accept(self)
        if node.operator == UnaryOp.NEG:
            return UnaryOpIR(UnaryOp.NEG, d_op, node.location or self._loc)
        if node.operator == UnaryOp.POS:
            return d_op
        raise ValueError(f"Autodiff: unsupported unary op in d/d(wrt): {node.operator}")

    # -- function call --

    def visit_function_call(self, node: FunctionCallIR) -> ExpressionIR:
        loc = node.location or self._loc
        callee_defid = node.function_defid
        args = node.arguments or []

        if callee_defid is None or callee_defid not in self._bindings:
            callee_name = None
            if getattr(node, "callee_expr", None) and isinstance(node.callee_expr, IdentifierIR):
                callee_name = node.callee_expr.name
            if callee_name and len(args) == 1:
                d_arg = args[0].accept(self)
                if callee_name == "exp":
                    return BinaryOpIR(BinaryOp.MUL, node, d_arg, loc)
                if callee_name == "ln":
                    return BinaryOpIR(BinaryOp.DIV, d_arg, args[0], loc)
            raise ValueError(
                "Autodiff: function call callee is not a known user function"
            )
        binding = self._bindings[callee_defid]

        rule_body = getattr(binding.expr, 'custom_diff_body', None) if isinstance(binding.expr, FunctionValueIR) else None
        if rule_body is not None:
            fv = binding.expr
            params = fv.parameters or []
            if len(params) == len(args):
                replace_map = {p.defid: args[j] for j, p in enumerate(params) if p.defid is not None}
                terms: List[ExpressionIR] = []
                for i, param in enumerate(params):
                    if param.defid is None:
                        continue
                    diff_replace_i = {
                        params[j].defid: (_float_lit(1, loc) if j == i else _float_lit(0, loc))
                        for j in range(len(params)) if params[j].defid is not None
                    }
                    coef = _substitute_with_diffs(rule_body, replace_map, diff_replace_i, loc)
                    d_arg = args[i].accept(self)
                    terms.append(BinaryOpIR(BinaryOp.MUL, coef, d_arg, loc))
                if terms:
                    out = terms[0]
                    for t in terms[1:]:
                        out = BinaryOpIR(BinaryOp.ADD, out, t, loc)
                    return out

        if not isinstance(binding.expr, FunctionValueIR):
            raise ValueError("Autodiff: callee is not a function value")
        fv = binding.expr
        body = fv.body
        params = fv.parameters or []
        if body is None:
            raise ValueError("Autodiff: user function has no body")
        if len(params) != len(args):
            raise ValueError(f"Autodiff: arity mismatch: {len(args)} args vs {len(params)} params")

        replace_map = {p.defid: args[j] for j, p in enumerate(params) if p.defid is not None}
        terms = []
        for i, param in enumerate(params):
            if param.defid is None:
                raise ValueError("Autodiff: function parameter has no defid")
            inner_vis = _SymbolicDiffVisitor(param.defid, loc, self._bindings, self._resolver)
            partial = body.accept(inner_vis)
            partial_subst = _substitute(partial, replace_map, loc)
            d_arg = args[i].accept(self)
            terms.append(BinaryOpIR(BinaryOp.MUL, partial_subst, d_arg, loc))
        if not terms:
            return _float_lit(0, loc)
        out = terms[0]
        for t in terms[1:]:
            out = BinaryOpIR(BinaryOp.ADD, out, t, loc)
        return out

    # -- cast --

    def visit_cast_expression(self, node: CastExpressionIR) -> ExpressionIR:
        return node.expr.accept(self)

    # -- if --

    def visit_if_expression(self, node: IfExpressionIR) -> ExpressionIR:
        d_then = node.then_expr.accept(self)
        d_else = node.else_expr.accept(self) if node.else_expr is not None else _float_lit(0, self._loc)
        return IfExpressionIR(
            condition=node.condition,
            then_expr=d_then,
            else_expr=d_else,
            location=node.location or self._loc,
            type_info=getattr(node, "type_info", None),
            shape_info=getattr(node, "shape_info", None),
        )

    # -- rectangular access --

    def visit_rectangular_access(self, node: RectangularAccessIR) -> ExpressionIR:
        loc = node.location or self._loc
        d_array = node.array.accept(self)
        if isinstance(d_array, LiteralIR) and d_array.value == 0:
            return _float_lit(0, loc)
        indices = list(node.indices or [])
        if isinstance(d_array, EinsteinIR) and (d_array.clauses or []):
            clauses = d_array.clauses
            if len(clauses) == 1:
                c = clauses[0]
                clause_indices = c.indices or []
                if len(clause_indices) == len(indices):
                    replace_map: Dict[DefId, ExpressionIR] = {}
                    for j, cidx in enumerate(clause_indices):
                        if isinstance(cidx, (IndexVarIR, IdentifierIR)) and cidx.defid is not None and j < len(indices):
                            replace_map[cidx.defid] = indices[j]
                    inlined = _substitute(c.value, replace_map, loc)
                    ti = getattr(node, "type_info", None)
                    si = getattr(node, "shape_info", None)
                    if ti is not None or si is not None:
                        _set_type_info(inlined, ti, si)
                    return inlined
        return RectangularAccessIR(
            d_array, indices, loc,
            type_info=getattr(node, "type_info", None),
            shape_info=getattr(node, "shape_info", None),
        )

    # -- block --

    def visit_block_expression(self, node: BlockExpressionIR) -> ExpressionIR:
        if node.final_expr is not None:
            return node.final_expr.accept(self)
        raise ValueError("Autodiff: cannot differentiate block expression with no final expression")

    # -- reduction --

    def visit_reduction_expression(self, node: ReductionExpressionIR) -> ExpressionIR:
        loc = node.location or self._loc
        d_body = node.body.accept(self)
        op = node.operation
        if op == ReductionOp.SUM:
            return ReductionExpressionIR(
                ReductionOp.SUM, node.loop_vars, d_body, loc,
                where_clause=node.where_clause,
                loop_var_ranges=node.loop_var_ranges,
                type_info=getattr(node, "type_info", None),
                shape_info=getattr(node, "shape_info", None),
            )
        if op == ReductionOp.MAX:
            return SelectAtArgmaxIR(
                node.body, d_body, node.loop_vars,
                loop_var_ranges=node.loop_var_ranges,
                location=loc,
                type_info=getattr(node, "type_info", None),
                shape_info=getattr(node, "shape_info", None),
                use_argmin=False,
            )
        if op == ReductionOp.MIN:
            return SelectAtArgmaxIR(
                node.body, d_body, node.loop_vars,
                loop_var_ranges=node.loop_var_ranges,
                location=loc,
                type_info=getattr(node, "type_info", None),
                shape_info=getattr(node, "shape_info", None),
                use_argmin=True,
            )
        if op == ReductionOp.PROD:
            return BinaryOpIR(
                BinaryOp.MUL,
                BinaryOpIR(BinaryOp.DIV, node, node.body, loc),
                d_body,
                loc,
            )
        raise ValueError(f"Autodiff: unsupported reduction in d/d(wrt): {op}")

    # -- einstein --

    def visit_einstein(self, node: EinsteinIR) -> ExpressionIR:
        return _diff_einstein_wrt(node, self._wrt, self._loc, self._bindings, self._resolver)

    # -- stubs for non-differentiable nodes --

    def visit_jagged_access(self, node: Any) -> ExpressionIR:
        return _float_lit(0, self._loc)

    def visit_lambda(self, node: Any) -> ExpressionIR:
        raise ValueError("Autodiff: cannot differentiate lambda")

    def visit_range(self, node: RangeIR) -> ExpressionIR:
        return _float_lit(0, self._loc)

    def visit_array_comprehension(self, node: Any) -> ExpressionIR:
        return _float_lit(0, self._loc)

    def visit_tuple_expression(self, node: Any) -> ExpressionIR:
        return _float_lit(0, self._loc)

    def visit_tuple_access(self, node: Any) -> ExpressionIR:
        return _float_lit(0, self._loc)

    def visit_interpolated_string(self, node: Any) -> ExpressionIR:
        return _float_lit(0, self._loc)

    def visit_member_access(self, node: MemberAccessIR) -> ExpressionIR:
        return _float_lit(0, self._loc)

    def visit_try_expression(self, node: Any) -> ExpressionIR:
        return _float_lit(0, self._loc)

    def visit_match_expression(self, node: Any) -> ExpressionIR:
        return _float_lit(0, self._loc)

    def visit_where_expression(self, node: Any) -> ExpressionIR:
        return _float_lit(0, self._loc)

    def visit_pipeline_expression(self, node: Any) -> ExpressionIR:
        return _float_lit(0, self._loc)

    def visit_builtin_call(self, node: BuiltinCallIR) -> ExpressionIR:
        return _float_lit(0, self._loc)

    def visit_module(self, node: Any) -> ExpressionIR:
        return _float_lit(0, self._loc)

    def visit_program(self, node: ProgramIR) -> ExpressionIR:
        return _float_lit(0, self._loc)

    def visit_function_value(self, node: FunctionValueIR) -> ExpressionIR:
        return _float_lit(0, self._loc)

    def visit_select_at_argmax(self, node: SelectAtArgmaxIR) -> ExpressionIR:
        return node

    def visit_differential(self, node: DifferentialIR) -> ExpressionIR:
        return _float_lit(0, self._loc)

    def visit_binding(self, node: BindingIR) -> ExpressionIR:
        if node.expr is not None:
            return node.expr.accept(self)
        return _float_lit(0, self._loc)

    def visit_index_var(self, node: Any) -> ExpressionIR:
        return _float_lit(0, self._loc)

    def visit_index_rest(self, node: Any) -> ExpressionIR:
        return _float_lit(0, self._loc)

    def visit_einstein_clause(self, node: Any) -> ExpressionIR:
        return _float_lit(0, self._loc)

    def visit_lowered_reduction(self, node: Any) -> ExpressionIR:
        return _float_lit(0, self._loc)

    def visit_lowered_select_at_argmax(self, node: Any) -> ExpressionIR:
        return _float_lit(0, self._loc)

    def visit_lowered_comprehension(self, node: Any) -> ExpressionIR:
        return _float_lit(0, self._loc)

    def visit_lowered_einstein_clause(self, node: Any) -> ExpressionIR:
        return _float_lit(0, self._loc)

    def visit_lowered_einstein(self, node: Any) -> ExpressionIR:
        return _float_lit(0, self._loc)

    def visit_lowered_recurrence(self, node: Any) -> ExpressionIR:
        return _float_lit(0, self._loc)

    def visit_literal_pattern(self, node: Any) -> ExpressionIR:
        return _float_lit(0, self._loc)

    def visit_identifier_pattern(self, node: Any) -> ExpressionIR:
        return _float_lit(0, self._loc)

    def visit_wildcard_pattern(self, node: Any) -> ExpressionIR:
        return _float_lit(0, self._loc)

    def visit_tuple_pattern(self, node: Any) -> ExpressionIR:
        return _float_lit(0, self._loc)

    def visit_array_pattern(self, node: Any) -> ExpressionIR:
        return _float_lit(0, self._loc)

    def visit_rest_pattern(self, node: Any) -> ExpressionIR:
        return _float_lit(0, self._loc)

    def visit_guard_pattern(self, node: Any) -> ExpressionIR:
        return _float_lit(0, self._loc)

    def visit_or_pattern(self, node: Any) -> ExpressionIR:
        return _float_lit(0, self._loc)

    def visit_constructor_pattern(self, node: Any) -> ExpressionIR:
        return _float_lit(0, self._loc)

    def visit_binding_pattern(self, node: Any) -> ExpressionIR:
        return _float_lit(0, self._loc)

    def visit_range_pattern(self, node: Any) -> ExpressionIR:
        return _float_lit(0, self._loc)


# ---------------------------------------------------------------------------
# Einstein differentiation (used by _SymbolicDiffVisitor.visit_einstein)
# ---------------------------------------------------------------------------

def _build_derivative_index_vars(
    clause_indices: List[Any],
    wrt_indices: List[Any],
    wrt_id: IdentifierIR,
    resolver: Any,
    loc: SourceLocation,
    allow_reuse: bool,
) -> Tuple[List[Any], Set[DefId], Dict[Any, Any]]:
    """Build index vars for derivative tensor dimensions.
    If allow_reuse, reuse clause index when same DefId; else always new _ad_p."""
    clause_index_by_defid: Dict[DefId, Any] = {}
    for c in clause_indices:
        if getattr(c, "defid", None) is not None:
            clause_index_by_defid[c.defid] = c
    derivative_index_vars: List[Any] = []
    new_defids: Set[DefId] = set()
    new_variable_ranges: Dict[Any, Any] = {}
    for p in range(len(wrt_indices)):
        idx_p = wrt_indices[p]
        defid_p = getattr(idx_p, "defid", None) if idx_p is not None else None
        if allow_reuse and defid_p is not None and defid_p in clause_index_by_defid:
            derivative_index_vars.append(clause_index_by_defid[defid_p])
        else:
            new_defid = resolver.allocate_for_local()
            shape_member = MemberAccessIR(object=wrt_id, member="shape", location=loc, type_info=UNKNOWN)
            dim_lit = LiteralIR(p, loc, type_info=PrimitiveType("i32"))
            shape_dim = RectangularAccessIR(array=shape_member, indices=[dim_lit], location=loc, type_info=UNKNOWN)
            start_lit = LiteralIR(0, loc, type_info=PrimitiveType("i32"))
            range_ir = RangeIR(start=start_lit, end=shape_dim, location=loc, type_info=UNKNOWN)
            new_iv = IndexVarIR("_ad_%d" % p, loc, new_defid, range_ir=range_ir)
            derivative_index_vars.append(new_iv)
            new_defids.add(new_defid)
            if new_iv.defid is not None and new_iv.range_ir is not None:
                new_variable_ranges[new_iv.defid] = new_iv.range_ir
    return derivative_index_vars, new_defids, new_variable_ranges


def _diff_einstein_wrt(
    expr: EinsteinIR,
    wrt_defid: DefId,
    loc: SourceLocation,
    binding_by_defid: Dict[DefId, BindingIR],
    resolver: Any,
) -> ExpressionIR:
    """d(EinsteinIR)/d(wrt): build derivative EinsteinIR following AUTODIFF_EINSTEIN_OPS.md."""
    derivative_clauses: List[EinsteinClauseIR] = []

    for clause in expr.clauses or []:
        val = clause.value
        if not isinstance(val, ReductionExpressionIR):
            vis = _SymbolicDiffVisitor(wrt_defid, loc, binding_by_defid, resolver)
            d_val = val.accept(vis)
            derivative_clauses.append(EinsteinClauseIR(
                indices=clause.indices,
                value=d_val,
                location=clause.location,
                where_clause=clause.where_clause,
                variable_ranges=dict(clause.variable_ranges or {}),
            ))
            continue

        inner = val.body
        factors = _flatten_product(inner) if inner else None

        if not factors:
            if val.operation == ReductionOp.SUM:
                raise ValueError("Autodiff: Einstein clause body is not a product of indexed arrays")
            vis = _SymbolicDiffVisitor(wrt_defid, loc, binding_by_defid, resolver)
            d_val = val.accept(vis)
            derivative_clauses.append(EinsteinClauseIR(
                indices=clause.indices,
                value=d_val,
                location=clause.location,
                where_clause=clause.where_clause,
                variable_ranges=dict(clause.variable_ranges or {}),
            ))
            continue

        wrt_positions: List[int] = [
            i for i, (arr_expr, _) in enumerate(factors)
            if isinstance(arr_expr, IdentifierIR) and arr_expr.defid == wrt_defid
        ]

        if not wrt_positions:
            continue

        first_wrt_indices = factors[wrt_positions[0]][1]
        wrt_binding = binding_by_defid.get(wrt_defid)
        wrt_name = (wrt_binding.name if wrt_binding else "") or "?"
        wrt_id = IdentifierIR(wrt_name, loc, wrt_defid, type_info=UNKNOWN)

        clause_indices_list = list(clause.indices or [])
        allow_reuse = len(clause_indices_list) < len(first_wrt_indices)
        derivative_index_vars, new_defids, new_var_ranges = _build_derivative_index_vars(
            clause_indices_list, first_wrt_indices, wrt_id, resolver, loc, allow_reuse
        )

        loop_vars = list(val.loop_vars or [])

        # MAX / MIN: SelectAtArgmaxIR
        if val.operation == ReductionOp.MAX or val.operation == ReductionOp.MIN:
            diff_body: ExpressionIR = _float_lit(1, loc)
            scalar_type = val.type_info if getattr(val, "type_info", None) is not None else F32
            for p in range(len(first_wrt_indices)):
                if getattr(derivative_index_vars[p], "defid", None) in new_defids:
                    idx_expr = first_wrt_indices[p]
                    der_var = derivative_index_vars[p]
                    eq = BinaryOpIR(BinaryOp.EQ, idx_expr, der_var, loc, type_info=BOOL)
                    diff_body = IfExpressionIR(
                        eq, diff_body, loc, else_expr=_float_lit(0, loc),
                        type_info=scalar_type,
                        shape_info=getattr(diff_body, "shape_info", None),
                    )
            sel = SelectAtArgmaxIR(
                val.body, diff_body, loop_vars,
                loop_var_ranges=dict(val.loop_var_ranges or {}),
                location=loc,
                type_info=val.type_info,
                shape_info=val.shape_info,
                use_argmin=(val.operation == ReductionOp.MIN),
            )
            new_variable_ranges = dict(clause.variable_ranges or {})
            new_variable_ranges.update(new_var_ranges)
            derivative_clauses.append(EinsteinClauseIR(
                indices=derivative_index_vars,
                value=sel,
                location=clause.location,
                where_clause=clause.where_clause,
                variable_ranges=new_variable_ranges,
            ))
            continue

        # PROD: prod over all factors except wrt (k != j constraint)
        if val.operation == ReductionOp.PROD and allow_reuse:
            prod_exclude_constraints = [
                BinaryOpIR(BinaryOp.NE, first_wrt_indices[p], derivative_index_vars[p], loc, type_info=BOOL)
                for p in range(len(first_wrt_indices))
                if getattr(derivative_index_vars[p], "defid", None) in new_defids
            ]
            original_where = getattr(val, "where_clause", None)
            orig_c = list(getattr(original_where, "constraints", None) or []) if original_where else []
            prod_where = WhereClauseIR(constraints=orig_c + prod_exclude_constraints, location=loc) if (orig_c or prod_exclude_constraints) else None
            prod_red = ReductionExpressionIR(
                ReductionOp.PROD, loop_vars, val.body, loc,
                where_clause=prod_where,
                loop_var_ranges=dict(val.loop_var_ranges or {}),
                type_info=val.type_info,
                shape_info=val.shape_info,
            )
            new_variable_ranges = dict(clause.variable_ranges or {})
            new_variable_ranges.update(new_var_ranges)
            derivative_clauses.append(EinsteinClauseIR(
                indices=derivative_index_vars,
                value=prod_red,
                location=clause.location,
                where_clause=clause.where_clause,
                variable_ranges=new_variable_ranges,
            ))
            continue

        if val.operation != ReductionOp.SUM:
            vis = _SymbolicDiffVisitor(wrt_defid, loc, binding_by_defid, resolver)
            d_val = val.accept(vis)
            derivative_clauses.append(EinsteinClauseIR(
                indices=clause.indices,
                value=d_val,
                location=clause.location,
                where_clause=clause.where_clause,
                variable_ranges=dict(clause.variable_ranges or {}),
            ))
            continue

        # SUM: product rule with delta constraints
        original_red_constraints: List[ExpressionIR] = (
            list(getattr(val.where_clause, "constraints", None) or [])
            if getattr(val, "where_clause", None) is not None else []
        )
        reduction_terms: List[ExpressionIR] = []
        for pos in wrt_positions:
            _, wrt_indices = factors[pos]
            other_factors = [factors[i] for i in range(len(factors)) if i != pos]
            delta_constraints = [
                BinaryOpIR(BinaryOp.EQ, wrt_indices[p], derivative_index_vars[p], loc)
                for p in range(len(wrt_indices))
                if getattr(derivative_index_vars[p], "defid", None) in new_defids
            ]
            combined_constraints = original_red_constraints + delta_constraints
            where = WhereClauseIR(constraints=combined_constraints, location=loc) if combined_constraints else None

            if not other_factors:
                body: ExpressionIR = _float_lit(1, loc)
            else:
                b = binding_by_defid.get(other_factors[0][0].defid) if isinstance(other_factors[0][0], IdentifierIR) else None
                nm = other_factors[0][0].name or (b.name if b else "") or ""
                ref0 = IdentifierIR(nm, loc, other_factors[0][0].defid) if isinstance(other_factors[0][0], IdentifierIR) else other_factors[0][0]
                body = RectangularAccessIR(ref0, list(other_factors[0][1]), loc)
                for arr_expr, idx_list in other_factors[1:]:
                    b = binding_by_defid.get(arr_expr.defid) if isinstance(arr_expr, IdentifierIR) else None
                    nm = arr_expr.name or (b.name if b else "") or ""
                    ref = IdentifierIR(nm, loc, arr_expr.defid) if isinstance(arr_expr, IdentifierIR) else arr_expr
                    body = BinaryOpIR(BinaryOp.MUL, body, RectangularAccessIR(ref, list(idx_list), loc), loc)

            red = ReductionExpressionIR(
                operation=val.operation,
                loop_vars=loop_vars,
                body=body,
                location=loc,
                where_clause=where,
                loop_var_ranges=dict(val.loop_var_ranges or {}),
                type_info=val.type_info,
                shape_info=val.shape_info,
            )
            reduction_terms.append(red)

        clause_value: ExpressionIR = reduction_terms[0]
        for r in reduction_terms[1:]:
            clause_value = BinaryOpIR(BinaryOp.ADD, clause_value, r, loc)

        new_indices = derivative_index_vars if allow_reuse else (clause_indices_list + derivative_index_vars)
        new_variable_ranges = dict(clause.variable_ranges or {})
        new_variable_ranges.update(new_var_ranges)
        derivative_clauses.append(EinsteinClauseIR(
            indices=new_indices,
            value=clause_value,
            location=clause.location,
            where_clause=clause.where_clause,
            variable_ranges=new_variable_ranges,
        ))

    if not derivative_clauses:
        return _float_lit(0, loc)

    return EinsteinIR(
        clauses=derivative_clauses,
        shape=None,
        element_type=expr.element_type,
        location=expr.location,
        type_info=expr.type_info,
        shape_info=None,
    )


# ---------------------------------------------------------------------------
# Visitor 3: _ForwardDiffVisitor  (d(expr) in terms of d_x refs)
# ---------------------------------------------------------------------------

class _ForwardDiffVisitor(IRVisitor[ExpressionIR]):
    """Build d(expr) using d_x refs (forward mode chain rule)."""

    def __init__(self, defid_to_d_ident: Dict[DefId, IdentifierIR], loc: SourceLocation) -> None:
        self._d_map = defid_to_d_ident
        self._loc = loc

    def visit_identifier(self, node: IdentifierIR) -> ExpressionIR:
        if node.defid is not None:
            ref = self._d_map.get(node.defid)
            if ref is not None:
                return IdentifierIR(ref.name, node.location or self._loc, ref.defid)
        raise ValueError("Autodiff: identifier not in differential map for d(expr)")

    def visit_literal(self, node: LiteralIR) -> ExpressionIR:
        return _float_lit(0, self._loc)

    def visit_binary_op(self, node: BinaryOpIR) -> ExpressionIR:
        loc = node.location or self._loc
        d_left = node.left.accept(self)
        d_right = node.right.accept(self)
        op = node.operator
        if op == BinaryOp.ADD:
            return BinaryOpIR(BinaryOp.ADD, d_left, d_right, loc)
        if op == BinaryOp.SUB:
            return BinaryOpIR(BinaryOp.SUB, d_left, d_right, loc)
        if op == BinaryOp.MUL:
            return BinaryOpIR(
                BinaryOp.ADD,
                BinaryOpIR(BinaryOp.MUL, d_left, node.right, loc),
                BinaryOpIR(BinaryOp.MUL, node.left, d_right, loc),
                loc,
            )
        if op == BinaryOp.DIV:
            t1 = BinaryOpIR(BinaryOp.DIV, d_left, node.right, loc)
            t2 = BinaryOpIR(
                BinaryOp.DIV,
                BinaryOpIR(BinaryOp.MUL, node.left, d_right, loc),
                BinaryOpIR(BinaryOp.POW, node.right, _float_lit(2, loc), loc),
                loc,
            )
            return BinaryOpIR(BinaryOp.SUB, t1, t2, loc)
        if op == BinaryOp.POW:
            b_minus_one = BinaryOpIR(BinaryOp.SUB, node.right, _float_lit(1, loc), loc)
            a_bm1 = BinaryOpIR(BinaryOp.POW, node.left, b_minus_one, loc)
            term_left = BinaryOpIR(BinaryOp.MUL, BinaryOpIR(BinaryOp.MUL, node.right, a_bm1, loc), d_left, loc)
            a_b = BinaryOpIR(BinaryOp.POW, node.left, node.right, loc)
            term_right = BinaryOpIR(BinaryOp.MUL, a_b, d_right, loc)
            return BinaryOpIR(BinaryOp.ADD, term_left, term_right, loc)
        if op == BinaryOp.MOD:
            return d_left
        raise ValueError(f"Autodiff: unsupported binary op for d(expr): {op}")

    def visit_unary_op(self, node: UnaryOpIR) -> ExpressionIR:
        d_op = node.operand.accept(self)
        if node.operator == UnaryOp.NEG:
            return UnaryOpIR(UnaryOp.NEG, d_op, node.location or self._loc)
        if node.operator == UnaryOp.POS:
            return d_op
        raise ValueError(f"Autodiff: unsupported unary op for d(expr): {node.operator}")

    def visit_reduction_expression(self, node: ReductionExpressionIR) -> ExpressionIR:
        loc = node.location or self._loc
        d_body = node.body.accept(self)
        op = node.operation
        if op == ReductionOp.SUM:
            return ReductionExpressionIR(
                ReductionOp.SUM, node.loop_vars, d_body, loc,
                where_clause=node.where_clause,
                loop_var_ranges=node.loop_var_ranges,
                type_info=getattr(node, "type_info", None),
                shape_info=getattr(node, "shape_info", None),
            )
        if op == ReductionOp.MAX:
            return SelectAtArgmaxIR(
                node.body, d_body, node.loop_vars,
                loop_var_ranges=node.loop_var_ranges,
                location=loc,
                type_info=getattr(node, "type_info", None),
                shape_info=getattr(node, "shape_info", None),
                use_argmin=False,
            )
        if op == ReductionOp.MIN:
            return SelectAtArgmaxIR(
                node.body, d_body, node.loop_vars,
                loop_var_ranges=node.loop_var_ranges,
                location=loc,
                type_info=getattr(node, "type_info", None),
                shape_info=getattr(node, "shape_info", None),
                use_argmin=True,
            )
        if op == ReductionOp.PROD:
            return BinaryOpIR(
                BinaryOp.MUL,
                BinaryOpIR(BinaryOp.DIV, node, node.body, loc),
                d_body,
                loc,
            )
        raise ValueError(f"Autodiff: unsupported reduction for d(expr): {op}")

    def visit_block_expression(self, node: BlockExpressionIR) -> ExpressionIR:
        if node.final_expr is not None:
            return node.final_expr.accept(self)
        return _float_lit(0, self._loc)

    def visit_select_at_argmax(self, node: SelectAtArgmaxIR) -> ExpressionIR:
        return node

    # stubs
    def visit_function_call(self, node: FunctionCallIR) -> ExpressionIR:
        return _float_lit(0, self._loc)

    def visit_rectangular_access(self, node: RectangularAccessIR) -> ExpressionIR:
        return _float_lit(0, self._loc)

    def visit_jagged_access(self, node: Any) -> ExpressionIR:
        return _float_lit(0, self._loc)

    def visit_if_expression(self, node: IfExpressionIR) -> ExpressionIR:
        return _float_lit(0, self._loc)

    def visit_cast_expression(self, node: CastExpressionIR) -> ExpressionIR:
        return _float_lit(0, self._loc)

    def visit_lambda(self, node: Any) -> ExpressionIR:
        return _float_lit(0, self._loc)

    def visit_range(self, node: RangeIR) -> ExpressionIR:
        return _float_lit(0, self._loc)

    def visit_array_comprehension(self, node: Any) -> ExpressionIR:
        return _float_lit(0, self._loc)

    def visit_array_literal(self, node: ArrayLiteralIR) -> ExpressionIR:
        return _float_lit(0, self._loc)

    def visit_tuple_expression(self, node: Any) -> ExpressionIR:
        return _float_lit(0, self._loc)

    def visit_tuple_access(self, node: Any) -> ExpressionIR:
        return _float_lit(0, self._loc)

    def visit_interpolated_string(self, node: Any) -> ExpressionIR:
        return _float_lit(0, self._loc)

    def visit_member_access(self, node: MemberAccessIR) -> ExpressionIR:
        return _float_lit(0, self._loc)

    def visit_function_value(self, node: FunctionValueIR) -> ExpressionIR:
        return _float_lit(0, self._loc)

    def visit_try_expression(self, node: Any) -> ExpressionIR:
        return _float_lit(0, self._loc)

    def visit_match_expression(self, node: Any) -> ExpressionIR:
        return _float_lit(0, self._loc)

    def visit_where_expression(self, node: Any) -> ExpressionIR:
        return _float_lit(0, self._loc)

    def visit_pipeline_expression(self, node: Any) -> ExpressionIR:
        return _float_lit(0, self._loc)

    def visit_builtin_call(self, node: BuiltinCallIR) -> ExpressionIR:
        return _float_lit(0, self._loc)

    def visit_module(self, node: Any) -> ExpressionIR:
        return _float_lit(0, self._loc)

    def visit_program(self, node: ProgramIR) -> ExpressionIR:
        return _float_lit(0, self._loc)

    def visit_einstein(self, node: EinsteinIR) -> ExpressionIR:
        return _float_lit(0, self._loc)

    def visit_differential(self, node: DifferentialIR) -> ExpressionIR:
        return _float_lit(0, self._loc)

    def visit_binding(self, node: BindingIR) -> ExpressionIR:
        if node.expr is not None:
            return node.expr.accept(self)
        return _float_lit(0, self._loc)

    def visit_index_var(self, node: Any) -> ExpressionIR:
        return _float_lit(0, self._loc)

    def visit_index_rest(self, node: Any) -> ExpressionIR:
        return _float_lit(0, self._loc)

    def visit_einstein_clause(self, node: Any) -> ExpressionIR:
        return _float_lit(0, self._loc)

    def visit_lowered_reduction(self, node: Any) -> ExpressionIR:
        return _float_lit(0, self._loc)

    def visit_lowered_select_at_argmax(self, node: Any) -> ExpressionIR:
        return _float_lit(0, self._loc)

    def visit_lowered_comprehension(self, node: Any) -> ExpressionIR:
        return _float_lit(0, self._loc)

    def visit_lowered_einstein_clause(self, node: Any) -> ExpressionIR:
        return _float_lit(0, self._loc)

    def visit_lowered_einstein(self, node: Any) -> ExpressionIR:
        return _float_lit(0, self._loc)

    def visit_lowered_recurrence(self, node: Any) -> ExpressionIR:
        return _float_lit(0, self._loc)

    def visit_literal_pattern(self, node: Any) -> ExpressionIR:
        return _float_lit(0, self._loc)

    def visit_identifier_pattern(self, node: Any) -> ExpressionIR:
        return _float_lit(0, self._loc)

    def visit_wildcard_pattern(self, node: Any) -> ExpressionIR:
        return _float_lit(0, self._loc)

    def visit_tuple_pattern(self, node: Any) -> ExpressionIR:
        return _float_lit(0, self._loc)

    def visit_array_pattern(self, node: Any) -> ExpressionIR:
        return _float_lit(0, self._loc)

    def visit_rest_pattern(self, node: Any) -> ExpressionIR:
        return _float_lit(0, self._loc)

    def visit_guard_pattern(self, node: Any) -> ExpressionIR:
        return _float_lit(0, self._loc)

    def visit_or_pattern(self, node: Any) -> ExpressionIR:
        return _float_lit(0, self._loc)

    def visit_constructor_pattern(self, node: Any) -> ExpressionIR:
        return _float_lit(0, self._loc)

    def visit_binding_pattern(self, node: Any) -> ExpressionIR:
        return _float_lit(0, self._loc)

    def visit_range_pattern(self, node: Any) -> ExpressionIR:
        return _float_lit(0, self._loc)


# ---------------------------------------------------------------------------
# Substitution helpers
# ---------------------------------------------------------------------------

def _substitute(expr: ExpressionIR, replace_map: Dict[DefId, ExpressionIR],
                loc: SourceLocation) -> ExpressionIR:
    if isinstance(expr, IdentifierIR) and expr.defid is not None and expr.defid in replace_map:
        return replace_map[expr.defid]
    if isinstance(expr, IndexVarIR) and expr.defid is not None and expr.defid in replace_map:
        return replace_map[expr.defid]
    if isinstance(expr, LiteralIR):
        return LiteralIR(expr.value, loc, type_info=getattr(expr, "type_info", None))
    if isinstance(expr, BinaryOpIR):
        return BinaryOpIR(expr.operator,
                          _substitute(expr.left, replace_map, loc),
                          _substitute(expr.right, replace_map, loc), loc)
    if isinstance(expr, UnaryOpIR):
        return UnaryOpIR(expr.operator, _substitute(expr.operand, replace_map, loc), loc)
    if isinstance(expr, RectangularAccessIR):
        arr = _substitute(expr.array, replace_map, loc)
        new_indices = [_substitute(i, replace_map, loc) for i in (expr.indices or [])]
        return RectangularAccessIR(arr, new_indices, loc,
                                   type_info=getattr(expr, "type_info", None),
                                   shape_info=getattr(expr, "shape_info", None))
    if isinstance(expr, BlockExpressionIR) and expr.final_expr is not None:
        return _substitute(expr.final_expr, replace_map, loc)
    if isinstance(expr, IfExpressionIR):
        cond = _substitute(expr.condition, replace_map, loc)
        then_e = _substitute(expr.then_expr, replace_map, loc)
        else_e = _substitute(expr.else_expr, replace_map, loc) if expr.else_expr else None
        return IfExpressionIR(condition=cond, then_expr=then_e, else_expr=else_e, location=loc)
    if isinstance(expr, ReductionExpressionIR) and expr.body is not None:
        return ReductionExpressionIR(
            expr.operation, expr.loop_vars,
            _substitute(expr.body, replace_map, loc),
            expr.location,
            where_clause=expr.where_clause,
            loop_var_ranges=expr.loop_var_ranges,
            type_info=getattr(expr, "type_info", None),
            shape_info=getattr(expr, "shape_info", None),
        )
    if isinstance(expr, SelectAtArgmaxIR):
        return SelectAtArgmaxIR(
            _substitute(expr.primal_body, replace_map, loc),
            _substitute(expr.diff_body, replace_map, loc),
            expr.loop_vars,
            loop_var_ranges=expr.loop_var_ranges,
            location=expr.location,
            type_info=getattr(expr, "type_info", None),
            shape_info=getattr(expr, "shape_info", None),
            use_argmin=getattr(expr, "use_argmin", False),
        )
    return expr


def _substitute_with_diffs(
    expr: ExpressionIR,
    replace_map: Dict[DefId, ExpressionIR],
    diff_replace_map: Dict[DefId, ExpressionIR],
    loc: SourceLocation,
) -> ExpressionIR:
    """Substitute identifiers by replace_map and DifferentialIR(IdentifierIR(defid)) by diff_replace_map[defid]."""
    if isinstance(expr, IdentifierIR) and expr.defid is not None:
        if expr.defid in replace_map:
            return replace_map[expr.defid]
    if isinstance(expr, DifferentialIR):
        op = expr.operand
        if isinstance(op, IdentifierIR) and op.defid is not None and op.defid in diff_replace_map:
            return diff_replace_map[op.defid]
        return DifferentialIR(operand=_substitute_with_diffs(op, replace_map, diff_replace_map, loc), location=loc)
    if isinstance(expr, LiteralIR):
        return LiteralIR(expr.value, loc, type_info=getattr(expr, "type_info", None))
    if isinstance(expr, BinaryOpIR):
        return BinaryOpIR(
            expr.operator,
            _substitute_with_diffs(expr.left, replace_map, diff_replace_map, loc),
            _substitute_with_diffs(expr.right, replace_map, diff_replace_map, loc),
            loc,
            type_info=getattr(expr, "type_info", None),
            shape_info=getattr(expr, "shape_info", None),
        )
    if isinstance(expr, UnaryOpIR):
        return UnaryOpIR(
            expr.operator,
            _substitute_with_diffs(expr.operand, replace_map, diff_replace_map, loc),
            loc,
            type_info=getattr(expr, "type_info", None),
            shape_info=getattr(expr, "shape_info", None),
        )
    if isinstance(expr, BlockExpressionIR) and expr.final_expr is not None:
        return _substitute_with_diffs(expr.final_expr, replace_map, diff_replace_map, loc)
    if isinstance(expr, FunctionCallIR):
        new_args = [_substitute_with_diffs(a, replace_map, diff_replace_map, loc) for a in (expr.arguments or [])]
        return FunctionCallIR(
            callee_expr=expr.callee_expr,
            location=expr.location,
            arguments=new_args,
            module_path=expr.module_path,
            type_info=expr.type_info,
            shape_info=getattr(expr, "shape_info", None),
        )
    if isinstance(expr, RectangularAccessIR):
        arr = _substitute_with_diffs(expr.array, replace_map, diff_replace_map, loc)
        indices = [_substitute_with_diffs(i, replace_map, diff_replace_map, loc) for i in (expr.indices or [])]
        return RectangularAccessIR(
            array=arr, indices=indices, location=expr.location,
            type_info=getattr(expr, "type_info", None),
            shape_info=getattr(expr, "shape_info", None),
        )
    if isinstance(expr, IfExpressionIR):
        cond = _substitute_with_diffs(expr.condition, replace_map, diff_replace_map, loc)
        then_e = _substitute_with_diffs(expr.then_expr, replace_map, diff_replace_map, loc)
        else_e = _substitute_with_diffs(expr.else_expr, replace_map, diff_replace_map, loc) if expr.else_expr else None
        return IfExpressionIR(
            cond, then_e, loc, else_expr=else_e,
            type_info=getattr(expr, "type_info", None),
            shape_info=getattr(expr, "shape_info", None),
        )
    if isinstance(expr, ReductionExpressionIR) and expr.body is not None:
        return ReductionExpressionIR(
            expr.operation, expr.loop_vars,
            _substitute_with_diffs(expr.body, replace_map, diff_replace_map, loc),
            expr.location,
            where_clause=expr.where_clause,
            loop_var_ranges=expr.loop_var_ranges,
            type_info=getattr(expr, "type_info", None),
            shape_info=getattr(expr, "shape_info", None),
        )
    if isinstance(expr, SelectAtArgmaxIR):
        return SelectAtArgmaxIR(
            _substitute_with_diffs(expr.primal_body, replace_map, diff_replace_map, loc),
            _substitute_with_diffs(expr.diff_body, replace_map, diff_replace_map, loc),
            expr.loop_vars,
            loop_var_ranges=expr.loop_var_ranges,
            location=expr.location,
            type_info=getattr(expr, "type_info", None),
            shape_info=getattr(expr, "shape_info", None),
            use_argmin=getattr(expr, "use_argmin", False),
        )
    if isinstance(expr, CastExpressionIR):
        inner = _substitute_with_diffs(expr.expr, replace_map, diff_replace_map, loc)
        return CastExpressionIR(
            inner, expr.target_type, loc,
            type_info=getattr(expr, "type_info", None),
            shape_info=getattr(expr, "shape_info", None),
        )
    return expr


# ---------------------------------------------------------------------------
# Forward-diff for Einstein  (d_C from d_A, d_B)
# ---------------------------------------------------------------------------

def _forward_einstein_ir(
    expr: EinsteinIR,
    d_ref_by_defid: Dict[DefId, IdentifierIR],
    binding_by_defid: Dict[DefId, BindingIR],
    loc: SourceLocation,
) -> Optional[ExpressionIR]:
    clause_terms: List[ExpressionIR] = []
    for clause in expr.clauses or []:
        val = clause.value
        if not isinstance(val, ReductionExpressionIR) or val.operation != ReductionOp.SUM:
            continue
        inner = val.body
        factors = _flatten_product(inner) if inner else None
        if not factors or len(factors) < 1:
            continue
        loop_vars = list(val.loop_vars or [])
        reduction_parts: List[ExpressionIR] = []
        for arr_expr, index_list in factors:
            if not isinstance(arr_expr, IdentifierIR) or arr_expr.defid is None:
                continue
            if arr_expr.defid not in d_ref_by_defid:
                continue
            d_ref = d_ref_by_defid[arr_expr.defid]
            prod: ExpressionIR = RectangularAccessIR(d_ref, index_list, loc)
            for other_expr, other_indices in factors:
                if other_expr is arr_expr:
                    continue
                if not isinstance(other_expr, IdentifierIR) or other_expr.defid is None:
                    continue
                b = binding_by_defid.get(other_expr.defid)
                nm = other_expr.name or (b.name if b else "") or ""
                other_ref = IdentifierIR(nm, loc, other_expr.defid)
                prod = BinaryOpIR(BinaryOp.MUL, prod, RectangularAccessIR(other_ref, other_indices, loc), loc)
            red = ReductionExpressionIR(
                operation=val.operation,
                loop_vars=loop_vars,
                body=prod,
                location=loc,
                where_clause=val.where_clause,
                loop_var_ranges=dict(val.loop_var_ranges or {}),
                type_info=val.type_info,
                shape_info=val.shape_info,
            )
            reduction_parts.append(red)
        if not reduction_parts:
            continue
        combined_val: ExpressionIR = reduction_parts[0]
        for rp in reduction_parts[1:]:
            combined_val = BinaryOpIR(BinaryOp.ADD, combined_val, rp, loc)
        new_clause = EinsteinClauseIR(
            indices=list(clause.indices or []),
            value=combined_val,
            location=clause.location,
            where_clause=clause.where_clause,
            variable_ranges=dict(clause.variable_ranges or {}),
        )
        clause_terms.append(
            EinsteinIR(
                clauses=[new_clause],
                shape=expr.shape,
                element_type=expr.element_type,
                location=expr.location,
                type_info=expr.type_info,
                shape_info=expr.shape_info,
            )
        )
    if not clause_terms:
        return None
    out = clause_terms[0]
    for t in clause_terms[1:]:
        out = BinaryOpIR(BinaryOp.ADD, out, t, loc)
    return out


def _forward_d_y_expr(
    binding: BindingIR,
    defid_to_d_ref: Dict[DefId, ExpressionIR],
    binding_by_defid: Dict[DefId, BindingIR],
    binding_to_deps: Dict[DefId, Set[DefId]],
    loc: SourceLocation,
    resolver: Optional[Any] = None,
) -> Optional[ExpressionIR]:
    """Forward: expression for d_y from d_x1, d_x2 (deps)."""
    expr = binding.expr
    if expr is None:
        return None
    deps = binding_to_deps.get(binding.defid) or set()
    if not deps:
        return None

    def _defid_from(e: ExpressionIR) -> Optional[DefId]:
        return e.defid if isinstance(e, IdentifierIR) else None

    if isinstance(expr, BinaryOpIR):
        left_defid = _defid_from(expr.left)
        right_defid = _defid_from(expr.right)
        d_left = defid_to_d_ref.get(left_defid) if left_defid else _float_lit(0, loc)
        d_right = defid_to_d_ref.get(right_defid) if right_defid else _float_lit(0, loc)
        if not isinstance(d_left, ExpressionIR):
            d_left = _float_lit(0, loc)
        if not isinstance(d_right, ExpressionIR):
            d_right = _float_lit(0, loc)
        op = expr.operator
        if op == BinaryOp.ADD:
            return BinaryOpIR(BinaryOp.ADD, d_left, d_right, loc)
        if op == BinaryOp.SUB:
            return BinaryOpIR(BinaryOp.SUB, d_left, d_right, loc)
        if op == BinaryOp.MUL:
            return BinaryOpIR(
                BinaryOp.ADD,
                BinaryOpIR(BinaryOp.MUL, d_left, expr.right, loc),
                BinaryOpIR(BinaryOp.MUL, expr.left, d_right, loc),
                loc,
            )
        if op == BinaryOp.DIV:
            t1 = BinaryOpIR(BinaryOp.DIV, d_left, expr.right, loc)
            t2 = BinaryOpIR(BinaryOp.DIV,
                            BinaryOpIR(BinaryOp.MUL, expr.left, d_right, loc),
                            BinaryOpIR(BinaryOp.POW, expr.right, _float_lit(2, loc), loc), loc)
            return BinaryOpIR(BinaryOp.SUB, t1, t2, loc)
        if op == BinaryOp.POW:
            b_m1 = BinaryOpIR(BinaryOp.SUB, expr.right, _float_lit(1, loc), loc)
            a_bm1 = BinaryOpIR(BinaryOp.POW, expr.left, b_m1, loc)
            tl = BinaryOpIR(BinaryOp.MUL, BinaryOpIR(BinaryOp.MUL, expr.right, a_bm1, loc), d_left, loc)
            a_b = BinaryOpIR(BinaryOp.POW, expr.left, expr.right, loc)
            tr = BinaryOpIR(BinaryOp.MUL, a_b, d_right, loc)
            return BinaryOpIR(BinaryOp.ADD, tl, tr, loc)
        if op == BinaryOp.MOD:
            return d_left
        return None

    if isinstance(expr, UnaryOpIR):
        operand_defid = _defid_from(expr.operand)
        d_operand = defid_to_d_ref.get(operand_defid) if operand_defid else _float_lit(0, loc)
        if not isinstance(d_operand, ExpressionIR):
            d_operand = _float_lit(0, loc)
        if expr.operator == UnaryOp.NEG:
            return UnaryOpIR(UnaryOp.NEG, d_operand, loc)
        if expr.operator == UnaryOp.POS:
            return d_operand
        return None

    if isinstance(expr, BuiltinCallIR):
        return None

    if isinstance(expr, FunctionCallIR):
        callee_defid = expr.function_defid
        args = expr.arguments or []
        if callee_defid is None or len(args) == 0:
            return None
        callee_binding = binding_by_defid.get(callee_defid)
        if callee_binding is None or not isinstance(callee_binding.expr, FunctionValueIR):
            return None
        fv = callee_binding.expr
        rule_body = getattr(fv, 'custom_diff_body', None)
        if rule_body is not None:
            params = fv.parameters or []
            if len(params) == len(args):
                replace_map = {p.defid: args[j] for j, p in enumerate(params) if p.defid is not None}
                diff_replace_map: Dict[DefId, ExpressionIR] = {}
                for i, param in enumerate(params):
                    if param.defid is None:
                        continue
                    arg_defid = _defid_from(args[i])
                    d_arg = defid_to_d_ref.get(arg_defid) if arg_defid else None
                    diff_replace_map[param.defid] = d_arg if isinstance(d_arg, ExpressionIR) else _float_lit(0, loc)
                d_y = _substitute_with_diffs(rule_body, replace_map, diff_replace_map, expr.location or loc)
                return d_y
        params = fv.parameters or []
        body = fv.body
        if body is None or len(params) != len(args):
            return None
        replace_map = {p.defid: args[j] for j, p in enumerate(params)}
        terms: List[ExpressionIR] = []
        for i, param in enumerate(params):
            vis = _SymbolicDiffVisitor(param.defid, expr.location or loc, binding_by_defid, resolver)
            partial = body.accept(vis)
            partial_at_call = _substitute(partial, replace_map, expr.location or loc)
            arg_defid = _defid_from(args[i])
            if arg_defid is None:
                continue
            d_arg = defid_to_d_ref.get(arg_defid)
            if d_arg is None:
                continue
            terms.append(BinaryOpIR(BinaryOp.MUL, d_arg, partial_at_call, expr.location or loc))
        if not terms:
            return None
        out: ExpressionIR = terms[0]
        for t in terms[1:]:
            out = BinaryOpIR(BinaryOp.ADD, out, t, loc)
        return out

    if isinstance(expr, EinsteinIR):
        d_ref_map = {did: ref for did, ref in defid_to_d_ref.items() if isinstance(ref, IdentifierIR)}
        return _forward_einstein_ir(expr, d_ref_map, binding_by_defid, loc)

    return None


# ---------------------------------------------------------------------------
# Collection helpers
# ---------------------------------------------------------------------------

def _collect_differential_targets(program: ProgramIR) -> List[Tuple[DefId, str]]:
    """Collect (defid, name) for every @id in the program."""
    targets: List[Tuple[DefId, str]] = []

    def _target_from(operand: ExpressionIR) -> Optional[Tuple[DefId, str]]:
        if isinstance(operand, IdentifierIR) and operand.defid is not None:
            return (operand.defid, operand.name or "")
        return None

    def walk(e: Any) -> None:
        if e is None:
            return
        if isinstance(e, DifferentialIR):
            t = _target_from(e.operand)
            if t is not None:
                targets.append(t)
            walk(e.operand)
        if isinstance(e, BinaryOpIR):
            if e.operator == BinaryOp.DIV and isinstance(e.left, DifferentialIR) and isinstance(e.right, DifferentialIR):
                for op in (e.left.operand, e.right.operand):
                    t = _target_from(op)
                    if t is not None:
                        targets.append(t)
            walk(e.left)
            walk(e.right)
        if isinstance(e, UnaryOpIR):
            walk(e.operand)
        if isinstance(e, BlockExpressionIR):
            for s in e.statements or []:
                walk(s)
            walk(e.final_expr)
        if isinstance(e, BindingIR):
            walk(e.expr)
        if isinstance(e, ProgramIR):
            for b in e.bindings or []:
                walk(b)
            for s in e.statements or []:
                if not isinstance(s, BindingIR):
                    walk(s)
        if isinstance(e, FunctionValueIR) and e.body is not None:
            walk(e.body)
        if isinstance(e, IfExpressionIR):
            walk(e.condition)
            walk(e.then_expr)
            walk(e.else_expr)
        if isinstance(e, (EinsteinIR,)):
            for c in getattr(e, "clauses", None) or []:
                walk(getattr(c, "value", c))
        if isinstance(e, FunctionCallIR):
            for a in e.arguments or []:
                walk(a)
        if isinstance(e, RectangularAccessIR):
            walk(e.array)
        if isinstance(e, BuiltinCallIR):
            for a in e.args or []:
                walk(a)

    walk(program)
    return targets


def _collect_quotient_pairs(program: ProgramIR) -> List[Tuple[DefId, DefId]]:
    """Collect (numerator_defid, denominator_defid) from BinaryOpIR(DIV, @num, @den)."""
    pairs: List[Tuple[DefId, DefId]] = []

    def _defid_from_diff_or_id(e: Any) -> Optional[DefId]:
        if isinstance(e, DifferentialIR):
            if isinstance(e.operand, IdentifierIR):
                return e.operand.defid
        if isinstance(e, IdentifierIR):
            return e.defid
        return None

    def walk(e: Any) -> None:
        if e is None:
            return
        if isinstance(e, BinaryOpIR):
            if e.operator == BinaryOp.DIV:
                num = _defid_from_diff_or_id(e.left)
                den = _defid_from_diff_or_id(e.right)
                if num is not None and den is not None:
                    pairs.append((num, den))
            walk(e.left)
            walk(e.right)
        if isinstance(e, UnaryOpIR):
            walk(e.operand)
        if isinstance(e, BlockExpressionIR):
            for s in e.statements or []:
                walk(s)
            walk(e.final_expr)
        if isinstance(e, BindingIR):
            walk(e.expr)
        if isinstance(e, ProgramIR):
            for b in e.bindings or []:
                walk(b)
        if isinstance(e, (EinsteinIR, IfExpressionIR)):
            for c in getattr(e, "clauses", None) or []:
                walk(getattr(c, "value", c))
            if hasattr(e, "then_expr"):
                walk(e.then_expr)
            if hasattr(e, "else_expr"):
                walk(e.else_expr)
        if isinstance(e, FunctionCallIR):
            for a in e.arguments or []:
                walk(a)
        if isinstance(e, RectangularAccessIR):
            walk(e.array)
        if isinstance(e, BuiltinCallIR):
            for a in e.args or []:
                walk(a)

    walk(program)
    return pairs


def _collect_quotient_pairs_in_expr(expr: Any) -> List[Tuple[DefId, DefId]]:
    pairs: List[Tuple[DefId, DefId]] = []

    def _defid_from_diff_or_id(e: Any) -> Optional[DefId]:
        if isinstance(e, DifferentialIR) and isinstance(e.operand, IdentifierIR):
            return e.operand.defid
        if isinstance(e, IdentifierIR):
            return e.defid
        return None

    def walk(e: Any) -> None:
        if e is None:
            return
        if isinstance(e, BinaryOpIR):
            if e.operator == BinaryOp.DIV:
                num = _defid_from_diff_or_id(e.left)
                den = _defid_from_diff_or_id(e.right)
                if num is not None and den is not None:
                    pairs.append((num, den))
            walk(e.left)
            walk(e.right)
        if isinstance(e, UnaryOpIR):
            walk(e.operand)
        if isinstance(e, BlockExpressionIR):
            for s in e.statements or []:
                walk(s)
            walk(e.final_expr)
        if isinstance(e, BindingIR):
            walk(e.expr)
        if isinstance(e, (EinsteinIR, IfExpressionIR)):
            for c in getattr(e, "clauses", None) or []:
                walk(getattr(c, "value", c))
            if hasattr(e, "then_expr"):
                walk(e.then_expr)
            if hasattr(e, "else_expr"):
                walk(e.else_expr)
        if isinstance(e, FunctionCallIR):
            for a in e.arguments or []:
                walk(a)
        if isinstance(e, RectangularAccessIR):
            walk(e.array)

    walk(expr)
    return pairs


def _collect_differential_targets_in_expr(expr: Any) -> List[Tuple[DefId, str]]:
    targets: List[Tuple[DefId, str]] = []

    def _target_from(operand: ExpressionIR) -> Optional[Tuple[DefId, str]]:
        if isinstance(operand, IdentifierIR) and operand.defid is not None:
            return (operand.defid, operand.name or "")
        return None

    def walk(e: Any) -> None:
        if e is None:
            return
        if isinstance(e, DifferentialIR):
            t = _target_from(e.operand)
            if t is not None:
                targets.append(t)
            walk(e.operand)
        if isinstance(e, BinaryOpIR):
            if e.operator == BinaryOp.DIV and isinstance(e.left, DifferentialIR) and isinstance(e.right, DifferentialIR):
                for op in (e.left.operand, e.right.operand):
                    t = _target_from(op)
                    if t is not None:
                        targets.append(t)
            walk(e.left)
            walk(e.right)
        if isinstance(e, UnaryOpIR):
            walk(e.operand)
        if isinstance(e, BlockExpressionIR):
            for s in e.statements or []:
                walk(s)
            walk(e.final_expr)
        if isinstance(e, BindingIR):
            walk(e.expr)
        if isinstance(e, (EinsteinIR, IfExpressionIR)):
            for c in getattr(e, "clauses", None) or []:
                walk(getattr(c, "value", c))
            if hasattr(e, "final_expr"):
                walk(getattr(e, "final_expr"))
            if hasattr(e, "then_expr"):
                walk(e.then_expr)
            if hasattr(e, "else_expr"):
                walk(e.else_expr)
        if isinstance(e, FunctionCallIR):
            for a in e.arguments or []:
                walk(a)
        if isinstance(e, RectangularAccessIR):
            walk(e.array)
        if isinstance(e, BuiltinCallIR):
            for a in e.args or []:
                walk(a)

    walk(expr)
    return targets


# ---------------------------------------------------------------------------
# Block-level d_* creation
# ---------------------------------------------------------------------------

def _bindings_in_block(block: Any, program: Optional[ProgramIR] = None) -> List[BindingIR]:
    if block is program or isinstance(block, ProgramIR):
        return [b for b in (program.bindings or []) if isinstance(b, BindingIR)] if program else []
    if isinstance(block, BlockExpressionIR):
        return [s for s in (block.statements or []) if isinstance(s, BindingIR)]
    return []


def _ensure_block_has_d_bindings(
    block: BlockExpressionIR,
    scope_binding_by_defid: Dict[DefId, Any],
    scope_defid_to_expr: Dict[DefId, ExpressionIR],
    defid_to_d_ident: Dict[DefId, IdentifierIR],
    loc: SourceLocation,
    resolver: Optional[Any],
) -> None:
    """If block contains @ or @num/@den, create d_* bindings for block-local defids."""
    targets = _collect_differential_targets_in_expr(block)
    pairs = _collect_quotient_pairs_in_expr(block)
    if not targets and not pairs:
        return
    block_bindings = _bindings_in_block(block, None)
    if not block_bindings:
        return
    block_defids = {b.defid for b in block_bindings if b.defid is not None}
    target_defids: Set[DefId] = set()
    for did, _ in targets:
        target_defids.add(did)
    for num, den in pairs:
        target_defids.add(num)
        target_defids.add(den)
    target_defids &= block_defids
    if not target_defids:
        return

    block_binding_by_defid: Dict[DefId, BindingIR] = dict(scope_binding_by_defid)
    for b in block_bindings:
        if b.defid is not None:
            block_binding_by_defid[b.defid] = b
    binding_to_deps: Dict[DefId, Set[DefId]] = {}
    for b in block_bindings:
        if b.defid is not None and b.expr is not None:
            binding_to_deps[b.defid] = _collect_defids(b.expr)

    reachable: Set[DefId] = set(target_defids)
    work = list(reachable)
    while work:
        did = work.pop()
        for dep in binding_to_deps.get(did) or []:
            if dep in block_defids and dep not in reachable:
                reachable.add(dep)
                work.append(dep)

    forward_order: List[BindingIR] = []
    seen: Set[DefId] = set()

    def visit(did: DefId) -> None:
        if did in seen or did not in reachable:
            return
        seen.add(did)
        b = block_binding_by_defid.get(did)
        if b is None:
            return
        for dep in binding_to_deps.get(b.defid) or []:
            if dep in block_defids:
                visit(dep)
        forward_order.append(b)

    for did in target_defids:
        visit(did)

    if resolver is None:
        return

    loc0 = SourceLocation("", 0, 0)
    quotient_denominators = {den for _, den in pairs}
    leaves = {did for did in reachable if not (binding_to_deps.get(did) or set())}
    seed_value: Dict[DefId, int] = {}
    for b in forward_order:
        if b.defid is None:
            continue
        if b.defid in quotient_denominators:
            seed_value[b.defid] = 1
        elif b.defid in leaves and b.defid in target_defids:
            seed_value[b.defid] = 1
        else:
            seed_value[b.defid] = 0

    defid_to_d_ref_expr: Dict[DefId, ExpressionIR] = {}
    for did, ref in defid_to_d_ident.items():
        defid_to_d_ref_expr[did] = ref

    use_per_quotient_seeds = len(pairs) > 0
    defid_to_d_binding_block: Dict[DefId, BindingIR] = {}
    for b in forward_order:
        if b.defid is None or b.defid in defid_to_d_ident:
            continue
        bloc = b.location or loc0
        if use_per_quotient_seeds and b.defid in leaves:
            d_rhs: ExpressionIR = _float_lit(0, bloc)
        elif b.defid in seed_value and seed_value[b.defid] == 1:
            d_rhs = _float_lit(1, bloc)
        else:
            for dep in binding_to_deps.get(b.defid) or []:
                if dep not in defid_to_d_ref_expr:
                    defid_to_d_ref_expr[dep] = _float_lit(0, bloc)
            rhs = _forward_d_y_expr(b, defid_to_d_ref_expr, block_binding_by_defid, binding_to_deps, bloc, resolver)
            d_rhs = rhs if rhs is not None else _float_lit(0, bloc)

        d_defid = resolver.allocate_for_local()
        d_name = "d_" + (b.name or "")
        d_ref = IdentifierIR(d_name, bloc, d_defid)
        defid_to_d_ident[b.defid] = d_ref
        defid_to_d_ref_expr[b.defid] = d_ref
        ti = getattr(b, "type_info", None) or (getattr(b.expr, "type_info", None) if b.expr else None)
        si = getattr(b, "shape_info", None) or (getattr(b.expr, "shape_info", None) if b.expr else None)
        _set_type_info(d_rhs, ti, si)
        d_binding = BindingIR(name=d_name, expr=d_rhs, location=b.location, defid=d_defid, type_info=ti)
        defid_to_d_binding_block[b.defid] = d_binding

    new_stmts: List[Any] = []
    for stmt in block.statements or []:
        new_stmts.append(stmt)
        if isinstance(stmt, BindingIR) and stmt.defid is not None and stmt.defid in defid_to_d_binding_block:
            new_stmts.append(defid_to_d_binding_block[stmt.defid])
    object.__setattr__(block, "statements", new_stmts)


# ---------------------------------------------------------------------------
# Expansion: replace DifferentialIR and @num/@den in program
# ---------------------------------------------------------------------------

def _expand_derivative_in_expr(
    expr: ExpressionIR,
    defid_to_d_ident: Dict[DefId, IdentifierIR],
    scope_binding_by_defid: Dict[DefId, Any],
    scope_defid_to_expr: Dict[DefId, ExpressionIR],
    loc: SourceLocation,
    resolver: Optional[Any] = None,
) -> ExpressionIR:
    """Rewrite DifferentialIR -> d_ref; expand @num/@den to derivative.

    Quotient decision:
      - Direct dep (den_defid in num_expr AST): symbolic Jacobian via _SymbolicDiffVisitor
      - Transitive dep (reachable but not in AST): d_num / d_den quotient form
      - No dep: LiteralIR(0)
    """
    if isinstance(expr, DifferentialIR):
        operand = expr.operand
        if isinstance(operand, IdentifierIR) and operand.defid is not None:
            ref = defid_to_d_ident.get(operand.defid)
            if ref is not None:
                return IdentifierIR(
                    ref.name, expr.location, ref.defid,
                    type_info=getattr(expr, "type_info", None),
                    shape_info=getattr(expr, "shape_info", None),
                )
        vis = _ForwardDiffVisitor(defid_to_d_ident, loc)
        out = operand.accept(vis)
        if getattr(expr, "type_info", None) is not None and hasattr(out, "type_info"):
            out.type_info = expr.type_info
        if getattr(expr, "shape_info", None) is not None and hasattr(out, "shape_info"):
            out.shape_info = expr.shape_info
        return out

    if isinstance(expr, BinaryOpIR):
        if expr.operator == BinaryOp.DIV and isinstance(expr.left, DifferentialIR) and isinstance(expr.right, DifferentialIR):
            loc_q = expr.location or loc
            num_operand = expr.left.operand
            den_operand = expr.right.operand

            if isinstance(den_operand, IdentifierIR) and den_operand.defid is not None:
                den_defid = den_operand.defid

                if isinstance(num_operand, IdentifierIR) and num_operand.defid is not None:
                    num_expr = scope_defid_to_expr.get(num_operand.defid)
                    if num_expr is None:
                        raise ValueError("Autodiff: numerator of @num/@den has no defining expression")
                else:
                    num_expr = num_operand

                direct_deps = _collect_defids(num_expr)

                if den_defid in direct_deps:
                    vis = _SymbolicDiffVisitor(den_defid, loc_q, scope_binding_by_defid, resolver)
                    der = num_expr.accept(vis)
                    der = _simplify(der, loc_q)
                    ti = getattr(expr, "type_info", None)
                    si = getattr(expr, "shape_info", None)
                    if ti is not None or si is not None:
                        _set_type_info(der, ti, si)
                    return der
                elif (
                    isinstance(num_operand, IdentifierIR)
                    and num_operand.defid is not None
                    and _is_reachable(num_operand.defid, den_defid, scope_binding_by_defid)
                ):
                    if _path_has_einstein(num_operand.defid, den_defid, scope_binding_by_defid):
                        if num_operand.defid in defid_to_d_ident and den_defid in defid_to_d_ident:
                            quot = BinaryOpIR(
                                BinaryOp.DIV,
                                defid_to_d_ident[num_operand.defid],
                                defid_to_d_ident[den_defid],
                                loc_q,
                            )
                            ti = getattr(expr, "type_info", None)
                            si = getattr(expr, "shape_info", None)
                            if ti is not None or si is not None:
                                _set_type_info(quot, ti, si)
                            return quot
                    vis = _SymbolicDiffVisitor(den_defid, loc_q, scope_binding_by_defid, resolver)
                    der = num_expr.accept(vis)
                    der = _simplify(der, loc_q)
                    ti = getattr(expr, "type_info", None)
                    si = getattr(expr, "shape_info", None)
                    if ti is not None or si is not None:
                        _set_type_info(der, ti, si)
                    return der
                else:
                    return _float_lit(0, loc_q)
            else:
                if isinstance(num_operand, IdentifierIR) and num_operand.defid is not None:
                    num_e = num_operand
                else:
                    num_e = num_operand
                den_expr = den_operand
                defids = _collect_defids(den_expr)
                if len(defids) == 0:
                    raise ValueError("Autodiff: @num/@(expr) denominator depends on no variables")
                if len(defids) > 1:
                    raise ValueError("Autodiff: @num/@(expr) denominator depends on more than one variable")
                wrt_defid = next(iter(defids))
                vis_num = _SymbolicDiffVisitor(wrt_defid, loc, scope_binding_by_defid, resolver)
                vis_den = _SymbolicDiffVisitor(wrt_defid, loc, scope_binding_by_defid, resolver)
                d_num = num_e.accept(vis_num)
                d_den = den_expr.accept(vis_den)
                der = BinaryOpIR(BinaryOp.DIV, d_num, d_den, loc)
                ti = getattr(expr, "type_info", None)
                si = getattr(expr, "shape_info", None)
                if ti is not None or si is not None:
                    _set_type_info(der, ti, si)
                return der

        new_left = _expand_derivative_in_expr(expr.left, defid_to_d_ident, scope_binding_by_defid, scope_defid_to_expr, loc, resolver)
        new_right = _expand_derivative_in_expr(expr.right, defid_to_d_ident, scope_binding_by_defid, scope_defid_to_expr, loc, resolver)
        return BinaryOpIR(
            expr.operator, new_left, new_right, expr.location,
            type_info=getattr(expr, "type_info", None),
            shape_info=getattr(expr, "shape_info", None),
        )

    if isinstance(expr, UnaryOpIR):
        return UnaryOpIR(
            expr.operator,
            _expand_derivative_in_expr(expr.operand, defid_to_d_ident, scope_binding_by_defid, scope_defid_to_expr, loc, resolver),
            expr.location,
            type_info=getattr(expr, "type_info", None),
            shape_info=getattr(expr, "shape_info", None),
        )

    if isinstance(expr, EinsteinIR):
        new_clauses = []
        for c in (expr.clauses or []):
            new_value = _expand_derivative_in_expr(
                c.value, defid_to_d_ident, scope_binding_by_defid, scope_defid_to_expr, loc, resolver
            )
            new_clauses.append(EinsteinClauseIR(
                indices=c.indices, value=new_value, location=c.location,
                where_clause=c.where_clause,
                variable_ranges=dict(c.variable_ranges) if c.variable_ranges else {},
            ))
        return EinsteinIR(
            clauses=new_clauses, shape=expr.shape, element_type=expr.element_type,
            location=expr.location,
            type_info=getattr(expr, "type_info", None),
            shape_info=getattr(expr, "shape_info", None),
        )

    if isinstance(expr, BlockExpressionIR):
        _ensure_block_has_d_bindings(
            expr, scope_binding_by_defid, scope_defid_to_expr, defid_to_d_ident, loc, resolver
        )
        new_scope_binding = dict(scope_binding_by_defid)
        new_scope_expr = dict(scope_defid_to_expr)
        new_stmts: List[Any] = []
        for stmt in expr.statements or []:
            if isinstance(stmt, BindingIR):
                expanded_expr = _expand_derivative_in_expr(
                    stmt.expr, defid_to_d_ident, new_scope_binding, new_scope_expr, loc, resolver
                )
                new_binding = BindingIR(
                    name=stmt.name, expr=expanded_expr, location=stmt.location,
                    defid=stmt.defid,
                    type_info=getattr(stmt, "type_info", None),
                )
                if new_binding.defid is not None:
                    new_scope_binding[new_binding.defid] = new_binding
                    new_scope_expr[new_binding.defid] = expanded_expr
                new_stmts.append(new_binding)
            elif isinstance(stmt, ExpressionIR):
                new_stmts.append(_expand_derivative_in_expr(stmt, defid_to_d_ident, new_scope_binding, new_scope_expr, loc, resolver))
            else:
                new_stmts.append(stmt)
        new_final = _expand_derivative_in_expr(expr.final_expr, defid_to_d_ident, new_scope_binding, new_scope_expr, loc, resolver) if expr.final_expr is not None else None
        return BlockExpressionIR(
            statements=new_stmts, location=expr.location, final_expr=new_final,
            type_info=getattr(expr, "type_info", None),
            shape_info=getattr(expr, "shape_info", None),
        )

    if isinstance(expr, FunctionCallIR):
        new_args = [
            _expand_derivative_in_expr(a, defid_to_d_ident, scope_binding_by_defid, scope_defid_to_expr, loc, resolver)
            for a in (expr.arguments or [])
        ]
        return FunctionCallIR(
            callee_expr=expr.callee_expr, location=expr.location,
            arguments=new_args,
            module_path=getattr(expr, "module_path", None),
            type_info=getattr(expr, "type_info", None),
            shape_info=getattr(expr, "shape_info", None),
        )

    if isinstance(expr, BuiltinCallIR):
        args = expr.args or []
        if expr.builtin_name == "print" and len(args) == 1 and isinstance(args[0], DifferentialIR):
            t = _differential_target_from_operand(args[0].operand)
            if t is not None:
                y_defid, y_name = t
                y_expr = scope_defid_to_expr.get(y_defid)
                if y_expr is not None:
                    try:
                        fwd_vis = _ForwardDiffVisitor(defid_to_d_ident, loc)
                        diff_rhs = y_expr.accept(fwd_vis)
                        diff_rhs = _simplify(diff_rhs, expr.location or loc)
                        d_defid_to_at_name: Dict[DefId, str] = {}
                        for p in defid_to_d_ident:
                            b = scope_binding_by_defid.get(p)
                            d_defid_to_at_name[defid_to_d_ident[p].defid] = "@" + (b.name if b and getattr(b, "name", None) else "?")
                        rhs_str = _expr_to_diff_source(diff_rhs, d_defid_to_at_name, scope_binding_by_defid)
                        msg = "@" + (y_name or "?") + "=" + rhs_str + ";"
                        return BuiltinCallIR(
                            "print",
                            [LiteralIR(msg, expr.location, type_info=STR)],
                            expr.location,
                            defid=getattr(expr, "defid", None),
                            type_info=getattr(expr, "type_info", None),
                            shape_info=getattr(expr, "shape_info", None),
                        )
                    except (ValueError, KeyError):
                        pass
        new_args = [
            _expand_derivative_in_expr(a, defid_to_d_ident, scope_binding_by_defid, scope_defid_to_expr, loc, resolver)
            for a in args
        ]
        return BuiltinCallIR(
            expr.builtin_name, new_args, expr.location,
            defid=getattr(expr, "defid", None),
            type_info=getattr(expr, "type_info", None),
            shape_info=getattr(expr, "shape_info", None),
        )

    return expr


def _expand_derivative_nodes_in_program(
    program: ProgramIR,
    defid_to_d_ident: Dict[DefId, IdentifierIR],
    loc: SourceLocation,
    resolver: Optional[Any] = None,
    initial_binding_by_defid: Optional[Dict[DefId, Any]] = None,
) -> None:
    """In-place: replace DifferentialIR and DIV(@.,@.) in program bindings."""
    scope_binding_by_defid: Dict[DefId, Any] = {}
    scope_defid_to_expr: Dict[DefId, ExpressionIR] = {}
    if initial_binding_by_defid:
        scope_binding_by_defid = dict(initial_binding_by_defid)
        scope_defid_to_expr = {
            did: b.expr for did, b in initial_binding_by_defid.items()
            if getattr(b, "expr", None) is not None
        }
    for binding in program.bindings or []:
        if not isinstance(binding, BindingIR) or binding.expr is None:
            continue
        binding.expr = _expand_derivative_in_expr(
            binding.expr, defid_to_d_ident,
            scope_binding_by_defid, scope_defid_to_expr,
            binding.expr.location or loc, resolver,
        )
        if binding.defid is not None:
            scope_binding_by_defid[binding.defid] = binding
            scope_defid_to_expr[binding.defid] = binding.expr
    stmts = program.statements or []
    for i, stmt in enumerate(stmts):
        if not isinstance(stmt, BindingIR) and isinstance(stmt, ExpressionIR):
            stmts[i] = _expand_derivative_in_expr(
                stmt, defid_to_d_ident,
                scope_binding_by_defid, scope_defid_to_expr,
                getattr(stmt, "location", None) or loc, resolver,
            )


# ---------------------------------------------------------------------------
# AutodiffPass
# ---------------------------------------------------------------------------

class AutodiffPass(BasePass):
    """Expand @ and @y/@x into plain IR via forward diff."""

    requires = [
        TypeInferencePass,
        UnifiedShapeAnalysisPass,
    ]

    def run(self, ir: ProgramIR, tcx: TyCtxt) -> ProgramIR:
        program = ir
        try:
            return self._run_autodiff_core(program, tcx)
        finally:
            from ..ir.nodes import clear_autodiff_only_fields

            clear_autodiff_only_fields(program)

    def _run_autodiff_core(self, program: ProgramIR, tcx: TyCtxt) -> ProgramIR:
        bindings = _bindings_in_block(program, program) or []
        if not bindings:
            tcx.set_analysis(AutodiffPass, {
                "diff_block": None,
                "differential_targets": set(),
                "differential_buffer_by_defid": {},
            })
            return program

        # 1. Collect targets
        differential_targets = _collect_differential_targets(program)
        quotient_pairs = _collect_quotient_pairs(program)

        # 2. Build binding map
        binding_by_defid: Dict[DefId, BindingIR] = {}
        for b in bindings:
            if b.defid is not None:
                binding_by_defid[b.defid] = b
        function_ir_map = getattr(tcx, "function_ir_map", None) or {}
        for defid, fn in function_ir_map.items():
            if defid is not None and defid not in binding_by_defid and isinstance(fn, BindingIR) and is_function_binding(fn):
                binding_by_defid[defid] = fn

        # 3. Build dep graph
        binding_to_deps: Dict[DefId, Set[DefId]] = {}
        for b in bindings:
            if b.defid is not None:
                binding_to_deps[b.defid] = _collect_defids(b.expr)

        # 4. Compute reachable set from targets
        target_defids: Set[DefId] = set()
        for did, _ in differential_targets:
            target_defids.add(did)
        for num, den in quotient_pairs:
            target_defids.add(num)
            target_defids.add(den)

        top_level_defids = {b.defid for b in bindings if b.defid is not None}
        target_defids_for_diff = target_defids & top_level_defids

        reachable: Set[DefId] = set()
        work = list(target_defids_for_diff)
        while work:
            did = work.pop()
            if did in reachable:
                continue
            reachable.add(did)
            b = binding_by_defid.get(did)
            if b is None:
                continue
            for dep in binding_to_deps.get(b.defid) or []:
                if dep not in reachable:
                    work.append(dep)

        # 5. Topo sort
        forward_order: List[BindingIR] = []
        seen: Set[DefId] = set()

        def _visit(did: DefId) -> None:
            if did in seen or did not in reachable:
                return
            seen.add(did)
            b = binding_by_defid.get(did)
            if b is None:
                return
            for dep in binding_to_deps.get(b.defid) or []:
                _visit(dep)
            forward_order.append(b)

        for did in target_defids_for_diff:
            _visit(did)

        resolver = getattr(tcx, "resolver", None)
        if resolver is None:
            tcx.set_analysis(AutodiffPass, {
                "diff_block": None,
                "differential_targets": set(differential_targets),
                "differential_buffer_by_defid": {},
            })
            return program

        # 6. Create d_* identifiers
        defid_to_d_ident: Dict[DefId, IdentifierIR] = {}
        defid_to_d_binding: Dict[DefId, BindingIR] = {}
        seed_value: Dict[DefId, int] = {}
        quotient_denominators = {den for _, den in quotient_pairs}
        leaves = {did for did in reachable if not (binding_to_deps.get(did) or set())}

        for b in forward_order:
            if b.defid is None or b.defid not in reachable:
                continue
            d_name = "d_" + (b.name or "")
            d_defid = resolver.allocate_for_local()
            d_ref = IdentifierIR(d_name, b.location or SourceLocation("", 0, 0), d_defid)
            defid_to_d_ident[b.defid] = d_ref
            if b.defid in quotient_denominators:
                seed_value[b.defid] = 1
            elif b.defid in leaves and (b.defid in target_defids or len(quotient_pairs) == 0):
                seed_value[b.defid] = 1
            else:
                seed_value[b.defid] = 0

        # 7. Build d_* RHS expressions
        defid_to_d_ref_expr: Dict[DefId, ExpressionIR] = {}
        for did, ref in defid_to_d_ident.items():
            defid_to_d_ref_expr[did] = ref

        d_rhs_by_defid: Dict[DefId, ExpressionIR] = {}
        loc0 = SourceLocation("", 0, 0)
        use_per_quotient_seeds = len(quotient_pairs) > 0

        for b in forward_order:
            if b.defid is None or b.defid not in reachable:
                continue
            bloc = b.location or loc0
            if use_per_quotient_seeds and b.defid in leaves:
                d_rhs_by_defid[b.defid] = _float_lit(0, bloc)
            elif b.defid in seed_value and seed_value[b.defid] == 1:
                d_rhs_by_defid[b.defid] = _float_lit(1, bloc)
            else:
                for dep in binding_to_deps.get(b.defid) or []:
                    if dep not in defid_to_d_ref_expr:
                        defid_to_d_ref_expr[dep] = _float_lit(0, bloc)
                rhs = _forward_d_y_expr(b, defid_to_d_ref_expr, binding_by_defid, binding_to_deps, bloc, resolver)
                d_rhs_by_defid[b.defid] = rhs if rhs is not None else _float_lit(0, bloc)

        # 8. Create d_* bindings with type info
        type_info = None
        shape_info = None
        for b in bindings:
            if b.expr is not None and not isinstance(b.expr, FunctionValueIR):
                type_info = getattr(b, "type_info", None) or getattr(b.expr, "type_info", None)
                shape_info = getattr(b, "shape_info", None) or getattr(b.expr, "shape_info", None)
                if type_info is not None:
                    break

        for b in forward_order:
            if b.defid is None or b.defid not in reachable:
                continue
            rhs = d_rhs_by_defid.get(b.defid) or _float_lit(0, b.location or loc0)
            ti = getattr(b, "type_info", None) or (
                getattr(b.expr, "type_info", None) if b.expr is not None else None
            ) or type_info
            si = getattr(b, "shape_info", None) or (
                getattr(b.expr, "shape_info", None) if b.expr is not None else None
            ) or shape_info
            _set_type_info(rhs, ti, si)
            d_ref = defid_to_d_ident[b.defid]
            d_binding = BindingIR(
                name=d_ref.name, expr=rhs, location=b.location,
                defid=d_ref.defid, type_info=ti,
            )
            defid_to_d_binding[b.defid] = d_binding

        # 9. Insert d_* bindings after their primal
        new_bindings: List[BindingIR] = []
        for b in bindings:
            new_bindings.append(b)
            d_b = defid_to_d_binding.get(b.defid)
            if d_b is not None:
                new_bindings.append(d_b)

        program.bindings = new_bindings
        non_binding_stmts = [s for s in (program.statements or []) if not isinstance(s, BindingIR)]
        program.statements = new_bindings + non_binding_stmts

        # 10. Expand @ and @num/@den in program
        _expand_derivative_nodes_in_program(
            program, defid_to_d_ident, loc0, resolver,
            initial_binding_by_defid=binding_by_defid,
        )

        # 11. Store analysis for backend
        diff_block_list: List[BindingIR] = [
            defid_to_d_binding[b.defid]
            for b in forward_order
            if b.defid in defid_to_d_binding
        ]
        diff_block: Optional[List[BindingIR]] = diff_block_list if diff_block_list else None
        autodiff_differential_map: Dict[DefId, DefId] = {
            primal: d_binding.defid
            for primal, d_binding in defid_to_d_binding.items()
        }
        differential_leaves: Set[DefId] = leaves

        tcx.set_analysis(
            AutodiffPass,
            {
                "diff_block": diff_block,
                "differential_targets": set(differential_targets),
                "differential_buffer_by_defid": {},
                "autodiff_differential_map": autodiff_differential_map,
                "differential_leaves": differential_leaves,
            },
        )

        return program
