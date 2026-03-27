"""NumPy backend Einstein helpers."""

from .numpy_einstein_analysis import *
from .numpy_einstein_analysis import (
    _BodyReferencesDefidVisitor,
    _DefidsByNameCollector,
)

def _collect_defids_by_name(expr: Any) -> Dict[str, List[Any]]:
    """Collect all (name -> list of defids) from IdentifierIR/IndexVarIR in expr (e.g. for binding aliases)."""
    if expr is None:
        return {}
    return expr.accept(_DefidsByNameCollector())


class _BodyContainsCallUsingLoopVarVisitor(IRVisitor[bool]):
    """True if the tree contains a FunctionCallIR whose callee or any argument references a loop var."""

    def __init__(self, loop_defids: List[Any]) -> None:
        self._loop_defids = loop_defids

    def visit_function_call(self, node: Any) -> bool:
        for defid in self._loop_defids:
            if _BodyReferencesDefidVisitor(defid).references(node.callee_expr):
                return True
            for a in node.arguments:
                if _BodyReferencesDefidVisitor(defid).references(a):
                    return True
        return False

    def visit_rectangular_access(self, node: Any) -> bool:
        if node.array.accept(self):
            return True
        return any(idx.accept(self) for idx in node.indices)

    def visit_binary_op(self, node: Any) -> bool:
        return node.left.accept(self) or node.right.accept(self)

    def visit_block_expression(self, node: Any) -> bool:
        for stmt in node.statements:
            if stmt.accept(self):
                return True
        return (
            node.final_expr.accept(self)
            if node.final_expr is not None
            else False
        )

    def visit_binding(self, node: Any) -> bool:
        return node.expr.accept(self)

    def visit_guard_pattern(self, node: Any) -> bool:
        return node.inner_pattern.accept(self) or node.guard_expr.accept(self)

    def visit_or_pattern(self, node: Any) -> bool:
        return any(alt.accept(self) for alt in ((node.alternatives or [])))

    def visit_constructor_pattern(self, node: Any) -> bool:
        return any(p.accept(self) for p in node.patterns)

    def _false(self) -> bool:
        return False

    def visit_literal(self, node: Any) -> bool:
        return False

    def visit_identifier(self, node: Any) -> bool:
        return False

    def visit_index_var(self, node: Any) -> bool:
        return (
            node.range_ir.accept(self)
            if node.range_ir is not None
            else False
        )

    def visit_index_rest(self, node: Any) -> bool:
        return False

    def visit_unary_op(self, node: Any) -> bool:
        return node.operand.accept(self)

    def visit_if_expression(self, node: Any) -> bool:
        return (
            node.condition.accept(self)
            or node.then_expr.accept(self)
            or (node.else_expr.accept(self) if node.else_expr is not None else False)
        )

    def visit_lambda(self, node: Any) -> bool:
        return node.body.accept(self)

    def visit_range(self, node: Any) -> bool:
        return node.start.accept(self) or node.end.accept(self)

    def visit_array_comprehension(self, node: Any) -> bool:
        return node.body.accept(self)

    def visit_jagged_access(self, node: Any) -> bool:
        if node.base.accept(self):
            return True
        return any(idx.accept(self) for idx in node.index_chain)

    def visit_module(self, node: Any) -> bool:
        return False

    def visit_array_literal(self, node: Any) -> bool:
        return any(e.accept(self) for e in node.elements)

    def visit_tuple_expression(self, node: Any) -> bool:
        return any(e.accept(self) for e in node.elements)

    def visit_tuple_access(self, node: Any) -> bool:
        return node.tuple_expr.accept(self)

    def visit_interpolated_string(self, node: Any) -> bool:
        return any(p.accept(self) for p in node.parts)

    def visit_cast_expression(self, node: Any) -> bool:
        return node.expr.accept(self)

    def visit_member_access(self, node: Any) -> bool:
        return node.object.accept(self)

    def visit_try_expression(self, node: Any) -> bool:
        return node.operand.accept(self)

    def visit_match_expression(self, node: Any) -> bool:
        if node.scrutinee.accept(self):
            return True
        return any(arm.body.accept(self) for arm in node.arms)

    def visit_reduction_expression(self, node: Any) -> bool:
        return node.body.accept(self)

    def visit_where_expression(self, node: Any) -> bool:
        if node.expr.accept(self):
            return True
        return any(c.accept(self) for c in node.constraints)

    def visit_pipeline_expression(self, node: Any) -> bool:
        return node.left.accept(self) or node.right.accept(self)

    def visit_builtin_call(self, node: Any) -> bool:
        return any(a.accept(self) for a in node.args)

    def visit_literal_pattern(self, node: Any) -> bool:
        return False

    def visit_identifier_pattern(self, node: Any) -> bool:
        return False

    def visit_wildcard_pattern(self, node: Any) -> bool:
        return False

    def visit_tuple_pattern(self, node: Any) -> bool:
        return any(e.accept(self) for e in node.elements)

    def visit_array_pattern(self, node: Any) -> bool:
        return any(e.accept(self) for e in node.elements)

    def visit_rest_pattern(self, node: Any) -> bool:
        return False

    def visit_binding_pattern(self, node: Any) -> bool:
        return node.inner_pattern.accept(self)

    def visit_range_pattern(self, node: Any) -> bool:
        return False

    def visit_program(self, node: Any) -> bool:
        return any(stmt.accept(self) for stmt in node.statements)

    def visit_lowered_reduction(self, node: Any) -> bool:
        return node.body.accept(self)

    def visit_lowered_comprehension(self, node: Any) -> bool:
        return node.body.accept(self)

    def visit_lowered_einstein_clause(self, node: Any) -> bool:
        return node.body.accept(self)

    def visit_lowered_einstein(self, node: Any) -> bool:
        return any(item.accept(self) for item in node.items)

    def visit_lowered_recurrence(self, node: Any) -> bool:
        if node.initial is not None and node.initial.accept(self):
            return True
        rloop = node.recurrence_loop
        if rloop is not None and rloop.iterable is not None:
            if rloop.iterable.accept(self):
                return True
        return (
            node.body is not None
            and node.body.accept(self)
        )

    def visit_function_value(self, node: Any) -> bool:
        return node.body.accept(self) if node.body is not None else False

    def visit_einstein(self, node: Any) -> bool:
        return False

    def visit_einstein_clause(self, node: Any) -> bool:
        return False


def _body_contains_call_using_loop_var(expr: Any, loop_defids: List[Any]) -> bool:
    """True if body contains a FunctionCallIR whose arguments (or callee) reference a loop var."""
    if not expr or not loop_defids:
        return False
    return expr.accept(_BodyContainsCallUsingLoopVarVisitor(loop_defids))


class _DefidsInCallArgsCollector(IRVisitor[None]):
    """Collect into _out any loop defid referenced inside a call's arguments or callee. Tracks _inside_call for RectangularAccessIR indices."""

    def __init__(self, loop_defids: List[Any], out: set, inside_call: bool = False) -> None:
        self._loop_set = {d for d in loop_defids if d is not None}
        self._out = out
        self._inside_call = inside_call

    def visit_identifier(self, node: Any) -> None:
        if self._inside_call and node.defid is not None and node.defid in self._loop_set:
            self._out.add(node.defid)

    def visit_index_var(self, node: Any) -> None:
        if self._inside_call and node.defid is not None and node.defid in self._loop_set:
            self._out.add(node.defid)
        if node.range_ir is not None:
            node.range_ir.accept(self)

    def visit_index_rest(self, node: Any) -> None:
        if self._inside_call and node.defid is not None and node.defid in self._loop_set:
            self._out.add(node.defid)

    def visit_rectangular_access(self, node: Any) -> None:
        node.array.accept(self)
        if self._inside_call:
            for idx in node.indices:
                idx.accept(self)

    def visit_binary_op(self, node: Any) -> None:
        node.left.accept(self)
        node.right.accept(self)

    def visit_function_call(self, node: Any) -> None:
        prev = self._inside_call
        self._inside_call = True
        try:
            for a in node.arguments:
                a.accept(self)
            node.callee_expr.accept(self)
        finally:
            self._inside_call = prev

    def visit_block_expression(self, node: Any) -> None:
        for stmt in node.statements:
            stmt.accept(self)
        if node.final_expr is not None:
            node.final_expr.accept(self)

    def visit_binding(self, node: Any) -> None:
        node.expr.accept(self)

    def visit_guard_pattern(self, node: Any) -> None:
        node.inner_pattern.accept(self)
        node.guard_expr.accept(self)

    def visit_or_pattern(self, node: Any) -> None:
        for alt in (node.alternatives or []):
            alt.accept(self)

    def visit_constructor_pattern(self, node: Any) -> None:
        for p in node.patterns:
            p.accept(self)

    def _recurse_none(self) -> None:
        return None

    def visit_literal(self, node: Any) -> None:
        return None

    def visit_unary_op(self, node: Any) -> None:
        node.operand.accept(self)

    def visit_if_expression(self, node: Any) -> None:
        node.condition.accept(self)
        node.then_expr.accept(self)
        if node.else_expr is not None:
            node.else_expr.accept(self)

    def visit_lambda(self, node: Any) -> None:
        node.body.accept(self)

    def visit_range(self, node: Any) -> None:
        node.start.accept(self)
        node.end.accept(self)

    def visit_array_comprehension(self, node: Any) -> None:
        node.body.accept(self)

    def visit_jagged_access(self, node: Any) -> None:
        node.base.accept(self)
        for idx in node.index_chain:
            idx.accept(self)

    def visit_module(self, node: Any) -> None:
        return None

    def visit_array_literal(self, node: Any) -> None:
        for e in node.elements:
            e.accept(self)

    def visit_tuple_expression(self, node: Any) -> None:
        for e in node.elements:
            e.accept(self)

    def visit_tuple_access(self, node: Any) -> None:
        node.tuple_expr.accept(self)

    def visit_interpolated_string(self, node: Any) -> None:
        for p in node.parts:
            p.accept(self)

    def visit_cast_expression(self, node: Any) -> None:
        node.expr.accept(self)

    def visit_member_access(self, node: Any) -> None:
        node.object.accept(self)

    def visit_try_expression(self, node: Any) -> None:
        node.operand.accept(self)

    def visit_match_expression(self, node: Any) -> None:
        node.scrutinee.accept(self)
        for arm in node.arms:
            arm.body.accept(self)

    def visit_reduction_expression(self, node: Any) -> None:
        node.body.accept(self)

    def visit_where_expression(self, node: Any) -> None:
        node.expr.accept(self)
        for c in node.constraints:
            c.accept(self)

    def visit_pipeline_expression(self, node: Any) -> None:
        node.left.accept(self)
        node.right.accept(self)

    def visit_builtin_call(self, node: Any) -> None:
        for a in node.args:
            a.accept(self)

    def visit_literal_pattern(self, node: Any) -> None:
        return None

    def visit_identifier_pattern(self, node: Any) -> None:
        return None

    def visit_wildcard_pattern(self, node: Any) -> None:
        return None

    def visit_tuple_pattern(self, node: Any) -> None:
        for e in node.elements:
            e.accept(self)

    def visit_array_pattern(self, node: Any) -> None:
        for e in node.elements:
            e.accept(self)

    def visit_rest_pattern(self, node: Any) -> None:
        return None

    def visit_binding_pattern(self, node: Any) -> None:
        node.inner_pattern.accept(self)

    def visit_range_pattern(self, node: Any) -> None:
        return None

    def visit_program(self, node: Any) -> None:
        for stmt in node.statements:
            stmt.accept(self)

    def visit_lowered_reduction(self, node: Any) -> None:
        node.body.accept(self)

    def visit_lowered_comprehension(self, node: Any) -> None:
        node.body.accept(self)

    def visit_lowered_einstein_clause(self, node: Any) -> None:
        node.body.accept(self)

    def visit_lowered_einstein(self, node: Any) -> None:
        for item in node.items:
            item.accept(self)

    def visit_lowered_recurrence(self, node: Any) -> None:
        if node.initial is not None:
            node.initial.accept(self)
        rloop = node.recurrence_loop
        if rloop is not None and rloop.iterable is not None:
            rloop.iterable.accept(self)
        if node.body is not None:
            node.body.accept(self)

    def visit_function_value(self, node: Any) -> None:
        if node.body is not None:
            node.body.accept(self)

    def visit_einstein(self, node: Any) -> None:
        return None

    def visit_einstein_clause(self, node: Any) -> None:
        return None


def _collect_defids_in_call_args(expr: Any, loop_defids: List[Any], out: set, inside_call: bool = False) -> None:
    """Add to out any loop defid referenced inside a call's arguments or callee (not in outer indexing like result[j])."""
    if expr is None or not loop_defids:
        return
    expr.accept(_DefidsInCallArgsCollector(loop_defids, out, inside_call))


def _loop_defids_in_call_args(body: Any, loop_defids: List[Any]) -> set:
    """Loop defids that appear in any call's arguments (or callee). Those dims must be scalar for call-scalar hybrid."""
    loop_set = {d for d in loop_defids if d is not None}
    out: set = set()
    _collect_defids_in_call_args(body, list(loop_set), out)
    return out


def _body_is_elementwise_call(body: Any, loop_defids: List[Any]) -> bool:
    """True if the body is (or ends in) a single function call and every loop var appears in that call's arguments.
    Such clauses are element-wise and must use the vectorized path (one call with full array), not the scalar loop."""
    if body is None or not loop_defids:
        return False
    loop_set = {d for d in loop_defids if d is not None}
    if not loop_set:
        return False
    in_call = _loop_defids_in_call_args(body, loop_defids)
    if in_call != loop_set:
        return False
    if isinstance(body, FunctionCallIR):
        return True
    if isinstance(body, BlockExpressionIR):
        return _body_is_elementwise_call(body.final_expr, loop_defids)
    return False


class _IndexExprIsLoopVarVisitor(IRVisitor[bool]):
    """True iff expr is IdentifierIR or IndexVarIR with defid == loop_defid."""

    def __init__(self, loop_defid: Any) -> None:
        self._loop_defid = loop_defid

    def visit_identifier(self, node: Any) -> bool:
        return node.defid == self._loop_defid

    def visit_index_var(self, node: Any) -> bool:
        return node.defid == self._loop_defid

    def visit_literal(self, node: Any) -> bool:
        return False

    def visit_binary_op(self, node: Any) -> bool:
        return False

    def visit_index_rest(self, node: Any) -> bool:
        return False

    def visit_rectangular_access(self, node: Any) -> bool:
        return False

    def visit_function_call(self, node: Any) -> bool:
        return False

    def visit_unary_op(self, node: Any) -> bool:
        return False

    def visit_block_expression(self, node: Any) -> bool:
        return False

    def visit_if_expression(self, node: Any) -> bool:
        return False

    def visit_lambda(self, node: Any) -> bool:
        return False

    def visit_range(self, node: Any) -> bool:
        return False

    def visit_array_comprehension(self, node: Any) -> bool:
        return False

    def visit_jagged_access(self, node: Any) -> bool:
        return False

    def visit_module(self, node: Any) -> bool:
        return False

    def visit_array_literal(self, node: Any) -> bool:
        return False

    def visit_tuple_expression(self, node: Any) -> bool:
        return False

    def visit_tuple_access(self, node: Any) -> bool:
        return False

    def visit_interpolated_string(self, node: Any) -> bool:
        return False

    def visit_cast_expression(self, node: Any) -> bool:
        return False

    def visit_member_access(self, node: Any) -> bool:
        return False

    def visit_try_expression(self, node: Any) -> bool:
        return False

    def visit_match_expression(self, node: Any) -> bool:
        return False

    def visit_reduction_expression(self, node: Any) -> bool:
        return False

    def visit_where_expression(self, node: Any) -> bool:
        return False

    def visit_pipeline_expression(self, node: Any) -> bool:
        return False

    def visit_builtin_call(self, node: Any) -> bool:
        return False

    def visit_literal_pattern(self, node: Any) -> bool:
        return False

    def visit_identifier_pattern(self, node: Any) -> bool:
        return False

    def visit_wildcard_pattern(self, node: Any) -> bool:
        return False

    def visit_tuple_pattern(self, node: Any) -> bool:
        return False

    def visit_array_pattern(self, node: Any) -> bool:
        return False

    def visit_rest_pattern(self, node: Any) -> bool:
        return False

    def visit_guard_pattern(self, node: Any) -> bool:
        return False

    def visit_binding(self, node: Any) -> bool:
        return False

    def visit_program(self, node: Any) -> bool:
        return False

    def visit_lowered_reduction(self, node: Any) -> bool:
        return False

    def visit_lowered_comprehension(self, node: Any) -> bool:
        return False

    def visit_lowered_einstein_clause(self, node: Any) -> bool:
        return False

    def visit_lowered_einstein(self, node: Any) -> bool:
        return False

    def visit_lowered_recurrence(self, node: Any) -> bool:
        return False

    def visit_or_pattern(self, node: Any) -> bool:
        return False

    def visit_constructor_pattern(self, node: Any) -> bool:
        return False

    def visit_binding_pattern(self, node: Any) -> bool:
        return False

    def visit_range_pattern(self, node: Any) -> bool:
        return False

    def visit_function_value(self, node: Any) -> bool:
        return False

    def visit_einstein(self, node: Any) -> bool:
        return False

    def visit_einstein_clause(self, node: Any) -> bool:
        return False


def _index_expr_is_loop_var(expr: Any, loop_defid: Any) -> bool:
    if expr is None or loop_defid is None:
        return False
    return expr.accept(_IndexExprIsLoopVarVisitor(loop_defid))


class _IndexExprIsBackwardVisitor(IRVisitor[bool]):
    """True if expr is loop_var or (loop_var - positive_const)."""

    def __init__(self, loop_defid: Any) -> None:
        self._loop_defid = loop_defid

    def visit_identifier(self, node: Any) -> bool:
        return node.defid == self._loop_defid

    def visit_index_var(self, node: Any) -> bool:
        return node.defid == self._loop_defid

    def visit_binary_op(self, node: Any) -> bool:
        if node.operator != BinaryOp.SUB:
            return False
        if not _index_expr_is_loop_var(node.left, self._loop_defid):
            return False
        if isinstance(node.right, LiteralIR):
            try:
                return int(node.right.value) > 0
            except (TypeError, ValueError):
                pass
        return False

    def visit_literal(self, node: Any) -> bool:
        return False

    def visit_index_rest(self, node: Any) -> bool:
        return False

    def visit_rectangular_access(self, node: Any) -> bool:
        return False

    def visit_function_call(self, node: Any) -> bool:
        return False

    def visit_unary_op(self, node: Any) -> bool:
        return False

    def visit_block_expression(self, node: Any) -> bool:
        return False

    def visit_if_expression(self, node: Any) -> bool:
        return False

    def visit_lambda(self, node: Any) -> bool:
        return False

    def visit_range(self, node: Any) -> bool:
        return False

    def visit_array_comprehension(self, node: Any) -> bool:
        return False

    def visit_jagged_access(self, node: Any) -> bool:
        return False

    def visit_module(self, node: Any) -> bool:
        return False

    def visit_array_literal(self, node: Any) -> bool:
        return False

    def visit_tuple_expression(self, node: Any) -> bool:
        return False

    def visit_tuple_access(self, node: Any) -> bool:
        return False

    def visit_interpolated_string(self, node: Any) -> bool:
        return False

    def visit_cast_expression(self, node: Any) -> bool:
        return False

    def visit_member_access(self, node: Any) -> bool:
        return False

    def visit_try_expression(self, node: Any) -> bool:
        return False

    def visit_match_expression(self, node: Any) -> bool:
        return False

    def visit_reduction_expression(self, node: Any) -> bool:
        return False

    def visit_where_expression(self, node: Any) -> bool:
        return False

    def visit_pipeline_expression(self, node: Any) -> bool:
        return False

    def visit_builtin_call(self, node: Any) -> bool:
        return False

    def visit_literal_pattern(self, node: Any) -> bool:
        return False

    def visit_identifier_pattern(self, node: Any) -> bool:
        return False

    def visit_wildcard_pattern(self, node: Any) -> bool:
        return False

    def visit_tuple_pattern(self, node: Any) -> bool:
        return False

    def visit_array_pattern(self, node: Any) -> bool:
        return False

    def visit_rest_pattern(self, node: Any) -> bool:
        return False

    def visit_guard_pattern(self, node: Any) -> bool:
        return False

    def visit_binding(self, node: Any) -> bool:
        return False

    def visit_program(self, node: Any) -> bool:
        return False

    def visit_lowered_reduction(self, node: Any) -> bool:
        return False

    def visit_lowered_comprehension(self, node: Any) -> bool:
        return False

    def visit_lowered_einstein_clause(self, node: Any) -> bool:
        return False

    def visit_lowered_einstein(self, node: Any) -> bool:
        return False

    def visit_lowered_recurrence(self, node: Any) -> bool:
        return False

    def visit_or_pattern(self, node: Any) -> bool:
        return False

    def visit_constructor_pattern(self, node: Any) -> bool:
        return False

    def visit_binding_pattern(self, node: Any) -> bool:
        return False

    def visit_range_pattern(self, node: Any) -> bool:
        return False

    def visit_function_value(self, node: Any) -> bool:
        return False

    def visit_einstein(self, node: Any) -> bool:
        return False

    def visit_einstein_clause(self, node: Any) -> bool:
        return False


def _index_expr_is_backward(expr: Any, loop_defid: Any) -> bool:
    """True if expr is loop_var or (loop_var - positive_const). Recurrence dim: no loop var on RHS, only t-1 style."""
    if expr is None or loop_defid is None:
        return False
    return expr.accept(_IndexExprIsBackwardVisitor(loop_defid))


class _IndexExprIsStrictlyBackwardVisitor(IRVisitor[bool]):
    """True only for (loop_var - positive_const). False for loop_var (same index)."""

    def __init__(self, loop_defid: Any) -> None:
        self._loop_defid = loop_defid

    def visit_identifier(self, node: Any) -> bool:
        return False

    def visit_index_var(self, node: Any) -> bool:
        return False

    def visit_binary_op(self, node: Any) -> bool:
        if node.operator != BinaryOp.SUB:
            return False
        if not _index_expr_is_loop_var(node.left, self._loop_defid):
            return False
        if isinstance(node.right, LiteralIR):
            try:
                return int(node.right.value) > 0
            except (TypeError, ValueError):
                pass
        return False

    def visit_literal(self, node: Any) -> bool:
        return False

    def visit_index_rest(self, node: Any) -> bool:
        return False

    def visit_rectangular_access(self, node: Any) -> bool:
        return False

    def visit_function_call(self, node: Any) -> bool:
        return False

    def visit_unary_op(self, node: Any) -> bool:
        return False

    def visit_block_expression(self, node: Any) -> bool:
        return False

    def visit_if_expression(self, node: Any) -> bool:
        return False

    def visit_lambda(self, node: Any) -> bool:
        return False

    def visit_range(self, node: Any) -> bool:
        return False

    def visit_array_comprehension(self, node: Any) -> bool:
        return False

    def visit_jagged_access(self, node: Any) -> bool:
        return False

    def visit_module(self, node: Any) -> bool:
        return False

    def visit_array_literal(self, node: Any) -> bool:
        return False

    def visit_tuple_expression(self, node: Any) -> bool:
        return False

    def visit_tuple_access(self, node: Any) -> bool:
        return False

    def visit_interpolated_string(self, node: Any) -> bool:
        return False

    def visit_cast_expression(self, node: Any) -> bool:
        return False

    def visit_member_access(self, node: Any) -> bool:
        return False

    def visit_try_expression(self, node: Any) -> bool:
        return False

    def visit_match_expression(self, node: Any) -> bool:
        return False

    def visit_reduction_expression(self, node: Any) -> bool:
        return False

    def visit_where_expression(self, node: Any) -> bool:
        return False

    def visit_pipeline_expression(self, node: Any) -> bool:
        return False

    def visit_builtin_call(self, node: Any) -> bool:
        return False

    def visit_literal_pattern(self, node: Any) -> bool:
        return False

    def visit_identifier_pattern(self, node: Any) -> bool:
        return False

    def visit_wildcard_pattern(self, node: Any) -> bool:
        return False

    def visit_tuple_pattern(self, node: Any) -> bool:
        return False

    def visit_array_pattern(self, node: Any) -> bool:
        return False

    def visit_rest_pattern(self, node: Any) -> bool:
        return False

    def visit_guard_pattern(self, node: Any) -> bool:
        return False

    def visit_binding(self, node: Any) -> bool:
        return False

    def visit_program(self, node: Any) -> bool:
        return False

    def visit_lowered_reduction(self, node: Any) -> bool:
        return False

    def visit_lowered_comprehension(self, node: Any) -> bool:
        return False

    def visit_lowered_einstein_clause(self, node: Any) -> bool:
        return False

    def visit_lowered_einstein(self, node: Any) -> bool:
        return False

    def visit_lowered_recurrence(self, node: Any) -> bool:
        return False

    def visit_or_pattern(self, node: Any) -> bool:
        return False

    def visit_constructor_pattern(self, node: Any) -> bool:
        return False

    def visit_binding_pattern(self, node: Any) -> bool:
        return False

    def visit_range_pattern(self, node: Any) -> bool:
        return False

    def visit_function_value(self, node: Any) -> bool:
        return False

    def visit_einstein(self, node: Any) -> bool:
        return False

    def visit_einstein_clause(self, node: Any) -> bool:
        return False
