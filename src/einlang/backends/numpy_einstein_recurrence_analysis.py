"""NumPy backend Einstein helpers."""

from .numpy_einstein_call_index_analysis import *
from .numpy_einstein_analysis import _BodyReferencesDefidVisitor
from .numpy_einstein_call_index_analysis import (
    _IndexExprIsStrictlyBackwardVisitor,
    _index_expr_is_loop_var,
)

def _index_expr_is_strictly_backward(expr: Any, loop_defid: Any) -> bool:
    """True only for (loop_var - positive_const). False for loop_var (same index). Used so hybrid recurrence is only t-1 style."""
    if expr is None or loop_defid is None:
        return False
    return expr.accept(_IndexExprIsStrictlyBackwardVisitor(loop_defid))


class _IndexExprIsLoopVarOrOffsetVisitor(IRVisitor[bool]):
    """True if expr is loop_var or (loop_var ± const)."""

    def __init__(self, loop_defid: Any) -> None:
        self._loop_defid = loop_defid

    def visit_identifier(self, node: Any) -> bool:
        return node.defid == self._loop_defid

    def visit_index_var(self, node: Any) -> bool:
        return node.defid == self._loop_defid

    def visit_binary_op(self, node: Any) -> bool:
        op = node.operator
        if op not in (BinaryOp.ADD, BinaryOp.SUB):
            return False
        if not _index_expr_is_loop_var(node.left, self._loop_defid):
            return False
        if isinstance(node.right, LiteralIR):
            try:
                int(node.right.value)
                return True
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


def _index_expr_is_loop_var_or_offset(expr: Any, loop_defid: Any) -> bool:
    """True if expr is loop_var or (loop_var ± const). Such dims are not recurrence; we can vectorize (e.g. i, i-1, i+1)."""
    if expr is None or loop_defid is None:
        return False
    return expr.accept(_IndexExprIsLoopVarOrOffsetVisitor(loop_defid))


class _ExprIsLoopVarOrMinusOneVisitor(IRVisitor[bool]):
    """True if expr is loop_var or (loop_var - 1)."""

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
                return int(node.right.value) == 1
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


def _expr_is_loop_var_or_minus_one(expr: Any, loop_defid: Any) -> bool:
    """True if expr is loop_var or (loop_var - 1). Used to detect reduction upper bound like 0..(j-1)."""
    if expr is None or loop_defid is None:
        return False
    return expr.accept(_ExprIsLoopVarOrMinusOneVisitor(loop_defid))


def _recurrence_dims_for_hybrid(
    lowered: Any, variable_defid: Any, clause_indices: Optional[List[Any]] = None
) -> List[int]:
    """Loop indices k where every LHS read is strictly backward (e.g. t-1). Same timestep (e.g. t) is not recurrence.
    Not used for partition/step: we need to accept both backward-in-time and same-timestep; use _recurrence_dims."""
    loops = (lowered.loops or [])
    if not loops or variable_defid is None:
        return []
    loop_defids = [lp.variable.defid for lp in loops]
    read_index_lists = _collect_lhs_read_index_lists(lowered.body, variable_defid)
    if not read_index_lists:
        return []
    loop_dims = _loop_dims_from_clause_indices(clause_indices, loops) if clause_indices else None
    result: List[int] = []
    for k in range(len(loops)):
        out_d = loop_dims[k] if loop_dims is not None and k < len(loop_dims) else k
        if all(
            out_d < len(idx_list) and _index_expr_is_strictly_backward(idx_list[out_d], loop_defids[k])
            for idx_list in read_index_lists
        ):
            result.append(k)
    return result


def _recurrence_dims_for_hybrid_or_full(
    lowered: Any, variable_defid: Any, clause_indices: Optional[List[Any]] = None
) -> List[int]:
    """Strict backward only (every read t-1). Returns [] if any read is same timestep (t).
    For partition/step we accept backward-in-time and same-timestep; use _recurrence_dims. This is for hybrid vectorized path only."""
    loops = (lowered.loops or [])
    if not loops or variable_defid is None:
        return []
    recurrence_for_hybrid = _recurrence_dims_for_hybrid(lowered, variable_defid, clause_indices)
    if not recurrence_for_hybrid:
        return []
    loop_defids = [lp.variable.defid for lp in loops]
    read_index_lists = _collect_lhs_read_index_lists(lowered.body, variable_defid)
    loop_dims = _loop_dims_from_clause_indices(clause_indices, loops) if clause_indices else None
    for k in range(len(loops)):
        if k in recurrence_for_hybrid:
            continue
        out_d = loop_dims[k] if loop_dims is not None and k < len(loop_dims) else k
        for idx_list in read_index_lists:
            if out_d >= len(idx_list):
                continue
            if not _index_expr_is_loop_var_or_offset(idx_list[out_d], loop_defids[k]):
                return _recurrence_dims(lowered, variable_defid, clause_indices)
    return recurrence_for_hybrid


class _LHSReadIndexListsCollector(IRVisitor[List[List[Any]]]):
    """Collect index lists from RectangularAccessIR nodes whose array references target_defid (LHS read positions)."""

    def __init__(self, target_defid: Any) -> None:
        self._target_defid = target_defid

    def _empty(self) -> List[List[Any]]:
        return []

    def visit_rectangular_access(self, node: Any) -> List[List[Any]]:
        out: List[List[Any]] = []
        if _BodyReferencesDefidVisitor(self._target_defid).references(node.array):
            if node.indices:
                out.append(list(node.indices))
        out.extend(node.array.accept(self))
        return out

    def visit_binary_op(self, node: Any) -> List[List[Any]]:
        out = node.left.accept(self)
        out.extend(node.right.accept(self))
        return out

    def visit_function_call(self, node: Any) -> List[List[Any]]:
        out = self._empty()
        for a in node.arguments:
            out.extend(a.accept(self))
        out.extend(node.callee_expr.accept(self))
        return out

    def visit_block_expression(self, node: Any) -> List[List[Any]]:
        out = self._empty()
        for stmt in node.statements:
            out.extend(stmt.accept(self))
        if node.final_expr is not None:
            out.extend(node.final_expr.accept(self))
        return out

    def visit_binding(self, node: Any) -> List[List[Any]]:
        return node.expr.accept(self)

    def visit_lambda(self, node: Any) -> List[List[Any]]:
        return node.body.accept(self)

    def visit_if_expression(self, node: Any) -> List[List[Any]]:
        out = node.condition.accept(self)
        out.extend(node.then_expr.accept(self))
        if node.else_expr is not None:
            out.extend(node.else_expr.accept(self))
        return out

    def visit_literal(self, node: Any) -> List[List[Any]]:
        return self._empty()

    def visit_identifier(self, node: Any) -> List[List[Any]]:
        return self._empty()

    def visit_index_var(self, node: Any) -> List[List[Any]]:
        return (
            node.range_ir.accept(self)
            if node.range_ir is not None
            else self._empty()
        )

    def visit_index_rest(self, node: Any) -> List[List[Any]]:
        return self._empty()

    def visit_unary_op(self, node: Any) -> List[List[Any]]:
        return node.operand.accept(self)

    def visit_range(self, node: Any) -> List[List[Any]]:
        out = node.start.accept(self)
        out.extend(node.end.accept(self))
        return out

    def visit_array_comprehension(self, node: Any) -> List[List[Any]]:
        return node.body.accept(self)

    def visit_jagged_access(self, node: Any) -> List[List[Any]]:
        out = node.base.accept(self)
        for idx in node.index_chain:
            out.extend(idx.accept(self))
        return out

    def visit_module(self, node: Any) -> List[List[Any]]:
        return self._empty()

    def visit_array_literal(self, node: Any) -> List[List[Any]]:
        out = self._empty()
        for e in node.elements:
            out.extend(e.accept(self))
        return out

    def visit_tuple_expression(self, node: Any) -> List[List[Any]]:
        out = self._empty()
        for e in node.elements:
            out.extend(e.accept(self))
        return out

    def visit_tuple_access(self, node: Any) -> List[List[Any]]:
        return node.tuple_expr.accept(self)

    def visit_interpolated_string(self, node: Any) -> List[List[Any]]:
        out = self._empty()
        for p in node.parts:
            out.extend(p.accept(self))
        return out

    def visit_cast_expression(self, node: Any) -> List[List[Any]]:
        return node.expr.accept(self)

    def visit_member_access(self, node: Any) -> List[List[Any]]:
        return node.object.accept(self)

    def visit_try_expression(self, node: Any) -> List[List[Any]]:
        return node.operand.accept(self)

    def visit_match_expression(self, node: Any) -> List[List[Any]]:
        out = node.scrutinee.accept(self)
        for arm in node.arms:
            out.extend(arm.body.accept(self))
        return out

    def visit_reduction_expression(self, node: Any) -> List[List[Any]]:
        return node.body.accept(self)

    def visit_where_expression(self, node: Any) -> List[List[Any]]:
        out = node.expr.accept(self)
        for c in node.constraints:
            out.extend(c.accept(self))
        return out

    def visit_pipeline_expression(self, node: Any) -> List[List[Any]]:
        out = node.left.accept(self)
        out.extend(node.right.accept(self))
        return out

    def visit_builtin_call(self, node: Any) -> List[List[Any]]:
        out = self._empty()
        for a in node.args:
            out.extend(a.accept(self))
        return out

    def visit_literal_pattern(self, node: Any) -> List[List[Any]]:
        return self._empty()

    def visit_identifier_pattern(self, node: Any) -> List[List[Any]]:
        return self._empty()

    def visit_wildcard_pattern(self, node: Any) -> List[List[Any]]:
        return self._empty()

    def visit_tuple_pattern(self, node: Any) -> List[List[Any]]:
        out = self._empty()
        for e in node.elements:
            out.extend(e.accept(self))
        return out

    def visit_array_pattern(self, node: Any) -> List[List[Any]]:
        out = self._empty()
        for e in node.elements:
            out.extend(e.accept(self))
        return out

    def visit_rest_pattern(self, node: Any) -> List[List[Any]]:
        return self._empty()

    def visit_guard_pattern(self, node: Any) -> List[List[Any]]:
        out = node.inner_pattern.accept(self)
        out.extend(node.guard_expr.accept(self))
        return out

    def visit_program(self, node: Any) -> List[List[Any]]:
        out = self._empty()
        for stmt in node.statements:
            out.extend(stmt.accept(self))
        return out

    def visit_lowered_reduction(self, node: Any) -> List[List[Any]]:
        return node.body.accept(self)

    def visit_lowered_comprehension(self, node: Any) -> List[List[Any]]:
        return node.body.accept(self)

    def visit_lowered_einstein_clause(self, node: Any) -> List[List[Any]]:
        return node.body.accept(self)

    def visit_lowered_einstein(self, node: Any) -> List[List[Any]]:
        out = self._empty()
        for item in node.items:
            out.extend(item.accept(self))
        return out

    def visit_lowered_recurrence(self, node: Any) -> List[List[Any]]:
        out = self._empty()
        if node.initial is not None:
            out.extend(node.initial.accept(self))
        rloop = node.recurrence_loop
        if rloop is not None and rloop.iterable is not None:
            out.extend(rloop.iterable.accept(self))
        if node.body is not None:
            out.extend(node.body.accept(self))
        return out

    def visit_or_pattern(self, node: Any) -> List[List[Any]]:
        out = self._empty()
        for alt in (node.alternatives or []):
            out.extend(alt.accept(self))
        return out

    def visit_constructor_pattern(self, node: Any) -> List[List[Any]]:
        out = self._empty()
        for p in node.patterns:
            out.extend(p.accept(self))
        return out

    def visit_binding_pattern(self, node: Any) -> List[List[Any]]:
        return node.inner_pattern.accept(self)

    def visit_range_pattern(self, node: Any) -> List[List[Any]]:
        return self._empty()

    def visit_function_value(self, node: Any) -> List[List[Any]]:
        return node.body.accept(self) if node.body is not None else self._empty()

    def visit_einstein(self, node: Any) -> List[List[Any]]:
        return self._empty()

    def visit_einstein_clause(self, node: Any) -> List[List[Any]]:
        return self._empty()


def _collect_lhs_read_index_lists(body: Any, target_defid: Any) -> List[List[Any]]:
    """Collect index lists from array accesses whose array references target_defid (LHS read positions)."""
    if body is None:
        return []
    return body.accept(_LHSReadIndexListsCollector(target_defid))


def _recurrence_dims(lowered: Any, variable_defid: Any, clause_indices: Optional[List[Any]] = None) -> List[int]:
    """Return loop indices k where any LHS read differs from the write (recurrence).
    Accepts both backward-in-time (e.g. t-1) and same-timestep (e.g. state[t,0] when writing state[t,1]);
    we add k if any read is not the loop var, so clauses with mixed t-1 and t reads still get k and run in timestep-major.
    When clause_indices is set, index read lists by output dim loop_dims[k], not k."""
    loops = (lowered.loops or [])
    if not loops or variable_defid is None:
        return []
    loop_defids = [lp.variable.defid for lp in loops]
    read_index_lists = _collect_lhs_read_index_lists(lowered.body, variable_defid)
    if not read_index_lists:
        return []
    loop_dims = _loop_dims_from_clause_indices(clause_indices, loops) if clause_indices else None
    recurrence: List[int] = []
    for k in range(len(loops)):
        out_d = loop_dims[k] if loop_dims is not None and k < len(loop_dims) else k
        for idx_list in read_index_lists:
            if out_d >= len(idx_list):
                continue
            if not _index_expr_is_loop_var(idx_list[out_d], loop_defids[k]):
                recurrence.append(k)
                break
    return recurrence


def _indices_exact_diagonal_match(rect_indices: List[Any], clause_indices: List[Any]) -> bool:
    """True iff each index is an IndexVarIR and pairwise defids match clause output indices."""
    ri = list(rect_indices or [])
    ci = list(clause_indices or [])
    if len(ri) != len(ci):
        return False
    for a, b in zip(ri, ci):
        if not isinstance(a, IndexVarIR) or not isinstance(b, IndexVarIR):
            return False
        if a.defid is None or b.defid is None or a.defid != b.defid:
            return False
    return True


def _has_recurrence_style_read_of_defid(
    body: Any, target_defid: Any, clause_indices: List[Any],
) -> bool:
    """True if some read of ``target_defid`` is a recurrence (e.g. u[t-1]), not input diagonal (x[i])."""
    found = False

    def walk(expr: Any) -> None:
        nonlocal found
        if expr is None or found:
            return
        if isinstance(expr, RectangularAccessIR):
            arr = expr.array
            if isinstance(arr, IdentifierIR) and arr.defid == target_defid:
                ridx = list(expr.indices or [])
                cidx = list(clause_indices or [])
                if _indices_exact_diagonal_match(ridx, cidx):
                    pass
                elif len(ridx) == len(cidx):
                    for ix in ridx:
                        if not isinstance(ix, IndexVarIR):
                            found = True
                            return
                    for a, b in zip(ridx, cidx):
                        if isinstance(a, IndexVarIR) and isinstance(b, IndexVarIR) and a.defid != b.defid:
                            found = True
                            return
                else:
                    pass
            walk(arr)
            for ix in expr.indices or []:
                walk(ix)
        elif isinstance(expr, BinaryOpIR):
            walk(expr.left)
            walk(expr.right)
        elif isinstance(expr, UnaryOpIR):
            walk(expr.operand)
        elif isinstance(expr, IfExpressionIR):
            walk(expr.condition)
            walk(expr.then_expr)
            if expr.else_expr is not None:
                walk(expr.else_expr)
        elif isinstance(expr, FunctionCallIR):
            walk(expr.callee_expr)
            for a in expr.arguments or []:
                walk(a)
        elif isinstance(expr, BlockExpressionIR):
            for st in expr.statements or []:
                walk(st)
            if expr.final_expr is not None:
                walk(expr.final_expr)

    walk(body)
    return found


def _infer_lowered_einstein_output_defid(lowered: LoweredEinsteinIR) -> Any:
    """
    DefId of the tensor this Einstein declaration writes: any DefId that appears as the array of a
    RectangularAccessIR in a clause body and is self-referenced there. Used when Autodiff nests
    LoweredEinsteinIR under RectangularAccessIR (pullback) so execution can push a synthetic decl stack.

    When the array is itself a nested LoweredEinsteinIR (re-materialized primal), recurse to the inner
    tensor (same DefId as the leaf read, e.g. u inside euler_decay).

    Diagonal input reads (``x[i]`` with clause ``[i]``) are excluded: they must use a fresh buffer so
    nested pullback does not alias ``x`` and corrupt it. Recurrence reads (``u[t - 1]`` with clause ``[t]``)
    still resolve to ``u`` so in-place recurrence execution works.
    """
    for item in lowered.items or []:
        if item.body is None:
            continue
        candidates: set = set()
        inner_einsteins: List[LoweredEinsteinIR] = []

        def collect_rect_array_defids(expr: Any) -> None:
            if expr is None:
                return
            if isinstance(expr, RectangularAccessIR):
                arr = expr.array
                if isinstance(arr, IdentifierIR) and arr.defid is not None:
                    candidates.add(arr.defid)
                elif isinstance(arr, LoweredEinsteinIR):
                    inner_einsteins.append(arr)
                collect_rect_array_defids(arr)
                for ix in expr.indices or []:
                    collect_rect_array_defids(ix)
            elif isinstance(expr, BinaryOpIR):
                collect_rect_array_defids(expr.left)
                collect_rect_array_defids(expr.right)
            elif isinstance(expr, UnaryOpIR):
                collect_rect_array_defids(expr.operand)
            elif isinstance(expr, IfExpressionIR):
                collect_rect_array_defids(expr.condition)
                collect_rect_array_defids(expr.then_expr)
                if expr.else_expr is not None:
                    collect_rect_array_defids(expr.else_expr)
            elif isinstance(expr, FunctionCallIR):
                collect_rect_array_defids(expr.callee_expr)
                for a in expr.arguments or []:
                    collect_rect_array_defids(a)
            elif isinstance(expr, BlockExpressionIR):
                for st in expr.statements or []:
                    collect_rect_array_defids(st)
                if expr.final_expr is not None:
                    collect_rect_array_defids(expr.final_expr)

        collect_rect_array_defids(item.body)
        clause_indices = list(item.indices or [])
        for d in candidates:
            if not _has_recurrence_style_read_of_defid(item.body, d, clause_indices):
                continue
            if _BodyReferencesDefidVisitor(d).references(item.body):
                return d
        for inner in inner_einsteins:
            inner_d = _infer_lowered_einstein_output_defid(inner)
            if inner_d is not None:
                return inner_d
    return None


def _reduction_var_bounded_by_loop_var(
    read_index_expr: Any,
    loop_defid: Any,
    reduction_ranges: Any,
) -> bool:
    """True if read_index_expr is a reduction variable whose range end is loop_var or loop_var-1."""
    if reduction_ranges is None or not read_index_expr or loop_defid is None:
        return False
    read_defid = getattr(read_index_expr, "defid", None)
    if read_defid is None:
        return False
    loop_struct = reduction_ranges.get(read_defid) if isinstance(reduction_ranges, dict) else None
    if loop_struct is None:
        return False
    iterable = loop_struct.iterable
    if not isinstance(iterable, RangeIR):
        return False
    end = iterable.end
    return _expr_is_loop_var_or_minus_one(end, loop_defid)


def _loop_dims_from_clause_indices(clause_indices: List[Any], loops: List[Any]) -> Optional[List[int]]:
    """For each loop index k, return the output-dimension index (position in clause_indices). Literals have no loop."""
    if not clause_indices or not loops:
        return None
    out: List[int] = []
    loop_pos = 0
    for pos, idx in enumerate(clause_indices):
        if isinstance(idx, LiteralIR):
            continue
        if loop_pos >= len(loops):
            return None
        out.append(pos)
        loop_pos += 1
    return out if loop_pos == len(loops) else None


def _slice_list_from_clause_indices(
    clause_indices: List[Any],
    lowered: Any,
    expr_evaluator: Any,
) -> Optional[List[Any]]:
    """Build full-dimension slice list: literal idx -> scalar (int), other indices -> slice from loop range.
    Rule: literal idx / constant value -> scalar, other indices -> vectorize (slice)."""
    if not clause_indices:
        return None
    out: List[Any] = []
    loop_pos = 0
    loops = (lowered.loops or [])
    for idx in clause_indices:
        literal_val = None
        if isinstance(idx, LiteralIR):
            try:
                literal_val = int(idx.value)
            except (TypeError, ValueError):
                pass
        elif getattr(idx, "value", None) is not None:
            try:
                literal_val = int(getattr(idx, "value", None))
            except (TypeError, ValueError):
                pass
        if literal_val is not None:
            out.append(literal_val)
            continue
        if loop_pos < len(loops):
            try:
                start, end = _extract_loop_range(loops[loop_pos], expr_evaluator)
                out.append(slice(int(start), int(end)))
            except (RuntimeError, TypeError, ValueError):
                return None
            loop_pos += 1
        else:
            return None
    return out if loop_pos == len(loops) else None


def _extract_loop_range(loop, evaluator) -> Tuple[int, int]:
    """Return (start, end) for the loop range; both must be concrete int. Raises if missing or dependent."""
    it = loop.iterable
    # IR validation (visit_lowered_einstein_clause) fails compilation if iterable is None; this is a safeguard.
    if it is None:
        raise RuntimeError("loop has no iterable; cannot extract range")
    if isinstance(it, LiteralIR) and isinstance(it.value, range):
        r = it.value
        start = r.start
        stop = r.stop
        try:
            return (int(start), int(stop))
        except (TypeError, ValueError) as e:
            raise RuntimeError("loop range start/stop must be int; got dependent or non-int") from e
    if isinstance(it, RangeIR):
        if isinstance(it.end, LiteralIR):
            try:
                end_lit = int(it.end.value)
                if it.start is None:
                    return (0, end_lit)
                if isinstance(it.start, LiteralIR):
                    return (int(it.start.value), end_lit)
            except (TypeError, ValueError):
                pass
        end_ev = evaluator(it.end)
        if not isinstance(end_ev, (int, np.integer)):
            raise RuntimeError("loop range end must be int; got dependent or non-int")
        start_node = it.start
        if start_node is not None:
            start_ev = evaluator(start_node)
            if not isinstance(start_ev, (int, np.integer)):
                raise RuntimeError("loop range start must be int; got dependent or non-int")
            return (int(start_ev), int(end_ev))
        return (0, int(end_ev))
    raise RuntimeError("loop iterable is not a range or literal range; cannot extract (start, end)")


def _literal_numeric_value(expr: Any) -> Optional[float]:
    if not isinstance(expr, LiteralIR):
        return None
    try:
        return float(expr.value)
    except (TypeError, ValueError):
        return None


def _index_expr_offset(expr: Any, loop_defid: Any) -> Optional[int]:
    if _index_expr_is_loop_var(expr, loop_defid):
        return 0
    if not isinstance(expr, BinaryOpIR):
        return None
    if expr.operator == BinaryOp.ADD:
        if _index_expr_is_loop_var(expr.left, loop_defid):
            val = _literal_numeric_value(expr.right)
            return int(val) if val is not None and float(val).is_integer() else None
        if _index_expr_is_loop_var(expr.right, loop_defid):
            val = _literal_numeric_value(expr.left)
            return int(val) if val is not None and float(val).is_integer() else None
        return None
    if expr.operator == BinaryOp.SUB and _index_expr_is_loop_var(expr.left, loop_defid):
        val = _literal_numeric_value(expr.right)
        return -int(val) if val is not None and float(val).is_integer() else None
    return None


def _flatten_additive_terms(expr: Any, sign: float = 1.0) -> List[Tuple[float, Any]]:
    if isinstance(expr, BinaryOpIR) and expr.operator == BinaryOp.ADD:
        return _flatten_additive_terms(expr.left, sign) + _flatten_additive_terms(expr.right, sign)
    if isinstance(expr, BinaryOpIR) and expr.operator == BinaryOp.SUB:
        return _flatten_additive_terms(expr.left, sign) + _flatten_additive_terms(expr.right, -sign)
    return [(sign, expr)]


def _split_scalar_mul(expr: Any) -> Tuple[float, Any]:
    if isinstance(expr, BinaryOpIR) and expr.operator == BinaryOp.MUL:
        left_num = _literal_numeric_value(expr.left)
        if left_num is not None:
            return left_num, expr.right
        right_num = _literal_numeric_value(expr.right)
        if right_num is not None:
            return right_num, expr.left
    return 1.0, expr


def _match_rect_access_offsets(expr: Any, array_defid: Any, loop_defids: List[Any], offsets: Tuple[int, ...]) -> bool:
    if not isinstance(expr, RectangularAccessIR):
        return False
    if not isinstance(expr.array, IdentifierIR) or expr.array.defid != array_defid:
        return False
    indices = list(expr.indices or [])
    if len(indices) != len(loop_defids) or len(indices) != len(offsets):
        return False
    for idx_expr, loop_defid, expected_offset in zip(indices, loop_defids, offsets):
        if _index_expr_offset(idx_expr, loop_defid) != expected_offset:
            return False
    return True
