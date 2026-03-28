"""
Recurrence order pass: mark clauses with same-timestep dependency so they run in timestep-major.

When a clause reads the variable at the same recurrence index (e.g. u[t, 0]) as another clause's
write (e.g. writing u[t, 1]), the backend must run them per timestep so the second sees the first.
This pass sets recurrence_dims_override on such clauses so the backend treats them as recurrence.

It also records conservative storage metadata for isolated recurrence nodes so a later backend
optimization can switch to a circular buffer when only bounded history and bounded tail reads are
required.
"""

from typing import Any, Iterator, List, Optional, Sequence, Tuple

from ..passes.base import BasePass, TyCtxt
from ..ir.nodes import (
    BinaryOpIR,
    BlockExpressionIR,
    BindingIR,
    FunctionValueIR,
    IdentifierIR,
    IfExpressionIR,
    IRNode,
    LambdaIR,
    LiteralIR,
    MatchExpressionIR,
    LoweredEinsteinClauseIR,
    LoweredEinsteinIR,
    LoweredRecurrenceIR,
    ProgramIR,
    RectangularAccessIR,
)
from ..shared.types import BinaryOp

# Reuse backend helpers for recurrence analysis (no circular import: backend does not import this pass)
from ..backends.numpy_einstein import (
    _BodyReferencesDefidVisitor,
    _collect_lhs_read_index_lists,
    _index_expr_is_loop_var,
    _index_expr_is_loop_var_or_offset,
    _loop_dims_from_clause_indices,
    _recurrence_dims,
    _recurrence_dims_for_hybrid,
    _reduction_var_bounded_by_loop_var,
)


def _body_reads_same_recurrence_index_as_write(
    clause: LoweredEinsteinClauseIR,
    variable_defid: Any,
    recurrence_dims: List[int],
) -> bool:
    """True if the clause body reads the variable at indices that match the write on recurrence_dims (same t)."""
    if not recurrence_dims or variable_defid is None:
        return False
    loops = clause.loops or []
    if not loops:
        return False
    loop_dims = _loop_dims_from_clause_indices(clause.indices, loops)
    if not loop_dims:
        return False
    read_lists = _collect_lhs_read_index_lists(clause.body, variable_defid)
    if not read_lists:
        return False
    loop_defids = [lp.variable.defid for lp in loops]
    for read in read_lists:
        if all(
            k < len(loop_dims)
            and loop_dims[k] < len(read)
            and _index_expr_is_loop_var(read[loop_dims[k]], loop_defids[k])
            for k in recurrence_dims
        ):
            return True
    return False


def _infer_recurrence_order_override(
    clause: LoweredEinsteinClauseIR,
    variable_defid: Any,
) -> Optional[List[int]]:
    """Infer recurrence dim order when two index vars appear on the same dim (e.g. Cholesky).
    If exactly some dims are 'strictly backward' (every read is loop_var/offset or reduction bounded by that loop var),
    put those first so column-major order is used. Returns override list or None."""
    clause_indices = clause.indices or []
    loops = clause.loops or []
    loop_dims = _loop_dims_from_clause_indices(clause_indices, loops)
    if not loop_dims:
        return None
    read_lists = _collect_lhs_read_index_lists(clause.body, variable_defid)
    if not read_lists:
        return None
    reduction_ranges = clause.reduction_ranges or {}
    loop_defids = [lp.variable.defid for lp in loops]
    rec_dims = set(_recurrence_dims(clause, variable_defid, clause_indices))
    bounded_dims = set()
    for k in range(len(loops)):
        out_d = loop_dims[k] if k < len(loop_dims) else k
        for idx_list in read_lists:
            if out_d >= len(idx_list):
                continue
            expr = idx_list[out_d]
            if _reduction_var_bounded_by_loop_var(expr, loop_defids[k], reduction_ranges):
                bounded_dims.add(k)
                break
    candidate_dims = sorted(rec_dims | bounded_dims)
    if len(candidate_dims) < 2:
        return None

    strictly_backward: List[int] = []
    mixed: List[int] = []
    for k in candidate_dims:
        out_d = loop_dims[k] if k < len(loop_dims) else k
        all_backward = True
        for idx_list in read_lists:
            if out_d >= len(idx_list):
                continue
            expr = idx_list[out_d]
            if _index_expr_is_loop_var_or_offset(expr, loop_defids[k]):
                continue
            if _reduction_var_bounded_by_loop_var(expr, loop_defids[k], reduction_ranges):
                continue
            all_backward = False
            break
        if all_backward:
            strictly_backward.append(k)
        else:
            mixed.append(k)

    if not strictly_backward:
        return None
    return strictly_backward + mixed


def _annotate_recurrence_override(
    lowered: LoweredEinsteinIR,
    variable_defid: Any,
) -> None:
    """Set recurrence_dims_override on clauses that have same-t dependency so backend runs them in timestep-major."""
    items = lowered.items or []
    if variable_defid is None:
        return
    for it in items:
        if it.recurrence_dims_override is not None:
            continue
        rec_order = _infer_recurrence_order_override(it, variable_defid)
        if rec_order:
            object.__setattr__(it, "recurrence_dims_override", rec_order)
    if len(items) < 2:
        return
    recurrence_dims: Optional[List[int]] = None
    for it in items:
        rec = it.recurrence_dims_override
        if not rec:
            rec = _recurrence_dims(it, variable_defid, it.indices)
        if rec:
            recurrence_dims = rec
            break
    if not recurrence_dims:
        return
    for it in items:
        if it.recurrence_dims_override is not None:
            continue
        rec = _recurrence_dims(it, variable_defid, it.indices)
        if rec:
            continue
        if not _BodyReferencesDefidVisitor(variable_defid).references(it.body):
            continue
        if _body_reads_same_recurrence_index_as_write(it, variable_defid, recurrence_dims):
            object.__setattr__(it, "recurrence_dims_override", recurrence_dims)


def _partition_recurrence(
    lowered: LoweredEinsteinIR,
    variable_defid: Any,
) -> tuple:
    """Partition items into (non_recurrence_items, recurrence_items, recurrence_loops_for_outer).
    Mirrors backend logic. recurrence_loops_for_outer is a list of LoopStructure (one per recurrence dim)."""
    items = lowered.items or []
    non_recurrence_items: List[Any] = []
    recurrence_items: List[Any] = []
    recurrence_loops_for_outer: Optional[List[Any]] = None
    if len(items) <= 1 or variable_defid is None:
        return (non_recurrence_items, recurrence_items, recurrence_loops_for_outer)
    for it in items:
        clause_indices = it.indices or []
        loops_it = it.loops or []
        rec_dims = it.recurrence_dims_override
        if rec_dims is None:
            rec_dims = _recurrence_dims_for_hybrid(it, variable_defid, clause_indices)
        if not rec_dims:
            rec_dims = _recurrence_dims(it, variable_defid, clause_indices)
        body_refs = _BodyReferencesDefidVisitor(variable_defid).references(it.body)
        has_rec = bool(
            rec_dims
            and body_refs
            and 0 < len(rec_dims) <= len(loops_it)
        )
        if has_rec:
            recurrence_items.append(it)
            if recurrence_loops_for_outer is None:
                recurrence_loops_for_outer = [it.loops[d] for d in rec_dims]
        else:
            non_recurrence_items.append(it)
    return (non_recurrence_items, recurrence_items, recurrence_loops_for_outer)


def _iter_ir_children(node: Any) -> Iterator[IRNode]:
    """Yield IR-node children from a slot-based IR object."""
    if node is None:
        return
    slots: List[str] = []
    for cls in type(node).__mro__:
        cls_slots = getattr(cls, "__slots__", ())
        if isinstance(cls_slots, str):
            slots.append(cls_slots)
        else:
            slots.extend(cls_slots)
    seen = set()
    for slot in slots:
        if slot in seen:
            continue
        seen.add(slot)
        value = getattr(node, slot, None)
        if value is None:
            continue
        if hasattr(value, "accept"):
            yield value
        elif isinstance(value, (list, tuple)):
            for item in value:
                if item is not None and hasattr(item, "accept"):
                    yield item
        elif isinstance(value, dict):
            for item in value.keys():
                if item is not None and hasattr(item, "accept"):
                    yield item
            for item in value.values():
                if item is not None and hasattr(item, "accept"):
                    yield item


def _int_literal_value(expr: Any) -> Optional[int]:
    if isinstance(expr, LiteralIR):
        try:
            return int(expr.value)
        except (TypeError, ValueError):
            return None
    if isinstance(expr, int):
        return int(expr)
    return None


def _extent_from_shape_dim(shape_dim: Any) -> Optional[int]:
    if shape_dim is None:
        return None
    if isinstance(shape_dim, int):
        return int(shape_dim)
    if isinstance(shape_dim, LiteralIR):
        try:
            return int(shape_dim.value)
        except (TypeError, ValueError):
            return None
    return None


def _loop_dim_and_output_dim_for_recurrence_item(
    item: Any,
    variable_defid: Any,
) -> Tuple[Optional[int], Optional[int]]:
    clause_indices = item.indices or []
    loops = item.loops or []
    rec_dims = item.recurrence_dims_override
    if rec_dims is None:
        rec_dims = _recurrence_dims_for_hybrid(item, variable_defid, clause_indices)
    if not rec_dims:
        rec_dims = _recurrence_dims(item, variable_defid, clause_indices)
    if not rec_dims:
        return (None, None)
    loop_dim = rec_dims[0]
    loop_dims = _loop_dims_from_clause_indices(clause_indices, loops)
    if loop_dim >= len(loop_dims):
        return (loop_dim, loop_dim)
    return (loop_dim, loop_dims[loop_dim])


def _backward_offset_from_index_expr(expr: Any, loop_defid: Any) -> Optional[int]:
    if expr is None or loop_defid is None:
        return None
    if _index_expr_is_loop_var(expr, loop_defid):
        return 0
    if isinstance(expr, BinaryOpIR) and expr.operator == BinaryOp.SUB:
        if not _index_expr_is_loop_var(expr.left, loop_defid):
            return None
        rhs = _int_literal_value(expr.right)
        if rhs is None or rhs < 0:
            return None
        return rhs
    return None


def _infer_history_lookback_steps(
    body: LoweredEinsteinIR,
    variable_defid: Any,
    recurrence_output_dim: Optional[int],
) -> Optional[int]:
    if recurrence_output_dim is None:
        return None
    lookback = 0
    for item in body.items or []:
        loops = item.loops or []
        if not loops:
            continue
        loop_dims = _loop_dims_from_clause_indices(item.indices or [], loops)
        if recurrence_output_dim not in loop_dims:
            continue
        loop_dim = loop_dims.index(recurrence_output_dim)
        loop_defid = loops[loop_dim].variable.defid
        if loop_defid is None:
            return None
        read_lists = _collect_lhs_read_index_lists(item.body, variable_defid)
        for idx_list in read_lists:
            if recurrence_output_dim >= len(idx_list):
                continue
            offset = _backward_offset_from_index_expr(idx_list[recurrence_output_dim], loop_defid)
            if offset is None:
                return None
            if offset > lookback:
                lookback = offset
    return lookback


def _analyze_tail_use_in_node(
    node: Any,
    *,
    target_defid: Any,
    recurrence_output_dim: Optional[int],
    recurrence_extent: Optional[int],
) -> Tuple[Optional[int], bool]:
    """Return (tail_steps, requires_full_output) for references under one IR node."""
    if node is None:
        return (0, False)
    if isinstance(node, RectangularAccessIR):
        arr = node.array
        if isinstance(arr, IdentifierIR) and arr.defid == target_defid:
            if recurrence_output_dim is None or recurrence_extent is None:
                return (None, True)
            indices = list(node.indices or [])
            if recurrence_output_dim >= len(indices):
                return (None, True)
            idx_val = _int_literal_value(indices[recurrence_output_dim])
            if idx_val is None:
                return (None, True)
            if idx_val < 0 or idx_val >= recurrence_extent:
                return (None, True)
            best_tail = recurrence_extent - idx_val
            for idx in indices:
                child_tail, child_full = _analyze_tail_use_in_node(
                    idx,
                    target_defid=target_defid,
                    recurrence_output_dim=recurrence_output_dim,
                    recurrence_extent=recurrence_extent,
                )
                if child_full:
                    return (None, True)
                if child_tail is not None and child_tail > best_tail:
                    best_tail = child_tail
            return (best_tail, False)
    if isinstance(node, IdentifierIR) and node.defid == target_defid:
        return (None, True)
    best_tail = 0
    for child in _iter_ir_children(node):
        child_tail, child_full = _analyze_tail_use_in_node(
            child,
            target_defid=target_defid,
            recurrence_output_dim=recurrence_output_dim,
            recurrence_extent=recurrence_extent,
        )
        if child_full:
            return (None, True)
        if child_tail is not None and child_tail > best_tail:
            best_tail = child_tail
    return (best_tail, False)


def _infer_downstream_tail_steps(
    later_statements: Sequence[Any],
    *,
    target_defid: Any,
    recurrence_output_dim: Optional[int],
    recurrence_extent: Optional[int],
) -> Tuple[Optional[int], bool]:
    tail_steps = 0
    for stmt in later_statements:
        stmt_tail, requires_full = _analyze_tail_use_in_node(
            stmt,
            target_defid=target_defid,
            recurrence_output_dim=recurrence_output_dim,
            recurrence_extent=recurrence_extent,
        )
        if requires_full:
            return (None, True)
        if stmt_tail is not None and stmt_tail > tail_steps:
            tail_steps = stmt_tail
    return (tail_steps, False)


def _build_recurrence_storage_metadata(
    lowered: LoweredEinsteinIR,
    body: LoweredEinsteinIR,
    variable_defid: Any,
    later_statements: Sequence[Any],
) -> dict:
    recurrence_loop_dim: Optional[int] = None
    recurrence_output_dim: Optional[int] = None
    for item in body.items or []:
        recurrence_loop_dim, recurrence_output_dim = _loop_dim_and_output_dim_for_recurrence_item(
            item, variable_defid
        )
        if recurrence_output_dim is not None:
            break

    recurrence_extent = None
    if (
        recurrence_output_dim is not None
        and lowered.shape
        and recurrence_output_dim < len(lowered.shape)
    ):
        recurrence_extent = _extent_from_shape_dim(lowered.shape[recurrence_output_dim])

    history_lookback_steps = _infer_history_lookback_steps(
        body,
        variable_defid,
        recurrence_output_dim,
    )
    downstream_tail_steps, requires_full_output = _infer_downstream_tail_steps(
        later_statements,
        target_defid=variable_defid,
        recurrence_output_dim=recurrence_output_dim,
        recurrence_extent=recurrence_extent,
    )

    if requires_full_output:
        preserve_steps = None
    else:
        history_window = (
            history_lookback_steps + 1
            if history_lookback_steps is not None
            else 1
        )
        preserve_steps = max(history_window, downstream_tail_steps or 0)

    return {
        "recurrence_loop_dim": recurrence_loop_dim,
        "recurrence_output_dim": recurrence_output_dim,
        "history_lookback_steps": history_lookback_steps,
        "downstream_tail_steps": downstream_tail_steps,
        "preserve_steps": preserve_steps,
        "requires_full_output": requires_full_output,
    }


def _isolate_recurrence(
    binding: BindingIR,
    lowered: LoweredEinsteinIR,
    variable_defid: Any,
    later_statements: Sequence[Any],
) -> bool:
    """If lowered has both non-recurrence and recurrence items, replace binding.expr with LoweredRecurrenceIR. Return True if replaced."""
    non_rec, rec_items, rec_loops = _partition_recurrence(lowered, variable_defid)
    if not non_rec or not rec_items or not rec_loops:
        return False
    shape = lowered.shape
    element_type = lowered.element_type
    loc = lowered.location
    initial = LoweredEinsteinIR(
        items=non_rec,
        shape=shape,
        element_type=element_type,
        location=loc,
    )
    body = LoweredEinsteinIR(
        items=rec_items,
        shape=shape,
        element_type=element_type,
        location=loc,
    )
    storage_meta = _build_recurrence_storage_metadata(
        lowered,
        body,
        variable_defid,
        later_statements,
    )
    # Single recurrence loop for LoweredRecurrenceIR (first recurrence dim, e.g. t).
    recurrence_loop = rec_loops[0]
    new_expr = LoweredRecurrenceIR(
        initial=initial,
        recurrence_loop=recurrence_loop,
        body=body,
        recurrence_loop_dim=storage_meta["recurrence_loop_dim"],
        recurrence_output_dim=storage_meta["recurrence_output_dim"],
        history_lookback_steps=storage_meta["history_lookback_steps"],
        downstream_tail_steps=storage_meta["downstream_tail_steps"],
        preserve_steps=storage_meta["preserve_steps"],
        requires_full_output=storage_meta["requires_full_output"],
        location=loc,
    )
    object.__setattr__(binding, "expr", new_expr)
    return True


class RecurrenceOrderPass(BasePass):
    """Pass that sets recurrence_dims_override on clauses with same-timestep dependency."""

    def _process_expr(self, expr: Any, *, allow_isolate: bool) -> None:
        if expr is None:
            return
        if isinstance(expr, FunctionValueIR):
            self._process_expr(expr.body, allow_isolate=False)
            return
        if isinstance(expr, BlockExpressionIR):
            extra_tail = [expr.final_expr] if expr.final_expr is not None else []
            self._process_statement_sequence(
                list(expr.statements or []),
                extra_tail,
                allow_isolate=allow_isolate,
            )
            if expr.final_expr is not None:
                self._process_expr(expr.final_expr, allow_isolate=allow_isolate)
            return
        if isinstance(expr, IfExpressionIR):
            self._process_expr(expr.condition, allow_isolate=allow_isolate)
            self._process_expr(expr.then_expr, allow_isolate=allow_isolate)
            self._process_expr(expr.else_expr, allow_isolate=allow_isolate)
            return
        if isinstance(expr, MatchExpressionIR):
            self._process_expr(expr.scrutinee, allow_isolate=allow_isolate)
            for arm in expr.arms or []:
                self._process_expr(arm.body, allow_isolate=allow_isolate)
            return
        if isinstance(expr, LambdaIR):
            self._process_expr(expr.body, allow_isolate=False)
            return
        for child in _iter_ir_children(expr):
            self._process_expr(child, allow_isolate=allow_isolate)

    def _process_statement_sequence(
        self,
        statements: Sequence[Any],
        extra_tail: Sequence[Any] = (),
        *,
        allow_isolate: bool,
    ) -> None:
        seq = list(statements or [])
        tail = list(extra_tail or [])
        for i, stmt in enumerate(seq):
            if isinstance(stmt, BindingIR):
                value = stmt.expr
                if value is not None and isinstance(value, LoweredEinsteinIR):
                    variable_defid = stmt.defid
                    if variable_defid is not None:
                        _annotate_recurrence_override(value, variable_defid)
                        if allow_isolate:
                            later = seq[i + 1 :] + tail
                            _isolate_recurrence(stmt, value, variable_defid, later)
                            value = stmt.expr
                self._process_expr(value, allow_isolate=allow_isolate)
            else:
                self._process_expr(stmt, allow_isolate=allow_isolate)

    def run(self, ir: ProgramIR, tcx: TyCtxt) -> ProgramIR:
        self._process_statement_sequence(list(ir.statements or []), allow_isolate=True)
        return ir
