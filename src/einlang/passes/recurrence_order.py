"""
Recurrence order pass: mark clauses with same-timestep dependency so they run in timestep-major.

When a clause reads the variable at the same recurrence index (e.g. u[t, 0]) as another clause's
write (e.g. writing u[t, 1]), the backend must run them per timestep so the second sees the first.
This pass sets recurrence_dims_override on such clauses so the backend treats them as recurrence.
"""

from typing import Any, List, Optional

from ..passes.base import BasePass, TyCtxt
from ..ir.nodes import (
    ProgramIR,
    BindingIR,
    LoweredEinsteinIR,
    LoweredEinsteinClauseIR,
    LoweredRecurrenceIR,
)

# Reuse backend helpers for recurrence analysis (no circular import: backend does not import this pass)
from ..backends.numpy_einstein import (
    _recurrence_dims,
    _recurrence_dims_for_hybrid,
    _collect_lhs_read_index_lists,
    _loop_dims_from_clause_indices,
    _index_expr_is_loop_var,
    _index_expr_is_loop_var_or_offset,
    _reduction_var_bounded_by_loop_var,
)
from ..backends.numpy_einstein import _BodyReferencesDefidVisitor


def _body_reads_same_recurrence_index_as_write(
    clause: LoweredEinsteinClauseIR,
    variable_defid: Any,
    recurrence_dims: List[int],
) -> bool:
    """True if the clause body reads the variable at indices that match the write on recurrence_dims (same t)."""
    if not recurrence_dims or variable_defid is None:
        return False
    loops = getattr(clause, "loops", None) or []
    if not loops:
        return False
    loop_dims = _loop_dims_from_clause_indices(
        getattr(clause, "indices", None), loops
    )
    if not loop_dims:
        return False
    read_lists = _collect_lhs_read_index_lists(clause.body, variable_defid)
    if not read_lists:
        return False
    loop_defids = [getattr(lp.variable, "defid", None) for lp in loops]
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
    clause_indices = getattr(clause, "indices", None) or []
    loops = getattr(clause, "loops", None) or []
    rec_dims = _recurrence_dims(clause, variable_defid, clause_indices)
    if len(rec_dims) < 2:
        return None
    loop_dims = _loop_dims_from_clause_indices(clause_indices, loops)
    if not loop_dims:
        return None
    read_lists = _collect_lhs_read_index_lists(clause.body, variable_defid)
    if not read_lists:
        return None
    reduction_ranges = getattr(clause, "reduction_ranges", None) or {}
    loop_defids = [getattr(lp.variable, "defid", None) for lp in loops]

    strictly_backward: List[int] = []
    mixed: List[int] = []
    for k in rec_dims:
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
    items = getattr(lowered, "items", None) or []
    if len(items) < 2 or variable_defid is None:
        return
    recurrence_dims: Optional[List[int]] = None
    for it in items:
        rec = _recurrence_dims(it, variable_defid, getattr(it, "indices", None))
        if rec:
            recurrence_dims = rec
            break
    if not recurrence_dims:
        return
    for it in items:
        if getattr(it, "recurrence_dims_override", None) is not None:
            continue
        rec = _recurrence_dims(it, variable_defid, getattr(it, "indices", None))
        if rec:
            continue
        if not _BodyReferencesDefidVisitor(variable_defid).references(
            getattr(it, "body", None)
        ):
            continue
        if _body_reads_same_recurrence_index_as_write(it, variable_defid, recurrence_dims):
            object.__setattr__(it, "recurrence_dims_override", recurrence_dims)


def _partition_recurrence(
    lowered: LoweredEinsteinIR,
    variable_defid: Any,
) -> tuple:
    """Partition items into (non_recurrence_items, recurrence_items, recurrence_loops_for_outer).
    Mirrors backend logic. recurrence_loops_for_outer is a list of LoopStructure (one per recurrence dim)."""
    items = getattr(lowered, "items", None) or []
    non_recurrence_items: List[Any] = []
    recurrence_items: List[Any] = []
    recurrence_loops_for_outer: Optional[List[Any]] = None
    if len(items) <= 1 or variable_defid is None:
        return (non_recurrence_items, recurrence_items, recurrence_loops_for_outer)
    for it in items:
        clause_indices = getattr(it, "indices", None) or []
        loops_it = getattr(it, "loops", None) or []
        rec_dims = getattr(it, "recurrence_dims_override", None)
        if rec_dims is None:
            rec_dims = _recurrence_dims_for_hybrid(it, variable_defid, clause_indices)
        if not rec_dims:
            rec_dims = _recurrence_dims(it, variable_defid, clause_indices)
        body_refs = _BodyReferencesDefidVisitor(variable_defid).references(getattr(it, "body", None))
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


def _isolate_recurrence(
    binding: BindingIR,
    lowered: LoweredEinsteinIR,
    variable_defid: Any,
) -> bool:
    """If lowered has both non-recurrence and recurrence items, replace binding.expr with LoweredRecurrenceIR. Return True if replaced."""
    non_rec, rec_items, rec_loops = _partition_recurrence(lowered, variable_defid)
    if not non_rec or not rec_items or not rec_loops:
        return False
    shape = getattr(lowered, "shape", None)
    element_type = getattr(lowered, "element_type", None)
    loc = getattr(lowered, "location", None)
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
    # Single recurrence loop for LoweredRecurrenceIR (first recurrence dim, e.g. t).
    recurrence_loop = rec_loops[0]
    new_expr = LoweredRecurrenceIR(
        initial=initial,
        recurrence_loop=recurrence_loop,
        body=body,
        location=loc,
    )
    object.__setattr__(binding, "expr", new_expr)
    return True


class RecurrenceOrderPass(BasePass):
    """Pass that sets recurrence_dims_override on clauses with same-timestep dependency."""

    def run(self, ir: ProgramIR, tcx: TyCtxt) -> ProgramIR:
        for stmt in ir.statements:
            if isinstance(stmt, BindingIR):
                value = getattr(stmt, "value", None)
                if value is not None and isinstance(value, LoweredEinsteinIR):
                    variable_defid = getattr(stmt, "defid", None)
                    if variable_defid is not None:
                        _annotate_recurrence_override(value, variable_defid)
                        # Isolate recurrence loop out of the Einstein: replace with LoweredRecurrenceIR when we have both initial and recurrence clauses.
                        _isolate_recurrence(stmt, value, variable_defid)
        return ir
