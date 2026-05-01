"""
Lowered Where Clause Execution Patterns
==========================================================

Pure execution patterns for LoweredIteration structures.
These functions provide the computational logic for executing lowered iteration constructs.

No direct AST/IR dependencies - works with the lowered structures as data.

Lowered execution model.
"""

import os
from typing import Dict, List, Callable, Any, Iterator, Optional, Tuple, Sequence

import numpy as np
from ...shared.defid import DefId
from ...shared.types import ReductionOp


def _try_vectorized_reduction(
    reduction_op: ReductionOp,
    reduction_loops: List[Any],
    body_evaluator: Callable,
    expr_evaluator: Callable,
    parallel_shape: Optional[Tuple[int, ...]] = None,
    initial_context: Optional[Dict[Any, Any]] = None,
) -> Tuple[bool, Any]:
    """
    Vectorized reduction: single rule parallel_shape + reduction_shape.
    Clause loop -> parallel; reduction loop -> reduction. Broadcast reduction
    vars to parallel_shape + red_shape, evaluate body once, reduce over last n axes.
    When parallel_shape is None (standalone reduction), infer it by evaluating
    body with scalar reduction indices. Falls back to False, None on failure.
    """
    if reduction_op not in (
        ReductionOp.SUM,
        ReductionOp.MAX,
        ReductionOp.MIN,
        ReductionOp.PROD,
        ReductionOp.ARGMAX,
        ReductionOp.ARGMIN,
    ):
        return False, None
    if reduction_op in (ReductionOp.ARGMAX, ReductionOp.ARGMIN) and len(reduction_loops) != 1:
        return False, None
    try:
        arrs: List[np.ndarray] = []
        defids: List[Any] = []
        for loop in reduction_loops:
            var_defid = loop.variable.defid  # Compiler guarantees set (IRValidationPass)
            iterable = expr_evaluator(loop.iterable)
            if isinstance(iterable, range):
                step = iterable.step if iterable.step is not None else 1
                arr = np.arange(iterable.start, iterable.stop, step, dtype=np.intp)
            else:
                arr = np.array(list(iterable), dtype=np.intp)
            if arr.size == 0:
                return True, (0 if reduction_op == ReductionOp.SUM else None)
            arrs.append(arr)
            defids.append(var_defid)

        if not arrs:
            return False, None

        n = len(arrs)
        expected_shape = tuple(arr.size for arr in arrs)
        if parallel_shape is None:
            spot_ctx: Dict[Any, Any] = {}
            for defid, arr in zip(defids, arrs):
                spot_ctx[defid] = int(arr.flat[0])
            spot_val = body_evaluator(spot_ctx)
            if isinstance(spot_val, np.ndarray):
                parallel_shape = tuple(spot_val.shape)
            else:
                parallel_shape = ()

        ctx: Dict[Any, Any] = {}
        # Broadcast any initial context (typically parallel indices like b) to full shape so
        # advanced indexing in the reduction body sees compatible index array shapes.
        if parallel_shape and initial_context:
            full_shape = tuple(parallel_shape) + expected_shape
            n_red = len(expected_shape)
            for defid, val in initial_context.items():
                if defid is None or defid in defids:
                    continue
                try:
                    v = np.asarray(val, dtype=np.intp) if isinstance(val, (list, tuple, range, np.ndarray)) else val
                    if isinstance(v, np.ndarray):
                        if v.ndim == 1 and len(parallel_shape) == 1 and v.shape[0] == parallel_shape[0]:
                            v2 = v.reshape((parallel_shape[0],) + (1,) * n_red)
                            ctx[defid] = np.broadcast_to(v2, full_shape)
                            continue
                        if v.shape == parallel_shape:
                            v2 = v.reshape(tuple(parallel_shape) + (1,) * n_red)
                            ctx[defid] = np.broadcast_to(v2, full_shape)
                            continue
                        ctx[defid] = np.broadcast_to(v, full_shape)
                    else:
                        ctx[defid] = np.broadcast_to(np.asarray(v, dtype=np.intp), full_shape)
                except Exception:
                    pass
        for i, (defid, arr) in enumerate(zip(defids, arrs)):
            if n == 1:
                red_shape = (arr.size,)
            else:
                red_shape = [1] * n
                red_shape[i] = arr.size
                red_shape = tuple(red_shape)
            red_arr = arr.reshape(red_shape)
            if parallel_shape:
                ctx[defid] = np.broadcast_to(
                    red_arr, tuple(parallel_shape) + tuple(red_shape)
                )
            else:
                ctx[defid] = red_arr

        result = body_evaluator(ctx)

        if not isinstance(result, np.ndarray):
            # Nested autodiff-expanded reductions can look scalar under a spot evaluation
            # even when the true body varies with the reduction indices. Fall back to the
            # scalar loop in that case instead of analytically multiplying by extent.
            return False, None

        if parallel_shape:
            reduction_axes = tuple(range(-n, 0))
            if result.shape != parallel_shape + expected_shape:
                if (result.ndim >= len(parallel_shape) + n
                        and result.shape[-n:] == expected_shape
                        and np.prod(result.shape[:-n]) == np.prod(parallel_shape)):
                    try:
                        if reduction_op == ReductionOp.SUM:
                            reduced = result.sum(axis=reduction_axes)
                        elif reduction_op == ReductionOp.MAX:
                            reduced = result.max(axis=reduction_axes)
                        elif reduction_op == ReductionOp.MIN:
                            reduced = result.min(axis=reduction_axes)
                        elif reduction_op == ReductionOp.PROD:
                            reduced = result.prod(axis=reduction_axes)
                        else:
                            reduced = None
                        if reduced is not None and reduced.size == np.prod(parallel_shape):
                            reduced = reduced.reshape(parallel_shape)
                            return True, reduced
                    except (ValueError, AttributeError):
                        pass
                return False, None
            if reduction_op == ReductionOp.SUM:
                reduced = result.sum(axis=reduction_axes)
            elif reduction_op == ReductionOp.MAX:
                reduced = result.max(axis=reduction_axes)
            elif reduction_op == ReductionOp.MIN:
                reduced = result.min(axis=reduction_axes)
            elif reduction_op == ReductionOp.PROD:
                reduced = result.prod(axis=reduction_axes)
            elif reduction_op in (ReductionOp.ARGMAX, ReductionOp.ARGMIN):
                if n != 1:
                    return False, None
                if reduction_op == ReductionOp.ARGMAX:
                    idx_pos = np.argmax(result, axis=reduction_axes[0])
                else:
                    idx_pos = np.argmin(result, axis=reduction_axes[0])
                reduced = arrs[0][idx_pos].astype(np.int32, copy=False)
            else:
                return False, None
            return True, reduced
        else:
            if result.shape != expected_shape:
                return False, None
            if reduction_op == ReductionOp.SUM:
                return True, result.sum()
            elif reduction_op == ReductionOp.MAX:
                return True, result.max()
            elif reduction_op == ReductionOp.MIN:
                return True, result.min()
            elif reduction_op == ReductionOp.PROD:
                return True, result.prod()
            elif reduction_op in (ReductionOp.ARGMAX, ReductionOp.ARGMIN):
                if n != 1:
                    return False, None
                idx_pos = int(np.argmax(result) if reduction_op == ReductionOp.ARGMAX else np.argmin(result))
                return True, int(arrs[0][idx_pos])
    except Exception:
        pass
    return False, None


def execute_select_at_argmax_vectorized(
    primal_body_ev: Callable,
    diff_body_ev: Callable,
    reduction_loops: List[Any],
    expr_evaluator: Callable,
    parallel_shape: Optional[Tuple[int, ...]] = None,
    initial_context: Optional[Sequence[Tuple[Any, Any]]] = None,
    use_argmin: bool = False,
    precomputed_idx_flat: Optional[Any] = None,
) -> Tuple[bool, Any]:
    """
    Vectorized select-at-argmax: evaluate primal and diff bodies over the reduction
    window, then return the diff value at the winning reduction index for each
    parallel position. Result shape is ``parallel_shape`` when parallel dims are
    present, otherwise a scalar.
    initial_context: optional list of (defid, array) in parallel-dim order; each array has shape
    (parallel_shape[i],) to set parallel (batch) indices when body uses them.
    """
    try:
        def _selected_diff_result(
            idx_flat: Any,
            red_shape: Tuple[int, ...],
            loop_values: List[np.ndarray],
            loop_defids: List[Any],
            current_parallel_shape: Tuple[int, ...],
        ) -> Tuple[bool, Any]:
            try:
                if current_parallel_shape:
                    if initial_context:
                        result = None
                        for outer_idx in np.ndindex(current_parallel_shape):
                            red_multi = np.unravel_index(idx_flat[outer_idx], red_shape)
                            chosen_ctx = {
                                did: int(loop_values[axis][red_multi[axis]])
                                for axis, did in enumerate(loop_defids)
                            }
                            for axis, (did, val) in enumerate(initial_context):
                                arr = np.asarray(val)
                                if arr.size == 0:
                                    continue
                                pos = outer_idx[axis] if axis < len(outer_idx) else 0
                                chosen_ctx[did] = int(arr.reshape(-1)[pos])
                            chosen = diff_body_ev(chosen_ctx)
                            chosen_arr = np.asarray(chosen)
                            if result is None:
                                result_shape = current_parallel_shape + tuple(chosen_arr.shape)
                                result = np.empty(result_shape, dtype=chosen_arr.dtype)
                            result[outer_idx] = chosen_arr
                        if result is not None:
                            return True, result
                    red_multi = np.unravel_index(idx_flat, red_shape)
                    chosen_ctx = {
                        did: np.asarray(vals[red_multi[axis]], dtype=np.intp)
                        for axis, (did, vals) in enumerate(zip(loop_defids, loop_values))
                    }
                    chosen = diff_body_ev(chosen_ctx)
                    chosen_arr = np.asarray(chosen)
                    if chosen_arr.shape[: len(current_parallel_shape)] == current_parallel_shape:
                        return True, chosen
                    if chosen_arr.shape == ():
                        return True, np.broadcast_to(chosen_arr, current_parallel_shape)
                    return False, None
                red_multi = np.unravel_index(int(idx_flat), red_shape)
                chosen_ctx = {
                    did: int(loop_values[axis][red_multi[axis]])
                    for axis, did in enumerate(loop_defids)
                }
                return True, diff_body_ev(chosen_ctx)
            except Exception:
                return False, None

        arrs: List[np.ndarray] = []
        defids: List[Any] = []
        for loop in reduction_loops:
            var_defid = loop.variable.defid
            iterable = expr_evaluator(loop.iterable)
            if isinstance(iterable, range):
                step = iterable.step if iterable.step is not None else 1
                arr = np.arange(iterable.start, iterable.stop, step, dtype=np.intp)
            else:
                arr = np.array(list(iterable), dtype=np.intp)
            if arr.size == 0:
                return False, None
            arrs.append(arr)
            defids.append(var_defid)
        if not arrs:
            return False, None
        n = len(arrs)
        red_shape_tuple = tuple(int(arr.size) for arr in arrs)
        if parallel_shape is None:
            spot_ctx: Dict[Any, Any] = {}
            for defid, arr in (initial_context or []):
                spot_ctx[defid] = int(np.asarray(arr).flat[0]) if np.asarray(arr).size else 0
            for defid, arr in zip(defids, arrs):
                spot_ctx[defid] = int(arr.flat[0])
            spot_val = primal_body_ev(spot_ctx)
            if isinstance(spot_val, np.ndarray):
                parallel_shape = tuple(spot_val.shape)
            else:
                parallel_shape = ()
        full_shape = tuple(parallel_shape) + red_shape_tuple
        ctx: Dict[Any, Any] = {}
        if initial_context and parallel_shape:
            k_par = len(parallel_shape)
            for i, (defid, val) in enumerate(initial_context):
                v = np.asarray(val, dtype=np.intp)
                v = v.reshape(
                    (1,) * i + (v.size,) + (1,) * (k_par - 1 - i) + (1,) * n
                )
                ctx[defid] = np.broadcast_to(v, full_shape)
        for i, (defid, arr) in enumerate(zip(defids, arrs)):
            if n == 1:
                red_shape = (arr.size,)
            else:
                red_shape = [1] * n
                red_shape[i] = arr.size
                red_shape = tuple(red_shape)
            red_arr = arr.reshape(red_shape)
            if parallel_shape:
                ctx[defid] = np.broadcast_to(
                    red_arr, full_shape
                )
            else:
                ctx[defid] = red_arr
        primal_result = None
        idx_flat = precomputed_idx_flat
        if idx_flat is None:
            primal_result = primal_body_ev(ctx)
            if not isinstance(primal_result, np.ndarray):
                return False, None
            if parallel_shape:
                primal_flat = primal_result.reshape(parallel_shape + (-1,))
                idx_flat = np.argmin(primal_flat, axis=-1) if use_argmin else np.argmax(primal_flat, axis=-1)
            else:
                idx_flat = int(np.argmin(primal_result) if use_argmin else np.argmax(primal_result))
        if parallel_shape:
            ok_selected, selected = _selected_diff_result(
                idx_flat,
                red_shape_tuple,
                arrs,
                defids,
                tuple(parallel_shape),
            )
            if ok_selected:
                return True, selected
        else:
            ok_selected, selected = _selected_diff_result(
                idx_flat,
                red_shape_tuple,
                arrs,
                defids,
                (),
            )
            if ok_selected:
                return True, selected

        if primal_result is None:
            primal_result = primal_body_ev(ctx)
            if not isinstance(primal_result, np.ndarray):
                return False, None
        diff_result = diff_body_ev(ctx)
        if not isinstance(diff_result, np.ndarray):
            diff_result = np.broadcast_to(
                np.asarray(diff_result),
                primal_result.shape,
            )
        primal_shape = tuple(primal_result.shape)
        diff_shape = tuple(diff_result.shape)
        if diff_shape == primal_shape:
            tail_shape: Tuple[int, ...] = ()
        elif diff_shape[: len(primal_shape)] == primal_shape:
            tail_shape = diff_shape[len(primal_shape) :]
        else:
            return False, None
        if parallel_shape:
            primal_flat = primal_result.reshape(parallel_shape + (-1,))
            diff_flat = diff_result.reshape(parallel_shape + (-1,) + tail_shape)
            idx_flat = np.argmin(primal_flat, axis=-1) if use_argmin else np.argmax(primal_flat, axis=-1)
            gather_idx = np.expand_dims(idx_flat, axis=-1)
            for _ in tail_shape:
                gather_idx = np.expand_dims(gather_idx, axis=-1)
            out = np.take_along_axis(
                diff_flat,
                gather_idx,
                axis=len(parallel_shape),
            ).squeeze(axis=len(parallel_shape))
            return True, out
        if tail_shape:
            return True, diff_result.reshape((-1,) + tail_shape)[idx_flat]
        scalar = diff_result.reshape(-1)[idx_flat]
        return True, scalar.item() if hasattr(scalar, "item") else scalar
    except Exception:
        pass
    return False, None


def execute_lowered_loops(
    loops: List[Any],  # List[LoopStructure]
    context: Dict[DefId, Any],
    evaluator: Callable[[Any], Any]
) -> Iterator[Dict[DefId, Any]]:
    """
    Execute nested loops. Context is keyed by DefId (loop variable identity).
    """
    if not loops:
        yield {}
        return

    def _execute_loop_level(level: int, current_context: Dict[DefId, Any]):
        if level >= len(loops):
            yield dict(current_context)
            return
        loop = loops[level]
        var_defid = loop.variable.defid  # Compiler guarantees set (IRValidationPass)
        iterable = evaluator(loop.iterable)
        if iterable is None:
            return
        for value in iterable:
            current_context[var_defid] = value
            yield from _execute_loop_level(level + 1, current_context)

    yield from _execute_loop_level(0, dict(context))


def execute_lowered_bindings(
    bindings: List[Any],  # List[BindingIR]
    context: Dict[DefId, Any],
    evaluator: Callable[[Any], Any]
) -> Dict[DefId, Any]:
    """Execute local bindings. Context keyed by DefId."""
    result_context = dict(context)
    for binding in bindings:
        defid = binding.defid  # Compiler guarantees set (IRValidationPass)
        result_context[defid] = evaluator(binding.expr)
    return result_context


def check_lowered_guards(
    guards: List[Any],
    context: Dict[DefId, Any],
    evaluator: Callable[[Any], bool]
) -> bool:
    for guard in guards:
        if not evaluator(guard.condition):
            return False
    return True


def execute_full_lowered_iteration(
    lowered_iteration: Any,  # LoweredIteration
    body_evaluator: Callable[[Dict[DefId, Any]], Any],
    expr_evaluator: Callable[[Any], Any],
    initial_context: Optional[Dict[DefId, Any]] = None
) -> List[Any]:
    """Execute complete lowered iteration. Context keyed by DefId."""
    if initial_context is None:
        initial_context = {}
    results = []
    for loop_context in execute_lowered_loops(
        lowered_iteration.loops,
        initial_context,
        expr_evaluator
    ):
        full_context = {**initial_context, **loop_context}
        if lowered_iteration.bindings:
            full_context = execute_lowered_bindings(
                lowered_iteration.bindings,
                full_context,
                expr_evaluator
            )
        if lowered_iteration.guards:
            if not check_lowered_guards(lowered_iteration.guards, full_context, expr_evaluator):
                continue
        results.append(body_evaluator(full_context))
    return results


def execute_reduction_with_loops(
    reduction_op: ReductionOp,
    reduction_ranges: Dict[Any, Any],  # Dict[DefId, LoopStructure] for compatibility / lookups
    body_evaluator: Callable[[Dict[DefId, Any]], Any],
    expr_evaluator: Callable[[Any], Any],
    guard_evaluator: Optional[Callable[[Dict[DefId, Any]], bool]] = None,
    initial_context: Optional[Dict[DefId, Any]] = None,
    profile_callback: Optional[Callable[[str], None]] = None,
    parallel_shape: Optional[Tuple[int, ...]] = None,
    vector_parallel_context: Optional[Dict[Any, Any]] = None,
    reduction_loops_ordered: Optional[Sequence[Any]] = None,
    *,
    allow_speculative_vectorized_reduction: bool = True,
) -> Any:
    """
    Execute reduction operation using nested loops with accumulators.

    Args:
        reduction_op: Operation name ('sum', 'prod', 'min', 'max', 'all', 'any')
        reduction_ranges: Reduction variable ranges keyed by DefId (may collapse duplicate defids).
        reduction_loops_ordered: When set, nested loop order for execution (must match lowered .loops).
        body_evaluator: Function that evaluates body given index bindings
        expr_evaluator: Function to evaluate sub-expressions
        guard_evaluator: Optional function that evaluates guard conditions
        initial_context: Initial variable bindings

    Returns:
        Reduced value
    
    Examples:
        >>> # Execute: sum[k in 0..5](k * 2)
        >>> def body(bindings): return bindings['k'] * 2
        >>> ranges = {'k': LoopStructure('k', LiteralIR(range(5)))}
        >>> execute_reduction_with_loops(ReductionOp.SUM, ranges, body, evaluator)
        20
    """
    if initial_context is None:
        initial_context = {}
    if vector_parallel_context is None:
        vector_parallel_context = {}
    merged_for_vector = dict(initial_context)
    merged_for_vector.update(vector_parallel_context)

    if reduction_loops_ordered is not None:
        reduction_loops = list(reduction_loops_ordered)
    else:
        reduction_loops = list(reduction_ranges.values())

    # Standalone speculative vectorization is only safe when the reduction has no
    # surrounding loop context. With outer loop bindings in initial_context, spot
    # evaluation can incorrectly collapse a nested reduction body to a scalar and
    # over-count by the reduction extent.
    can_try_vectorized = (
        allow_speculative_vectorized_reduction
        and guard_evaluator is None
        and (
            parallel_shape is not None
            or (
                not initial_context
                and not vector_parallel_context
            )
        )
    )
    if can_try_vectorized:
        ok, vec_result = _try_vectorized_reduction(
            reduction_op,
            reduction_loops,
            body_evaluator,
            expr_evaluator,
            parallel_shape=parallel_shape,
            initial_context=merged_for_vector,
        )
        if ok and vec_result is not None:
            if profile_callback is not None:
                profile_callback("vectorized")
            return vec_result
    if profile_callback is not None:
        profile_callback("scalar")

    # Initialize accumulator based on operation
    if reduction_op == ReductionOp.SUM:
        accumulator = 0
        def combine(acc, val): return acc + val
    elif reduction_op == ReductionOp.PROD:
        accumulator = 1
        def combine(acc, val): return acc * val
    elif reduction_op == ReductionOp.MIN:
        accumulator = None
        def combine(acc, val):
            if acc is None:
                return val
            return min(acc, val)
    elif reduction_op == ReductionOp.MAX:
        accumulator = None
        def combine(acc, val):
            if acc is None:
                return val
            # : use np.maximum for numpy arrays/scalars (handles f32 correctly)
            if isinstance(acc, np.ndarray) or isinstance(val, np.ndarray):
                return np.maximum(acc, val)
            return max(acc, val)
    elif reduction_op == ReductionOp.ALL:
        accumulator = True
        def combine(acc, val):
            v = bool(np.all(val)) if isinstance(val, np.ndarray) else bool(val)
            return acc and v
    elif reduction_op == ReductionOp.ANY:
        accumulator = False
        def combine(acc, val):
            v = bool(np.any(val)) if isinstance(val, np.ndarray) else bool(val)
            return acc or v
    elif reduction_op in (ReductionOp.ARGMAX, ReductionOp.ARGMIN):
        if len(reduction_loops) != 1:
            raise ValueError(f"{reduction_op.value} currently supports exactly one reduction variable")
        accumulator = None
        loop_defid = reduction_loops[0].variable.defid

        def combine(acc, val, ctx=None):
            index_value = ctx.get(loop_defid) if ctx is not None else None
            if isinstance(index_value, np.ndarray):
                index_value = int(index_value.item()) if index_value.size == 1 else index_value
            if acc is None:
                return (val, index_value)
            best_val, best_index = acc
            if reduction_op == ReductionOp.ARGMAX:
                better = val > best_val
            else:
                better = val < best_val
            if isinstance(better, np.ndarray):
                return (np.where(better, val, best_val), np.where(better, index_value, best_index))
            return (val, index_value) if better else (best_val, best_index)
    else:
        raise ValueError(f"Unknown reduction operation: {reduction_op}")
    
    # Execute loops for reduction variables (initial_context carries parallel indices when reduction has guards)
    for reduction_context in execute_lowered_loops(
        reduction_loops,
        initial_context,
        expr_evaluator
    ):
        full_context = {**initial_context, **reduction_context}
        # Check guards if provided (guard may reference parallel indices from initial_context)
        if guard_evaluator and not guard_evaluator(full_context):
            continue
        
        # Evaluate body
        value = body_evaluator(full_context)
        
        # Skip None values (from where expressions that filtered this item)
        if value is None:
            continue
        
        if reduction_op in (ReductionOp.ALL, ReductionOp.ANY):
            if isinstance(value, np.ndarray):
                value = bool(value.item()) if value.size == 1 else (bool(np.all(value)) if reduction_op == ReductionOp.ALL else bool(np.any(value)))
            else:
                value = bool(value)
        
        # Accumulate
        if reduction_op in (ReductionOp.ARGMAX, ReductionOp.ARGMIN):
            accumulator = combine(accumulator, value, full_context)
        else:
            accumulator = combine(accumulator, value)
    
    # Handle empty reduction case for max/min
    if accumulator is None and reduction_op in (ReductionOp.MIN, ReductionOp.MAX):
        raise ValueError(f"{reduction_op}() arg is an empty sequence")
    if accumulator is None and reduction_op in (ReductionOp.ARGMAX, ReductionOp.ARGMIN):
        raise ValueError(f"{reduction_op.value}() arg is an empty sequence")
    if reduction_op in (ReductionOp.ARGMAX, ReductionOp.ARGMIN):
        _best_val, best_index = accumulator
        if isinstance(best_index, np.ndarray):
            return best_index.astype(np.int32, copy=False)
        return int(best_index)
    
    return accumulator
