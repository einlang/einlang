"""
Lowered Where Clause Execution Patterns
==========================================================

Pure execution patterns for LoweredIteration structures.
These functions provide the computational logic for executing lowered iteration constructs.

No direct AST/IR dependencies - works with the lowered structures as data.

Lowered execution model.
"""

import os
from typing import Dict, List, Callable, Any, Iterator, Optional, Tuple

import numpy as np

from ...shared.defid import DefId


def _try_vectorized_reduction(
    reduction_op: str,
    reduction_loops: List[Any],
    body_evaluator: Callable,
    expr_evaluator: Callable,
    parallel_shape: Optional[Tuple[int, ...]] = None,
) -> Tuple[bool, Any]:
    """
    Vectorized reduction: single rule parallel_shape + reduction_shape.
    Clause loop -> parallel; reduction loop -> reduction. Broadcast reduction
    vars to parallel_shape + red_shape, evaluate body once, reduce over last n axes.
    When parallel_shape is None (standalone reduction), infer it by evaluating
    body with scalar reduction indices. Falls back to False, None on failure.
    """
    if reduction_op not in ('sum', 'max', 'min', 'product', 'prod'):
        return False, None
    try:
        arrs: List[np.ndarray] = []
        defids: List[Any] = []
        for loop in reduction_loops:
            var_defid = getattr(loop.variable, 'defid', None)
            if var_defid is None:
                return False, None
            iterable = expr_evaluator(loop.iterable)
            if isinstance(iterable, range):
                step = iterable.step if iterable.step is not None else 1
                arr = np.arange(iterable.start, iterable.stop, step, dtype=np.intp)
            else:
                arr = np.array(list(iterable), dtype=np.intp)
            if arr.size == 0:
                return True, (0 if reduction_op == 'sum' else None)
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
            return False, None

        if parallel_shape:
            reduction_axes = tuple(range(-n, 0))
            if result.shape != parallel_shape + expected_shape:
                if (result.ndim >= len(parallel_shape) + n
                        and result.shape[-n:] == expected_shape
                        and np.prod(result.shape[:-n]) == np.prod(parallel_shape)):
                    try:
                        if reduction_op == 'sum':
                            reduced = result.sum(axis=reduction_axes)
                        elif reduction_op == 'max':
                            reduced = result.max(axis=reduction_axes)
                        elif reduction_op == 'min':
                            reduced = result.min(axis=reduction_axes)
                        elif reduction_op in ('product', 'prod'):
                            reduced = result.prod(axis=reduction_axes)
                        else:
                            reduced = None
                        if reduced is not None and reduced.size == np.prod(parallel_shape):
                            reduced = reduced.reshape(parallel_shape)
                            return True, reduced
                    except (ValueError, AttributeError):
                        pass
                return False, None
            if reduction_op == 'sum':
                reduced = result.sum(axis=reduction_axes)
            elif reduction_op == 'max':
                reduced = result.max(axis=reduction_axes)
            elif reduction_op == 'min':
                reduced = result.min(axis=reduction_axes)
            elif reduction_op in ('product', 'prod'):
                reduced = result.prod(axis=reduction_axes)
            else:
                return False, None
            return True, reduced
        else:
            if result.shape != expected_shape:
                return False, None
            if reduction_op == 'sum':
                return True, result.sum()
            elif reduction_op == 'max':
                return True, result.max()
            elif reduction_op == 'min':
                return True, result.min()
            elif reduction_op in ('product', 'prod'):
                return True, result.prod()
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
        var_defid = getattr(loop.variable, 'defid', None)
        if var_defid is None:
            var_name = getattr(loop.variable, 'name', '?')
            raise RuntimeError(
                f"Loop variable '{var_name}' has no defid; cannot bind at runtime. "
                "Ensure name resolution (AST) and rest_pattern / einstein_lowering set defid on index variables."
            )
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
        defid = getattr(binding, 'defid', None)
        if defid is None:
            name = getattr(binding, 'name', '?')
            raise RuntimeError(
                f"Binding '{name}' has no defid; cannot bind at runtime. "
                "Ensure name resolution sets defid on where-clause bindings."
            )
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
    reduction_op: str,
    reduction_ranges: Dict[Any, Any],  # Dict[DefId, LoopStructure]; use .values() for loops
    body_evaluator: Callable[[Dict[DefId, Any]], Any],
    expr_evaluator: Callable[[Any], Any],
    guard_evaluator: Optional[Callable[[Dict[DefId, Any]], bool]] = None,
    initial_context: Optional[Dict[DefId, Any]] = None,
    profile_callback: Optional[Callable[[str], None]] = None,
    parallel_shape: Optional[Tuple[int, ...]] = None,
) -> Any:
    """
    Execute reduction operation using nested loops with accumulators.

    Args:
        reduction_op: Operation name ('sum', 'prod', 'min', 'max', 'all', 'any')
        reduction_ranges: Dictionary mapping reduction variable DefId to LoopStructure (use .values() for loop list)
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
        >>> execute_reduction_with_loops('sum', ranges, body, evaluator)
        20
    """
    if initial_context is None:
        initial_context = {}

    # Convert reduction_ranges to list of loops
    reduction_loops = list(reduction_ranges.values())

    # Vectorized path does not support guards; when guard_evaluator is set use scalar loop path only.
    if guard_evaluator is None:
        ok, vec_result = _try_vectorized_reduction(
            reduction_op,
            reduction_loops,
            body_evaluator,
            expr_evaluator,
            parallel_shape=parallel_shape,
        )
        if ok and vec_result is not None:
            if profile_callback is not None:
                profile_callback("vectorized")
            return vec_result
    if profile_callback is not None:
        profile_callback("scalar")

    # Initialize accumulator based on operation
    if reduction_op == 'sum':
        accumulator = 0
        def combine(acc, val): return acc + val
    elif reduction_op == 'product' or reduction_op == 'prod':
        accumulator = 1
        def combine(acc, val): return acc * val
    elif reduction_op == 'min':
        accumulator = None
        def combine(acc, val):
            if acc is None:
                return val
            return min(acc, val)
    elif reduction_op == 'max':
        accumulator = None
        def combine(acc, val):
            import numpy as np
            if acc is None:
                return val
            # : use np.maximum for numpy arrays/scalars (handles f32 correctly)
            if isinstance(acc, np.ndarray) or isinstance(val, np.ndarray):
                return np.maximum(acc, val)
            return max(acc, val)
    elif reduction_op == 'all':
        accumulator = True
        def combine(acc, val):
            import numpy as np
            v = bool(np.all(val)) if isinstance(val, np.ndarray) else bool(val)
            return acc and v
    elif reduction_op == 'any':
        accumulator = False
        def combine(acc, val):
            import numpy as np
            v = bool(np.any(val)) if isinstance(val, np.ndarray) else bool(val)
            return acc or v
    else:
        raise ValueError(f"Unknown reduction operation: {reduction_op}")
    
    # Execute loops for reduction variables
    for reduction_context in execute_lowered_loops(
        reduction_loops,
        initial_context,
        expr_evaluator
    ):
        # Check guards if provided
        if guard_evaluator and not guard_evaluator(reduction_context):
            continue
        
        # Evaluate body
        value = body_evaluator(reduction_context)
        
        # Skip None values (from where expressions that filtered this item)
        if value is None:
            continue
        
        if reduction_op in ['all', 'any']:
            import numpy as np
            if isinstance(value, np.ndarray):
                value = bool(value.item()) if value.size == 1 else (bool(np.all(value)) if reduction_op == 'all' else bool(np.any(value)))
            else:
                value = bool(value)
        
        # Accumulate
        accumulator = combine(accumulator, value)
    
    # Handle empty reduction case for max/min
    if accumulator is None and reduction_op in ['min', 'max']:
        raise ValueError(f"{reduction_op}() arg is an empty sequence")
    
    return accumulator

