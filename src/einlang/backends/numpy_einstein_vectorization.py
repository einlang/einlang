"""NumPy backend Einstein helpers."""

from .numpy_einstein_recurrence_analysis import *
from .numpy_einstein_analysis import (
    _count_reduction_dims_in_expr,
    _reduction_uses_clause_var_in_bounds,
)
from .numpy_einstein_call_index_analysis import _collect_defids_by_name
from .numpy_einstein_recurrence_analysis import (
    _extract_loop_range,
    _flatten_additive_terms,
    _loop_dims_from_clause_indices,
    _match_rect_access_offsets,
    _recurrence_dims_for_hybrid_or_full,
    _split_scalar_mul,
)


def _literal_int_value(idx: Any) -> Optional[int]:
    if not isinstance(idx, LiteralIR):
        return None
    try:
        return int(idx.value)
    except (TypeError, ValueError):
        return None


def _coerce_scalar_index(value: Any) -> Any:
    return int(value) if hasattr(value, "__int__") else value


def _extract_vectorization_loop_info(
    loops: List[Any],
    evaluator: Any,
    backend: Any = None,
    *,
    allow_dynamic_ranges: bool = False,
    loop_ranges_override: Optional[List[Tuple[int, int]]] = None,
) -> Optional[List[Tuple[Any, Optional[Tuple[int, int]], str]]]:
    loop_info: List[Tuple[Any, Optional[Tuple[int, int]], str]] = []
    for dim, loop in enumerate(loops):
        defid = loop.variable.defid
        if defid is None:
            return None
        if loop_ranges_override is not None and dim < len(loop_ranges_override):
            range_value: Optional[Tuple[int, int]] = loop_ranges_override[dim]
        else:
            try:
                cached_loop_range = getattr(backend, "_cached_loop_range", None) if backend is not None else None
                if callable(cached_loop_range):
                    range_value = cached_loop_range(loop, evaluator)
                else:
                    range_value = _extract_loop_range(loop, evaluator)
            except RuntimeError:
                if not allow_dynamic_ranges:
                    return None
                range_value = None
        loop_info.append((defid, range_value, loop.variable.name))
    return loop_info


def _resolve_loop_range_at_dim(
    loop_info: List[Tuple[Any, Optional[Tuple[int, int]], str]],
    loops: List[Any],
    dim: int,
    backend: Any,
) -> Optional[Tuple[int, int]]:
    range_value = loop_info[dim][1]
    if range_value is not None:
        return range_value
    try:
        cached_loop_range = getattr(backend, "_cached_loop_range", None)
        if callable(cached_loop_range):
            return cached_loop_range(loops[dim], lambda e: e.accept(backend))
        return _extract_loop_range(loops[dim], lambda e: e.accept(backend))
    except RuntimeError:
        return None


def _bind_vectorized_backend_state(
    backend: Any,
    parallel_shape: Tuple[int, ...],
    clause_loop_defids: List[Any],
    *,
    safe_oob: bool = False,
) -> Dict[str, Any]:
    return {
        "parallel_shape": parallel_shape,
        "parallel_defids_order": tuple(clause_loop_defids),
        "safe_oob": safe_oob,
    }


def _restore_vectorized_backend_state(backend: Any, saved: Dict[str, Any]) -> None:
    return None


def _squeeze_vectorized_result(
    result: np.ndarray,
    ndim: int,
    squeeze_dims: List[int],
) -> Optional[np.ndarray]:
    try:
        if result.ndim == ndim:
            return np.squeeze(result, axis=tuple(squeeze_dims))
        if result.ndim == ndim - len(squeeze_dims):
            return result
        axes = [dim for dim in squeeze_dims if dim < result.ndim]
        return np.squeeze(result, axis=tuple(axes)) if axes else result
    except ValueError:
        return None


def _split_binding_factor(expr: Any, binding_defid: Any) -> Optional[Any]:
    if isinstance(expr, IdentifierIR) and expr.defid == binding_defid:
        return LiteralIR(1.0)
    if isinstance(expr, BinaryOpIR) and expr.operator == BinaryOp.MUL:
        if isinstance(expr.left, IdentifierIR) and expr.left.defid == binding_defid:
            return expr.right
        if isinstance(expr.right, IdentifierIR) and expr.right.defid == binding_defid:
            return expr.left
    return None


def _try_fast_2d_wave_step(
    item: Any,
    output: np.ndarray,
    variable_defid: Any,
    rec_context: Dict[Any, Any],
    outer_rec_defids: List[Tuple[int, Any]],
    loop_info: List[Tuple[Any, Tuple[int, int], str]],
    expr_eval: Any,
) -> Optional[np.ndarray]:
    """Fast path for 2D leapfrog-style stencils: scalar in recurrence dim, sliced NumPy over space."""
    if not isinstance(item.body, BlockExpressionIR):
        return None
    body = item.body
    if len(loop_info) != 3 or len(outer_rec_defids) != 1 or outer_rec_defids[0][0] != 0:
        return None
    if len(body.statements or []) != 1 or body.final_expr is None:
        return None
    lap_binding = body.statements[0]
    lap_defid = getattr(lap_binding, "defid", None)
    lap_expr = getattr(lap_binding, "expr", None)
    if lap_defid is None or lap_expr is None:
        return None
    loop_defids = [info[0] for info in loop_info]
    lap_terms: Dict[Tuple[int, int, int], float] = {}
    for sign, term in _flatten_additive_terms(lap_expr):
        mul_coeff, base = _split_scalar_mul(term)
        coeff = sign * mul_coeff
        matched = False
        for offsets in (
            (-1, -1, 0),
            (-1, 1, 0),
            (-1, 0, -1),
            (-1, 0, 1),
            (-1, 0, 0),
        ):
            if _match_rect_access_offsets(base, variable_defid, loop_defids, offsets):
                lap_terms[offsets] = lap_terms.get(offsets, 0.0) + coeff
                matched = True
                break
        if not matched:
            return None
    expected_lap = {
        (-1, -1, 0): 1.0,
        (-1, 1, 0): 1.0,
        (-1, 0, -1): 1.0,
        (-1, 0, 1): 1.0,
        (-1, 0, 0): -4.0,
    }
    if lap_terms != expected_lap:
        return None

    center_coeffs: Dict[Tuple[int, int, int], float] = {}
    lap_factor_expr = None
    for sign, term in _flatten_additive_terms(body.final_expr):
        mul_coeff, base = _split_scalar_mul(term)
        coeff = sign * mul_coeff
        if _match_rect_access_offsets(base, variable_defid, loop_defids, (-1, 0, 0)):
            center_coeffs[(-1, 0, 0)] = center_coeffs.get((-1, 0, 0), 0.0) + coeff
            continue
        if _match_rect_access_offsets(base, variable_defid, loop_defids, (-2, 0, 0)):
            center_coeffs[(-2, 0, 0)] = center_coeffs.get((-2, 0, 0), 0.0) + coeff
            continue
        factor_expr = _split_binding_factor(base, lap_defid)
        if factor_expr is not None:
            if lap_factor_expr is not None or coeff != 1.0:
                return None
            lap_factor_expr = factor_expr
            continue
        return None
    if center_coeffs != {(-1, 0, 0): 2.0, (-2, 0, 0): -1.0} or lap_factor_expr is None:
        return None

    t_outer_defid = outer_rec_defids[0][1]
    if t_outer_defid not in rec_context:
        return None
    t = int(rec_context[t_outer_defid])
    _, (i_start, i_end), _ = loop_info[1]
    _, (j_start, j_end), _ = loop_info[2]
    if t < 2:
        return None
    if i_start <= 0 or j_start <= 0:
        return None
    if i_end >= output.shape[1] or j_end >= output.shape[2]:
        return None
    r_val = expr_eval(lap_factor_expr)
    if isinstance(r_val, np.ndarray):
        if r_val.ndim != 0 and r_val.size != 1:
            return None
        r_val = float(np.asarray(r_val).reshape(-1)[0])
    elif not isinstance(r_val, (int, float, np.integer, np.floating)):
        return None
    prev = output[t - 1]
    prev2 = output[t - 2]
    center = prev[i_start:i_end, j_start:j_end]
    lap = (
        prev[i_start - 1:i_end - 1, j_start:j_end]
        + prev[i_start + 1:i_end + 1, j_start:j_end]
        + prev[i_start:i_end, j_start - 1:j_end - 1]
        + prev[i_start:i_end, j_start + 1:j_end + 1]
        - 4.0 * center
    )
    output[t, i_start:i_end, j_start:j_end] = (
        2.0 * center
        - prev2[i_start:i_end, j_start:j_end]
        + float(r_val) * lap
    ).astype(output.dtype, copy=False)
    return output


def _eval_clause_body_with_broadcast_loops(
    clause: Any,
    output_shape: List[int],
    evaluator: Any,
    backend: Any,
    loop_ranges_override: Optional[List[Tuple[int, int]]] = None,
) -> Optional[Any]:
    """Evaluate clause body once with loop vars set to broadcast arrays. Returns result or None on failure.
    If loop_ranges_override is set, use (start,end) per dimension instead of extracting from clause (for chunked execution)."""
    loops = (clause.loops or [])
    if not loops or clause.guards or clause.bindings:
        return None
    clause_ndim = len(loops)
    clause_facts = None
    lowered_clause_facts = getattr(backend, "_lowered_einstein_clause_facts", None)
    if callable(lowered_clause_facts):
        clause_facts = lowered_clause_facts(clause)
    if clause_facts is not None:
        n_red = int(getattr(clause_facts, "body_reduction_dim_count", 0) or 0)
    else:
        n_red = _count_reduction_dims_in_expr(clause.body)
    ndim = clause_ndim + n_red
    loop_info = _extract_vectorization_loop_info(
        loops,
        evaluator,
        backend,
        loop_ranges_override=loop_ranges_override,
    )
    if loop_info is None:
        return None
    clause_loop_defids = [defid for (defid, _, _) in loop_info]
    if clause_facts is not None:
        uses_clause_var_in_bounds = bool(getattr(clause_facts, "body_reduction_uses_clause_var_in_bounds", False))
    else:
        uses_clause_var_in_bounds = _reduction_uses_clause_var_in_bounds(clause.body, clause_loop_defids)
    if uses_clause_var_in_bounds:
        return None
    if clause_facts is not None:
        body_defids_by_name = {
            name: list(dids)
            for name, dids in (getattr(clause_facts, "body_defids_by_name", {}) or {}).items()
        }
        safe_oob = bool(getattr(clause_facts, "body_contains_if_expression", False))
    else:
        body_defids_by_name = _collect_defids_by_name(clause.body)
        safe_oob = isinstance(clause.body, IfExpressionIR)
    try:
        with backend.env.scope():
            for dim, (defid, rng, name) in enumerate(loop_info):
                start, end = rng
                sz = end - start
                shape = [1] * ndim
                shape[dim] = sz
                arr = np.arange(start, end, dtype=np.intp).reshape(shape)
                backend.env.set_value(defid, arr, name=name)
                for other_defid in body_defids_by_name.get(name, []):
                    if other_defid != defid:
                        backend.env.set_value(other_defid, arr, name=name)
            parallel_shape_tuple = tuple(output_shape)
            saved_state = _bind_vectorized_backend_state(
                backend,
                parallel_shape_tuple,
                clause_loop_defids,
                safe_oob=safe_oob,
            )
            with backend._vectorization_scope(**saved_state):
                return clause.body.accept(backend)
    except Exception as exc:
        if backend is not None and getattr(backend, "_vectorize_debug_enabled", None) and backend._vectorize_debug_enabled():
            loc = getattr(clause, "location", None)
            line = int(getattr(loc, "line", 0) or 0)
            seen = getattr(backend, "_vectorize_eval_failed_seen", None)
            if seen is None:
                seen = set()
                backend._vectorize_eval_failed_seen = seen
            key = (line, type(exc).__name__, str(exc))
            if key not in seen:
                seen.add(key)
                print(
                    f"[vectorize] eval-failed L{line or '?'} {type(exc).__name__}: {exc}",
                    flush=True,
                )
        return None


def _try_slice_vectorize_if_clause(
    lowered: Any,
    output: np.ndarray,
    expr_evaluator: Any,
    backend: Any,
) -> Optional[np.ndarray]:
    """
    When body is (if loop_var < bound then then_expr else else_expr), vectorize over
    [0..bound) and fill the rest with else_expr. Speeds up patterns like
    emb[p,d] = if p < t then dec_tok_emb[tokens[p], d] + ... else 0.
    """
    body = lowered.body
    if not isinstance(body, IfExpressionIR):
        return None
    cond = body.condition
    if not isinstance(cond, BinaryOpIR):
        return None
    op = cond.operator
    lt_ops = (BinaryOp.LT, "<")
    if op not in lt_ops:
        return None
    left = cond.left
    right = cond.right
    if left is None or right is None:
        return None
    loops = (lowered.loops or [])
    if not loops:
        return None
    loop_defids = [lp.variable.defid for lp in loops]
    left_defid = left.defid if isinstance(left, (IdentifierIR, IndexVarIR)) else None
    right_defid = right.defid if isinstance(right, (IdentifierIR, IndexVarIR)) else None
    bound_side = None
    dim = -1
    if left_defid is not None and left_defid in loop_defids:
        bound_side = right
        dim = loop_defids.index(left_defid)
    elif right_defid is not None and right_defid in loop_defids:
        bound_side = left
        dim = loop_defids.index(right_defid)
    if bound_side is None or dim < 0:
        return None
    try:
        if isinstance(bound_side, LiteralIR):
            bound = int(bound_side.value)
        elif opt_defid(bound_side) is not None:
            bound = backend.env.get_value(bound_side.defid)
            bound = int(bound) if bound is not None else None
        else:
            bound = bound_side.accept(backend)
            bound = int(bound) if bound is not None else None
        if bound is None or bound < 0:
            return None
    except (TypeError, ValueError, AttributeError):
        return None
    full_shape = list(output.shape)
    if dim >= len(full_shape) or bound > full_shape[dim]:
        return None
    override: List[Tuple[int, int]] = []
    for i, lp in enumerate(loops):
        if i == dim:
            override.append((0, bound))
        else:
            try:
                r = _extract_loop_range(lp, expr_evaluator)
                override.append(r)
            except RuntimeError:
                return None
    slice_shape = [override[i][1] - override[i][0] for i in range(len(override))]
    result = _try_vectorize_clause(
        lowered, slice_shape, output.dtype, expr_evaluator, backend,
        loop_ranges_override=override,
    )
    if result is None:
        return None
    result = np.asarray(result)
    if result.shape != tuple(slice_shape):
        return None
    else_expr = body.else_expr
    else_val = 0
    if isinstance(else_expr, LiteralIR):
        v = else_expr.value
        try:
            else_val = float(v) if isinstance(v, (int, float)) else 0
        except (TypeError, ValueError):
            return None
    elif else_expr is not None:
        try:
            else_val = else_expr.accept(backend)
            if isinstance(else_val, np.ndarray):
                if else_val.ndim == 0:
                    else_val = float(else_val)
                else:
                    return None
            elif not isinstance(else_val, (int, float)):
                return None
        except Exception:
            return None
    output.fill(else_val)
    sl: List[Any] = [slice(None)] * output.ndim
    sl[dim] = slice(0, bound)
    output[tuple(sl)] = result.astype(output.dtype, copy=False)
    return output


def _try_vectorize_clause(
    clause,
    output_shape,
    dtype,
    evaluator,
    backend=None,
    loop_ranges_override: Optional[List[Tuple[int, int]]] = None,
):
    """
    General vectorization: set loop variables to broadcast numpy arrays,
    evaluate the body once, and let numpy handle everything.
    Falls back to None if any operation doesn't support array-valued indices.
    If loop_ranges_override is set, use those (start,end) per dimension for chunked execution.
    """
    if backend is None:
        return None
    loops = clause.loops or []
    if not loops:
        return None
    if clause.guards:
        return None
    if clause.bindings:
        return None

    clause_ndim = len(loops)
    clause_facts = None
    lowered_clause_facts = getattr(backend, "_lowered_einstein_clause_facts", None)
    if callable(lowered_clause_facts):
        clause_facts = lowered_clause_facts(clause)
    if clause_facts is not None:
        n_red = int(getattr(clause_facts, "body_reduction_dim_count", 0) or 0)
    else:
        n_red = _count_reduction_dims_in_expr(clause.body)
    ndim = clause_ndim + n_red

    loop_info: List[Tuple[Any, Tuple[int, int], str]] = []
    for dim, lp in enumerate(loops):
        defid = lp.variable.defid
        if defid is None:
            return None
        if loop_ranges_override is not None and dim < len(loop_ranges_override):
            r = loop_ranges_override[dim]
        else:
            r = _extract_loop_range(lp, evaluator)
        name = lp.variable.name
        loop_info.append((defid, r, name))

    clause_loop_defids = [defid for (defid, _, _) in loop_info]
    if clause_facts is not None:
        uses_clause_var_in_bounds = bool(getattr(clause_facts, "body_reduction_uses_clause_var_in_bounds", False))
    else:
        uses_clause_var_in_bounds = _reduction_uses_clause_var_in_bounds(clause.body, clause_loop_defids)
    if uses_clause_var_in_bounds:
        return None

    try:
        result = _eval_clause_body_with_broadcast_loops(
            clause, output_shape, evaluator, backend, loop_ranges_override=loop_ranges_override
        )
        if result is None:
            return None
        if isinstance(result, np.ndarray):
            expected = tuple(output_shape)
            ranges = [(start, end) for (_, (start, end), _) in loop_info]
            range_is_full = len(ranges) == len(expected) and all(
                start == 0 and end == expected[dim] for dim, (start, end) in enumerate(ranges)
            )
            if result.shape == expected:
                return result.astype(dtype, copy=False)
            if not range_is_full:
                full = np.zeros(expected, dtype=dtype)
                slices = tuple(slice(int(start), int(end)) for (start, end) in ranges)
                full[slices] = result.astype(dtype, copy=False)
                return full
            if result.size == np.prod(expected):
                return result.reshape(expected).astype(dtype, copy=False)
            try:
                return np.broadcast_to(result, expected).copy().astype(dtype, copy=False)
            except ValueError:
                return None
        if isinstance(result, (int, float, np.integer, np.floating)):
            return np.full(output_shape, result, dtype=dtype)
    except Exception:
        return None
    return None


def _try_hybrid_vectorize_clause(
    clause: Any,
    output_shape: List[int],
    output: np.ndarray,
    variable_defid: Any,
    expr_evaluator: Any,
    backend: Any,
    clause_indices: Optional[List[Any]] = None,
) -> Optional[np.ndarray]:
    """
    When the body has recurrence on a subset of dimensions: iterate over those
    (scalar), vectorize over the rest. Writes into output and returns it, or None on failure.
    When clause_indices is set (and has literals), build slice in output space and use
    recurrence_dims from _recurrence_dims_for_hybrid_or_full(..., clause_indices).
    """
    from ..runtime.compute.lowered_execution import execute_lowered_loops
    loops = (clause.loops or [])
    if not loops or clause.guards:
        return None
    if clause.bindings:
        return None
    ndim = len(loops)
    loop_dims = _loop_dims_from_clause_indices(clause_indices, loops) if clause_indices else None
    recurrence_dims = (
        _recurrence_dims_for_hybrid_or_full(clause, variable_defid, clause_indices)
        if variable_defid else []
    )
    if not (0 < len(recurrence_dims) < ndim):
        return None
    loop_info = _extract_vectorization_loop_info(
        loops,
        expr_evaluator,
        backend,
        allow_dynamic_ranges=True,
    )
    if loop_info is None:
        return None
    recurrence_loops = [loops[d] for d in recurrence_dims]
    _MAX = int(DEFAULT_EINSTEIN_LOOP_MAX)
    n_iter = [0]
    output_ndim = output.ndim
    has_literal = bool(clause_indices and any(isinstance(idx, LiteralIR) for idx in clause_indices))
    clause_facts = None
    lowered_clause_facts = getattr(backend, "_lowered_einstein_clause_facts", None)
    if callable(lowered_clause_facts):
        clause_facts = lowered_clause_facts(clause)
    if clause_facts is not None:
        body_defids_by_name = {
            name: list(dids)
            for name, dids in (getattr(clause_facts, "body_defids_by_name", {}) or {}).items()
        }
    else:
        body_defids_by_name = _collect_defids_by_name(clause.body)

    try:
        for rec_context in execute_lowered_loops(recurrence_loops, {}, expr_evaluator):
            n_iter[0] += 1
            if n_iter[0] > _MAX:
                raise RuntimeError(
                    f"Einstein clause loop iterations exceeded limit ({_MAX}). "
                    "Reduce clause range or increase config.DEFAULT_EINSTEIN_LOOP_MAX."
                )
            with backend.env.scope():
                for dim in range(ndim):
                    defid, range_val, name = loop_info[dim]
                    if dim in recurrence_dims:
                        val = rec_context[defid]
                        backend.env.set_value(defid, val, name=name)
                        for other_defid in body_defids_by_name.get(name, []):
                            if other_defid != defid:
                                backend.env.set_value(other_defid, val, name=name)
                    else:
                        start, end = range_val if range_val is not None else (None, None)
                        if range_val is None:
                            try:
                                cached_loop_range = getattr(backend, "_cached_loop_range", None)
                                if callable(cached_loop_range):
                                    start, end = cached_loop_range(loops[dim], lambda e: e.accept(backend))
                                else:
                                    start, end = _extract_loop_range(
                                        loops[dim], lambda e: e.accept(backend)
                                    )
                            except RuntimeError:
                                return None
                        sz = end - start
                        shape = [1] * ndim
                        shape[dim] = sz
                        arr = np.arange(start, end, dtype=np.intp).reshape(shape)
                        backend.env.set_value(defid, arr, name=name)
                result = clause.body.accept(backend)
            if not isinstance(result, np.ndarray):
                return None
            if has_literal and clause_indices is not None and loop_dims is not None and len(clause_indices) == output_ndim:
                slice_list_out: List[Any] = []
                loop_pos = 0
                for out_d, idx in enumerate(clause_indices):
                    if isinstance(idx, LiteralIR):
                        literal_value = _literal_int_value(idx)
                        if literal_value is None:
                            return None
                        slice_list_out.append(literal_value)
                    elif loop_pos < len(loops):
                        k = loop_pos
                        loop_pos += 1
                        if k in recurrence_dims:
                            v = rec_context[loop_info[k][0]]
                            slice_list_out.append(_coerce_scalar_index(v))
                        else:
                            r = _resolve_loop_range_at_dim(loop_info, loops, k, backend)
                            if r is None:
                                return None
                            start, end = r
                            slice_list_out.append(slice(int(start), int(end)))
                    else:
                        return None
                if len(slice_list_out) != output_ndim:
                    return None
                to_write = result
                to_write = _squeeze_vectorized_result(to_write, ndim, recurrence_dims)
                if to_write is None:
                    return None
                output[tuple(slice_list_out)] = to_write.astype(output.dtype)
            else:
                slice_list: List[Any] = []
                for dim in range(ndim):
                    if dim in recurrence_dims:
                        slice_list.append(rec_context[loop_info[dim][0]])
                    else:
                        r = _resolve_loop_range_at_dim(loop_info, loops, dim, backend)
                        if r is None:
                            return None
                        start, end = r
                        slice_list.append(slice(int(start), int(end)))
                squeezed = _squeeze_vectorized_result(result, ndim, recurrence_dims)
                if squeezed is None:
                    return None
                output[tuple(slice_list)] = squeezed.astype(output.dtype)
        return output
    except Exception:
        return None


def _try_call_scalar_vectorize_clause(
    clause: Any,
    output_shape: List[int],
    output: np.ndarray,
    scalar_loop_indices: List[int],
    expr_evaluator: Any,
    backend: Any,
) -> Optional[np.ndarray]:
    """
    When body has a non-element-wise call using some loop vars: iterate over those (scalar),
    vectorize over the rest. E.g. topk_2d_row_values(X, i, ...)[j]: scalar over i, vector over j.
    """
    from ..runtime.compute.lowered_execution import execute_lowered_loops
    loops = (clause.loops or [])
    if not loops or clause.guards or clause.bindings:
        return None
    ndim = len(loops)
    scalar_set = set(scalar_loop_indices)
    if not (0 < len(scalar_set) < ndim):
        return None
    vector_dims = [d for d in range(ndim) if d not in scalar_set]
    loop_info = _extract_vectorization_loop_info(loops, expr_evaluator, backend)
    if loop_info is None:
        return None
    scalar_loops = [loops[d] for d in scalar_loop_indices]
    _MAX = int(DEFAULT_EINSTEIN_LOOP_MAX)
    n_iter = [0]
    try:
        for scalar_context in execute_lowered_loops(scalar_loops, {}, expr_evaluator):
            n_iter[0] += 1
            if n_iter[0] > _MAX:
                return None
            with backend.env.scope():
                for dim in range(ndim):
                    defid, (start, end), name = loop_info[dim]
                    if dim in scalar_set:
                        backend.env.set_value(defid, scalar_context[defid], name=name)
                    else:
                        sz = end - start
                        shape = [1] * ndim
                        shape[dim] = sz
                        arr = np.arange(start, end, dtype=np.intp).reshape(shape)
                        backend.env.set_value(defid, arr, name=name)
                result = clause.body.accept(backend)
            if not isinstance(result, np.ndarray):
                return None
            slice_list: List[Any] = []
            for dim in range(ndim):
                if dim in scalar_set:
                    slice_list.append(scalar_context[loop_info[dim][0]])
                else:
                    start, end = loop_info[dim][1]
                    slice_list.append(slice(int(start), int(end)))
            squeezed = _squeeze_vectorized_result(result, ndim, scalar_loop_indices)
            if squeezed is None:
                return None
            output[tuple(slice_list)] = squeezed.astype(output.dtype)
        return output
    except Exception:
        return None
