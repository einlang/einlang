"""NumPy backend Einstein execution mixin."""

from .numpy_einstein_vectorization import *
from .numpy_einstein_analysis import (
    _allocate_numpy_output,
    _evaluate_shape_dim,
    _vectorize_axes_scalar_vs_vector,
    _BodyReferencesDefidVisitor,
)
from .numpy_einstein_call_index_analysis import (
    _body_contains_call_using_loop_var,
    _body_is_elementwise_call,
    _collect_defids_by_name,
    _loop_defids_in_call_args,
)
from .numpy_einstein_recurrence_analysis import (
    _extract_loop_range,
    _recurrence_dims,
    _slice_list_from_clause_indices,
)
from .numpy_einstein_vectorization import (
    _eval_clause_body_with_broadcast_loops,
    _try_call_scalar_vectorize_clause,
    _try_hybrid_vectorize_clause,
    _try_slice_vectorize_if_clause,
    _try_vectorize_clause,
)

class EinsteinExecutionClauseMixin:
    def _execute_lowered_einstein_clause(
        self,
        lowered: LoweredEinsteinClauseIR,
        variable_decl: Any,
        shape: Optional[List] = None,
        element_type: Optional[Any] = None,
        pre_allocated_output: Optional[Any] = None,
    ) -> Any:
        from ..runtime.compute.lowered_execution import execute_lowered_loops, execute_lowered_bindings, check_lowered_guards
        loc = lowered.location or variable_decl.location
        line = int(getattr(loc, "line", 0) or 0)
        _clause_name = (
            getattr(variable_decl, "name", None)
            or getattr(getattr(variable_decl, "_binding", None), "name", None)
            or ""
        )
        _clause_rhs = str(lowered.body)[:60] if lowered.body is not None else "?"
        bucket_size = getattr(self, "_profile_bucket_size", 0)
        _profile_clauses = getattr(self, "_profile_functions", False) or getattr(self, "_profile_statements", False)
        t0 = time.perf_counter() if (bucket_size > 0 or _profile_clauses) else 0
        def _record_profile(shape: Optional[tuple] = None, path: Optional[str] = None) -> None:
            if bucket_size > 0 and getattr(self, "_profile_buckets", None) is not None:
                key = (line // bucket_size) * bucket_size
                self._profile_buckets[key] = self._profile_buckets.get(key, 0) + (time.perf_counter() - t0)
            if _profile_clauses and t0:
                elapsed = time.perf_counter() - t0
                parts = [f"[profile] clause L{line or '?'}"]
                if _clause_name or _clause_rhs:
                    lhs = _clause_name or "?"
                    parts.append(f" {lhs} = {_clause_rhs}")
                if shape is not None:
                    parts.append(f" {shape}")
                parts.append(f": \033[32m{elapsed:.3f}s\033[0m")
                if path:
                    parts.append(f" [{path}]")
                print("".join(parts), flush=True)

        clause_indices = lowered.indices or []
        binding = getattr(variable_decl, "_binding", None)
        variable_defid = (binding.defid if binding else None) or getattr(variable_decl, "defid", None)
        body_node = lowered.body
        has_recurrence = bool(
            variable_defid is not None
            and body_node is not None
            and _BodyReferencesDefidVisitor(variable_defid).references(body_node)
            and len(_recurrence_dims(lowered, variable_defid, clause_indices)) > 0
        )
        self._einstein_recurrence_clause = has_recurrence

        def cell_index(full_context: dict) -> Optional[tuple]:
            out = []
            loop_pos = 0
            for idx in clause_indices:
                if isinstance(idx, LiteralIR):
                    try:
                        out.append(int(idx.value))
                    except (TypeError, ValueError):
                        break
                elif loop_pos < len(lowered.loops):
                    defid = lowered.loops[loop_pos].variable.defid
                    v = full_context.get(defid)
                    if v is None and defid is not None:
                        v = self.env.get_value(defid)
                    if v is None:
                        break
                    out.append(v)
                    loop_pos += 1
                else:
                    break
            return tuple(out) if len(out) == len(clause_indices) else None

        if pre_allocated_output is not None:
            output = pre_allocated_output
            if variable_defid is not None:
                self.env.set_value(variable_defid, output)
        else:
            output_shape = None
            if shape:
                output_shape = []
                for shape_dim in shape:
                    dim_value = _evaluate_shape_dim(shape_dim, self)
                    if isinstance(dim_value, (int, np.integer)):
                        output_shape.append(int(dim_value))
                    elif isinstance(dim_value, np.ndarray) and dim_value.ndim == 0:
                        try:
                            output_shape.append(int(dim_value))
                        except (TypeError, ValueError):
                            output_shape = None
                            break
                    else:
                        output_shape = None
                        break
            if output_shape is None and lowered.loops:
                output_shape = []
                for loop in lowered.loops:
                    it = loop.iterable
                    if it is None:
                        output_shape = None
                        break
                    if isinstance(it, LiteralIR) and isinstance(it.value, range):
                        output_shape.append(int(it.value.stop))
                    elif isinstance(it, RangeIR):
                        try:
                            end_val = int(it.end.accept(self))
                        except (TypeError, ValueError):
                            output_shape = None
                            break
                        output_shape.append(end_val)
                    else:
                        try:
                            r = it.accept(self)
                            output_shape.append(len(r) if hasattr(r, "__len__") else None)
                        except Exception:
                            output_shape = None
                        if output_shape and output_shape[-1] is None:
                            output_shape = None
                            break
            if output_shape is None:
                output_shape = [int(idx.value) + 1 if isinstance(idx, LiteralIR) else 1 for idx in clause_indices] if clause_indices else [1]
            dtype = self._dtype_for_clause_result(lowered.body, element_type)
            output = _allocate_numpy_output(output_shape, dtype)

        def expr_evaluator(expr: Any) -> Any:
            return expr.accept(self)

        has_literal_idx = any(isinstance(idx, LiteralIR) for idx in clause_indices)
        object_output = isinstance(output, np.ndarray) and output.dtype == object
        body_node = lowered.body
        loop_defids = [lp.variable.defid for lp in (lowered.loops or [])]
        has_call_using_loop = _body_contains_call_using_loop_var(body_node, [d for d in loop_defids if d is not None])
        # When body has a call that uses loop vars in its args (e.g. topk_2d_row_values(X, i, ...)), those vars must be scalar.
        # Try call-scalar first so we don't use wrong full-vectorize result (array-valued row index).
        if (
            lowered.loops
            and not has_literal_idx
            and has_call_using_loop
            and not object_output
        ):
            scalar_defids = _loop_defids_in_call_args(body_node, loop_defids)
            scalar_loop_indices_call = [
                dim
                for dim, lp in enumerate(lowered.loops)
                if lp.variable.defid in scalar_defids
            ]
            if 0 < len(scalar_loop_indices_call) < len(lowered.loops):
                call_hybrid_out = _try_call_scalar_vectorize_clause(
                    lowered,
                    list(output.shape),
                    output,
                    scalar_loop_indices_call,
                    expr_evaluator,
                    backend=self,
                )
                if call_hybrid_out is not None:
                    if variable_defid:
                        self._clause_set_output(variable_defid, output)
                    self._einstein_call_scalar = getattr(self, "_einstein_call_scalar", 0) + 1
                    self._vectorize_debug_log(
                        "call-scalar",
                        lowered,
                        variable_decl,
                        axes=_vectorize_axes_scalar_vs_vector(lowered, scalar_loop_indices_call),
                    )
                    _record_profile(tuple(output.shape) if getattr(output, "shape", None) is not None else None, path="call-scalar")
                    return output
        # Literal idx / self-ref (recurrence) -> scalar; other indices -> vectorize.
        # When body has recurrence (reads LHS at different index), try hybrid first so we read prior timestep correctly.
        recurrence_needs_scalar = False
        if (
            lowered.loops
            and variable_defid is not None
            and _BodyReferencesDefidVisitor(variable_defid).references(body_node)
        ):
            recurrence_dims = _recurrence_dims(lowered, variable_defid, clause_indices)
            if 0 < len(recurrence_dims) < len(lowered.loops):
                hybrid_out = _try_hybrid_vectorize_clause(
                    lowered, list(output.shape), output, variable_defid, expr_evaluator, backend=self,
                    clause_indices=clause_indices,
                )
                if hybrid_out is not None and not object_output:
                    if variable_defid:
                        self._clause_set_output(variable_defid, output)
                    self._einstein_hybrid = getattr(self, "_einstein_hybrid", 0) + 1
                    self._vectorize_debug_log("hybrid", lowered, variable_decl, axes="recurrence_hybrid")
                    _record_profile(tuple(output.shape) if getattr(output, "shape", None) is not None else None, path="hybrid")
                    return output
                recurrence_needs_scalar = True  # hybrid failed; use scalar path so we read LHS[t-1] correctly
            elif len(recurrence_dims) == len(lowered.loops) and recurrence_dims:
                # Every loop dim is recurrence (e.g. u[t] = f(u[t-1]) with a single t). Cannot vectorize over t;
                # must run scalar loop so prior indices of u are visible (e.g. numerics::euler_decay).
                recurrence_needs_scalar = True
        # Try full vectorize over loop dims (literal idx -> fixed slice; other dims -> vectorize).
        if lowered.loops and not object_output:
            # Slice-vectorize: body "if p < t then ... else 0" -> vectorize over [0..t), fill rest (e.g. emb in decode).
            if not recurrence_needs_scalar and not lowered.guards and not lowered.bindings:
                slice_vec = _try_slice_vectorize_if_clause(lowered, output, expr_evaluator, backend=self)
                if slice_vec is not None:
                    if variable_defid:
                        self._clause_set_output(variable_defid, output)
                    self._einstein_vectorized = getattr(self, "_einstein_vectorized", 0) + 1
                    self._vectorize_debug_log(
                        "vectorized",
                        lowered,
                        variable_decl,
                        axes="vectorized=all_loop_axes slice_if_pattern",
                    )
                    _record_profile(tuple(output.shape) if getattr(output, "shape", None) is not None else None, path="vectorized")
                    return output
            # Optional chunked execution to reduce peak memory (env EINLANG_CHUNK_ELEMENTS > 0).
            chunk_threshold = int(os.environ.get("EINLANG_CHUNK_ELEMENTS", "0") or "0")
            if (
                chunk_threshold > 0
                and output.size > chunk_threshold
                and not recurrence_needs_scalar
                and not has_literal_idx
                and len(lowered.loops) == output.ndim
            ):
                try:
                    full_ranges = [_extract_loop_range(lp, expr_evaluator) for lp in lowered.loops]
                    if len(full_ranges) == output.ndim and output.shape[0] > 1:
                        rest_size = max(1, output.size // output.shape[0])
                        chunk_rows = max(1, min(output.shape[0], chunk_threshold // rest_size))
                        if chunk_rows < output.shape[0]:
                            all_ok = True
                            for start in range(0, output.shape[0], chunk_rows):
                                end = min(start + chunk_rows, output.shape[0])
                                override = [(start, end)] + list(full_ranges[1:])
                                chunk_shape = [end - start] + list(output.shape[1:])
                                chunk_result = _try_vectorize_clause(
                                    lowered, chunk_shape, output.dtype, expr_evaluator, backend=self,
                                    loop_ranges_override=override,
                                )
                                if chunk_result is None:
                                    all_ok = False
                                    break
                                output[start:end, ...] = chunk_result.astype(output.dtype, copy=False)
                            if all_ok:
                                if variable_defid:
                                    self._clause_set_output(variable_defid, output)
                                self._einstein_vectorized = getattr(self, "_einstein_vectorized", 0) + 1
                                _path = getattr(self, "_last_reduction_fast_path", None) or "vectorized"
                                if hasattr(self, "_last_reduction_fast_path"):
                                    delattr(self, "_last_reduction_fast_path")
                                self._vectorize_debug_log(
                                    f"vectorized-chunk:{_path}",
                                    lowered,
                                    variable_decl,
                                    axes=f"vectorized=chunked fast={_path}",
                                )
                                _record_profile(tuple(output.shape) if getattr(output, "shape", None) is not None else None, path=_path)
                                return output
                except (RuntimeError, TypeError, ValueError):
                    pass
            # When clause has literal indices, vectorize only over loop dims so result shape matches loop dims.
            vec_shape = list(output.shape)
            if has_literal_idx and len(clause_indices) == output.ndim:
                try:
                    vec_shape = [int(_extract_loop_range(lp, expr_evaluator)[1]) - int(_extract_loop_range(lp, expr_evaluator)[0]) for lp in lowered.loops]
                except (RuntimeError, TypeError, ValueError):
                    vec_shape = list(output.shape)
            vec_result = _try_vectorize_clause(
                lowered, vec_shape, output.dtype, expr_evaluator, backend=self,
            )
            if recurrence_needs_scalar and vec_result is not None:
                vec_result = None  # force scalar path so recurrence reads prior timestep correctly
            if vec_result is not None:
                vec_result = np.asarray(vec_result)
                slice_list_from_indices = (
                    _slice_list_from_clause_indices(clause_indices, lowered, expr_evaluator)
                    if has_literal_idx and len(clause_indices) == output.ndim
                    else None
                )
                if vec_result.shape == output.shape:
                    if pre_allocated_output is not None and lowered.loops:
                        slices_list_partial: List[Any] = []
                        try:
                            if slice_list_from_indices is not None:
                                slices_list_partial = slice_list_from_indices
                            else:
                                for lp in lowered.loops:
                                    start, end = _extract_loop_range(lp, expr_evaluator)
                                    slices_list_partial.append(slice(int(start), int(end)))
                        except RuntimeError:
                            slices_list_partial = []
                        if slice_list_from_indices is not None and len(slices_list_partial) == output.ndim:
                            output[tuple(slices_list_partial)] = vec_result.astype(output.dtype, copy=False)
                        else:
                            range_is_full_partial = (
                                len(slices_list_partial) == len(lowered.loops)
                                and all(s.start == 0 and s.stop == output.shape[i] for i, s in enumerate(slices_list_partial) if isinstance(s, slice))
                            )
                            if len(slices_list_partial) == len(lowered.loops) and not range_is_full_partial:
                                recurrence_dims = _recurrence_dims(lowered, variable_defid, clause_indices) if _BodyReferencesDefidVisitor(variable_defid).references(lowered.body) else []
                                if recurrence_dims:
                                    hybrid_out = _try_hybrid_vectorize_clause(
                                        lowered, list(output.shape), output, variable_defid, expr_evaluator, backend=self,
                                        clause_indices=clause_indices,
                                    )
                                    if hybrid_out is not None:
                                        if variable_defid:
                                            self._clause_set_output(variable_defid, output)
                                        self._einstein_hybrid = getattr(self, "_einstein_hybrid", 0) + 1
                                        self._vectorize_debug_log(
                                            "hybrid-partial",
                                            lowered,
                                            variable_decl,
                                            axes="hybrid_partial+sliced_write",
                                        )
                                        _record_profile(tuple(output.shape) if getattr(output, "shape", None) is not None else None, path="hybrid")
                                        return output
                                    vec_result = None
                                else:
                                    output[tuple(slices_list_partial)] = vec_result[tuple(slices_list_partial)].astype(output.dtype, copy=False)
                            else:
                                output[:] = vec_result
                    else:
                        output[:] = vec_result
                if vec_result is not None:
                    if slice_list_from_indices is not None and len(slice_list_from_indices) == output.ndim:
                        output[tuple(slice_list_from_indices)] = vec_result.astype(output.dtype, copy=False)
                    elif vec_result.shape != output.shape and vec_result.size == output.size:
                        output[:] = vec_result.reshape(output.shape)
                    elif pre_allocated_output is not None and vec_result.ndim == output.ndim:
                        slices_list: List[Any] = []
                        try:
                            if slice_list_from_indices is not None and len(slice_list_from_indices) == output.ndim:
                                slices_list = slice_list_from_indices
                            else:
                                for lp in lowered.loops:
                                    start, end = _extract_loop_range(lp, expr_evaluator)
                                    slices_list.append(slice(int(start), int(end)))
                        except RuntimeError:
                            slices_list = []
                        if slice_list_from_indices is not None and len(slices_list) == output.ndim:
                            output[tuple(slices_list)] = vec_result.astype(output.dtype, copy=False)
                        else:
                            range_is_full = (
                                len(slices_list) == len(lowered.loops)
                                and all(s.start == 0 and s.stop == output.shape[i] for i, s in enumerate(slices_list))
                            )
                            if len(slices_list) == len(lowered.loops):
                                if range_is_full:
                                    np.copyto(output, np.broadcast_to(vec_result.astype(output.dtype, copy=False), output.shape))
                                else:
                                    output[tuple(slices_list)] = vec_result[tuple(slices_list)].astype(output.dtype, copy=False)
                            else:
                                output[:] = np.broadcast_to(vec_result, output.shape)
                    else:
                        output[:] = np.broadcast_to(vec_result, output.shape)
                    if variable_defid:
                        self._clause_set_output(variable_defid, output)
                    self._einstein_vectorized = getattr(self, "_einstein_vectorized", 0) + 1
                    _path = getattr(self, "_last_reduction_fast_path", None) or "vectorized"
                    if hasattr(self, "_last_reduction_fast_path"):
                        delattr(self, "_last_reduction_fast_path")
                    self._vectorize_debug_log(
                        _path,
                        lowered,
                        variable_decl,
                        axes=f"vectorized=all_loop_axes fast={_path}",
                    )
                    _record_profile(tuple(output.shape) if getattr(output, "shape", None) is not None else None, path=_path)
                    return output

        # Fallback: call-scalar hybrid when only some loop vars in call args (e.g. topk) and full vectorize failed.
        if (
            lowered.loops
            and not has_literal_idx
            and has_call_using_loop
        ):
            scalar_defids = _loop_defids_in_call_args(body_node, loop_defids)
            scalar_loop_indices = [
                dim
                for dim, lp in enumerate(lowered.loops)
                if lp.variable.defid in scalar_defids
            ]
            if 0 < len(scalar_loop_indices) < len(lowered.loops):
                call_hybrid_out = _try_call_scalar_vectorize_clause(
                    lowered,
                    list(output.shape),
                    output,
                    scalar_loop_indices,
                    expr_evaluator,
                    backend=self,
                )
                if call_hybrid_out is not None:
                    if variable_defid:
                        self._clause_set_output(variable_defid, output)
                    self._einstein_call_scalar = getattr(self, "_einstein_call_scalar", 0) + 1
                    self._vectorize_debug_log(
                        "call-scalar-fallback",
                        lowered,
                        variable_decl,
                        axes=_vectorize_axes_scalar_vs_vector(lowered, scalar_loop_indices),
                    )
                    _record_profile(tuple(output.shape) if getattr(output, "shape", None) is not None else None, path="call-scalar")
                    return output

        # Element-wise call (e.g. gelu(fc1[s,k])): must run once with full array, not scalar loop.
        if (
            lowered.loops
            and _body_is_elementwise_call(body_node, loop_defids)
        ):
            elem_result = _eval_clause_body_with_broadcast_loops(
                lowered, list(output.shape), expr_evaluator, self
            )
            if elem_result is not None and isinstance(elem_result, np.ndarray):
                assigned = False
                if elem_result.shape == output.shape:
                    output[:] = elem_result.astype(output.dtype, copy=False)
                    assigned = True
                elif elem_result.size == output.size:
                    output.reshape(-1)[:] = elem_result.reshape(-1).astype(output.dtype, copy=False)
                    assigned = True
                else:
                    try:
                        np.copyto(output, np.broadcast_to(elem_result, output.shape))
                        assigned = True
                    except (ValueError, TypeError):
                        pass
                if assigned:
                    if variable_defid:
                        self._clause_set_output(variable_defid, output)
                    self._einstein_vectorized = getattr(self, "_einstein_vectorized", 0) + 1
                    _path = getattr(self, "_last_reduction_fast_path", None) or "vectorized"
                    if hasattr(self, "_last_reduction_fast_path"):
                        delattr(self, "_last_reduction_fast_path")
                    self._vectorize_debug_log(
                        f"elem-call:{_path}",
                        lowered,
                        variable_decl,
                        axes=f"vectorized=elem_broadcast fast={_path}",
                    )
                    _record_profile(tuple(output.shape) if getattr(output, "shape", None) is not None else None, path=_path)
                    return output

        _loop_defid_to_name = {}
        _loops = lowered.loops
        for lp in _loops:
            v = lp.variable
            if v and v.defid:
                _loop_defid_to_name[v.defid] = v.name
        _body = lowered.body
        _bindings = lowered.bindings or []
        _guards = lowered.guards or []
        # Precompute cell_index spec: list of (is_literal, value_or_defid) so we avoid getattr per iteration.
        _cell_index_spec: List[Any] = []
        _loop_pos = 0
        for idx in clause_indices:
            if isinstance(idx, LiteralIR):
                try:
                    _cell_index_spec.append((True, int(idx.value)))
                except (TypeError, ValueError):
                    _cell_index_spec = None
                    break
            elif _loop_pos < len(_loops):
                _defid = _loops[_loop_pos].variable.defid
                _cell_index_spec.append((False, _defid))
                _loop_pos += 1
            else:
                _cell_index_spec = None
                break
        if _cell_index_spec is not None and _loop_pos != len(_loops):
            _cell_index_spec = None
        _loop_defids_tuple = tuple(lp.variable.defid for lp in _loops)

        with self.env.scope():
            # Child scope must see the clause output tensor (e.g. u in u[t-1]).
            if variable_defid is not None:
                self.env.set_value(variable_defid, output)
            if not _loops:
                if all(isinstance(idx, LiteralIR) for idx in clause_indices):
                    idx_tuple = tuple(int(idx.value) for idx in clause_indices)
                else:
                    idx_tuple = None
                if idx_tuple is not None:
                    value = _body.accept(self)
                    if value is not None:
                        if isinstance(value, np.ndarray):
                            if value.shape == output.shape:
                                output[:] = value.astype(output.dtype, copy=False)
                            else:
                                if value.ndim == 0:
                                    value = value.item()
                                elif value.size == 1:
                                    value = value.flatten()[0].item()
                                if not isinstance(value, np.ndarray):
                                    output[idx_tuple] = value
                        elif isinstance(value, np.generic):
                            output[idx_tuple] = value.item()
                        else:
                            output[idx_tuple] = value
            else:
                _MAX = int(DEFAULT_EINSTEIN_LOOP_MAX)
                _n = [0]
                _set_value = self.env.set_value
                _to_bool = self._to_bool
                for loop_context in execute_lowered_loops(_loops, {}, expr_evaluator):
                    _n[0] += 1
                    if _n[0] > _MAX:
                        raise RuntimeError("Einstein clause loop iterations exceeded limit.")
                    if _bindings:
                        full_context = execute_lowered_bindings(_bindings, loop_context, expr_evaluator)
                    else:
                        full_context = loop_context
                    for defid, val in full_context.items():
                        if defid is not None:
                            _set_value(defid, val, name=_loop_defid_to_name.get(defid))
                    if _guards and not check_lowered_guards(_guards, full_context, lambda e: _to_bool(e.accept(self))):
                        continue
                    try:
                        # Pass parallel indices into reduction so guard/body can see them.
                        # For SelectAtArgmaxIR bodies, primal_body and diff_body may use
                        # *different* defids for the same outer-loop variable name, so
                        # alias every defid found (via _collect_defids_by_name) not just first.
                        _ri_ctx = dict(full_context)
                        if _loops and _body is not None:
                            _outer_val = full_context
                            _body_defids_by_name = _collect_defids_by_name(_body)
                            for _lp in _loops:
                                _vv = _lp.variable
                                if (
                                    _vv is not None
                                    and _vv.defid is not None
                                    and _vv.name
                                    and _vv.defid in _outer_val
                                ):
                                    _val = _outer_val[_vv.defid]
                                    for _body_did in _body_defids_by_name.get(_vv.name, []):
                                        if _body_did != _vv.defid:
                                            _ri_ctx[_body_did] = _val
                        setattr(self, "_reduction_initial_context", _ri_ctx)
                        setattr(self, "_select_outer_index_defids", _loop_defids_tuple)
                        try:
                            value = _body.accept(self)
                        finally:
                            if hasattr(self, "_reduction_initial_context"):
                                delattr(self, "_reduction_initial_context")
                            if hasattr(self, "_select_outer_index_defids"):
                                delattr(self, "_select_outer_index_defids")
                    except IndexError:
                        continue
                    if _cell_index_spec is not None:
                        _parts = []
                        for _is_lit, _v in _cell_index_spec:
                            if _is_lit:
                                _parts.append(_v)
                            else:
                                _p = full_context.get(_v)
                                if _p is None and _v is not None:
                                    _p = self.env.get_value(_v)
                                if _p is None:
                                    _parts = None
                                    break
                                _parts.append(_p)
                        idx_tuple = tuple(_parts) if _parts is not None and len(_parts) == len(clause_indices) else None
                    else:
                        idx_tuple = cell_index(full_context)
                    if idx_tuple is None:
                        idx_tuple = tuple(full_context.get(d) for d in _loop_defids_tuple)
                    if idx_tuple is None:
                        continue
                    if len(idx_tuple) > output.ndim:
                        continue
                    # ∂(reduction)/∂x has the same shape as x. Pullback IR may use only batch index b (shape (B,))
                    # while the body returns a full (B, J) tensor or a single row as length-J vector (prod chain rule).
                    if (
                        len(idx_tuple) == 1
                        and isinstance(value, np.ndarray)
                        and output.ndim == 1
                        and value.size > output.size
                    ):
                        if (
                            value.ndim >= 2
                            and value.shape[0] == output.shape[0]
                        ):
                            output = _allocate_numpy_output(list(value.shape), output.dtype)
                            if variable_defid is not None:
                                self.env.set_value(variable_defid, output)
                        elif value.ndim == 1 and tuple(output.shape) == (1,):
                            output = _allocate_numpy_output([1, int(value.size)], output.dtype)
                            if variable_defid is not None:
                                self.env.set_value(variable_defid, output)
                    if len(idx_tuple) != output.ndim:
                        if not (len(idx_tuple) == 1 and output.ndim > 1):
                            continue
                    if isinstance(value, np.ndarray):
                        if value.ndim == 0:
                            value = value.item()
                        elif value.size == 1:
                            value = value.flatten()[0].item()
                    elif isinstance(value, np.generic):
                        value = value.item()
                    if len(idx_tuple) == 1:
                        # Row slice output[i] expects shape output.shape[1:]; reduction bodies may
                        # return (1, n1, n2, ...) (leading batch of 1) and trigger NumPy
                        # "setting an array element with a sequence" without reshape.
                        if isinstance(value, np.ndarray) and output.ndim > 1:
                            tail = output.shape[1:]
                            if tail and value.shape == (1,) + tail:
                                value = value.reshape(tail)
                            elif tail and value.shape == tail:
                                pass
                            elif tail and value.shape == output.shape:
                                ri = int(np.asarray(idx_tuple[0]).item())
                                value = value[ri]
                        output[idx_tuple[0]] = value
                    else:
                        output[idx_tuple] = value

        if variable_defid:
            self._clause_set_output(variable_defid, output)
        self._einstein_scalar = getattr(self, "_einstein_scalar", 0) + 1
        self._vectorize_debug_log(
            "scalar",
            lowered,
            variable_decl,
            axes="scalar=all_loop_axes_nested_loops",
        )
        _record_profile(tuple(output.shape) if getattr(output, "shape", None) is not None else None, path="scalar")
        return output
