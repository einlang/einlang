"""NumPy backend Einstein execution mixin."""

from .numpy_einstein_vectorization import *
from .numpy_einstein_analysis import _einlang_recurrence_block_vectorized_binding_enabled
from .numpy_einstein_recurrence_analysis import (
    _extract_loop_range,
    _recurrence_dims,
)
from .numpy_einstein_vectorization import _try_fast_2d_wave_step

class EinsteinExecutionRecurrenceMixin:
    def _assign_recurrence_step_result(
        self,
        output: np.ndarray,
        slice_list: List[Any],
        res: Any,
        *,
        object_output: bool,
    ) -> bool:
        idx = tuple(slice_list)
        target = output[idx]
        if object_output:
            output[idx] = res
            return True
        arr = np.asarray(res, dtype=output.dtype)
        if isinstance(target, np.ndarray):
            if arr.shape != target.shape:
                return False
            output[idx] = arr
            return True
        if arr.size != 1:
            return False
        output[idx] = arr.reshape(-1)[0]
        return True

    def _outer_recurrence_defid_for_dim(
        self,
        outer_rec_defids: List[Tuple[int, Any]],
        dim: int,
    ) -> Optional[Any]:
        for outer_dim, outer_defid in outer_rec_defids:
            if outer_dim == dim:
                return outer_defid
        return None

    def _set_loop_value_or_range(
        self,
        defid: Any,
        bounds: Tuple[int, int],
        name: str,
        *,
        scalar_value: Optional[Any] = None,
        reshape_rank: Optional[int] = None,
        reshape_dim: Optional[int] = None,
    ) -> None:
        if scalar_value is not None:
            self.env.set_value(defid, scalar_value, name=name)
            return
        start, end = bounds
        arr = np.arange(start, end, dtype=np.intp)
        if reshape_rank is not None and reshape_dim is not None:
            shape = [1] * reshape_rank
            shape[reshape_dim] = end - start
            arr = arr.reshape(shape)
        self.env.set_value(defid, arr, name=name)

    def _bind_outer_recurrence_context(
        self,
        outer_rec_defids: List[Tuple[int, Any]],
        rec_context: Dict[Any, Any],
    ) -> None:
        for _dim, outer_defid in outer_rec_defids:
            if outer_defid in rec_context:
                self.env.set_value(outer_defid, rec_context[outer_defid])

    def _apply_lowered_bindings(
        self,
        bindings: List[Any],
        loops: List[Any],
        expr_eval: Any,
        execute_lowered_bindings: Any,
    ) -> None:
        if not bindings:
            return
        loop_context = {}
        for loop in loops:
            defid = loop.variable.defid
            if defid is not None:
                loop_context[defid] = self.env.get_value(defid)
        full_context = execute_lowered_bindings(bindings, loop_context, expr_eval)
        for defid, val in full_context.items():
            if defid is not None:
                self.env.set_value(defid, val)

    def _literal_index_value(self, idx: Any) -> Optional[int]:
        if isinstance(idx, LiteralIR):
            try:
                return int(idx.value)
            except (TypeError, ValueError):
                return None
        if getattr(idx, "value", None) is not None:
            try:
                return int(getattr(idx, "value"))
            except (TypeError, ValueError):
                return None
        return None

    def _coerce_scalar_index(self, value: Any) -> Any:
        if isinstance(value, (np.integer, np.floating, int, float)):
            return int(value)
        return value

    def _execute_lowered_einstein_clause_one_recurrence_step(
        self,
        item: Any,
        variable_decl: Any,
        output: np.ndarray,
        variable_key: Any,
        variable_defid: Any,
        rec_context: Dict[Any, Any],
        recurrence_loops_outer: List[Any],
        expr_eval: Any,
        tensor_shape: Optional[List] = None,
        tensor_element_type: Optional[Any] = None,
    ) -> Optional[Any]:
        """Run one clause for one recurrence step (rec_context); vectorize over other dims. Used by timestep-major.
        recurrence_loops_outer: loops we iterate over (same order as rec_context keys); use their variable defids
        so every clause gets the current timestep even if its own loop var has a different defid."""
        from ..runtime.compute.lowered_execution import execute_lowered_bindings
        object_output = isinstance(output, np.ndarray) and output.dtype == object
        loops = item.loops or []
        if not loops:
            return None
        guards = item.guards or []
        clause_indices = item.indices or []
        recurrence_dims = item.recurrence_dims_override
        if recurrence_dims is None:
            recurrence_dims = _recurrence_dims(item, variable_defid, clause_indices)
        elif not recurrence_dims:
            recurrence_dims = _recurrence_dims(item, variable_defid, clause_indices)
        recurrence_dims = recurrence_dims or []
        ndim = len(loops)
        # Allow clause to run when we have at least as many loops as outer (so we can bind current step).
        if not recurrence_dims or len(loops) < len(recurrence_loops_outer):
            return None
        loop_info: List[Tuple[Any, Tuple[int, int], str]] = []
        for dim, lp in enumerate(loops):
            defid = lp.variable.defid
            if defid is None:
                return None
            try:
                r = _extract_loop_range(lp, expr_eval)
            except RuntimeError:
                return None
            name = lp.variable.name
            loop_info.append((defid, r, name))
        # Map clause dim k to k-th outer loop's defid (so all outer vars bound from rec_context for current step).
        outer_rec_defids = []
        for k in range(min(len(recurrence_loops_outer), ndim)):
            outer_defid = recurrence_loops_outer[k].variable.defid
            if outer_defid is None:
                return None
            outer_rec_defids.append((k, outer_defid))
        outer_dims_set = {d for d, _ in outer_rec_defids}
        inner_recurrence_dims = [d for d in recurrence_dims if d not in outer_dims_set]
        # All clause dims bound from current step (e.g. diagonal clause with i==j in row-major): run body once, write one cell.
        if len(outer_dims_set) == ndim and not inner_recurrence_dims:
            with self.env.scope():
                self.env.set_value(variable_defid, output)
                self._bind_outer_recurrence_context(outer_rec_defids, rec_context)
                for dim in range(ndim):
                    defid, bounds, name = loop_info[dim]
                    outer_defid = self._outer_recurrence_defid_for_dim(outer_rec_defids, dim)
                    scalar_value = rec_context.get(outer_defid) if outer_defid in rec_context else None
                    self._set_loop_value_or_range(defid, bounds, name, scalar_value=scalar_value)
                bindings = item.bindings or []
                self._apply_lowered_bindings(bindings, loops, expr_eval, execute_lowered_bindings)
                if guards:
                    from ..runtime.compute.lowered_execution import check_lowered_guards
                    step_ctx = {loop_info[dim][0]: rec_context.get(odef) for dim, (_, odef) in enumerate(outer_rec_defids) if odef in rec_context}
                    if not check_lowered_guards(guards, step_ctx, lambda e: self._to_bool(e.accept(self))):
                        if variable_key is not None:
                            self.env.set_value(variable_key, output)
                        return output
                with self._vectorization_scope(
                    recurrence_clause=True,
                    parallel_shape=None,
                ):
                    # Use scalar reduction path so recurrence reads the current env, not a vectorized fast path.
                    res = item.body.accept(self)
            if res is not None:
                slice_list: List[Any] = []
                for dim in range(ndim):
                    outer_defid = self._outer_recurrence_defid_for_dim(outer_rec_defids, dim)
                    if outer_defid is not None and outer_defid in rec_context:
                        slice_list.append(self._coerce_scalar_index(rec_context[outer_defid]))
                    else:
                        slice_list.append(slice(loop_info[dim][1][0], loop_info[dim][1][1]))
                if self._assign_recurrence_step_result(
                    output,
                    slice_list,
                    res,
                    object_output=object_output,
                ):
                    if variable_key is not None:
                        self.env.set_value(variable_key, output)
                    return output
        # Inner recurrence dims: isolate as outer loop (iterate in order). In the body they are literals; vectorize other dims.
        if inner_recurrence_dims:
            from ..runtime.compute.lowered_execution import execute_lowered_loops
            inner_loops = [loops[d] for d in inner_recurrence_dims]
            with self._vectorization_scope(recurrence_clause=True):
                for inner_ctx in execute_lowered_loops(inner_loops, {}, expr_eval):
                    with self.env.scope():
                        self.env.set_value(variable_defid, output)
                        self._bind_outer_recurrence_context(outer_rec_defids, rec_context)
                        for dim in range(ndim):
                            defid, (start, end), name = loop_info[dim]
                            if dim in outer_dims_set:
                                outer_defid = self._outer_recurrence_defid_for_dim(outer_rec_defids, dim)
                                scalar_value = rec_context.get(outer_defid) if outer_defid in rec_context else None
                                self._set_loop_value_or_range(defid, (start, end), name, scalar_value=scalar_value)
                            elif dim in inner_recurrence_dims:
                                self._set_loop_value_or_range(
                                    defid,
                                    (start, end),
                                    name,
                                    scalar_value=inner_ctx.get(defid),
                                )
                        vector_dims_list = [
                            d
                            for d in range(ndim)
                            if d not in outer_dims_set and d not in inner_recurrence_dims
                        ]
                        vec_rank = len(vector_dims_list)
                        for vec_idx, vdim in enumerate(vector_dims_list):
                            vdefid, (vstart, vend), vname = loop_info[vdim]
                            self._set_loop_value_or_range(
                                vdefid,
                                (vstart, vend),
                                vname,
                                reshape_rank=vec_rank,
                                reshape_dim=vec_idx,
                            )
                        bindings = item.bindings or []
                        self._apply_lowered_bindings(bindings, loops, expr_eval, execute_lowered_bindings)
                        par_shape_inner = tuple(
                            int(loop_info[d][1][1] - loop_info[d][1][0]) for d in vector_dims_list
                        )
                        with self._vectorization_scope(
                            parallel_shape=par_shape_inner,
                            parallel_defids_order=tuple(loop_info[d][0] for d in vector_dims_list),
                        ):
                            res = item.body.accept(self)
                    if res is None:
                        continue
                        res = np.asarray(res, dtype=output.dtype)
                    # Build slice: literals (outer + inner recurrence) as scalar indices; other dims as slice() for vectorized write.
                    # Use loop_pos to index loop_info (only advance when we consume a loop), so literal indices don't skip a dimension.
                    slice_list: List[Any] = []
                    loop_pos = 0
                    for pos in range(min(len(clause_indices), output.ndim)):
                        idx = clause_indices[pos] if pos < len(clause_indices) else None
                        if isinstance(idx, LiteralIR):
                            try:
                                slice_list.append(int(idx.value))
                            except (TypeError, ValueError):
                                break
                            continue
                        if loop_pos >= len(loop_info):
                            break
                        if loop_pos in outer_dims_set:
                            outer_defid = self._outer_recurrence_defid_for_dim(outer_rec_defids, loop_pos)
                            value = rec_context[outer_defid] if outer_defid is not None and outer_defid in rec_context else inner_ctx.get(loop_info[loop_pos][0], 0)
                            slice_list.append(self._coerce_scalar_index(value))
                        elif loop_pos in inner_recurrence_dims:
                            v = inner_ctx.get(loop_info[loop_pos][0], 0)
                            slice_list.append(self._coerce_scalar_index(v))
                        else:
                            _, (start, end), _ = loop_info[loop_pos]
                            slice_list.append(slice(int(start), int(end)))
                        loop_pos += 1
                    if self._assign_recurrence_step_result(
                        output,
                        slice_list,
                        res,
                        object_output=object_output,
                    ):
                        continue
                self.env.set_value(variable_key, output)
                return output
        fast_wave = _try_fast_2d_wave_step(
            item, output, variable_defid, rec_context, outer_rec_defids, loop_info, expr_eval,
        )
        if fast_wave is not None:
            if variable_key is not None:
                self.env.set_value(variable_key, output)
            return output
        if recurrence_dims and len(recurrence_dims) == ndim and clause_indices and len(clause_indices) == output.ndim:
            with self.env.scope():
                self.env.set_value(variable_defid, output)
                self._bind_outer_recurrence_context(outer_rec_defids, rec_context)
                for dim in range(ndim):
                    defid, (start, end), name = loop_info[dim]
                    outer_defid = self._outer_recurrence_defid_for_dim(outer_rec_defids, dim)
                    scalar_value = rec_context.get(outer_defid) if outer_defid in rec_context else None
                    self._set_loop_value_or_range(defid, (start, end), name, scalar_value=scalar_value)
                bindings = item.bindings or []
                self._apply_lowered_bindings(bindings, loops, expr_eval, execute_lowered_bindings)
                res = item.body.accept(self)
            slice_list: List[Any] = []
            loop_pos = 0
            for idx in clause_indices:
                literal_val = self._literal_index_value(idx)
                if literal_val is not None:
                    slice_list.append(literal_val)
                    continue
                if loop_pos < len(loops):
                    outer_defid = self._outer_recurrence_defid_for_dim(outer_rec_defids, loop_pos)
                    if outer_defid is not None and outer_defid in rec_context:
                        slice_list.append(self._coerce_scalar_index(rec_context[outer_defid]))
                    else:
                        slice_list.append(0)
                    loop_pos += 1
                else:
                    return None
            if self._assign_recurrence_step_result(
                output,
                slice_list,
                res,
                object_output=object_output,
            ):
                self.env.set_value(variable_key, output)
                return output
        # When body is a block (e.g. RNN recurrence with let + if), use scalar iteration over non-recurrence dims
        # so that reductions in the body see the same env as the clause's loop vars (avoids vectorization bugs).
        # Only when clause indices match loop count (no literal indices like LSTM state[t, slot, b, h]) so slice building is correct.
        non_rec_loops = [loops[i] for i in range(ndim) if i not in recurrence_dims]
        if (
            isinstance(item.body, BlockExpressionIR)
            and non_rec_loops
            and clause_indices
            and len(clause_indices) == len(loops)
            and not _einlang_recurrence_block_vectorized_binding_enabled()
        ):
            from ..runtime.compute.lowered_execution import execute_lowered_loops
            with self._vectorization_scope(recurrence_clause=True):
                self._recurrence_block_strategy_log("scalar-loop", item, variable_decl, rec_context)
                for inner_ctx in execute_lowered_loops(non_rec_loops, {}, expr_eval):
                    with self.env.scope():
                        self.env.set_value(variable_defid, output)
                        self._bind_outer_recurrence_context(outer_rec_defids, rec_context)
                        for dim in range(ndim):
                            defid, (start, end), name = loop_info[dim]
                            if dim in recurrence_dims:
                                outer_defid = self._outer_recurrence_defid_for_dim(outer_rec_defids, dim)
                                scalar_value = rec_context.get(outer_defid) if outer_defid in rec_context else None
                                self._set_loop_value_or_range(defid, (start, end), name, scalar_value=scalar_value)
                            else:
                                self._set_loop_value_or_range(
                                    defid,
                                    (start, end),
                                    name,
                                    scalar_value=inner_ctx.get(defid),
                                )
                        bindings = item.bindings or []
                        self._apply_lowered_bindings(bindings, loops, expr_eval, execute_lowered_bindings)
                        res = item.body.accept(self)
                    if object_output:
                        scalar = res
                    elif not isinstance(res, np.ndarray):
                        continue
                    else:
                        scalar = res.flat[0] if res.size == 1 else res
                    slice_list_scalar: List[Any] = []
                    for pos, idx in enumerate(clause_indices):
                        if pos >= len(loops):
                            break
                        if isinstance(idx, LiteralIR):
                            try:
                                slice_list_scalar.append(int(idx.value))
                            except (TypeError, ValueError):
                                break
                        elif pos in recurrence_dims:
                            outer_defid = self._outer_recurrence_defid_for_dim(outer_rec_defids, pos)
                            if outer_defid is not None and outer_defid in rec_context:
                                slice_list_scalar.append(self._coerce_scalar_index(rec_context[outer_defid]))
                            else:
                                slice_list_scalar.append(self._coerce_scalar_index(inner_ctx.get(loop_info[pos][0], 0)))
                        else:
                            slice_list_scalar.append(self._coerce_scalar_index(inner_ctx.get(loop_info[pos][0], 0)))
                    if len(slice_list_scalar) == output.ndim:
                        if object_output:
                            output[tuple(slice_list_scalar)] = scalar
                        else:
                            output[tuple(slice_list_scalar)] = np.asarray(scalar, dtype=output.dtype)
                self.env.set_value(variable_key, output)
                return output
        with self._vectorization_scope(recurrence_clause=True):
            # Keep matmul/einsum fast paths off in evaluate_lowered_reduction.
            # One timestep per call: only outer recurrence dims bound from rec_context are scalar; other
            # clause loops use the broadcast grid (not full output.shape, which includes all steps).
            scalar_clause_dims = set()
            for d in range(ndim):
                if d in outer_dims_set:
                    od = next((o for od_d, o in outer_rec_defids if od_d == d), None)
                    if od is not None and od in rec_context:
                        scalar_clause_dims.add(d)
            par_shape = tuple(
                int(loop_info[d][1][1] - loop_info[d][1][0])
                for d in range(ndim)
                if d not in scalar_clause_dims
            )
            with self._vectorization_scope(
                parallel_shape=par_shape,
                parallel_defids_order=tuple(loop_info[d][0] for d in range(ndim) if d not in scalar_clause_dims),
            ):
                with self.env.scope():
                    self.env.set_value(variable_defid, output)
                    # Block bodies log when IR is still BlockExpressionIR; braced `{ expr }` often lowers to
                    # a plain expression, so also log when this clause has non-recurrence loop dims (i,j,…):
                    # same broadcast-index path runs here (see _recurrence_block_strategy_log gates).
                    if isinstance(item.body, BlockExpressionIR) or (
                        _einlang_recurrence_block_vectorized_binding_enabled() and non_rec_loops
                    ):
                        self._recurrence_block_strategy_log("broadcast-binding", item, variable_decl, rec_context)
                    # Bind outer recurrence loop vars so body sees current timestep (body may reference outer defid).
                    self._bind_outer_recurrence_context(outer_rec_defids, rec_context)
                    for dim in range(ndim):
                        defid, (start, end), name = loop_info[dim]
                        if dim in recurrence_dims:
                            outer_defid = self._outer_recurrence_defid_for_dim(outer_rec_defids, dim)
                            scalar_value = rec_context.get(outer_defid) if outer_defid in rec_context else None
                            self._set_loop_value_or_range(
                                defid,
                                (start, end),
                                name,
                                scalar_value=scalar_value,
                                reshape_rank=None if scalar_value is not None else ndim,
                                reshape_dim=None if scalar_value is not None else dim,
                            )
                        else:
                            self._set_loop_value_or_range(
                                defid,
                                (start, end),
                                name,
                                reshape_rank=ndim,
                                reshape_dim=dim,
                            )
                    bindings = item.bindings or []
                    self._apply_lowered_bindings(bindings, loops, expr_eval, execute_lowered_bindings)
                    result = item.body.accept(self)
            if object_output:
                if result is None:
                    return None
            else:
                result = np.asarray(result) if result is not None else None
            if not object_output and result is not None and not isinstance(result, np.ndarray):
                result = np.asarray(result)
            if result is None or (not object_output and not isinstance(result, np.ndarray)):
                return None
            # Build output slice from clause indices (same length as output.ndim; literals and recurrence bound).
            if not clause_indices or len(clause_indices) != output.ndim:
                return None
            slice_list: List[Any] = []
            loop_pos = 0
            for idx in clause_indices:
                # Literal or constant (e.g. 10, 0 in u[t, i, 10, 0]) so we don't consume a loop.
                literal_val = self._literal_index_value(idx)
                if literal_val is not None:
                    slice_list.append(literal_val)
                    continue
                if loop_pos < len(loops):
                    if loop_pos in recurrence_dims:
                        outer_defid = self._outer_recurrence_defid_for_dim(outer_rec_defids, loop_pos)
                        if outer_defid is not None and outer_defid in rec_context:
                            slice_list.append(self._coerce_scalar_index(rec_context[outer_defid]))
                        else:
                            start, end = loop_info[loop_pos][1]
                            slice_list.append(slice(int(start), int(end)))
                    else:
                        start, end = loop_info[loop_pos][1]
                        slice_list.append(slice(int(start), int(end)))
                    loop_pos += 1
                else:
                    return None
            if loop_pos != len(loops):
                return None
            if object_output:
                output[tuple(slice_list)] = result
            else:
                try:
                    n_outer = len(recurrence_loops_outer)
                    if result.ndim == ndim:
                        squeezed = np.squeeze(result, axis=tuple(range(n_outer)))
                    elif result.ndim == ndim - n_outer:
                        squeezed = result
                    else:
                        axes = [d for d in range(n_outer) if d < result.ndim]
                        squeezed = np.squeeze(result, axis=tuple(axes)) if axes else result
                except ValueError:
                    return None
                output[tuple(slice_list)] = squeezed.astype(output.dtype)
            return output

    def _shape_from_all_items(self, items: List) -> Optional[List[int]]:
        """Compute output shape from the max absolute end across ALL items (not just the first).

        All non-empty items must use the same number of loops; otherwise return None so the
        compiler-provided shape is used (mixed loop ranks would mis-align axes, e.g. ``[i,j]``
        vs ``[t,i,j]``).  Autodiff softmax-style cases with a too-small buffer are handled in
        ``_execute_lowered_einstein_clause`` / result adoption, not by cross-rank union here.
        """
        if not items:
            return None
        rank = None
        max_ends: Optional[List[int]] = None
        body_suffix: Optional[List[int]] = None
        for item in items:
            loops = item.loops or []
            if not loops:
                continue
            if rank is None:
                rank = len(loops)
                max_ends = [0] * rank
            if len(loops) != rank:
                return None
            candidate_suffix = None
            body_shape = getattr(item.body, "shape_info", None)
            if not body_shape:
                body_type = getattr(item.body, "type_info", None)
                body_shape = getattr(body_type, "shape", None)
            if isinstance(body_shape, (list, tuple)):
                concrete_suffix: List[int] = []
                for dim in body_shape:
                    if isinstance(dim, (int, np.integer)):
                        concrete_suffix.append(int(dim))
                    elif isinstance(dim, LiteralIR) and isinstance(dim.value, (int, float)):
                        concrete_suffix.append(int(dim.value))
                    else:
                        concrete_suffix = []
                        break
                if concrete_suffix:
                    candidate_suffix = concrete_suffix
            if candidate_suffix is not None:
                if body_suffix is None:
                    body_suffix = candidate_suffix
                elif body_suffix != candidate_suffix:
                    return None
            for d, loop in enumerate(loops):
                it = loop.iterable
                if it is None:
                    return None
                if isinstance(it, LiteralIR) and isinstance(it.value, range):
                    end = int(it.value.stop)
                elif isinstance(it, RangeIR):
                    try:
                        end = int(it.end.accept(self))
                    except (TypeError, ValueError):
                        return None
                else:
                    try:
                        r = it.accept(self)
                        end = len(r) if hasattr(r, "__len__") else None
                    except Exception:
                        return None
                    if end is None:
                        return None
                if end > max_ends[d]:
                    max_ends[d] = end
        if max_ends is None:
            return body_suffix
        return max_ends + (body_suffix or [])

    def _clause_set_output(self, variable_defid: Any, output: Any) -> None:
        """Set clause result in env."""
        if variable_defid is not None:
            self.env.set_value(variable_defid, output)
