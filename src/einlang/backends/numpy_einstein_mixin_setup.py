"""NumPy backend Einstein execution mixin."""

from .numpy_einstein_vectorization import *
from .numpy_einstein_analysis import (
    _TYPE_NAME_TO_NUMPY_DTYPE,
    _allocate_numpy_output,
    _einlang_debug_recurrence_block_enabled,
    _einlang_vectorize_debug_detail_enabled,
    _evaluate_shape_dim,
    _lowered_clause_loop_axis_names,
    _BodyReferencesDefidVisitor,
)
from .numpy_einstein_recurrence_analysis import (
    _extract_loop_range,
    _infer_lowered_einstein_output_defid,
    _recurrence_dims,
    _recurrence_dims_for_hybrid,
    _slice_list_from_clause_indices,
)

class EinsteinExecutionSetupMixin:
    def _vectorize_debug_log(
        self,
        path: str,
        lowered: Any,
        variable_decl: Any,
        *,
        axes: str = "",
    ) -> None:
        if not _einlang_vectorize_debug_detail_enabled():
            return
        loc = getattr(lowered, "location", None)
        if loc is None and variable_decl is not None:
            loc = getattr(variable_decl, "location", None)
        line = int(getattr(loc, "line", 0) or 0)
        binder = getattr(variable_decl, "_binding", None) or variable_decl
        name = getattr(binder, "name", None) or getattr(variable_decl, "name", None) or "?"
        loop_s = _lowered_clause_loop_axis_names(lowered)
        loop_part = f" loops={loop_s}" if loop_s else ""
        axes_part = f" | {axes}" if axes else ""
        print(f"[vectorize] detail {path} L{line} {name}{loop_part}{axes_part}", flush=True)

    def _recurrence_block_strategy_log(
        self,
        strategy: str,
        item: Any,
        variable_decl: Any,
        rec_context: Optional[Dict[Any, Any]] = None,
    ) -> None:
        """``[recurrence-block]`` lines: broadcast-binding when verbose vectorize or DEBUG_RECURRENCE_BLOCK; scalar-loop when DEBUG_RECURRENCE_BLOCK only."""
        if strategy == "scalar-loop":
            if not _einlang_debug_recurrence_block_enabled():
                return
        elif strategy == "broadcast-binding":
            if not (
                _einlang_vectorize_debug_detail_enabled()
                or _einlang_debug_recurrence_block_enabled()
            ):
                return
        else:
            return
        loc = getattr(item, "location", None)
        if loc is None and variable_decl is not None:
            loc = getattr(variable_decl, "location", None)
        line = int(getattr(loc, "line", 0) or 0)
        binder = getattr(variable_decl, "_binding", None) or variable_decl
        name = getattr(binder, "name", None) or getattr(variable_decl, "name", None) or "?"
        step_s = "?"
        if rec_context:
            try:
                first = next(iter(rec_context.values()))
                step_s = str(int(first) if isinstance(first, (np.integer, int)) else int(np.asarray(first).item()))
            except (StopIteration, TypeError, ValueError):
                step_s = "?"
        print(f"[recurrence-block] {strategy} L{line} {name} step={step_s}", flush=True)

    def _evaluate_lowered_einstein_subexpr(self, lowered: LoweredEinsteinIR) -> Any:
        """Run a nested LoweredEinsteinIR (e.g. under RectangularAccessIR from autodiff) with a synthetic decl.

        Prefer the inferred output DefId for recurrence (``u[t-1]`` reads) so execution
        reuses ``u``'s storage. Use a fresh DefId when inference finds only diagonal
        input reads (``x[i]``) so pullback output does not alias ``x``.
        """
        out_defid = _infer_lowered_einstein_output_defid(lowered)
        if out_defid is None:
            seq = getattr(self, "_nested_einstein_synth_seq", 0) + 1
            self._nested_einstein_synth_seq = seq
            out_defid = DefId(RUNTIME_CRATE, seq)

        class _SyntheticEinsteinDecl:
            __slots__ = ("defid", "name")

            def __init__(self, defid: Any) -> None:
                self.defid = defid
                self.name = "?"

        decl = _SyntheticEinsteinDecl(out_defid)
        stack = getattr(self, "_variable_decl_stack", None)
        if stack is None:
            self._variable_decl_stack = []
            stack = self._variable_decl_stack
        stack.append(decl)
        try:
            return lowered.accept(self)
        finally:
            stack.pop()

    def visit_lowered_einstein_clause(self, node: LoweredEinsteinClauseIR) -> Any:
        stack = getattr(self, "_variable_decl_stack", None)
        variable_decl = stack[-1] if stack else None
        return self._execute_lowered_einstein_clause(node, variable_decl)

    def visit_lowered_einstein(self, node: LoweredEinsteinIR) -> Any:
        stack = getattr(self, "_variable_decl_stack", None)
        variable_decl = stack[-1] if stack else None
        return self._execute_lowered_einstein(node, variable_decl)

    def visit_lowered_recurrence(self, node: LoweredRecurrenceIR) -> Any:
        """Execute recurrence isolated out of Einstein: run initial once, then for each recurrence_loop value run body clauses."""
        stack = getattr(self, "_variable_decl_stack", None)
        variable_decl = stack[-1] if stack else None
        if variable_decl is None:
            raise RuntimeError("LoweredRecurrenceIR executed without variable_decl on stack")
        from ..runtime.compute.lowered_execution import execute_lowered_loops
        output = self._execute_lowered_einstein(node.initial, variable_decl)
        binding = getattr(variable_decl, "_binding", None) or variable_decl
        variable_key = binding.defid or getattr(variable_decl, "defid", None)
        variable_defid = variable_key
        tensor_shape = list(output.shape) if output is not None else None
        tensor_element_type = node.initial.element_type or None

        def expr_eval(e: Any) -> Any:
            return e.accept(self)

        recurrence_loops_for_outer = [node.recurrence_loop]
        _MAX = int(DEFAULT_EINSTEIN_LOOP_MAX)
        n_iter = [0]
        first_ctx_iter = execute_lowered_loops(recurrence_loops_for_outer, {}, expr_eval)
        first_ctx = next(first_ctx_iter, None)
        step_items: List[Any] = []
        if first_ctx is not None:
            for item in node.body.items:
                step_ok = self._execute_lowered_einstein_clause_one_recurrence_step(
                    item, variable_decl, output, variable_key, variable_defid,
                    first_ctx, recurrence_loops_for_outer, expr_eval, tensor_shape, tensor_element_type,
                )
                if step_ok is not None and variable_key is not None:
                    self.env.set_value(variable_key, output)
                    step_items.append(item)
                else:
                    self.env.set_value(variable_key, output)
                    full_result = self._execute_lowered_einstein_clause(
                        item, variable_decl,
                        shape=tensor_shape, element_type=tensor_element_type,
                        pre_allocated_output=None,
                    )
                    if full_result is not None and variable_key is not None:
                        clause_indices = item.indices or []
                        if clause_indices and len(clause_indices) == output.ndim:
                            slices_list = _slice_list_from_clause_indices(clause_indices, item, expr_eval) or []
                            if len(slices_list) == output.ndim:
                                if full_result.shape == output.shape:
                                    output[tuple(slices_list)] = full_result[tuple(slices_list)].astype(output.dtype)
                                else:
                                    output[tuple(slices_list)] = full_result.astype(output.dtype)
                                self.env.set_value(variable_key, output)
        else:
            step_items = list(node.body.items)
        for rec_context in first_ctx_iter:
            n_iter[0] += 1
            if n_iter[0] > _MAX:
                raise RuntimeError(
                    f"Einstein recurrence loop iterations exceeded limit ({_MAX}). "
                    "Reduce clause range or increase config.DEFAULT_EINSTEIN_LOOP_MAX."
                )
            for item in step_items:
                result = self._execute_lowered_einstein_clause_one_recurrence_step(
                    item, variable_decl, output, variable_key, variable_defid,
                    rec_context, recurrence_loops_for_outer, expr_eval, tensor_shape, tensor_element_type,
                )
                if result is not None and variable_key is not None:
                    self.env.set_value(variable_key, output)
        return output

    def _primitive_type_to_numpy_dtype(self, type_obj: Any) -> Optional[Any]:
        if not isinstance(type_obj, PrimitiveType):
            return None
        return _TYPE_NAME_TO_NUMPY_DTYPE.get(type_obj)

    def _type_info_to_numpy_dtype(self, type_info: Any) -> Optional[Any]:
        if type_info is None:
            return None
        if isinstance(type_info, Type) and hasattr(type_info, "kind") and type_info.kind == TypeKind.UNKNOWN:
            return None
        if isinstance(type_info, PrimitiveType):
            return self._primitive_type_to_numpy_dtype(type_info)
        if isinstance(type_info, TupleType):
            return object
        if isinstance(type_info, RectangularType):
            return self._type_info_to_numpy_dtype(type_info.element_type)
        return None

    def _dtype_for_clause_result(self, clause_body: Any, tensor_element_type: Any) -> Any:
        """Dtype from type pass only: tensor_element_type, then clause body type_info."""
        dtype = self._type_info_to_numpy_dtype(tensor_element_type)
        if dtype is not None:
            return dtype
        if clause_body is None:
            return np.int32
        type_info = clause_body.type_info
        if type_info is not None:
            dtype = self._type_info_to_numpy_dtype(type_info)
            if dtype is not None:
                return dtype
        if isinstance(clause_body, (LoweredReductionIR, ReductionExpressionIR)):
            body_expr = clause_body.body
            if body_expr is not None:
                ti = body_expr.type_info
                if ti is not None:
                    dtype = self._type_info_to_numpy_dtype(ti)
                    if dtype is not None:
                        return dtype
        return np.int32

    def _get_defid_for_pattern_var(self, var_name: str, pattern: Any) -> Optional[DefId]:
        if hasattr(pattern, "name") and pattern.name == var_name:
            return pattern.defid
        if hasattr(pattern, "inner_pattern"):
            return self._get_defid_for_pattern_var(var_name, pattern.inner_pattern)
        if hasattr(pattern, "patterns"):
            for nested in pattern.patterns:
                defid = self._get_defid_for_pattern_var(var_name, nested)
                if defid is not None:
                    return defid
        return None

    def _execute_lowered_einstein(self, lowered_einstein: LoweredEinsteinIR, variable_decl: Any) -> Any:
        from ..runtime.compute.lowered_execution import execute_lowered_loops, execute_lowered_bindings, check_lowered_guards
        binding = getattr(variable_decl, "_binding", None)
        variable_key = (binding.defid if binding else None) or getattr(variable_decl, "defid", None)
        tensor_shape = lowered_einstein.shape
        tensor_element_type = lowered_einstein.element_type
        output_shape = None
        if tensor_shape:
            output_shape = []
            for shape_dim in tensor_shape:
                try:
                    dim_value = _evaluate_shape_dim(shape_dim, self)
                except RuntimeError:
                    # Symbolic shape may reference vars not in env during pullback (e.g. len(x[0])); infer from clauses.
                    output_shape = None
                    break
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
        if not output_shape and lowered_einstein.items:
            output_shape = self._shape_from_all_items(lowered_einstein.items)
        elif output_shape and lowered_einstein.items:
            # Compiler shape may underestimate for multi-segment declarations
            # (e.g. _compute_shape_union picks a symbolic expr that evaluates to
            # the first clause's end, not the union).  Widen only when loop ranks match
            # across items — clauses with different loop *counts* (e.g. [i,j] vs [t,i,j])
            # must not be merged dimension-wise (would corrupt axes).
            items_shape = self._shape_from_all_items(lowered_einstein.items)
            if items_shape and len(items_shape) == len(output_shape):
                output_shape = [max(a, b) for a, b in zip(output_shape, items_shape)]
        if not output_shape and lowered_einstein.items:
            raise RuntimeError(
                "Einstein declaration has no shape from compiler. "
                "Compiler must set shape (union of clause ranges) on LoweredEinsteinIR."
            )
        if not output_shape:
            output_shape = [1]
        dtype = self._type_info_to_numpy_dtype(tensor_element_type)
        if dtype is None:
            dtype = np.int32

        # Multi-segment: reuse existing array if this variable was already
        # declared (e.g. pad's `let result[i in 0..p] = ...; let result[i in p..n] = ...;`)
        existing = self.env.get_value(variable_key) if variable_key is not None else None
        if existing is not None and isinstance(existing, np.ndarray):
            needed = tuple(output_shape)
            current = existing.shape
            if len(needed) == len(current) and any(n > c for n, c in zip(needed, current)):
                new_shape = tuple(max(n, c) for n, c in zip(needed, current))
                output = _allocate_numpy_output(list(new_shape), existing.dtype)
                slices = tuple(slice(0, s) for s in current)
                output[slices] = existing
                self.env.set_value(variable_key, output)
            else:
                output = existing
        else:
            output = _allocate_numpy_output(output_shape, dtype)
            if variable_key is not None:
                self.env.set_value(variable_key, output)

        def expr_eval(e: Any) -> Any:
            return e.accept(self)

        items = lowered_einstein.items
        binding = getattr(variable_decl, "_binding", None)
        variable_defid = (binding.defid if binding else None) or getattr(variable_decl, "defid", None)

        # Execution order (mental model): (1) run all non-recurrence clauses in source order; (2) for each value of
        # the recurrence dimension (outermost), run all recurrence clauses in source order. So: recurrence dim
        # outermost, all clauses that write at that index run inside, preserving .ein clause order.
        # Partition: recurrence = any backward-in-time (t-1) or same-timestep (state[t,0] when writing state[t,1]).
        # Use _recurrence_dims so we accept both; clauses like LSTM hidden (reads t-1 and t) stay in recurrence_items and run after cell.
        recurrence_items: List[Any] = []
        non_recurrence_items: List[Any] = []
        recurrence_loops_for_outer: Optional[List[Any]] = None
        if len(items) > 1 and variable_defid is not None:
            for it in items:
                clause_indices = (it.indices or [])
                loops_it = (it.loops or [])
                rec_dims = it.recurrence_dims_override
                if rec_dims is None:
                    rec_dims = _recurrence_dims_for_hybrid(it, variable_defid, clause_indices)
                if not rec_dims:
                    rec_dims = _recurrence_dims(it, variable_defid, clause_indices)
                body_refs = _BodyReferencesDefidVisitor(variable_defid).references(it.body)
                # Recurrence = has recurrence dim(s). Allow pure recurrence (only t) so t is extracted as outer loop.
                # When len(rec_dims) < len(loops_it) we vectorize over the rest; when equal we run one scalar/point per t.
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
        use_timestep_major = bool(recurrence_items and recurrence_loops_for_outer)

        # Run non-recurrence items first (e.g. state[0,...] initial conditions).
        for item in non_recurrence_items:
            result = self._execute_lowered_einstein_clause(
                item, variable_decl,
                shape=tensor_shape, element_type=tensor_element_type,
                pre_allocated_output=output,
            )
            if result is not None and variable_key is not None:
                if result.shape == output.shape:
                    if result is not output:
                        output[:] = result.astype(output.dtype)
                elif result.shape != output.shape:
                    # Autodiff pullback only: 1D buffer (e.g. shape (B,)) vs Jacobian row (B, J, ...).
                    # Do not adopt for general primals — higher-rank clause results can be slice writes.
                    if (
                        result.ndim > output.ndim
                        and output.ndim == 1
                        and result.shape[0] == output.shape[0]
                        and result.size > output.size
                    ):
                        output = np.asarray(result, dtype=output.dtype)
                        self.env.set_value(variable_key, output)
                    else:
                        slices_list_nr: List[Any] = []
                        clause_indices = item.indices or []
                        if clause_indices and len(clause_indices) == output.ndim:
                            slices_list_nr = _slice_list_from_clause_indices(clause_indices, item, expr_eval) or []
                        if len(slices_list_nr) != output.ndim and item.loops:
                            try:
                                for lp in item.loops:
                                    start, end = _extract_loop_range(lp, expr_eval)
                                    slices_list_nr.append(slice(int(start), int(end)))
                            except RuntimeError:
                                slices_list_nr = []
                        if len(slices_list_nr) == output.ndim:
                            output[tuple(slices_list_nr)] = result.astype(output.dtype)
                        elif result.size == 1 and item.indices and all(
                            isinstance(idx, LiteralIR) for idx in item.indices
                        ):
                            idx_tuple = tuple(int(idx.value) for idx in item.indices)
                            output[idx_tuple] = result.flat[0] if result.size == 1 else result
                self.env.set_value(variable_key, output)

        # When we have recurrence items: run recurrence dim outermost, all recurrence clauses inside (timestep-major).
        if use_timestep_major and recurrence_loops_for_outer:
            from ..runtime.compute.lowered_execution import execute_lowered_loops
            _MAX = int(DEFAULT_EINSTEIN_LOOP_MAX)
            n_iter = [0]
            try:
                # Items whose step returns None: run full clause once and write slice (e.g. clauses with literal indices).
                step_items: List[Any] = []
                first_ctx_iter = execute_lowered_loops(recurrence_loops_for_outer, {}, expr_eval)
                first_ctx = next(first_ctx_iter, None)
                if first_ctx is not None:
                    for idx, item in enumerate(recurrence_items):
                        step_ok = self._execute_lowered_einstein_clause_one_recurrence_step(
                            item, variable_decl, output, variable_key, variable_defid,
                            first_ctx, recurrence_loops_for_outer, expr_eval, tensor_shape, tensor_element_type,
                        )
                        if step_ok is not None and variable_key is not None:
                            self.env.set_value(variable_key, output)
                            step_items.append(item)
                        else:
                            # Run full clause using existing output so body reads current state (e.g. state[t,0] for hidden).
                            self.env.set_value(variable_key, output)
                            full_result = self._execute_lowered_einstein_clause(
                                item, variable_decl,
                                shape=tensor_shape, element_type=tensor_element_type,
                                pre_allocated_output=output,
                            )
                            if full_result is not None and variable_key is not None:
                                clause_indices = item.indices or []
                                if clause_indices and len(clause_indices) == output.ndim:
                                    slices_list = _slice_list_from_clause_indices(clause_indices, item, expr_eval) or []
                                    if len(slices_list) == output.ndim:
                                        if full_result.shape == output.shape:
                                            output[tuple(slices_list)] = full_result[tuple(slices_list)].astype(output.dtype)
                                        else:
                                            output[tuple(slices_list)] = full_result.astype(output.dtype)
                                        self.env.set_value(variable_key, output)
                else:
                    step_items = list(recurrence_items)
                rec_loop = first_ctx_iter if first_ctx is not None else execute_lowered_loops(recurrence_loops_for_outer, {}, expr_eval)
                for rec_context in rec_loop:
                    n_iter[0] += 1
                    if n_iter[0] > _MAX:
                        raise RuntimeError(
                            f"Einstein recurrence loop iterations exceeded limit ({_MAX}). "
                            "Reduce clause range or increase config.DEFAULT_EINSTEIN_LOOP_MAX."
                        )
                    for item in step_items:
                        result = self._execute_lowered_einstein_clause_one_recurrence_step(
                            item, variable_decl, output, variable_key, variable_defid,
                            rec_context, recurrence_loops_for_outer, expr_eval, tensor_shape, tensor_element_type,
                        )
                        if result is not None and variable_key is not None:
                            self.env.set_value(variable_key, output)
                return output
            except Exception:
                use_timestep_major = False
        # Recurrence items not run in timestep-major (fallback or single-clause): run them in clause order.
        run_in_clause_order = recurrence_items if recurrence_items else items
        for clause_idx, item in enumerate(run_in_clause_order):
            result = self._execute_lowered_einstein_clause(
                item, variable_decl,
                shape=tensor_shape, element_type=tensor_element_type,
                pre_allocated_output=output,
            )
            if result is not None and variable_key is not None:
                if result.shape == output.shape:
                    if result is not output:
                        output[:] = result.astype(output.dtype)
                elif result.shape != output.shape:
                    if (
                        result.ndim > output.ndim
                        and output.ndim == 1
                        and result.shape[0] == output.shape[0]
                        and result.size > output.size
                    ):
                        output = np.asarray(result, dtype=output.dtype)
                        self.env.set_value(variable_key, output)
                    else:
                        slices_list: List[Any] = []
                        clause_indices = item.indices or []
                        if clause_indices and len(clause_indices) == output.ndim:
                            slices_list = _slice_list_from_clause_indices(clause_indices, item, expr_eval) or []
                        if len(slices_list) != output.ndim and item.loops:
                            slices_list = []
                            try:
                                for lp in item.loops:
                                    start, end = _extract_loop_range(lp, expr_eval)
                                    slices_list.append(slice(int(start), int(end)))
                            except RuntimeError:
                                slices_list = []
                        if len(slices_list) == output.ndim:
                            output[tuple(slices_list)] = result.astype(output.dtype)
                        elif result.size == 1 and item.indices and all(
                            isinstance(idx, LiteralIR) for idx in item.indices
                        ):
                            idx_tuple = tuple(int(idx.value) for idx in item.indices)
                            output[idx_tuple] = result.flat[0] if result.size == 1 else result
                self.env.set_value(variable_key, output)
        return output
