"""NumPy backend Einstein execution mixin."""

from typing import Callable

from ..ir.nodes import BlockExpressionIR
from ..shared.defid import DefId

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
    _slice_list_from_clause_indices,
)


class _CircularRecurrenceBuffer:
    """Logical full tensor backed by a ring buffer on one recurrence dimension."""

    is_circular_recurrence_buffer = True

    def __init__(
        self,
        full_shape: List[int],
        dtype: Any,
        recurrence_dim: int,
        preserve_steps: int,
        materializer: Optional[Callable[[], np.ndarray]] = None,
    ) -> None:
        self._full_shape = tuple(int(v) for v in full_shape)
        self._recurrence_dim = int(recurrence_dim)
        self._preserve_steps = int(preserve_steps)
        self._materializer = materializer
        self._materialized: Optional[np.ndarray] = None
        buffer_shape = list(self._full_shape)
        buffer_shape[self._recurrence_dim] = self._preserve_steps
        self._buffer = _allocate_numpy_output(buffer_shape, dtype)
        self.shape = self._full_shape
        self.ndim = len(self._full_shape)
        self.dtype = self._buffer.dtype

    def is_materialized(self) -> bool:
        return self._materialized is not None

    def materialize(self) -> np.ndarray:
        if self._materialized is None:
            if self._materializer is None:
                raise RuntimeError("circular recurrence buffer cannot materialize without a callback")
            self._materialized = np.asarray(self._materializer())
        return self._materialized

    def _expand_key(self, key: Any) -> tuple:
        if not isinstance(key, tuple):
            key = (key,)
        items = list(key)
        if Ellipsis in items:
            ell_idx = items.index(Ellipsis)
            missing = self.ndim - (len(items) - 1)
            items = items[:ell_idx] + [slice(None)] * missing + items[ell_idx + 1 :]
        if len(items) < self.ndim:
            items.extend([slice(None)] * (self.ndim - len(items)))
        return tuple(items[: self.ndim])

    def _translate_recurrence_index(self, idx: Any) -> Any:
        if isinstance(idx, (np.integer, int)):
            value = int(idx)
            if value < 0 or value >= self._full_shape[self._recurrence_dim]:
                raise IndexError("recurrence index out of ring-buffer bounds")
            return value % self._preserve_steps
        if isinstance(idx, slice):
            start, stop, step = idx.indices(self._full_shape[self._recurrence_dim])
            arr = np.arange(start, stop, step, dtype=np.intp)
            if arr.size > self._preserve_steps:
                raise RuntimeError("slice exceeds preserved recurrence window")
            if arr.size == 0:
                return slice(0, 0, 1)
            mapped = arr % self._preserve_steps
            if arr.size > 1 and np.all(np.diff(mapped) == 1):
                return slice(int(mapped[0]), int(mapped[-1]) + 1, 1)
            return mapped
        if isinstance(idx, np.ndarray):
            arr = np.asarray(idx, dtype=np.intp)
            if np.any(arr < 0) or np.any(arr >= self._full_shape[self._recurrence_dim]):
                raise IndexError("recurrence index array out of ring-buffer bounds")
            if arr.size > self._preserve_steps:
                raise RuntimeError("index array exceeds preserved recurrence window")
            return arr % self._preserve_steps
        if isinstance(idx, list):
            return self._translate_recurrence_index(np.asarray(idx, dtype=np.intp))
        raise RuntimeError(f"unsupported recurrence index type: {type(idx).__name__}")

    def _translated_key(self, key: Any) -> tuple:
        expanded = list(self._expand_key(key))
        expanded[self._recurrence_dim] = self._translate_recurrence_index(expanded[self._recurrence_dim])
        return tuple(expanded)

    def __getitem__(self, key: Any) -> Any:
        if self._materialized is not None:
            return self._materialized[key]
        try:
            return self._buffer[self._translated_key(key)]
        except Exception:
            return self.materialize()[key]

    def __setitem__(self, key: Any, value: Any) -> None:
        if self._materialized is not None:
            self._materialized[key] = value
            return
        try:
            self._buffer[self._translated_key(key)] = value
        except Exception:
            self.materialize()[key] = value

    def __array__(self, dtype: Optional[Any] = None) -> np.ndarray:
        arr = self.materialize()
        if dtype is None:
            return arr
        return np.asarray(arr, dtype=dtype)


class EinsteinExecutionSetupMixin:
    def _snapshot_visible_env(self) -> Dict[Any, Any]:
        snapshot: Dict[Any, Any] = {}
        for scope in getattr(self.env, "_scope_stack", []) or []:
            snapshot.update(scope)
        return snapshot

    def _materialize_recurrence_with_snapshot(
        self,
        node: LoweredRecurrenceIR,
        variable_decl: Any,
        snapshot: Dict[Any, Any],
    ) -> np.ndarray:
        with self.env.scope():
            for defid, value in snapshot.items():
                if defid is not None:
                    self.env.set_value(defid, value)
            return np.asarray(self._execute_lowered_recurrence_full(node, variable_decl))

    def _resolve_lowered_output_shape(self, lowered_einstein: LoweredEinsteinIR) -> Optional[List[int]]:
        output_shape = None
        tensor_shape = lowered_einstein.shape
        if tensor_shape:
            output_shape = []
            for shape_dim in tensor_shape:
                try:
                    dim_value = _evaluate_shape_dim(shape_dim, self)
                except RuntimeError:
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
            items_shape = self._shape_from_all_items(lowered_einstein.items)
            if items_shape and len(items_shape) == len(output_shape):
                output_shape = [max(a, b) for a, b in zip(output_shape, items_shape)]
        return output_shape

    def _execute_lowered_recurrence_full(self, node: LoweredRecurrenceIR, variable_decl: Any) -> Any:
        """Existing full-history recurrence execution path."""
        output = self._execute_lowered_einstein(node.initial, variable_decl)
        binding = getattr(variable_decl, "_binding", None) or variable_decl
        variable_key = binding.defid or getattr(variable_decl, "defid", None)
        variable_defid = variable_key
        tensor_shape = list(output.shape) if output is not None else None
        tensor_element_type = node.initial.element_type or None

        def expr_eval(e: Any) -> Any:
            return e.accept(self)

        recurrence_loops_for_outer = [node.recurrence_loop]
        outer_loop_defid = node.recurrence_loop.variable.defid
        recurrence_iterable = expr_eval(node.recurrence_loop.iterable)
        _MAX = int(DEFAULT_EINSTEIN_LOOP_MAX)
        n_iter = [0]
        if outer_loop_defid is None or recurrence_iterable is None:
            first_ctx_iter = iter(())
        else:
            def _iter_outer_contexts() -> Any:
                rec_context = {}
                for value in recurrence_iterable:
                    rec_context[outer_loop_defid] = value
                    yield rec_context
            first_ctx_iter = _iter_outer_contexts()
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

    def _maybe_execute_lowered_recurrence_circular(self, node: LoweredRecurrenceIR, variable_decl: Any) -> Optional[Any]:
        preserve_steps = getattr(node, "preserve_steps", None)
        recurrence_output_dim = getattr(node, "recurrence_output_dim", None)
        if (
            getattr(node, "requires_full_output", False)
            or preserve_steps is None
            or recurrence_output_dim is None
            or preserve_steps <= 0
        ):
            return None
        body_items = list(node.body.items or [])
        if len(body_items) != 1:
            return None
        item0 = body_items[0]
        if len(item0.loops or []) != 1 or item0.bindings or item0.guards:
            return None
        # Block-bodied recurrence steps can contain nested reductions that still rely on
        # outer timestep bindings staying scalar in the ambient env. The circular-buffer
        # optimized path currently routes those through the broadcast-binding executor,
        # which is not yet robust for these block+reduction combinations.
        if any(isinstance(getattr(item, "body", None), BlockExpressionIR) for item in body_items):
            return None
        shape = self._resolve_lowered_output_shape(node.initial or node.body)
        if not shape or recurrence_output_dim >= len(shape):
            return None
        full_extent = int(shape[recurrence_output_dim])
        if preserve_steps >= full_extent:
            return None
        tensor_element_type = node.initial.element_type or node.body.element_type
        dtype = self._type_info_to_numpy_dtype(tensor_element_type) or np.int32
        binding = getattr(variable_decl, "_binding", None) or variable_decl
        variable_key = binding.defid or getattr(variable_decl, "defid", None)
        variable_defid = variable_key
        env_snapshot = self._snapshot_visible_env()

        ring = _CircularRecurrenceBuffer(
            shape,
            dtype,
            recurrence_output_dim,
            preserve_steps,
            materializer=lambda: self._materialize_recurrence_with_snapshot(
                node,
                variable_decl,
                env_snapshot,
            ),
        )
        if variable_key is not None:
            self.env.set_value(variable_key, ring)

        def expr_eval(e: Any) -> Any:
            return e.accept(self)

        recurrence_loops_for_outer = [node.recurrence_loop]
        outer_loop_defid = node.recurrence_loop.variable.defid
        recurrence_iterable = expr_eval(node.recurrence_loop.iterable)
        _MAX = int(DEFAULT_EINSTEIN_LOOP_MAX)
        n_iter = [0]

        for item in node.initial.items or []:
            result = self._execute_lowered_einstein_clause(
                item,
                variable_decl,
                shape=shape,
                element_type=tensor_element_type,
                pre_allocated_output=ring,
            )
            if ring.is_materialized():
                return ring.materialize()
            if result is not None and variable_key is not None:
                self.env.set_value(variable_key, ring)

        if outer_loop_defid is None or recurrence_iterable is None:
            first_ctx_iter = iter(())
        else:
            def _iter_outer_contexts() -> Any:
                rec_context = {}
                for value in recurrence_iterable:
                    rec_context[outer_loop_defid] = value
                    yield rec_context
            first_ctx_iter = _iter_outer_contexts()

        first_ctx = next(first_ctx_iter, None)
        step_items: List[Any] = []
        if first_ctx is not None:
            for item in node.body.items:
                step_ok = self._execute_lowered_einstein_clause_one_recurrence_step(
                    item, variable_decl, ring, variable_key, variable_defid,
                    first_ctx, recurrence_loops_for_outer, expr_eval, shape, tensor_element_type,
                )
                if ring.is_materialized():
                    return ring.materialize()
                if step_ok is None:
                    return ring.materialize()
                if variable_key is not None:
                    self.env.set_value(variable_key, ring)
                step_items.append(item)
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
                    item, variable_decl, ring, variable_key, variable_defid,
                    rec_context, recurrence_loops_for_outer, expr_eval, shape, tensor_element_type,
                )
                if ring.is_materialized():
                    return ring.materialize()
                if result is None:
                    return ring.materialize()
                if variable_key is not None:
                    self.env.set_value(variable_key, ring)
        return ring

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

    def _evaluate_lowered_einstein_subexpr(
        self,
        lowered: LoweredEinsteinIR,
        *,
        allow_outer_slot_eval: bool = True,
        scalar_index_bindings: Optional[Dict[Any, Any]] = None,
    ) -> Any:
        """Run a nested LoweredEinsteinIR (e.g. under RectangularAccessIR from autodiff) with a synthetic decl.

        Prefer the inferred output DefId for recurrence (``u[t-1]`` reads) so execution
        reuses ``u``'s storage. Use a fresh DefId when inference finds only diagonal
        input reads (``x[i]``) so pullback output does not alias ``x``.
        """
        out_defid = _infer_lowered_einstein_output_defid(lowered)
        if out_defid is None:
            raise RuntimeError(
                "Nested LoweredEinsteinIR is missing a compile-time output DefId. "
                "Compiler lowering must annotate the storage target instead of synthesizing it at runtime."
            )

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
            from .numpy_einstein_call_index_analysis import _collect_defids_by_name

            body_name_cache = getattr(self, "_cached_defids_by_name", None)
            if callable(body_name_cache):
                defids_by_name = body_name_cache(lowered)
            else:
                defids_by_name = _collect_defids_by_name(lowered)

            captured_ctx = {}
            for dids in defids_by_name.values():
                for did in dids:
                    if did is None:
                        continue
                    cur = self.env.get_value(did)
                    if cur is not None:
                        captured_ctx[did] = cur
            cache = None
            cache_key = None
            if allow_outer_slot_eval and self._vectorization_parallel_shape() is not None:
                cache = getattr(self, "_nested_lowered_eval_cache", None)
            if cache is not None:
                def _sort_key(item: Tuple[Any, Any]) -> Tuple[int, int]:
                    did = item[0]
                    return (getattr(did, "krate", 0), getattr(did, "index", 0))

                captured_fingerprint = tuple(
                    (did, self._cache_value_fingerprint(val))
                    for did, val in sorted(captured_ctx.items(), key=_sort_key)
                )
                scalar_fingerprint = tuple(
                    (did, self._cache_value_fingerprint(val))
                    for did, val in sorted((scalar_index_bindings or {}).items(), key=_sort_key)
                )
                cache_key = (
                    id(lowered),
                    captured_fingerprint,
                    scalar_fingerprint,
                )
                if cache_key in cache:
                    return cache[cache_key]
            with self.env.scope():
                env_names = getattr(self.env, "_defid_names", {}) or {}
                for did, val in captured_ctx.items():
                    self.env.set_value(did, val, name=env_names.get(did))
                for did, val in (scalar_index_bindings or {}).items():
                    self.env.set_value(did, val, name=env_names.get(did))
                if allow_outer_slot_eval:
                    slot_eval = self._evaluate_lowered_per_outer_slot(lowered)
                    if slot_eval is not None:
                        if cache is not None and cache_key is not None:
                            cache[cache_key] = slot_eval
                        return slot_eval
                value = lowered.accept(self)
                if cache is not None and cache_key is not None:
                    cache[cache_key] = value
                return value
        finally:
            stack.pop()

    def _evaluate_lowered_einstein_at_indices(self, lowered: LoweredEinsteinIR, indices: List[Any]) -> Any:
        """Evaluate a scalar cell of a nested LoweredEinsteinIR without materializing the full tensor."""
        try:
            target = tuple(int(np.asarray(idx).reshape(-1)[0]) for idx in indices)
        except Exception:
            return None
        from ..runtime.compute.lowered_execution import execute_lowered_bindings, check_lowered_guards

        def expr_eval(e: Any) -> Any:
            return e.accept(self)

        result = None
        for item in lowered.items or []:
            clause_indices = list(item.indices or [])
            if len(clause_indices) != len(target):
                continue
            loop_context = {}
            matches = True
            for idx_expr, wanted in zip(clause_indices, target):
                if isinstance(idx_expr, LiteralIR):
                    try:
                        if int(idx_expr.value) != wanted:
                            matches = False
                            break
                    except (TypeError, ValueError):
                        matches = False
                        break
                    continue
                did = getattr(idx_expr, "defid", None)
                if did is None:
                    matches = False
                    break
                loop_context[did] = wanted
            if not matches:
                continue
            with self.env.scope():
                for did, val in loop_context.items():
                    self.env.set_value(did, val)
                full_context = execute_lowered_bindings(item.bindings or [], loop_context, expr_eval) if item.bindings else loop_context
                for did, val in full_context.items():
                    if did is not None:
                        self.env.set_value(did, val)
                if item.guards and not check_lowered_guards(item.guards, full_context, lambda e: self._to_bool(e.accept(self))):
                    continue
                value = item.body.accept(self)
            if value is None:
                continue
            if result is None:
                result = value
            else:
                result = result + value
        if result is not None:
            if isinstance(result, np.ndarray) and result.ndim == 0:
                return result.reshape(-1)[0].item()
            return result
        return 0.0

    def _evaluate_lowered_per_outer_slot(self, lowered: Any) -> Any:
        parallel_shape = self._vectorization_parallel_shape()
        if not parallel_shape:
            return None
        outer_shape = tuple(int(dim) for dim in parallel_shape)
        outer_names = getattr(self.env, "_defid_names", {})
        scalarizable: List[Tuple[Any, np.ndarray]] = []
        seen: Set[Any] = set()
        for scope in reversed(getattr(self.env, "_scope_stack", []) or []):
            for did, cur in scope.items():
                if did is None or did in seen:
                    continue
                if not isinstance(cur, np.ndarray):
                    continue
                if cur.ndim == 0 or not np.issubdtype(cur.dtype, np.integer):
                    continue
                try:
                    arr = np.broadcast_to(cur, outer_shape)
                except ValueError:
                    continue
                scalarizable.append((did, arr))
                seen.add(did)
        if not scalarizable:
            return None
        result = None
        for outer_idx in np.ndindex(outer_shape):
            with self.env.scope():
                for did, arr in scalarizable:
                    scalar = np.asarray(arr[outer_idx]).reshape(-1)[0].item()
                    self.env.set_value(did, scalar, name=outer_names.get(did))
                with self._vectorization_scope(
                    parallel_shape=None,
                    parallel_defids_order=None,
                ):
                    cell = lowered.accept(self)
            cell_arr = np.asarray(cell)
            if result is None:
                result_shape = tuple(cell_arr.shape) + outer_shape
                result = np.empty(result_shape, dtype=cell_arr.dtype)
            result[(Ellipsis,) + outer_idx] = cell_arr
        return result

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
        circular = self._maybe_execute_lowered_recurrence_circular(node, variable_decl)
        if circular is not None:
            return circular
        return self._execute_lowered_recurrence_full(node, variable_decl)

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
        if hasattr(self, "_lowered_expr_is_zero") and self._lowered_expr_is_zero(lowered_einstein):
            zero_value = self._zero_value_for_lowered_expr(lowered_einstein)
            if zero_value is not None:
                if variable_key is not None:
                    self.env.set_value(variable_key, zero_value)
                return zero_value
        output_shape = None
        if tensor_shape is not None:
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
        if output_shape is None and lowered_einstein.items:
            output_shape = self._shape_from_all_items(lowered_einstein.items)
        elif output_shape is not None and lowered_einstein.items:
            # Compiler shape may underestimate for multi-segment declarations
            # (e.g. _compute_shape_union picks a symbolic expr that evaluates to
            # the first clause's end, not the union).  Widen only when loop ranks match
            # across items — clauses with different loop *counts* (e.g. [i,j] vs [t,i,j])
            # must not be merged dimension-wise (would corrupt axes).
            items_shape = self._shape_from_all_items(lowered_einstein.items)
            if items_shape and len(items_shape) == len(output_shape):
                output_shape = [max(a, b) for a, b in zip(output_shape, items_shape)]
        if output_shape is None and lowered_einstein.items:
            raise RuntimeError(
                "Einstein declaration has no shape from compiler. "
                "Compiler must set shape (union of clause ranges) on LoweredEinsteinIR."
            )
        if output_shape is None:
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
                    raise RuntimeError(
                        "Lowered Einstein clause missing compiler-owned recurrence_dims_override. "
                        "RecurrenceOrderPass must annotate recurrence metadata before execution."
                    )
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
