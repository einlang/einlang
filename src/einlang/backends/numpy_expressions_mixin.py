"""NumPy backend expression visitor mixin."""

from .numpy_expressions_support import *
from .numpy_expressions_support import (
    _BINARY_OP_MAP,
    _UNARY_OP_MAP,
    _extract_binding,
    _first_parallel_index_defid,
    _invoke_runtime_builtin,
    _normalize_literal_sequence,
    _safe_oob_ndarray_access,
    _try_einsum_reduction,
    _try_matmul_reduction,
    _try_windowed_sumprod_einsum,
)
from .numpy_helpers import _PatternMatcher

class ExpressionVisitorMixin:
    """Expression visit_*; function/builtin lookup via env only."""

    def _raise_here(self, exc: Exception, expr) -> None:
        """Re-raise *exc* as an EinlangSourceError pinned to *expr*.location (or exc.clause_location if set)."""
        from ..shared.errors import EinlangSourceError
        if isinstance(exc, EinlangSourceError):
            raise
        clause_loc = getattr(exc, "clause_location", None)
        loc = clause_loc if (clause_loc and (getattr(clause_loc, "line", 0) or getattr(clause_loc, "file", ""))) else expr.location
        source_code = None
        tcx = getattr(self, "_tcx", None)
        if tcx and loc:
            sf = getattr(tcx, "source_files", None)
            if sf and getattr(loc, "file", None):
                source_code = sf.get(loc.file)
        raise EinlangSourceError(
            message=str(exc),
            location=loc,
            error_code="E0007",
            category="runtime",
            source_code=source_code,
        ) from exc

    @staticmethod
    def _to_bool(value: Any) -> bool:
        if isinstance(value, np.ndarray):
            return bool(value.all())
        return bool(value)

    def visit_literal(self, expr: LiteralIR) -> Any:
        val = expr.value
        if isinstance(val, range):
            size = len(val)
            if size > 1_000_000:
                raise RuntimeError(f"Loop range too large: size={size}")
            return val
        if isinstance(val, tuple):
            return tuple(val)
        if isinstance(val, list):
            # Rectangular literals are serialized as LiteralIR tuples/lists, but the
            # NumPy backend expects ndarray semantics for multi-index access and
            # elementwise arithmetic inside Einstein clauses.
            return _normalize_literal_sequence(val)
        return val

    def visit_identifier(self, expr) -> Any:
        from ..shared.defid import DefId
        defid = expr.defid
        if defid is None:
            raise RuntimeError(f"Variable not found (defid=None). Name: {expr.name or '?'}")
        value = self.env.get_value(defid)
        if value is None:
            raise RuntimeError(f"Variable not found (defid={defid}). Name: {expr.name or '?'}")
        return value

    def visit_differential(self, expr: DifferentialIR) -> Any:
        """Return differential of operand; from buffer filled by backward pass (AUTODIFF_IMPLEMENTATION §9)."""
        op = expr.operand
        target_defid = op.defid if isinstance(op, IdentifierIR) and op.defid is not None else None
        if target_defid is None:
            raise RuntimeError("DifferentialIR operand must be an identifier in v1 backend.")
        buffers = getattr(self, "_differential_buffers", {})
        if target_defid not in buffers:
            raise RuntimeError(
                f"Differential for {getattr(op, 'name', '?')} not in _differential_buffers; "
                "run backward pass first or ensure AutodiffPass produced backward IR."
            )
        return buffers[target_defid]

    def visit_index_var(self, expr) -> Any:
        defid = expr.defid
        if defid is None:
            raise RuntimeError(f"Index variable has no DefId. Name: {expr.name or '?'}")
        value = self.env.get_value(defid)
        if value is None:
            raise RuntimeError(f"Index variable not found (defid={defid}). Name: {expr.name or '?'}")
        return value

    def visit_binary_op(self, expr: BinaryOpIR) -> Any:
        left = expr.left.accept(self)
        right = expr.right.accept(self)
        op = expr.operator
        fn = _BINARY_OP_MAP.get(op) if isinstance(op, BinaryOp) else None
        if fn is None:
            raise RuntimeError(f"Unknown operator: {expr.operator}")
        if isinstance(left, np.ndarray) and isinstance(right, np.ndarray):
            if op != BinaryOp.IN:
                if left.ndim != right.ndim:
                    if left.ndim < right.ndim:
                        left = np.reshape(left, left.shape + (1,) * (right.ndim - left.ndim))
                    else:
                        right = np.reshape(right, right.shape + (1,) * (left.ndim - right.ndim))
                if left.shape != right.shape:
                    left, right = np.broadcast_arrays(left, right)
        try:
            return fn(self, left, right)
        except (ZeroDivisionError, FloatingPointError) as e:
            self._raise_here(e, (expr.right or expr))
        except Exception as e:
            self._raise_here(e, expr)

    def visit_unary_op(self, expr: UnaryOpIR) -> Any:
        operand = expr.operand.accept(self)
        op = expr.operator
        fn = _UNARY_OP_MAP.get(op) if isinstance(op, UnaryOp) else None
        if fn is None:
            raise RuntimeError(f"Unknown unary operator: {expr.operator}")
        try:
            return fn(self, operand)
        except Exception as e:
            self._raise_here(e, expr)

    def visit_function_call(self, expr: FunctionCallIR) -> Any:
        try:
            return self._visit_function_call_inner(expr)
        except Exception as e:
            self._raise_here(e, expr)

    def _visit_function_call_inner(self, expr: FunctionCallIR) -> Any:
        module_path = (expr.module_path or ())
        if expr.function_defid is None and module_path and len(module_path) > 0 and module_path[0] == "python":
            args = [arg.accept(self) for arg in expr.arguments]
            return self._call_python_module(module_path, expr.function_name, args)
        callee_expr = expr.callee_expr
        if callee_expr is not None:
            callee_value = callee_expr.accept(self)
            if isinstance(callee_value, FunctionValue):
                effective_defid = callee_value.defid
            elif hasattr(callee_value, "parameters") and hasattr(callee_value, "body"):
                effective_defid = getattr(callee_value, "defid", None)
                if effective_defid is not None and self.env.get_value(effective_defid) is None:
                    self.env.set_value(effective_defid, callee_value)
            else:
                raise RuntimeError(f"Callable did not evaluate to function (got {type(callee_value).__name__})")
            if effective_defid is None:
                raise RuntimeError("Callable has no DefId")
            args = [arg.accept(self) for arg in expr.arguments]
            func_def = self.env.get_value(effective_defid)
            if func_def is None:
                raise RuntimeError(f"Function (DefId: {effective_defid}) not found")
            return self._call_function(func_def, args)
        if expr.function_defid is None:
            raise RuntimeError("Function call has no DefId")
        args = [arg.accept(self) for arg in expr.arguments]
        callee = self.env.get_value(expr.function_defid)
        if callee is None:
            raise RuntimeError(f"Function not found (DefId: {expr.function_defid})")
        if isinstance(callee, FunctionValue):
            callee = self.env.get_value(callee.defid)
            if callee is None:
                raise RuntimeError(f"Lambda/function (DefId: {expr.function_defid}) not found")
        if hasattr(callee, "body") and hasattr(callee, "parameters"):
            return self._call_function(callee, args)
        if callee == builtin_assert:
            return _invoke_runtime_builtin(callee, args)
        return callee(*args)

    def visit_rectangular_access(self, expr: RectangularAccessIR) -> Any:
        # Autodiff pullback may place LoweredEinsteinIR here; evaluate with synthetic variable_decl stack.
        if isinstance(expr.array, LoweredEinsteinIR):
            array = self._evaluate_lowered_einstein_subexpr(expr.array)
        else:
            array = expr.array.accept(self)
        indices = [idx.accept(self) for idx in (expr.indices or []) if idx is not None]
        try:
            if getattr(array, "is_circular_recurrence_buffer", False):
                return array[tuple(indices)]
            if isinstance(array, np.ndarray):
                if self._vectorization_safe_oob_enabled():
                    return _safe_oob_ndarray_access(array, indices)
                return array[tuple(indices)]
            if isinstance(array, str):
                idx = indices[0] if indices else 0
                return array[int(idx)]
            if isinstance(array, (list, tuple)):
                if not indices:
                    # Preserve ragged lists as Python sequences; only coerce when NumPy can represent them.
                    try:
                        return np.asarray(array)
                    except ValueError:
                        return array
                current = array
                for raw_idx in indices:
                    idx = int(raw_idx) if isinstance(raw_idx, (np.integer, int, float)) else raw_idx
                    current = current[idx]
                return current
            # Forward AD: seeded derivative may be scalar 1 (broadcast over indices)
            if np.isscalar(array) or (isinstance(array, np.ndarray) and array.ndim == 0):
                return array
        except (IndexError, KeyError, TypeError) as e:
            self._raise_here(e, expr)
        raise RuntimeError(f"rectangular_access: expected ndarray, list, or str, got {type(array).__name__}")

    def visit_jagged_access(self, expr: JaggedAccessIR) -> Any:
        array = expr.base.accept(self)
        for idx in (expr.index_chain or []):
            array = array[idx.accept(self)]
        return array

    def visit_block_expression(self, expr: BlockExpressionIR) -> Any:
        with self.env.scope():
            for stmt in (expr.statements or []) or []:
                result_value = stmt.accept(self)
                binding = getattr(stmt, "_binding", None)
                variable_defid = binding.defid if binding else None
                if variable_defid is not None:
                    var_name = binding.name or stmt.name
                    self.env.set_value(variable_defid, result_value, name=var_name)
            if expr.final_expr:
                return expr.final_expr.accept(self)
        return None

    def visit_if_expression(self, expr) -> Any:
        cond = expr.condition.accept(self)
        if isinstance(cond, np.ndarray) and cond.ndim > 0:
            then_val = expr.then_expr.accept(self)
            else_val = expr.else_expr.accept(self) if expr.else_expr else None
            return np.where(cond, then_val, else_val)
        if self._to_bool(cond):
            return expr.then_expr.accept(self)
        if expr.else_expr:
            return expr.else_expr.accept(self)
        return None

    def visit_lambda(self, expr) -> Any:
        defid = getattr(expr, "defid", None)
        if defid is None:
            resolver = getattr(self, "resolver", None)
            if resolver is not None:
                defid = resolver.allocate_for_local()
            else:
                raise RuntimeError("Lambda has no DefId and backend has no resolver to allocate one")
        self.env.set_value(defid, expr)
        return FunctionValue(defid=defid, closure_env=self.env)

    def visit_range(self, expr: RangeIR) -> Any:
        start = expr.start.accept(self)
        end = expr.end.accept(self)
        end_int = int(end) + 1 if (expr.inclusive or False) else int(end)
        return range(int(start), end_int)

    def visit_array_comprehension(self, expr: ArrayComprehensionIR) -> Any:
        _reject_non_lowered(type(expr).__name__)

    def visit_lowered_comprehension(self, expr: LoweredComprehensionIR) -> Any:
        from ..runtime.compute.lowered_execution import execute_lowered_loops, check_lowered_guards
        results = []
        def ev(e): return e.accept(self)
        for context in execute_lowered_loops(expr.loops, {}, ev):
            with self.env.scope():
                for defid, val in context.items():
                    if defid is not None:
                        self.env.set_value(defid, val)
                full = {}
                for binding in (expr.bindings or []) or []:
                    defid = binding.defid
                    if defid is not None:
                        val = binding.expr.accept(self)
                        full[defid] = val
                        self.env.set_value(defid, val)
                if not (expr.guards and not check_lowered_guards(expr.guards, full, lambda c: self._to_bool(c.accept(self)))):
                    results.append(expr.body.accept(self))
        return np.array(results) if results else np.array([])

    def visit_array_literal(self, expr: ArrayLiteralIR) -> Any:
        type_info = expr.type_info
        if type_info is not None and getattr(type_info, "kind", None) == TypeKind.JAGGED:
            evaluated = [e.accept(self) for e in expr.elements]
            return list(evaluated)
        dtype = None
        if type_info is not None:
            converter = getattr(self, "_type_info_to_numpy_dtype", None)
            if callable(converter):
                el = getattr(type_info, "element_type", None) or type_info
                dtype = converter(el)
        if dtype is None and expr.elements:
            for e in expr.elements:
                v = (e.value if isinstance(e, LiteralIR) else None)
                if v is not None and isinstance(v, (float, np.floating)):
                    dtype = np.float32
                    break
        evaluated = [e.accept(self) for e in expr.elements]
        if dtype is None and evaluated:
            if isinstance(evaluated[0], (float, np.floating)):
                dtype = np.float32
            elif isinstance(evaluated[0], (int, np.integer)) and not isinstance(evaluated[0], (bool, np.bool_)):
                if any(isinstance(x, (float, np.floating)) for x in evaluated):
                    dtype = np.float32
        if dtype is not None:
            return np.array(evaluated, dtype=dtype)
        return np.array(evaluated)

    def visit_tuple_expression(self, expr: TupleExpressionIR) -> Any:
        return tuple(e.accept(self) for e in expr.elements)

    def visit_tuple_access(self, expr: TupleAccessIR) -> Any:
        t = expr.tuple_expr.accept(self)
        return t[expr.index]

    def visit_interpolated_string(self, expr: InterpolatedStringIR) -> Any:
        parts = []
        for part in (expr.parts or []) or []:
            parts.append(str(part.accept(self)) if hasattr(part, "accept") else str(part))
        return "".join(parts)

    def visit_cast_expression(self, expr: CastExpressionIR) -> Any:
        inner = expr.expr or expr.operand
        if inner is None:
            raise RuntimeError("CastExpressionIR has no expr or operand")
        val = inner.accept(self)
        target = expr.target_type
        if target is None:
            return val
        name = getattr(target, "name", None) or (target if isinstance(target, str) else None)
        _cast_dtype = self._resolve_cast_dtype(name)
        if _cast_dtype is not None:
            if val is None:
                return None
            if isinstance(val, np.ndarray):
                return val.astype(_cast_dtype)
            if name == "bool":
                return bool(val)
            if name in ("i8", "i32", "i64"):
                return int(val)
            return float(val)
        elem_type = getattr(target, "element_type", None)
        if elem_type is not None and val is not None:
            elem_name = getattr(elem_type, "name", None) or (elem_type if isinstance(elem_type, str) else None)
            _elem_dtype = self._resolve_cast_dtype(elem_name)
            if _elem_dtype is not None:
                return np.asarray(val, dtype=_elem_dtype)
        return val

    @staticmethod
    def _resolve_cast_dtype(name):
        _CAST_DTYPES = {
            "i8": np.int8, "i32": np.int32, "i64": np.int64,
            "f16": np.float16, "f32": np.float32, "f64": np.float64,
            "bool": np.bool_,
        }
        dt = _CAST_DTYPES.get(name)
        if dt is not None:
            return dt
        try:
            import ml_dtypes
            _ML_DTYPES = {"bf16": ml_dtypes.bfloat16, "f8e4m3": ml_dtypes.float8_e4m3fn}
            return _ML_DTYPES.get(name)
        except ImportError:
            return None

    def _call_python_module(
        self,
        module_path: tuple,
        function_name: str,
        args: List[Any],
    ) -> Any:
        import importlib
        parts = list(module_path)
        if parts and parts[0] == "python":
            parts = parts[1:]
        if not parts:
            raise RuntimeError(f"Invalid module path for Python module call: {module_path}")
        module_name = ".".join(parts)
        if module_path == ("python", "builtins") or module_path == ("builtins",):
            import builtins
            callable_func = getattr(builtins, function_name, None)
            if callable_func is None:
                raise RuntimeError(f"Python builtin '{function_name}' not found")
            return callable_func(*args)
        try:
            module = importlib.import_module(module_name)
        except ImportError as e:
            raise RuntimeError(f"Failed to import Python module '{module_name}': {e}")
        value = getattr(module, function_name, None)
        if value is None:
            raise RuntimeError(
                f"Function/property '{function_name}' not found in Python module '{module_name}'"
            )
        if not callable(value):
            if len(args) == 0:
                return value
            raise RuntimeError(
                f"Cannot call non-callable property {module_name}::{function_name}: {type(value)}"
            )
        with np.errstate(all="ignore"):
            return value(*args)

    def visit_member_access(self, expr: MemberAccessIR) -> Any:
        obj = expr.object.accept(self)
        member = expr.member
        if member is None:
            return None
        if isinstance(member, str):
            if member == "length" and isinstance(obj, (np.ndarray, list, tuple, str)):
                return len(obj)
            if member == "size" and isinstance(obj, np.ndarray):
                return obj.size
            if member == "shape" and isinstance(obj, np.ndarray):
                return obj.shape
            return getattr(obj, member, None)
        if isinstance(member, int):
            return obj[member]
        key = member.accept(self) if hasattr(member, "accept") else getattr(member, "value", member)
        return obj[key] if key is not None else None

    def visit_try_expression(self, expr: TryExpressionIR) -> Any:
        try:
            return expr.operand.accept(self)
        except Exception:
            raise

    def visit_match_expression(self, expr: MatchExpressionIR) -> Any:
        scrutinee_value = expr.scrutinee.accept(self)
        matcher = _PatternMatcher(scrutinee_value, self)
        for arm in expr.arms:
            bindings = arm.pattern.accept(matcher)
            if bindings is None:
                continue
            if hasattr(arm.pattern, "guard_expr"):
                with self.env.scope():
                    for var_defid, var_value in bindings.items():
                        if var_defid is not None:
                            self.env.set_value(var_defid, var_value)
                    if not self._to_bool(arm.pattern.guard_expr.accept(self)):
                        continue
            with self.env.scope():
                for var_defid, var_value in bindings.items():
                    if var_defid is not None:
                        self.env.set_value(var_defid, var_value)
                return arm.body.accept(self)
        try:
            raise RuntimeError(f"Match not exhaustive: no pattern matched {scrutinee_value}")
        except RuntimeError as e:
            self._raise_here(e, expr)

    def visit_reduction_expression(self, expr: ReductionExpressionIR) -> Any:
        _reject_non_lowered(type(expr).__name__)

    def evaluate_lowered_reduction(
        self, expr: LoweredReductionIR, parallel_shape: Optional[Tuple[int, ...]] = None
    ) -> Any:
        """Evaluate a lowered reduction, optionally with vectorized path when parallel_shape is set.
        When parallel_shape is None, uses the backend vectorization state when present (e.g. in a vectorized clause).
        Fast paths (matmul, conv via einsum) only when parallel_shape is set; stricter conditions avoid LSTM."""
        import os
        if parallel_shape is None:
            parallel_shape = self._vectorization_parallel_shape()
        # Recurrence clauses may use partial vectorization but must not use fast_matmul / fast_conv.
        if parallel_shape is not None and not self._in_recurrence_vectorization_clause():
            windowed_result = _try_windowed_sumprod_einsum(expr, self)
            if windowed_result is not None and isinstance(windowed_result, np.ndarray):
                if windowed_result.shape == tuple(parallel_shape):
                    setattr(self, "_last_reduction_fast_path", "windowed-einsum")
                    return windowed_result
            matmul_result = _try_matmul_reduction(expr, self)
            if matmul_result is not None and isinstance(matmul_result, np.ndarray):
                if matmul_result.shape == tuple(parallel_shape):
                    setattr(self, "_last_reduction_fast_path", "matmul")
                    return matmul_result
            einsum_result = _try_einsum_reduction(expr, self)
            if einsum_result is not None and isinstance(einsum_result, np.ndarray):
                if einsum_result.shape == tuple(parallel_shape):
                    setattr(self, "_last_reduction_fast_path", "einsum")
                    return einsum_result
        from ..runtime.compute.lowered_execution import (
            execute_reduction_with_loops,
            execute_select_at_argmax_vectorized,
        )
        from ..passes.visitor_helpers import ArrayAccessCollector
        loc = expr.location
        line = int(getattr(loc, "line", 0) or 0)
        profile_reductions = bool(os.environ.get("EINLANG_PROFILE_REDUCTIONS", ""))
        seen = getattr(self, "_reduction_profile_seen", None)
        if seen is None:
            seen = set()
            self._reduction_profile_seen = seen
        def reduction_profile(path: str) -> None:
            if profile_reductions:
                key = (line, path)
                if key not in seen:
                    seen.add(key)
                    print(f"[reduction] {path} L{line}", flush=True)
        def ev(e): return e.accept(self)
        _reduction_defid_names = {}
        for _lp in (expr.loops or []) or []:
            _v = _lp.variable
            if _v is not None and _v.defid is not None:
                _reduction_defid_names[_v.defid] = _v.name

        def body_ev(ctx):
            _ctx = {
                defid: val for defid, val in (ctx or {}).items()
                if defid is not None
            }
            saved: Dict[Any, Any] = {}
            for defid in _ctx:
                if defid is not None:
                    saved[defid] = self.env.get_value(defid)
            try:
                for defid, val in _ctx.items():
                    if defid is not None:
                        self.env.set_value(defid, val, name=_reduction_defid_names.get(defid))
                return expr.body.accept(self)
            finally:
                for defid, val in saved.items():
                    self.env.set_value(defid, val, name=_reduction_defid_names.get(defid))
        def guard_ev(ctx):
            if not expr.guards:
                return True
            _ctx = {
                defid: val for defid, val in (ctx or {}).items()
                if defid is not None
            }
            for defid, val in _ctx.items():
                if defid is not None:
                    self.env.set_value(defid, val, name=_reduction_defid_names.get(defid))
            from ..runtime.compute.lowered_execution import check_lowered_guards
            return check_lowered_guards(expr.guards, _ctx, lambda c: self._to_bool(c.accept(self)))
        # Pass parallel indices into reduction so body/guards can see them.
        # In scalar Einstein execution, numpy_einstein sets `_reduction_initial_context` per-iteration.
        # In fully vectorized execution via _eval_clause_body_with_broadcast_loops, parallel vars are
        # already in the env with correct ndim (clause_ndim + n_red). Do NOT rebuild initial_ctx in
        # that case — rebuilding would create arrays with wrong dimensionality (parallel-only ndim)
        # that clobber the correct env values when body_ev sets them.
        initial_ctx = dict(getattr(self, "_reduction_initial_context", None) or {})
        vector_parallel_ctx: Dict[Any, Any] = {}
        if (not initial_ctx) and parallel_shape is None:
            try:
                reduction_body_defids = {
                    loop.variable.defid
                    for loop in (expr.loops or [])
                    if loop.variable is not None and loop.variable.defid is not None
                }
                seen_parallel_defids = set()
                collector = ArrayAccessCollector()
                scan_exprs = [expr.body]
                scan_exprs.extend(g.condition for g in (expr.guards or []))
                for scan_expr in scan_exprs:
                    if scan_expr is None or not hasattr(scan_expr, "accept"):
                        continue
                    for access in (scan_expr.accept(collector) or []):
                        for idx in (access.indices or []):
                            did = _first_parallel_index_defid(idx, reduction_body_defids)
                            if did is None or did in seen_parallel_defids:
                                continue
                            cur = self.env.get_value(did)
                            if isinstance(cur, np.ndarray) and cur.ndim >= 1:
                                initial_ctx[did] = cur
                                seen_parallel_defids.add(did)
            except Exception:
                pass
        if (not initial_ctx) and parallel_shape:
            order_defids = self._vectorization_parallel_defids_order()
            if order_defids is not None and len(order_defids) == len(parallel_shape):
                n_red_ix = len(expr.reduction_ranges or {})
                if n_red_ix >= 1:
                    for idx, did in enumerate(order_defids):
                        if did is None:
                            continue
                        cur = self.env.get_value(did)
                        if not isinstance(cur, np.ndarray):
                            continue
                        sz = int(parallel_shape[idx])
                        flat = np.asarray(cur, dtype=np.intp).reshape(-1)
                        start = int(flat[0]) if flat.size else 0
                        shape_pv = [1] * (len(parallel_shape) + n_red_ix)
                        shape_pv[idx] = sz
                        arr = np.arange(start, start + sz, dtype=np.intp).reshape(shape_pv)
                        vector_parallel_ctx[did] = arr
            else:
                try:
                    reduction_body_defids = {
                        loop.variable.defid
                        for loop in (expr.loops or [])
                        if loop.variable is not None and loop.variable.defid is not None
                    }
                    if isinstance(expr.body, RectangularAccessIR):
                        par_defids: List[Any] = []
                        for idx in (expr.body.indices or []):
                            d = getattr(idx, "defid", None)
                            if d is not None and d not in reduction_body_defids:
                                par_defids.append(d)
                        if len(par_defids) == len(parallel_shape):
                            par_shape = tuple(parallel_shape)
                            initial_ctx = {}
                            for i, did in enumerate(par_defids):
                                n = par_shape[i]
                                shape = (1,) * i + (n,) + (1,) * (len(par_shape) - i - 1)
                                initial_ctx[did] = np.arange(n, dtype=np.intp).reshape(shape)
                except Exception:
                    pass

        # Guarded vectorized reduction: evaluate guards elementwise as arrays and keep the reduction
        # on a vectorized path (instead of scalar loops). This is safe when guards/body are pure
        # NumPy expressions. Works both with parallel dims (parallel_shape) and standalone guarded
        # reductions (treat parallel_shape as ()).
        if expr.guards:
            from ..shared.types import ReductionOp
            op = expr.operation
            if op in (ReductionOp.SUM, ReductionOp.PROD, ReductionOp.MIN, ReductionOp.MAX):
                try:
                    reduction_loops = list((expr.reduction_ranges or {}).values())
                    n_red = len(reduction_loops)
                    par_shape = tuple(parallel_shape) if parallel_shape is not None else ()
                    # Build broadcasted index arrays for reduction loops (same strategy as vectorized reduction).
                    arrs: List[np.ndarray] = []
                    loop_defids: List[Any] = []
                    for loop in reduction_loops:
                        d = loop.variable.defid
                        iterable = ev(loop.iterable)
                        if isinstance(iterable, range):
                            step = iterable.step if iterable.step is not None else 1
                            a = np.arange(iterable.start, iterable.stop, step, dtype=np.intp)
                        else:
                            a = np.array(list(iterable), dtype=np.intp)
                        arrs.append(a)
                        loop_defids.append(d)
                    expected_shape = tuple(int(a.size) for a in arrs)
                    full_shape = par_shape + expected_shape
                    ctx: Dict[Any, Any] = {}
                    # Parallel indices (e.g. b) → broadcast to full shape
                    _par_idx = dict(initial_ctx)
                    _par_idx.update(vector_parallel_ctx)
                    if _par_idx:
                        for did, val in _par_idx.items():
                            if did is None:
                                continue
                            v = np.asarray(val, dtype=np.intp)
                            if par_shape:
                                if v.ndim == 1 and len(par_shape) == 1 and v.shape[0] == par_shape[0]:
                                    v = v.reshape((par_shape[0],) + (1,) * n_red)
                                elif v.shape == tuple(par_shape):
                                    v = v.reshape(tuple(par_shape) + (1,) * n_red)
                            else:
                                # Standalone reduction: scalar initial_ctx values broadcast over reduction grid.
                                if v.ndim == 0:
                                    v = v.reshape((1,) * n_red)
                            ctx[did] = np.broadcast_to(v, full_shape)
                    # Reduction indices
                    for i, (did, a) in enumerate(zip(loop_defids, arrs)):
                        red_shape = (a.size,) if n_red == 1 else tuple((a.size if j == i else 1) for j in range(n_red))
                        red_arr = a.reshape(red_shape)
                        ctx[did] = np.broadcast_to(red_arr, full_shape)

                    # Evaluate body + guards elementwise.
                    body_val = body_ev(ctx)
                    body_arr = np.asarray(body_val) if not isinstance(body_val, np.ndarray) else body_val
                    if body_arr.shape != full_shape:
                        body_arr = np.broadcast_to(body_arr, full_shape)
                    mask = np.ones(full_shape, dtype=bool)
                    for g in (expr.guards or []):
                        with self.env.scope():
                            _ctx_g = {
                                defid: val for defid, val in ctx.items()
                                if defid is not None
                            }
                            for defid, val in _ctx_g.items():
                                if defid is not None:
                                    self.env.set_value(
                                        defid, val, name=_reduction_defid_names.get(defid)
                                    )
                            gv = g.condition.accept(self)
                        gv_arr = np.asarray(gv, dtype=bool) if not isinstance(gv, np.ndarray) else gv.astype(bool, copy=False)
                        if gv_arr.shape != full_shape:
                            gv_arr = np.broadcast_to(gv_arr, full_shape)
                        mask &= gv_arr

                    # Apply mask then reduce over last n_red axes.
                    red_axes = tuple(range(-n_red, 0))
                    if op == ReductionOp.SUM:
                        out = np.where(mask, body_arr, 0).sum(axis=red_axes)
                    elif op == ReductionOp.PROD:
                        out = np.where(mask, body_arr, 1).prod(axis=red_axes)
                    elif op == ReductionOp.MIN:
                        out = np.where(mask, body_arr, np.inf).min(axis=red_axes)
                    elif op == ReductionOp.MAX:
                        out = np.where(mask, body_arr, -np.inf).max(axis=red_axes)
                    else:
                        out = None
                    if out is not None:
                        # NumPy may return np.ndarray for parallel outputs, but for scalar outputs
                        # it may return a NumPy scalar (np.generic). Treat both as success.
                        if isinstance(out, np.ndarray):
                            if out.shape == par_shape:
                                reduction_profile("vectorized")
                                return out
                        else:
                            if par_shape == () and isinstance(out, (np.generic, int, float, bool)):
                                reduction_profile("vectorized")
                                return out
                except Exception:
                    pass
        return execute_reduction_with_loops(
            expr.operation,
            (expr.reduction_ranges or {}),
            body_ev,
            ev,
            guard_evaluator=guard_ev if expr.guards else None,
            initial_context=initial_ctx,
            profile_callback=reduction_profile if profile_reductions else None,
            parallel_shape=parallel_shape,
            vector_parallel_context=vector_parallel_ctx,
        )

    def visit_lowered_reduction(self, expr: LoweredReductionIR) -> Any:
        return self.evaluate_lowered_reduction(expr, parallel_shape=None)

    def visit_lowered_select_at_argmax(self, expr: LoweredSelectAtArgmaxIR) -> Any:
        from ..passes.visitor_helpers import all_defids_of_var_in_expr
        from ..runtime.compute.lowered_execution import execute_select_at_argmax_vectorized
        from ..ir.nodes import RectangularAccessIR
        from ..passes.autodiff._graph import _collect_defids
        reduction_loops = list((expr.reduction_ranges or {}).values())
        if not reduction_loops:
            raise RuntimeError("SelectAtArgmax has no reduction loops")
        n_red = len(reduction_loops)

        # Build loop_var_defid -> ALL body defids (covering both primal and diff bodies)
        # and a union set of all reduction body defids.
        # primal_body and diff_body may use different defids for the same variable name
        # (e.g. batch.0 in primal vs diff), so we must alias every one.
        _loop_to_all_body_defids: Dict[Any, List[Any]] = {}  # loop_var_defid -> [body_defid, ...]
        _reduction_defid_names: Dict[Any, str] = {}
        for _lp in expr.loops or []:
            _v = _lp.variable
            if _v is not None and _v.defid is not None and _v.name:
                _vname = _v.name
                _all_bds = all_defids_of_var_in_expr(expr.primal_body, _vname) | \
                           all_defids_of_var_in_expr(expr.diff_body, _vname)
                _bds = [d for d in _all_bds if d != _v.defid]
                if _bds:
                    _loop_to_all_body_defids[_v.defid] = _bds
                for _bd in _all_bds:
                    _reduction_defid_names[_bd] = _vname
                _reduction_defid_names[_v.defid] = _vname

        # Collect union of ALL reduction-loop body defids (for parallel-index detection)
        reduction_body_defids: Set[Any] = set()
        for _lp in expr.loops or []:
            if _lp.variable is not None and _lp.variable.defid is not None:
                reduction_body_defids.add(_lp.variable.defid)
                for _bd in _loop_to_all_body_defids.get(_lp.variable.defid) or []:
                    reduction_body_defids.add(_bd)

        outer_index_defids = tuple(getattr(self, "_select_outer_index_defids", ()) or ())
        parallel_shape = None
        parallel_defids_list: List[Any] = []
        initial_context: List[Tuple[Any, Any]] = []
        _ri0 = getattr(self, "_reduction_initial_context", None) or {}
        clause_parallel_shape = self._vectorization_parallel_shape()
        if clause_parallel_shape is not None and not outer_index_defids:
            try:
                parallel_shape = tuple(int(dim) for dim in clause_parallel_shape)
            except (TypeError, ValueError):
                parallel_shape = None
        if not outer_index_defids and isinstance(expr.primal_body, RectangularAccessIR) and expr.primal_body.array is not None:
            try:
                arr = expr.primal_body.array.accept(self)
                if parallel_shape is None and not _ri0 and isinstance(arr, np.ndarray) and arr.ndim >= n_red:
                    parallel_shape = tuple(arr.shape[:-n_red])
                if parallel_shape is None and not _ri0:
                    for idx in (expr.primal_body.indices or []):
                        d = _first_parallel_index_defid(idx, reduction_body_defids)
                        if d is not None:
                            parallel_defids_list.append(d)
            except Exception:
                pass
        if parallel_shape and len(parallel_defids_list) == len(parallel_shape):
            initial_context = [
                (defid, np.arange(parallel_shape[i], dtype=np.intp))
                for i, defid in enumerate(parallel_defids_list)
            ]

        def _expand_ctx_to_all_body_defids(ctx: Dict) -> Dict:
            """Given a ctx keyed by loop_var_defids, expand to also include all body-defid aliases."""
            out: Dict[Any, Any] = {}
            for loop_defid, val in (ctx or {}).items():
                if loop_defid is None:
                    continue
                out[loop_defid] = val
                for body_defid in _loop_to_all_body_defids.get(loop_defid) or []:
                    out[body_defid] = val
            return out

        def _apply_ctx_then_eval(ctx: Dict, body_expr: Any) -> Any:
            ri = getattr(self, "_reduction_initial_context", None) or {}
            # Expand reduction-loop ctx to all body defid aliases
            _ctx = _expand_ctx_to_all_body_defids(ctx)
            # ri already contains outer-loop aliases (both primal and diff body defids,
            # built by _collect_defids_by_name in _execute_lowered_einstein_clause)
            merged = {**ri, **_ctx}
            saved: Dict[Any, Any] = {}
            try:
                for defid in merged:
                    if defid is not None:
                        saved[defid] = self.env.get_value(defid)
                for defid, val in merged.items():
                    if defid is not None:
                        self.env.set_value(defid, val, name=_reduction_defid_names.get(defid))
                return body_expr.accept(self)
            finally:
                for defid, val in saved.items():
                    self.env.set_value(defid, val, name=_reduction_defid_names.get(defid))

        def primal_body_ev(ctx):
            return _apply_ctx_then_eval(ctx, expr.primal_body)

        def diff_body_ev(ctx):
            return _apply_ctx_then_eval(ctx, expr.diff_body)

        def ev(e):
            return e.accept(self)

        # Quotient gradients for max/min often lower to SelectAtArgmax whose diff body is a
        # read from the denominator tangent buffer (e.g. @x[...]). When this select executes
        # inside an outer Einstein loop, the current outer loop indices identify the derivative
        # slot we are filling. In the scalar/no-parallel execution path we must preserve that
        # slot match instead of treating the selected tangent read as an unconditional scalar.
        ri0 = getattr(self, "_reduction_initial_context", None) or {}
        if (
            not parallel_shape
            and ri0
            and isinstance(expr.primal_body, RectangularAccessIR)
            and isinstance(expr.diff_body, RectangularAccessIR)
            and len(expr.primal_body.indices or []) == len(expr.diff_body.indices or [])
        ):
            try:
                arrs: List[np.ndarray] = []
                defids: List[Any] = []
                for loop in reduction_loops:
                    var_defid = loop.variable.defid
                    iterable = ev(loop.iterable)
                    if isinstance(iterable, range):
                        step = iterable.step if iterable.step is not None else 1
                        arr = np.arange(iterable.start, iterable.stop, step, dtype=np.intp)
                    else:
                        arr = np.array(list(iterable), dtype=np.intp)
                    if arr.size == 0:
                        raise ValueError("empty reduction loop")
                    arrs.append(arr)
                    defids.append(var_defid)
                red_shape_tuple = tuple(int(arr.size) for arr in arrs)
                ctx: Dict[Any, Any] = {}
                for i, (defid, arr) in enumerate(zip(defids, arrs)):
                    red_shape = [1] * len(arrs)
                    red_shape[i] = arr.size
                    ctx[defid] = arr.reshape(tuple(red_shape))
                primal_result = primal_body_ev(ctx)
                if isinstance(primal_result, np.ndarray):
                    idx_flat = int(np.argmin(primal_result) if getattr(expr, "use_argmin", False) else np.argmax(primal_result))
                else:
                    idx_flat = 0
                red_multi = np.unravel_index(idx_flat, red_shape_tuple)
                chosen_ctx = {did: int(arrs[i][red_multi[i]]) for i, did in enumerate(defids)}

                current_scalar_ctx = {}
                for did, val in ri0.items():
                    arr = np.asarray(val)
                    if arr.ndim == 0 or arr.size == 1:
                        current_scalar_ctx[did] = int(arr.reshape(-1)[0])
                reduction_defids = set(defids)
                chosen_full_ctx = {**current_scalar_ctx, **chosen_ctx}
                matches_current_slot = True
                for pos, idx_expr in enumerate(expr.diff_body.indices or []):
                    if pos >= len(outer_index_defids):
                        continue
                    expected_did = outer_index_defids[pos]
                    if expected_did is None or expected_did not in current_scalar_ctx or expected_did in reduction_defids:
                        continue
                    expected = current_scalar_ctx[expected_did]
                    actual = int(np.asarray(_apply_ctx_then_eval(chosen_full_ctx, idx_expr)).reshape(-1)[0])
                    if actual != expected:
                        matches_current_slot = False
                        break
                if not matches_current_slot:
                    return 0.0
            except Exception:
                pass

        ok, result = execute_select_at_argmax_vectorized(
            primal_body_ev,
            diff_body_ev,
            reduction_loops,
            ev,
            parallel_shape=parallel_shape,
            initial_context=initial_context if initial_context else None,
            use_argmin=getattr(expr, "use_argmin", False),
        )
        if not ok or result is None:
            from ..runtime.compute.lowered_execution import execute_lowered_loops
            best_primal = None
            best_diff = None
            scalar_init: Dict[Any, Any] = {}
            ri_scalar = getattr(self, "_reduction_initial_context", None) or {}
            for did, val in ri_scalar.items():
                arr = np.asarray(val)
                if arr.ndim == 0 or arr.size == 1:
                    scalar_init[did] = int(arr.reshape(-1)[0]) if np.issubdtype(arr.dtype, np.integer) else arr.reshape(-1)[0].item()
            for loop_ctx in execute_lowered_loops(reduction_loops, scalar_init, ev):
                full_ctx = dict(loop_ctx)
                primal_val = primal_body_ev(full_ctx)
                primal_arr = np.asarray(primal_val)
                if primal_arr.size != 1:
                    raise RuntimeError("SelectAtArgmax scalar fallback expected scalar primal body")
                primal_scalar = primal_arr.reshape(-1)[0].item()
                choose = False
                if best_primal is None:
                    choose = True
                elif getattr(expr, "use_argmin", False):
                    choose = primal_scalar < best_primal
                else:
                    choose = primal_scalar > best_primal
                if choose:
                    best_primal = primal_scalar
                    best_diff = diff_body_ev(full_ctx)
            if best_diff is None:
                raise RuntimeError("SelectAtArgmax vectorized execution failed")
            result = best_diff
        return result

    def visit_where_expression(self, expr: WhereExpressionIR) -> Any:
        constraints = (expr.constraints or []) or []
        needs_scope = any(_extract_binding(c) and _extract_binding(c)[0] for c in constraints)
        if needs_scope:
            with self.env.scope():
                for c in constraints:
                    b = _extract_binding(c)
                    if b:
                        var_defid, value_expr = b
                        if var_defid:
                            self.env.set_value(var_defid, value_expr.accept(self))
                for c in constraints:
                    b = _extract_binding(c)
                    if b and b[0]:
                        continue
                    if not self._to_bool(c.accept(self)):
                        return None
                return expr.expr.accept(self)
        for c in constraints:
            b = _extract_binding(c)
            if b and b[0]:
                self.env.set_value(b[0], b[1].accept(self))
        for c in constraints:
            b = _extract_binding(c)
            if b and b[0]:
                continue
            if not self._to_bool(c.accept(self)):
                return None
        return expr.expr.accept(self)

    def visit_pipeline_expression(self, expr: PipelineExpressionIR) -> Any:
        left_value = expr.left.accept(self)
        from .numpy_arrow_pipeline import apply_pipeline_right
        return apply_pipeline_right(expr.right, left_value, expr.location, self)

    def visit_builtin_call(self, expr: BuiltinCallIR) -> Any:
        from ..shared.defid import DefId
        raw = expr.defid
        if raw is None:
            raise RuntimeError("Builtin call has no DefId")
        if isinstance(raw, (list, tuple)) and len(raw) >= 2:
            defid = DefId(krate=int(raw[0]), index=int(raw[1]))
        elif hasattr(raw, "krate") and hasattr(raw, "index"):
            defid = DefId(krate=int(raw.krate), index=int(raw.index))
        else:
            defid = raw
        fn = self.env.get_value(defid)
        if fn is None or not callable(fn):
            raise RuntimeError(f"Builtin not found (DefId: {defid})")
        args_list = (expr.args or [])
        args = [arg.accept(self) for arg in args_list]
        try:
            return _invoke_runtime_builtin(fn, args)
        except Exception as e:
            self._raise_here(e, expr)
