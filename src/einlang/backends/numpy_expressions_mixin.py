"""NumPy backend expression visitor mixin."""

from types import SimpleNamespace
from typing import Any, List, Optional, Set, Tuple

from ..ir.nodes import (
    EinsteinIR,
    BinaryOpIR,
    BindingIR,
    BlockExpressionIR,
    BuiltinCallIR,
    CastExpressionIR,
    FunctionCallIR,
    IfExpressionIR,
    IRNode,
    LazyJacobianIR,
    LiteralIR,
    LoweredEinsteinIR,
    LoweredReductionIR,
    LoweredSelectAtArgmaxIR,
    MemberAccessIR,
    RectangularAccessIR,
    UnaryOpIR,
    is_function_binding,
)
from ..ir.predicate_visitor import RecursivePredicateVisitor
from ..shared.debug_trace import emit_debug_log
from ..shared.autodiff_intrinsics import autodiff_builtin_kind
from ..shared.types import ReductionOp, RectangularType
from ..runtime.autodiff_requests import AutodiffRuntimeEngine

from .numpy_expressions_support import *
from .numpy_expressions_support import (
    _BINARY_OP_MAP,
    _UNARY_OP_MAP,
    _first_parallel_index_defid,
    _invoke_runtime_builtin,
    _normalize_literal_sequence,
    _reject_non_lowered,
    _safe_oob_ndarray_access,
    _try_einsum_reduction,
    _try_matmul_reduction,
    _try_windowed_sumprod_einsum,
)
from .numpy_helpers import _PatternMatcher
from .numpy_einstein_analysis import (
    _BodyReferencesDefidVisitor,
    _count_reduction_dims_in_expr,
    _reduction_uses_clause_var_in_bounds,
)


class _ContainsNodeTypeVisitor(RecursivePredicateVisitor):
    def __init__(self, *node_types: type) -> None:
        self._node_types = node_types

    def matches(self, node: Any) -> bool:
        return isinstance(node, self._node_types)


def _contains_ir_node_type(node: Any, *node_types: type) -> bool:
    if node is None:
        return False
    return _ContainsNodeTypeVisitor(*node_types).visit(node)


def _contains_nested_lowered_reduction(node: Any) -> bool:
    return _contains_ir_node_type(node, LoweredReductionIR, LoweredSelectAtArgmaxIR)


def _contains_if_expression(node: Any) -> bool:
    return _contains_ir_node_type(node, IfExpressionIR)


def _contains_lowered_einstein(node: Any) -> bool:
    return _contains_ir_node_type(node, LoweredEinsteinIR)


def _collect_ir_defids(node: Any) -> Set[Any]:
    out: Set[Any] = set()
    seen: Set[int] = set()
    stack: List[Any] = [node]
    while stack:
        cur = stack.pop()
        if cur is None:
            continue
        oid = id(cur)
        if oid in seen:
            continue
        seen.add(oid)
        did = getattr(cur, "defid", None)
        if did is not None and hasattr(cur, "name"):
            out.add(did)
        if isinstance(cur, dict):
            stack.extend(cur.keys())
            stack.extend(cur.values())
            continue
        if isinstance(cur, (list, tuple)):
            stack.extend(cur)
            continue
        if isinstance(cur, IRNode):
            for cls in type(cur).__mro__:
                for slot in getattr(cls, "__slots__", ()):
                    stack.append(getattr(cur, slot, None))
    return out


def _concrete_shape_from_type_info(type_info: Any) -> Optional[Tuple[int, ...]]:
    if not isinstance(type_info, RectangularType):
        return None
    shape = getattr(type_info, "shape", None)
    if shape is None or getattr(type_info, "is_dynamic_rank", False):
        return None
    dims: List[int] = []
    for dim in shape:
        if not isinstance(dim, int):
            return None
        dims.append(int(dim))
    return tuple(dims)


def _is_zero_literal_like(node: Any) -> bool:
    return isinstance(node, LiteralIR) and getattr(node, "value", None) == 0


class ExpressionVisitorMixin:
    """Expression visit_*; function/builtin lookup via env only."""

    def _lowered_execution_analysis(self) -> Dict[str, Any]:
        tcx = getattr(self, "_tcx", None)
        if tcx is None:
            return {}
        try:
            from ..passes.lowered_execution_facts import LoweredExecutionFactsPass

            return tcx.get_analysis(LoweredExecutionFactsPass) or {}
        except RuntimeError:
            return {}

    def _lowered_reduction_facts(self, expr: LoweredReductionIR) -> Optional[Any]:
        facts_id = getattr(expr, "execution_facts_id", None)
        if facts_id is None:
            raise RuntimeError(
                "LoweredReductionIR missing compiler-owned execution_facts_id. "
                "Recurrence ordering must annotate lowered execution facts before execution."
            )
        analysis = self._lowered_execution_analysis()
        facts = (analysis.get("reduction_facts_by_id") or {}).get(facts_id)
        if facts is None:
            raise RuntimeError(
                f"Missing lowered reduction execution facts for id={facts_id}. "
                "Compiler/runtime lowered-execution analysis mismatch."
            )
        return facts

    def _lowered_reduction_kernel_plan(self, expr: LoweredReductionIR) -> Optional[Any]:
        plan_id = getattr(expr, "kernel_plan_id", None)
        if plan_id is None:
            return None
        analysis = self._lowered_execution_analysis()
        return (analysis.get("reduction_kernel_plans_by_id") or {}).get(plan_id)

    def _lowered_einstein_clause_facts(self, clause: Any) -> Optional[Any]:
        facts_id = getattr(clause, "execution_facts_id", None)
        if facts_id is None:
            return None
        analysis = self._lowered_execution_analysis()
        return (analysis.get("clause_facts_by_id") or {}).get(facts_id)

    def _lowered_select_at_argmax_facts(self, expr: Any) -> Optional[Any]:
        facts_id = getattr(expr, "execution_facts_id", None)
        if facts_id is None:
            return None
        analysis = self._lowered_execution_analysis()
        return (analysis.get("select_facts_by_id") or {}).get(facts_id)

    def _lowered_einstein_facts(self, expr: Any) -> Optional[Any]:
        facts_id = getattr(expr, "execution_facts_id", None)
        if facts_id is None:
            return None
        analysis = self._lowered_execution_analysis()
        return (analysis.get("einstein_facts_by_id") or {}).get(facts_id)

    def _analysis_cache_bucket(self, name: str) -> Dict[Any, Any]:
        cache = getattr(self, "_analysis_cache", None)
        if cache is None:
            cache = {}
            self._analysis_cache = cache
        bucket = cache.get(name)
        if bucket is None:
            bucket = {}
            cache[name] = bucket
        return bucket

    def _cached_contains_ir_types(self, node: Any, *node_types: type) -> bool:
        if node is None:
            return False
        bucket = self._analysis_cache_bucket("contains_ir_types")
        key = (id(node), tuple(t.__name__ for t in node_types))
        hit = bucket.get(key)
        if hit is None:
            hit = _contains_ir_node_type(node, *node_types)
            bucket[key] = hit
        return hit

    def _cached_defids_by_name(self, node: Any) -> Dict[str, List[Any]]:
        if node is None:
            return {}
        bucket = self._analysis_cache_bucket("defids_by_name")
        key = id(node)
        hit = bucket.get(key)
        if hit is None:
            from .numpy_einstein_call_index_analysis import _collect_defids_by_name

            hit = _collect_defids_by_name(node)
            bucket[key] = hit
        return hit

    def _cached_count_reduction_dims(self, node: Any) -> int:
        if node is None:
            return 0
        bucket = self._analysis_cache_bucket("count_reduction_dims")
        key = id(node)
        hit = bucket.get(key)
        if hit is None:
            hit = _count_reduction_dims_in_expr(node)
            bucket[key] = hit
        return hit

    def _cached_reduction_uses_clause_var_in_bounds(self, node: Any, clause_loop_defids: List[Any]) -> bool:
        if node is None:
            return False
        bucket = self._analysis_cache_bucket("reduction_uses_clause_var_in_bounds")
        key = (id(node), tuple(clause_loop_defids))
        hit = bucket.get(key)
        if hit is None:
            hit = _reduction_uses_clause_var_in_bounds(node, clause_loop_defids)
            bucket[key] = hit
        return hit

    def _cached_loop_range(self, loop: Any, evaluator: Any) -> Tuple[int, int]:
        dep_bucket = self._analysis_cache_bucket("loop_range_depids")
        loop_key = id(loop)
        depids = dep_bucket.get(loop_key)
        if depids is None:
            iterable = getattr(loop, "iterable", None)
            depids = tuple(
                sorted(
                    (did for did in _collect_ir_defids(iterable) if did is not None),
                    key=lambda d: (d.krate, d.index),
                )
            )
            dep_bucket[loop_key] = depids
        bucket = self._analysis_cache_bucket("loop_range")
        fingerprint = tuple(
            (did, self._cache_value_fingerprint(self.env.get_value(did)))
            for did in depids
        )
        key = (loop_key, fingerprint)
        hit = bucket.get(key)
        if hit is None:
            from .numpy_einstein_recurrence_analysis import _extract_loop_range

            hit = _extract_loop_range(loop, evaluator)
            bucket[key] = hit
        return hit

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

    def _block_needs_per_outer_slot(self, expr: BlockExpressionIR) -> bool:
        parallel_defids = {
            did for did in (self._vectorization_parallel_defids_order() or ()) if did is not None
        }
        if not parallel_defids:
            return False
        for stmt in (expr.statements or []) or []:
            binding = getattr(stmt, "_binding", None)
            if binding is None and isinstance(stmt, BindingIR):
                binding = stmt
            binding_expr = getattr(binding, "expr", None) if binding is not None else None
            if binding_expr is None:
                continue
            if not self._cached_contains_ir_types(
                binding_expr,
                LoweredEinsteinIR,
                LoweredReductionIR,
                LoweredSelectAtArgmaxIR,
            ):
                continue
            if any(_BodyReferencesDefidVisitor(d).references(binding_expr) for d in parallel_defids):
                return True
        final_expr = getattr(expr, "final_expr", None)
        if final_expr is not None and self._cached_contains_ir_types(
            final_expr,
            LoweredEinsteinIR,
            LoweredReductionIR,
            LoweredSelectAtArgmaxIR,
        ):
            if any(_BodyReferencesDefidVisitor(d).references(final_expr) for d in parallel_defids):
                return True
        return False

    def _evaluate_block_per_outer_slot(self, expr: BlockExpressionIR) -> Any:
        parallel_shape = self._vectorization_parallel_shape()
        if not parallel_shape:
            return None
        outer_shape = tuple(int(dim) for dim in parallel_shape)
        outer_names = getattr(self.env, "_defid_names", {}) or {}
        parallel_defids = [
            did for did in (self._vectorization_parallel_defids_order() or ()) if did is not None
        ]
        if not parallel_defids:
            return None
        scalarizable: List[Tuple[Any, np.ndarray]] = []
        for did in parallel_defids:
            cur = self.env.get_value(did)
            if not isinstance(cur, np.ndarray):
                continue
            if cur.ndim == 0 or not np.issubdtype(cur.dtype, np.integer):
                continue
            arr = np.asarray(cur)
            try:
                arr = np.broadcast_to(arr, outer_shape)
            except ValueError:
                if arr.size != int(np.prod(outer_shape)):
                    continue
                try:
                    arr = np.reshape(arr, outer_shape)
                except ValueError:
                    continue
            scalarizable.append((did, np.asarray(arr, dtype=np.intp).reshape(-1)))
        if not scalarizable:
            return None
        result = None
        for linear_idx, outer_idx in enumerate(np.ndindex(outer_shape)):
            with self.env.scope():
                for did, flat_values in scalarizable:
                    self.env.set_value(did, int(flat_values[linear_idx]), name=outer_names.get(did))
                with self._vectorization_scope(
                    parallel_shape=None,
                    parallel_defids_order=None,
                ):
                    cell = self.visit_block_expression(expr)
            cell_arr = np.asarray(cell)
            if result is None:
                result_shape = tuple(cell_arr.shape) + outer_shape
                result = np.empty(result_shape, dtype=cell_arr.dtype)
            result[(Ellipsis,) + outer_idx] = cell_arr
        return result

    def _lowered_expr_is_zero(self, expr: Any) -> bool:
        if expr is None:
            return False
        if _is_zero_literal_like(expr):
            return True
        if isinstance(expr, CastExpressionIR):
            return self._lowered_expr_is_zero(expr.expr)
        if isinstance(expr, BlockExpressionIR) and expr.final_expr is not None:
            return self._lowered_expr_is_zero(expr.final_expr)
        if isinstance(expr, BinaryOpIR):
            if expr.operator == BinaryOp.ADD:
                return self._lowered_expr_is_zero(expr.left) and self._lowered_expr_is_zero(expr.right)
            if expr.operator == BinaryOp.SUB:
                return self._lowered_expr_is_zero(expr.left) and self._lowered_expr_is_zero(expr.right)
        if isinstance(expr, LoweredReductionIR):
            return expr.operation == ReductionOp.SUM and self._lowered_expr_is_zero(expr.body)
        if isinstance(expr, LoweredEinsteinIR):
            items = list(expr.items or [])
            return bool(items) and all(self._lowered_expr_is_zero(item.body) for item in items)
        return False

    def _zero_value_for_lowered_expr(self, expr: Any) -> Any:
        if expr is None:
            return None
        if isinstance(expr, LiteralIR):
            return expr.value
        if isinstance(expr, CastExpressionIR):
            return self._zero_value_for_lowered_expr(expr.expr)
        if isinstance(expr, BlockExpressionIR) and expr.final_expr is not None:
            return self._zero_value_for_lowered_expr(expr.final_expr)
        if isinstance(expr, LoweredReductionIR):
            return self._zero_value_for_lowered_expr(expr.body)
        if isinstance(expr, LoweredEinsteinIR):
            shape_fn = getattr(self, "_resolve_lowered_output_shape", None)
            dtype_fn = getattr(self, "_type_info_to_numpy_dtype", None)
            output_shape = shape_fn(expr) if callable(shape_fn) else None
            if not output_shape:
                return None
            dtype = dtype_fn(expr.element_type) if callable(dtype_fn) else None
            if dtype is None:
                dtype = np.float32
            return np.zeros(tuple(int(v) for v in output_shape), dtype=dtype)
        return None

    def _binding_cache_depids(self, binding: BindingIR) -> Tuple[Any, ...]:
        cache = getattr(self, "_binding_cache_depids_by_defid", None)
        if cache is None:
            cache = {}
            self._binding_cache_depids_by_defid = cache
        if binding.defid in cache:
            return cache[binding.defid]
        depids = tuple(
            sorted(
                (d for d in _collect_ir_defids(binding.expr) if d is not None),
                key=lambda d: (d.krate, d.index),
            )
        )
        cache[binding.defid] = depids
        return depids

    @staticmethod
    def _cache_value_fingerprint(value: Any) -> Any:
        if value is None or isinstance(value, (int, float, bool, str, bytes)):
            return value
        if isinstance(value, np.ndarray):
            ptr = None
            try:
                ptr = value.__array_interface__.get("data", (None,))[0]
            except Exception:
                pass
            return ("ndarray", id(value), ptr, tuple(value.shape), str(value.dtype))
        if isinstance(value, tuple):
            return ("tuple", tuple(ExpressionVisitorMixin._cache_value_fingerprint(v) for v in value))
        if isinstance(value, list):
            return ("list", tuple(ExpressionVisitorMixin._cache_value_fingerprint(v) for v in value))
        return (type(value).__name__, id(value))

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
            last_values = getattr(self.env, "_last_values", {}) or {}
            if defid in last_values:
                return last_values[defid]
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
        lowered_array = isinstance(expr.array, LoweredEinsteinIR)
        indices = [idx.accept(self) for idx in (expr.indices or []) if idx is not None]
        if not lowered_array and isinstance(expr.array, IfExpressionIR):
            cond = expr.array.condition.accept(self)
            cond_scalar = None
            if isinstance(cond, np.ndarray):
                cond_indexed = None
                if indices:
                    try:
                        cond_indexed = cond[tuple(indices)]
                    except Exception:
                        cond_indexed = None
                if isinstance(cond_indexed, np.ndarray):
                    if cond_indexed.ndim == 0 or cond_indexed.size == 1:
                        cond_scalar = self._to_bool(cond_indexed.reshape(-1)[0])
                    elif bool(cond_indexed.all()):
                        cond_scalar = True
                    elif not bool(cond_indexed.any()):
                        cond_scalar = False
                elif cond_indexed is not None:
                    cond_scalar = self._to_bool(cond_indexed)
                elif cond.ndim == 0 or cond.size == 1:
                    cond_scalar = self._to_bool(cond.reshape(-1)[0])
            else:
                cond_scalar = self._to_bool(cond)
            if cond_scalar is not None:
                branch = expr.array.then_expr if cond_scalar else expr.array.else_expr
                if branch is None:
                    return None
                return RectangularAccessIR(
                    branch,
                    list(expr.indices or []),
                    expr.location,
                    type_info=expr.type_info,
                    shape_info=expr.shape_info,
                ).accept(self)
        if not lowered_array and isinstance(expr.array, BlockExpressionIR) and expr.array.final_expr is not None:
            block = expr.array
            final_access = RectangularAccessIR(
                block.final_expr,
                list(expr.indices or []),
                expr.location,
                type_info=expr.type_info,
                shape_info=expr.shape_info,
            )
            with self.env.scope():
                previous_binding_cache = getattr(self, "_binding_eval_cache", None)
                self._binding_eval_cache = {}
                previous_nested_lowered_cache = getattr(self, "_nested_lowered_eval_cache", None)
                self._nested_lowered_eval_cache = {}
                try:
                    binding_cache = self._binding_eval_cache
                    for stmt in (block.statements or []) or []:
                        binding = getattr(stmt, "_binding", None)
                        if binding is None and isinstance(stmt, BindingIR):
                            binding = stmt
                        result_value = None
                        binding_expr = binding.expr if binding is not None else None
                        use_binding_cache = bool(
                            binding is not None
                            and binding.defid is not None
                            and binding_expr is not None
                            and _contains_ir_node_type(
                                binding_expr,
                                LoweredEinsteinIR,
                                LoweredReductionIR,
                                LoweredSelectAtArgmaxIR,
                            )
                        )
                        if use_binding_cache and binding is not None and binding.defid is not None and binding.expr is not None:
                            depids = self._binding_cache_depids(binding)
                            dep_key = tuple(
                                (depid, self._cache_value_fingerprint(self.env.get_value(depid)))
                                for depid in depids
                            )
                            cache_key = (binding.defid, dep_key)
                            cached = binding_cache.get(cache_key)
                            if cached is not None:
                                result_value = cached
                            else:
                                result_value = stmt.accept(self)
                                binding_cache[cache_key] = result_value
                        else:
                            result_value = stmt.accept(self)
                        variable_defid = binding.defid if binding else None
                        if variable_defid is not None:
                            var_name = binding.name or stmt.name
                            if result_value is None and binding.expr is not None and not is_function_binding(binding):
                                result_value = stmt.accept(self)
                            if result_value is None and is_function_binding(binding):
                                result_value = self.env.get_value(variable_defid)
                            self.env.set_value(variable_defid, result_value, name=var_name)
                    return final_access.accept(self)
                finally:
                    if previous_binding_cache is None:
                        delattr(self, "_binding_eval_cache")
                    else:
                        self._binding_eval_cache = previous_binding_cache
                    if previous_nested_lowered_cache is None:
                        delattr(self, "_nested_lowered_eval_cache")
                    else:
                        self._nested_lowered_eval_cache = previous_nested_lowered_cache
        if lowered_array:
            lowered_rank = 0
            shape = getattr(expr.array, "shape", None)
            items = getattr(expr.array, "items", None) or []
            if isinstance(shape, (list, tuple)) and len(shape) > 0:
                lowered_rank = len(shape)
            elif items:
                first_indices = getattr(items[0], "indices", None) or []
                lowered_rank = len(first_indices)
            scalar_index_bindings = {}
            for idx_node, idx_value in zip((expr.indices or []), indices):
                did = getattr(idx_node, "defid", None)
                if did is None:
                    continue
                try:
                    idx_array = np.asarray(idx_value)
                    if idx_array.ndim == 0:
                        scalar_index_bindings[did] = int(idx_array.item())
                    else:
                        scalar_index_bindings[did] = idx_value
                except Exception:
                    scalar_index_bindings[did] = idx_value
            all_scalar_indices = bool(indices) and all(
                not isinstance(np.asarray(idx), np.ndarray) or np.asarray(idx).ndim == 0 or np.asarray(idx).size == 1
                for idx in indices
            )
            lowered_einstein_facts = self._lowered_einstein_facts(expr.array)
            contains_select = (
                bool(getattr(lowered_einstein_facts, "contains_select_at_argmax", False))
                if lowered_einstein_facts is not None
                else self._cached_contains_ir_types(expr.array, LoweredSelectAtArgmaxIR)
            )
            if len(indices) == lowered_rank and all_scalar_indices and not contains_select:
                direct_cell = self._evaluate_lowered_einstein_at_indices(expr.array, indices)
                if direct_cell is not None:
                    return direct_cell
            array = self._evaluate_lowered_einstein_subexpr(
                expr.array,
                allow_outer_slot_eval=len(indices) > lowered_rank,
                scalar_index_bindings=scalar_index_bindings,
            )
        else:
            array = expr.array.accept(self)
        try:
            if getattr(array, "is_circular_recurrence_buffer", False):
                return array[tuple(indices)]
            if isinstance(array, np.ndarray):
                if array.size == 0 and isinstance(expr.array, EinsteinIR):
                    clauses = list(getattr(expr.array, "clauses", ()) or ())
                    if len(clauses) == 1 and getattr(clauses[0], "value", None) is not None:
                        return clauses[0].value.accept(self)
                if array.dtype == object and any(
                    isinstance(raw_idx, np.ndarray) and raw_idx.ndim > 0 for raw_idx in indices
                ):
                    idx_arrays = [np.asarray(raw_idx) for raw_idx in indices]
                    broadcasted = np.broadcast_arrays(*idx_arrays)
                    gathered = np.empty(broadcasted[0].shape, dtype=object)
                    for pos in np.ndindex(gathered.shape):
                        key = tuple(int(arr[pos]) for arr in broadcasted)
                        gathered[pos] = array[key]
                    return gathered
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
            if hasattr(array, "__getitem__") and hasattr(array, "shape"):
                key = tuple(
                    int(raw_idx) if isinstance(raw_idx, (np.integer, int, float)) else raw_idx
                    for raw_idx in indices
                )
                return array[key]
            # Forward AD: seeded derivative may be scalar 1 (broadcast over indices)
            if np.isscalar(array) or (isinstance(array, np.ndarray) and array.ndim == 0):
                return array
        except (IndexError, KeyError, TypeError) as e:
            detail = RuntimeError(
                f"{e} | array_type={type(array).__name__} array_shape={getattr(array, 'shape', None)} indices={indices} array_name={getattr(expr.array, 'name', None)}"
            )
            self._raise_here(detail, expr)
        raise RuntimeError(f"rectangular_access: expected ndarray, list, or str, got {type(array).__name__}")

    def visit_jagged_access(self, expr: JaggedAccessIR) -> Any:
        array = expr.base.accept(self)
        for idx in (expr.index_chain or []):
            array = array[idx.accept(self)]
        return array

    def visit_block_expression(self, expr: BlockExpressionIR) -> Any:
        if (
            expr.statements
            and self._vectorization_parallel_shape() is not None
            and not self._in_recurrence_vectorization_clause()
            and self._block_needs_per_outer_slot(expr)
        ):
            per_slot = self._evaluate_block_per_outer_slot(expr)
            if per_slot is not None:
                return per_slot
        with self.env.scope():
            # Keep binding-result caching local to this block invocation. Reusing it
            # across function calls lets stale local bindings leak when dependency
            # tracking misses a scalar parameter, which corrupts repeated stdlib calls.
            previous_binding_cache = getattr(self, "_binding_eval_cache", None)
            self._binding_eval_cache = {}
            previous_nested_lowered_cache = getattr(self, "_nested_lowered_eval_cache", None)
            self._nested_lowered_eval_cache = {}
            try:
                binding_cache = self._binding_eval_cache
                for stmt in (expr.statements or []) or []:
                    binding = getattr(stmt, "_binding", None)
                    if binding is None and isinstance(stmt, BindingIR):
                        binding = stmt
                    result_value = None
                    used_cache = False
                    cacheable_name = (binding.name or stmt.name) if binding is not None else None
                    binding_expr = binding.expr if binding is not None else None
                    use_binding_cache = bool(
                        binding is not None
                        and binding.defid is not None
                        and binding_expr is not None
                        and _contains_ir_node_type(
                            binding_expr,
                            LoweredEinsteinIR,
                            LoweredReductionIR,
                            LoweredSelectAtArgmaxIR,
                        )
                    )
                    if use_binding_cache and binding is not None and binding.defid is not None and binding.expr is not None:
                        depids = self._binding_cache_depids(binding)
                        dep_key = tuple(
                            (depid, self._cache_value_fingerprint(self.env.get_value(depid)))
                            for depid in depids
                        )
                        cache_key = (binding.defid, dep_key)
                        cached = binding_cache.get(cache_key)
                        if cached is not None:
                            result_value = cached
                            used_cache = True
                        else:
                            result_value = stmt.accept(self)
                            binding_cache[cache_key] = result_value
                    else:
                        result_value = stmt.accept(self)
                    variable_defid = binding.defid if binding else None
                    if variable_defid is not None:
                        var_name = binding.name or stmt.name
                        if result_value is None and binding.expr is not None and not is_function_binding(binding):
                            result_value = stmt.accept(self)
                        if result_value is None and is_function_binding(binding):
                            result_value = self.env.get_value(variable_defid)
                        if var_name in {"Xp", "conv_sum", "output", "_@Xp", "_@conv_sum", "_@output"}:
                            emit_debug_log(
                                "runtime.autodiff_probe",
                                "numpy_expressions_mixin.py:visit_block_expression",
                                "local_binding_value",
                                {
                                    "binding_name": var_name,
                                    "used_cache": used_cache,
                                    "shape": list(np.asarray(result_value).shape) if result_value is not None else None,
                                    "value": np.asarray(result_value).tolist() if result_value is not None else None,
                                },
                            )
                        self.env.set_value(variable_defid, result_value, name=var_name)
                if expr.final_expr:
                    return expr.final_expr.accept(self)
            finally:
                if previous_binding_cache is None:
                    delattr(self, "_binding_eval_cache")
                else:
                    self._binding_eval_cache = previous_binding_cache
                if previous_nested_lowered_cache is None:
                    delattr(self, "_nested_lowered_eval_cache")
                else:
                    self._nested_lowered_eval_cache = previous_nested_lowered_cache
        return None

    def visit_if_expression(self, expr) -> Any:
        cond = expr.condition.accept(self)
        if isinstance(cond, np.ndarray):
            if self._vectorization_safe_oob_enabled():
                then_val = expr.then_expr.accept(self)
                if expr.else_expr:
                    else_val = expr.else_expr.accept(self)
                else:
                    if isinstance(then_val, np.ndarray):
                        else_val = np.zeros_like(then_val)
                    else:
                        else_val = 0
            else:
                then_val = expr.then_expr.accept(self)
                if expr.else_expr:
                    else_val = expr.else_expr.accept(self)
                else:
                    if isinstance(then_val, np.ndarray):
                        else_val = np.zeros_like(then_val)
                    else:
                        else_val = 0
            target_ndim = cond.ndim
            if isinstance(then_val, np.ndarray):
                target_ndim = max(target_ndim, then_val.ndim)
            if isinstance(else_val, np.ndarray):
                target_ndim = max(target_ndim, else_val.ndim)
            if cond.ndim < target_ndim:
                cond = cond.reshape(cond.shape + (1,) * (target_ndim - cond.ndim))
            if isinstance(then_val, np.ndarray) and then_val.ndim < target_ndim:
                then_val = then_val.reshape(then_val.shape + (1,) * (target_ndim - then_val.ndim))
            if isinstance(else_val, np.ndarray) and else_val.ndim < target_ndim:
                else_val = else_val.reshape(else_val.shape + (1,) * (target_ndim - else_val.ndim))
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
        loop_vars = list(getattr(expr, "loop_vars", None) or [])
        ranges = list(getattr(expr, "ranges", None) or [])
        constraints = list(getattr(expr, "constraints", None) or [])
        if not loop_vars:
            return expr.body.accept(self)

        iterables: List[List[Any]] = []
        for rng in ranges:
            value = rng.accept(self)
            if isinstance(value, range):
                iterables.append(list(value))
            else:
                iterables.append(list(value))

        def evaluate(depth: int) -> Any:
            if depth >= len(loop_vars):
                for constraint in constraints:
                    if not self._to_bool(constraint.accept(self)):
                        return None
                return expr.body.accept(self)
            var = loop_vars[depth]
            values = iterables[depth] if depth < len(iterables) else []
            out: List[Any] = []
            for item in values:
                with self.env.scope():
                    if getattr(var, "defid", None) is not None:
                        self.env.set_value(var.defid, item, name=getattr(var, "name", None))
                    out.append(evaluate(depth + 1))
            return out

        evaluated = evaluate(0)
        type_info = getattr(expr, "type_info", None)
        dtype = None
        converter = getattr(self, "_type_info_to_numpy_dtype", None)
        if callable(converter) and type_info is not None:
            dtype = converter(getattr(type_info, "element_type", None) or type_info)
        if dtype is not None:
            return np.array(evaluated, dtype=dtype)
        return np.array(evaluated)

    def visit_einstein(self, expr: EinsteinIR) -> Any:
        from ..runtime.compute.lowered_execution import execute_lowered_loops

        def ev(e): return e.accept(self)

        shape: Optional[Tuple[int, ...]] = None
        if expr.shape is not None:
            shape = tuple(int(np.asarray(dim.accept(self)).reshape(-1)[0]) for dim in (expr.shape or ()))
        elif isinstance(expr.shape_info, tuple) and all(isinstance(dim, int) for dim in expr.shape_info):
            shape = tuple(int(dim) for dim in expr.shape_info)
        else:
            type_shape = _concrete_shape_from_type_info(getattr(expr, "type_info", None))
            if type_shape is not None:
                shape = type_shape
        if shape is None:
            inferred: List[Optional[int]] = []
            for clause in expr.clauses or ():
                for axis, idx in enumerate(clause.indices or ()):
                    dim_value: Optional[int] = None
                    rng = None
                    if isinstance(idx, IndexVarIR):
                        rng = (clause.variable_ranges or {}).get(getattr(idx, "defid", None)) or getattr(idx, "range_ir", None)
                    if isinstance(rng, RangeIR):
                        try:
                            start = int(np.asarray(rng.start.accept(self)).reshape(-1)[0])
                            end = int(np.asarray(rng.end.accept(self)).reshape(-1)[0])
                            dim_value = end - start + (1 if getattr(rng, "inclusive", False) else 0)
                        except Exception:
                            dim_value = None
                    elif isinstance(idx, LiteralIR) and isinstance(idx.value, int):
                        dim_value = int(idx.value) + 1
                    if dim_value is None:
                        continue
                    while len(inferred) <= axis:
                        inferred.append(None)
                    current = inferred[axis]
                    inferred[axis] = dim_value if current is None else max(current, dim_value)
            if inferred and all(dim is not None for dim in inferred):
                shape = tuple(int(dim) for dim in inferred if dim is not None)
        if shape is None:
            direct_only = bool(expr.clauses)
            for clause in expr.clauses or ():
                if not all(isinstance(idx, LiteralIR) for idx in (clause.indices or ())):
                    direct_only = False
                    break
            if direct_only:
                direct_result = None
                for clause in expr.clauses or ():
                    with self.env.scope():
                        if clause.where_clause is not None:
                            guards_ok = all(
                                self._to_bool(constraint.accept(self))
                                for constraint in (clause.where_clause.constraints or ())
                            )
                            if not guards_ok:
                                continue
                        value = clause.value.accept(self)
                    if value is None:
                        continue
                    direct_result = value if direct_result is None else direct_result + value
                if direct_result is not None:
                    return direct_result
            raise RuntimeError(
                "EinsteinIR missing concrete shape metadata; expected expr.shape, "
                "expr.shape_info, or concrete RectangularType.shape"
            )
        dtype = None
        converter = getattr(self, "_type_info_to_numpy_dtype", None)
        if callable(converter):
            tensor_type = getattr(expr, "type_info", None)
            if tensor_type is not None:
                dtype = converter(tensor_type)
            if dtype is None:
                dtype = converter(getattr(expr, "element_type", None))
        if dtype is None:
            dtype = np.float64

        if not shape:
            result = None
            for clause in expr.clauses or ():
                with self.env.scope():
                    if clause.where_clause is not None:
                        guards_ok = all(self._to_bool(constraint.accept(self)) for constraint in (clause.where_clause.constraints or ()))
                        if not guards_ok:
                            continue
                    result = clause.value.accept(self)
            return result

        output = np.zeros(shape, dtype=dtype)
        for clause in expr.clauses or ():
            ordered_loops = []
            for idx in clause.indices or ():
                if isinstance(idx, IndexVarIR) and idx.defid is not None:
                    iterable = clause.variable_ranges.get(idx.defid) or getattr(idx, "range_ir", None)
                    if iterable is not None:
                        ordered_loops.append(SimpleNamespace(variable=idx, iterable=iterable))
            loop_contexts = execute_lowered_loops(ordered_loops, {}, ev) if ordered_loops else [dict()]
            for context in loop_contexts:
                with self.env.scope():
                    for defid, value in context.items():
                        if defid is not None:
                            self.env.set_value(defid, value)
                    if clause.where_clause is not None:
                        guards_ok = all(
                            self._to_bool(constraint.accept(self))
                            for constraint in (clause.where_clause.constraints or ())
                        )
                        if not guards_ok:
                            continue
                    idx_tuple = tuple(int(np.asarray(idx.accept(self)).reshape(-1)[0]) for idx in (clause.indices or ()))
                    value = clause.value.accept(self)
                    arr = np.asarray(value, dtype=output.dtype)
                    if output.ndim == 0 and arr.ndim > 0:
                        return arr
                    if arr.size == 1:
                        output[idx_tuple] = arr.reshape(-1)[0]
                        continue
                    if arr.shape == output.shape:
                        output[...] = arr
                        return output
                    target = output[idx_tuple] if idx_tuple else output
                    if isinstance(target, np.ndarray) and arr.shape == target.shape:
                        output[idx_tuple] = arr
                        continue
                    if (
                        arr.ndim > 0
                        and output.ndim >= arr.ndim
                        and arr.shape == output.shape[output.ndim - arr.ndim :]
                    ):
                        prefix_len = output.ndim - arr.ndim
                        prefix = idx_tuple[:prefix_len]
                        suffix = idx_tuple[prefix_len:]
                        if len(prefix) < len(idx_tuple) and all(int(v) == 0 for v in suffix):
                            output[prefix] = arr
                            continue
                    if isinstance(target, np.ndarray):
                        try:
                            output[idx_tuple] = np.broadcast_to(arr, target.shape)
                            continue
                        except ValueError:
                            pass
                    raise RuntimeError(
                        f"visit_einstein: clause value shape {arr.shape!r} does not fit target "
                        f"{getattr(target, 'shape', ())!r} at indices {idx_tuple!r}"
                    )
        return output

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

    def _project_object_array_member(self, array: np.ndarray, member_index: int) -> Any:
        projected = np.empty(array.shape, dtype=object)
        for idx in np.ndindex(array.shape):
            projected[idx] = array[idx][member_index]
        if projected.size == 0:
            return projected
        first = projected.reshape(-1)[0]
        flat = [projected[idx] for idx in np.ndindex(projected.shape)]
        if isinstance(first, np.ndarray) and all(
            isinstance(value, np.ndarray) and value.shape == first.shape for value in flat
        ):
            return np.stack(flat, axis=0).reshape(projected.shape + first.shape)
        try:
            return np.asarray(projected.tolist())
        except Exception:
            return projected

    def visit_tuple_access(self, expr: TupleAccessIR) -> Any:
        t = expr.tuple_expr.accept(self)
        if isinstance(t, np.ndarray) and t.dtype == object:
            return self._project_object_array_member(t, expr.index)
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
            if member.isdigit():
                member_index = int(member)
                if isinstance(obj, np.ndarray) and obj.dtype == object:
                    return self._project_object_array_member(obj, member_index)
                if isinstance(obj, (list, tuple, np.ndarray)):
                    return obj[member_index]
            if member == "length" and isinstance(obj, (np.ndarray, list, tuple, str)):
                return len(obj)
            if member == "size" and isinstance(obj, np.ndarray):
                return obj.size
            if member == "shape" and isinstance(obj, np.ndarray):
                return obj.shape
            return getattr(obj, member, None)
        if isinstance(member, int):
            if isinstance(obj, np.ndarray) and obj.dtype == object:
                return self._project_object_array_member(obj, member)
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
        from ..runtime.compute.lowered_execution import execute_reduction_with_loops

        def ev(e): return e.accept(self)

        ordered_loops = []
        reduction_ranges = {}
        for loop_var in (expr.loop_vars or []) or []:
            loop_defid = getattr(loop_var, "defid", None)
            iterable = None
            if loop_defid is not None:
                iterable = (expr.loop_var_ranges or {}).get(loop_defid)
            if iterable is None:
                iterable = getattr(loop_var, "range_ir", None)
            if iterable is None or loop_defid is None:
                continue
            loop = SimpleNamespace(variable=loop_var, iterable=iterable)
            ordered_loops.append(loop)
            reduction_ranges[loop_defid] = loop

        def body_ev(context):
            with self.env.scope():
                for defid, value in context.items():
                    if defid is not None:
                        self.env.set_value(defid, value)
                return expr.body.accept(self)

        guard_exprs = tuple(getattr(expr.where_clause, "constraints", ()) or ()) if getattr(expr, "where_clause", None) is not None else ()

        def guard_ev(context):
            if not guard_exprs:
                return True
            with self.env.scope():
                for defid, value in context.items():
                    if defid is not None:
                        self.env.set_value(defid, value)
                for guard in guard_exprs:
                    if not self._to_bool(guard.accept(self)):
                        return False
            return True

        return execute_reduction_with_loops(
            expr.operation,
            reduction_ranges,
            body_ev,
            ev,
            guard_evaluator=guard_ev if guard_exprs else None,
            reduction_loops_ordered=ordered_loops,
        )

    def _reduction_loop_defid_alias_maps(
        self, expr: LoweredReductionIR, reduction_facts: Any
    ) -> Tuple[Dict[Any, List[Any]], Dict[Any, str]]:
        loop_to_body_defids: Dict[Any, List[Any]] = {}
        reduction_defid_names: Dict[Any, str] = {}
        body_defids_by_name = dict(getattr(reduction_facts, "body_defids_by_name", {}) or {})
        for _lp in expr.loops or []:
            _v = _lp.variable
            if _v is not None and _v.defid is not None and _v.name:
                _vname = _v.name
                _all_bds = set(body_defids_by_name.get(_vname, ()))
                _bds = [d for d in _all_bds if d != _v.defid]
                if _bds:
                    loop_to_body_defids[_v.defid] = _bds
                for _bd in _all_bds:
                    reduction_defid_names[_bd] = _vname
                reduction_defid_names[_v.defid] = _vname
        return loop_to_body_defids, reduction_defid_names

    def evaluate_lowered_reduction(
        self, expr: LoweredReductionIR, parallel_shape: Optional[Tuple[int, ...]] = None
    ) -> Any:
        """Evaluate a lowered reduction, optionally with vectorized path when parallel_shape is set.
        When parallel_shape is None, uses the backend vectorization state when present (e.g. in a vectorized clause).
        Fast paths (matmul, conv via einsum) only when parallel_shape is set; stricter conditions avoid LSTM."""
        import os
        if expr.operation == ReductionOp.SUM and self._lowered_expr_is_zero(expr.body):
            zero_value = self._zero_value_for_lowered_expr(expr.body)
            if zero_value is not None:
                return zero_value
        reduction_facts = self._lowered_reduction_facts(expr)
        force_scalar_reduction = bool(
            getattr(reduction_facts, "contains_nested_reduction_or_select", False)
            or getattr(reduction_facts, "contains_if_expression", False)
        )
        if parallel_shape is None:
            parallel_shape = self._vectorization_parallel_shape()
        _loop_alias_map, _reduction_defid_names = self._reduction_loop_defid_alias_maps(expr, reduction_facts)
        # Only block windowed/matmul/einsum and speculative vectorization when the same loop
        # variable name is tied to multiple *distinct* body defids (e.g. autodiff primal+diff).
        # A single alternate defid vs. the loop header still needs ctx expansion below but is
        # safe for the same structural fast paths as a unified defid.
        has_defid_aliases = any(len(alist) > 1 for alist in _loop_alias_map.values())
        # Recurrence clauses may use partial vectorization but must not use fast_matmul / fast_conv.
        if (
            parallel_shape is not None
            and not force_scalar_reduction
            and not self._in_recurrence_vectorization_clause()
            and not has_defid_aliases
        ):
            plan = self._lowered_reduction_kernel_plan(expr)
            kind = getattr(plan, "kind", None)
            if kind == "windowed_sumprod":
                windowed_result = _try_windowed_sumprod_einsum(expr, self, plan)
                if windowed_result is not None and isinstance(windowed_result, np.ndarray):
                    if windowed_result.shape == tuple(parallel_shape):
                        setattr(self, "_last_reduction_fast_path", "windowed-einsum")
                        return windowed_result
            elif kind == "matmul_sumprod":
                matmul_result = _try_matmul_reduction(expr, self, plan)
                if matmul_result is not None and isinstance(matmul_result, np.ndarray):
                    if matmul_result.shape == tuple(parallel_shape):
                        setattr(self, "_last_reduction_fast_path", "matmul")
                        return matmul_result
                einsum_result = _try_einsum_reduction(expr, self, plan)
                if einsum_result is not None and isinstance(einsum_result, np.ndarray):
                    if einsum_result.shape == tuple(parallel_shape):
                        setattr(self, "_last_reduction_fast_path", "einsum")
                        return einsum_result
            elif kind == "einsum_sumprod":
                einsum_result = _try_einsum_reduction(expr, self, plan)
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

        _loop_to_all_body_defids = _loop_alias_map
        _reduction_body_defids_with_aliases = set()
        _reduction_loop_defids = set()
        for loop in (expr.loops or []):
            if loop.variable is not None and loop.variable.defid is not None:
                d = loop.variable.defid
                _reduction_loop_defids.add(d)
                _reduction_body_defids_with_aliases.add(d)
                for bd in _loop_to_all_body_defids.get(d) or []:
                    _reduction_body_defids_with_aliases.add(bd)

        def _expand_reduction_ctx(ctx: Dict[Any, Any]) -> Dict[Any, Any]:
            out: Dict[Any, Any] = {}
            for loop_defid, val in (ctx or {}).items():
                if loop_defid is None:
                    continue
                out[loop_defid] = val
                for body_defid in _loop_to_all_body_defids.get(loop_defid) or []:
                    if body_defid in out:
                        continue
                    out[body_defid] = val
            return out

        body_defids_by_name = {
            name: list(defids)
            for name, defids in (getattr(reduction_facts, "body_defids_by_name", {}) or {}).items()
        }
        has_cross_scope_name_shadow = any(
            any(d in _reduction_body_defids_with_aliases for d in dids)
            and any(d not in _reduction_body_defids_with_aliases for d in dids)
            for dids in body_defids_by_name.values()
        )
        guard_defids_by_name: Dict[str, List[Any]] = {
            name: list(defids)
            for name, defids in (getattr(reduction_facts, "guard_defids_by_name", {}) or {}).items()
        }

        def _apply_name_aliases(ctx_map: Dict[Any, Any], target_defids_by_name: Dict[str, List[Any]]) -> Dict[Any, Any]:
            merged = dict(ctx_map)
            env_names = getattr(self.env, "_defid_names", {}) or {}
            name_to_value: Dict[str, Any] = {}
            for defid, val in merged.items():
                if defid is None:
                    continue
                # Prefer loop defids as the authoritative source for name -> value
                # so we can propagate their values to alias defids in the body.
                if defid in _reduction_body_defids_with_aliases and defid not in _reduction_loop_defids:
                    continue
                nm = _reduction_defid_names.get(defid) or env_names.get(defid)
                if nm and nm not in name_to_value:
                    name_to_value[nm] = val
            for nm, dids in target_defids_by_name.items():
                if nm not in name_to_value:
                    continue
                val = name_to_value[nm]
                for did in dids:
                    if did is not None and did not in merged:
                        merged[did] = val
            return merged

        def body_ev(ctx):
            _ctx = _apply_name_aliases(
                _expand_reduction_ctx(
                    {
                        defid: val
                        for defid, val in (ctx or {}).items()
                        if defid is not None
                    }
                ),
                body_defids_by_name,
            )
            saved: Dict[Any, Any] = {}
            for defid in _ctx:
                if defid is not None:
                    saved[defid] = self.env.get_value(defid)
            had_outer_ctx = hasattr(self, "_reduction_initial_context")
            prev_outer_ctx = getattr(self, "_reduction_initial_context", None)
            had_select_outer = hasattr(self, "_select_outer_index_defids")
            prev_select_outer = getattr(self, "_select_outer_index_defids", None)
            needs_outer_reduction_ctx = _contains_nested_lowered_reduction(expr.body) or isinstance(
                expr.body, LoweredSelectAtArgmaxIR
            )
            if needs_outer_reduction_ctx:
                outer_ctx = dict(prev_outer_ctx or {})
                outer_ctx.update(_ctx)
                setattr(self, "_reduction_initial_context", outer_ctx)
                setattr(
                    self,
                    "_select_outer_index_defids",
                    tuple(
                        loop.variable.defid
                        for loop in (expr.loops or [])
                        if loop.variable is not None and loop.variable.defid is not None
                    ),
                )
            try:
                for defid, val in _ctx.items():
                    if defid is not None:
                        self.env.set_value(defid, val, name=_reduction_defid_names.get(defid))
                return expr.body.accept(self)
            finally:
                if needs_outer_reduction_ctx:
                    if had_outer_ctx:
                        setattr(self, "_reduction_initial_context", prev_outer_ctx)
                    elif hasattr(self, "_reduction_initial_context"):
                        delattr(self, "_reduction_initial_context")
                    if had_select_outer:
                        setattr(self, "_select_outer_index_defids", prev_select_outer)
                    elif hasattr(self, "_select_outer_index_defids"):
                        delattr(self, "_select_outer_index_defids")
                for defid, val in saved.items():
                    self.env.set_value(defid, val, name=_reduction_defid_names.get(defid))
        def guard_ev(ctx):
            if not expr.guards:
                return True
            _ctx = _apply_name_aliases(
                _expand_reduction_ctx(
                    {
                        defid: val
                        for defid, val in (ctx or {}).items()
                        if defid is not None
                    }
                ),
                guard_defids_by_name,
            )
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

        referenced_defids = set(_collect_ir_defids(expr.body))
        for loop in (expr.loops or []):
            referenced_defids.update(_collect_ir_defids(loop.iterable))
        for g in (expr.guards or []):
            referenced_defids.update(_collect_ir_defids(g.condition))
        for did in referenced_defids:
            if did is None or did in _reduction_body_defids_with_aliases or did in initial_ctx:
                continue
            cur = self.env.get_value(did)
            if isinstance(cur, np.ndarray) and cur.ndim > 0:
                continue
            if isinstance(cur, (list, tuple)) and not np.isscalar(cur):
                continue
            if cur is not None:
                initial_ctx[did] = cur
        vector_parallel_ctx: Dict[Any, Any] = {}
        if (not initial_ctx) and parallel_shape is None:
            try:
                reduction_body_defids = set(_reduction_body_defids_with_aliases)
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
                n_loops = len(expr.loops or [])
                if n_loops >= 1:
                    for idx, did in enumerate(order_defids):
                        if did is None:
                            continue
                        cur = self.env.get_value(did)
                        if not isinstance(cur, np.ndarray):
                            continue
                        sz = int(parallel_shape[idx])
                        flat = np.asarray(cur, dtype=np.intp).reshape(-1)
                        start = int(flat[0]) if flat.size else 0
                        shape_pv = [1] * (len(parallel_shape) + n_loops)
                        shape_pv[idx] = sz
                        arr = np.arange(start, start + sz, dtype=np.intp).reshape(shape_pv)
                        vector_parallel_ctx[did] = arr
            else:
                try:
                    reduction_body_defids = set(_reduction_body_defids_with_aliases)
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

        _use_vectorized_guarded_lowered_reduction = not has_defid_aliases and not has_cross_scope_name_shadow
        if _use_vectorized_guarded_lowered_reduction and expr.guards:
            op = expr.operation
            if op in (ReductionOp.SUM, ReductionOp.PROD, ReductionOp.MIN, ReductionOp.MAX):
                try:
                    reduction_loops = list(expr.loops or [])
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
                            _ctx_g = _expand_reduction_ctx(
                                {
                                    defid: val
                                    for defid, val in ctx.items()
                                    if defid is not None
                                }
                            )
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
            reduction_loops_ordered=list(expr.loops or []),
            allow_speculative_vectorized_reduction=(
                not has_defid_aliases
                and not has_cross_scope_name_shadow
                and not bool(getattr(reduction_facts, "contains_lowered_einstein", False))
            ),
        )

    def visit_lowered_reduction(self, expr: LoweredReductionIR) -> Any:
        return self.evaluate_lowered_reduction(expr, parallel_shape=None)

    def visit_lowered_select_at_argmax(self, expr: LoweredSelectAtArgmaxIR) -> Any:
        from ..runtime.compute.lowered_execution import execute_select_at_argmax_vectorized
        from ..ir.nodes import RectangularAccessIR
        reduction_loops = list(expr.loops or [])
        if not reduction_loops:
            raise RuntimeError("SelectAtArgmax has no reduction loops")
        n_red = len(reduction_loops)
        dep_bucket = self._analysis_cache_bucket("select_at_argmax_depids")
        primal_dep_bucket = self._analysis_cache_bucket("select_at_argmax_primal_depids")
        cache_bucket = self._analysis_cache_bucket("select_at_argmax_result")
        winner_cache_bucket = self._analysis_cache_bucket("select_at_argmax_winner")
        expr_key = id(expr)
        select_facts = self._lowered_select_at_argmax_facts(expr)
        if select_facts is not None:
            depids = tuple(getattr(select_facts, "depids", ()) or ())
            primal_depids = tuple(getattr(select_facts, "primal_depids", ()) or ())
        else:
            depids = dep_bucket.get(expr_key)
            if depids is None:
                from ..passes.autodiff.compiletime import _collect_all_defids_ir

                depids = tuple(
                    sorted(
                        (
                            did for did in (
                                _collect_all_defids_ir(expr.primal_body)
                                | _collect_all_defids_ir(expr.diff_body)
                                | {
                                    d
                                    for loop in reduction_loops
                                    for d in _collect_all_defids_ir(loop.iterable)
                                }
                            )
                            if did is not None
                        ),
                        key=lambda d: (d.krate, d.index),
                    )
                )
                dep_bucket[expr_key] = depids
            primal_depids = primal_dep_bucket.get(expr_key)
            if primal_depids is None:
                from ..passes.autodiff.compiletime import _collect_all_defids_ir

                primal_depids = tuple(
                    sorted(
                        (
                            did for did in (
                                _collect_all_defids_ir(expr.primal_body)
                                | {
                                    d
                                    for loop in reduction_loops
                                    for d in _collect_all_defids_ir(loop.iterable)
                                }
                            )
                            if did is not None
                        ),
                        key=lambda d: (d.krate, d.index),
                    )
                )
                primal_dep_bucket[expr_key] = primal_depids

        reduction_body_defids: Set[Any] = set()
        for _lp in expr.loops or []:
            if _lp.variable is not None and _lp.variable.defid is not None:
                reduction_body_defids.add(_lp.variable.defid)

        if select_facts is not None:
            loop_name_by_defid = dict(getattr(select_facts, "loop_names_by_defid", {}) or {})
            body_defids_by_name = {
                name: list(dids)
                for name, dids in (getattr(select_facts, "body_defids_by_name", {}) or {}).items()
            }
        else:
            loop_name_by_defid = {
                _lp.variable.defid: _lp.variable.name
                for _lp in expr.loops or []
                if _lp.variable is not None and _lp.variable.defid is not None and _lp.variable.name
            }
            body_defids_by_name: Dict[str, List[Any]] = {}
            for _node in (expr.primal_body, expr.diff_body):
                for _name, _dids in self._cached_defids_by_name(_node).items():
                    bucket = body_defids_by_name.setdefault(_name, [])
                    for _did in _dids:
                        if _did is not None and _did not in bucket:
                            bucket.append(_did)

        outer_index_defids = tuple(getattr(self, "_select_outer_index_defids", ()) or ())
        parallel_shape = None
        parallel_defids_list: List[Any] = []
        initial_context: List[Tuple[Any, Any]] = []
        _ri0 = getattr(self, "_reduction_initial_context", None) or {}
        loop_defid_set = {
            loop.variable.defid for loop in reduction_loops
            if loop.variable is not None and loop.variable.defid is not None
        }
        dep_fingerprint = tuple(
            (did, self._cache_value_fingerprint(self.env.get_value(did)))
            for did in depids
            if did not in loop_defid_set
        )
        primal_dep_fingerprint = tuple(
            (did, self._cache_value_fingerprint(self.env.get_value(did)))
            for did in primal_depids
            if did not in loop_defid_set
        )
        ri_fingerprint = tuple(
            sorted(
                (
                    (did, self._cache_value_fingerprint(val))
                    for did, val in _ri0.items()
                    if did is not None
                ),
                key=lambda item: (item[0].krate, item[0].index),
            )
        )
        primal_ri_fingerprint = tuple(
            sorted(
                (
                    (did, self._cache_value_fingerprint(val))
                    for did, val in _ri0.items()
                    if did is not None and did in primal_depids
                ),
                key=lambda item: (item[0].krate, item[0].index),
            )
        )
        clause_parallel_shape = self._vectorization_parallel_shape()
        if clause_parallel_shape is not None and not outer_index_defids:
            try:
                parallel_shape = tuple(int(dim) for dim in clause_parallel_shape)
            except (TypeError, ValueError):
                parallel_shape = None
        enable_result_cache = not _ri0 and not outer_index_defids
        cache_key = None
        if enable_result_cache:
            cache_key = (
                expr_key,
                tuple(parallel_shape or ()),
                tuple(outer_index_defids),
                bool(getattr(expr, "use_argmin", False)),
                dep_fingerprint,
                ri_fingerprint,
            )
            cached_result = cache_bucket.get(cache_key)
            if cached_result is not None:
                return cached_result
        if not outer_index_defids and isinstance(expr.primal_body, RectangularAccessIR) and expr.primal_body.array is not None:
            try:
                arr = expr.primal_body.array.accept(self)
                if parallel_shape is None and not _ri0 and isinstance(arr, np.ndarray) and arr.ndim >= n_red:
                    parallel_shape = tuple(arr.shape[:-n_red])
                if not _ri0:
                    for idx in (expr.primal_body.indices or []):
                        d = _first_parallel_index_defid(idx, reduction_body_defids)
                        if d is not None and d not in parallel_defids_list:
                            parallel_defids_list.append(d)
            except Exception:
                pass
        if parallel_shape and len(parallel_defids_list) == len(parallel_shape):
            initial_context = [
                (defid, np.arange(parallel_shape[i], dtype=np.intp))
                for i, defid in enumerate(parallel_defids_list)
            ]
        elif parallel_shape and len(parallel_defids_list) != len(parallel_shape):
            fallback_parallel_defids = list(self._vectorization_parallel_defids_order() or ())
            if len(fallback_parallel_defids) == len(parallel_shape):
                parallel_defids_list = [did for did in fallback_parallel_defids if did is not None]
                if len(parallel_defids_list) == len(parallel_shape):
                    initial_context = [
                        (defid, np.arange(parallel_shape[i], dtype=np.intp))
                        for i, defid in enumerate(parallel_defids_list)
                    ]
                else:
                    parallel_shape = None
            else:
                parallel_shape = None
        if parallel_shape and all(int(dim) == 1 for dim in parallel_shape):
            parallel_shape = None
            initial_context = []

        def _expand_ctx_to_all_body_defids(ctx: Dict, outer_ctx: Dict) -> Dict:
            env_names = getattr(self.env, "_defid_names", {}) or {}
            authoritative_defids = set(reduction_body_defids)
            authoritative_defids.update(did for did in outer_index_defids if did is not None)
            merged_seed: Dict[Any, Any] = {}
            for source in (outer_ctx or {}, ctx or {}):
                for defid, val in source.items():
                    if defid is None:
                        continue
                    if source is outer_ctx and defid not in authoritative_defids:
                        continue
                    merged_seed[defid] = val
            expanded = dict(merged_seed)
            name_to_value: Dict[str, Any] = {}
            for defid, val in merged_seed.items():
                nm = loop_name_by_defid.get(defid) or env_names.get(defid)
                if nm and nm not in name_to_value:
                    name_to_value[nm] = val
            for nm, dids in body_defids_by_name.items():
                if nm not in name_to_value:
                    continue
                val = name_to_value[nm]
                for did in dids:
                    if did is not None:
                        expanded[did] = val
            return expanded

        def _apply_ctx_then_eval(ctx: Dict, body_expr: Any) -> Any:
            ri = getattr(self, "_reduction_initial_context", None) or {}
            _ctx = _expand_ctx_to_all_body_defids(ctx, ri)
            merged = {**ri, **_ctx}
            saved: Dict[Any, Any] = {}
            try:
                for defid in merged:
                    if defid is not None:
                        saved[defid] = self.env.get_value(defid)
                for defid, val in merged.items():
                    if defid is not None:
                        self.env.set_value(defid, val, name=loop_name_by_defid.get(defid))
                return body_expr.accept(self)
            finally:
                for defid, val in saved.items():
                    self.env.set_value(defid, val, name=loop_name_by_defid.get(defid))

        def primal_body_ev(ctx):
            return _apply_ctx_then_eval(ctx, expr.primal_body)

        def diff_body_ev(ctx):
            return _apply_ctx_then_eval(ctx, expr.diff_body)

        def ev(e):
            return e.accept(self)

        def _compute_primal_winner() -> Tuple[Optional[Tuple[int, ...]], Any]:
            arrs: List[np.ndarray] = []
            defids: List[Any] = []
            for loop in reduction_loops:
                iterable = ev(loop.iterable)
                if isinstance(iterable, range):
                    step = iterable.step if iterable.step is not None else 1
                    arr = np.arange(iterable.start, iterable.stop, step, dtype=np.intp)
                else:
                    arr = np.array(list(iterable), dtype=np.intp)
                if arr.size == 0:
                    return None, None
                arrs.append(arr)
                defids.append(loop.variable.defid)
            if not arrs:
                return None, None
            local_parallel_shape = parallel_shape
            if local_parallel_shape is None:
                spot_ctx: Dict[Any, Any] = {}
                for defid, arr in (initial_context or []):
                    arr_np = np.asarray(arr)
                    spot_ctx[defid] = int(arr_np.flat[0]) if arr_np.size else 0
                for defid, arr in zip(defids, arrs):
                    spot_ctx[defid] = int(arr.flat[0])
                spot_val = primal_body_ev(spot_ctx)
                if isinstance(spot_val, np.ndarray):
                    local_parallel_shape = tuple(spot_val.shape)
                else:
                    local_parallel_shape = ()
            full_shape = tuple(local_parallel_shape) + tuple(int(arr.size) for arr in arrs)
            ctx: Dict[Any, Any] = {}
            if initial_context and local_parallel_shape:
                k_par = len(local_parallel_shape)
                for i, (defid, val) in enumerate(initial_context):
                    v = np.asarray(val, dtype=np.intp)
                    v = v.reshape((1,) * i + (v.size,) + (1,) * (k_par - 1 - i) + (1,) * len(arrs))
                    ctx[defid] = np.broadcast_to(v, full_shape)
            for i, (defid, arr) in enumerate(zip(defids, arrs)):
                if len(arrs) == 1:
                    red_shape = (arr.size,)
                else:
                    red_shape = tuple(arr.size if j == i else 1 for j in range(len(arrs)))
                red_arr = arr.reshape(red_shape)
                if local_parallel_shape:
                    ctx[defid] = np.broadcast_to(red_arr, full_shape)
                else:
                    ctx[defid] = red_arr
            primal_result = primal_body_ev(ctx)
            if not isinstance(primal_result, np.ndarray):
                return local_parallel_shape, None
            if local_parallel_shape:
                primal_flat = primal_result.reshape(local_parallel_shape + (-1,))
                idx_flat = np.argmin(primal_flat, axis=-1) if getattr(expr, "use_argmin", False) else np.argmax(primal_flat, axis=-1)
            else:
                idx_flat = int(np.argmin(primal_result) if getattr(expr, "use_argmin", False) else np.argmax(primal_result))
            return tuple(local_parallel_shape), idx_flat

        ri0 = getattr(self, "_reduction_initial_context", None) or {}
        force_scalar_select = False
        diff_shape = getattr(expr.diff_body, "shape", None)
        diff_shape_info = getattr(expr.diff_body, "shape_info", None)
        diff_type_info = getattr(expr.diff_body, "type_info", None)
        scalar_result_expected = (
            not parallel_shape
            and diff_shape in (None, ())
            and diff_shape_info in (None, ())
            and not isinstance(diff_type_info, RectangularType)
        )

        # Quotient gradients for max/min often lower to SelectAtArgmax whose diff body is a
        # read from the denominator tangent buffer (e.g. @x[...]). When this select executes
        # inside an outer Einstein loop, the current outer loop indices identify the derivative
        # slot we are filling. In the scalar/no-parallel execution path we must preserve that
        # slot match instead of treating the selected tangent read as an unconditional scalar.
        if (
            not parallel_shape
            and ri0
            and isinstance(expr.primal_body, RectangularAccessIR)
            and isinstance(expr.diff_body, RectangularAccessIR)
            and len(expr.primal_body.indices or []) == len(expr.diff_body.indices or [])
        ):
            try:
                direct_slot_projected = True
                for pos, idx_expr in enumerate(expr.diff_body.indices or []):
                    if pos >= len(outer_index_defids):
                        continue
                    expected_did = outer_index_defids[pos]
                    if expected_did is None:
                        continue
                    if isinstance(idx_expr, (IdentifierIR, IndexVarIR)) and idx_expr.defid == expected_did:
                        continue
                    direct_slot_projected = False
                    break
                if not direct_slot_projected:
                    raise ValueError("diff body indices are not a direct outer-slot projection")
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

        ok = False
        result = None
        winner_cache_key = (
            expr_key,
            tuple(parallel_shape or ()),
            tuple(outer_index_defids),
            bool(getattr(expr, "use_argmin", False)),
            primal_dep_fingerprint,
            primal_ri_fingerprint,
        )
        winner_cached = winner_cache_bucket.get(winner_cache_key)
        winner_parallel_shape = None
        winner_idx_flat = None
        if winner_cached is not None:
            winner_parallel_shape, winner_idx_flat = winner_cached
            if winner_parallel_shape is not None:
                parallel_shape = winner_parallel_shape
        else:
            winner_parallel_shape, winner_idx_flat = _compute_primal_winner()
            if winner_parallel_shape is not None:
                parallel_shape = winner_parallel_shape
            winner_cache_bucket[winner_cache_key] = (winner_parallel_shape, winner_idx_flat)
        if not force_scalar_select:
            ok, result = execute_select_at_argmax_vectorized(
                primal_body_ev,
                diff_body_ev,
                reduction_loops,
                ev,
                parallel_shape=parallel_shape,
                initial_context=initial_context if initial_context else None,
                use_argmin=getattr(expr, "use_argmin", False),
                precomputed_idx_flat=winner_idx_flat,
            )
        if not ok or result is None:
            from ..runtime.compute.lowered_execution import execute_lowered_loops
            best_primal = None
            best_diff = None
            scalar_init: Dict[Any, Any] = {}
            ri_scalar = getattr(self, "_reduction_initial_context", None) or {}
            for did, val in ri_scalar.items():
                if isinstance(val, IRNode):
                    continue
                try:
                    arr = np.asarray(val)
                except Exception:
                    continue
                if arr.ndim == 0 or arr.size == 1:
                    flat0 = arr.reshape(-1)[0]
                    if isinstance(flat0, IRNode):
                        continue
                    try:
                        scalar_init[did] = int(flat0) if np.issubdtype(arr.dtype, np.integer) else flat0.item()
                    except Exception:
                        continue
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
        if scalar_result_expected:
            try:
                result_arr = np.asarray(result)
                if result_arr.size >= 1:
                    scalar0 = result_arr.reshape(-1)[0]
                    result = scalar0.item() if hasattr(scalar0, "item") else scalar0
            except Exception:
                pass
        if enable_result_cache and cache_key is not None:
            cache_bucket[cache_key] = result
        return result

    def visit_where_expression(self, expr: WhereExpressionIR) -> Any:
        bindings = getattr(expr, "binding_constraints", None)
        guards = getattr(expr, "guard_constraints", None)
        if bindings is None or guards is None:
            raise RuntimeError(
                "Unnormalized WhereExpressionIR reached backend. "
                "Where expressions must be normalized during AST lowering."
            )
        needs_scope = bool(bindings)
        if needs_scope:
            with self.env.scope():
                for binding in bindings:
                    if binding.defid is not None and binding.expr is not None:
                        self.env.set_value(binding.defid, binding.expr.accept(self), name=binding.name)
                for guard in guards:
                    if not self._to_bool(guard.accept(self)):
                        return None
                return expr.expr.accept(self)
        for guard in guards:
            if not self._to_bool(guard.accept(self)):
                return None
        return expr.expr.accept(self)

    def visit_pipeline_expression(self, expr: PipelineExpressionIR) -> Any:
        left_value = expr.left.accept(self)
        from .numpy_arrow_pipeline import apply_pipeline_right
        return apply_pipeline_right(expr.right, left_value, expr.location, self)

    def visit_builtin_call(self, expr: BuiltinCallIR) -> Any:
        from ..shared.defid import DefId

        def _coerce_defid(raw: Any) -> Any:
            if raw is None:
                return None
            if isinstance(raw, (list, tuple)) and len(raw) >= 2:
                return DefId(krate=int(raw[0]), index=int(raw[1]))
            if hasattr(raw, "krate") and hasattr(raw, "index"):
                return DefId(krate=int(raw.krate), index=int(raw.index))
            return raw

        defid = _coerce_defid(expr.defid)
        autodiff_kind = autodiff_builtin_kind(defid)
        if autodiff_kind is not None:
            engine = getattr(self, "_autodiff_runtime_engine", None)
            if engine is None:
                engine = AutodiffRuntimeEngine(self)
                self._autodiff_runtime_engine = engine
            return engine.execute_builtin(autodiff_kind, expr)
        raw = expr.defid
        if raw is None:
            raise RuntimeError("Builtin call has no DefId")
        fn = self.env.get_value(defid)
        if fn is None or not callable(fn):
            raise RuntimeError(f"Builtin not found (DefId: {defid})")
        args_list = (expr.args or [])
        args = [arg.accept(self) for arg in args_list]
        try:
            return _invoke_runtime_builtin(fn, args)
        except Exception as e:
            self._raise_here(e, expr)
