"""NumPy backend Einstein execution: variable decl, lowered einstein/clause; env only."""

import os
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..ir.nodes import (
    LiteralIR, RangeIR, LoweredEinsteinIR, LoweredEinsteinClauseIR,
    LoweredReductionIR, ReductionExpressionIR, BinaryOpIR, RectangularAccessIR, IndexVarIR,
    FunctionCallIR, IdentifierIR, IfExpressionIR,
    is_function_binding, is_einstein_binding,
)
from ..shared.defid import DefId
from .numpy_helpers import _reject_non_lowered


def _body_references_defid(expr: Any, target_defid: Any) -> bool:
    if target_defid is None:
        return False
    if isinstance(expr, IdentifierIR):
        return getattr(expr, "defid", None) == target_defid
    if isinstance(expr, RectangularAccessIR):
        if _body_references_defid(expr.array, target_defid):
            return True
        for idx in getattr(expr, "indices", []) or []:
            if _body_references_defid(idx, target_defid):
                return True
        return False
    if isinstance(expr, BinaryOpIR):
        return _body_references_defid(getattr(expr, "left", None), target_defid) or _body_references_defid(getattr(expr, "right", None), target_defid)
    if isinstance(expr, FunctionCallIR):
        if _body_references_defid(getattr(expr, "callee_expr", None), target_defid):
            return True
        for a in getattr(expr, "arguments", []) or []:
            if _body_references_defid(a, target_defid):
                return True
        return False
    return False


def _body_contains_call_using_loop_var(expr: Any, loop_defids: List[Any]) -> bool:
    """True if body contains a FunctionCallIR whose arguments (or callee) reference a loop var."""
    if not expr or not loop_defids:
        return False
    if isinstance(expr, FunctionCallIR):
        for defid in loop_defids:
            if _body_references_defid(getattr(expr, "callee_expr", None), defid):
                return True
            for a in getattr(expr, "arguments", []) or []:
                if _body_references_defid(a, defid):
                    return True
        return False
    if isinstance(expr, RectangularAccessIR):
        return _body_contains_call_using_loop_var(getattr(expr, "array", None), loop_defids) or any(
            _body_contains_call_using_loop_var(idx, loop_defids) for idx in (getattr(expr, "indices", None) or [])
        )
    if isinstance(expr, BinaryOpIR):
        return _body_contains_call_using_loop_var(getattr(expr, "left", None), loop_defids) or _body_contains_call_using_loop_var(getattr(expr, "right", None), loop_defids)
    if hasattr(expr, "expr"):
        return _body_contains_call_using_loop_var(getattr(expr, "expr", None), loop_defids)
    if hasattr(expr, "inner_pattern"):
        return _body_contains_call_using_loop_var(getattr(expr, "inner_pattern", None), loop_defids)
    if hasattr(expr, "patterns"):
        return any(_body_contains_call_using_loop_var(p, loop_defids) for p in (getattr(expr, "patterns", None) or []))
    return False


def _index_expr_is_loop_var(expr: Any, loop_defid: Any) -> bool:
    if expr is None or loop_defid is None:
        return False
    return (isinstance(expr, (IdentifierIR, IndexVarIR)) and
            getattr(expr, "defid", None) == loop_defid)


def _collect_lhs_read_index_lists(body: Any, target_defid: Any) -> List[List[Any]]:
    out: List[List[Any]] = []
    if isinstance(body, RectangularAccessIR):
        if _body_references_defid(body.array, target_defid):
            indices = getattr(body, "indices", None) or []
            if indices:
                out.append(list(indices))
        out.extend(_collect_lhs_read_index_lists(body.array, target_defid))
        return out
    if isinstance(body, BinaryOpIR):
        out.extend(_collect_lhs_read_index_lists(getattr(body, "left", None), target_defid))
        out.extend(_collect_lhs_read_index_lists(getattr(body, "right", None), target_defid))
    elif isinstance(body, FunctionCallIR):
        for a in getattr(body, "arguments", []) or []:
            out.extend(_collect_lhs_read_index_lists(a, target_defid))
        out.extend(_collect_lhs_read_index_lists(getattr(body, "callee_expr", None), target_defid))
    return out


def _recurrence_dims(lowered: Any, variable_defid: Any) -> List[int]:
    """Return dimension indices where LHS is read at an index that differs from the write index (recurrence)."""
    loops = getattr(lowered, "loops", None) or []
    if not loops or variable_defid is None:
        return []
    loop_defids = [getattr(lp.variable, "defid", None) for lp in loops]
    read_index_lists = _collect_lhs_read_index_lists(lowered.body, variable_defid)
    if not read_index_lists:
        return []
    recurrence: List[int] = []
    for d in range(len(loops)):
        for idx_list in read_index_lists:
            if d >= len(idx_list):
                continue
            if not _index_expr_is_loop_var(idx_list[d], loop_defids[d]):
                recurrence.append(d)
                break
    return recurrence


def _extract_loop_range(loop, evaluator) -> Tuple[int, int]:
    """Return (start, end) for the loop range; both must be concrete int. Raises if missing or dependent."""
    it = getattr(loop, "iterable", None)
    if it is None:
        raise RuntimeError("loop has no iterable; cannot extract range")
    if isinstance(it, LiteralIR) and isinstance(getattr(it, "value", None), range):
        r = it.value
        start = getattr(r, "start", 0)
        stop = getattr(r, "stop", r.start)
        try:
            return (int(start), int(stop))
        except (TypeError, ValueError) as e:
            raise RuntimeError("loop range start/stop must be int; got dependent or non-int") from e
    if isinstance(it, RangeIR):
        end_ev = evaluator(it.end)
        if not isinstance(end_ev, (int, np.integer)):
            raise RuntimeError("loop range end must be int; got dependent or non-int")
        start_node = getattr(it, "start", None)
        if start_node is not None:
            start_ev = evaluator(start_node)
            if not isinstance(start_ev, (int, np.integer)):
                raise RuntimeError("loop range start must be int; got dependent or non-int")
            return (int(start_ev), int(end_ev))
        return (0, int(end_ev))
    raise RuntimeError("loop iterable is not a range or literal range; cannot extract (start, end)")


def _try_vectorize_clause(clause, output_shape, dtype, evaluator, backend=None):
    """
    General vectorization: set loop variables to broadcast numpy arrays,
    evaluate the body once, and let numpy handle everything.
    Falls back to None if any operation doesn't support array-valued indices.
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

    ndim = len(loops)
    loop_info: List[Tuple[Any, Tuple[int, int], str]] = []
    for dim, lp in enumerate(loops):
        defid = getattr(lp.variable, "defid", None)
        if defid is None:
            return None
        r = _extract_loop_range(lp, evaluator)
        name = getattr(lp.variable, "name", None)
        loop_info.append((defid, r, name))

    try:
        with backend.env.scope():
            for dim, (defid, rng, name) in enumerate(loop_info):
                start, end = rng
                sz = end - start
                shape = [1] * ndim
                shape[dim] = sz
                arr = np.arange(start, end, dtype=np.intp).reshape(shape)
                backend.env.set_value(defid, arr, name=name)

            body = getattr(clause, "body", None)
            if isinstance(body, IfExpressionIR):
                cond = body.condition.accept(backend)
                if isinstance(cond, np.ndarray) and cond.ndim > 0:
                    # If-expr as scalar RHS: evaluate only the taken branch at each point via vectorized indexing.
                    valid = np.asarray(cond, dtype=bool)
                    expected_shape = tuple(output_shape)
                    result = np.zeros(expected_shape, dtype=dtype)
                    n_valid = int(np.sum(valid))
                    valid_indices = np.where(valid)
                    if len(valid_indices) == ndim and n_valid > 0:
                        with backend.env.scope():
                            for dim, (defid, _, name) in enumerate(loop_info):
                                backend.env.set_value(defid, valid_indices[dim], name=name)
                        then_val = body.then_expr.accept(backend)
                        then_flat = np.asarray(then_val, dtype=dtype).ravel()[:n_valid]
                        if then_flat.size >= n_valid:
                            result[valid] = then_flat[:n_valid]
                    n_invalid = int(np.sum(~valid))
                    invalid_indices = np.where(~valid)
                    if len(invalid_indices) == ndim and n_invalid > 0:
                        with backend.env.scope():
                            for dim, (defid, _, name) in enumerate(loop_info):
                                backend.env.set_value(defid, invalid_indices[dim], name=name)
                        else_expr = getattr(body, "else_expr", None)
                        else_val = else_expr.accept(backend) if else_expr else None
                        if else_val is not None:
                            else_flat = np.asarray(else_val, dtype=dtype).ravel()[:n_invalid]
                            if else_flat.size >= n_invalid:
                                result[~valid] = else_flat[:n_invalid]
                            else:
                                result[~valid] = dtype(0.0)
                        else:
                            result[~valid] = dtype(0.0)
                    elif n_invalid > 0:
                        result[~valid] = dtype(0.0)
                else:
                    result = clause.body.accept(backend)
            else:
                result = clause.body.accept(backend)

            if isinstance(result, np.ndarray):
                expected = tuple(output_shape)
                ranges = [(start, end) for (_, (start, end), _) in loop_info]
                range_is_full = len(ranges) == len(expected) and all(
                    start == 0 and end == expected[dim] for dim, (start, end) in enumerate(ranges)
                )
                if result.shape == expected:
                    return result.astype(dtype)
                if not range_is_full:
                    full = np.zeros(expected, dtype=dtype)
                    slices = tuple(slice(int(start), int(end)) for (start, end) in ranges)
                    full[slices] = result.astype(dtype)
                    return full
                if result.size == np.prod(expected):
                    return result.reshape(expected).astype(dtype)
                try:
                    return np.broadcast_to(result, expected).copy().astype(dtype)
                except ValueError:
                    return None
            elif isinstance(result, (int, float, np.integer, np.floating)):
                return np.full(output_shape, result, dtype=dtype)
    except Exception as e:
        if os.environ.get("EINLANG_DEBUG_VECTORIZE"):
            import traceback
            traceback.print_exc()
        return None
    return None


def _try_hybrid_vectorize_clause(
    clause: Any,
    output_shape: List[int],
    output: np.ndarray,
    variable_defid: Any,
    expr_evaluator: Any,
    backend: Any,
) -> Optional[np.ndarray]:
    """
    When the body has recurrence on a subset of dimensions: iterate over those
    (scalar), vectorize over the rest. Writes into output and returns it, or None on failure.
    """
    from ..runtime.compute.lowered_execution import execute_lowered_loops
    loops = getattr(clause, "loops", None) or []
    if not loops or clause.guards or clause.bindings:
        return None
    ndim = len(loops)
    recurrence_dims = _recurrence_dims(clause, variable_defid) if variable_defid else []
    if not (0 < len(recurrence_dims) < ndim):
        return None
    loop_info: List[Tuple[Any, Tuple[int, int], str]] = []
    for dim, lp in enumerate(loops):
        defid = getattr(lp.variable, "defid", None)
        if defid is None:
            return None
        try:
            r = _extract_loop_range(lp, expr_evaluator)
        except RuntimeError:
            return None
        name = getattr(lp.variable, "name", None)
        loop_info.append((defid, r, name))
    recurrence_loops = [loops[d] for d in recurrence_dims]
    _MAX = int(os.environ.get("EINLANG_EINSTEIN_LOOP_MAX", "1000000"))
    n_iter = [0]
    try:
        for rec_context in execute_lowered_loops(recurrence_loops, {}, expr_evaluator):
            n_iter[0] += 1
            if n_iter[0] > _MAX:
                return None
            with backend.env.scope():
                for dim in range(ndim):
                    defid, (start, end), name = loop_info[dim]
                    if dim in recurrence_dims:
                        backend.env.set_value(defid, rec_context[defid], name=name)
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
                if dim in recurrence_dims:
                    slice_list.append(rec_context[loop_info[dim][0]])
                else:
                    start, end = loop_info[dim][1]
                    slice_list.append(slice(int(start), int(end)))
            try:
                squeezed = np.squeeze(result, axis=tuple(recurrence_dims))
            except ValueError:
                return None
            output[tuple(slice_list)] = squeezed.astype(output.dtype)
        return output
    except Exception:
        return None


class EinsteinExecutionMixin:
    def visit_lowered_einstein_clause(self, node: LoweredEinsteinClauseIR) -> Any:
        stack = getattr(self, "_variable_decl_stack", None)
        variable_decl = stack[-1] if stack else None
        return self._execute_lowered_einstein_clause(node, variable_decl)

    def visit_lowered_einstein(self, node: LoweredEinsteinIR) -> Any:
        stack = getattr(self, "_variable_decl_stack", None)
        variable_decl = stack[-1] if stack else None
        return self._execute_lowered_einstein(node, variable_decl)

    def _primitive_type_to_numpy_dtype(self, type_obj: Any) -> Optional[Any]:
        from ..shared.types import PrimitiveType
        if not isinstance(type_obj, PrimitiveType):
            return None
        type_name = type_obj.name.lower()
        dtype_map = {
            "i8": np.int8, "i32": np.int32, "i64": np.int64,
            "f16": np.float16, "f32": np.float32, "f64": np.float64,
            "bool": np.bool_, "int": np.int32, "float": np.float32,
        }
        try:
            import ml_dtypes
            dtype_map["bf16"] = ml_dtypes.bfloat16
            dtype_map["f8e4m3"] = ml_dtypes.float8_e4m3fn
        except ImportError:
            pass
        return dtype_map.get(type_name)

    def _type_info_to_numpy_dtype(self, type_info: Any) -> Optional[Any]:
        from ..shared.types import PrimitiveType, RectangularType, Type, TypeKind
        if type_info is None:
            return None
        if isinstance(type_info, Type) and hasattr(type_info, "kind") and type_info.kind == TypeKind.UNKNOWN:
            return None
        if isinstance(type_info, PrimitiveType):
            return self._primitive_type_to_numpy_dtype(type_info)
        if isinstance(type_info, RectangularType):
            return self._type_info_to_numpy_dtype(type_info.element_type)
        return None

    def _dtype_for_clause_result(self, clause_body: Any, tensor_element_type: Any) -> Any:
        """Dtype for values produced by evaluating the clause body (scalar path: value = body.accept(self)).
        Use same source of truth for both main allocation and per-clause allocation."""
        dtype = self._type_info_to_numpy_dtype(tensor_element_type)
        if dtype is not None:
            return dtype
        if clause_body is None:
            return np.int32
        type_info = getattr(clause_body, "type_info", None)
        if type_info is not None:
            dtype = self._type_info_to_numpy_dtype(type_info)
            if dtype is not None:
                return dtype
        if isinstance(clause_body, (LoweredReductionIR, ReductionExpressionIR)):
            body_expr = getattr(clause_body, "body", None)
            if body_expr is not None:
                ti = getattr(body_expr, "type_info", None)
                if ti is not None:
                    dtype = self._type_info_to_numpy_dtype(ti)
                    if dtype is not None:
                        return dtype
        if self._body_implies_float(clause_body):
            return np.float32
        return np.int32

    def _body_implies_float(self, body: Any) -> bool:
        from ..ir.nodes import FunctionCallIR, BinaryOpIR
        from ..shared.types import PrimitiveType
        if body is None:
            return False
        if isinstance(body, (LoweredReductionIR, ReductionExpressionIR)):
            return self._body_implies_float(getattr(body, "body", None))
        if isinstance(body, LiteralIR):
            v = getattr(body, "value", None)
            return v is not None and isinstance(v, (float, np.floating))
        if isinstance(body, RectangularAccessIR):
            arr = getattr(body, "array", None)
            if arr is not None:
                t = getattr(arr, "type_info", None)
                if t is not None:
                    el = getattr(t, "element_type", None) or t
                    if isinstance(el, PrimitiveType) and (getattr(el, "name", None) or "").lower() in ("f32", "f64", "float"):
                        return True
                defid = getattr(arr, "defid", None)
                if defid is not None and hasattr(self, "env"):
                    try:
                        val = self.env.get_value(defid)
                        if isinstance(val, np.ndarray) and np.issubdtype(val.dtype, np.floating):
                            return True
                    except Exception:
                        pass
            t = getattr(body, "type_info", None)
            if t is not None:
                el = getattr(t, "element_type", None) or t
                if isinstance(el, PrimitiveType) and (getattr(el, "name", None) or "").lower() in ("f32", "f64", "float"):
                    return True
        if isinstance(body, FunctionCallIR):
            name = (body.function_name or "").lower()
            if name in ("exp", "ln", "log", "sqrt", "sigmoid", "tanh"):
                return True
        if isinstance(body, BinaryOpIR):
            from ..shared.types import BinaryOp
            op = getattr(body, "operator", None)
            op_val = getattr(op, "value", None) if op is not None else None
            if op in (BinaryOp.MUL, BinaryOp.DIV, BinaryOp.POW) or op_val in ("/", "**", "*"):
                for operand in (body.left, body.right):
                    if self._body_implies_float(operand):
                        return True
            for operand in (body.left, body.right):
                if operand is not None:
                    t = getattr(operand, "type_info", None)
                    if t is not None:
                        el = getattr(t, "element_type", None) or t
                        if isinstance(el, PrimitiveType) and (getattr(el, "name", None) or "").lower() in ("f32", "f64", "float"):
                            return True
            if body.left and self._body_implies_float(body.left):
                return True
            if body.right and self._body_implies_float(body.right):
                return True
        return False

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
        variable_key = (getattr(binding, "defid", None) if binding else None) or getattr(variable_decl, "defid", None)
        tensor_shape = lowered_einstein.shape
        tensor_element_type = lowered_einstein.element_type
        output_shape = None
        if tensor_shape:
            output_shape = []
            for shape_dim in tensor_shape:
                dim_value = shape_dim.accept(self)
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
            # the first clause's end, not the union).  Widen to cover all items.
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
        if dtype is None and lowered_einstein.items:
            first_body = lowered_einstein.items[0].body
            dtype = self._dtype_for_clause_result(first_body, None)
        if dtype is None:
            dtype = np.int32
        if dtype == np.int32 and lowered_einstein.items:
            first_body = lowered_einstein.items[0].body
            if isinstance(first_body, ReductionExpressionIR):
                body = getattr(first_body, "body", None)
                if isinstance(body, BinaryOpIR):
                    from ..shared.types import BinaryOp
                    op = getattr(body, "operator", None)
                    if op in (BinaryOp.MUL, BinaryOp.DIV) or (op is not None and getattr(op, "value", None) in ("*", "/")):
                        if self._body_implies_float(body):
                            dtype = np.float32
                        elif getattr(body, "left", None) is not None and getattr(body, "right", None) is not None:
                            if isinstance(body.left, RectangularAccessIR) or isinstance(body.right, RectangularAccessIR):
                                dtype = np.float32

        # Multi-segment: reuse existing array if this variable was already
        # declared (e.g. pad's `let result[i in 0..p] = ...; let result[i in p..n] = ...;`)
        existing = self.env.get_value(variable_key) if variable_key is not None else None
        if existing is not None and isinstance(existing, np.ndarray):
            needed = tuple(output_shape)
            current = existing.shape
            if len(needed) == len(current) and any(n > c for n, c in zip(needed, current)):
                new_shape = tuple(max(n, c) for n, c in zip(needed, current))
                output = np.zeros(new_shape, dtype=existing.dtype)
                slices = tuple(slice(0, s) for s in current)
                output[slices] = existing
                self.env.set_value(variable_key, output)
            else:
                output = existing
        else:
            output = np.zeros(output_shape, dtype=dtype)
            if variable_key is not None:
                self.env.set_value(variable_key, output)

        def expr_eval(e: Any) -> Any:
            return e.accept(self)

        for clause_idx, item in enumerate(lowered_einstein.items):
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
                    if result.ndim == output.ndim and getattr(item, "loops", None):
                        slices_list: List[slice] = []
                        try:
                            for lp in item.loops:
                                start, end = _extract_loop_range(lp, expr_eval)
                                slices_list.append(slice(int(start), int(end)))
                        except RuntimeError:
                            slices_list = []
                        if len(slices_list) == len(item.loops):
                            output[tuple(slices_list)] = result.astype(output.dtype)
                    elif result.size == 1 and getattr(item, "indices", None) and all(
                        isinstance(idx, LiteralIR) for idx in item.indices
                    ):
                        idx_tuple = tuple(int(idx.value) for idx in item.indices)
                        output[idx_tuple] = result.flat[0] if result.size == 1 else result
                self.env.set_value(variable_key, output)
        return output

    def _shape_from_all_items(self, items: List) -> Optional[List[int]]:
        """Compute output shape from the max absolute end across ALL items (not just the first)."""
        if not items:
            return None
        rank = None
        max_ends: Optional[List[int]] = None
        for item in items:
            loops = getattr(item, "loops", None) or []
            if not loops:
                continue
            if rank is None:
                rank = len(loops)
                max_ends = [0] * rank
            if len(loops) != rank:
                return None
            for d, loop in enumerate(loops):
                it = getattr(loop, "iterable", None)
                if it is None:
                    return None
                if isinstance(it, LiteralIR) and isinstance(getattr(it, "value", None), range):
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
        return max_ends

    def _execute_lowered_einstein_clause(
        self,
        lowered: LoweredEinsteinClauseIR,
        variable_decl: Any,
        shape: Optional[List] = None,
        element_type: Optional[Any] = None,
        pre_allocated_output: Optional[Any] = None,
    ) -> Any:
        from ..runtime.compute.lowered_execution import execute_lowered_loops, execute_lowered_bindings, check_lowered_guards
        loc = getattr(lowered, "location", None) or getattr(variable_decl, "location", None)
        line = int(getattr(loc, "line", 0) or 0)
        bucket_size = getattr(self, "_profile_bucket_size", 0)
        t0 = time.perf_counter() if bucket_size > 0 else 0
        _debug_vec = os.environ.get("EINLANG_DEBUG_VECTORIZE", "").strip().lower() in ("1", "true", "yes")

        def _record_profile():
            if bucket_size > 0 and getattr(self, "_profile_buckets", None) is not None:
                key = (line // bucket_size) * bucket_size
                self._profile_buckets[key] = self._profile_buckets.get(key, 0) + (time.perf_counter() - t0)

        clause_indices = getattr(lowered, "indices", None) or []
        binding = getattr(variable_decl, "_binding", None)
        variable_defid = (getattr(binding, "defid", None) if binding else None) or getattr(variable_decl, "defid", None)

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
                    defid = getattr(lowered.loops[loop_pos].variable, "defid", None)
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
        else:
            output_shape = None
            if shape:
                output_shape = []
                for shape_dim in shape:
                    dim_value = shape_dim.accept(self)
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
                    it = getattr(loop, "iterable", None)
                    if it is None:
                        output_shape = None
                        break
                    if isinstance(it, LiteralIR) and isinstance(getattr(it, "value", None), range):
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
                output_shape = [int(getattr(idx, "value", 0)) + 1 if isinstance(idx, LiteralIR) else 1 for idx in clause_indices] if clause_indices else [1]
            dtype = self._dtype_for_clause_result(lowered.body, element_type)
            output = np.zeros(output_shape, dtype=dtype)

        def expr_evaluator(expr: Any) -> Any:
            return expr.accept(self)

        has_literal_idx = any(isinstance(idx, LiteralIR) for idx in clause_indices)
        # Skip vectorization for clauses with if/else or bodies that call functions with loop-var args; use scalar path.
        body_node = getattr(lowered, "body", None)
        has_if_body = isinstance(body_node, IfExpressionIR)
        loop_defids = [getattr(lp.variable, "defid", None) for lp in (lowered.loops or [])]
        has_call_using_loop = _body_contains_call_using_loop_var(body_node, [d for d in loop_defids if d is not None])
        if lowered.loops and not has_literal_idx and not has_if_body and not has_call_using_loop:
            vec_result = _try_vectorize_clause(
                lowered, list(output.shape), output.dtype, expr_evaluator, backend=self,
            )
            if vec_result is not None:
                vec_result = np.asarray(vec_result)
                if vec_result.shape == output.shape:
                    if pre_allocated_output is not None and lowered.loops:
                        slices_list_partial: List[slice] = []
                        try:
                            for lp in lowered.loops:
                                start, end = _extract_loop_range(lp, expr_evaluator)
                                slices_list_partial.append(slice(int(start), int(end)))
                        except RuntimeError:
                            slices_list_partial = []
                        range_is_full_partial = (
                            len(slices_list_partial) == len(lowered.loops)
                            and all(s.start == 0 and s.stop == output.shape[i] for i, s in enumerate(slices_list_partial))
                        )
                        if len(slices_list_partial) == len(lowered.loops) and not range_is_full_partial:
                            recurrence_dims = _recurrence_dims(lowered, variable_defid) if _body_references_defid(lowered.body, variable_defid) else []
                            if recurrence_dims:
                                hybrid_out = _try_hybrid_vectorize_clause(
                                    lowered, list(output.shape), output, variable_defid, expr_evaluator, backend=self,
                                )
                                if hybrid_out is not None:
                                    if variable_defid:
                                        self.env.set_value(variable_defid, output)
                                    if _debug_vec:
                                        print(f"[hybrid] L{line}", flush=True)
                                    _record_profile()
                                    return output
                                vec_result = None
                            else:
                                output[tuple(slices_list_partial)] = vec_result[tuple(slices_list_partial)].astype(output.dtype)
                        else:
                            output[:] = vec_result
                    else:
                        output[:] = vec_result
                if vec_result is not None:
                    if vec_result.shape != output.shape and vec_result.size == output.size:
                        output[:] = vec_result.reshape(output.shape)
                    elif pre_allocated_output is not None and vec_result.ndim == output.ndim:
                        slices_list: List[slice] = []
                        try:
                            for lp in lowered.loops:
                                start, end = _extract_loop_range(lp, expr_evaluator)
                                slices_list.append(slice(int(start), int(end)))
                        except RuntimeError:
                            slices_list = []
                        range_is_full = (
                            len(slices_list) == len(lowered.loops)
                            and all(s.start == 0 and s.stop == output.shape[i] for i, s in enumerate(slices_list))
                        )
                        if len(slices_list) == len(lowered.loops):
                            if range_is_full:
                                output[:] = np.broadcast_to(vec_result.astype(output.dtype), output.shape)
                            else:
                                output[tuple(slices_list)] = vec_result[tuple(slices_list)].astype(output.dtype)
                        else:
                            output[:] = np.broadcast_to(vec_result, output.shape)
                    else:
                        output[:] = np.broadcast_to(vec_result, output.shape)
                    if variable_defid:
                        self.env.set_value(variable_defid, output)
                    if _debug_vec:
                        print(f"[vectorized] L{line}", flush=True)
                    _record_profile()
                    return output

        if _debug_vec and lowered.loops:
            print(f"[scalar] L{line}", flush=True)
        _loop_defid_to_name = {}
        for lp in lowered.loops:
            v = getattr(lp, "variable", None)
            if v and getattr(v, "defid", None):
                _loop_defid_to_name[v.defid] = getattr(v, "name", None)

        with self.env.scope():
            if not lowered.loops:
                if all(isinstance(idx, LiteralIR) for idx in clause_indices):
                    idx_tuple = tuple(int(idx.value) for idx in clause_indices)
                else:
                    idx_tuple = None
                if idx_tuple is not None:
                    value = lowered.body.accept(self)
                    if value is not None:
                        if isinstance(value, np.ndarray):
                            if value.ndim == 0:
                                value = value.item()
                            elif value.size == 1:
                                value = value.flatten()[0].item()
                        elif isinstance(value, np.generic):
                            value = value.item()
                        output[idx_tuple] = value
            else:
                _MAX = int(os.environ.get("EINLANG_EINSTEIN_LOOP_MAX", "1000000"))
                _n = [0]
                for loop_context in execute_lowered_loops(lowered.loops, {}, expr_evaluator):
                    _n[0] += 1
                    if _n[0] > _MAX:
                        raise RuntimeError("Einstein clause loop iterations exceeded limit.")
                    full_context = execute_lowered_bindings(lowered.bindings, loop_context, expr_evaluator)
                    for defid, val in full_context.items():
                        if defid is not None:
                            vname = _loop_defid_to_name.get(defid)
                            self.env.set_value(defid, val, name=vname)
                    if lowered.guards and not check_lowered_guards(lowered.guards, full_context, lambda e: self._to_bool(e.accept(self))):
                        continue
                    try:
                        value = lowered.body.accept(self)
                    except IndexError:
                        continue
                    idx_tuple = cell_index(full_context)
                    if idx_tuple is None and clause_indices:
                        out_idx = []
                        loop_pos = 0
                        for idx in clause_indices:
                            if isinstance(idx, LiteralIR):
                                try:
                                    out_idx.append(int(idx.value))
                                except (TypeError, ValueError):
                                    break
                            elif loop_pos < len(lowered.loops):
                                v = full_context.get(getattr(lowered.loops[loop_pos].variable, "defid", None))
                                if v is None:
                                    break
                                out_idx.append(v)
                                loop_pos += 1
                            else:
                                break
                        if len(out_idx) == len(clause_indices):
                            idx_tuple = tuple(out_idx)
                    if idx_tuple is None:
                        idx_tuple = tuple(full_context.get(getattr(loop.variable, "defid", None)) for loop in lowered.loops)
                    if idx_tuple is None or len(idx_tuple) != output.ndim:
                        continue
                    if isinstance(value, np.ndarray):
                        if value.ndim == 0:
                            value = value.item()
                        elif value.size == 1:
                            value = value.flatten()[0].item()
                    elif isinstance(value, np.generic):
                        value = value.item()
                    if len(idx_tuple) == 1:
                        output[idx_tuple[0]] = value
                    else:
                        output[idx_tuple] = value

        if variable_defid:
            self.env.set_value(variable_defid, output)
        _record_profile()
        return output
