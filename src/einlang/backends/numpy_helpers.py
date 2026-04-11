"""Backend helpers shared by the NumPy execution backend."""

from dataclasses import dataclass
from typing import Any, Dict, Generic, List, Optional, Tuple, TypeVar

import numpy as np

from ..ir.nodes import (
    LiteralIR, IdentifierIR, IRVisitor,
    LiteralPatternIR, IdentifierPatternIR,
    TuplePatternIR, ArrayPatternIR, RestPatternIR, GuardPatternIR,
    BindingIR,
)
from ..shared.defid import DefId


T = TypeVar("T")


@dataclass
class NumPyVectorizationState:
    parallel_shape: Optional[Tuple[int, ...]] = None
    parallel_defids_order: Optional[Tuple[Any, ...]] = None
    safe_oob: bool = False
    recurrence_clause: bool = False


try:
    import ml_dtypes as _ml_dtypes
except ImportError:
    _ml_dtypes = None


def _is_numpy_scalar(value: Any) -> bool:
    return isinstance(value, (np.integer, np.floating, np.bool_))


def _as_list_like(value: Any) -> Optional[List[Any]]:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return list(value)
    return None


def _reject_non_lowered(node_type_name: str) -> None:
    raise RuntimeError(
        f"Non-lowered IR at runtime: {node_type_name}. "
        "Lowering passes must replace with lowered form before execution."
    )


def builtin_assert(condition: Any, message: str = "Assertion failed") -> None:
    def _all_true(v: Any) -> bool:
        if _is_numpy_scalar(v):
            return bool(v)
        if isinstance(v, np.ndarray) and v.ndim == 0:
            return bool(v.item())
        if hasattr(v, "__iter__") and hasattr(v, "__len__") and not hasattr(v, "all"):
            return all(_all_true(e) for e in v)
        if hasattr(v, "all") and callable(v.all):
            return bool(v.all())
        return bool(v)
    if not _all_true(condition):
        raise RuntimeError(f"assertion failed: {message}")


def builtin_print(*args: Any) -> None:
    out = []
    for a in args:
        out.append(a.tolist() if hasattr(a, "tolist") else (list(a) if isinstance(a, (list, tuple)) else a))
    print(*out, flush=True)


def builtin_len(collection: Any) -> int:
    if hasattr(collection, "__len__"):
        return len(collection)
    if isinstance(collection, np.ndarray):
        return int(collection.size)
    raise TypeError(f"Object of type {type(collection).__name__} has no len()")


def builtin_shape(array: Any) -> Any:
    if isinstance(array, np.ndarray):
        return np.array(array.shape, dtype=int)
    sequence = _as_list_like(array)
    if sequence is not None:
        if not sequence:
            return np.array([0], dtype=int)
        head = sequence[0]
        nested_shape = builtin_shape(head) if isinstance(head, (list, tuple, np.ndarray)) else None
        if nested_shape is not None:
            return np.array([len(sequence), *list(nested_shape)], dtype=int)
        return np.array([len(sequence)], dtype=int)
    return np.array([], dtype=int)


def builtin_typeof(value: Any) -> str:
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, np.int8):
        return "i8"
    if isinstance(value, np.float16):
        return "f16"
    if isinstance(value, np.float32):
        return "f32"
    if isinstance(value, np.float64):
        return "f64"
    if _ml_dtypes is not None:
        if isinstance(value, _ml_dtypes.bfloat16):
            return "bf16"
        if isinstance(value, _ml_dtypes.float8_e4m3fn):
            return "f8e4m3"
    if isinstance(value, (int, np.integer)):
        return "i32"
    if isinstance(value, float):
        return "f32"
    if isinstance(value, str):
        return "str"
    if isinstance(value, np.ndarray):
        return "rectangular"
    sequence = _as_list_like(value)
    if sequence is not None:
        if len(sequence) == 0:
            return "rectangular"
        first_len = None
        for e in sequence:
            if not isinstance(e, (list, tuple, np.ndarray)):
                return "rectangular"
            current_len = len(e) if not isinstance(e, np.ndarray) else (e.shape[0] if len(e.shape) > 0 else 0)
            if first_len is None:
                first_len = current_len
            elif current_len != first_len:
                return "array"
        return "rectangular"
    if value is None:
        return "null"
    return type(value).__name__


def builtin_sum(array: Any) -> Any:
    if isinstance(array, np.ndarray):
        return np.sum(array)
    if isinstance(array, (list, tuple)):
        return np.sum(np.array(array))
    return array


def builtin_max(*args: Any) -> Any:
    if not args:
        raise TypeError("max() requires at least one argument")
    if len(args) == 1 and isinstance(args[0], np.ndarray):
        return np.max(args[0])
    if len(args) == 2:
        return np.maximum(np.asarray(args[0]), np.asarray(args[1]))
    return max(*args)


def builtin_min(*args: Any) -> Any:
    if not args:
        raise TypeError("min() requires at least one argument")
    if len(args) == 1 and isinstance(args[0], np.ndarray):
        return np.min(args[0])
    if len(args) == 2:
        return np.minimum(np.asarray(args[0]), np.asarray(args[1]))
    return min(*args)


def builtin_array_append(array: Any, value: Any) -> Any:
    lst = _as_list_like(array)
    if lst is None:
        lst = [array]
    lst.append(value)
    return np.array(lst, dtype=array.dtype) if isinstance(array, np.ndarray) else lst


class _DefaultVisitor(IRVisitor[T], Generic[T]):
    """IRVisitor base with a shared default result for nodes a subclass ignores."""

    def _default_result(self) -> T:
        raise NotImplementedError

    def visit_program(self, node: Any) -> T:
        return self._default_result()

    def visit_literal(self, node: Any) -> T:
        return self._default_result()

    def visit_identifier(self, node: Any) -> T:
        return self._default_result()

    def visit_binary_op(self, node: Any) -> T:
        return self._default_result()

    def visit_function_call(self, node: Any) -> T:
        return self._default_result()

    def visit_unary_op(self, node: Any) -> T:
        return self._default_result()

    def visit_rectangular_access(self, node: Any) -> T:
        return self._default_result()

    def visit_jagged_access(self, node: Any) -> T:
        return self._default_result()

    def visit_block_expression(self, node: Any) -> T:
        return self._default_result()

    def visit_if_expression(self, node: Any) -> T:
        return self._default_result()

    def visit_lambda(self, node: Any) -> T:
        return self._default_result()

    def visit_range(self, node: Any) -> T:
        return self._default_result()

    def visit_array_comprehension(self, node: Any) -> T:
        return self._default_result()

    def visit_array_literal(self, node: Any) -> T:
        return self._default_result()

    def visit_tuple_expression(self, node: Any) -> T:
        return self._default_result()

    def visit_tuple_access(self, node: Any) -> T:
        return self._default_result()

    def visit_interpolated_string(self, node: Any) -> T:
        return self._default_result()

    def visit_cast_expression(self, node: Any) -> T:
        return self._default_result()

    def visit_member_access(self, node: Any) -> T:
        return self._default_result()

    def visit_try_expression(self, node: Any) -> T:
        return self._default_result()

    def visit_match_expression(self, node: Any) -> T:
        return self._default_result()

    def visit_reduction_expression(self, node: Any) -> T:
        return self._default_result()

    def visit_where_expression(self, node: Any) -> T:
        return self._default_result()

    def visit_pipeline_expression(self, node: Any) -> T:
        return self._default_result()

    def visit_builtin_call(self, node: Any) -> T:
        return self._default_result()

    def visit_literal_pattern(self, node: Any) -> T:
        return self._default_result()

    def visit_identifier_pattern(self, node: Any) -> T:
        return self._default_result()

    def visit_wildcard_pattern(self, node: Any) -> T:
        return self._default_result()

    def visit_tuple_pattern(self, node: Any) -> T:
        return self._default_result()

    def visit_array_pattern(self, node: Any) -> T:
        return self._default_result()

    def visit_rest_pattern(self, node: Any) -> T:
        return self._default_result()

    def visit_guard_pattern(self, node: Any) -> T:
        return self._default_result()

    def visit_binding(self, node: Any) -> T:
        return self._default_result()

    def visit_module(self, node: Any) -> T:
        return self._default_result()


class _OptionalDefaultVisitor(_DefaultVisitor[Optional[T]], Generic[T]):
    def _default_result(self) -> Optional[T]:
        return None


class _NoneDefaultVisitor(_DefaultVisitor[None]):
    def _default_result(self) -> None:
        return None


class _PatternMatcher(_OptionalDefaultVisitor[Dict[DefId, Any]]):
    def __init__(self, value: Any, backend: Any):
        self.value = value
        self.backend = backend

    def visit_literal_pattern(self, node: LiteralPatternIR) -> Optional[Dict[DefId, Any]]:
        return {} if self.value == node.value else None

    def visit_identifier_pattern(self, node: IdentifierPatternIR) -> Optional[Dict[DefId, Any]]:
        did = node.defid
        return {did: self.value} if did else {}

    def visit_wildcard_pattern(self, node: Any) -> Optional[Dict[DefId, Any]]:
        return {}

    def visit_tuple_pattern(self, node: TuplePatternIR) -> Optional[Dict[DefId, Any]]:
        if not isinstance(self.value, tuple):
            return None
        val_list = list(self.value)
        has_rest = any(isinstance(p, RestPatternIR) for p in node.patterns)
        if has_rest:
            ri = next((i for i, p in enumerate(node.patterns) if isinstance(p, RestPatternIR)), None)
            if ri is None or len(val_list) < len(node.patterns) - 1:
                return None
            bindings: Dict[DefId, Any] = {}
            for i in range(ri):
                r = node.patterns[i].accept(_PatternMatcher(val_list[i], self.backend))
                if r is None: return None
                bindings.update(r)
            end = len(val_list) - (len(node.patterns) - ri - 1)
            rp = node.patterns[ri]
            if rp.pattern.defid is not None:
                bindings[rp.pattern.defid] = tuple(val_list[ri:end])
            for i in range(ri + 1, len(node.patterns)):
                r = node.patterns[i].accept(_PatternMatcher(val_list[end + (i - ri - 1)], self.backend))
                if r is None: return None
                bindings.update(r)
            return bindings
        if len(val_list) != len(node.patterns):
            return None
        bindings = {}
        for p, v in zip(node.patterns, val_list):
            r = p.accept(_PatternMatcher(v, self.backend))
            if r is None: return None
            bindings.update(r)
        return bindings

    def visit_array_pattern(self, node: ArrayPatternIR) -> Optional[Dict[DefId, Any]]:
        lst = _as_list_like(self.value)
        if lst is None: return None
        has_rest = any(isinstance(p, RestPatternIR) for p in node.patterns)
        if has_rest:
            ri = next((i for i, p in enumerate(node.patterns) if isinstance(p, RestPatternIR)), None)
            if ri is None or len(lst) < len(node.patterns) - 1: return None
            bindings: Dict[DefId, Any] = {}
            for i in range(ri):
                r = node.patterns[i].accept(_PatternMatcher(lst[i], self.backend))
                if r is None: return None
                bindings.update(r)
            end = len(lst) - (len(node.patterns) - ri - 1)
            rp = node.patterns[ri]
            if rp.pattern.defid is not None:
                bindings[rp.pattern.defid] = lst[ri:end]
            for i in range(ri + 1, len(node.patterns)):
                r = node.patterns[i].accept(_PatternMatcher(lst[end + (i - ri - 1)], self.backend))
                if r is None: return None
                bindings.update(r)
            return bindings
        if len(lst) != len(node.patterns): return None
        bindings = {}
        for p, v in zip(node.patterns, lst):
            r = p.accept(_PatternMatcher(v, self.backend))
            if r is None: return None
            bindings.update(r)
        return bindings

    def visit_rest_pattern(self, node: RestPatternIR) -> Optional[Dict[DefId, Any]]:
        did = node.pattern.defid
        return {did: self.value} if did else {}

    def visit_guard_pattern(self, node: GuardPatternIR) -> Optional[Dict[DefId, Any]]:
        return node.inner_pattern.accept(_PatternMatcher(self.value, self.backend))

    def visit_or_pattern(self, node) -> Optional[Dict[DefId, Any]]:
        for alt in node.alternatives:
            result = alt.accept(_PatternMatcher(self.value, self.backend))
            if result is not None:
                return result
        return None

    def visit_constructor_pattern(self, node) -> Optional[Dict[DefId, Any]]:
        if not isinstance(self.value, tuple) or len(self.value) < 1:
            return None
        tag = self.value[0] if isinstance(self.value, tuple) else None
        if tag != node.constructor_name:
            return None
        fields = self.value[1:] if len(self.value) > 1 else ()
        if len(fields) != len(node.patterns):
            return None
        bindings: Dict[DefId, Any] = {}
        for p, v in zip(node.patterns, fields):
            r = p.accept(_PatternMatcher(v, self.backend))
            if r is None:
                return None
            bindings.update(r)
        return bindings

    def visit_binding_pattern(self, node) -> Optional[Dict[DefId, Any]]:
        result = node.inner_pattern.accept(_PatternMatcher(self.value, self.backend))
        if result is None:
            return None
        did = node.identifier_pattern.defid
        if did is not None:
            result[did] = self.value
        return result

    def visit_range_pattern(self, node) -> Optional[Dict[DefId, Any]]:
        try:
            val = float(self.value) if not isinstance(self.value, (int, float)) else self.value
        except (TypeError, ValueError):
            return None
        if node.inclusive:
            return {} if node.start <= val <= node.end else None
        return {} if node.start <= val < node.end else None
