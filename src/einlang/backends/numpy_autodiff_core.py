"""Core JVP/VJP NumPy machinery for Einlang autodiff.

This module is the backend execution core behind the native autodiff design:

- traced tensor graph
- shared primitive JVP/VJP rules
- lazy Jacobians
- symbolic JVP printing
- direct IR-native tensor/primal evaluation for the subset currently supported in Einlang
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
import itertools
from typing import Any, Callable, Dict, Optional, Sequence, Tuple

import numpy as np

from ..ir.nodes import (
    BuiltinCallIR,
    ExpressionIR,
)
from ..shared.defid import DefId, fixed_builtin_defid


ArrayLike = Any
Index = Any


_LEN_BUILTIN_DEFID = fixed_builtin_defid("len")
_SHAPE_BUILTIN_DEFID = fixed_builtin_defid("shape")
_TYPEOF_BUILTIN_DEFID = fixed_builtin_defid("typeof")
_SUM_BUILTIN_DEFID = fixed_builtin_defid("sum")
_MAX_BUILTIN_DEFID = fixed_builtin_defid("max")
_MIN_BUILTIN_DEFID = fixed_builtin_defid("min")
_ASSERT_BUILTIN_DEFID = fixed_builtin_defid("assert")


def _as_array(value: ArrayLike) -> np.ndarray:
    return np.asarray(value, dtype=np.float64)


def _zeros_like(value: np.ndarray) -> np.ndarray:
    return np.zeros_like(np.asarray(value, dtype=np.float64))


def _ones_like(value: np.ndarray) -> np.ndarray:
    return np.ones_like(np.asarray(value, dtype=np.float64))


def _identity_seed(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return np.ones_like(np.asarray(value, dtype=np.float64))
    return 1.0


def _builtin_call_defid(expr: BuiltinCallIR) -> Optional[DefId]:
    raw = getattr(expr, "defid", None)
    if isinstance(raw, DefId):
        return raw
    return fixed_builtin_defid(expr.builtin_name)


def _eval_builtin_len(args: Sequence[Any]) -> Any:
    target = args[0]
    if isinstance(target, np.ndarray):
        return int(target.shape[0]) if target.ndim > 0 else int(target.size)
    return len(target)


def _eval_builtin_shape(args: Sequence[Any]) -> Any:
    target = args[0]
    if isinstance(target, np.ndarray):
        return np.asarray(target.shape, dtype=np.int32)
    return np.asarray(np.shape(target), dtype=np.int32)


def _eval_builtin_typeof(args: Sequence[Any]) -> Any:
    target = args[0]
    if isinstance(target, np.ndarray):
        return "rectangular"
    if isinstance(target, str):
        return "str"
    if isinstance(target, bool):
        return "bool"
    if isinstance(target, (int, np.integer)):
        return "i32"
    if isinstance(target, (float, np.floating)):
        return "f32"
    return type(target).__name__


def _eval_builtin_sum(args: Sequence[Any]) -> Any:
    return np.sum(np.asarray(args[0]))


def _eval_builtin_max(args: Sequence[Any]) -> Any:
    if len(args) == 1:
        return np.max(np.asarray(args[0]))
    return np.maximum(np.asarray(args[0]), np.asarray(args[1]))


def _eval_builtin_min(args: Sequence[Any]) -> Any:
    if len(args) == 1:
        return np.min(np.asarray(args[0]))
    return np.minimum(np.asarray(args[0]), np.asarray(args[1]))


def _eval_builtin_assert(args: Sequence[Any]) -> Any:
    cond = bool(np.asarray(args[0]).all())
    if not cond:
        raise IRAutodiffError(str(args[1]) if len(args) > 1 else "assertion failed")
    return None


_PRIMAL_BUILTIN_EVALUATORS: Dict[Optional[DefId], Callable[[Sequence[Any]], Any]] = {
    _LEN_BUILTIN_DEFID: _eval_builtin_len,
    _SHAPE_BUILTIN_DEFID: _eval_builtin_shape,
    _TYPEOF_BUILTIN_DEFID: _eval_builtin_typeof,
    _SUM_BUILTIN_DEFID: _eval_builtin_sum,
    _MAX_BUILTIN_DEFID: _eval_builtin_max,
    _MIN_BUILTIN_DEFID: _eval_builtin_min,
    _ASSERT_BUILTIN_DEFID: _eval_builtin_assert,
}


def _eval_member_shape(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return np.asarray(obj.shape, dtype=np.int32)
    return np.asarray(np.shape(obj), dtype=np.int32)


def _eval_member_size(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return int(obj.size)
    return int(np.size(obj))


_PRIMAL_MEMBER_ACCESSORS: Dict[Any, Callable[[Any], Any]] = {
    "shape": _eval_member_shape,
    "size": _eval_member_size,
}

_PYTHON_PRIMAL_MODULES: Dict[Tuple[str, ...], Any] = {
    ("python", "numpy"): np,
}


def _sum_to_shape(value: ArrayLike, shape: Tuple[int, ...]) -> np.ndarray:
    """Reduce a broadcasted cotangent back to ``shape``."""
    arr = _as_array(value)
    target = tuple(shape)
    if arr.shape == target:
        return arr
    if target == ():
        return np.asarray(arr.sum(), dtype=np.float64)

    while arr.ndim > len(target):
        arr = arr.sum(axis=0)

    for axis, size in enumerate(target):
        if size == 1 and arr.shape[axis] != 1:
            arr = arr.sum(axis=axis, keepdims=True)

    if arr.shape != target:
        arr = np.broadcast_to(arr, target).copy()
    return np.asarray(arr, dtype=np.float64)


def _normalize_axis(axis: Optional[Sequence[int] | int], ndim: int) -> Optional[Tuple[int, ...]]:
    if axis is None:
        return None
    if isinstance(axis, int):
        axes = (axis,)
    else:
        axes = tuple(int(a) for a in axis)
    out = []
    for a in axes:
        if a < 0:
            a += ndim
        if not (0 <= a < ndim):
            raise ValueError(f"axis {axis!r} out of bounds for ndim={ndim}")
        if a not in out:
            out.append(a)
    return tuple(sorted(out))


def _expand_reduced_like(
    value: ArrayLike,
    input_shape: Tuple[int, ...],
    axis: Optional[Tuple[int, ...]],
    keepdims: bool,
) -> np.ndarray:
    arr = _as_array(value)
    if keepdims:
        return np.broadcast_to(arr, input_shape).astype(np.float64, copy=False)
    if axis is None:
        while arr.ndim < len(input_shape):
            arr = np.expand_dims(arr, axis=0)
        return np.broadcast_to(arr, input_shape).astype(np.float64, copy=False)
    out = arr
    for a in axis:
        out = np.expand_dims(out, axis=a)
    return np.broadcast_to(out, input_shape).astype(np.float64, copy=False)


def _basis(shape: Tuple[int, ...], flat_index: int) -> np.ndarray:
    out = np.zeros(shape or (), dtype=np.float64)
    out.reshape(-1)[flat_index] = 1.0
    return out


def _ravel_index(index: Tuple[int, ...], shape: Tuple[int, ...]) -> int:
    if shape == ():
        return 0
    return int(np.ravel_multi_index(index, shape))


@dataclass(frozen=True)
class PrimitiveRule:
    jvp: Callable[[Tuple[np.ndarray, ...], Tuple[np.ndarray, ...], Dict[str, Any], np.ndarray], np.ndarray]
    vjp: Callable[[Tuple[np.ndarray, ...], np.ndarray, Dict[str, Any], np.ndarray], Tuple[np.ndarray, ...]]


@dataclass(frozen=True)
class SymbolicExpr:
    text: str
    is_zero: bool = False


class TensorOp(Enum):
    ADD = auto()
    SUB = auto()
    MUL = auto()
    DIV = auto()
    POW = auto()
    NEG = auto()
    EXP = auto()
    LOG = auto()
    SIN = auto()
    COS = auto()
    SUM = auto()
    RESHAPE = auto()
    GETITEM = auto()
    STACK = auto()
    WHERE = auto()
    CUSTOM_DIFF_CALL = auto()


class LinearizationMode(Enum):
    JVP = auto()
    VJP = auto()


_SYM_ZERO = SymbolicExpr("0.0", is_zero=True)


def _sym(text: str) -> SymbolicExpr:
    return SymbolicExpr(text, is_zero=False)


def _sym_parens(expr: SymbolicExpr) -> str:
    return expr.text if expr.is_zero else f"({expr.text})"


def _sym_add(a: SymbolicExpr, b: SymbolicExpr) -> SymbolicExpr:
    if a.is_zero:
        return b
    if b.is_zero:
        return a
    return _sym(f"{_sym_parens(a)} + {_sym_parens(b)}")


def _sym_sub(a: SymbolicExpr, b: SymbolicExpr) -> SymbolicExpr:
    if b.is_zero:
        return a
    if a.is_zero:
        return _sym(f"-{_sym_parens(b)}")
    return _sym(f"{_sym_parens(a)} - {_sym_parens(b)}")


def _sym_mul(a: SymbolicExpr, b: str) -> SymbolicExpr:
    if a.is_zero:
        return _SYM_ZERO
    return _sym(f"{_sym_parens(a)} * ({b})")


def _sym_apply_jacobian(label: str, wrt_name: str) -> str:
    return f"(@{label} / @{wrt_name}) · @{wrt_name}"


def _sym_tangent_ref(node: "Tensor", expr: SymbolicExpr) -> str:
    if node.name is not None:
        return f"@{node.name}"
    return expr.text


class Tensor:
    """Tiny traced NumPy tensor for standalone JVP/VJP experiments."""

    __array_priority__ = 1000

    def __init__(
        self,
        value: ArrayLike,
        *,
        parents: Sequence["Tensor"] = (),
        op: Optional[TensorOp] = None,
        meta: Optional[Dict[str, Any]] = None,
        name: Optional[str] = None,
        ir_node: Optional[ExpressionIR] = None,
    ) -> None:
        self.value = _as_array(value)
        self.parents = tuple(parents)
        self.op = op
        self.meta = dict(meta or {})
        self.name = name
        self.ir_node = ir_node

    @property
    def shape(self) -> Tuple[int, ...]:
        return tuple(self.value.shape)

    @property
    def ndim(self) -> int:
        return int(self.value.ndim)

    @property
    def size(self) -> int:
        return int(self.value.size)

    def __repr__(self) -> str:
        label = self.name if self.name is not None else (_TENSOR_OP_LABELS.get(self.op, "leaf"))
        return f"Tensor(name={label!r}, shape={self.shape}, value={self.value!r})"

    @staticmethod
    def leaf(value: ArrayLike, name: Optional[str] = None) -> "Tensor":
        return Tensor(value, name=name)

    def _binary(self, op: TensorOp, other: ArrayLike, *, ir_node: Optional[ExpressionIR] = None) -> "Tensor":
        rhs = ensure_tensor(other)
        value = _BINARY_IMPL[op](self.value, rhs.value)
        return Tensor(value, parents=(self, rhs), op=op, ir_node=ir_node)

    def __add__(self, other: ArrayLike) -> "Tensor":
        return self._binary(TensorOp.ADD, other)

    def __radd__(self, other: ArrayLike) -> "Tensor":
        return ensure_tensor(other)._binary(TensorOp.ADD, self)

    def __sub__(self, other: ArrayLike) -> "Tensor":
        return self._binary(TensorOp.SUB, other)

    def __rsub__(self, other: ArrayLike) -> "Tensor":
        return ensure_tensor(other)._binary(TensorOp.SUB, self)

    def __mul__(self, other: ArrayLike) -> "Tensor":
        return self._binary(TensorOp.MUL, other)

    def __rmul__(self, other: ArrayLike) -> "Tensor":
        return ensure_tensor(other)._binary(TensorOp.MUL, self)

    def __truediv__(self, other: ArrayLike) -> "Tensor":
        return self._binary(TensorOp.DIV, other)

    def __rtruediv__(self, other: ArrayLike) -> "Tensor":
        return ensure_tensor(other)._binary(TensorOp.DIV, self)

    def __pow__(self, other: ArrayLike) -> "Tensor":
        return self._binary(TensorOp.POW, other)

    def __rpow__(self, other: ArrayLike) -> "Tensor":
        return ensure_tensor(other)._binary(TensorOp.POW, self)

    def __neg__(self) -> "Tensor":
        return neg_tensor(self)

    def exp(self) -> "Tensor":
        return exp_tensor(self)

    def log(self) -> "Tensor":
        return log_tensor(self)

    def sin(self) -> "Tensor":
        return sin_tensor(self)

    def cos(self) -> "Tensor":
        return cos_tensor(self)

    def sum(
        self,
        axis: Optional[Sequence[int] | int] = None,
        keepdims: bool = False,
    ) -> "Tensor":
        meta = {"axis": _normalize_axis(axis, self.ndim), "keepdims": bool(keepdims)}
        return Tensor(
            self.value.sum(axis=axis, keepdims=keepdims),
            parents=(self,),
            op=TensorOp.SUM,
            meta=meta,
        )

    def reshape(self, *shape: int) -> "Tensor":
        if len(shape) == 1 and isinstance(shape[0], tuple):
            target = tuple(int(x) for x in shape[0])
        else:
            target = tuple(int(x) for x in shape)
        return reshape_tensor(self, target)

    def __getitem__(self, index: Index) -> "Tensor":
        return getitem_tensor(self, index)

    def named(self, name: str) -> "Tensor":
        self.name = name
        return self


def ensure_tensor(value: ArrayLike) -> Tensor:
    if isinstance(value, Tensor):
        return value
    return Tensor.leaf(value)


def binary_tensor(op: TensorOp, left: ArrayLike, right: ArrayLike, *, ir_node: Optional[ExpressionIR] = None) -> Tensor:
    return ensure_tensor(left)._binary(op, right, ir_node=ir_node)


def neg_tensor(value: ArrayLike, *, ir_node: Optional[ExpressionIR] = None) -> Tensor:
    tensor = ensure_tensor(value)
    return Tensor(-tensor.value, parents=(tensor,), op=TensorOp.NEG, ir_node=ir_node)


def exp_tensor(value: ArrayLike, *, ir_node: Optional[ExpressionIR] = None) -> Tensor:
    tensor = ensure_tensor(value)
    return Tensor(np.exp(tensor.value), parents=(tensor,), op=TensorOp.EXP, ir_node=ir_node)


def log_tensor(value: ArrayLike, *, ir_node: Optional[ExpressionIR] = None) -> Tensor:
    tensor = ensure_tensor(value)
    return Tensor(np.log(tensor.value), parents=(tensor,), op=TensorOp.LOG, ir_node=ir_node)


def sin_tensor(value: ArrayLike, *, ir_node: Optional[ExpressionIR] = None) -> Tensor:
    tensor = ensure_tensor(value)
    return Tensor(np.sin(tensor.value), parents=(tensor,), op=TensorOp.SIN, ir_node=ir_node)


def cos_tensor(value: ArrayLike, *, ir_node: Optional[ExpressionIR] = None) -> Tensor:
    tensor = ensure_tensor(value)
    return Tensor(np.cos(tensor.value), parents=(tensor,), op=TensorOp.COS, ir_node=ir_node)


def reshape_tensor(value: ArrayLike, shape: Tuple[int, ...], *, ir_node: Optional[ExpressionIR] = None) -> Tensor:
    tensor = ensure_tensor(value)
    return Tensor(tensor.value.reshape(shape), parents=(tensor,), op=TensorOp.RESHAPE, meta={"shape": shape}, ir_node=ir_node)


def getitem_tensor(value: ArrayLike, index: Index, *, ir_node: Optional[ExpressionIR] = None) -> Tensor:
    tensor = ensure_tensor(value)
    return Tensor(tensor.value[index], parents=(tensor,), op=TensorOp.GETITEM, meta={"index": index}, ir_node=ir_node)


def where_tensors(
    condition: ArrayLike,
    then_value: ArrayLike,
    else_value: ArrayLike,
    *,
    ir_node: Optional[ExpressionIR] = None,
) -> Tensor:
    cond_arr = np.asarray(condition)
    then_t = ensure_tensor(then_value)
    else_t = ensure_tensor(else_value)
    value = np.where(cond_arr, _as_array(then_t.value), _as_array(else_t.value))
    return Tensor(value, parents=(then_t, else_t), op=TensorOp.WHERE, meta={"condition": cond_arr}, ir_node=ir_node)


def stack_tensors(values: Sequence[Tensor], axis: int = 0, *, ir_node: Optional[ExpressionIR] = None) -> Tensor:
    items = [ensure_tensor(v) for v in values]
    if not items:
        raise ValueError("stack_tensors requires at least one tensor")
    value = np.stack([_as_array(item.value) for item in items], axis=axis)
    return Tensor(value, parents=tuple(items), op=TensorOp.STACK, meta={"axis": int(axis)}, ir_node=ir_node)


def custom_diff_call(
    value: ArrayLike,
    parents: Sequence[Tensor],
    *,
    call_text: str,
    jvp_fn: Callable[[Tuple[np.ndarray, ...], Tuple[np.ndarray, ...]], np.ndarray],
    vjp_fn: Callable[[Tuple[np.ndarray, ...], np.ndarray], Tuple[np.ndarray, ...]],
    symbolic_fn: Callable[[Sequence[Tensor], Sequence[SymbolicExpr], Dict[int, str]], SymbolicExpr],
    ir_node: Optional[ExpressionIR] = None,
) -> Tensor:
    return Tensor(
        value,
        parents=tuple(parents),
        op=TensorOp.CUSTOM_DIFF_CALL,
        meta={
            "call_text": call_text,
            "jvp_fn": jvp_fn,
            "vjp_fn": vjp_fn,
            "symbolic_fn": symbolic_fn,
        },
        ir_node=ir_node,
    )


_TENSOR_OP_LABELS: Dict[Optional[TensorOp], str] = {
    None: "leaf",
    TensorOp.ADD: "add",
    TensorOp.SUB: "sub",
    TensorOp.MUL: "mul",
    TensorOp.DIV: "div",
    TensorOp.POW: "pow",
    TensorOp.NEG: "neg",
    TensorOp.EXP: "exp",
    TensorOp.LOG: "log",
    TensorOp.SIN: "sin",
    TensorOp.COS: "cos",
    TensorOp.SUM: "sum",
    TensorOp.RESHAPE: "reshape",
    TensorOp.GETITEM: "getitem",
    TensorOp.STACK: "stack",
    TensorOp.WHERE: "where",
    TensorOp.CUSTOM_DIFF_CALL: "custom_diff_call",
}
_BINARY_TENSOR_OPS = frozenset(
    {
        TensorOp.ADD,
        TensorOp.SUB,
        TensorOp.MUL,
        TensorOp.DIV,
        TensorOp.POW,
    }
)
_UNARY_ELEMENTWISE_TENSOR_OPS = frozenset(
    {
        TensorOp.EXP,
        TensorOp.LOG,
        TensorOp.SIN,
        TensorOp.COS,
    }
)
_PRIMAL_BINARY_SYMBOLS: Dict[TensorOp, str] = {
    TensorOp.ADD: "+",
    TensorOp.SUB: "-",
    TensorOp.MUL: "*",
    TensorOp.DIV: "/",
    TensorOp.POW: "**",
}


def _format_meta(meta: Dict[str, Any], keys: Sequence[str]) -> str:
    parts = []
    for key in keys:
        if key in meta:
            parts.append(repr(meta[key]))
    return ", ".join(parts)


def primal_expr(node: Tensor, _cache: Optional[Dict[int, str]] = None) -> str:
    cache = {} if _cache is None else _cache
    oid = id(node)
    if oid in cache:
        return cache[oid]
    if node.name is not None:
        text = node.name
    elif node.op is None:
        text = np.array2string(node.value, separator=", ")
    elif node.op in _BINARY_TENSOR_OPS:
        left = primal_expr(node.parents[0], cache)
        right = primal_expr(node.parents[1], cache)
        symbol = _PRIMAL_BINARY_SYMBOLS[node.op]
        text = f"({left} {symbol} {right})"
    elif node.op is TensorOp.NEG:
        text = f"(-{primal_expr(node.parents[0], cache)})"
    elif node.op in _UNARY_ELEMENTWISE_TENSOR_OPS:
        text = f"{_TENSOR_OP_LABELS[node.op]}({primal_expr(node.parents[0], cache)})"
    elif node.op is TensorOp.SUM:
        text = (
            f"sum({primal_expr(node.parents[0], cache)}, "
            f"axis={node.meta.get('axis')}, keepdims={node.meta.get('keepdims', False)})"
        )
    elif node.op is TensorOp.RESHAPE:
        text = f"reshape({primal_expr(node.parents[0], cache)}, {node.meta['shape']})"
    elif node.op is TensorOp.GETITEM:
        text = f"{primal_expr(node.parents[0], cache)}[{node.meta['index']!r}]"
    elif node.op is TensorOp.STACK:
        text = f"stack([{', '.join(primal_expr(parent, cache) for parent in node.parents)}], axis={node.meta.get('axis', 0)})"
    elif node.op is TensorOp.WHERE:
        cond = node.meta.get("condition")
        text = f"where({np.asarray(cond).tolist()}, {primal_expr(node.parents[0], cache)}, {primal_expr(node.parents[1], cache)})"
    elif node.op is TensorOp.CUSTOM_DIFF_CALL:
        text = node.meta.get("call_text", node.name or "<custom_diff_call>")
    else:
        text = node.name if node.name is not None else f"<{_TENSOR_OP_LABELS.get(node.op, 'op')}>"
    cache[oid] = text
    return text


def symbolic_tangent_expr(
    node: Tensor,
    *,
    wrt: Optional[Tensor] = None,
    include_named_leaves: bool = True,
    _cache: Optional[Dict[int, SymbolicExpr]] = None,
    _primal_cache: Optional[Dict[int, str]] = None,
) -> SymbolicExpr:
    cache = {} if _cache is None else _cache
    primal_cache = {} if _primal_cache is None else _primal_cache
    oid = id(node)
    if oid in cache:
        return cache[oid]

    if node.op is None:
        if wrt is not None:
            result = _sym(f"@{node.name or 'x'}") if node is wrt else _SYM_ZERO
        elif include_named_leaves and node.name is not None:
            result = _sym(f"@{node.name}")
        else:
            result = _SYM_ZERO
        cache[oid] = result
        return result

    parents = node.parents
    parent_primal = [primal_expr(parent, primal_cache) for parent in parents]
    tangents = [
        symbolic_tangent_expr(
            parent,
            wrt=wrt,
            include_named_leaves=include_named_leaves,
            _cache=cache,
            _primal_cache=primal_cache,
        )
        for parent in parents
    ]
    tangent_refs = [_sym_tangent_ref(parent, tangent) for parent, tangent in zip(parents, tangents)]

    if node.op is TensorOp.ADD:
        result = _sym_add(tangents[0], tangents[1])
    elif node.op is TensorOp.SUB:
        result = _sym_sub(tangents[0], tangents[1])
    elif node.op is TensorOp.MUL:
        result = _sym_add(
            _SYM_ZERO if tangents[0].is_zero else _sym(f"({tangent_refs[0]}) * ({parent_primal[1]})"),
            _SYM_ZERO if tangents[1].is_zero else _sym(f"({tangent_refs[1]}) * ({parent_primal[0]})"),
        )
    elif node.op is TensorOp.DIV:
        if tangents[0].is_zero and tangents[1].is_zero:
            result = _SYM_ZERO
        else:
            result = _sym(
                f"(({parent_primal[1]}) * ({tangent_refs[0]}) - ({parent_primal[0]}) * ({tangent_refs[1]})) / "
                f"(({parent_primal[1]}) ** 2.0)"
            )
    elif node.op is TensorOp.POW:
        if tangents[0].is_zero and tangents[1].is_zero:
            result = _SYM_ZERO
        else:
            result = _sym(
                f"({primal_expr(node, primal_cache)}) * "
                f"(({tangent_refs[1]}) * log({parent_primal[0]}) + ({parent_primal[1]}) * ({tangent_refs[0]}) / ({parent_primal[0]}))"
            )
    elif node.op is TensorOp.NEG:
        result = _SYM_ZERO if tangents[0].is_zero else _sym(f"-({tangent_refs[0]})")
    elif node.op is TensorOp.EXP:
        result = _SYM_ZERO if tangents[0].is_zero else _sym(f"exp({parent_primal[0]}) * ({tangent_refs[0]})")
    elif node.op is TensorOp.LOG:
        result = _SYM_ZERO if tangents[0].is_zero else _sym(f"({tangent_refs[0]}) / ({parent_primal[0]})")
    elif node.op is TensorOp.SIN:
        result = _SYM_ZERO if tangents[0].is_zero else _sym(f"cos({parent_primal[0]}) * ({tangent_refs[0]})")
    elif node.op is TensorOp.COS:
        result = _SYM_ZERO if tangents[0].is_zero else _sym(f"-sin({parent_primal[0]}) * ({tangent_refs[0]})")
    elif node.op is TensorOp.SUM:
        result = _SYM_ZERO if tangents[0].is_zero else _sym(
            f"sum({tangent_refs[0]}, axis={node.meta.get('axis')}, keepdims={node.meta.get('keepdims', False)})"
        )
    elif node.op is TensorOp.RESHAPE:
        result = _SYM_ZERO if tangents[0].is_zero else _sym(f"reshape({tangent_refs[0]}, {node.meta['shape']})")
    elif node.op is TensorOp.GETITEM:
        result = _SYM_ZERO if tangents[0].is_zero else _sym(f"({tangent_refs[0]})[{node.meta['index']!r}]")
    elif node.op is TensorOp.STACK:
        if all(t.is_zero for t in tangents):
            result = _SYM_ZERO
        else:
            parts = [ref if not tangent.is_zero else "0.0" for ref, tangent in zip(tangent_refs, tangents)]
            result = _sym(f"stack([{', '.join(parts)}], axis={node.meta.get('axis', 0)})")
    elif node.op is TensorOp.WHERE:
        if tangents[0].is_zero and tangents[1].is_zero:
            result = _SYM_ZERO
        else:
            result = _sym(
                f"where({np.asarray(node.meta.get('condition')).tolist()}, "
                f"{tangent_refs[0] if not tangents[0].is_zero else '0.0'}, "
                f"{tangent_refs[1] if not tangents[1].is_zero else '0.0'})"
            )
    elif node.op is TensorOp.CUSTOM_DIFF_CALL:
        result = node.meta["symbolic_fn"](parents, tangents, primal_cache)
    else:
        result = _sym(f"@{node.name or _TENSOR_OP_LABELS.get(node.op, 'op')}")

    cache[oid] = result
    return result


def symbolic_tangent_program(
    root: Tensor,
    *,
    wrt: Optional[Tensor] = None,
    include_named_leaves: bool = True,
) -> str:
    topo = _toposort(root)
    cache: Dict[int, SymbolicExpr] = {}
    primal_cache: Dict[int, str] = {}
    lines = []
    for node in topo:
        expr = symbolic_tangent_expr(
            node,
            wrt=wrt,
            include_named_leaves=include_named_leaves,
            _cache=cache,
            _primal_cache=primal_cache,
        )
        if node.name is not None and (node.parents or node is root):
            lines.append(f"let @{node.name} = {expr.text};")
    if lines:
        return "\n".join(lines)
    return symbolic_tangent_expr(root, wrt=wrt, include_named_leaves=include_named_leaves, _cache=cache, _primal_cache=primal_cache).text


def symbolic_jacobian_application(output: Tensor, wrt: Tensor) -> str:
    out_name = output.name or "out"
    wrt_name = wrt.name or "x"
    return _sym_apply_jacobian(out_name, wrt_name)


def _toposort(root: Tensor) -> list[Tensor]:
    order: list[Tensor] = []
    seen: set[int] = set()

    def dfs(node: Tensor) -> None:
        oid = id(node)
        if oid in seen:
            return
        seen.add(oid)
        for parent in node.parents:
            dfs(parent)
        order.append(node)

    dfs(root)
    return order


def _jvp_add(inputs: Tuple[np.ndarray, ...], tangents: Tuple[np.ndarray, ...], _meta: Dict[str, Any], _out: np.ndarray) -> np.ndarray:
    return tangents[0] + tangents[1]


def _vjp_add(_inputs: Tuple[np.ndarray, ...], cotangent: np.ndarray, _meta: Dict[str, Any], _out: np.ndarray) -> Tuple[np.ndarray, ...]:
    return cotangent, cotangent


def _jvp_sub(inputs: Tuple[np.ndarray, ...], tangents: Tuple[np.ndarray, ...], _meta: Dict[str, Any], _out: np.ndarray) -> np.ndarray:
    del inputs
    return tangents[0] - tangents[1]


def _vjp_sub(_inputs: Tuple[np.ndarray, ...], cotangent: np.ndarray, _meta: Dict[str, Any], _out: np.ndarray) -> Tuple[np.ndarray, ...]:
    return cotangent, -cotangent


def _jvp_mul(inputs: Tuple[np.ndarray, ...], tangents: Tuple[np.ndarray, ...], _meta: Dict[str, Any], _out: np.ndarray) -> np.ndarray:
    a, b = inputs
    da, db = tangents
    return da * b + a * db


def _vjp_mul(inputs: Tuple[np.ndarray, ...], cotangent: np.ndarray, _meta: Dict[str, Any], _out: np.ndarray) -> Tuple[np.ndarray, ...]:
    a, b = inputs
    return cotangent * b, cotangent * a


def _jvp_div(inputs: Tuple[np.ndarray, ...], tangents: Tuple[np.ndarray, ...], _meta: Dict[str, Any], _out: np.ndarray) -> np.ndarray:
    a, b = inputs
    da, db = tangents
    return (da * b - a * db) / (b * b)


def _vjp_div(inputs: Tuple[np.ndarray, ...], cotangent: np.ndarray, _meta: Dict[str, Any], _out: np.ndarray) -> Tuple[np.ndarray, ...]:
    a, b = inputs
    return cotangent / b, -(cotangent * a) / (b * b)


def _jvp_pow(inputs: Tuple[np.ndarray, ...], tangents: Tuple[np.ndarray, ...], _meta: Dict[str, Any], out: np.ndarray) -> np.ndarray:
    base, exp = inputs
    dbase, dexp = tangents
    return out * (dexp * np.log(base) + exp * dbase / base)


def _vjp_pow(inputs: Tuple[np.ndarray, ...], cotangent: np.ndarray, _meta: Dict[str, Any], out: np.ndarray) -> Tuple[np.ndarray, ...]:
    base, exp = inputs
    dbase = cotangent * out * exp / base
    dexp = cotangent * out * np.log(base)
    return dbase, dexp


def _jvp_neg(_inputs: Tuple[np.ndarray, ...], tangents: Tuple[np.ndarray, ...], _meta: Dict[str, Any], _out: np.ndarray) -> np.ndarray:
    return -tangents[0]


def _vjp_neg(_inputs: Tuple[np.ndarray, ...], cotangent: np.ndarray, _meta: Dict[str, Any], _out: np.ndarray) -> Tuple[np.ndarray, ...]:
    return (-cotangent,)


def _jvp_exp(_inputs: Tuple[np.ndarray, ...], tangents: Tuple[np.ndarray, ...], _meta: Dict[str, Any], out: np.ndarray) -> np.ndarray:
    return out * tangents[0]


def _vjp_exp(_inputs: Tuple[np.ndarray, ...], cotangent: np.ndarray, _meta: Dict[str, Any], out: np.ndarray) -> Tuple[np.ndarray, ...]:
    return (cotangent * out,)


def _jvp_log(inputs: Tuple[np.ndarray, ...], tangents: Tuple[np.ndarray, ...], _meta: Dict[str, Any], _out: np.ndarray) -> np.ndarray:
    return tangents[0] / inputs[0]


def _vjp_log(inputs: Tuple[np.ndarray, ...], cotangent: np.ndarray, _meta: Dict[str, Any], _out: np.ndarray) -> Tuple[np.ndarray, ...]:
    return (cotangent / inputs[0],)


def _jvp_sin(inputs: Tuple[np.ndarray, ...], tangents: Tuple[np.ndarray, ...], _meta: Dict[str, Any], _out: np.ndarray) -> np.ndarray:
    return np.cos(inputs[0]) * tangents[0]


def _vjp_sin(inputs: Tuple[np.ndarray, ...], cotangent: np.ndarray, _meta: Dict[str, Any], _out: np.ndarray) -> Tuple[np.ndarray, ...]:
    return (cotangent * np.cos(inputs[0]),)


def _jvp_cos(inputs: Tuple[np.ndarray, ...], tangents: Tuple[np.ndarray, ...], _meta: Dict[str, Any], _out: np.ndarray) -> np.ndarray:
    return -np.sin(inputs[0]) * tangents[0]


def _vjp_cos(inputs: Tuple[np.ndarray, ...], cotangent: np.ndarray, _meta: Dict[str, Any], _out: np.ndarray) -> Tuple[np.ndarray, ...]:
    return (-cotangent * np.sin(inputs[0]),)


def _jvp_sum(_inputs: Tuple[np.ndarray, ...], tangents: Tuple[np.ndarray, ...], meta: Dict[str, Any], _out: np.ndarray) -> np.ndarray:
    axis = meta.get("axis")
    keepdims = bool(meta.get("keepdims", False))
    return tangents[0].sum(axis=axis, keepdims=keepdims)


def _vjp_sum(inputs: Tuple[np.ndarray, ...], cotangent: np.ndarray, meta: Dict[str, Any], _out: np.ndarray) -> Tuple[np.ndarray, ...]:
    axis = meta.get("axis")
    keepdims = bool(meta.get("keepdims", False))
    return (_expand_reduced_like(cotangent, tuple(inputs[0].shape), axis, keepdims),)


def _jvp_reshape(_inputs: Tuple[np.ndarray, ...], tangents: Tuple[np.ndarray, ...], meta: Dict[str, Any], _out: np.ndarray) -> np.ndarray:
    return tangents[0].reshape(meta["shape"])


def _vjp_reshape(inputs: Tuple[np.ndarray, ...], cotangent: np.ndarray, _meta: Dict[str, Any], _out: np.ndarray) -> Tuple[np.ndarray, ...]:
    return (cotangent.reshape(inputs[0].shape),)


def _jvp_getitem(_inputs: Tuple[np.ndarray, ...], tangents: Tuple[np.ndarray, ...], meta: Dict[str, Any], _out: np.ndarray) -> np.ndarray:
    return tangents[0][meta["index"]]


def _vjp_getitem(inputs: Tuple[np.ndarray, ...], cotangent: np.ndarray, meta: Dict[str, Any], _out: np.ndarray) -> Tuple[np.ndarray, ...]:
    base = np.zeros_like(inputs[0], dtype=np.float64)
    base[meta["index"]] += cotangent
    return (base,)


def _jvp_stack(_inputs: Tuple[np.ndarray, ...], tangents: Tuple[np.ndarray, ...], meta: Dict[str, Any], _out: np.ndarray) -> np.ndarray:
    return np.stack([_as_array(t) for t in tangents], axis=int(meta.get("axis", 0)))


def _vjp_stack(_inputs: Tuple[np.ndarray, ...], cotangent: np.ndarray, meta: Dict[str, Any], _out: np.ndarray) -> Tuple[np.ndarray, ...]:
    axis = int(meta.get("axis", 0))
    moved = np.moveaxis(_as_array(cotangent), axis, 0)
    return tuple(np.asarray(moved[i], dtype=np.float64) for i in range(moved.shape[0]))


def _jvp_where(_inputs: Tuple[np.ndarray, ...], tangents: Tuple[np.ndarray, ...], meta: Dict[str, Any], _out: np.ndarray) -> np.ndarray:
    cond = np.asarray(meta["condition"])
    return np.where(cond, _as_array(tangents[0]), _as_array(tangents[1]))


def _vjp_where(inputs: Tuple[np.ndarray, ...], cotangent: np.ndarray, meta: Dict[str, Any], _out: np.ndarray) -> Tuple[np.ndarray, ...]:
    cond = np.asarray(meta["condition"])
    then_shape = tuple(np.asarray(inputs[0]).shape)
    else_shape = tuple(np.asarray(inputs[1]).shape)
    then_ct = _sum_to_shape(np.where(cond, cotangent, 0.0), then_shape)
    else_ct = _sum_to_shape(np.where(cond, 0.0, cotangent), else_shape)
    return then_ct, else_ct


_RULES: Dict[TensorOp, PrimitiveRule] = {
    TensorOp.ADD: PrimitiveRule(_jvp_add, _vjp_add),
    TensorOp.SUB: PrimitiveRule(_jvp_sub, _vjp_sub),
    TensorOp.MUL: PrimitiveRule(_jvp_mul, _vjp_mul),
    TensorOp.DIV: PrimitiveRule(_jvp_div, _vjp_div),
    TensorOp.POW: PrimitiveRule(_jvp_pow, _vjp_pow),
    TensorOp.NEG: PrimitiveRule(_jvp_neg, _vjp_neg),
    TensorOp.EXP: PrimitiveRule(_jvp_exp, _vjp_exp),
    TensorOp.LOG: PrimitiveRule(_jvp_log, _vjp_log),
    TensorOp.SIN: PrimitiveRule(_jvp_sin, _vjp_sin),
    TensorOp.COS: PrimitiveRule(_jvp_cos, _vjp_cos),
    TensorOp.SUM: PrimitiveRule(_jvp_sum, _vjp_sum),
    TensorOp.RESHAPE: PrimitiveRule(_jvp_reshape, _vjp_reshape),
    TensorOp.GETITEM: PrimitiveRule(_jvp_getitem, _vjp_getitem),
    TensorOp.STACK: PrimitiveRule(_jvp_stack, _vjp_stack),
    TensorOp.WHERE: PrimitiveRule(_jvp_where, _vjp_where),
}


_BINARY_IMPL: Dict[TensorOp, Callable[[np.ndarray, np.ndarray], np.ndarray]] = {
    TensorOp.ADD: np.add,
    TensorOp.SUB: np.subtract,
    TensorOp.MUL: np.multiply,
    TensorOp.DIV: np.divide,
    TensorOp.POW: np.power,
}


def jvp(root: Tensor, tangents: Dict[Tensor, ArrayLike]) -> np.ndarray:
    topo = _toposort(root)
    tangent_map: Dict[Tensor, np.ndarray] = {}

    for node in topo:
        if node in tangents:
            tangent_map[node] = _sum_to_shape(_as_array(tangents[node]), node.shape)
            continue
        if node.op is None:
            tangent_map[node] = _as_array(tangents.get(node, _zeros_like(node.value)))
            continue
        if node.op is TensorOp.CUSTOM_DIFF_CALL:
            in_values = tuple(parent.value for parent in node.parents)
            in_tangents = tuple(tangent_map[parent] for parent in node.parents)
            tangent_map[node] = _sum_to_shape(node.meta["jvp_fn"](in_values, in_tangents), node.shape)
            continue
        inputs = tuple(parent.value for parent in node.parents)
        in_tangents = tuple(tangent_map[parent] for parent in node.parents)
        rule = _RULES[node.op]
        tangent = rule.jvp(inputs, in_tangents, node.meta, node.value)
        tangent_map[node] = _sum_to_shape(tangent, node.shape)

    return tangent_map[root]


def vjp(root: Tensor, cotangent: ArrayLike, wrt: Optional[Tensor] = None) -> Any:
    topo = _toposort(root)
    cotangent_map: Dict[Tensor, np.ndarray] = {root: _sum_to_shape(cotangent, root.shape)}

    for node in reversed(topo):
        current = cotangent_map.get(node)
        if current is None or node.op is None:
            continue
        if node.op is TensorOp.CUSTOM_DIFF_CALL:
            inputs = tuple(parent.value for parent in node.parents)
            parent_cotangents = node.meta["vjp_fn"](inputs, current)
            for parent, parent_cotangent in zip(node.parents, parent_cotangents):
                reduced = _sum_to_shape(parent_cotangent, parent.shape)
                if parent in cotangent_map:
                    cotangent_map[parent] = cotangent_map[parent] + reduced
                else:
                    cotangent_map[parent] = reduced
            continue
        rule = _RULES[node.op]
        inputs = tuple(parent.value for parent in node.parents)
        parent_cotangents = rule.vjp(inputs, current, node.meta, node.value)
        for parent, parent_cotangent in zip(node.parents, parent_cotangents):
            reduced = _sum_to_shape(parent_cotangent, parent.shape)
            if parent in cotangent_map:
                cotangent_map[parent] = cotangent_map[parent] + reduced
            else:
                cotangent_map[parent] = reduced

    if wrt is not None:
        return cotangent_map.get(wrt, _zeros_like(wrt.value))
    return cotangent_map


def tangent_of(target: Tensor, wrt: Tensor) -> np.ndarray:
    return jvp(target, {wrt: _ones_like(wrt.value)})


def grad(loss: Tensor, wrt: Tensor) -> np.ndarray:
    if loss.shape != ():
        raise ValueError(f"grad requires a scalar output, got shape {loss.shape}")
    return vjp(loss, np.array(1.0, dtype=np.float64), wrt)


class LazyJacobianTensor:
    """Lazy Jacobian view backed by JVP or VJP basis evaluations."""

    def __init__(self, output: Tensor, wrt: Tensor) -> None:
        self.output = output
        self.wrt = wrt
        self._materialized: Optional[np.ndarray] = None

    @property
    def shape(self) -> Tuple[int, ...]:
        return self.output.shape + self.wrt.shape

    @property
    def ndim(self) -> int:
        return len(self.shape)

    def mode(self) -> LinearizationMode:
        return LinearizationMode.JVP if self.wrt.size <= self.output.size else LinearizationMode.VJP

    def materialize_via_jvp(self) -> np.ndarray:
        y_size = self.output.size
        x_size = self.wrt.size
        out = np.zeros((y_size, x_size), dtype=np.float64)
        for flat_x in range(x_size):
            tangent = _basis(self.wrt.shape, flat_x)
            col = jvp(self.output, {self.wrt: tangent}).reshape(-1)
            out[:, flat_x] = col
        return out.reshape(self.shape)

    def materialize_via_vjp(self) -> np.ndarray:
        y_size = self.output.size
        x_size = self.wrt.size
        out = np.zeros((y_size, x_size), dtype=np.float64)
        for flat_y in range(y_size):
            cotangent = _basis(self.output.shape, flat_y)
            row = vjp(self.output, cotangent, self.wrt).reshape(-1)
            out[flat_y, :] = row
        return out.reshape(self.shape)

    def materialize(self) -> np.ndarray:
        if self._materialized is None:
            self._materialized = (
                self.materialize_via_jvp()
                if self.mode() is LinearizationMode.JVP
                else self.materialize_via_vjp()
            )
        return self._materialized

    def row(self, output_index: Tuple[int, ...]) -> np.ndarray:
        cotangent = _basis(self.output.shape, _ravel_index(output_index, self.output.shape))
        return vjp(self.output, cotangent, self.wrt)

    def column(self, wrt_index: Tuple[int, ...]) -> np.ndarray:
        tangent = _basis(self.wrt.shape, _ravel_index(wrt_index, self.wrt.shape))
        return jvp(self.output, {self.wrt: tangent})

    def entry(self, output_index: Tuple[int, ...], wrt_index: Tuple[int, ...]) -> float:
        if self.mode() is LinearizationMode.JVP:
            return float(np.asarray(self.column(wrt_index))[output_index])
        return float(np.asarray(self.row(output_index))[wrt_index])

    def __array__(self, dtype: Optional[np.dtype] = None) -> np.ndarray:
        arr = self.materialize()
        if dtype is not None:
            return arr.astype(dtype, copy=False)
        return arr

    def tolist(self) -> Any:
        return self.materialize().tolist()

    def __getitem__(self, key: Index) -> Any:
        if not isinstance(key, tuple):
            key = (key,)
        if Ellipsis in key or len(key) != self.ndim:
            return self.materialize()[key]

        out_rank = len(self.output.shape)
        out_key = key[:out_rank]
        wrt_key = key[out_rank:]
        out_all_int = all(isinstance(k, (int, np.integer)) for k in out_key)
        wrt_all_int = all(isinstance(k, (int, np.integer)) for k in wrt_key)

        if out_all_int and wrt_all_int:
            return self.entry(tuple(int(k) for k in out_key), tuple(int(k) for k in wrt_key))
        if out_all_int:
            return self.row(tuple(int(k) for k in out_key))[wrt_key]
        if wrt_all_int:
            return self.column(tuple(int(k) for k in wrt_key))[out_key]
        return self.materialize()[key]


def jacobian(output: Tensor, wrt: Tensor) -> LazyJacobianTensor:
    return LazyJacobianTensor(output, wrt)

def _assert_allclose(actual: ArrayLike, expected: ArrayLike, *, atol: float = 1e-8, rtol: float = 1e-8) -> None:
    a = _as_array(actual)
    e = _as_array(expected)
    if not np.allclose(a, e, atol=atol, rtol=rtol):
        raise AssertionError(f"not close\nactual={a}\nexpected={e}")
