"""Native IR JVP/VJP NumPy backend for Einlang autodiff.

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

from ..passes.autodiff.compiler import (
    AutodiffCompiledFacts,
    AutodiffBuiltinRequest,
    autodiff_builtin_request,
    binding_for_defid,
)
from ..ir.nodes import (
    ArrayLiteralIR,
    BinaryOpIR,
    BindingIR,
    BlockExpressionIR,
    BuiltinCallIR,
    CastExpressionIR,
    DifferentialIR,
    EinsteinIR,
    ExpressionIR,
    FunctionCallIR,
    FunctionValueIR,
    IdentifierIR,
    IfExpressionIR,
    IndexRestIR,
    IndexVarIR,
    LiteralIR,
    MemberAccessIR,
    LoweredEinsteinIR,
    LoweredRecurrenceIR,
    RangeIR,
    RectangularAccessIR,
    ReductionExpressionIR,
    UnaryOpIR,
    is_function_binding,
)
from ..shared.autodiff_intrinsics import AutodiffBuiltinKind
from ..shared.defid import DefId, fixed_builtin_defid
from ..shared.types import BinaryOp, ReductionOp, UnaryOp


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


class IRAutodiffError(RuntimeError):
    """Raised when native Einlang autodiff cannot evaluate a requested IR graph."""


class NativeIRAutodiffRuntime:
    def __init__(self, compiled_facts: AutodiffCompiledFacts, value_lookup: Callable[[DefId], Any]) -> None:
        self._compiled_facts = compiled_facts
        self._bindings: Dict[DefId, BindingIR] = dict(compiled_facts.get("bindings") or {})
        self._functions: Dict[DefId, BindingIR] = dict(compiled_facts.get("functions") or {})
        self._leaf_defids = set(compiled_facts.get("leaf_defids") or set())
        self._self_recursive_defids = set(compiled_facts.get("self_recursive_defids") or set())
        self._value_lookup = value_lookup
        self._tensor_cache: Dict[Tuple[DefId, bool], Tensor] = {}
        self._self_tensor_store_stack: list[Dict[DefId, Dict[Tuple[int, ...], Tensor]]] = []
        self._force_structural_depth = 0

    def binding_tensor(self, defid: DefId) -> Tensor:
        exact = self._force_structural_depth > 0
        cache_key = (defid, exact)
        cached = self._tensor_cache.get(cache_key)
        if cached is not None:
            return cached
        binding = self._bindings.get(defid)
        if binding is None:
            binding = self._functions.get(defid)
        name = getattr(binding, "name", None)
        current_primal = self._value_lookup(defid)
        if (
            binding is None
            or is_function_binding(binding)
            or binding.expr is None
            or defid in self._leaf_defids
            or (
                not exact
                and
                current_primal is not None
                and (
                    isinstance(binding.expr, LoweredRecurrenceIR)
                    or (
                        binding.defid is not None
                        and binding.defid in self._self_recursive_defids
                    )
                )
            )
        ):
            primal = current_primal
            if primal is None:
                raise IRAutodiffError(f"Missing primal value for autodiff leaf {name or defid}")
            tensor = Tensor.leaf(primal, name=name)
            self._tensor_cache[cache_key] = tensor
            return tensor
        tensor = self._eval_binding_tensor(binding, {})
        if name:
            tensor.named(name)
        self._tensor_cache[cache_key] = tensor
        return tensor

    def _binding_tensor_exact(self, defid: DefId) -> Tensor:
        self._force_structural_depth += 1
        try:
            return self.binding_tensor(defid)
        finally:
            self._force_structural_depth -= 1

    def _eval_binding_tensor(self, binding: BindingIR, locals_map: Dict[DefId, Tensor]) -> Tensor:
        expr = binding.expr
        if expr is None:
            raise IRAutodiffError(f"Binding {binding.name or binding.defid} has no expression")
        if isinstance(expr, EinsteinIR):
            return self._eval_einstein_tensor(
                expr,
                locals_map,
                owner_defid=binding.defid,
                owner_name=binding.name,
            )
        return self.eval_tensor(expr, locals_map)

    def _autodiff_request(self, expr: BuiltinCallIR) -> Optional[AutodiffBuiltinRequest]:
        return autodiff_builtin_request(self._compiled_facts, expr)

    def _resolve_autodiff_target_tensor(
        self,
        target_defid: DefId,
        locals_map: Dict[DefId, Tensor],
        *,
        exact: bool,
    ) -> Tensor:
        local = locals_map.get(target_defid)
        if local is not None:
            return local
        if exact:
            return self._binding_tensor_exact(target_defid)
        return self.binding_tensor(target_defid)

    def _resolve_autodiff_target_name(
        self,
        request: AutodiffBuiltinRequest,
        index: int,
        locals_map: Dict[DefId, Tensor],
        *,
        exact: bool,
    ) -> str:
        target_defid = request.target_defids[index]
        tensor = locals_map.get(target_defid)
        if tensor is not None and tensor.name:
            return tensor.name
        binding = self._bindings.get(target_defid) or self._functions.get(target_defid)
        name = getattr(binding, "name", None)
        if name:
            return name
        if tensor is None:
            tensor = self._resolve_autodiff_target_tensor(target_defid, locals_map, exact=exact)
        if tensor.name:
            return tensor.name
        if index < len(request.target_names):
            return request.target_names[index]
        return "x" if index else "y"

    def _eval_autodiff_builtin(
        self,
        expr: BuiltinCallIR,
        locals_map: Dict[DefId, Tensor],
        *,
        exact: bool,
    ) -> Any:
        request = self._autodiff_request(expr)
        if request is None:
            raise IRAutodiffError(f"Unsupported autodiff builtin in native autodiff: {expr.builtin_name}")
        kind = request.kind

        if kind is AutodiffBuiltinKind.TANGENT:
            target_defid = request.target_defids[0]
            target = locals_map.get(target_defid)
            if target is not None:
                return _identity_seed(target.value)
            primal = self._value_lookup(target_defid)
            if primal is None:
                primal = self.binding_tensor(target_defid).value
            return _identity_seed(primal)

        if kind is AutodiffBuiltinKind.JACOBIAN:
            numerator_defid, denominator_defid = request.target_defids
            numerator = self._resolve_autodiff_target_tensor(numerator_defid, locals_map, exact=exact)
            denominator = self._resolve_autodiff_target_tensor(denominator_defid, locals_map, exact=exact)
            lazy = jacobian(numerator, denominator)
            if numerator.size == 1 and denominator.size == 1:
                scalar = np.asarray(lazy).reshape(-1)[0]
                return scalar.item() if hasattr(scalar, "item") else scalar
            return lazy

        if kind is AutodiffBuiltinKind.SYMBOLIC_TANGENT:
            target_defid = request.target_defids[0]
            target = self._resolve_autodiff_target_tensor(target_defid, locals_map, exact=exact)
            if target.name is None:
                target.named(self._resolve_autodiff_target_name(request, 0, locals_map, exact=exact))
            return symbolic_tangent_program(target)

        if kind is AutodiffBuiltinKind.SYMBOLIC_JACOBIAN:
            num_name = self._resolve_autodiff_target_name(request, 0, locals_map, exact=exact)
            den_name = self._resolve_autodiff_target_name(request, 1, locals_map, exact=exact)
            return f"(@{num_name} / @{den_name}) · @{den_name}"

        raise IRAutodiffError(f"Unknown autodiff builtin in native autodiff: {expr.builtin_name}")

    def _eval_custom_diff_tensor(
        self,
        expr: ExpressionIR,
        primal_locals: Dict[DefId, Tensor],
        tangent_locals: Dict[DefId, Tensor],
    ) -> Tensor:
        if isinstance(expr, DifferentialIR):
            raise IRAutodiffError("DifferentialIR should be rewritten before native autodiff runtime execution")
        if isinstance(expr, LiteralIR):
            return Tensor.leaf(expr.value)
        if isinstance(expr, ArrayLiteralIR):
            return Tensor.leaf(self.eval_primal(expr, primal_locals))
        if isinstance(expr, BuiltinCallIR):
            request = self._autodiff_request(expr)
            if request is not None and request.kind is AutodiffBuiltinKind.TANGENT:
                target_defid = request.target_defids[0]
                tangent = tangent_locals.get(target_defid)
                if tangent is None:
                    raise IRAutodiffError(
                        f"Autodiff tangent builtin in custom diff body missing tangent for {target_defid}"
                    )
                return tangent
            if request is not None:
                return Tensor.leaf(self._eval_autodiff_builtin(expr, primal_locals, exact=False))
            return Tensor.leaf(self.eval_primal(expr, primal_locals))
        if isinstance(expr, MemberAccessIR):
            return Tensor.leaf(self.eval_primal(expr, primal_locals))
        if isinstance(expr, IdentifierIR):
            if expr.defid is None:
                raise IRAutodiffError(f"Unresolved identifier in custom diff body: {expr.name or '?'}")
            if expr.defid in primal_locals:
                return primal_locals[expr.defid]
            return self.binding_tensor(expr.defid)
        if isinstance(expr, (IndexVarIR, IndexRestIR)):
            if expr.defid is None or expr.defid not in primal_locals:
                raise IRAutodiffError(f"Missing index value in custom diff body: {getattr(expr, 'name', '?')}")
            return primal_locals[expr.defid]
        if isinstance(expr, BinaryOpIR):
            left = self._eval_custom_diff_tensor(expr.left, primal_locals, tangent_locals)
            right = self._eval_custom_diff_tensor(expr.right, primal_locals, tangent_locals)
            if expr.operator == BinaryOp.ADD:
                return binary_tensor(TensorOp.ADD, left, right, ir_node=expr)
            if expr.operator == BinaryOp.SUB:
                return binary_tensor(TensorOp.SUB, left, right, ir_node=expr)
            if expr.operator == BinaryOp.MUL:
                return binary_tensor(TensorOp.MUL, left, right, ir_node=expr)
            if expr.operator == BinaryOp.DIV:
                return binary_tensor(TensorOp.DIV, left, right, ir_node=expr)
            if expr.operator == BinaryOp.POW:
                return binary_tensor(TensorOp.POW, left, right, ir_node=expr)
            raise IRAutodiffError(f"Unsupported binary op in custom diff body: {expr.operator}")
        if isinstance(expr, UnaryOpIR):
            operand = self._eval_custom_diff_tensor(expr.operand, primal_locals, tangent_locals)
            if expr.operator == UnaryOp.NEG:
                return neg_tensor(operand, ir_node=expr)
            if expr.operator == UnaryOp.POS:
                return operand
            raise IRAutodiffError(f"Unsupported unary op in custom diff body: {expr.operator}")
        if isinstance(expr, RectangularAccessIR):
            array = self._eval_custom_diff_tensor(expr.array, primal_locals, tangent_locals)
            indices = tuple(int(np.asarray(self.eval_primal(idx, primal_locals)).reshape(-1)[0]) for idx in (expr.indices or []))
            return getitem_tensor(array, indices, ir_node=expr)
        if isinstance(expr, CastExpressionIR):
            return self._eval_custom_diff_tensor(expr.expr, primal_locals, tangent_locals)
        if isinstance(expr, BlockExpressionIR):
            child_primal = dict(primal_locals)
            child_tangent = dict(tangent_locals)
            for stmt in expr.statements or []:
                if isinstance(stmt, BindingIR) and stmt.defid is not None and stmt.expr is not None and not is_function_binding(stmt):
                    child_primal[stmt.defid] = self._eval_custom_diff_tensor(stmt.expr, child_primal, child_tangent)
                    if stmt.name:
                        child_primal[stmt.defid].named(stmt.name)
                elif isinstance(stmt, ExpressionIR):
                    self.eval_primal(stmt, child_primal)
            if expr.final_expr is None:
                raise IRAutodiffError("Custom diff block has no final expression")
            return self._eval_custom_diff_tensor(expr.final_expr, child_primal, child_tangent)
        if isinstance(expr, IfExpressionIR):
            cond = self.eval_primal(expr.condition, primal_locals)
            branch = expr.then_expr if bool(np.asarray(cond).all()) else expr.else_expr
            if branch is None:
                raise IRAutodiffError("Custom diff if-expression missing else branch")
            return self._eval_custom_diff_tensor(branch, primal_locals, tangent_locals)
        if isinstance(expr, ReductionExpressionIR):
            # Use primal-local bindings when evaluating custom diff reductions.
            return self._eval_reduction_tensor(expr, primal_locals)
        if isinstance(expr, EinsteinIR):
            return self._eval_einstein_tensor(expr, primal_locals)
        if isinstance(expr, FunctionCallIR):
            return Tensor.leaf(self.eval_primal(expr, primal_locals))
        raise IRAutodiffError(f"Unsupported IR node in custom diff body: {type(expr).__name__}")

    def _lookup_self_tensor_store(self, defid: Optional[DefId]) -> Optional[Dict[Tuple[int, ...], Tensor]]:
        if defid is None:
            return None
        for frame in reversed(self._self_tensor_store_stack):
            store = frame.get(defid)
            if store is not None:
                return store
        return None

    def eval_tensor(self, expr: ExpressionIR, locals_map: Dict[DefId, Tensor]) -> Tensor:
        if isinstance(expr, LiteralIR):
            return Tensor.leaf(expr.value)
        if isinstance(expr, ArrayLiteralIR):
            return Tensor.leaf(self.eval_primal(expr, locals_map))
        if isinstance(expr, BuiltinCallIR):
            if self._autodiff_request(expr) is not None:
                return Tensor.leaf(self._eval_autodiff_builtin(expr, locals_map, exact=True))
            return Tensor.leaf(self.eval_primal(expr, locals_map))
        if isinstance(expr, MemberAccessIR):
            return Tensor.leaf(self.eval_primal(expr, locals_map))
        if isinstance(expr, IdentifierIR):
            if expr.defid is None:
                raise IRAutodiffError(f"Unresolved identifier in autodiff graph: {expr.name or '?'}")
            if expr.defid in locals_map:
                return locals_map[expr.defid]
            return self.binding_tensor(expr.defid)
        if isinstance(expr, (IndexVarIR, IndexRestIR)):
            if expr.defid is None:
                raise IRAutodiffError(f"Unresolved index identifier in autodiff graph: {getattr(expr, 'name', '?')}")
            if expr.defid not in locals_map:
                raise IRAutodiffError(f"Missing loop/index value for autodiff graph: {getattr(expr, 'name', '?')}")
            return locals_map[expr.defid]
        if isinstance(expr, BinaryOpIR):
            left = self.eval_tensor(expr.left, locals_map)
            right = self.eval_tensor(expr.right, locals_map)
            if expr.operator == BinaryOp.ADD:
                return binary_tensor(TensorOp.ADD, left, right, ir_node=expr)
            if expr.operator == BinaryOp.SUB:
                return binary_tensor(TensorOp.SUB, left, right, ir_node=expr)
            if expr.operator == BinaryOp.MUL:
                return binary_tensor(TensorOp.MUL, left, right, ir_node=expr)
            if expr.operator == BinaryOp.DIV:
                return binary_tensor(TensorOp.DIV, left, right, ir_node=expr)
            if expr.operator == BinaryOp.POW:
                return binary_tensor(TensorOp.POW, left, right, ir_node=expr)
            raise IRAutodiffError(f"Unsupported binary op in native autodiff: {expr.operator}")
        if isinstance(expr, UnaryOpIR):
            operand = self.eval_tensor(expr.operand, locals_map)
            if expr.operator == UnaryOp.NEG:
                return neg_tensor(operand, ir_node=expr)
            if expr.operator == UnaryOp.POS:
                return operand
            raise IRAutodiffError(f"Unsupported unary op in native autodiff: {expr.operator}")
        if isinstance(expr, RectangularAccessIR):
            indices = tuple(int(np.asarray(self.eval_primal(idx, locals_map)).reshape(-1)[0]) for idx in (expr.indices or []))
            if isinstance(expr.array, IdentifierIR):
                store = self._lookup_self_tensor_store(expr.array.defid)
                if store is not None:
                    if indices not in store:
                        raise IRAutodiffError(
                            f"Self-referential Einstein access requested unavailable index {indices} "
                            f"for {expr.array.name or expr.array.defid}"
                        )
                    return store[indices]
            array = self.eval_tensor(expr.array, locals_map)
            return getitem_tensor(array, indices, ir_node=expr)
        if isinstance(expr, CastExpressionIR):
            return self.eval_tensor(expr.expr, locals_map)
        if isinstance(expr, BlockExpressionIR):
            child = dict(locals_map)
            for stmt in expr.statements or []:
                if isinstance(stmt, BindingIR) and stmt.defid is not None and stmt.expr is not None and not is_function_binding(stmt):
                    child[stmt.defid] = self._eval_binding_tensor(stmt, child)
                    if stmt.name:
                        child[stmt.defid].named(stmt.name)
                elif isinstance(stmt, ExpressionIR):
                    self.eval_primal(stmt, child)
            if expr.final_expr is None:
                raise IRAutodiffError("Block expression in autodiff graph has no final expression")
            return self.eval_tensor(expr.final_expr, child)
        if isinstance(expr, IfExpressionIR):
            cond = self.eval_primal(expr.condition, locals_map)
            if expr.else_expr is None:
                raise IRAutodiffError("If expression in autodiff graph missing else branch")
            cond_arr = np.asarray(cond)
            if cond_arr.ndim == 0:
                if bool(cond_arr):
                    return self.eval_tensor(expr.then_expr, locals_map)
                return self.eval_tensor(expr.else_expr, locals_map)
            then_t = self.eval_tensor(expr.then_expr, locals_map)
            else_t = self.eval_tensor(expr.else_expr, locals_map)
            return where_tensors(cond_arr, then_t, else_t, ir_node=expr)
        if isinstance(expr, ReductionExpressionIR):
            return self._eval_reduction_tensor(expr, locals_map)
        if isinstance(expr, EinsteinIR):
            return self._eval_einstein_tensor(expr, locals_map)
        if isinstance(expr, FunctionCallIR):
            return self._eval_function_call(expr, locals_map)
        raise IRAutodiffError(f"Unsupported IR node in native autodiff graph: {type(expr).__name__}")

    def _eval_reduction_tensor(self, expr: ReductionExpressionIR, locals_map: Dict[DefId, Tensor]) -> Tensor:
        if expr.operation not in (ReductionOp.SUM, ReductionOp.MAX, ReductionOp.MIN):
            raise IRAutodiffError(f"Unsupported reduction op in native autodiff: {expr.operation}")

        loop_vars = list(expr.loop_vars or [])
        values: list[Tensor] = []

        def walk(i: int, current_locals: Dict[DefId, Tensor]) -> None:
            if i >= len(loop_vars):
                if expr.where_clause is not None and expr.where_clause.constraints:
                    for constraint in expr.where_clause.constraints:
                        if not bool(np.asarray(self.eval_primal(constraint, current_locals)).all()):
                            return
                values.append(self.eval_tensor(expr.body, current_locals))
                return

            loop_var = loop_vars[i]
            did = getattr(loop_var, "defid", None)
            if did is None:
                raise IRAutodiffError("Reduction loop variable has no DefId")
            range_expr = None
            if did in (expr.loop_var_ranges or {}):
                range_expr = expr.loop_var_ranges[did]
            elif getattr(loop_var, "range_ir", None) is not None:
                range_expr = getattr(loop_var, "range_ir", None)
            if range_expr is None:
                raise IRAutodiffError(f"Reduction loop variable {getattr(loop_var, 'name', '?')} missing range")
            iter_range = self.eval_primal(range_expr, current_locals)
            for value in iter_range:
                child = dict(current_locals)
                child[did] = Tensor.leaf(np.array(value, dtype=np.float64), name=getattr(loop_var, "name", None))
                walk(i + 1, child)

        walk(0, dict(locals_map))

        if not values:
            return Tensor.leaf(0.0)
        if expr.operation == ReductionOp.SUM:
            acc = values[0]
            for item in values[1:]:
                acc = acc + item
            return acc

        best = values[0]
        best_primal = np.asarray(best.value)
        for item in values[1:]:
            primal = np.asarray(item.value)
            if expr.operation == ReductionOp.MAX:
                if bool((primal > best_primal).all()):
                    best = item
                    best_primal = primal
            else:
                if bool((primal < best_primal).all()):
                    best = item
                    best_primal = primal
        return best

    def _eval_einstein_tensor(
        self,
        expr: EinsteinIR,
        locals_map: Dict[DefId, Tensor],
        *,
        owner_defid: Optional[DefId] = None,
        owner_name: Optional[str] = None,
    ) -> Tensor:
        clauses = list(expr.clauses or [])
        if not clauses:
            raise IRAutodiffError("EinsteinIR has no clauses")

        storage: Dict[Tuple[int, ...], Tensor] = {}
        frame: Dict[DefId, Dict[Tuple[int, ...], Tensor]] = {}
        if owner_defid is not None:
            frame[owner_defid] = storage
        self._self_tensor_store_stack.append(frame)
        try:
            for clause in clauses:
                self._eval_einstein_clause_into_storage(clause, locals_map, storage)
        finally:
            self._self_tensor_store_stack.pop()

        shape = self._shape_for_einstein(expr, locals_map, storage)
        tensor = self._tensor_from_storage(shape, storage)
        if owner_name:
            tensor.named(owner_name)
        return tensor

    def _eval_einstein_clause_into_storage(
        self,
        clause: Any,
        locals_map: Dict[DefId, Tensor],
        storage: Dict[Tuple[int, ...], Tensor],
    ) -> None:
        indices = list(getattr(clause, "indices", ()) or ())
        variable_ranges = getattr(clause, "variable_ranges", None) or {}

        def populate(depth: int, current_locals: Dict[DefId, Tensor], current_index: List[int]) -> None:
            if depth >= len(indices):
                if getattr(clause, "where_clause", None) is not None:
                    for constraint in clause.where_clause.constraints or ():
                        if not bool(np.asarray(self.eval_primal(constraint, current_locals)).all()):
                            return
                value = self.eval_tensor(clause.value, current_locals)
                key = tuple(current_index)
                if key in storage:
                    storage[key] = storage[key] + value
                else:
                    storage[key] = value
                return

            idx = indices[depth]
            if isinstance(idx, LiteralIR):
                current_index.append(int(np.asarray(idx.value).reshape(-1)[0]))
                populate(depth + 1, current_locals, current_index)
                current_index.pop()
                return

            did = getattr(idx, "defid", None)
            range_expr = getattr(idx, "range_ir", None) or (variable_ranges.get(did) if did is not None else None)
            if range_expr is not None:
                iter_range = self.eval_primal(range_expr, current_locals)
                for value in iter_range:
                    child = dict(current_locals)
                    if did is not None:
                        child[did] = Tensor.leaf(np.array(value, dtype=np.float64), name=getattr(idx, "name", None))
                    current_index.append(int(value))
                    populate(depth + 1, child, current_index)
                    current_index.pop()
                return

            fixed_value = int(np.asarray(self.eval_primal(idx, current_locals)).reshape(-1)[0])
            if did is not None and did not in current_locals:
                current_locals = dict(current_locals)
                current_locals[did] = Tensor.leaf(np.array(fixed_value, dtype=np.float64), name=getattr(idx, "name", None))
            current_index.append(fixed_value)
            populate(depth + 1, current_locals, current_index)
            current_index.pop()

        populate(0, dict(locals_map), [])

    def _shape_for_einstein(
        self,
        expr: EinsteinIR,
        locals_map: Dict[DefId, Tensor],
        storage: Dict[Tuple[int, ...], Tensor],
    ) -> Tuple[int, ...]:
        if expr.shape:
            dims = []
            for dim in expr.shape:
                dims.append(int(np.asarray(self.eval_primal(dim, locals_map)).reshape(-1)[0]))
            return tuple(dims)
        if not storage:
            return ()
        rank = max((len(idx) for idx in storage.keys()), default=0)
        if rank == 0:
            return ()
        dims = [0] * rank
        for idx in storage.keys():
            for axis, value in enumerate(idx):
                dims[axis] = max(dims[axis], int(value) + 1)
        return tuple(dims)

    def _tensor_from_storage(
        self,
        shape: Tuple[int, ...],
        storage: Dict[Tuple[int, ...], Tensor],
        prefix: Tuple[int, ...] = (),
    ) -> Tensor:
        if len(prefix) >= len(shape):
            return storage.get(prefix, Tensor.leaf(0.0))
        axis = len(prefix)
        elems = [self._tensor_from_storage(shape, storage, prefix + (i,)) for i in range(shape[axis])]
        if not elems:
            return Tensor.leaf(0.0)
        return stack_tensors(elems, axis=0)

    def _eval_function_call(self, expr: FunctionCallIR, locals_map: Dict[DefId, Tensor]) -> Tensor:
        args = list(expr.arguments or [])
        callee_binding = self._functions.get(expr.function_defid) or self._bindings.get(expr.function_defid)
        custom = self._try_eval_custom_diff_call(callee_binding, expr, args, locals_map)
        if custom is not None:
            return custom
        if isinstance(callee_binding, BindingIR) and isinstance(callee_binding.expr, FunctionValueIR):
            fv = callee_binding.expr
            child_locals = dict(locals_map)
            for param, arg in zip(fv.parameters or [], args):
                if param.defid is not None:
                    child_locals[param.defid] = self.eval_tensor(arg, locals_map)
                    if param.name:
                        child_locals[param.defid].named(param.name)
            if fv.body is None:
                raise IRAutodiffError(f"Function {expr.function_name or expr.function_defid} has no body")
            return self.eval_tensor(fv.body, child_locals)

        raise IRAutodiffError(f"Unsupported function call in native autodiff: {expr.function_name or expr.function_defid}")

    def _try_eval_custom_diff_call(
        self,
        callee_binding: Any,
        expr: FunctionCallIR,
        args: List[ExpressionIR],
        locals_map: Dict[DefId, Tensor],
    ) -> Optional[Tensor]:
        if not (isinstance(callee_binding, BindingIR) and isinstance(callee_binding.expr, FunctionValueIR)):
            return None
        fv = callee_binding.expr
        if fv.custom_diff_body is None:
            return None

        arg_tensors = [self.eval_tensor(arg, locals_map) for arg in args]
        primal_locals = dict(locals_map)
        for param, arg_tensor in zip(fv.parameters or [], arg_tensors):
            if param.defid is not None:
                primal_locals[param.defid] = arg_tensor
                if param.name:
                    primal_locals[param.defid].named(param.name)

        if fv.body is None:
            raise IRAutodiffError(f"Custom-diff function {expr.function_name or expr.function_defid} has no body")
        primal_value = self.eval_primal(fv.body, primal_locals)
        call_text = f"{expr.function_name}({', '.join(primal_expr(t) for t in arg_tensors)})"

        def jvp_fn(inputs: Tuple[np.ndarray, ...], tangents: Tuple[np.ndarray, ...]) -> np.ndarray:
            local_primal = dict(primal_locals)
            tangent_locals: Dict[DefId, Tensor] = {}
            for param, primal_value_i, tangent_value_i in zip(fv.parameters or [], inputs, tangents):
                if param.defid is None:
                    continue
                local_primal[param.defid] = Tensor.leaf(primal_value_i, name=param.name)
                tangent_locals[param.defid] = Tensor.leaf(tangent_value_i, name=f"@{param.name}" if param.name else None)
            result = self._eval_custom_diff_tensor(fv.custom_diff_body, local_primal, tangent_locals)
            return np.asarray(result.value, dtype=np.float64)

        def vjp_fn(inputs: Tuple[np.ndarray, ...], cotangent: np.ndarray) -> Tuple[np.ndarray, ...]:
            out: List[np.ndarray] = []
            cotangent_arr = np.asarray(cotangent, dtype=np.float64)
            input_shapes = [np.asarray(v).shape for v in inputs]
            for i, shape in enumerate(input_shapes):
                tangents = []
                for j, inp in enumerate(inputs):
                    if i == j:
                        tangents.append(cotangent_arr)
                    else:
                        tangents.append(np.zeros_like(np.asarray(inp), dtype=np.float64))
                contrib = jvp_fn(inputs, tuple(tangents))
                out.append(_sum_to_shape(contrib, tuple(shape)))
            return tuple(out)

        def symbolic_fn(parents: Sequence[Tensor], tangents: Sequence[SymbolicExpr], primal_cache: Dict[int, str]) -> SymbolicExpr:
            local_primal = dict(primal_locals)
            tangent_locals: Dict[DefId, Tensor] = {}
            for param, parent in zip(fv.parameters or [], parents):
                if param.defid is not None:
                    local_primal[param.defid] = parent
                    tangent_locals[param.defid] = Tensor.leaf(0.0, name=f"@{param.name}" if param.name else None)
            expr_tensor = self._eval_custom_diff_tensor(fv.custom_diff_body, local_primal, tangent_locals)
            # Rebuild symbolically from the translated Tensor graph so custom rules compose.
            return symbolic_tangent_expr(expr_tensor, include_named_leaves=True, _primal_cache=primal_cache)

        return custom_diff_call(
            primal_value,
            arg_tensors,
            call_text=call_text,
            jvp_fn=jvp_fn,
            vjp_fn=vjp_fn,
            symbolic_fn=symbolic_fn,
            ir_node=expr,
        )

    def eval_primal(self, expr: ExpressionIR, locals_map: Dict[DefId, Tensor]) -> Any:
        if isinstance(expr, LiteralIR):
            return expr.value
        if isinstance(expr, IdentifierIR):
            if expr.defid is None:
                raise IRAutodiffError(f"Unresolved identifier in autodiff primal eval: {expr.name or '?'}")
            if expr.defid in locals_map:
                return locals_map[expr.defid].value
            value = self._value_lookup(expr.defid)
            if value is None:
                tensor = self.binding_tensor(expr.defid)
                return tensor.value
            return value
        if isinstance(expr, (IndexVarIR, IndexRestIR)):
            if expr.defid is None:
                raise IRAutodiffError(f"Unresolved index identifier in autodiff primal eval: {getattr(expr, 'name', '?')}")
            if expr.defid not in locals_map:
                raise IRAutodiffError(f"Missing loop/index value in autodiff primal eval: {getattr(expr, 'name', '?')}")
            return locals_map[expr.defid].value
        if isinstance(expr, ArrayLiteralIR):
            return np.asarray([self.eval_primal(elem, locals_map) for elem in (expr.elements or [])], dtype=np.float64)
        if isinstance(expr, BuiltinCallIR):
            if self._autodiff_request(expr) is not None:
                return self._eval_autodiff_builtin(expr, locals_map, exact=True)
            args = [self.eval_primal(arg, locals_map) for arg in (expr.args or [])]
            builtin_defid = _builtin_call_defid(expr)
            evaluator = _PRIMAL_BUILTIN_EVALUATORS.get(builtin_defid)
            if evaluator is None:
                raise IRAutodiffError(f"Unsupported builtin in native autodiff primal eval: {expr.builtin_name}")
            return evaluator(args)
        if isinstance(expr, MemberAccessIR):
            obj = self.eval_primal(expr.object, locals_map)
            evaluator = _PRIMAL_MEMBER_ACCESSORS.get(expr.member)
            if evaluator is not None:
                return evaluator(obj)
            raise IRAutodiffError(f"Unsupported member access in native autodiff primal eval: {expr.member}")
        if isinstance(expr, RectangularAccessIR):
            base = np.asarray(self.eval_primal(expr.array, locals_map))
            indices = tuple(int(np.asarray(self.eval_primal(idx, locals_map)).reshape(-1)[0]) for idx in (expr.indices or []))
            return base[indices]
        if isinstance(expr, CastExpressionIR):
            return self.eval_primal(expr.expr, locals_map)
        if isinstance(expr, UnaryOpIR):
            value = self.eval_primal(expr.operand, locals_map)
            if expr.operator == UnaryOp.NEG:
                return -np.asarray(value)
            if expr.operator == UnaryOp.POS:
                return np.asarray(value)
        if isinstance(expr, BinaryOpIR):
            left = self.eval_primal(expr.left, locals_map)
            right = self.eval_primal(expr.right, locals_map)
            if expr.operator == BinaryOp.ADD:
                return np.asarray(left) + np.asarray(right)
            if expr.operator == BinaryOp.SUB:
                return np.asarray(left) - np.asarray(right)
            if expr.operator == BinaryOp.MUL:
                return np.asarray(left) * np.asarray(right)
            if expr.operator == BinaryOp.DIV:
                return np.asarray(left) / np.asarray(right)
            if expr.operator == BinaryOp.POW:
                return np.asarray(left) ** np.asarray(right)
            if expr.operator == BinaryOp.GT:
                return np.asarray(left) > np.asarray(right)
            if expr.operator == BinaryOp.GE:
                return np.asarray(left) >= np.asarray(right)
            if expr.operator == BinaryOp.LT:
                return np.asarray(left) < np.asarray(right)
            if expr.operator == BinaryOp.LE:
                return np.asarray(left) <= np.asarray(right)
            if expr.operator == BinaryOp.EQ:
                return np.asarray(left) == np.asarray(right)
            if expr.operator == BinaryOp.NE:
                return np.asarray(left) != np.asarray(right)
            if expr.operator == BinaryOp.AND:
                return np.asarray(left).astype(bool) & np.asarray(right).astype(bool)
            if expr.operator == BinaryOp.OR:
                return np.asarray(left).astype(bool) | np.asarray(right).astype(bool)
            if expr.operator == BinaryOp.MOD:
                return np.asarray(left) % np.asarray(right)
        if isinstance(expr, BlockExpressionIR):
            child = dict(locals_map)
            for stmt in expr.statements or []:
                if isinstance(stmt, BindingIR) and stmt.defid is not None and stmt.expr is not None and not is_function_binding(stmt):
                    child[stmt.defid] = Tensor.leaf(self.eval_primal(stmt.expr, child), name=stmt.name)
            if expr.final_expr is None:
                raise IRAutodiffError("Block expression in autodiff primal eval has no final expression")
            return self.eval_primal(expr.final_expr, child)
        if isinstance(expr, IfExpressionIR):
            cond = self.eval_primal(expr.condition, locals_map)
            if bool(np.asarray(cond).all()):
                return self.eval_primal(expr.then_expr, locals_map)
            if expr.else_expr is None:
                raise IRAutodiffError("If expression in autodiff primal eval missing else branch")
            return self.eval_primal(expr.else_expr, locals_map)
        if isinstance(expr, RangeIR):
            start = self.eval_primal(expr.start, locals_map)
            end = self.eval_primal(expr.end, locals_map)
            end_int = int(np.asarray(end).reshape(-1)[0])
            if expr.inclusive:
                end_int += 1
            return range(int(np.asarray(start).reshape(-1)[0]), end_int)
        if isinstance(expr, ReductionExpressionIR):
            return self._eval_reduction_tensor(expr, locals_map).value
        if isinstance(expr, EinsteinIR):
            return self._eval_einstein_tensor(expr, locals_map).value
        if isinstance(expr, FunctionCallIR):
            args = [self.eval_primal(arg, locals_map) for arg in (expr.arguments or [])]
            module_path = tuple(getattr(expr, "module_path", ()) or ())
            module = _PYTHON_PRIMAL_MODULES.get(module_path[:2])
            if module is not None:
                fn_name = expr.function_name or ""
                fn = getattr(module, fn_name, None)
                if fn is None:
                    raise IRAutodiffError(f"Unsupported python primal call: {fn_name}")
                return fn(*args)
            callee_binding = self._functions.get(expr.function_defid) or self._bindings.get(expr.function_defid)
            if isinstance(callee_binding, BindingIR) and isinstance(callee_binding.expr, FunctionValueIR):
                fv = callee_binding.expr
                child = dict(locals_map)
                for param, arg_value in zip(fv.parameters or [], args):
                    if param.defid is not None:
                        child[param.defid] = Tensor.leaf(arg_value, name=param.name)
                if fv.body is None:
                    raise IRAutodiffError(f"Function {expr.function_name or expr.function_defid} has no body")
                return self.eval_primal(fv.body, child)
        raise IRAutodiffError(f"Unsupported primal eval node in native autodiff: {type(expr).__name__}")


def tangent_value_for_defid(
    target_defid: DefId,
    compiled_facts: AutodiffCompiledFacts,
    value_lookup: Callable[[DefId], Any],
) -> Any:
    del compiled_facts
    target = value_lookup(target_defid)
    if target is None:
        raise IRAutodiffError(f"Missing primal value for autodiff tangent target {target_defid}")
    return _identity_seed(target)


def jacobian_value_for_defids(
    numerator_defid: DefId,
    denominator_defid: DefId,
    compiled_facts: AutodiffCompiledFacts,
    value_lookup: Callable[[DefId], Any],
) -> Any:
    runtime = NativeIRAutodiffRuntime(compiled_facts, value_lookup)
    numerator = runtime.binding_tensor(numerator_defid)
    denominator = runtime.binding_tensor(denominator_defid)
    lazy = jacobian(numerator, denominator)
    if numerator.size == 1 and denominator.size == 1:
        scalar = np.asarray(lazy).reshape(-1)[0]
        return scalar.item() if hasattr(scalar, "item") else scalar
    return lazy


def symbolic_tangent_for_defid(
    target_defid: DefId,
    compiled_facts: AutodiffCompiledFacts,
    value_lookup: Callable[[DefId], Any],
) -> str:
    runtime = NativeIRAutodiffRuntime(compiled_facts, value_lookup)
    target = runtime.binding_tensor(target_defid)
    if target.name is None:
        binding = binding_for_defid(compiled_facts, target_defid)
        if isinstance(binding, BindingIR) and binding.name:
            target.named(binding.name)
    return symbolic_tangent_program(target)


def symbolic_jacobian_relation(
    numerator_defid: DefId,
    denominator_defid: DefId,
    compiled_facts: AutodiffCompiledFacts,
    value_lookup: Callable[[DefId], Any],
) -> str:
    del value_lookup
    num_binding = binding_for_defid(compiled_facts, numerator_defid)
    den_binding = binding_for_defid(compiled_facts, denominator_defid)
    num_name = getattr(num_binding, "name", None) or "y"
    den_name = getattr(den_binding, "name", None) or "x"
    return f"(@{num_name} / @{den_name}) · @{den_name}"


def _assert_allclose(actual: ArrayLike, expected: ArrayLike, *, atol: float = 1e-8, rtol: float = 1e-8) -> None:
    a = _as_array(actual)
    e = _as_array(expected)
    if not np.allclose(a, e, atol=atol, rtol=rtol):
        raise AssertionError(f"not close\nactual={a}\nexpected={e}")
