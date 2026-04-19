"""Standalone NumPy AD prototype: JVP, VJP, and lazy Jacobians.

This file is intentionally outside Einlang proper. It is a small executable
reference for the rewrite in ``docs/AUTODIFF.md``:

- one traced tensor graph
- one primitive rule registry
- ``jvp`` for forward-mode linearization
- ``vjp`` for reverse-mode pullback
- ``LazyJacobianTensor`` for ``@y / @x``-style Jacobian values

Run:

    python3 examples/numpy_jvp_vjp_lazy_jacobian.py
"""

from __future__ import annotations

from dataclasses import dataclass
import itertools
from typing import Any, Callable, Dict, Optional, Sequence, Tuple

import numpy as np


ArrayLike = Any
Index = Any


def _as_array(value: ArrayLike) -> np.ndarray:
    return np.asarray(value, dtype=np.float64)


def _zeros_like(value: np.ndarray) -> np.ndarray:
    return np.zeros_like(np.asarray(value, dtype=np.float64))


def _ones_like(value: np.ndarray) -> np.ndarray:
    return np.ones_like(np.asarray(value, dtype=np.float64))


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
        op: Optional[str] = None,
        meta: Optional[Dict[str, Any]] = None,
        name: Optional[str] = None,
    ) -> None:
        self.value = _as_array(value)
        self.parents = tuple(parents)
        self.op = op
        self.meta = dict(meta or {})
        self.name = name

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
        label = self.name if self.name is not None else (self.op or "leaf")
        return f"Tensor(name={label!r}, shape={self.shape}, value={self.value!r})"

    @staticmethod
    def leaf(value: ArrayLike, name: Optional[str] = None) -> "Tensor":
        return Tensor(value, name=name)

    def _binary(self, op: str, other: ArrayLike) -> "Tensor":
        rhs = ensure_tensor(other)
        value = _BINARY_IMPL[op](self.value, rhs.value)
        return Tensor(value, parents=(self, rhs), op=op)

    def __add__(self, other: ArrayLike) -> "Tensor":
        return self._binary("add", other)

    def __radd__(self, other: ArrayLike) -> "Tensor":
        return ensure_tensor(other)._binary("add", self)

    def __sub__(self, other: ArrayLike) -> "Tensor":
        return self._binary("sub", other)

    def __rsub__(self, other: ArrayLike) -> "Tensor":
        return ensure_tensor(other)._binary("sub", self)

    def __mul__(self, other: ArrayLike) -> "Tensor":
        return self._binary("mul", other)

    def __rmul__(self, other: ArrayLike) -> "Tensor":
        return ensure_tensor(other)._binary("mul", self)

    def __truediv__(self, other: ArrayLike) -> "Tensor":
        return self._binary("div", other)

    def __rtruediv__(self, other: ArrayLike) -> "Tensor":
        return ensure_tensor(other)._binary("div", self)

    def __pow__(self, other: ArrayLike) -> "Tensor":
        return self._binary("pow", other)

    def __rpow__(self, other: ArrayLike) -> "Tensor":
        return ensure_tensor(other)._binary("pow", self)

    def __neg__(self) -> "Tensor":
        return Tensor(-self.value, parents=(self,), op="neg")

    def exp(self) -> "Tensor":
        return Tensor(np.exp(self.value), parents=(self,), op="exp")

    def log(self) -> "Tensor":
        return Tensor(np.log(self.value), parents=(self,), op="log")

    def sin(self) -> "Tensor":
        return Tensor(np.sin(self.value), parents=(self,), op="sin")

    def cos(self) -> "Tensor":
        return Tensor(np.cos(self.value), parents=(self,), op="cos")

    def relu(self) -> "Tensor":
        return Tensor(np.maximum(self.value, 0.0), parents=(self,), op="relu")

    def sum(
        self,
        axis: Optional[Sequence[int] | int] = None,
        keepdims: bool = False,
    ) -> "Tensor":
        meta = {"axis": _normalize_axis(axis, self.ndim), "keepdims": bool(keepdims)}
        return Tensor(
            self.value.sum(axis=axis, keepdims=keepdims),
            parents=(self,),
            op="sum",
            meta=meta,
        )

    def reshape(self, *shape: int) -> "Tensor":
        if len(shape) == 1 and isinstance(shape[0], tuple):
            target = tuple(int(x) for x in shape[0])
        else:
            target = tuple(int(x) for x in shape)
        return Tensor(self.value.reshape(target), parents=(self,), op="reshape", meta={"shape": target})

    def __getitem__(self, index: Index) -> "Tensor":
        return Tensor(self.value[index], parents=(self,), op="getitem", meta={"index": index})

    def named(self, name: str) -> "Tensor":
        self.name = name
        return self


def ensure_tensor(value: ArrayLike) -> Tensor:
    if isinstance(value, Tensor):
        return value
    return Tensor.leaf(value)


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
    elif node.op in ("add", "sub", "mul", "div", "pow"):
        left = primal_expr(node.parents[0], cache)
        right = primal_expr(node.parents[1], cache)
        symbol = {"add": "+", "sub": "-", "mul": "*", "div": "/", "pow": "**"}[node.op]
        text = f"({left} {symbol} {right})"
    elif node.op == "neg":
        text = f"(-{primal_expr(node.parents[0], cache)})"
    elif node.op in ("exp", "log", "sin", "cos", "relu"):
        text = f"{node.op}({primal_expr(node.parents[0], cache)})"
    elif node.op == "sum":
        text = (
            f"sum({primal_expr(node.parents[0], cache)}, "
            f"axis={node.meta.get('axis')}, keepdims={node.meta.get('keepdims', False)})"
        )
    elif node.op == "reshape":
        text = f"reshape({primal_expr(node.parents[0], cache)}, {node.meta['shape']})"
    elif node.op == "getitem":
        text = f"{primal_expr(node.parents[0], cache)}[{node.meta['index']!r}]"
    elif node.op == "conv":
        x, w, b = node.parents
        text = (
            f"conv({primal_expr(x, cache)}, {primal_expr(w, cache)}, {primal_expr(b, cache)}, "
            f"{_format_meta(node.meta, ('stride', 'pad_begin', 'pad_end', 'dilation', 'group'))})"
        )
    elif node.op == "max_pool":
        text = (
            f"max_pool({primal_expr(node.parents[0], cache)}, "
            f"{_format_meta(node.meta, ('kernel_shape', 'strides', 'pads'))})"
        )
    elif node.op == "avg_pool":
        text = (
            f"avg_pool({primal_expr(node.parents[0], cache)}, "
            f"{_format_meta(node.meta, ('kernel_shape', 'strides', 'pads'))})"
        )
    else:
        text = node.name if node.name is not None else f"<{node.op}>"
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

    if node.op == "add":
        result = _sym_add(tangents[0], tangents[1])
    elif node.op == "sub":
        result = _sym_sub(tangents[0], tangents[1])
    elif node.op == "mul":
        result = _sym_add(
            _SYM_ZERO if tangents[0].is_zero else _sym(f"({tangent_refs[0]}) * ({parent_primal[1]})"),
            _SYM_ZERO if tangents[1].is_zero else _sym(f"({tangent_refs[1]}) * ({parent_primal[0]})"),
        )
    elif node.op == "div":
        if tangents[0].is_zero and tangents[1].is_zero:
            result = _SYM_ZERO
        else:
            result = _sym(
                f"(({parent_primal[1]}) * ({tangent_refs[0]}) - ({parent_primal[0]}) * ({tangent_refs[1]})) / "
                f"(({parent_primal[1]}) ** 2.0)"
            )
    elif node.op == "pow":
        if tangents[0].is_zero and tangents[1].is_zero:
            result = _SYM_ZERO
        else:
            result = _sym(
                f"({primal_expr(node, primal_cache)}) * "
                f"(({tangent_refs[1]}) * log({parent_primal[0]}) + ({parent_primal[1]}) * ({tangent_refs[0]}) / ({parent_primal[0]}))"
            )
    elif node.op == "neg":
        result = _SYM_ZERO if tangents[0].is_zero else _sym(f"-({tangent_refs[0]})")
    elif node.op == "exp":
        result = _SYM_ZERO if tangents[0].is_zero else _sym(f"exp({parent_primal[0]}) * ({tangent_refs[0]})")
    elif node.op == "log":
        result = _SYM_ZERO if tangents[0].is_zero else _sym(f"({tangent_refs[0]}) / ({parent_primal[0]})")
    elif node.op == "sin":
        result = _SYM_ZERO if tangents[0].is_zero else _sym(f"cos({parent_primal[0]}) * ({tangent_refs[0]})")
    elif node.op == "cos":
        result = _SYM_ZERO if tangents[0].is_zero else _sym(f"-sin({parent_primal[0]}) * ({tangent_refs[0]})")
    elif node.op == "relu":
        result = _SYM_ZERO if tangents[0].is_zero else _sym(f"if {parent_primal[0]} > 0.0 {{ {tangent_refs[0]} }} else {{ 0.0 }}")
    elif node.op == "sum":
        result = _SYM_ZERO if tangents[0].is_zero else _sym(
            f"sum({tangent_refs[0]}, axis={node.meta.get('axis')}, keepdims={node.meta.get('keepdims', False)})"
        )
    elif node.op == "reshape":
        result = _SYM_ZERO if tangents[0].is_zero else _sym(f"reshape({tangent_refs[0]}, {node.meta['shape']})")
    elif node.op == "getitem":
        result = _SYM_ZERO if tangents[0].is_zero else _sym(f"({tangent_refs[0]})[{node.meta['index']!r}]")
    elif node.op == "conv":
        dx, dw, db = tangents
        terms = []
        if not dx.is_zero:
            terms.append(
                "conv("
                f"{tangent_refs[0]}, {parent_primal[1]}, 0.0, "
                f"{_format_meta(node.meta, ('stride', 'pad_begin', 'pad_end', 'dilation', 'group'))})"
            )
        if not dw.is_zero or not db.is_zero:
            terms.append(
                "conv("
                f"{parent_primal[0]}, {tangent_refs[1] if not dw.is_zero else '0.0'}, {tangent_refs[2] if not db.is_zero else '0.0'}, "
                f"{_format_meta(node.meta, ('stride', 'pad_begin', 'pad_end', 'dilation', 'group'))})"
            )
        result = _SYM_ZERO if not terms else _sym(" + ".join(terms))
    elif node.op == "max_pool":
        result = _SYM_ZERO if tangents[0].is_zero else _sym(
            f"select_at_argmax({parent_primal[0]}, {tangent_refs[0]}, "
            f"{_format_meta(node.meta, ('kernel_shape', 'strides', 'pads'))})"
        )
    elif node.op == "avg_pool":
        result = _SYM_ZERO if tangents[0].is_zero else _sym(
            f"avg_pool({tangent_refs[0]}, {_format_meta(node.meta, ('kernel_shape', 'strides', 'pads'))})"
        )
    else:
        result = _sym(f"@{node.name or node.op}")

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


def _pad_spatial(
    x: np.ndarray,
    pad_begin: Sequence[int],
    pad_end: Sequence[int],
) -> np.ndarray:
    rank = len(pad_begin)
    if len(pad_end) != rank:
        raise ValueError("pad_begin and pad_end must have the same rank")
    return np.pad(
        _as_array(x),
        [(0, 0), (0, 0)] + [(int(pad_begin[d]), int(pad_end[d])) for d in range(rank)],
        mode="constant",
        constant_values=0.0,
    )


def _conv_output_spatial_shape(
    padded_spatial: Sequence[int],
    kernel: Sequence[int],
    stride: Sequence[int],
    dilation: Sequence[int],
) -> Tuple[int, ...]:
    out = []
    for d in range(len(padded_spatial)):
        val = (int(padded_spatial[d]) - int(dilation[d]) * (int(kernel[d]) - 1) - 1) // int(stride[d]) + 1
        out.append(int(val))
    return tuple(out)


def _tensor_vjp(
    dy: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    out_index: np.ndarray,
    u_index: np.ndarray,
    v_index: np.ndarray,
    *,
    accumulate_dv: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    dyf = _as_array(dy).reshape(-1)
    uf = _as_array(u).reshape(-1)
    vf = _as_array(v).reshape(-1)
    oi = np.asarray(out_index, dtype=np.int64).reshape(-1)
    ui = np.asarray(u_index, dtype=np.int64).reshape(-1)
    vi = np.asarray(v_index, dtype=np.int64).reshape(-1)
    if not (oi.size == ui.size == vi.size):
        raise ValueError("index arrays must have matching lengths")
    du = np.zeros_like(uf)
    dv = np.zeros_like(vf)
    if oi.size:
        np.add.at(du, ui, dyf[oi] * vf[vi])
        if accumulate_dv:
            np.add.at(dv, vi, dyf[oi] * uf[ui])
    return du.reshape(u.shape), dv.reshape(v.shape)


def conv_forward(
    x: np.ndarray,
    w: np.ndarray,
    b: np.ndarray,
    stride: Sequence[int],
    pad_begin: Sequence[int],
    pad_end: Sequence[int],
    dilation: Sequence[int],
    group: int = 1,
) -> np.ndarray:
    x = _as_array(x)
    w = _as_array(w)
    b = _as_array(b)
    rank = x.ndim - 2
    if w.ndim != 2 + rank:
        raise ValueError("weight rank must match input spatial rank")
    n, c_in = x.shape[0], x.shape[1]
    c_out, cpg = w.shape[0], w.shape[1]
    if c_in % group != 0 or c_out % group != 0:
        raise ValueError("group must divide input and output channels")
    if cpg != c_in // group:
        raise ValueError("weight second dimension must be C_in / group")
    fpg = c_out // group
    kernel = list(w.shape[2:])
    x_p = _pad_spatial(x, pad_begin, pad_end)
    padded_spatial = [int(x_p.shape[2 + d]) for d in range(rank)]
    out_sp = _conv_output_spatial_shape(padded_spatial, kernel, stride, dilation)
    y = np.zeros((n, c_out) + out_sp, dtype=np.float64)

    for b_ in range(n):
        for co in range(c_out):
            g = co // fpg
            base_cl = g * cpg
            bias = float(b[co])
            for rest_out in itertools.product(*[range(o) for o in out_sp]):
                acc = bias
                for cl_off in range(cpg):
                    cl = base_cl + cl_off
                    for k_rest in itertools.product(*[range(kernel[d]) for d in range(rank)]):
                        pos = []
                        ok = True
                        for d in range(rank):
                            p = int(rest_out[d]) * int(stride[d]) + int(k_rest[d]) * int(dilation[d])
                            if not (0 <= p < padded_spatial[d]):
                                ok = False
                                break
                            pos.append(p)
                        if ok:
                            acc += x_p[(b_, cl) + tuple(pos)] * w[(co, cl_off) + tuple(k_rest)]
                y[(b_, co) + rest_out] = acc
    return y


def conv_vjp(
    x: np.ndarray,
    w: np.ndarray,
    dy: np.ndarray,
    stride: Sequence[int],
    pad_begin: Sequence[int],
    pad_end: Sequence[int],
    dilation: Sequence[int],
    group: int = 1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = _as_array(x)
    w = _as_array(w)
    dy = _as_array(dy)
    rank = x.ndim - 2
    n, c_in = x.shape[0], x.shape[1]
    c_out, cpg = w.shape[0], w.shape[1]
    if c_in % group != 0 or c_out % group != 0 or cpg != c_in // group:
        raise ValueError("invalid grouped convolution shapes")
    fpg = c_out // group
    kernel = list(w.shape[2:])
    x_p = _pad_spatial(x, pad_begin, pad_end)
    padded_spatial = [int(x_p.shape[2 + d]) for d in range(rank)]
    out_sp = _conv_output_spatial_shape(padded_spatial, kernel, stride, dilation)
    if dy.shape != (n, c_out) + out_sp:
        raise ValueError("dy shape mismatch for conv_vjp")

    o_list: list[int] = []
    iu_list: list[int] = []
    iv_list: list[int] = []
    db = np.zeros(c_out, dtype=np.float64)

    for b_ in range(n):
        for co in range(c_out):
            g = co // fpg
            base_cl = g * cpg
            for rest_out in itertools.product(*[range(o) for o in out_sp]):
                db[co] += float(dy[(b_, co) + rest_out])
                o_flat = int(np.ravel_multi_index((b_, co) + rest_out, dy.shape))
                for cl_off in range(cpg):
                    cl = base_cl + cl_off
                    for k_rest in itertools.product(*[range(kernel[d]) for d in range(rank)]):
                        pos = []
                        ok = True
                        for d in range(rank):
                            p = int(rest_out[d]) * int(stride[d]) + int(k_rest[d]) * int(dilation[d])
                            if not (0 <= p < padded_spatial[d]):
                                ok = False
                                break
                            pos.append(p)
                        if ok:
                            o_list.append(o_flat)
                            iu_list.append(int(np.ravel_multi_index((b_, cl) + tuple(pos), x_p.shape)))
                            iv_list.append(int(np.ravel_multi_index((co, cl_off) + tuple(k_rest), w.shape)))

    dx_p, dw = _tensor_vjp(
        dy,
        x_p,
        w,
        np.asarray(o_list, dtype=np.int64),
        np.asarray(iu_list, dtype=np.int64),
        np.asarray(iv_list, dtype=np.int64),
    )
    slices = [slice(None), slice(None)]
    for d in range(rank):
        pb = int(pad_begin[d])
        lim = int(x.shape[2 + d])
        slices.append(slice(pb, pb + lim))
    dx = np.array(dx_p[tuple(slices)], dtype=np.float64, copy=True)
    return dx, dw, db


def _pool_output_spatial_shape(
    in_spatial: Sequence[int],
    kernel_shape: Sequence[int],
    strides: Sequence[int],
    pads: Sequence[int],
) -> Tuple[int, ...]:
    out = []
    for d in range(len(in_spatial)):
        kd = int(kernel_shape[d])
        sd = int(strides[d])
        pd = int(pads[d])
        i = 0
        while True:
            hit = False
            for m in range(kd):
                t = i * sd - pd + m
                if 0 <= t < int(in_spatial[d]):
                    hit = True
                    break
            if not hit:
                break
            i += 1
        out.append(i)
    return tuple(out)


def _pool_window_value_max(
    x: np.ndarray,
    b_: int,
    ch: int,
    rest_out: Tuple[int, ...],
    in_sp: Sequence[int],
    ks: Sequence[int],
    st: Sequence[int],
    pd: Sequence[int],
) -> Tuple[float, Optional[Tuple[int, ...]]]:
    rank = len(ks)
    best = -np.inf
    winner: Optional[Tuple[int, ...]] = None
    for k_rest in itertools.product(*[range(int(ks[d])) for d in range(rank)]):
        pos = []
        ok = True
        for d in range(rank):
            t = int(rest_out[d]) * int(st[d]) - int(pd[d]) + int(k_rest[d])
            if not (0 <= t < int(in_sp[d])):
                ok = False
                break
            pos.append(t)
        if ok:
            value = float(x[(b_, ch) + tuple(pos)])
        else:
            value = -np.inf
        if value > best:
            best = value
            winner = tuple(pos) if ok else None
    return float(best), winner


def max_pool_forward(
    x: np.ndarray,
    kernel_shape: Sequence[int],
    strides: Sequence[int],
    pads: Sequence[int],
) -> np.ndarray:
    x = _as_array(x)
    rank = x.ndim - 2
    n, c = x.shape[0], x.shape[1]
    in_sp = [int(x.shape[2 + d]) for d in range(rank)]
    ks = [int(kernel_shape[d]) for d in range(rank)]
    st = [int(strides[d]) for d in range(rank)]
    pd = [int(pads[d]) for d in range(rank)]
    out_sp = _pool_output_spatial_shape(in_sp, ks, st, pd)
    y = np.empty((n, c) + out_sp, dtype=np.float64)
    for b_ in range(n):
        for ch in range(c):
            for rest_out in itertools.product(*[range(o) for o in out_sp]):
                value, _ = _pool_window_value_max(x, b_, ch, rest_out, in_sp, ks, st, pd)
                y[(b_, ch) + rest_out] = value
    return y


def max_pool_jvp(
    x: np.ndarray,
    dx: np.ndarray,
    kernel_shape: Sequence[int],
    strides: Sequence[int],
    pads: Sequence[int],
) -> np.ndarray:
    x = _as_array(x)
    dx = _as_array(dx)
    rank = x.ndim - 2
    n, c = x.shape[0], x.shape[1]
    in_sp = [int(x.shape[2 + d]) for d in range(rank)]
    ks = [int(kernel_shape[d]) for d in range(rank)]
    st = [int(strides[d]) for d in range(rank)]
    pd = [int(pads[d]) for d in range(rank)]
    out_sp = _pool_output_spatial_shape(in_sp, ks, st, pd)
    dy = np.zeros((n, c) + out_sp, dtype=np.float64)
    for b_ in range(n):
        for ch in range(c):
            for rest_out in itertools.product(*[range(o) for o in out_sp]):
                _, winner = _pool_window_value_max(x, b_, ch, rest_out, in_sp, ks, st, pd)
                if winner is not None:
                    dy[(b_, ch) + rest_out] = dx[(b_, ch) + winner]
    return dy


def max_pool_vjp(
    x: np.ndarray,
    dy: np.ndarray,
    kernel_shape: Sequence[int],
    strides: Sequence[int],
    pads: Sequence[int],
) -> np.ndarray:
    x = _as_array(x)
    dy = _as_array(dy)
    rank = x.ndim - 2
    n, c = x.shape[0], x.shape[1]
    in_sp = [int(x.shape[2 + d]) for d in range(rank)]
    ks = [int(kernel_shape[d]) for d in range(rank)]
    st = [int(strides[d]) for d in range(rank)]
    pd = [int(pads[d]) for d in range(rank)]
    out_sp = _pool_output_spatial_shape(in_sp, ks, st, pd)
    if dy.shape != (n, c) + out_sp:
        raise ValueError("dy shape mismatch for max_pool_vjp")

    o_list: list[int] = []
    iu_list: list[int] = []
    for b_ in range(n):
        for ch in range(c):
            for rest_out in itertools.product(*[range(o) for o in out_sp]):
                _, winner = _pool_window_value_max(x, b_, ch, rest_out, in_sp, ks, st, pd)
                if winner is None:
                    continue
                o_list.append(int(np.ravel_multi_index((b_, ch) + rest_out, dy.shape)))
                iu_list.append(int(np.ravel_multi_index((b_, ch) + winner, x.shape)))

    vi0 = np.zeros(len(o_list), dtype=np.int64)
    du, _ = _tensor_vjp(
        dy,
        x,
        np.array([1.0], dtype=np.float64),
        np.asarray(o_list, dtype=np.int64),
        np.asarray(iu_list, dtype=np.int64),
        vi0,
        accumulate_dv=False,
    )
    return du


def average_pool_forward(
    x: np.ndarray,
    kernel_shape: Sequence[int],
    strides: Sequence[int],
    pads: Sequence[int],
) -> np.ndarray:
    x = _as_array(x)
    rank = x.ndim - 2
    n, c = x.shape[0], x.shape[1]
    in_sp = [int(x.shape[2 + d]) for d in range(rank)]
    ks = [int(kernel_shape[d]) for d in range(rank)]
    st = [int(strides[d]) for d in range(rank)]
    pd = [int(pads[d]) for d in range(rank)]
    out_sp = _pool_output_spatial_shape(in_sp, ks, st, pd)
    vol = float(np.prod(ks))
    y = np.zeros((n, c) + out_sp, dtype=np.float64)
    for b_ in range(n):
        for ch in range(c):
            for rest_out in itertools.product(*[range(o) for o in out_sp]):
                acc = 0.0
                for k_rest in itertools.product(*[range(ks[d]) for d in range(rank)]):
                    pos = []
                    ok = True
                    for d in range(rank):
                        t = int(rest_out[d]) * int(st[d]) - int(pd[d]) + int(k_rest[d])
                        if not (0 <= t < int(in_sp[d])):
                            ok = False
                            break
                        pos.append(t)
                    if ok:
                        acc += float(x[(b_, ch) + tuple(pos)])
                y[(b_, ch) + rest_out] = acc / vol
    return y


def average_pool_vjp(
    x: np.ndarray,
    dy: np.ndarray,
    kernel_shape: Sequence[int],
    strides: Sequence[int],
    pads: Sequence[int],
) -> np.ndarray:
    x = _as_array(x)
    dy = _as_array(dy)
    rank = x.ndim - 2
    n, c = x.shape[0], x.shape[1]
    in_sp = [int(x.shape[2 + d]) for d in range(rank)]
    ks = [int(kernel_shape[d]) for d in range(rank)]
    st = [int(strides[d]) for d in range(rank)]
    pd = [int(pads[d]) for d in range(rank)]
    out_sp = _pool_output_spatial_shape(in_sp, ks, st, pd)
    if dy.shape != (n, c) + out_sp:
        raise ValueError("dy shape mismatch for average_pool_vjp")

    vol = float(np.prod(ks))
    o_list: list[int] = []
    iu_list: list[int] = []
    for b_ in range(n):
        for ch in range(c):
            for rest_out in itertools.product(*[range(o) for o in out_sp]):
                o_flat = int(np.ravel_multi_index((b_, ch) + rest_out, dy.shape))
                for k_rest in itertools.product(*[range(ks[d]) for d in range(rank)]):
                    pos = []
                    ok = True
                    for d in range(rank):
                        t = int(rest_out[d]) * int(st[d]) - int(pd[d]) + int(k_rest[d])
                        if not (0 <= t < int(in_sp[d])):
                            ok = False
                            break
                        pos.append(t)
                    if ok:
                        o_list.append(o_flat)
                        iu_list.append(int(np.ravel_multi_index((b_, ch) + tuple(pos), x.shape)))

    vi0 = np.zeros(len(o_list), dtype=np.int64)
    du, _ = _tensor_vjp(
        dy,
        x,
        np.array([1.0 / vol], dtype=np.float64),
        np.asarray(o_list, dtype=np.int64),
        np.asarray(iu_list, dtype=np.int64),
        vi0,
        accumulate_dv=False,
    )
    return du


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


def _jvp_relu(inputs: Tuple[np.ndarray, ...], tangents: Tuple[np.ndarray, ...], _meta: Dict[str, Any], _out: np.ndarray) -> np.ndarray:
    return (inputs[0] > 0.0).astype(np.float64) * tangents[0]


def _vjp_relu(inputs: Tuple[np.ndarray, ...], cotangent: np.ndarray, _meta: Dict[str, Any], _out: np.ndarray) -> Tuple[np.ndarray, ...]:
    return ((inputs[0] > 0.0).astype(np.float64) * cotangent,)


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


def _jvp_conv(inputs: Tuple[np.ndarray, ...], tangents: Tuple[np.ndarray, ...], meta: Dict[str, Any], _out: np.ndarray) -> np.ndarray:
    x, w, b = inputs
    dx, dw, db = tangents
    zero_b = np.zeros_like(b, dtype=np.float64)
    return conv_forward(dx, w, zero_b, meta["stride"], meta["pad_begin"], meta["pad_end"], meta["dilation"], meta["group"]) + conv_forward(
        x, dw, db, meta["stride"], meta["pad_begin"], meta["pad_end"], meta["dilation"], meta["group"]
    )


def _vjp_conv(inputs: Tuple[np.ndarray, ...], cotangent: np.ndarray, meta: Dict[str, Any], _out: np.ndarray) -> Tuple[np.ndarray, ...]:
    x, w, _b = inputs
    return conv_vjp(x, w, cotangent, meta["stride"], meta["pad_begin"], meta["pad_end"], meta["dilation"], meta["group"])


def _jvp_max_pool(inputs: Tuple[np.ndarray, ...], tangents: Tuple[np.ndarray, ...], meta: Dict[str, Any], _out: np.ndarray) -> np.ndarray:
    return max_pool_jvp(inputs[0], tangents[0], meta["kernel_shape"], meta["strides"], meta["pads"])


def _vjp_max_pool(inputs: Tuple[np.ndarray, ...], cotangent: np.ndarray, meta: Dict[str, Any], _out: np.ndarray) -> Tuple[np.ndarray, ...]:
    return (max_pool_vjp(inputs[0], cotangent, meta["kernel_shape"], meta["strides"], meta["pads"]),)


def _jvp_avg_pool(_inputs: Tuple[np.ndarray, ...], tangents: Tuple[np.ndarray, ...], meta: Dict[str, Any], _out: np.ndarray) -> np.ndarray:
    return average_pool_forward(tangents[0], meta["kernel_shape"], meta["strides"], meta["pads"])


def _vjp_avg_pool(inputs: Tuple[np.ndarray, ...], cotangent: np.ndarray, meta: Dict[str, Any], _out: np.ndarray) -> Tuple[np.ndarray, ...]:
    return (average_pool_vjp(inputs[0], cotangent, meta["kernel_shape"], meta["strides"], meta["pads"]),)


_RULES: Dict[str, PrimitiveRule] = {
    "add": PrimitiveRule(_jvp_add, _vjp_add),
    "sub": PrimitiveRule(_jvp_sub, _vjp_sub),
    "mul": PrimitiveRule(_jvp_mul, _vjp_mul),
    "div": PrimitiveRule(_jvp_div, _vjp_div),
    "pow": PrimitiveRule(_jvp_pow, _vjp_pow),
    "neg": PrimitiveRule(_jvp_neg, _vjp_neg),
    "exp": PrimitiveRule(_jvp_exp, _vjp_exp),
    "log": PrimitiveRule(_jvp_log, _vjp_log),
    "sin": PrimitiveRule(_jvp_sin, _vjp_sin),
    "cos": PrimitiveRule(_jvp_cos, _vjp_cos),
    "relu": PrimitiveRule(_jvp_relu, _vjp_relu),
    "sum": PrimitiveRule(_jvp_sum, _vjp_sum),
    "reshape": PrimitiveRule(_jvp_reshape, _vjp_reshape),
    "getitem": PrimitiveRule(_jvp_getitem, _vjp_getitem),
    "conv": PrimitiveRule(_jvp_conv, _vjp_conv),
    "max_pool": PrimitiveRule(_jvp_max_pool, _vjp_max_pool),
    "avg_pool": PrimitiveRule(_jvp_avg_pool, _vjp_avg_pool),
}


_BINARY_IMPL: Dict[str, Callable[[np.ndarray, np.ndarray], np.ndarray]] = {
    "add": np.add,
    "sub": np.subtract,
    "mul": np.multiply,
    "div": np.divide,
    "pow": np.power,
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

    def mode(self) -> str:
        return "jvp" if self.wrt.size <= self.output.size else "vjp"

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
                if self.mode() == "jvp"
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
        if self.mode() == "jvp":
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


def conv(
    x: ArrayLike,
    w: ArrayLike,
    b: ArrayLike,
    *,
    stride: Sequence[int],
    pad_begin: Sequence[int],
    pad_end: Sequence[int],
    dilation: Sequence[int],
    group: int = 1,
) -> Tensor:
    x_t = ensure_tensor(x)
    w_t = ensure_tensor(w)
    b_t = ensure_tensor(b)
    meta = {
        "stride": tuple(int(v) for v in stride),
        "pad_begin": tuple(int(v) for v in pad_begin),
        "pad_end": tuple(int(v) for v in pad_end),
        "dilation": tuple(int(v) for v in dilation),
        "group": int(group),
    }
    value = conv_forward(x_t.value, w_t.value, b_t.value, meta["stride"], meta["pad_begin"], meta["pad_end"], meta["dilation"], meta["group"])
    return Tensor(value, parents=(x_t, w_t, b_t), op="conv", meta=meta)


def max_pool(
    x: ArrayLike,
    *,
    kernel_shape: Sequence[int],
    strides: Sequence[int],
    pads: Sequence[int],
) -> Tensor:
    x_t = ensure_tensor(x)
    meta = {
        "kernel_shape": tuple(int(v) for v in kernel_shape),
        "strides": tuple(int(v) for v in strides),
        "pads": tuple(int(v) for v in pads),
    }
    value = max_pool_forward(x_t.value, meta["kernel_shape"], meta["strides"], meta["pads"])
    return Tensor(value, parents=(x_t,), op="max_pool", meta=meta)


def avg_pool(
    x: ArrayLike,
    *,
    kernel_shape: Sequence[int],
    strides: Sequence[int],
    pads: Sequence[int],
) -> Tensor:
    x_t = ensure_tensor(x)
    meta = {
        "kernel_shape": tuple(int(v) for v in kernel_shape),
        "strides": tuple(int(v) for v in strides),
        "pads": tuple(int(v) for v in pads),
    }
    value = average_pool_forward(x_t.value, meta["kernel_shape"], meta["strides"], meta["pads"])
    return Tensor(value, parents=(x_t,), op="avg_pool", meta=meta)


def softmax(x: Tensor, axis: int = -1) -> Tensor:
    ex = x.exp()
    return ex / ex.sum(axis=axis, keepdims=True)


def relu(x: ArrayLike) -> Tensor:
    return ensure_tensor(x).relu()


def _assert_allclose(actual: ArrayLike, expected: ArrayLike, *, atol: float = 1e-8, rtol: float = 1e-8) -> None:
    a = _as_array(actual)
    e = _as_array(expected)
    if not np.allclose(a, e, atol=atol, rtol=rtol):
        raise AssertionError(f"not close\nactual={a}\nexpected={e}")


def _demo() -> None:
    x = Tensor.leaf(np.array([0.2, -0.4, 0.7]), name="x")
    y = softmax(x)
    J = jacobian(y, x)

    y_val = _as_array(y.value)
    ref = np.diag(y_val) - np.outer(y_val, y_val)

    _assert_allclose(J.materialize_via_jvp(), ref)
    _assert_allclose(J.materialize_via_vjp(), ref)
    _assert_allclose(np.asarray(J), ref)
    _assert_allclose(J.column((1,)), ref[:, 1])
    _assert_allclose(J.row((2,)), ref[2, :])
    _assert_allclose(J[2, 1], ref[2, 1])

    W = Tensor.leaf(np.array([[1.0, 2.0], [3.0, 4.0]]), name="W")
    e = W[1, 0] * W[1, 0]
    d_e_d_W = jacobian(e, W)
    ref_alias = np.zeros_like(W.value)
    ref_alias[1, 0] = 6.0
    _assert_allclose(np.asarray(d_e_d_W), ref_alias)

    s = Tensor.leaf(np.array(3.0), name="s")
    loss = s * s + 2.0 * s + 1.0
    _assert_allclose(jvp(loss, {s: np.array(1.0)}), np.array(8.0))
    _assert_allclose(vjp(loss, np.array(1.0), s), np.array(8.0))

    x2 = Tensor.leaf(np.array([[[[1.0, -2.0, 0.5], [0.3, 1.2, -0.7], [2.0, -1.0, 0.8]]]], dtype=np.float64), name="x")
    w2 = Tensor.leaf(np.array([[[[0.4, -0.1], [0.2, 0.3]]]], dtype=np.float64), name="w")
    b2 = Tensor.leaf(np.array([0.15], dtype=np.float64), name="b")
    c = conv(x2, w2, b2, stride=(1, 1), pad_begin=(0, 0), pad_end=(0, 0), dilation=(1, 1), group=1).named("c")
    r = relu(c).named("r")
    p = max_pool(r, kernel_shape=(2, 2), strides=(1, 1), pads=(0, 0)).named("p")
    symbolic = symbolic_tangent_program(p)
    assert "let @c =" in symbolic
    assert "select_at_argmax" in symbolic
    assert "conv(@x" in symbolic or "conv(@x," in symbolic

    print("ok: NumPy JVP/VJP/lazy-Jacobian prototype passes self-checks")


if __name__ == "__main__":
    _demo()
