#!/usr/bin/env python3
"""
Standalone verifier for analytic gradients of NumPy reductions on a 3D tensor.

We test gradients of:
  - sum, mean, prod, max, min

for multiple axis/keepdims configurations by comparing analytic grads to
finite-difference grads of a scalar objective:

    f(x) = sum(reduce(x, axis=axis, keepdims=keepdims) * upstream)

This way, the upstream gradient has the same shape as the reduction output,
and the expected gradient wrt x is the reduction's VJP applied to upstream.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Callable, Iterable, Optional, Sequence, Tuple, Union

import numpy as np

Axis = Optional[Union[int, Tuple[int, ...]]]


def _as_tuple_axis(axis: Axis) -> Optional[Tuple[int, ...]]:
    if axis is None:
        return None
    if isinstance(axis, int):
        return (axis,)
    return tuple(axis)


def _normalize_axis(axis: Axis, ndim: int) -> Optional[Tuple[int, ...]]:
    ax = _as_tuple_axis(axis)
    if ax is None:
        return None
    out = []
    for a in ax:
        if a < 0:
            a = ndim + a
        if not (0 <= a < ndim):
            raise ValueError(f"axis {axis} out of bounds for ndim={ndim}")
        out.append(int(a))
    # De-dup like NumPy effectively does (it errors on duplicates in many ops);
    # here we just canonicalize for our own shape math.
    return tuple(sorted(set(out)))


def _expand_like(x: np.ndarray, target_ndim: int, axis: Optional[Tuple[int, ...]]) -> np.ndarray:
    """
    Given a tensor x that is the result of a reduction with keepdims=False,
    re-insert singleton dims at the reduced axes so it can broadcast to the input.
    """
    if axis is None:
        # Reduction to scalar: expand to all singleton dims.
        while x.ndim < target_ndim:
            x = np.expand_dims(x, axis=0)
        return x
    for a in axis:
        x = np.expand_dims(x, axis=a)
    if x.ndim != target_ndim:
        raise RuntimeError(f"internal: expanded ndim mismatch: {x.ndim} vs {target_ndim}")
    return x


def grad_sum(x: np.ndarray, upstream: np.ndarray, axis: Axis, keepdims: bool) -> np.ndarray:
    ax = _normalize_axis(axis, x.ndim)
    g = upstream
    if not keepdims:
        g = _expand_like(g, x.ndim, ax)
    return np.broadcast_to(g, x.shape).astype(x.dtype, copy=False)


def grad_mean(x: np.ndarray, upstream: np.ndarray, axis: Axis, keepdims: bool) -> np.ndarray:
    ax = _normalize_axis(axis, x.ndim)
    if ax is None:
        denom = x.size
    else:
        denom = int(np.prod([x.shape[a] for a in ax]))
    return grad_sum(x, upstream, axis=axis, keepdims=keepdims) / np.array(denom, dtype=x.dtype)


def grad_prod(x: np.ndarray, upstream: np.ndarray, axis: Axis, keepdims: bool) -> np.ndarray:
    """
    For prod: y = prod(x, axis)
      dy/dx_i = y / x_i, but this is undefined at x_i=0.

    We implement a numerically stable rule that handles zeros:
      - If there are no zeros along the reduced axes: grad = upstream * y / x
      - If exactly one zero: gradient is upstream * prod(nonzero) at the zero position, 0 elsewhere
      - If >=2 zeros: gradient is 0 everywhere for that reduction slice
    """
    ax = _normalize_axis(axis, x.ndim)
    if ax is None:
        # treat as reduction over all dims
        ax = tuple(range(x.ndim))

    y = np.prod(x, axis=ax, keepdims=True)
    g = upstream
    if not keepdims:
        g = _expand_like(g, x.ndim, ax)

    xk = x
    zeros = (xk == 0)
    zero_count = zeros.sum(axis=ax, keepdims=True)

    # Base case: no zeros
    no_zero = zero_count == 0
    out = np.zeros_like(xk, dtype=np.result_type(xk.dtype, g.dtype))
    out = np.where(no_zero, g * (y / xk), out)

    # Exactly one zero: only that position gets upstream * prod(nonzero)
    one_zero = zero_count == 1
    if np.any(one_zero):
        # prod over nonzero elements (treat zeros as 1)
        prod_nonzero = np.prod(np.where(zeros, 1, xk), axis=ax, keepdims=True)
        out = np.where(one_zero & zeros, g * prod_nonzero, out)

    # Two or more zeros: stays 0
    return out.astype(x.dtype, copy=False)


def _reduce_forward(op: str, x: np.ndarray, axis: Axis, keepdims: bool) -> np.ndarray:
    if op == "sum":
        return np.sum(x, axis=axis, keepdims=keepdims)
    if op == "mean":
        return np.mean(x, axis=axis, keepdims=keepdims)
    if op == "prod":
        return np.prod(x, axis=axis, keepdims=keepdims)
    if op == "max":
        return np.max(x, axis=axis, keepdims=keepdims)
    if op == "min":
        return np.min(x, axis=axis, keepdims=keepdims)
    raise ValueError(f"unknown op: {op}")


def grad_maxmin(op: str, x: np.ndarray, upstream: np.ndarray, axis: Axis, keepdims: bool) -> np.ndarray:
    ax = _normalize_axis(axis, x.ndim)
    if ax is None:
        ax = tuple(range(x.ndim))

    keep_y = _reduce_forward(op, x, axis=ax, keepdims=True)
    g = upstream
    if not keepdims:
        g = _expand_like(g, x.ndim, ax)

    if op == "max":
        mask = (x == keep_y)
    elif op == "min":
        mask = (x == keep_y)
    else:
        raise ValueError(op)

    # Distribute gradient equally among ties.
    tie_count = mask.sum(axis=ax, keepdims=True)
    tie_count = np.maximum(tie_count, 1)
    return (g * mask / tie_count).astype(x.dtype, copy=False)


def analytic_grad(op: str, x: np.ndarray, upstream: np.ndarray, axis: Axis, keepdims: bool) -> np.ndarray:
    if op == "sum":
        return grad_sum(x, upstream, axis, keepdims)
    if op == "mean":
        return grad_mean(x, upstream, axis, keepdims)
    if op == "prod":
        return grad_prod(x, upstream, axis, keepdims)
    if op == "max":
        return grad_maxmin("max", x, upstream, axis, keepdims)
    if op == "min":
        return grad_maxmin("min", x, upstream, axis, keepdims)
    raise ValueError(op)


def finite_diff_grad(
    f: Callable[[np.ndarray], float],
    x: np.ndarray,
    eps: float = 1e-4,
) -> np.ndarray:
    x = np.array(x, dtype=np.float64, copy=True)
    g = np.zeros_like(x, dtype=np.float64)
    it = np.nditer(x, flags=["multi_index"], op_flags=["readwrite"])
    while not it.finished:
        idx = it.multi_index
        old = x[idx]
        x[idx] = old + eps
        f_pos = float(f(x))
        x[idx] = old - eps
        f_neg = float(f(x))
        x[idx] = old
        g[idx] = (f_pos - f_neg) / (2.0 * eps)
        it.iternext()
    return g


@dataclass(frozen=True)
class Case:
    op: str
    axis: Axis
    keepdims: bool


def run_case(
    rng: np.random.Generator,
    shape: Tuple[int, int, int],
    case: Case,
    *,
    eps: float,
    atol: float,
    rtol: float,
    return_tensors: bool = False,
) -> Tuple[bool, str]:
    # Use float64 for numerics; cast analytic grad to float64 before compare.
    x = rng.standard_normal(size=shape).astype(np.float64)
    # For prod, avoid being too close to 0 (fd becomes noisy). Still allow sign changes.
    if case.op == "prod":
        x = x + 0.3
    # For max/min, ties are very unlikely, but still handle them.
    y = _reduce_forward(case.op, x, axis=case.axis, keepdims=case.keepdims)
    upstream = rng.standard_normal(size=y.shape).astype(np.float64)

    def f(z: np.ndarray) -> float:
        yz = _reduce_forward(case.op, z, axis=case.axis, keepdims=case.keepdims)
        return float(np.sum(yz * upstream))

    g_ana = analytic_grad(case.op, x, upstream, axis=case.axis, keepdims=case.keepdims).astype(np.float64)
    g_fd = finite_diff_grad(f, x, eps=eps)

    diff = np.abs(g_ana - g_fd)
    denom = np.maximum(np.abs(g_fd), np.abs(g_ana))
    rel = diff / np.maximum(denom, 1e-12)

    ok = bool(np.all(diff <= atol + rtol * denom))
    msg = (
        f"{case.op:>4} axis={case.axis!s:<8} keepdims={str(case.keepdims):<5} "
        f"max_abs={diff.max():.3e} max_rel={rel.max():.3e}"
    )
    if return_tensors:
        # type: ignore[return-value]
        return ok, msg, x, y, upstream, g_ana, g_fd
    return ok, msg


def main(argv: Sequence[str]) -> int:
    rng = np.random.default_rng(0)
    shape = (3, 4, 5)

    argv_set = set(argv)
    if "--show" in argv_set or "--show-all" in argv_set:
        np.set_printoptions(precision=4, suppress=True, linewidth=140)

        # Defaults chosen to clearly show 3D broadcast-back behavior.
        axis: Axis = (1, 2)
        keepdims = False

        if "--axis0" in argv_set:
            axis = 0
        elif "--axis1" in argv_set:
            axis = 1
        elif "--axis2" in argv_set:
            axis = 2
        elif "--axis01" in argv_set:
            axis = (0, 1)
        elif "--axis02" in argv_set:
            axis = (0, 2)
        elif "--axis12" in argv_set:
            axis = (1, 2)
        elif "--axisNone" in argv_set:
            axis = None

        if "--keepdims" in argv_set:
            keepdims = True

        ops = ["sum"]
        if "--show-all" in argv_set:
            ops = ["sum", "prod", "min", "max"]
        else:
            for candidate in ("sum", "mean", "prod", "min", "max"):
                flag = f"--op={candidate}"
                if flag in argv_set:
                    ops = [candidate]
                    break

        overall_ok = True
        for op in ops:
            example = Case(op=op, axis=axis, keepdims=keepdims)
            atol = 5e-6
            rtol = 5e-4
            if op in ("prod", "min", "max"):
                atol = 5e-5
                rtol = 5e-3
            ok, msg, x, y, upstream, g_ana, g_fd = run_case(
                rng,
                shape,
                example,
                eps=1e-4,
                atol=atol,
                rtol=rtol,
                return_tensors=True,
            )
            overall_ok = overall_ok and ok
            print(("OK   " if ok else "FAIL ") + msg)
            print("\n--- example tensors ---")
            print(f"x.shape        = {x.shape}")
            print(f"reduce(x).shape = {y.shape}   (op={example.op!r}, axis={example.axis}, keepdims={example.keepdims})")
            print(f"upstream.shape  = {upstream.shape}")
            print(f"grad.shape      = {g_ana.shape}")
            print("\nupstream =")
            print(upstream)
            print("\ngrad (analytic) =")
            print(g_ana)
            print("\ngrad (finite-diff) =")
            print(g_fd)
            print("\nmax_abs_diff =", float(np.max(np.abs(g_ana - g_fd))))
            print("\n" + ("=" * 80) + "\n")
        return 0 if overall_ok else 1

    axes: Sequence[Axis] = [None, 0, 1, 2, (0, 1), (1, 2), (0, 2)]
    cases: list[Case] = []
    for op in ("sum", "mean", "prod", "max", "min"):
        for axis in axes:
            for keepdims in (False, True):
                cases.append(Case(op=op, axis=axis, keepdims=keepdims))

    eps = 1e-4
    # Tolerances: prod + max/min can be noisier (non-smooth); keep slightly looser.
    base_atol = 5e-6
    base_rtol = 5e-4

    failed: list[str] = []
    for case in cases:
        atol = base_atol
        rtol = base_rtol
        if case.op in ("prod", "max", "min"):
            atol = 5e-5
            rtol = 5e-3
        ok, msg = run_case(rng, shape, case, eps=eps, atol=atol, rtol=rtol)
        print(("OK   " if ok else "FAIL ") + msg)
        if not ok:
            failed.append(msg)

    if failed:
        print("\nFailures:", file=sys.stderr)
        for m in failed:
            print("  " + m, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))

