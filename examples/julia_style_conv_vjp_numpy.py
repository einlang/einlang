"""
Julia / Zygote-style ML ops in pure NumPy (no Einlang): conv, ReLU, max/average pool.

Conv: nested loops with zero-padded Xp; layout matches stdlib/ml/conv_ops.ein (NCHW / NCDHW),
``group`` as in ``pub fn conv``. ``einlang_pads_to_begin_end`` maps conv ``pads`` (length ``2 * rank``)
to ``(pad_begin, pad_end)``. ``conv_ml`` / ``conv_vjp_ml`` mirror the stdlib conv entry point.

ReLU matches stdlib/ml/activations.ein ``relu`` (elementwise ``max(0, x)``).

Pooling matches stdlib/ml/pool_ops.ein indexing ``i * stride - pad + offset`` per spatial dim; one
integer pad per dim (not conv's begin/end vector). Out-of-window positions contribute ``-inf`` to
max-pool argmax selection and ``0`` to average-pool sums; average divides by full kernel volume.

``tensor_vjp`` is the only reverse-mode primitive used here: for contributions
``y.flat[out] += sum_k u.flat[u_i[k]] * v.flat[v_i[k]]`` (accumulated with ``np.add.at``), it returns
``(du, dv)`` for ``L = sum(dy * y)`` with ``v`` treated as constant when ``accumulate_dv=False`` (ReLU
uses ``u=x`` and ``v`` the frozen mask). Conv, max-pool, and average-pool build sparse index triples
into ``tensor_vjp``; bias ``db`` for conv stays a separate reduction over ``dy``. ``test_autodiff_pass``
conv notes for ``ref_dx`` rank 2/3 still apply.

Run: python3 examples/julia_style_conv_vjp_numpy.py
"""

from __future__ import annotations

import itertools
from typing import Sequence, Tuple

import numpy as np


def einlang_pads_to_begin_end(
    pads: Sequence[int], rank: int
) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
    """Map ``std::ml::conv`` pads [begin0,…,begin_{r-1}, end0,…,end_{r-1}] to two length-r tuples."""
    pads_t = tuple(int(p) for p in pads)
    if len(pads_t) != 2 * rank:
        raise ValueError(f"pads length must be 2 * rank ({2 * rank}), got {len(pads_t)}")
    pad_begin = tuple(pads_t[d] for d in range(rank))
    pad_end = tuple(pads_t[rank + d] for d in range(rank))
    return pad_begin, pad_end


def conv_ml(
    x: np.ndarray,
    w: np.ndarray,
    b: np.ndarray,
    strides: Sequence[int],
    pads: Sequence[int],
    dilations: Sequence[int],
    group: int = 1,
) -> np.ndarray:
    rank = len(strides)
    pad_begin, pad_end = einlang_pads_to_begin_end(pads, rank)
    return conv_forward(x, w, b, strides, pad_begin, pad_end, dilations, group)


def conv_vjp_ml(
    x: np.ndarray,
    w: np.ndarray,
    dy: np.ndarray,
    strides: Sequence[int],
    pads: Sequence[int],
    dilations: Sequence[int],
    group: int = 1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rank = len(strides)
    pad_begin, pad_end = einlang_pads_to_begin_end(pads, rank)
    return conv_vjp(x, w, dy, strides, pad_begin, pad_end, dilations, group)


def _pad_spatial(
    x: np.ndarray,
    pad_begin: Sequence[int],
    pad_end: Sequence[int],
) -> np.ndarray:
    """Zero-pad spatial dims (last rank axes); x is (N, C, *spatial_in)."""
    x = np.asarray(x, dtype=np.float64)
    rank = len(pad_begin)
    assert x.ndim == 2 + rank
    assert len(pad_end) == rank
    pads_width = [(pad_begin[d], pad_end[d]) for d in range(rank)]
    return np.pad(x, [(0, 0), (0, 0)] + pads_width, mode="constant", constant_values=0.0)


def _output_spatial_shape(
    w_p: Sequence[int],
    kernel: Sequence[int],
    stride: Sequence[int],
    dilation: Sequence[int],
) -> Tuple[int, ...]:
    rank = len(w_p)
    out = []
    for d in range(rank):
        o = (w_p[d] - dilation[d] * (kernel[d] - 1) - 1) // stride[d] + 1
        out.append(int(o))
    return tuple(out)


def tensor_vjp(
    dy: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    out_index: np.ndarray,
    u_index: np.ndarray,
    v_index: np.ndarray,
    *,
    accumulate_dv: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    dyf = np.asarray(dy, dtype=np.float64).ravel()
    u = np.asarray(u, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    uf = u.ravel()
    vf = v.ravel()
    oi = np.asarray(out_index, dtype=np.int64).ravel()
    ui = np.asarray(u_index, dtype=np.int64).ravel()
    vi = np.asarray(v_index, dtype=np.int64).ravel()
    if oi.size == 0:
        return np.zeros_like(u), np.zeros_like(v)
    if ui.size != oi.size or vi.size != oi.size:
        raise ValueError("out_index, u_index, v_index must have the same length")
    du = np.zeros_like(uf)
    np.add.at(du, ui, dyf[oi] * vf[vi])
    dv = np.zeros_like(vf)
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
    """Forward conv. x (N,C_in,*in), w (C_out,C_in/group,*k), b (C_out,)."""
    x = np.asarray(x, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    rank = x.ndim - 2
    n, c_in = x.shape[0], x.shape[1]
    c_out, cpg = w.shape[0], w.shape[1]
    kernel = list(w.shape[2:])
    assert w.ndim == 2 + rank
    assert len(stride) == rank and len(dilation) == rank
    assert len(pad_begin) == rank and len(pad_end) == rank
    assert c_in % group == 0 and c_out % group == 0
    assert cpg == c_in // group
    fpg = c_out // group

    x_p = _pad_spatial(x, pad_begin, pad_end)
    w_p_shape = [x_p.shape[2 + d] for d in range(rank)]
    out_sp = _output_spatial_shape(w_p_shape, kernel, stride, dilation)
    y = np.zeros((n, c_out) + out_sp, dtype=np.float64)

    for b_ in range(n):
        for co in range(c_out):
            g = co // fpg
            base_cl = g * cpg
            acc_bias = float(b[co])
            for rest_out in itertools.product(*[range(o) for o in out_sp]):
                s = acc_bias
                for cl_off in range(cpg):
                    cl = base_cl + cl_off
                    for k_rest in itertools.product(*[range(kernel[d]) for d in range(rank)]):
                        pos = []
                        ok = True
                        for d in range(rank):
                            p = rest_out[d] * stride[d] + k_rest[d] * dilation[d]
                            if not (0 <= p < w_p_shape[d]):
                                ok = False
                                break
                            pos.append(p)
                        if not ok:
                            continue
                        idx_xp = (b_, cl) + tuple(pos)
                        idx_w = (co, cl_off) + k_rest
                        s += x_p[idx_xp] * w[idx_w]
                y[(b_, co) + rest_out] = s
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
    x = np.asarray(x, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)
    dy = np.asarray(dy, dtype=np.float64)
    rank = x.ndim - 2
    n, c_in = x.shape[0], x.shape[1]
    c_out, cpg = w.shape[0], w.shape[1]
    kernel = list(w.shape[2:])
    assert c_in % group == 0 and c_out % group == 0
    assert cpg == c_in // group
    fpg = c_out // group

    x_p = _pad_spatial(x, pad_begin, pad_end)
    w_p_shape = [x_p.shape[2 + d] for d in range(rank)]
    out_sp = _output_spatial_shape(w_p_shape, kernel, stride, dilation)
    assert dy.shape == (n, c_out) + out_sp

    sh_dy = dy.shape
    sh_xp = x_p.shape
    sh_w = w.shape
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
                o_flat = int(np.ravel_multi_index((b_, co) + rest_out, sh_dy))
                for cl_off in range(cpg):
                    cl = base_cl + cl_off
                    for k_rest in itertools.product(*[range(kernel[d]) for d in range(rank)]):
                        pos = []
                        ok = True
                        for d in range(rank):
                            p = rest_out[d] * stride[d] + k_rest[d] * dilation[d]
                            if not (0 <= p < w_p_shape[d]):
                                ok = False
                                break
                            pos.append(p)
                        if not ok:
                            continue
                        idx_xp = (b_, cl) + tuple(pos)
                        idx_w = (co, cl_off) + k_rest
                        o_list.append(o_flat)
                        iu_list.append(int(np.ravel_multi_index(idx_xp, sh_xp)))
                        iv_list.append(int(np.ravel_multi_index(idx_w, sh_w)))

    dx_p, dw = tensor_vjp(
        dy,
        x_p,
        w,
        np.array(o_list, dtype=np.int64),
        np.array(iu_list, dtype=np.int64),
        np.array(iv_list, dtype=np.int64),
    )

    slices = [slice(None), slice(None)]
    for d in range(rank):
        pb = pad_begin[d]
        lim = x.shape[2 + d]
        slices.append(slice(pb, pb + lim))
    dx = np.array(dx_p[tuple(slices)], dtype=np.float64, copy=True)
    return dx, dw, db


def relu_forward(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return np.maximum(x, 0.0)


def relu_vjp(x: np.ndarray, dy: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    dy = np.asarray(dy, dtype=np.float64)
    assert dy.shape == x.shape
    m = (x > 0.0).astype(np.float64)
    n = int(x.size)
    idx = np.arange(n, dtype=np.int64)
    du, _ = tensor_vjp(
        dy,
        x,
        m,
        idx,
        idx,
        idx,
        accumulate_dv=False,
    )
    return du


def _pool_output_spatial_shape(
    in_spatial: Sequence[int],
    kernel_shape: Sequence[int],
    strides: Sequence[int],
    pads: Sequence[int],
) -> Tuple[int, ...]:
    rank = len(in_spatial)
    out: list[int] = []
    for d in range(rank):
        kd = int(kernel_shape[d])
        sd = int(strides[d])
        pd_ = int(pads[d])
        assert kd >= 1 and sd >= 1
        i = 0
        while True:
            hit = False
            for m in range(kd):
                t = i * sd - pd_ + m
                if 0 <= t < in_spatial[d]:
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
    rank: int,
) -> Tuple[float, Tuple[int, ...] | None]:
    best = -np.inf
    winner: Tuple[int, ...] | None = None
    for k_rest in itertools.product(*[range(int(ks[d])) for d in range(rank)]):
        pos: list[int] = []
        ok = True
        for d in range(rank):
            t = int(rest_out[d]) * int(st[d]) - int(pd[d]) + int(k_rest[d])
            if not (0 <= t < in_sp[d]):
                ok = False
                break
            pos.append(t)
        if ok:
            v = float(x[(b_, ch) + tuple(pos)])
        else:
            v = -np.inf
        if v > best:
            best = v
            winner = tuple(pos) if ok else None
    return float(best), winner


def max_pool_forward(
    x: np.ndarray,
    kernel_shape: Sequence[int],
    strides: Sequence[int],
    pads: Sequence[int],
) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    rank = len(kernel_shape)
    assert x.ndim == 2 + rank
    n, c = x.shape[0], x.shape[1]
    in_sp = [int(x.shape[2 + d]) for d in range(rank)]
    ks = [int(kernel_shape[d]) for d in range(rank)]
    st = [int(strides[d]) for d in range(rank)]
    pd = [int(pads[d]) for d in range(rank)]
    assert len(st) == rank and len(pd) == rank
    out_sp = _pool_output_spatial_shape(in_sp, ks, st, pd)
    y = np.empty((n, c) + out_sp, dtype=np.float64)
    for b_ in range(n):
        for ch in range(c):
            for rest_out in itertools.product(*[range(o) for o in out_sp]):
                v, _ = _pool_window_value_max(x, b_, ch, rest_out, in_sp, ks, st, pd, rank)
                y[(b_, ch) + rest_out] = v
    return y


def max_pool_vjp(
    x: np.ndarray,
    dy: np.ndarray,
    kernel_shape: Sequence[int],
    strides: Sequence[int],
    pads: Sequence[int],
) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    dy = np.asarray(dy, dtype=np.float64)
    rank = len(kernel_shape)
    n, c = x.shape[0], x.shape[1]
    in_sp = [int(x.shape[2 + d]) for d in range(rank)]
    ks = [int(kernel_shape[d]) for d in range(rank)]
    st = [int(strides[d]) for d in range(rank)]
    pd = [int(pads[d]) for d in range(rank)]
    out_sp = _pool_output_spatial_shape(in_sp, ks, st, pd)
    assert dy.shape == (n, c) + out_sp
    sh_dy = dy.shape
    sh_x = x.shape
    o_list: list[int] = []
    iu_list: list[int] = []
    v_one = np.array([1.0], dtype=np.float64)
    for b_ in range(n):
        for ch in range(c):
            for rest_out in itertools.product(*[range(o) for o in out_sp]):
                _, winner = _pool_window_value_max(
                    x, b_, ch, rest_out, in_sp, ks, st, pd, rank
                )
                if winner is None:
                    continue
                o_list.append(
                    int(np.ravel_multi_index((b_, ch) + rest_out, sh_dy))
                )
                iu_list.append(
                    int(np.ravel_multi_index((b_, ch) + winner, sh_x))
                )
    vi0 = np.zeros(len(o_list), dtype=np.int64)
    du, _ = tensor_vjp(
        dy,
        x,
        v_one,
        np.array(o_list, dtype=np.int64),
        np.array(iu_list, dtype=np.int64),
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
    x = np.asarray(x, dtype=np.float64)
    rank = len(kernel_shape)
    assert x.ndim == 2 + rank
    n, c = x.shape[0], x.shape[1]
    in_sp = [int(x.shape[2 + d]) for d in range(rank)]
    ks = [int(kernel_shape[d]) for d in range(rank)]
    st = [int(strides[d]) for d in range(rank)]
    pd = [int(pads[d]) for d in range(rank)]
    vol = float(np.prod(ks))
    out_sp = _pool_output_spatial_shape(in_sp, ks, st, pd)
    y = np.zeros((n, c) + out_sp, dtype=np.float64)
    for b_ in range(n):
        for ch in range(c):
            for rest_out in itertools.product(*[range(o) for o in out_sp]):
                s = 0.0
                for k_rest in itertools.product(*[range(ks[d]) for d in range(rank)]):
                    pos = []
                    ok = True
                    for d in range(rank):
                        t = int(rest_out[d]) * st[d] - pd[d] + int(k_rest[d])
                        if not (0 <= t < in_sp[d]):
                            ok = False
                            break
                        pos.append(t)
                    if ok:
                        s += float(x[(b_, ch) + tuple(pos)])
                y[(b_, ch) + rest_out] = s / vol
    return y


def average_pool_vjp(
    x: np.ndarray,
    dy: np.ndarray,
    kernel_shape: Sequence[int],
    strides: Sequence[int],
    pads: Sequence[int],
) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    dy = np.asarray(dy, dtype=np.float64)
    rank = len(kernel_shape)
    n, c = x.shape[0], x.shape[1]
    in_sp = [int(x.shape[2 + d]) for d in range(rank)]
    ks = [int(kernel_shape[d]) for d in range(rank)]
    st = [int(strides[d]) for d in range(rank)]
    pd = [int(pads[d]) for d in range(rank)]
    vol = float(np.prod(ks))
    out_sp = _pool_output_spatial_shape(in_sp, ks, st, pd)
    assert dy.shape == (n, c) + out_sp
    sh_dy = dy.shape
    sh_x = x.shape
    o_list: list[int] = []
    iu_list: list[int] = []
    v_scale = np.array([1.0 / vol], dtype=np.float64)
    for b_ in range(n):
        for ch in range(c):
            for rest_out in itertools.product(*[range(o) for o in out_sp]):
                o_flat = int(np.ravel_multi_index((b_, ch) + rest_out, sh_dy))
                for k_rest in itertools.product(*[range(ks[d]) for d in range(rank)]):
                    pos = []
                    ok = True
                    for d in range(rank):
                        t = int(rest_out[d]) * st[d] - pd[d] + int(k_rest[d])
                        if not (0 <= t < in_sp[d]):
                            ok = False
                            break
                        pos.append(t)
                    if ok:
                        o_list.append(o_flat)
                        iu_list.append(
                            int(np.ravel_multi_index((b_, ch) + tuple(pos), sh_x))
                        )
    vi0 = np.zeros(len(o_list), dtype=np.int64)
    du, _ = tensor_vjp(
        dy,
        x,
        v_scale,
        np.array(o_list, dtype=np.int64),
        np.array(iu_list, dtype=np.int64),
        vi0,
        accumulate_dv=False,
    )
    return du


def _loss(
    x: np.ndarray,
    w: np.ndarray,
    b: np.ndarray,
    stride: Sequence[int],
    pad_begin: Sequence[int],
    pad_end: Sequence[int],
    dilation: Sequence[int],
    group: int,
    dy: np.ndarray,
) -> float:
    y = conv_forward(x, w, b, stride, pad_begin, pad_end, dilation, group)
    return float(np.sum(dy * y))


def _finite_diff_grads(
    x0: np.ndarray,
    w0: np.ndarray,
    b0: np.ndarray,
    stride: Sequence[int],
    pad_begin: Sequence[int],
    pad_end: Sequence[int],
    dilation: Sequence[int],
    group: int,
    dy: np.ndarray,
    eps: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    L0 = _loss(x0, w0, b0, stride, pad_begin, pad_end, dilation, group, dy)
    dx = np.zeros_like(x0)
    x = x0.copy()
    for idx in np.ndindex(x0.shape):
        x[idx] = x0[idx] + eps
        dx[idx] = (
            _loss(x, w0, b0, stride, pad_begin, pad_end, dilation, group, dy) - L0
        ) / eps
        x[idx] = x0[idx]
    dw = np.zeros_like(w0)
    w = w0.copy()
    for idx in np.ndindex(w0.shape):
        w[idx] = w0[idx] + eps
        dw[idx] = (
            _loss(x0, w, b0, stride, pad_begin, pad_end, dilation, group, dy) - L0
        ) / eps
        w[idx] = w0[idx]
    db = np.zeros_like(b0)
    b = b0.copy()
    for i in range(int(b0.shape[0])):
        b[i] = b0[i] + eps
        db[i] = (
            _loss(x0, w0, b, stride, pad_begin, pad_end, dilation, group, dy) - L0
        ) / eps
        b[i] = b0[i]
    return dx, dw, db


def _check_einlang_pads_explicit_rank2() -> None:
    rng = np.random.default_rng(0)
    x = rng.standard_normal((1, 1, 4, 5))
    w = rng.standard_normal((1, 1, 2, 3))
    b = np.zeros(1)
    stride = [1, 1]
    dil = [1, 1]
    pads = [1, 0, 2, 1]
    pb, pe = einlang_pads_to_begin_end(pads, 2)
    assert pb == (1, 0) and pe == (2, 1)
    y_ml = conv_ml(x, w, b, stride, pads, dil, 1)
    y_ex = conv_forward(x, w, b, stride, pb, pe, dil, 1)
    assert np.allclose(y_ml, y_ex), (y_ml, y_ex)


def _check_finite_diff_rank2_group_stride_dilation() -> None:
    rng = np.random.default_rng(1)
    x0 = rng.standard_normal((1, 2, 4, 5))
    w0 = rng.standard_normal((2, 1, 2, 3))
    b0 = rng.standard_normal(2)
    stride = [2, 1]
    pads = [1, 0, 1, 0]
    dil = [2, 1]
    group = 2
    y = conv_ml(x0, w0, b0, stride, pads, dil, group)
    dy = rng.standard_normal(y.shape)
    dx, dw, db = conv_vjp_ml(x0, w0, dy, stride, pads, dil, group)
    pb, pe = einlang_pads_to_begin_end(pads, 2)
    eps = 1e-5
    dx_fd, dw_fd, db_fd = _finite_diff_grads(
        x0, w0, b0, stride, pb, pe, dil, group, dy, eps
    )
    rtol, atol = 2e-3, 2e-3
    assert np.allclose(dx, dx_fd, rtol=rtol, atol=atol), np.max(np.abs(dx - dx_fd))
    assert np.allclose(dw, dw_fd, rtol=rtol, atol=atol), np.max(np.abs(dw - dw_fd))
    assert np.allclose(db, db_fd, rtol=rtol, atol=atol), np.max(np.abs(db - db_fd))


def _check_rank1() -> None:
    """Rank-1 conv VJP with dy=ones: reference for ``test_conv_autodiff_jacobian_rank1_matches_calculus`` (quotient @y/@x, @y/@w, @y/@b)."""
    x = np.array([[[1.0, 2.0, 3.0, 4.0]]])
    w = np.array([[[1.0, 0.5]]])
    b = np.array([0.0])
    stride = [1]
    pads = [0, 0]
    dil = [1]
    y = conv_ml(x, w, b, stride, pads, dil, 1)
    dy = np.ones_like(y)
    dx, dw, db = conv_vjp_ml(x, w, dy, stride, pads, dil, 1)
    ref_dx = np.array([[[1.0, 1.5, 1.5, 0.5]]])
    ref_dw = np.array([[[6.0, 9.0]]])
    ref_db = np.array([3.0])
    assert np.allclose(dx, ref_dx), (dx, ref_dx)
    assert np.allclose(dw, ref_dw), (dw, ref_dw)
    assert np.allclose(db, ref_db), (db, ref_db)


def _check_rank2() -> None:
    x = np.array([[[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]]])
    w = np.array([[[[1.0, 1.0], [1.0, 1.0]]]])
    b = np.array([0.0])
    stride = [1, 1]
    pb = [0, 0, 0, 0]
    dil = [1, 1]
    y = conv_ml(x, w, b, stride, pb, dil, 1)
    dy_ones = np.ones_like(y)
    _, dw, db = conv_vjp_ml(x, w, dy_ones, stride, pb, dil, 1)
    ref_dw = np.array([[[[12.0, 16.0], [24.0, 28.0]]]])
    ref_db = np.array([4.0])
    assert np.allclose(dw, ref_dw), (dw, ref_dw)
    assert np.allclose(db, ref_db), (db, ref_db)

    dy0 = np.zeros_like(y)
    dy0[(0,) * dy0.ndim] = 1.0
    dx, _, _ = conv_vjp_ml(x, w, dy0, stride, pb, dil, 1)
    ref_dx = np.array([[[[1.0, 1.0, 0.0], [1.0, 1.0, 0.0], [0.0, 0.0, 0.0]]]])
    assert np.allclose(dx, ref_dx), (dx, ref_dx)

    dx_full, _, _ = conv_vjp_ml(x, w, dy_ones, stride, pb, dil, 1)
    expected_overlap = np.array(
        [[[[1.0, 2.0, 1.0], [2.0, 4.0, 2.0], [1.0, 2.0, 1.0]]]],
        dtype=np.float64,
    )
    assert np.allclose(dx_full, expected_overlap), (dx_full, expected_overlap)


def _check_rank3() -> None:
    x = np.array([[[[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]]])
    assert x.shape == (1, 1, 2, 2, 2)
    w = np.array([[[[[1.0]]]]])
    assert w.shape == (1, 1, 1, 1, 1)
    b = np.array([0.0])
    stride = [1, 1, 1]
    pb = [0, 0, 0, 0, 0, 0]
    dil = [1, 1, 1]
    y = conv_ml(x, w, b, stride, pb, dil, 1)
    dy_ones = np.ones_like(y)
    _, dw, db = conv_vjp_ml(x, w, dy_ones, stride, pb, dil, 1)
    ref_dw = np.array([[[[[36.0]]]]])
    ref_db = np.array([8.0])
    assert np.allclose(dw, ref_dw), (dw, ref_dw)
    assert np.allclose(db, ref_db), (db, ref_db)

    dy0 = np.zeros_like(y)
    dy0[(0,) * dy0.ndim] = 1.0
    dx, _, _ = conv_vjp_ml(x, w, dy0, stride, pb, dil, 1)
    ref_dx = np.array([[[[[1.0, 0.0], [0.0, 0.0]], [[0.0, 0.0], [0.0, 0.0]]]]])
    assert np.allclose(dx, ref_dx), (dx, ref_dx)

    dx_full, _, _ = conv_vjp_ml(x, w, dy_ones, stride, pb, dil, 1)
    assert np.allclose(dx_full, np.ones_like(x)), (dx_full,)


if __name__ == "__main__":
    _check_einlang_pads_explicit_rank2()
    print("ok: conv_ml pads match std::ml::conv [begin*, end*] layout (rank2)")
    _check_finite_diff_rank2_group_stride_dilation()
    print("ok: rank2 VJP vs finite diff (group=2, stride, dilation, asymmetric pads)")
    _check_rank1()
    print("ok: rank1 dx, dw, db (dy=ones) match test_autodiff_pass")
    _check_rank2()
    print("ok: rank2 dw, db (dy=ones); dx matches test ref with dy one-hot first; full dy overlap dx checked")
    _check_rank3()
    print("ok: rank3 dw, db (dy=ones); dx matches test ref with dy one-hot first; full dy gives dx=ones")
