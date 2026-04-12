#!/usr/bin/env python3
"""Leaf-gradient comparison: Einlang direct tensor quotients vs NumPy."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import numpy as np


DT = np.float32

GRADIENT_KEYS = (
    "d_loss_d_x",
    "d_loss_d_y",
    "d_loss_d_c1_0",
    "d_loss_d_p1_0",
    "d_loss_d_c2_0",
    "d_loss_d_p2_0",
    "d_loss_d_flat0",
    "d_loss_d_logits0",
    "d_loss_d_conv1_w",
    "d_loss_d_conv1_b",
    "d_loss_d_conv2_w",
    "d_loss_d_conv2_b",
    "d_loss_d_fc_w",
    "d_loss_d_fc_b",
)

LEAF_GRADIENT_KEYS = (
    "d_loss_d_x",
    "d_loss_d_y",
    "d_loss_d_conv1_w",
    "d_loss_d_conv1_b",
    "d_loss_d_conv2_w",
    "d_loss_d_conv2_b",
    "d_loss_d_fc_w",
    "d_loss_d_fc_b",
)

GRADIENT_BINDINGS = {
    "d_loss_d_x": "let d_loss_d_x = @loss0_scalar / @x;",
    "d_loss_d_y": "let d_loss_d_y = @loss0_scalar / @y;",
    "d_loss_d_c1_0": "let d_loss_d_c1_0 = @loss0_scalar / @c1_0;",
    "d_loss_d_p1_0": "let d_loss_d_p1_0 = @loss0_scalar / @p1_0;",
    "d_loss_d_c2_0": "let d_loss_d_c2_0 = @loss0_scalar / @c2_0;",
    "d_loss_d_p2_0": "let d_loss_d_p2_0 = @loss0_scalar / @p2_0;",
    "d_loss_d_flat0": "let d_loss_d_flat0 = @loss0_scalar / @flat0;",
    "d_loss_d_logits0": "let d_loss_d_logits0 = @loss0_scalar / @logits0;",
    "d_loss_d_conv1_w": "let d_loss_d_conv1_w = @loss0_scalar / @conv1_w0;",
    "d_loss_d_conv1_b": "let d_loss_d_conv1_b = @loss0_scalar / @conv1_b0;",
    "d_loss_d_conv2_w": "let d_loss_d_conv2_w = @loss0_scalar / @conv2_w0;",
    "d_loss_d_conv2_b": "let d_loss_d_conv2_b = @loss0_scalar / @conv2_b0;",
    "d_loss_d_fc_w": "let d_loss_d_fc_w = @loss0_scalar / @fc_w0;",
    "d_loss_d_fc_b": "let d_loss_d_fc_b = @loss0_scalar / @fc_b0;",
}


def _run_einlang_gradients(
    repo_root: Path, gradient_keys: tuple[str, ...] = GRADIENT_KEYS
) -> Dict[str, np.ndarray]:
    import sys

    sys.path.insert(0, str(repo_root / "src"))
    from einlang.compiler.driver import CompilerDriver
    from einlang.runtime.runtime import EinlangRuntime

    requested = tuple(gradient_keys)
    unknown = [key for key in requested if key not in GRADIENT_BINDINGS]
    if unknown:
        raise KeyError(f"Unknown gradient keys: {unknown}")

    mnist_dir = repo_root / "examples" / "mnist"
    source_file = mnist_dir / "train_one_step.ein"
    full_source = source_file.read_text(encoding="utf-8")
    marker = "let loss0_scalar = loss0[0];"
    prefix, found, _ = full_source.partition(marker)
    if not found:
        raise RuntimeError(f"Could not find marker {marker!r} in {source_file}")
    source = prefix + marker + "\n" + "\n".join(GRADIENT_BINDINGS[key] for key in requested) + "\n"

    compiler = CompilerDriver()
    runtime = EinlangRuntime()
    result = compiler.compile(source, str(source_file), root_path=mnist_dir)
    if not result.success:
        raise RuntimeError("Compilation failed")
    exec_result = runtime.execute(result)
    if exec_result.error is not None:
        raise RuntimeError(str(exec_result.error))

    out = exec_result.outputs or {}
    return {key: np.asarray(out[key], dtype=np.float64) for key in requested}


def _conv2d_nchw(x: np.ndarray, w: np.ndarray, b: np.ndarray, pad: int = 1, stride: int = 1) -> np.ndarray:
    n, cin, h, wid = x.shape
    cout, cin2, kh, kw = w.shape
    assert cin == cin2
    xpad = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode="constant")
    hout = (h + 2 * pad - kh) // stride + 1
    wout = (wid + 2 * pad - kw) // stride + 1
    out = np.zeros((n, cout, hout, wout), dtype=DT)
    for bi in range(n):
        for co in range(cout):
            for oh in range(hout):
                ih = oh * stride
                for ow in range(wout):
                    iw = ow * stride
                    s = DT(0.0)
                    for ci in range(cin):
                        for ki in range(kh):
                            for kj in range(kw):
                                s = DT(s + xpad[bi, ci, ih + ki, iw + kj] * w[co, ci, ki, kj])
                    out[bi, co, oh, ow] = DT(s + b[co])
    return out


def _conv2d_backward(
    dout: np.ndarray, x: np.ndarray, w: np.ndarray, pad: int = 1, stride: int = 1
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    n, cin, h, wid = x.shape
    cout, _, kh, kw = w.shape
    _, _, hout, wout = dout.shape
    xpad = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode="constant")
    dxpad = np.zeros_like(xpad, dtype=DT)
    dw = np.zeros_like(w, dtype=DT)
    db = np.zeros((cout,), dtype=DT)
    for bi in range(n):
        for co in range(cout):
            for oh in range(hout):
                ih = oh * stride
                for ow in range(wout):
                    iw = ow * stride
                    g = dout[bi, co, oh, ow]
                    db[co] = DT(db[co] + g)
                    for ci in range(cin):
                        for ki in range(kh):
                            for kj in range(kw):
                                dw[co, ci, ki, kj] = DT(dw[co, ci, ki, kj] + xpad[bi, ci, ih + ki, iw + kj] * g)
                                dxpad[bi, ci, ih + ki, iw + kj] = DT(dxpad[bi, ci, ih + ki, iw + kj] + w[co, ci, ki, kj] * g)
    dx = dxpad[:, :, pad : pad + h, pad : pad + wid]
    return dx, dw, db


def _maxpool2x2(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    n, c, h, w = x.shape
    out = np.zeros((n, c, h // 2, w // 2), dtype=DT)
    arg = np.zeros((n, c, h // 2, w // 2, 2), dtype=np.int32)
    for bi in range(n):
        for ch in range(c):
            for oh in range(h // 2):
                for ow in range(w // 2):
                    hs, ws = oh * 2, ow * 2
                    patch = x[bi, ch, hs : hs + 2, ws : ws + 2]
                    idx = int(np.argmax(patch))
                    ph, pw = divmod(idx, 2)
                    out[bi, ch, oh, ow] = patch[ph, pw]
                    arg[bi, ch, oh, ow] = (hs + ph, ws + pw)
    return out, arg


def _maxpool2x2_backward(dout: np.ndarray, arg: np.ndarray, in_shape: Tuple[int, int, int, int]) -> np.ndarray:
    n, c, h, w = in_shape
    dx = np.zeros((n, c, h, w), dtype=DT)
    for bi in range(n):
        for ch in range(c):
            for oh in range(dout.shape[2]):
                for ow in range(dout.shape[3]):
                    hh, ww = arg[bi, ch, oh, ow]
                    dx[bi, ch, hh, ww] = DT(dx[bi, ch, hh, ww] + dout[bi, ch, oh, ow])
    return dx


def _run_numpy_gradients(
    gradient_keys: tuple[str, ...] = LEAF_GRADIENT_KEYS,
) -> Dict[str, np.ndarray]:
    x = np.zeros((1, 1, 8, 8), dtype=DT)
    for h in range(8):
        for w in range(8):
            x[0, 0, h, w] = DT((h * 8 + w + 1) / 64.0)
    y = np.zeros((10,), dtype=DT)
    y[3] = DT(1.0)

    conv1_w = np.zeros((2, 1, 3, 3), dtype=DT)
    for co in range(2):
        for ci in range(1):
            for kh in range(3):
                for kw in range(3):
                    conv1_w[co, ci, kh, kw] = DT(0.02 * (1.0 + co + kh + kw))
    conv1_b = np.full((2,), DT(0.05), dtype=DT)

    conv2_w = np.zeros((2, 2, 3, 3), dtype=DT)
    for co in range(2):
        for ci in range(2):
            for kh in range(3):
                for kw in range(3):
                    conv2_w[co, ci, kh, kw] = DT(0.015 * (1.0 + co + ci + kh + kw))
    conv2_b = np.full((2,), DT(0.03), dtype=DT)

    fc_w = np.zeros((8, 10), dtype=DT)
    for i in range(8):
        for j in range(10):
            fc_w[i, j] = DT(0.01 * (1.0 + i + j))
    fc_b = np.zeros((10,), dtype=DT)

    c1 = _conv2d_nchw(x, conv1_w, conv1_b, pad=1, stride=1)
    r1 = np.maximum(c1, DT(0.0))
    p1, p1_arg = _maxpool2x2(r1)

    c2 = _conv2d_nchw(p1, conv2_w, conv2_b, pad=1, stride=1)
    r2 = np.maximum(c2, DT(0.0))
    p2, p2_arg = _maxpool2x2(r2)

    flat = p2.reshape(8).astype(DT)
    logits = (flat @ fc_w + fc_b).astype(DT)
    dlogits = (DT(2.0) * (logits - y)).astype(DT)

    d_fc_w = np.outer(flat, dlogits).astype(DT)
    d_fc_b = dlogits.copy()
    dflat = (fc_w @ dlogits).astype(DT)
    dp2 = dflat.reshape(1, 2, 2, 2).astype(DT)

    dr2 = _maxpool2x2_backward(dp2, p2_arg, r2.shape)
    dc2 = (dr2 * (c2 > 0).astype(DT)).astype(DT)
    dp1, d_conv2_w, d_conv2_b = _conv2d_backward(dc2, p1, conv2_w, pad=1, stride=1)

    dr1 = _maxpool2x2_backward(dp1, p1_arg, r1.shape)
    dc1 = (dr1 * (c1 > 0).astype(DT)).astype(DT)
    d_x, d_conv1_w, d_conv1_b = _conv2d_backward(dc1, x, conv1_w, pad=1, stride=1)
    d_y = (-dlogits).astype(DT)

    results = {
        "d_loss_d_x": np.asarray(d_x, dtype=np.float64),
        "d_loss_d_y": np.asarray(d_y, dtype=np.float64),
        "d_loss_d_c1_0": np.asarray(dc1, dtype=np.float64),
        "d_loss_d_p1_0": np.asarray(dp1, dtype=np.float64),
        "d_loss_d_c2_0": np.asarray(dc2, dtype=np.float64),
        "d_loss_d_p2_0": np.asarray(dp2, dtype=np.float64),
        "d_loss_d_flat0": np.asarray(dflat, dtype=np.float64),
        "d_loss_d_logits0": np.asarray(dlogits, dtype=np.float64),
        "d_loss_d_conv1_w": np.asarray(d_conv1_w, dtype=np.float64),
        "d_loss_d_conv1_b": np.asarray(d_conv1_b, dtype=np.float64),
        "d_loss_d_conv2_w": np.asarray(d_conv2_w, dtype=np.float64),
        "d_loss_d_conv2_b": np.asarray(d_conv2_b, dtype=np.float64),
        "d_loss_d_fc_w": np.asarray(d_fc_w, dtype=np.float64),
        "d_loss_d_fc_b": np.asarray(d_fc_b, dtype=np.float64),
    }
    return {key: results[key] for key in gradient_keys}


def _print_entrywise(name: str, a: np.ndarray, b: np.ndarray) -> None:
    if a.shape != b.shape:
        print(f"{name}: shape mismatch autodiff={a.shape} numpy={b.shape}")
        return
    diff = np.abs(a - b)
    print(f"\\n{name}: shape={a.shape} max_abs_diff={diff.max():.10f} mean_abs_diff={diff.mean():.10f}")
    for idx in np.ndindex(a.shape):
        av = float(a[idx])
        bv = float(b[idx])
        dv = abs(av - bv)
        print(f"  {name}{idx}: autodiff={av:.10f} numpy={bv:.10f} diff={dv:.10f}")


def main() -> int:
    repo_root = Path(__file__).resolve().parents[2]
    ad = _run_einlang_gradients(repo_root)
    npg = _run_numpy_gradients()
    for key in GRADIENT_KEYS:
        _print_entrywise(key, ad[key], npg[key])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
