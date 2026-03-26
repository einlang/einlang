#!/usr/bin/env python3
"""Compare examples/mnist/train_one_step.ein against a NumPy reference step."""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import numpy as np

DT = np.float32


def _run_einlang(repo_root: Path) -> tuple[float, float]:
    mnist_dir = repo_root / "examples" / "mnist"
    cmd = [
        str(repo_root / ".venv" / "bin" / "python"),
        "-m",
        "einlang",
        "train_one_step.ein",
    ]
    env = dict(**__import__("os").environ)
    env["PYTHONPATH"] = str(repo_root / "src")
    proc = subprocess.run(
        cmd,
        cwd=str(mnist_dir),
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    nums = [float(x) for x in re.findall(r"[-+]?(?:\d+\.\d*|\d*\.\d+|\d+)", proc.stdout)]
    if len(nums) < 2:
        raise RuntimeError(f"Could not parse two losses from einlang output:\\n{proc.stdout}")
    return nums[0], nums[1]


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
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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


def _maxpool2x2(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
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


def _maxpool2x2_backward(dout: np.ndarray, arg: np.ndarray, in_shape: tuple[int, int, int, int]) -> np.ndarray:
    n, c, h, w = in_shape
    dx = np.zeros((n, c, h, w), dtype=DT)
    for bi in range(n):
        for ch in range(c):
            for oh in range(dout.shape[2]):
                for ow in range(dout.shape[3]):
                    hh, ww = arg[bi, ch, oh, ow]
                    dx[bi, ch, hh, ww] = DT(dx[bi, ch, hh, ww] + dout[bi, ch, oh, ow])
    return dx


def _run_numpy() -> tuple[float, float]:
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
    loss0 = float(np.sum((logits - y) ** 2, dtype=np.float64).astype(DT))

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
    _, d_conv1_w, d_conv1_b = _conv2d_backward(dc1, x, conv1_w, pad=1, stride=1)

    lr = DT(1e-3)
    conv1_w1 = (conv1_w - lr * d_conv1_w).astype(DT)
    conv1_b1 = (conv1_b - lr * d_conv1_b).astype(DT)
    conv2_w1 = (conv2_w - lr * d_conv2_w).astype(DT)
    conv2_b1 = (conv2_b - lr * d_conv2_b).astype(DT)
    fc_w1 = (fc_w - lr * d_fc_w).astype(DT)
    fc_b1 = (fc_b - lr * d_fc_b).astype(DT)

    c1_1 = _conv2d_nchw(x, conv1_w1, conv1_b1, pad=1, stride=1)
    r1_1 = np.maximum(c1_1, DT(0.0))
    p1_1, _ = _maxpool2x2(r1_1)
    c2_1 = _conv2d_nchw(p1_1, conv2_w1, conv2_b1, pad=1, stride=1)
    r2_1 = np.maximum(c2_1, DT(0.0))
    p2_1, _ = _maxpool2x2(r2_1)
    flat1 = p2_1.reshape(8).astype(DT)
    logits1 = (flat1 @ fc_w1 + fc_b1).astype(DT)
    loss1 = float(np.sum((logits1 - y) ** 2, dtype=np.float64).astype(DT))
    return loss0, loss1


def main() -> int:
    repo_root = Path(__file__).resolve().parents[2]
    ein_loss0, ein_loss1 = _run_einlang(repo_root)
    np_loss0, np_loss1 = _run_numpy()
    print(f"einlang loss0={ein_loss0:.10f} loss1={ein_loss1:.10f}")
    print(f"numpy   loss0={np_loss0:.10f} loss1={np_loss1:.10f}")
    print(f"delta   loss0={abs(ein_loss0 - np_loss0):.10f} loss1={abs(ein_loss1 - np_loss1):.10f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
