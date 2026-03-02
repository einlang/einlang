#!/usr/bin/env python3
"""Quantize MNIST float32 weights to symmetric int8 + per-tensor scale.
   Calibrate activation scales by running float forward on sample images.

Reads weights from ../mnist/weights/, quantizes each weight tensor to int8
using symmetric quantization (zero_point = 0), and writes:
  - weights/<name>_q.npy   (int8 quantized values)
  - weights/<name>_s.npy   (f32 per-tensor scale)
  - weights/<name>.npy     (f32 biases, copied unchanged)
  - weights/act1_s.npy    (f32 scale for activations after conv1+relu)
  - weights/flat_s.npy    (f32 scale for flattened features before FC)
"""
import numpy as np
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "mnist" / "weights"
DST = Path(__file__).resolve().parent / "weights"
SAMPLES = Path(__file__).resolve().parent / "samples"
DST.mkdir(exist_ok=True)

WEIGHT_NAMES = ["conv1_w", "conv2_w", "fc_w"]
BIAS_NAMES = ["conv1_b", "conv2_b", "fc_b"]


def conv2d_np(x, w, b, stride, pad):
    n, c_in, h, w_in = x.shape
    c_out, _, kh, kw = w.shape
    ph, pw = pad, pad
    h_out = (h + 2 * ph - kh) // stride + 1
    w_out = (w_in + 2 * pw - kw) // stride + 1
    xp = np.pad(x, ((0, 0), (0, 0), (ph, ph), (pw, pw)), constant_values=0.0)
    out = np.zeros((n, c_out, h_out, w_out), dtype=np.float32)
    for i in range(h_out):
        for j in range(w_out):
            patch = xp[:, :, i * stride : i * stride + kh, j * stride : j * stride + kw]
            out[:, :, i, j] = np.tensordot(patch, w, axes=([1, 2, 3], [1, 2, 3])) + b
    return out


def max_pool_np(x, kernel, stride):
    _, c, h, w = x.shape
    kh, kw = kernel
    h_out = (h - kh) // stride + 1
    w_out = (w - kw) // stride + 1
    out = np.zeros((x.shape[0], c, h_out, w_out), dtype=np.float32)
    for i in range(h_out):
        for j in range(w_out):
            out[:, :, i, j] = np.max(
                x[:, :, i * stride : i * stride + kh, j * stride : j * stride + kw],
                axis=(2, 3),
            )
    return out


def forward_float(images, conv1_w, conv1_b, conv2_w, conv2_b, fc_w, fc_b):
    input_ = 1.0 - images
    c1 = conv2d_np(input_, conv1_w, conv1_b, 1, 2)
    c1 = np.maximum(c1, 0.0)
    p1 = max_pool_np(c1, (2, 2), 2)
    c2 = conv2d_np(p1, conv2_w, conv2_b, 1, 2)
    c2 = np.maximum(c2, 0.0)
    p2 = max_pool_np(c2, (3, 3), 3)
    flat = p2.reshape(p2.shape[0], -1)
    logits = flat @ fc_w + fc_b
    return c1, p1, c2, p2, flat, logits


for name in WEIGHT_NAMES:
    w = np.load(SRC / f"{name}.npy")
    amax = np.max(np.abs(w))
    scale = amax / 127.0 if amax > 0 else np.float32(1.0)
    q = np.asarray(np.clip(np.round(w / scale), -128, 127), dtype=np.int8)
    out_q = DST / f"{name}_q.npy"
    np.save(out_q, q)
    check = np.load(out_q)
    assert check.dtype == np.int8, f"{out_q.name} must be int8 in storage, got {check.dtype}"
    np.save(DST / f"{name}_s.npy", np.array([scale], dtype=np.float32))
    print(f"{name}: shape={w.shape}  scale={scale:.6e}  "
          f"max_err={np.max(np.abs(w - q.astype(np.float32) * scale)):.6e}")

for name in BIAS_NAMES:
    b = np.load(SRC / f"{name}.npy")
    np.save(DST / f"{name}.npy", b)
    print(f"{name}: shape={b.shape}  (copied as f32)")

conv1_w = np.load(SRC / "conv1_w.npy")
conv1_b = np.load(SRC / "conv1_b.npy")
conv2_w = np.load(SRC / "conv2_w.npy")
conv2_b = np.load(SRC / "conv2_b.npy")
fc_w = np.load(SRC / "fc_w.npy")
fc_b = np.load(SRC / "fc_b.npy")

if SAMPLES.exists():
    try:
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from pgm_io import load as load_pgm
        images = np.stack([load_pgm(str(SAMPLES / f"{i}.pgm")) for i in range(10)], axis=0)
        images = images.astype(np.float32).reshape(10, 1, 28, 28)
        c1, p1, c2, p2, flat, _ = forward_float(images, conv1_w, conv1_b, conv2_w, conv2_b, fc_w, fc_b)
        act1_max = np.max(np.abs(p1))
        flat_max = np.max(np.abs(flat))
        act1_scale = np.float32(act1_max / 127.0) if act1_max > 0 else np.float32(1.0)
        flat_scale = np.float32(flat_max / 127.0) if flat_max > 0 else np.float32(1.0)
        np.save(DST / "act1_s.npy", np.array([act1_scale], dtype=np.float32))
        np.save(DST / "flat_s.npy", np.array([flat_scale], dtype=np.float32))
        print(f"act1_scale={act1_scale:.6e}  flat_scale={flat_scale:.6e}")
    except Exception as e:
        print("Calibration skipped (need samples/ and pgm_io):", e)
        np.save(DST / "act1_s.npy", np.array([0.05], dtype=np.float32))
        np.save(DST / "flat_s.npy", np.array([0.1], dtype=np.float32))
else:
    np.save(DST / "act1_s.npy", np.array([0.05], dtype=np.float32))
    np.save(DST / "flat_s.npy", np.array([0.1], dtype=np.float32))
    print("Calibration skipped (no samples/); using default act scales")
