"""Vectorized numpy implementations of ML ops for the MNIST demo.

Called from mnist_onnx_arch.ein via python::numpy_ops::*.
Each function is a single numpy call so no Python loop overhead.
"""
import numpy as np
from numpy.lib.stride_tricks import as_strided


def conv2d(X: np.ndarray, W: np.ndarray, B: np.ndarray,
           stride_h: int, stride_w: int,
           pad_h: int, pad_w: int) -> np.ndarray:
    N, C_in, H, W_in = X.shape
    C_out, _, kH, kW = W.shape
    H_out = (H  + 2 * pad_h - kH) // stride_h + 1
    W_out = (W_in + 2 * pad_w - kW) // stride_w + 1

    Xp = np.pad(X, ((0, 0), (0, 0), (pad_h, pad_h), (pad_w, pad_w))) if (pad_h or pad_w) else X
    s = Xp.strides
    patches = as_strided(
        Xp,
        shape=(N, C_in, H_out, W_out, kH, kW),
        strides=(s[0], s[1], s[2] * stride_h, s[3] * stride_w, s[2], s[3]),
    )
    out = np.tensordot(patches, W, axes=([1, 4, 5], [1, 2, 3]))  # (N, H_out, W_out, C_out)
    out = out.transpose(0, 3, 1, 2) + B[np.newaxis, :, np.newaxis, np.newaxis]
    return out.astype(np.float32)


def relu(X: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, X, dtype=np.float32)


def max_pool2d(X: np.ndarray, kh: int, kw: int,
               sh: int, sw: int) -> np.ndarray:
    N, C, H, W = X.shape
    H_out = (H - kh) // sh + 1
    W_out = (W - kw) // sw + 1
    s = X.strides
    windows = as_strided(
        X,
        shape=(N, C, H_out, W_out, kh, kw),
        strides=(s[0], s[1], s[2] * sh, s[3] * sw, s[2], s[3]),
    )
    return windows.max(axis=(-2, -1)).astype(np.float32)


def linear(X: np.ndarray, W: np.ndarray, B: np.ndarray) -> np.ndarray:
    flat = X.reshape(X.shape[0], -1)
    return (flat @ W + B).astype(np.float32)


def argmax1d(X: np.ndarray) -> int:
    return int(np.argmax(X))
