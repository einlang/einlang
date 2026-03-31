"""Julia / Zygote-style NumPy reference for softmax and log-softmax pullbacks.

This file is the smallest executable oracle for the remaining softmax-family quotient work.

Conventions:

- ``softmax_vjp(x, dy)`` returns the pullback wrt ``x`` for a cotangent ``dy`` on ``y = softmax(x)``.
- ``log_softmax_vjp(x, dy)`` returns the pullback wrt ``x`` for a cotangent ``dy`` on ``y = log_softmax(x)``.
- For a scalar alias ``a00 = y[0, 0]``, the quotient ``@a00 / @x00`` should be interpreted as:
  seed a one-hot cotangent on ``y`` at ``(0, 0)``, run the pullback, then project the storage cotangent at ``x00``.

These are the ChainRules / Zygote formulas:

- softmax:
  ``dx = y * (dy - sum(dy * y, axis=-1, keepdims=True))``
- log_softmax:
  ``dx = dy - softmax(x) * sum(dy, axis=-1, keepdims=True)``
"""

from __future__ import annotations

import numpy as np


def softmax_forward(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    m = np.max(x, axis=axis, keepdims=True)
    e = np.exp(x - m)
    return e / np.sum(e, axis=axis, keepdims=True)


def log_softmax_forward(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    m = np.max(x, axis=axis, keepdims=True)
    e = np.exp(x - m)
    logsum = np.log(np.sum(e, axis=axis, keepdims=True)) + m
    return x - logsum


def softmax_vjp(x: np.ndarray, dy: np.ndarray, axis: int = -1) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    dy = np.asarray(dy, dtype=np.float64)
    y = softmax_forward(x, axis=axis)
    proj = np.sum(dy * y, axis=axis, keepdims=True)
    return y * (dy - proj)


def log_softmax_vjp(x: np.ndarray, dy: np.ndarray, axis: int = -1) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    dy = np.asarray(dy, dtype=np.float64)
    y = softmax_forward(x, axis=axis)
    return dy - y * np.sum(dy, axis=axis, keepdims=True)


def one_hot_like(x: np.ndarray, index: tuple[int, ...]) -> np.ndarray:
    dy = np.zeros_like(np.asarray(x, dtype=np.float64))
    dy[index] = 1.0
    return dy


def alias_quotient_softmax_scalar(x: np.ndarray, y_index: tuple[int, ...], x_index: tuple[int, ...]) -> float:
    dy = one_hot_like(softmax_forward(x), y_index)
    dx = softmax_vjp(x, dy)
    return float(dx[x_index])


def alias_quotient_log_softmax_scalar(x: np.ndarray, y_index: tuple[int, ...], x_index: tuple[int, ...]) -> float:
    dy = one_hot_like(log_softmax_forward(x), y_index)
    dx = log_softmax_vjp(x, dy)
    return float(dx[x_index])


if __name__ == "__main__":
    x = np.array([[1.0, 2.0, 3.0]], dtype=np.float64)

    # All-ones cotangent: softmax pullback is zero, log-softmax is not.
    dy_all = np.ones_like(x)
    dx_softmax = softmax_vjp(x, dy_all)
    dx_log_softmax = log_softmax_vjp(x, dy_all)
    ref_softmax = np.zeros_like(x)
    ref_log_softmax = 1.0 - 3.0 * softmax_forward(x)
    assert np.allclose(dx_softmax, ref_softmax), (dx_softmax, ref_softmax)
    assert np.allclose(dx_log_softmax, ref_log_softmax), (dx_log_softmax, ref_log_softmax)

    # Scalar alias quotient: a00 = y[0,0], x00 = x[0,0].
    g_soft = alias_quotient_softmax_scalar(x, (0, 0), (0, 0))
    g_log = alias_quotient_log_softmax_scalar(x, (0, 0), (0, 0))
    y = softmax_forward(x)
    ref_g_soft = float(y[0, 0] * (1.0 - y[0, 0]))
    ref_g_log = float(1.0 - y[0, 0])
    assert np.allclose(g_soft, ref_g_soft), (g_soft, ref_g_soft)
    assert np.allclose(g_log, ref_g_log), (g_log, ref_g_log)
    print("ok: softmax/log_softmax pullbacks and scalar alias projections match Julia-style formulas")
