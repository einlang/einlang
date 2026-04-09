"""Reference NumPy prototype for masked conv weight pullback through relu + max-pool.

This mirrors the `test_conv_relu_pool_direct_tensor_weight_quotient_matches_numpy`
fixture and gives us a clean oracle before porting the logic back into Einlang IR.
"""

from __future__ import annotations

import numpy as np


def conv2d_same_masked_forward(x: np.ndarray, w: np.ndarray, b: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    n_batch, c_in, h_in, w_in = x.shape
    c_out, c_in_w, kh_size, kw_size = w.shape
    assert c_in == c_in_w
    out = np.zeros((n_batch, c_out, h_in, w_in), dtype=np.float64)
    for n in range(n_batch):
        for co in range(c_out):
            for h in range(h_in):
                for ww in range(w_in):
                    acc = float(b[co])
                    for ci in range(c_in):
                        for kh in range(kh_size):
                            for kw in range(kw_size):
                                ih = h + kh - 1
                                iw = ww + kw - 1
                                if 0 <= ih < h_in and 0 <= iw < w_in:
                                    acc += x[n, ci, ih, iw] * w[co, ci, kh, kw]
                    out[n, co, h, ww] = acc
    return out


def relu_max_pool_seed(c0: np.ndarray) -> np.ndarray:
    relu = np.maximum(c0, 0.0)
    n_batch, c_out, h_in, w_in = relu.shape
    out_h = h_in // 2
    out_w = w_in // 2
    d_c0 = np.zeros_like(c0, dtype=np.float64)
    for n in range(n_batch):
        for co in range(c_out):
            for ph in range(out_h):
                for pw in range(out_w):
                    h0 = ph * 2
                    w0 = pw * 2
                    window = relu[n, co, h0 : h0 + 2, w0 : w0 + 2]
                    flat_idx = int(np.argmax(window))
                    wh, ww = divmod(flat_idx, window.shape[1])
                    d_c0[n, co, h0 + wh, w0 + ww] += 1.0
    d_c0 *= (c0 > 0).astype(np.float64)
    return d_c0


def conv2d_same_masked_weight_pullback(x: np.ndarray, d_c0: np.ndarray, weight_shape: tuple[int, ...]) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    d_c0 = np.asarray(d_c0, dtype=np.float64)
    d_w = np.zeros(weight_shape, dtype=np.float64)
    n_batch, c_in, h_in, w_in = x.shape
    c_out, _, kh_size, kw_size = weight_shape
    for n in range(n_batch):
        for co in range(c_out):
            for ci in range(c_in):
                for kh in range(kh_size):
                    for kw in range(kw_size):
                        for h in range(h_in):
                            for ww in range(w_in):
                                ih = h + kh - 1
                                iw = ww + kw - 1
                                if 0 <= ih < h_in and 0 <= iw < w_in:
                                    d_w[co, ci, kh, kw] += d_c0[n, co, h, ww] * x[n, ci, ih, iw]
    return d_w


def prototype_reference() -> np.ndarray:
    x = np.zeros((1, 1, 4, 4), dtype=np.float64)
    for h in range(4):
        for w in range(4):
            x[0, 0, h, w] = (h * 4 + w + 1) / 16.0
    w0 = np.zeros((1, 1, 3, 3), dtype=np.float64)
    for kh in range(3):
        for kw in range(3):
            w0[0, 0, kh, kw] = 0.1 * (1.0 + kh + kw)
    b0 = np.array([0.05], dtype=np.float64)

    c0 = conv2d_same_masked_forward(x, w0, b0)
    d_c0 = relu_max_pool_seed(c0)
    return conv2d_same_masked_weight_pullback(x, d_c0, w0.shape)


if __name__ == "__main__":
    actual = prototype_reference()
    ref = np.array(
        [[[[0.875, 1.125, 1.375], [1.875, 2.125, 2.375], [2.875, 3.125, 3.375]]]],
        dtype=np.float64,
    )
    assert np.allclose(actual, ref), (actual, ref)
    print("ok: masked conv weight pullback matches reference")
