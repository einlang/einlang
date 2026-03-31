"""NumPy reference: 1D conv VJP with cotangent dy = ones on y (matches test_autodiff_pass rank1 setup).

Forward (valid conv, stride 1, dilation 1, no pad): for batch B, channels, length L, kernel K,
  y[b, co, t] = sum_{cl,m} Xp[b, f(co,cl), t+m] * W[co, cl, m] + B[co]
with Xp the padded input. Chain rule (Zygote-style): for fixed dy, scatter dy * flipped W onto dx
at positions that contributed to each y[t].
"""

import numpy as np


def conv1d_forward_reference(x, w, b, stride=1, dilation=1):
    x = np.asarray(x, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    assert x.ndim == 3 and w.ndim == 3
    _, c_in, w_in = x.shape
    c_out, cpg, kernel_w = w.shape
    assert cpg == c_in
    w_p = w_in
    y_len = (w_p - (kernel_w - 1) * dilation + stride - 1) // stride
    y = np.zeros((x.shape[0], c_out, y_len), dtype=np.float64)
    for b_ in range(x.shape[0]):
        for co in range(c_out):
            for t in range(y_len):
                s = float(b[co])
                for cl in range(cpg):
                    for m in range(kernel_w):
                        pos = t * stride + m * dilation
                        if 0 <= pos < w_p:
                            s += x[b_, cl, pos] * w[co, cl, m]
                y[b_, co, t] = s
    return y


def conv1d_vjp_x_reference(x, w, dy, stride=1, dilation=1):
    x = np.asarray(x, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)
    dy = np.asarray(dy, dtype=np.float64)
    _, c_in, w_in = x.shape
    c_out, cpg, kernel_w = w.shape
    w_p = w_in
    dx = np.zeros_like(x)
    for b_ in range(x.shape[0]):
        for co in range(c_out):
            for t in range(dy.shape[2]):
                for cl in range(cpg):
                    for m in range(kernel_w):
                        pos = t * stride + m * dilation
                        if 0 <= pos < w_p:
                            dx[b_, cl, pos] += dy[b_, co, t] * w[co, cl, m]
    return dx


if __name__ == "__main__":
    x = np.array([[[1.0, 2.0, 3.0, 4.0]]])
    w = np.array([[[1.0, 0.5]]])
    b = np.array([0.0])
    y = conv1d_forward_reference(x, w, b)
    dy = np.ones_like(y)
    dx = conv1d_vjp_x_reference(x, w, dy)
    ref_dx = np.array([[[1.0, 1.5, 1.5, 0.5]]])
    assert np.allclose(dx, ref_dx), (dx, ref_dx)
    print("ok: dx matches test_autodiff_pass conv rank1 ref_dx")
