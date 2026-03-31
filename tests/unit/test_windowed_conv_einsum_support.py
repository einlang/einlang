"""Numerical checks for ND sliding-window + einsum used by conv fast paths."""

import numpy as np

from einlang.backends.numpy_expressions_support import _sliding_window_nd_view


def test_sliding_window_nd_view_matches_naive_conv2d():
    rng = np.random.default_rng(0)
    c_in, h, w = 2, 5, 6
    co, kh, kw = 3, 2, 3
    x = rng.standard_normal((c_in, h, w), dtype=np.float64)
    weight = rng.standard_normal((co, c_in, kh, kw), dtype=np.float64)
    stride_h, stride_w = 1, 2
    windows = _sliding_window_nd_view(
        x, axes=(1, 2), window_shape=(kh, kw), strides=(stride_h, stride_w)
    )
    assert windows is not None
    ho = (h - kh) // stride_h + 1
    wo = (w - kw) // stride_w + 1
    assert windows.shape == (c_in, ho, wo, kh, kw)
    out = np.einsum("cijmn,ocmn->oij", windows, weight, optimize=True)
    ref = np.empty((co, ho, wo), dtype=np.float64)
    for o in range(co):
        for i in range(ho):
            for j in range(wo):
                h0 = i * stride_h
                w0 = j * stride_w
                patch = x[:, h0 : h0 + kh, w0 : w0 + kw]
                ref[o, i, j] = np.sum(patch * weight[o])
    np.testing.assert_allclose(out, ref, rtol=1e-10, atol=1e-10)


def test_sliding_window_nd_view_batched_conv2d():
    rng = np.random.default_rng(1)
    bsz, c_in, h, w = 2, 2, 4, 5
    co, kh, kw = 2, 2, 2
    x = rng.standard_normal((bsz, c_in, h, w), dtype=np.float64)
    weight = rng.standard_normal((co, c_in, kh, kw), dtype=np.float64)
    windows = _sliding_window_nd_view(
        x, axes=(2, 3), window_shape=(kh, kw), strides=(1, 1)
    )
    assert windows is not None
    ho, wo = h - kh + 1, w - kw + 1
    assert windows.shape == (bsz, c_in, ho, wo, kh, kw)
    out = np.einsum("bcijmn,ocmn->boij", windows, weight, optimize=True)
    assert out.shape == (bsz, co, ho, wo)
