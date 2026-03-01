#!/usr/bin/env python3
"""
Tests for std::ml::conv_transpose. Split from test_ml_convolution.
"""

import numpy as np
from tests.test_utils import compile_and_execute


def test_conv_transpose_1d(compiler, runtime):
    """Test conv_transpose 1D."""
    source = """use std::ml;
    let x = [[[1.0, 2.0, 3.0]]];
    let w = [[[1.0, 0.5]]];
    let b = [0.0];
    let result = std::ml::conv_transpose(x, w, b, [2], [0], [0]);
    """
    result = compile_and_execute(source, compiler, runtime)
    assert result.success, f"Execution failed: {result.errors}"
    expected = np.array([[[1.0, 0.5, 2.0, 1.0, 3.0, 1.5]]], dtype=np.float32)
    np.testing.assert_allclose(np.array(result.outputs['result']), expected, rtol=1e-5)


def test_conv_transpose_2d(compiler, runtime):
    """Test conv_transpose 2D."""
    source = """use std::ml;
    let x = [[[[1.0, 2.0], [3.0, 4.0]]]];
    let w = [[[[1.0, 0.5], [0.5, 1.0]]]];
    let b = [0.0];
    let result = std::ml::conv_transpose(x, w, b, [2, 2], [0, 0], [0, 0]);
    """
    result = compile_and_execute(source, compiler, runtime)
    assert result.success, f"Execution failed: {result.errors}"
    expected = np.array([[[[1.0, 0.5, 2.0, 1.0],
                           [0.5, 1.0, 1.0, 2.0],
                           [3.0, 1.5, 4.0, 2.0],
                           [1.5, 3.0, 2.0, 4.0]]]], dtype=np.float32)
    np.testing.assert_allclose(np.array(result.outputs['result']), expected, rtol=1e-5)


def test_conv_transpose_3d(compiler, runtime):
    """Test conv_transpose 3D."""
    source = """use std::ml;
    let x = [[[[[1.0, 2.0], [3.0, 4.0]]]]];
    let w = [[[[[1.0, 0.5], [0.5, 1.0]], [[0.5, 0.5], [0.5, 0.5]]]]];
    let b = [0.0];
    let result = std::ml::conv_transpose(x, w, b, [1, 1, 1], [0, 0, 0], [0, 0, 0]);
    """
    result = compile_and_execute(source, compiler, runtime)
    assert result.success, f"Execution failed: {result.errors}"
    x_np = np.array([[[[[1.0, 2.0], [3.0, 4.0]]]]], dtype=np.float32)
    w_np = np.array([[[[[1.0, 0.5], [0.5, 1.0]], [[0.5, 0.5], [0.5, 0.5]]]]], dtype=np.float32)
    expected = np.zeros((1, 1, 2, 3, 3), dtype=np.float32)
    for d_out in range(2):
        for i_out in range(3):
            for j_out in range(3):
                s = 0.0
                for kd in range(2):
                    for kh in range(2):
                        for kw in range(2):
                            d_in = d_out - kd
                            i_in = i_out - kh
                            j_in = j_out - kw
                            if 0 <= d_in < 1 and 0 <= i_in < 2 and 0 <= j_in < 2:
                                s += x_np[0, 0, d_in, i_in, j_in] * w_np[0, 0, kd, kh, kw]
                expected[0, 0, d_out, i_out, j_out] = s
    np.testing.assert_allclose(np.array(result.outputs['result']), expected, rtol=1e-5)

