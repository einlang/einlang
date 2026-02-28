#!/usr/bin/env python3
"""
Tests for std::ml::conv_transpose. Split from test_ml_convolution.
"""

import numpy as np
from tests.test_utils import compile_and_execute


def test_conv_transpose(compiler, runtime):
    """Test conv_transpose operation across all ranks (1D, 2D, 3D)"""
    source = """use std::ml;
    // ConvTranspose 1D
    let x_conv_transpose_1d = [[[1.0, 2.0, 3.0]]];
    let w_conv_transpose_1d = [[[1.0, 0.5]]];
    let b_conv_transpose_1d = [0.0];
    let strides_conv_transpose_1d = [2];
    let pads_conv_transpose_1d = [0];
    let output_padding_conv_transpose_1d = [0];
    let result_1d = std::ml::conv_transpose(x_conv_transpose_1d, w_conv_transpose_1d, b_conv_transpose_1d, strides_conv_transpose_1d, pads_conv_transpose_1d, output_padding_conv_transpose_1d);
    // ConvTranspose 2D
    let x_conv_transpose_2d = [[[[1.0, 2.0], [3.0, 4.0]]]];
    let w_conv_transpose_2d = [[[[1.0, 0.5], [0.5, 1.0]]]];
    let b_conv_transpose_2d = [0.0];
    let strides_conv_transpose_2d = [2, 2];
    let pads_conv_transpose_2d = [0, 0];
    let output_padding_conv_transpose_2d = [0, 0];
    let result_2d = std::ml::conv_transpose(x_conv_transpose_2d, w_conv_transpose_2d, b_conv_transpose_2d, strides_conv_transpose_2d, pads_conv_transpose_2d, output_padding_conv_transpose_2d);
    // ConvTranspose 3D
    let x_conv_transpose_3d = [[[[[1.0, 2.0], [3.0, 4.0]]]]];
    let w_conv_transpose_3d = [[[[[1.0, 0.5], [0.5, 1.0]], [[0.5, 0.5], [0.5, 0.5]]]]];
    let b_conv_transpose_3d = [0.0];
    let strides_conv_transpose_3d = [1, 1, 1];
    let pads_conv_transpose_3d = [0, 0, 0];
    let output_padding_conv_transpose_3d = [0, 0, 0];
    let result_3d = std::ml::conv_transpose(x_conv_transpose_3d, w_conv_transpose_3d, b_conv_transpose_3d, strides_conv_transpose_3d, pads_conv_transpose_3d, output_padding_conv_transpose_3d);
    """
    result = compile_and_execute(source, compiler, runtime)
    assert result.success, f"Execution failed: {result.errors}"

    # Verify conv_transpose 1D
    x_conv_transpose_1d = np.array([[[1.0, 2.0, 3.0]]], dtype=np.float32)
    w_conv_transpose_1d = np.array([[[1.0, 0.5]]], dtype=np.float32)
    expected_1d = np.array([[[1.0, 0.5, 2.0, 1.0, 3.0, 1.5]]], dtype=np.float32)
    actual_1d = np.array(result.outputs['result_1d'])
    np.testing.assert_allclose(actual_1d, expected_1d, rtol=1e-5)

    # Verify conv_transpose 2D
    x_conv_transpose_2d = np.array([[[[1.0, 2.0], [3.0, 4.0]]]], dtype=np.float32)
    w_conv_transpose_2d = np.array([[[[1.0, 0.5], [0.5, 1.0]]]], dtype=np.float32)
    expected_2d = np.array([[[[1.0, 0.5, 2.0, 1.0],
                              [0.5, 1.0, 1.0, 2.0],
                              [3.0, 1.5, 4.0, 2.0],
                              [1.5, 3.0, 2.0, 4.0]]]], dtype=np.float32)
    actual_2d = np.array(result.outputs['result_2d'])
    np.testing.assert_allclose(actual_2d, expected_2d, rtol=1e-5)

    # Verify conv_transpose 3D
    x_conv_transpose_3d = np.array([[[[[1.0, 2.0], [3.0, 4.0]]]]], dtype=np.float32)
    w_conv_transpose_3d = np.array([[[[[1.0, 0.5], [0.5, 1.0]], [[0.5, 0.5], [0.5, 0.5]]]]], dtype=np.float32)
    stride_d, stride_h, stride_w = 1, 1, 1
    pad_d, pad_h, pad_w = 0, 0, 0
    expected_3d = np.zeros((1, 1, 2, 3, 3), dtype=np.float32)
    for d_out in range(2):
        for i_out in range(3):
            for j_out in range(3):
                sum_val = 0.0
                for kd in range(2):
                    for kh in range(2):
                        for kw in range(2):
                            d_in = (d_out + pad_d - kd) / stride_d
                            i_in = (i_out + pad_h - kh) / stride_h
                            j_in = (j_out + pad_w - kw) / stride_w
                            if (d_out + pad_d - kd) % stride_d == 0 and \
                               (i_out + pad_h - kh) % stride_h == 0 and \
                               (j_out + pad_w - kw) % stride_w == 0 and \
                               0 <= d_in < 1 and 0 <= i_in < 2 and 0 <= j_in < 2:
                                d_in = int(d_in)
                                i_in = int(i_in)
                                j_in = int(j_in)
                                sum_val += x_conv_transpose_3d[0, 0, d_in, i_in, j_in] * w_conv_transpose_3d[0, 0, kd, kh, kw]
                expected_3d[0, 0, d_out, i_out, j_out] = sum_val
    actual_3d = np.array(result.outputs['result_3d'])
    np.testing.assert_allclose(actual_3d, expected_3d, rtol=1e-5)

