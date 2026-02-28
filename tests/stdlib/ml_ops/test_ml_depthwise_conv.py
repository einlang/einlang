#!/usr/bin/env python3
"""
Tests for std::ml::depthwise_conv. Split from test_ml_convolution.
"""

import numpy as np
from tests.test_utils import compile_and_execute


def test_depthwise_conv(compiler, runtime):
    """Test depthwise_conv operation with all scenarios"""
    source = """use std::ml;
    let x_depthwise = [[[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]]];
    let w_depthwise = [[[[1.0, 0.0], [0.0, 1.0]]]];
    let b_depthwise = [0.0];
    let result_depthwise = std::ml::depthwise_conv(x_depthwise, w_depthwise, b_depthwise, [1, 1], [0, 0], [1, 1]);
    let x_depthwise_multi = [[[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]];
    let w_depthwise_multi = [[[[1.0, 0.0], [0.0, 1.0]]], [[[0.5, 0.5], [0.5, 0.5]]]];
    let b_depthwise_multi = [0.0, 1.0];
    let result_depthwise_multi = std::ml::depthwise_conv(x_depthwise_multi, w_depthwise_multi, b_depthwise_multi, [1, 1], [0, 0], [1, 1]);
    let x_depthwise_pad = [[[[1.0, 2.0], [3.0, 4.0]]]];
    let w_depthwise_pad = [[[[1.0, 1.0], [1.0, 1.0]]]];
    let b_depthwise_pad = [0.0];
    let result_depthwise_pad = std::ml::depthwise_conv(x_depthwise_pad, w_depthwise_pad, b_depthwise_pad, [1, 1], [1, 1], [1, 1]);
    let x_depthwise_stride = [[[[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0], [13.0, 14.0, 15.0, 16.0]]]];
    let w_depthwise_stride = [[[[1.0, 0.0], [0.0, 1.0]]]];
    let b_depthwise_stride = [0.0];
    let result_depthwise_stride = std::ml::depthwise_conv(x_depthwise_stride, w_depthwise_stride, b_depthwise_stride, [2, 2], [0, 0], [1, 1]);
    """
    result = compile_and_execute(source, compiler, runtime)
    assert result.success, f"Execution failed: {result.errors}"

    # Verify depthwise_conv - Basic
    x_depthwise = np.array([[[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]]], dtype=np.float32)
    w_depthwise = np.array([[[[1.0, 0.0], [0.0, 1.0]]]], dtype=np.float32)
    b_depthwise = np.array([0.0], dtype=np.float32)
    expected_depthwise = np.zeros((1, 1, 2, 2), dtype=np.float32)
    for i in range(2):
        for j in range(2):
            expected_depthwise[0, 0, i, j] = np.sum(x_depthwise[0, 0, i:i+2, j:j+2] * w_depthwise[0, 0]) + b_depthwise[0]
    actual_depthwise = np.array(result.outputs['result_depthwise'])
    np.testing.assert_allclose(actual_depthwise, expected_depthwise, rtol=1e-5)

    # Verify depthwise_conv - Multi-channel
    x_depthwise_multi = np.array([[[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]], dtype=np.float32)
    w_depthwise_multi = np.array([[[[1.0, 0.0], [0.0, 1.0]]], [[[0.5, 0.5], [0.5, 0.5]]]], dtype=np.float32)
    b_depthwise_multi = np.array([0.0, 1.0], dtype=np.float32)
    expected_depthwise_multi = np.array([[[[5.0]], [[14.0]]]], dtype=np.float32)
    actual_depthwise_multi = np.array(result.outputs['result_depthwise_multi'])
    np.testing.assert_allclose(actual_depthwise_multi, expected_depthwise_multi, rtol=1e-5)

    # Verify depthwise_conv - With padding
    x_depthwise_pad = np.array([[[[1.0, 2.0], [3.0, 4.0]]]], dtype=np.float32)
    w_depthwise_pad = np.array([[[[1.0, 1.0], [1.0, 1.0]]]], dtype=np.float32)
    b_depthwise_pad = np.array([0.0], dtype=np.float32)
    expected_depthwise_pad = np.zeros((1, 1, 3, 3), dtype=np.float32)
    pad_h, pad_w = 1, 1
    stride_h, stride_w = 1, 1
    for i in range(3):
        for j in range(3):
            sum_val = 0.0
            for kh in range(2):
                for kw in range(2):
                    h_in = i * stride_h - pad_h + kh
                    w_in = j * stride_w - pad_w + kw
                    if 0 <= h_in < 2 and 0 <= w_in < 2:
                        sum_val += x_depthwise_pad[0, 0, h_in, w_in] * w_depthwise_pad[0, 0, kh, kw]
            expected_depthwise_pad[0, 0, i, j] = sum_val + b_depthwise_pad[0]
    actual_depthwise_pad = np.array(result.outputs['result_depthwise_pad'])
    np.testing.assert_allclose(actual_depthwise_pad, expected_depthwise_pad, rtol=1e-5)

    # Verify depthwise_conv - With stride
    x_depthwise_stride = np.array([[[[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0], [13.0, 14.0, 15.0, 16.0]]]], dtype=np.float32)
    w_depthwise_stride = np.array([[[[1.0, 0.0], [0.0, 1.0]]]], dtype=np.float32)
    b_depthwise_stride = np.array([0.0], dtype=np.float32)
    expected_depthwise_stride = np.zeros((1, 1, 2, 2), dtype=np.float32)
    for i in range(2):
        for j in range(2):
            row = i * 2
            col = j * 2
            expected_depthwise_stride[0, 0, i, j] = np.sum(x_depthwise_stride[0, 0, row:row+2, col:col+2] * w_depthwise_stride[0, 0]) + b_depthwise_stride[0]
    actual_depthwise_stride = np.array(result.outputs['result_depthwise_stride'])
    np.testing.assert_allclose(actual_depthwise_stride, expected_depthwise_stride, rtol=1e-5)
