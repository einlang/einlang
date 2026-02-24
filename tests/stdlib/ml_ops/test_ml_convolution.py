#!/usr/bin/env python3
"""
Comprehensive accuracy tests for std::ml operations against ONNX/NumPy reference implementations.
Tests all operations added to ml.ein for correctness.
"""

import pytest
import numpy as np
try:
    import scipy.special
except ImportError:
    scipy = None
from tests.test_utils import compile_and_execute, assert_float_close
# ===========================================================================
# Activation Function Tests
# ===========================================================================



# ===========================================================================
# Convolution Operation Tests
# Clustered tests for efficiency - all convolution ops tested together
# Compile/execute errors are failures, not skips
# ===========================================================================

def test_conv(compiler, runtime):
    """Test conv operation across all ranks (1D, 2D, 3D)"""
    source = """use std::ml;
    # Conv 1D
    let x_conv_1d = [[[1.0, 2.0, 3.0, 4.0]]];
    let w_conv_1d = [[[1.0, 0.5]]];
    let b_conv_1d = [0.0];
    let result_1d = std::ml::conv(x_conv_1d, w_conv_1d, b_conv_1d, [1], [0], [1]);
    # Conv 2D
    let x_conv_2d = [[[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]]];
    let w_conv_2d = [[[[1.0, 0.0], [0.0, 1.0]]]];
    let b_conv_2d = [0.0];
    let result_2d = std::ml::conv(x_conv_2d, w_conv_2d, b_conv_2d, [1, 1], [0, 0], [1, 1]);
    # Conv 3D
    let x_conv_3d = [[[[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]]];
    let w_conv_3d = [[[[[1.0, 0.0], [0.0, 1.0]], [[0.5, 0.5], [0.5, 0.5]]]]];
    let b_conv_3d = [0.0];
    let result_3d = std::ml::conv(x_conv_3d, w_conv_3d, b_conv_3d, [1, 1, 1], [0, 0, 0], [1, 1, 1]);
    """
    result = compile_and_execute(source, compiler, runtime)
    assert result.success, f"Execution failed: {result.errors}"

    # Verify conv 1D
    x_conv_1d = np.array([[[1.0, 2.0, 3.0, 4.0]]], dtype=np.float32)
    w_conv_1d = np.array([[[1.0, 0.5]]], dtype=np.float32)
    expected_1d = np.array([[[2.0, 3.5, 5.0]]], dtype=np.float32)  # 1*1+2*0.5=2, 2*1+3*0.5=3.5, 3*1+4*0.5=5
    actual_1d = np.array(result.outputs['result_1d'])
    np.testing.assert_allclose(actual_1d, expected_1d, rtol=1e-5)

    # Verify conv 2D
    x_conv_2d = np.array([[[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]]], dtype=np.float32)
    w_conv_2d = np.array([[[[1.0, 0.0], [0.0, 1.0]]]], dtype=np.float32)
    b_conv_2d = np.array([0.0], dtype=np.float32)
    # Manual convolution calculation: 2x2 kernel on 3x3 input with stride 1, pad 0
    expected_2d = np.zeros((1, 1, 2, 2), dtype=np.float32)
    for i in range(2):
        for j in range(2):
            expected_2d[0, 0, i, j] = np.sum(x_conv_2d[0, 0, i:i+2, j:j+2] * w_conv_2d[0, 0]) + b_conv_2d[0]
    actual_2d = np.array(result.outputs['result_2d'])
    np.testing.assert_allclose(actual_2d, expected_2d, rtol=1e-5)

    # Verify conv 3D
    x_conv_3d = np.array([[[[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]]], dtype=np.float32)
    w_conv_3d = np.array([[[[[1.0, 0.0], [0.0, 1.0]], [[0.5, 0.5], [0.5, 0.5]]]]], dtype=np.float32)
    # Front layer: 1*1 + 2*0 + 3*0 + 4*1 = 5
    # Back layer: 5*0.5 + 6*0.5 + 7*0.5 + 8*0.5 = 13
    # Total: 5 + 13 = 18
    expected_3d = np.array([[[[[18.0]]]]], dtype=np.float32)
    actual_3d = np.array(result.outputs['result_3d'])
    np.testing.assert_allclose(actual_3d, expected_3d, rtol=1e-5)


def test_conv_transpose(compiler, runtime):
    """Test conv_transpose operation across all ranks (1D, 2D, 3D)"""
    source = """use std::ml;
    # ConvTranspose 1D
    let x_conv_transpose_1d = [[[1.0, 2.0, 3.0]]];
    let w_conv_transpose_1d = [[[1.0, 0.5]]];
    let b_conv_transpose_1d = [0.0];
    let strides_conv_transpose_1d = [2];
    let pads_conv_transpose_1d = [0];
    let output_padding_conv_transpose_1d = [0];
    let result_1d = std::ml::conv_transpose(x_conv_transpose_1d, w_conv_transpose_1d, b_conv_transpose_1d, strides_conv_transpose_1d, pads_conv_transpose_1d, output_padding_conv_transpose_1d);
    # ConvTranspose 2D
    let x_conv_transpose_2d = [[[[1.0, 2.0], [3.0, 4.0]]]];
    let w_conv_transpose_2d = [[[[1.0, 0.5], [0.5, 1.0]]]];
    let b_conv_transpose_2d = [0.0];
    let strides_conv_transpose_2d = [2, 2];
    let pads_conv_transpose_2d = [0, 0];
    let output_padding_conv_transpose_2d = [0, 0];
    let result_2d = std::ml::conv_transpose(x_conv_transpose_2d, w_conv_transpose_2d, b_conv_transpose_2d, strides_conv_transpose_2d, pads_conv_transpose_2d, output_padding_conv_transpose_2d);
    # ConvTranspose 3D
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
    # Transposed convolution 1D: 3-length input, 2-length kernel, stride 2
    # Output length = (input_length - 1) * stride + kernel_length = (3-1)*2 + 2 = 6
    # Manual calculation for stride 2 transposed conv:
    # Output positions: 0, 1, 2, 3, 4, 5
    # Input position i contributes to output positions: i*stride to i*stride + kernel_length - 1
    # Input[0]=1.0 -> output[0:2] += [1.0*1.0, 1.0*0.5] = [1.0, 0.5]
    # Input[1]=2.0 -> output[2:4] += [2.0*1.0, 2.0*0.5] = [2.0, 1.0]
    # Input[2]=3.0 -> output[4:6] += [3.0*1.0, 3.0*0.5] = [3.0, 1.5]
    expected_1d = np.array([[[1.0, 0.5, 2.0, 1.0, 3.0, 1.5]]], dtype=np.float32)
    actual_1d = np.array(result.outputs['result_1d'])
    np.testing.assert_allclose(actual_1d, expected_1d, rtol=1e-5)

    # Verify conv_transpose 2D
    x_conv_transpose_2d = np.array([[[[1.0, 2.0], [3.0, 4.0]]]], dtype=np.float32)
    w_conv_transpose_2d = np.array([[[[1.0, 0.5], [0.5, 1.0]]]], dtype=np.float32)
    b_conv_transpose_2d = np.array([0.0], dtype=np.float32)
    # Transposed convolution: 2x2 input, 2x2 kernel, stride 2, pad 0 -> 4x4 output
    # Manual calculation for stride 2 transposed conv 2D:
    # Input[0,0]=1.0 -> output[0:2, 0:2] += 1.0 * [[1.0, 0.5], [0.5, 1.0]] = [[1.0, 0.5], [0.5, 1.0]]
    # Input[0,1]=2.0 -> output[0:2, 2:4] += 2.0 * [[1.0, 0.5], [0.5, 1.0]] = [[2.0, 1.0], [1.0, 2.0]]
    # Input[1,0]=3.0 -> output[2:4, 0:2] += 3.0 * [[1.0, 0.5], [0.5, 1.0]] = [[3.0, 1.5], [1.5, 3.0]]
    # Input[1,1]=4.0 -> output[2:4, 2:4] += 4.0 * [[1.0, 0.5], [0.5, 1.0]] = [[4.0, 2.0], [2.0, 4.0]]
    expected_2d = np.array([[[[1.0, 0.5, 2.0, 1.0],
                              [0.5, 1.0, 1.0, 2.0],
                              [3.0, 1.5, 4.0, 2.0],
                              [1.5, 3.0, 2.0, 4.0]]]], dtype=np.float32)
    actual_2d = np.array(result.outputs['result_2d'])
    np.testing.assert_allclose(actual_2d, expected_2d, rtol=1e-5)

    # Verify conv_transpose 3D
    x_conv_transpose_3d = np.array([[[[[1.0, 2.0], [3.0, 4.0]]]]], dtype=np.float32)  # [1, 1, 1, 2, 2]
    w_conv_transpose_3d = np.array([[[[[1.0, 0.5], [0.5, 1.0]], [[0.5, 0.5], [0.5, 0.5]]]]], dtype=np.float32)  # [1, 1, 2, 2, 2]
    # Transposed convolution 3D: 1x2x2 input, 2x2x2 kernel, stride 1, pad 0
    # Output shape: [1, 1, 2, 3, 3]
    # Formula: For each output position (d, i, j), find contributing input positions
    # Input position = (output_pos + pad - kernel_pos) / stride (must be divisible)
    stride_d, stride_h, stride_w = 1, 1, 1
    pad_d, pad_h, pad_w = 0, 0, 0
    expected_3d = np.zeros((1, 1, 2, 3, 3), dtype=np.float32)
    for d_out in range(2):
        for i_out in range(3):
            for j_out in range(3):
                sum_val = 0.0
                # Iterate over kernel positions
                for kd in range(2):
                    for kh in range(2):
                        for kw in range(2):
                            # Input position formula: (output_pos + pad - kernel_pos) / stride
                            d_in = (d_out + pad_d - kd) / stride_d
                            i_in = (i_out + pad_h - kh) / stride_h
                            j_in = (j_out + pad_w - kw) / stride_w
                            # Check if divisible and in bounds
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
    x_depthwise = np.array([[[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]]], dtype=np.float32)  # [1, 1, 3, 3]
    w_depthwise = np.array([[[[1.0, 0.0], [0.0, 1.0]]]], dtype=np.float32)  # [1, 1, 2, 2] - depthwise: each channel has its own filter
    b_depthwise = np.array([0.0], dtype=np.float32)
    # Depthwise conv: each input channel convolved separately
    # For 1 channel with multiplier=1: same as regular conv
    expected_depthwise = np.zeros((1, 1, 2, 2), dtype=np.float32)
    for i in range(2):
        for j in range(2):
            expected_depthwise[0, 0, i, j] = np.sum(x_depthwise[0, 0, i:i+2, j:j+2] * w_depthwise[0, 0]) + b_depthwise[0]
    actual_depthwise = np.array(result.outputs['result_depthwise'])
    np.testing.assert_allclose(actual_depthwise, expected_depthwise, rtol=1e-5)
    
    # Verify depthwise_conv - Multi-channel
    x_depthwise_multi = np.array([[[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]], dtype=np.float32)  # [1, 2, 2, 2]
    w_depthwise_multi = np.array([[[[1.0, 0.0], [0.0, 1.0]]], [[[0.5, 0.5], [0.5, 0.5]]]], dtype=np.float32)  # [2, 1, 2, 2]
    b_depthwise_multi = np.array([0.0, 1.0], dtype=np.float32)
    # Channel 0: 1*1 + 2*0 + 3*0 + 4*1 = 5
    # Channel 1: 5*0.5 + 6*0.5 + 7*0.5 + 8*0.5 + 1 = 14
    expected_depthwise_multi = np.array([[[[5.0]], [[14.0]]]], dtype=np.float32)
    actual_depthwise_multi = np.array(result.outputs['result_depthwise_multi'])
    np.testing.assert_allclose(actual_depthwise_multi, expected_depthwise_multi, rtol=1e-5)
    
    # Verify depthwise_conv - With padding
    x_depthwise_pad = np.array([[[[1.0, 2.0], [3.0, 4.0]]]], dtype=np.float32)  # [1, 1, 2, 2]
    w_depthwise_pad = np.array([[[[1.0, 1.0], [1.0, 1.0]]]], dtype=np.float32)  # [1, 1, 2, 2]
    b_depthwise_pad = np.array([0.0], dtype=np.float32)
    # Output shape = (2 + 2*1 - 2) / 1 + 1 = 3
    # For each output position (i, j), compute: sum over kernel positions
    # Input position = (output_pos * stride - pad + kernel_pos)
    expected_depthwise_pad = np.zeros((1, 1, 3, 3), dtype=np.float32)
    pad_h, pad_w = 1, 1
    stride_h, stride_w = 1, 1
    for i in range(3):
        for j in range(3):
            sum_val = 0.0
            for kh in range(2):
                for kw in range(2):
                    # Input position in original (unpadded) space
                    h_in = i * stride_h - pad_h + kh
                    w_in = j * stride_w - pad_w + kw
                    # Only add if within bounds (padding is handled by bounds check)
                    if 0 <= h_in < 2 and 0 <= w_in < 2:
                        sum_val += x_depthwise_pad[0, 0, h_in, w_in] * w_depthwise_pad[0, 0, kh, kw]
            expected_depthwise_pad[0, 0, i, j] = sum_val + b_depthwise_pad[0]
    actual_depthwise_pad = np.array(result.outputs['result_depthwise_pad'])
    np.testing.assert_allclose(actual_depthwise_pad, expected_depthwise_pad, rtol=1e-5)
    
    # Verify depthwise_conv - With stride
    x_depthwise_stride = np.array([[[[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0], [13.0, 14.0, 15.0, 16.0]]]], dtype=np.float32)  # [1, 1, 4, 4]
    w_depthwise_stride = np.array([[[[1.0, 0.0], [0.0, 1.0]]]], dtype=np.float32)  # [1, 1, 2, 2]
    b_depthwise_stride = np.array([0.0], dtype=np.float32)
    # With stride [2, 2], output shape = (4 - 2) / 2 + 1 = 2
    expected_depthwise_stride = np.zeros((1, 1, 2, 2), dtype=np.float32)
    for i in range(2):
        for j in range(2):
            row = i * 2
            col = j * 2
            expected_depthwise_stride[0, 0, i, j] = np.sum(x_depthwise_stride[0, 0, row:row+2, col:col+2] * w_depthwise_stride[0, 0]) + b_depthwise_stride[0]
    actual_depthwise_stride = np.array(result.outputs['result_depthwise_stride'])
    np.testing.assert_allclose(actual_depthwise_stride, expected_depthwise_stride, rtol=1e-5)




# ===========================================================================
# Linear Algebra Operation Tests
# ===========================================================================
