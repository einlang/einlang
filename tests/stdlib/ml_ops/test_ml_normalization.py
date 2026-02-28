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
from ...test_utils import compile_and_execute, assert_float_close
# Use conftest's compiler/runtime
# ===========================================================================
# Activation Function Tests
# ===========================================================================



# ===========================================================================
# Normalization Operation Tests
# Clustered tests for efficiency - all normalization ops tested together
# Compile/execute errors are failures, not skips
# ===========================================================================

def test_normalization_clustered_accuracy(compiler, runtime):
    """Test normalization operations - clustered for efficiency"""
    source = """use std::ml;
    let input1 = [[1.0, 2.0], [3.0, 4.0]];
    let scale1 = [1.0, 1.0];
    let bias1 = [0.0, 0.0];
    let mean1 = [2.0, 3.0];
    let var1 = [1.0, 1.0];
    let result1 = std::ml::batch_normalization(input1, scale1, bias1, mean1, var1, 1e-5);
    let input2 = [[[[1.0, 2.0], [3.0, 4.0]]]];
    let scale2 = [1.0];
    let bias2 = [0.0];
    let result2 = std::ml::instance_normalization(input2, scale2, bias2, 1e-5);
    let input3 = [[1.0, 2.0], [3.0, 4.0]];
    let scale3 = [1.0, 1.0];
    let bias3 = [0.0, 0.0];
    let input4 = [[[[1.0, 2.0], [3.0, 4.0]]]];
    let size = 2;
    let alpha = 1.0;
    let beta = 0.5;
    let bias = 1.0;
    let input5 = [[1.0, 2.0, 3.0]];
    let p = 2.0;
    let input6 = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
    let axes = [1];
    let result3 = std::ml::layer_normalization(input3, scale3, bias3, 1e-5, -1);
    let result4 = std::ml::lrn(input4, size, alpha, beta, bias);
    let result5 = std::ml::lp_normalization(input5, -1, p);
    let result6 = std::ml::mean_variance_normalization(input6, axes);
    // Batch normalization 3D, 4D, 5D
    let input_bn_3d = [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]];
    let scale_bn_3d = [1.0, 1.0];
    let bias_bn_3d = [0.0, 0.0];
    let mean_bn_3d = [2.0, 3.0];
    let var_bn_3d = [1.0, 1.0];
    let result_bn_3d = std::ml::batch_normalization(input_bn_3d, scale_bn_3d, bias_bn_3d, mean_bn_3d, var_bn_3d, 1e-5);
    let input_bn_4d = [[[[1.0, 2.0], [3.0, 4.0]]]];
    let scale_bn_4d = [1.0];
    let bias_bn_4d = [0.0];
    let mean_bn_4d = [2.0];
    let var_bn_4d = [1.0];
    let result_bn_4d = std::ml::batch_normalization(input_bn_4d, scale_bn_4d, bias_bn_4d, mean_bn_4d, var_bn_4d, 1e-5);
    let input_bn_5d = [[[[[1.0, 2.0], [3.0, 4.0]]]]];
    let scale_bn_5d = [1.0];
    let bias_bn_5d = [0.0];
    let mean_bn_5d = [2.0];
    let var_bn_5d = [1.0];
    let result_bn_5d = std::ml::batch_normalization(input_bn_5d, scale_bn_5d, bias_bn_5d, mean_bn_5d, var_bn_5d, 1e-5);
    """
    
    result = compile_and_execute(source, compiler, runtime)
    assert result.success, f"Execution failed: {result.errors}"

    # Verify each operation

    # Verify batch_normalization - Test Batch Normalization
    input1 = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    scale1 = np.array([1.0, 1.0], dtype=np.float32)
    bias1 = np.array([0.0, 0.0], dtype=np.float32)
    mean1 = np.array([2.0, 3.0], dtype=np.float32)
    var1 = np.array([1.0, 1.0], dtype=np.float32)
    expected = (input1 - mean1) / np.sqrt(var1 + 1e-5) * scale1 + bias1
    actual = np.array(result.outputs['result1'])
    np.testing.assert_allclose(actual, expected, rtol=1e-5)

    # Verify instance_normalization - Test Instance Normalization
    input2 = np.array([[[[1.0, 2.0], [3.0, 4.0]]]], dtype=np.float32)
    scale2 = np.array([1.0], dtype=np.float32)
    bias2 = np.array([0.0], dtype=np.float32)
    # Instance norm normalizes over spatial dimensions (H, W) per channel
    mean = np.mean(input2, axis=(2, 3), keepdims=True)
    var = np.var(input2, axis=(2, 3), keepdims=True)
    expected = scale2.reshape(1, -1, 1, 1) * (input2 - mean) / np.sqrt(var + 1e-5) + bias2.reshape(1, -1, 1, 1)
    actual = np.array(result.outputs['result2'])
    np.testing.assert_allclose(actual, expected, rtol=1e-5)

    # Verify layer_normalization - Test Layer Normalization
    input3 = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    scale3 = np.array([1.0, 1.0], dtype=np.float32)
    bias3 = np.array([0.0, 0.0], dtype=np.float32)
    # Layer norm normalizes over last dimension
    mean = np.mean(input3, axis=-1, keepdims=True)
    var = np.var(input3, axis=-1, keepdims=True)
    expected = scale3 * (input3 - mean) / np.sqrt(var + 1e-5) + bias3
    actual = np.array(result.outputs['result3'])
    np.testing.assert_allclose(actual, expected, rtol=1e-5)


    # Verify lrn - Test Local Response Normalization
    input4 = np.array([[[[1.0, 2.0], [3.0, 4.0]]]], dtype=np.float32)  # Using input4 from source
    size = 2  # Using size from source
    alpha = 1.0  # Using alpha from source
    beta = 0.5  # Using beta from source
    bias = 1.0  # Using bias from source
    # LRN: output = input / (bias + alpha * sum(neighbor_channels^2))^beta
    # Simplified check: verify output shape and approximate values
    actual = np.array(result.outputs['result4'])
    assert actual.shape == input4.shape, f"Expected shape {input4.shape}, got {actual.shape}"
    np.testing.assert_allclose(actual, input4 / np.power(bias + alpha * np.sum(input4**2, axis=1, keepdims=True), beta), rtol=1e-4)


    # Verify lp_normalization - Test Lp Normalization
    input5 = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)  # Using input5 from source
    p = 2.0  # Using p from source
    # Lp norm: normalize by Lp norm along specified axis
    lp_norm = np.power(np.sum(np.power(np.abs(input5), p), axis=-1, keepdims=True), 1.0/p)
    expected = input5 / lp_norm
    actual = np.array(result.outputs['result5'])
    np.testing.assert_allclose(actual, expected, rtol=1e-5)


    # Verify mean_variance_normalization - Test Mean Variance Normalization
    input6 = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)  # Using input6 from source
    axes = [1]  # Using axes from source
    # Normalize over specified axes
    mean = np.mean(input6, axis=tuple(axes), keepdims=True)
    var = np.var(input6, axis=tuple(axes), keepdims=True)
    expected = (input6 - mean) / np.sqrt(var + 1e-9)
    actual = np.array(result.outputs['result6'])
    np.testing.assert_allclose(actual, expected, rtol=1e-5)

    # Verify batch_normalization 3D - Test Batch Normalization (3D)
    # Input shape: [N, C, D1] = [2, 2, 2]
    input_bn_3d = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], dtype=np.float32)
    scale_bn_3d = np.array([1.0, 1.0], dtype=np.float32)
    bias_bn_3d = np.array([0.0, 0.0], dtype=np.float32)
    mean_bn_3d = np.array([2.0, 3.0], dtype=np.float32)
    var_bn_3d = np.array([1.0, 1.0], dtype=np.float32)
    # Formula: (X[n, c, d1] - mean[c]) / sqrt(var[c] + eps) * scale[c] + bias[c]
    # Broadcast mean/var/scale/bias along channel dimension
    expected_bn_3d = (input_bn_3d - mean_bn_3d[:, np.newaxis]) / np.sqrt(var_bn_3d[:, np.newaxis] + 1e-5) * scale_bn_3d[:, np.newaxis] + bias_bn_3d[:, np.newaxis]
    actual_bn_3d = np.array(result.outputs['result_bn_3d'])
    np.testing.assert_allclose(actual_bn_3d, expected_bn_3d, rtol=1e-5)

    # Verify batch_normalization 4D - Test Batch Normalization (4D)
    # Input shape: [N, C, D1, D2] = [1, 1, 2, 2]
    input_bn_4d = np.array([[[[1.0, 2.0], [3.0, 4.0]]]], dtype=np.float32)
    scale_bn_4d = np.array([1.0], dtype=np.float32)
    bias_bn_4d = np.array([0.0], dtype=np.float32)
    mean_bn_4d = np.array([2.0], dtype=np.float32)
    var_bn_4d = np.array([1.0], dtype=np.float32)
    # Formula: (X[n, c, d1, d2] - mean[c]) / sqrt(var[c] + eps) * scale[c] + bias[c]
    expected_bn_4d = (input_bn_4d - mean_bn_4d[:, np.newaxis, np.newaxis]) / np.sqrt(var_bn_4d[:, np.newaxis, np.newaxis] + 1e-5) * scale_bn_4d[:, np.newaxis, np.newaxis] + bias_bn_4d[:, np.newaxis, np.newaxis]
    actual_bn_4d = np.array(result.outputs['result_bn_4d'])
    np.testing.assert_allclose(actual_bn_4d, expected_bn_4d, rtol=1e-5)

    # Verify batch_normalization 5D - Test Batch Normalization (5D)
    # Input shape: [N, C, D1, D2, D3] = [1, 1, 1, 2, 2]
    input_bn_5d = np.array([[[[[1.0, 2.0], [3.0, 4.0]]]]], dtype=np.float32)
    scale_bn_5d = np.array([1.0], dtype=np.float32)
    bias_bn_5d = np.array([0.0], dtype=np.float32)
    mean_bn_5d = np.array([2.0], dtype=np.float32)
    var_bn_5d = np.array([1.0], dtype=np.float32)
    # Formula: (X[n, c, d1, d2, d3] - mean[c]) / sqrt(var[c] + eps) * scale[c] + bias[c]
    expected_bn_5d = (input_bn_5d - mean_bn_5d[:, np.newaxis, np.newaxis, np.newaxis]) / np.sqrt(var_bn_5d[:, np.newaxis, np.newaxis, np.newaxis] + 1e-5) * scale_bn_5d[:, np.newaxis, np.newaxis, np.newaxis] + bias_bn_5d[:, np.newaxis, np.newaxis, np.newaxis]
    actual_bn_5d = np.array(result.outputs['result_bn_5d'])
    np.testing.assert_allclose(actual_bn_5d, expected_bn_5d, rtol=1e-5)
