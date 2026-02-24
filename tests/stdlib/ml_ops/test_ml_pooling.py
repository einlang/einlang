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
# Pooling Operation Tests
# Clustered tests for efficiency - all pooling ops tested together
# Compile/execute errors are failures, not skips
# ===========================================================================

def test_pooling_clustered_accuracy(compiler, runtime):
    """Test pooling operations - clustered for efficiency"""
    source = """use std::ml;
    let x = [[[[1.0, 2.0], [3.0, 4.0]]]];
    let x_1d = [[[1.0, 2.0, 3.0, 4.0]]];
    let x_3d = [[[[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]]];
    let pool_h = 2;
    let pool_w = 2;
    let stride_h = 2;
    let stride_w = 2;
    let pad_h = 0;
    let pad_w = 0;
    let result_0 = std::ml::global_max_pool(x);
    let result_0_1d = std::ml::global_max_pool(x_1d);
    let result_0_3d = std::ml::global_max_pool(x_3d);
    let x_1d_pool = [[[1.0, 2.0, 3.0, 4.0, 5.0]]];
    let pool_1d = 2;
    let stride_1d = 2;
    let pad_1d = 0;
    let result_max_pool_1d = std::ml::max_pool(x_1d_pool, [pool_1d], [stride_1d], [pad_1d]);
    let result_avg_pool_1d = std::ml::average_pool(x_1d_pool, [pool_1d], [stride_1d], [pad_1d]);
    let result_lp_pool_1d = std::ml::lp_pool(x_1d_pool, [pool_1d], [stride_1d], 2);
    let result_1 = std::ml::max_pool(x, [pool_h, pool_w], [stride_h, stride_w], [pad_h, pad_w]);
    let result_2 = std::ml::average_pool(x, [pool_h, pool_w], [stride_h, stride_w], [pad_h, pad_w]);
    let result_3 = std::ml::global_average_pool(x);
    let result_3_1d = std::ml::global_average_pool(x_1d);
    let result_3_3d = std::ml::global_average_pool(x_3d);
    let result_4 = std::ml::lp_pool(x, [2, 2], [1, 1], 2);
    let x_3d_pool = [[[[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]]];
    let pool_3d_d = 2;
    let pool_3d_h = 2;
    let pool_3d_w = 2;
    let stride_3d_d = 1;
    let stride_3d_h = 1;
    let stride_3d_w = 1;
    let pad_3d_d = 0;
    let pad_3d_h = 0;
    let pad_3d_w = 0;
    let result_max_pool_3d = std::ml::max_pool(x_3d_pool, [pool_3d_d, pool_3d_h, pool_3d_w], [stride_3d_d, stride_3d_h, stride_3d_w], [pad_3d_d, pad_3d_h, pad_3d_w]);
    let result_avg_pool_3d = std::ml::average_pool(x_3d_pool, [pool_3d_d, pool_3d_h, pool_3d_w], [stride_3d_d, stride_3d_h, stride_3d_w], [pad_3d_d, pad_3d_h, pad_3d_w]);
    let result_lp_pool_3d = std::ml::lp_pool(x_3d_pool, [pool_3d_d, pool_3d_h, pool_3d_w], [stride_3d_d, stride_3d_h, stride_3d_w], 2);
    let x_roi = [[[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]]];
    let rois = [[0, 0, 0, 2, 2]];
    let pooled_shape = [2, 2];
    let spatial_scale = 1.0;
    let result_5 = std::ml::max_roi_pool(x_roi, rois, pooled_shape, spatial_scale);
    """

    result = compile_and_execute(source, compiler, runtime)
    assert result.success, f"Execution failed: {result.errors}"

    # Verify each operation


    # Verify global_max_pool - Test Global Max Pooling (2D spatial, rank 4)
    x = np.array([[[[1.0, 2.0], [3.0, 4.0]]]], dtype=np.float32)  # Using x from source
    expected = np.max(x, axis=(-2, -1), keepdims=True)  # [N, C, 1, 1]
    actual = np.array(result.outputs['result_0'])
    np.testing.assert_allclose(actual, expected, rtol=1e-6)


    # Verify global_max_pool - Test Global Max Pooling (1D spatial, rank 3)
    x_1d = np.array([[[1.0, 2.0, 3.0, 4.0]]], dtype=np.float32)  # Using x_1d from source
    expected = np.max(x_1d, axis=-1, keepdims=True)  # [N, C, 1]
    actual = np.array(result.outputs['result_0_1d'])
    np.testing.assert_allclose(actual, expected, rtol=1e-6)


    # Verify global_max_pool - Test Global Max Pooling (3D spatial, rank 5)
    x_3d = np.array([[[[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]]], dtype=np.float32)  # Using x_3d from source
    expected = np.max(x_3d, axis=(-3, -2, -1), keepdims=True)  # [N, C, 1, 1, 1]
    actual = np.array(result.outputs['result_0_3d'])
    np.testing.assert_allclose(actual, expected, rtol=1e-6)


    # Verify max_pool 1D - Test MaxPool1D
    x_1d_pool = np.array([[[1.0, 2.0, 3.0, 4.0, 5.0]]], dtype=np.float32)
    expected_1d = np.array([[[2.0, 4.0]]], dtype=np.float32)  # Max of [1,2]=2, [3,4]=4
    actual_1d = np.array(result.outputs['result_max_pool_1d'])
    np.testing.assert_allclose(actual_1d, expected_1d, rtol=1e-6)

    # Verify max_pool 2D - Test MaxPool2D
    x = np.array([[[[1.0, 2.0], [3.0, 4.0]]]])  # Using x from source (2x2 input)
    # MaxPool with kernel 2x2, stride 2x2, pad 0x0 on 2x2 input -> 1x1 output
    expected = np.array([[[[4.0]]]])  # Max of [1,2,3,4] = 4.0
    actual = np.array(result.outputs['result_1'])
    np.testing.assert_allclose(actual, expected, rtol=1e-6)

    # Verify max_pool 3D - Test MaxPool3D
    x_3d_pool = np.array([[[[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]]], dtype=np.float32)
    # MaxPool with kernel 2x2x2, stride 1x1x1 on 2x2x2 input -> 1x1x1 output
    expected_3d = np.array([[[[[8.0]]]]], dtype=np.float32)  # Max of all 8 values = 8.0
    actual_3d = np.array(result.outputs['result_max_pool_3d'])
    np.testing.assert_allclose(actual_3d, expected_3d, rtol=1e-6)


    # Verify average_pool 1D - Test AvgPool1D
    x_1d_pool = np.array([[[1.0, 2.0, 3.0, 4.0, 5.0]]], dtype=np.float32)
    expected_1d = np.array([[[1.5, 3.5]]], dtype=np.float32)  # Mean of [1,2]=1.5, [3,4]=3.5
    actual_1d = np.array(result.outputs['result_avg_pool_1d'])
    np.testing.assert_allclose(actual_1d, expected_1d, rtol=1e-5)

    # Verify average_pool 2D - Test AvgPool2D
    x = np.array([[[[1.0, 2.0], [3.0, 4.0]]]])  # Using x from source (2x2 input)
    # AvgPool with kernel 2x2, stride 2x2, pad 0x0 on 2x2 input -> 1x1 output
    expected = np.array([[[[2.5]]]])  # Mean of [1,2,3,4] = 2.5
    actual = np.array(result.outputs['result_2'])
    np.testing.assert_allclose(actual, expected, rtol=1e-5)

    # Verify average_pool 3D - Test AvgPool3D
    x_3d_pool = np.array([[[[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]]], dtype=np.float32)
    # AvgPool with kernel 2x2x2, stride 1x1x1 on 2x2x2 input -> 1x1x1 output
    expected_3d = np.array([[[[[4.5]]]]], dtype=np.float32)  # Mean of all 8 values = 4.5
    actual_3d = np.array(result.outputs['result_avg_pool_3d'])
    np.testing.assert_allclose(actual_3d, expected_3d, rtol=1e-5)


    # Verify global_average_pool - Test global_average_pool operation (2D spatial, rank 4)
    x = np.array([[[[1.0, 2.0], [3.0, 4.0]]]], dtype=np.float32)  # Using x from source
    expected = np.mean(x, axis=(-2, -1), keepdims=True)  # [N, C, 1, 1]
    actual = np.array(result.outputs['result_3'])
    np.testing.assert_allclose(actual, expected, rtol=1e-6)


    # Verify global_average_pool - Test global_average_pool operation (1D spatial, rank 3)
    x_1d = np.array([[[1.0, 2.0, 3.0, 4.0]]], dtype=np.float32)  # Using x_1d from source
    expected = np.mean(x_1d, axis=-1, keepdims=True)  # [N, C, 1]
    actual = np.array(result.outputs['result_3_1d'])
    np.testing.assert_allclose(actual, expected, rtol=1e-6)


    # Verify global_average_pool - Test global_average_pool operation (3D spatial, rank 5)
    x_3d = np.array([[[[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]]], dtype=np.float32)  # Using x_3d from source
    expected = np.mean(x_3d, axis=(-3, -2, -1), keepdims=True)  # [N, C, 1, 1, 1]
    actual = np.array(result.outputs['result_3_3d'])
    np.testing.assert_allclose(actual, expected, rtol=1e-6)


    # Verify lp_pool 1D - Test lp_pool operation (1D)
    x_1d_pool = np.array([[[1.0, 2.0, 3.0, 4.0, 5.0]]], dtype=np.float32)
    # With kernel=2, stride=2 on 5 elements: output positions 0 and 1
    # Position 0: elements [0:2] = [1, 2], L2 norm = sqrt(1^2 + 2^2) = sqrt(5) ≈ 2.236
    # Position 1: elements [2:4] = [3, 4], L2 norm = sqrt(3^2 + 4^2) = sqrt(25) = 5.0
    expected_1d = np.array([[[np.sqrt(1**2 + 2**2), np.sqrt(3**2 + 4**2)]]], dtype=np.float32)  # [sqrt(5), 5.0]
    actual_1d = np.array(result.outputs['result_lp_pool_1d'])
    np.testing.assert_allclose(actual_1d, expected_1d, rtol=1e-5)

    # Verify lp_pool 2D - Test lp_pool operation (2D)
    x = np.array([[[[1.0, 2.0], [3.0, 4.0]]]], dtype=np.float32)  # Using x from source
    expected = np.sqrt(np.sum(x**2, axis=(-2, -1), keepdims=True))
    actual = np.array(result.outputs['result_4'])
    np.testing.assert_allclose(actual, expected, rtol=1e-5)

    # Verify lp_pool 3D - Test lp_pool operation (3D)
    x_3d_pool = np.array([[[[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]]], dtype=np.float32)
    expected_3d = np.power(np.sum(np.power(np.abs(x_3d_pool), 2), axis=(-3, -2, -1), keepdims=True), 1.0/2)
    actual_3d = np.array(result.outputs['result_lp_pool_3d'])
    np.testing.assert_allclose(actual_3d, expected_3d, rtol=1e-5)


    # Verify max_roi_pool - Test MaxRoiPool operation
    x_roi = np.array([[[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]]], dtype=np.float32)  # Using x_roi from source
    rois = np.array([[0, 0, 0, 2, 2]], dtype=np.float32)  # Using rois from source: [batch_idx, x1, y1, x2, y2]
    pooled_shape = [2, 2]  # Using pooled_shape from source
    spatial_scale = 1.0  # Using spatial_scale from source
    # Extract ROI region [0:2, 0:2] = [[1, 2], [4, 5]]
    # Max pool to 2x2: each bin is 1x1, so output is [[1, 2], [4, 5]]
    roi_region = x_roi[0, 0, 0:2, 0:2]  # [[1, 2], [4, 5]]
    # For 2x2 pooled output from 2x2 region with bin size 1x1, output is same as input
    expected = roi_region.reshape(1, 1, 2, 2)  # [1, 1, 2, 2]
    actual = np.array(result.outputs['result_5'])
    np.testing.assert_allclose(actual, expected, rtol=1e-5)