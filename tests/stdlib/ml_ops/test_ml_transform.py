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
# Transform Operation Tests
# Clustered tests for efficiency - all transform ops tested together
# Compile/execute errors are failures, not skips
# ===========================================================================

def test_transform_clustered_accuracy(compiler, runtime):
    """Test transform operations - clustered for efficiency"""
    source = """use std::ml;
    let x = [[[1.0, 2.0], [3.0, 4.0]]];
    let data_1d = [1.0, 2.0, 3.0];
    let data_2d = [[1.0, 2.0], [3.0, 4.0]];
    let pads = [1];
    let x_1 = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
    let a = [[1.0, 2.0]];
    let b = [[3.0, 4.0]];
    let x_2 = [[1.0, 2.0]];
    let repeats = 3;
    let x_3 = [[[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]], [[9.0, 10.0], [11.0, 12.0]], [[13.0, 14.0], [15.0, 16.0]]]];
    let x_4 = [[[[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0], [13.0, 14.0, 15.0, 16.0]]]];
    let x_5 = [[1.0, 2.0, 3.0, 4.0]];
    let x_6 = [[[[1.0, 2.0]]]];
    let axes = [2];
    let x_6_1d = [[1.0, 2.0]];
    let axes_1d = [0];
    let x_6_3d = [[[[[1.0, 2.0]]]]];
    let axes_3d = [2];
    let x_7 = [[1.0, 2.0, 3.0]];
    let axes_1 = [1];
    let x_7_1d = [1.0, 2.0, 3.0];
    let axes_1_1d = [0];
    let x_7_3d = [[[1.0, 2.0], [3.0, 4.0]]];
    let axes_1_3d = [1];
    let start = 0.0;
    let limit = 5.0;
    let delta = 1.0;
    let shape_val = [5];
    let value = 42.0;
    let data_split = [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]];
    let split_sizes = [2];
    let data_split_1d = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let split_sizes_1d = [3];
    let data_split_3d = [[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]];
    let split_sizes_3d = [1];
    let data_expand = [[1.0, 2.0]];
    let expand_shape = [3, 2];
    let data_expand_1d = [1.0, 2.0, 3.0];
    let expand_shape_1d = [2, 3];
    let data_expand_3d = [[[1.0, 2.0], [3.0, 4.0]]];
    let expand_shape_3d = [2, 2, 2];
    let data_shape = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
    let x_resize = [[[[1.0, 2.0], [3.0, 4.0]]]];
    let scales_resize = [2.0, 2.0];
    let x_upsample = [[[[1.0, 2.0], [3.0, 4.0]]]];
    let scales_upsample = [2.0, 2.0];
    let result_0 = std::ml::transpose(x);
    let result_1d_1 = std::ml::pad(data_1d, pads, 0.0);
    let result_2d_1 = std::ml::pad(data_2d, pads, 0.0);
    let result_2 = std::ml::flatten(x_1);
    let result_3 = std::ml::concat(a, b);
    let result_4 = std::ml::tile(x_2, repeats);
    let result_5 = std::ml::depth_to_space(x_3, 2);
    let result_6 = std::ml::space_to_depth(x_4, 2);
    let result_7 = std::ml::reshape(x_5, [2, 2]);
    let result_8 = std::ml::squeeze(x_6, axes);
    let result_8_1d = std::ml::squeeze(x_6_1d, axes_1d);
    let result_8_3d = std::ml::squeeze(x_6_3d, axes_3d);
    let x_6_4d = [[[[1.0, 2.0]]]];
    let axes_4d = [2];
    let result_8_4d = std::ml::squeeze(x_6_4d, axes_4d);
    let x_6_5d = [[[[[1.0, 2.0]]]]];
    let axes_5d = [2];
    let result_8_5d = std::ml::squeeze(x_6_5d, axes_5d);
    let result_9 = std::ml::unsqueeze(x_7, axes_1);
    let result_9_1d = std::ml::unsqueeze(x_7_1d, axes_1_1d);
    let result_9_3d = std::ml::unsqueeze(x_7_3d, axes_1_3d);
    let result_10 = std::ml::range(start, limit, delta);
    let result_11 = std::ml::constant_of_shape(shape_val, value);
    let result_12 = std::ml::split(data_split, split_sizes, 0);
    let result_12_1d = std::ml::split(data_split_1d, split_sizes_1d, 0);
    let result_12_3d = std::ml::split(data_split_3d, split_sizes_3d, 0);
    let data_split_4d = [[[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]];
    let split_sizes_4d = [1];
    let result_12_4d = std::ml::split(data_split_4d, split_sizes_4d, 0);
    let result_13 = std::ml::expand(data_expand, expand_shape);
    let result_13_1d = std::ml::expand(data_expand_1d, expand_shape_1d);
    let result_13_3d = std::ml::expand(data_expand_3d, expand_shape_3d);
    let data_expand_2d_to_3d = [[1.0, 2.0], [3.0, 4.0]];
    let expand_shape_2d_to_3d = [2, 2, 2];
    let result_13_2d_to_3d = std::ml::expand(data_expand_2d_to_3d, expand_shape_2d_to_3d);
    let data_expand_3d_to_4d = [[[1.0, 2.0], [3.0, 4.0]]];
    let expand_shape_3d_to_4d = [2, 1, 2, 2];
    let result_13_3d_to_4d = std::ml::expand(data_expand_3d_to_4d, expand_shape_3d_to_4d);
    let result_14 = std::ml::shape(data_shape);
    let result_15 = std::ml::resize(x_resize, scales_resize, "nearest");
    let result_16 = std::ml::upsample(x_upsample, scales_upsample, "nearest");
    """

    result = compile_and_execute(source, compiler, runtime)
    assert result.success, f"Execution failed: {result.errors}"

    # Verify each operation


    # Verify transpose - Test Transpose
    x = np.array([[[1.0, 2.0], [3.0, 4.0]]])
    expected = np.transpose(x, (0, 2, 1))  # Swap last two dims
    actual = np.array(result.outputs['result_0'])
    np.testing.assert_allclose(actual, expected, rtol=1e-6)


    # Verify pad - Test ONNX Pad operator for 1D and 2D
    data_1d = np.array([1.0, 2.0, 3.0])  # Using data_1d from source
    pads = [1]  # Using pads from source
    expected_1d = np.pad(data_1d, pads[0], mode='constant', constant_values=0.0)
    actual_1d = np.array(result.outputs['result_1d_1'])
    np.testing.assert_allclose(actual_1d, expected_1d, rtol=1e-6)
    data_2d = np.array([[1.0, 2.0], [3.0, 4.0]])  # Using data_2d from source
    expected_2d = np.pad(data_2d, pads[0], mode='constant', constant_values=0.0)
    actual_2d = np.array(result.outputs['result_2d_1'])
    np.testing.assert_allclose(actual_2d, expected_2d, rtol=1e-6)


    # Verify flatten - Test flatten operation
    assert result.success, f"Execution failed: {result.errors}"
    x = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    expected = x.flatten()  # [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    actual = np.array(result.outputs['result_2'])
    np.testing.assert_allclose(actual, expected, rtol=1e-6)


    # Verify concat - Test concat operation
    assert result.success, f"Execution failed: {result.errors}"
    a = np.array([[1.0, 2.0]], dtype=np.float32)
    b = np.array([[3.0, 4.0]], dtype=np.float32)
    expected = np.concatenate([a, b], axis=-1)
    actual = np.array(result.outputs['result_3'])
    np.testing.assert_allclose(actual, expected, rtol=1e-6)


    # Verify tile - Test tile operation
    assert result.success, f"Execution failed: {result.errors}"
    x = np.array([[1.0, 2.0]], dtype=np.float32)
    expected = np.tile(x, (1, 3))
    actual = np.array(result.outputs['result_4'])
    np.testing.assert_allclose(actual, expected, rtol=1e-6)


    # Verify depth_to_space - Test depth_to_space operation
    x_3 = np.array([[[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]], [[9.0, 10.0], [11.0, 12.0]], [[13.0, 14.0], [15.0, 16.0]]]], dtype=np.float32)  # [1, 4, 2, 2]
    r = 2
    # depth_to_space: [batch, C*r*r, H, W] -> [batch, C, H*r, W*r]
    # Input: [1, 4, 2, 2] with r=2 -> C=1, output: [1, 1, 4, 4]
    expected_depth = np.zeros((1, 1, 4, 4), dtype=np.float32)
    for b in range(1):
        for c in range(1):
            for i in range(4):
                for j in range(4):
                    c_idx = c * (r * r) + (i % r) * r + (j % r)
                    h_idx = i // r
                    w_idx = j // r
                    expected_depth[b, c, i, j] = x_3[b, c_idx, h_idx, w_idx]
    actual = np.array(result.outputs['result_5'])
    np.testing.assert_allclose(actual, expected_depth, rtol=1e-6)

    # Verify space_to_depth - Test space_to_depth operation
    x_4 = np.array([[[[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0], [13.0, 14.0, 15.0, 16.0]]]], dtype=np.float32)  # [1, 1, 4, 4]
    # space_to_depth: [batch, C, H*r, W*r] -> [batch, C*r*r, H, W]
    # Input: [1, 1, 4, 4] with r=2 -> H=2, W=2, output: [1, 4, 2, 2]
    expected_space = np.zeros((1, 4, 2, 2), dtype=np.float32)
    for b in range(1):
        for c in range(1):
            for i in range(4):
                for j in range(4):
                    c_out = c * (r * r) + (i % r) * r + (j % r)
                    h_out = i // r
                    w_out = j // r
                    expected_space[b, c_out, h_out, w_out] = x_4[b, c, i, j]
    actual = np.array(result.outputs['result_6'])
    np.testing.assert_allclose(actual, expected_space, rtol=1e-6)


    # Verify reshape - Test reshape operation
    assert result.success, f"Execution failed: {result.errors}"
    x = np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32)
    expected = x.reshape(2, 2)
    actual = np.array(result.outputs['result_7'])
    np.testing.assert_allclose(actual, expected, rtol=1e-6)


    # Verify squeeze 1D - Test squeeze operation (1D)
    x_6_1d = np.array([[1.0, 2.0]], dtype=np.float32)
    expected_1d = np.squeeze(x_6_1d, axis=0)
    actual_1d = np.array(result.outputs['result_8_1d'])
    np.testing.assert_allclose(actual_1d, expected_1d, rtol=1e-6)

    # Verify squeeze 2D - Test squeeze operation (2D)
    x = np.array([[[[1.0, 2.0]]]], dtype=np.float32)
    expected = np.squeeze(x, axis=2)
    actual = np.array(result.outputs['result_8'])
    np.testing.assert_allclose(actual, expected, rtol=1e-6)

    # Verify squeeze 3D - Test squeeze operation (3D)
    x_6_3d = np.array([[[[[1.0, 2.0]]]]], dtype=np.float32)
    expected_3d = np.squeeze(x_6_3d, axis=2)
    actual_3d = np.array(result.outputs['result_8_3d'])
    np.testing.assert_allclose(actual_3d, expected_3d, rtol=1e-6)

    # Verify squeeze 4D - Test squeeze operation (4D)
    # Input: [[[[1.0, 2.0]]]] shape (1, 1, 1, 2), squeeze axis 2 -> (1, 1, 2)
    x_6_4d = np.array([[[[1.0, 2.0]]]], dtype=np.float32)
    expected_4d = np.squeeze(x_6_4d, axis=2)
    actual_4d = np.array(result.outputs['result_8_4d'])
    np.testing.assert_allclose(actual_4d, expected_4d, rtol=1e-6)

    # Verify squeeze 5D - Test squeeze operation (5D)
    # Input: [[[[[1.0, 2.0]]]]] shape (1, 1, 1, 1, 2), squeeze axis 2 -> (1, 1, 1, 2)
    x_6_5d = np.array([[[[[1.0, 2.0]]]]], dtype=np.float32)
    expected_5d = np.squeeze(x_6_5d, axis=2)
    actual_5d = np.array(result.outputs['result_8_5d'])
    np.testing.assert_allclose(actual_5d, expected_5d, rtol=1e-6)

    # Verify unsqueeze 1D - Test unsqueeze operation (1D)
    x_7_1d = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    expected_1d = np.expand_dims(x_7_1d, axis=0)
    actual_1d = np.array(result.outputs['result_9_1d'])
    np.testing.assert_allclose(actual_1d, expected_1d, rtol=1e-6)

    # Verify unsqueeze 2D - Test unsqueeze operation (2D)
    x = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
    expected = np.expand_dims(x, axis=1)
    actual = np.array(result.outputs['result_9'])
    np.testing.assert_allclose(actual, expected, rtol=1e-6)

    # Verify unsqueeze 3D - Test unsqueeze operation (3D)
    x_7_3d = np.array([[[1.0, 2.0], [3.0, 4.0]]], dtype=np.float32)
    expected_3d = np.expand_dims(x_7_3d, axis=1)
    actual_3d = np.array(result.outputs['result_9_3d'])
    np.testing.assert_allclose(actual_3d, expected_3d, rtol=1e-6)


    # Verify range - Test range operation
    start = 0.0  # Using start from source
    limit = 5.0  # Using limit from source
    delta = 1.0  # Using delta from source
    expected = np.arange(start, limit, delta, dtype=np.float32)  # [0.0, 1.0, 2.0, 3.0, 4.0]
    actual = np.array(result.outputs['result_10'])
    np.testing.assert_allclose(actual, expected, rtol=1e-6)


    # Verify constant_of_shape - Test constant_of_shape operation
    # Note: Implementation only supports 1D shapes currently
    shape_val = [5]  # Using shape_val from source (1D shape)
    value = 42.0  # Using value from source
    expected = np.full(shape_val, value, dtype=np.float32)  # [42.0, 42.0, 42.0, 42.0, 42.0]
    actual = np.array(result.outputs['result_11'])
    np.testing.assert_allclose(actual, expected, rtol=1e-6)


    # Verify split 1D - Test split operation (1D)
    data_split_1d = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=np.float32)
    split_sizes_1d = [3]
    expected_1d = data_split_1d[:3]  # [1.0, 2.0, 3.0]
    actual_1d = np.array(result.outputs['result_12_1d'])
    np.testing.assert_allclose(actual_1d, expected_1d, rtol=1e-6)

    # Verify split 2D - Test split operation (2D)
    data_split = np.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]], dtype=np.float32)  # Using data_split from source
    split_sizes = [2]  # Using split_sizes from source
    # Split along axis 0: take first 2 rows
    expected = data_split[:2, :]  # [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]
    actual = np.array(result.outputs['result_12'])
    np.testing.assert_allclose(actual, expected, rtol=1e-6)

    # Verify split 3D - Test split operation (3D)
    data_split_3d = np.array([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]], dtype=np.float32)
    split_sizes_3d = [1]
    expected_3d = data_split_3d[:1, :, :]  # [[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]]
    actual_3d = np.array(result.outputs['result_12_3d'])
    np.testing.assert_allclose(actual_3d, expected_3d, rtol=1e-6)

    # Verify split 4D - Test split operation (4D)
    data_split_4d = np.array([[[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]], dtype=np.float32)
    split_sizes_4d = [1]
    expected_4d = data_split_4d[:1, :, :, :]  # [[[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]]
    actual_4d = np.array(result.outputs['result_12_4d'])
    np.testing.assert_allclose(actual_4d, expected_4d, rtol=1e-6)

    # Verify expand 1D - Test expand operation (1D)
    data_expand_1d = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    expand_shape_1d = [2, 3]
    expected_1d = np.broadcast_to(data_expand_1d, expand_shape_1d)
    actual_1d = np.array(result.outputs['result_13_1d'])
    np.testing.assert_allclose(actual_1d, expected_1d, rtol=1e-6)

    # Verify expand 2D - Test expand operation (2D)
    data_expand = np.array([[1.0, 2.0]], dtype=np.float32)  # Using data_expand from source
    expand_shape = [3, 2]  # Using expand_shape from source
    # Expand [1, 2] to [3, 2] by broadcasting first dimension
    expected = np.broadcast_to(data_expand, expand_shape)  # [[1.0, 2.0], [1.0, 2.0], [1.0, 2.0]]
    actual = np.array(result.outputs['result_13'])
    np.testing.assert_allclose(actual, expected, rtol=1e-6)

    # Verify expand 3D - Test expand operation (3D)
    data_expand_3d = np.array([[[1.0, 2.0], [3.0, 4.0]]], dtype=np.float32)
    expand_shape_3d = [2, 2, 2]
    expected_3d = np.broadcast_to(data_expand_3d, expand_shape_3d)
    actual_3d = np.array(result.outputs['result_13_3d'])
    np.testing.assert_allclose(actual_3d, expected_3d, rtol=1e-6)

    # Verify expand 2D->3D - Test expand operation (2D to 3D)
    data_expand_2d_to_3d = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    expand_shape_2d_to_3d = [2, 2, 2]
    expected_2d_to_3d = np.broadcast_to(data_expand_2d_to_3d, expand_shape_2d_to_3d)
    actual_2d_to_3d = np.array(result.outputs['result_13_2d_to_3d'])
    np.testing.assert_allclose(actual_2d_to_3d, expected_2d_to_3d, rtol=1e-6)

    # Verify expand 3D->4D - Test expand operation (3D to 4D)
    data_expand_3d_to_4d = np.array([[[1.0, 2.0], [3.0, 4.0]]], dtype=np.float32)
    expand_shape_3d_to_4d = [2, 1, 2, 2]
    expected_3d_to_4d = np.broadcast_to(data_expand_3d_to_4d, expand_shape_3d_to_4d)
    actual_3d_to_4d = np.array(result.outputs['result_13_3d_to_4d'])
    np.testing.assert_allclose(actual_3d_to_4d, expected_3d_to_4d, rtol=1e-6)


    # Verify shape - Test shape operation
    data_shape = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)  # Using data_shape from source
    expected = np.array(data_shape.shape, dtype=np.float32)  # [2.0, 3.0]
    actual = np.array(result.outputs['result_14'])
    np.testing.assert_allclose(actual, expected, rtol=1e-6)


    # Verify resize - Test resize operation
    x_resize = np.array([[[[1.0, 2.0], [3.0, 4.0]]]], dtype=np.float32)  # Using x_resize from source
    scales_resize = np.array([2.0, 2.0], dtype=np.float32)  # Using scales_resize from source
    # Nearest neighbor upsampling: 2x2 -> 4x4
    expected = np.repeat(np.repeat(x_resize, 2, axis=2), 2, axis=3)
    actual = np.array(result.outputs['result_15'])
    np.testing.assert_allclose(actual, expected, rtol=1e-6)


    # Verify upsample - Test upsample operation (alias for resize)
    x_upsample = np.array([[[[1.0, 2.0], [3.0, 4.0]]]], dtype=np.float32)  # Using x_upsample from source
    scales_upsample = np.array([2.0, 2.0], dtype=np.float32)  # Using scales_upsample from source
    # Nearest neighbor upsampling: 2x2 -> 4x4
    expected = np.repeat(np.repeat(x_upsample, 2, axis=2), 2, axis=3)
    actual = np.array(result.outputs['result_16'])
    np.testing.assert_allclose(actual, expected, rtol=1e-6)