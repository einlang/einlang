#!/usr/bin/env python3
"""
Accuracy tests for std::ml transform operations against NumPy reference.
Split into smaller test functions for parallel execution.
"""

import numpy as np
from tests.test_utils import compile_and_execute


def test_transform_basic_ops(compiler, runtime):
    """Test transpose, pad, flatten, concat, and tile operations."""
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
    let result_0 = std::ml::transpose(x);
    let result_1d_1 = std::ml::pad(data_1d, pads, 0.0);
    let result_2d_1 = std::ml::pad(data_2d, pads, 0.0);
    let result_2 = std::ml::flatten(x_1);
    let result_3 = std::ml::concat(a, b);
    let result_4 = std::ml::tile(x_2, repeats);
    """
    result = compile_and_execute(source, compiler, runtime)
    assert result.success, f"Execution failed: {result.errors}"

    x = np.array([[[1.0, 2.0], [3.0, 4.0]]])
    np.testing.assert_allclose(np.array(result.outputs['result_0']),
                               np.transpose(x, (0, 2, 1)), rtol=1e-6)
    data_1d = np.array([1.0, 2.0, 3.0])
    np.testing.assert_allclose(np.array(result.outputs['result_1d_1']),
                               np.pad(data_1d, 1, mode='constant', constant_values=0.0), rtol=1e-6)
    data_2d = np.array([[1.0, 2.0], [3.0, 4.0]])
    np.testing.assert_allclose(np.array(result.outputs['result_2d_1']),
                               np.pad(data_2d, 1, mode='constant', constant_values=0.0), rtol=1e-6)
    x_1 = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    np.testing.assert_allclose(np.array(result.outputs['result_2']), x_1.flatten(), rtol=1e-6)
    a = np.array([[1.0, 2.0]], dtype=np.float32)
    b = np.array([[3.0, 4.0]], dtype=np.float32)
    np.testing.assert_allclose(np.array(result.outputs['result_3']),
                               np.concatenate([a, b], axis=-1), rtol=1e-6)
    x_2 = np.array([[1.0, 2.0]], dtype=np.float32)
    np.testing.assert_allclose(np.array(result.outputs['result_4']),
                               np.tile(x_2, (1, 3)), rtol=1e-6)


def test_transform_depth_reshape_ops(compiler, runtime):
    """Test depth_to_space, space_to_depth, reshape, and squeeze operations."""
    source = """use std::ml;
    let x_3 = [[[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]], [[9.0, 10.0], [11.0, 12.0]], [[13.0, 14.0], [15.0, 16.0]]]];
    let x_4 = [[[[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0], [13.0, 14.0, 15.0, 16.0]]]];
    let x_5 = [[1.0, 2.0, 3.0, 4.0]];
    let x_6 = [[[[1.0, 2.0]]]];
    let axes = [2];
    let x_6_1d = [[1.0, 2.0]];
    let axes_1d = [0];
    let x_6_3d = [[[[[1.0, 2.0]]]]];
    let axes_3d = [2];
    let x_6_4d = [[[[1.0, 2.0]]]];
    let axes_4d = [2];
    let x_6_5d = [[[[[1.0, 2.0]]]]];
    let axes_5d = [2];
    let result_5 = std::ml::depth_to_space(x_3, 2);
    let result_6 = std::ml::space_to_depth(x_4, 2);
    let result_7 = std::ml::reshape(x_5, [2, 2]);
    let result_8 = std::ml::squeeze(x_6, axes);
    let result_8_1d = std::ml::squeeze(x_6_1d, axes_1d);
    let result_8_3d = std::ml::squeeze(x_6_3d, axes_3d);
    let result_8_4d = std::ml::squeeze(x_6_4d, axes_4d);
    let result_8_5d = std::ml::squeeze(x_6_5d, axes_5d);
    """
    result = compile_and_execute(source, compiler, runtime)
    assert result.success, f"Execution failed: {result.errors}"

    r = 2
    x_3 = np.array([[[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]],
                      [[9.0, 10.0], [11.0, 12.0]], [[13.0, 14.0], [15.0, 16.0]]]], dtype=np.float32)
    expected_depth = np.zeros((1, 1, 4, 4), dtype=np.float32)
    for b in range(1):
        for c in range(1):
            for i in range(4):
                for j in range(4):
                    c_idx = c * (r * r) + (i % r) * r + (j % r)
                    expected_depth[b, c, i, j] = x_3[b, c_idx, i // r, j // r]
    np.testing.assert_allclose(np.array(result.outputs['result_5']), expected_depth, rtol=1e-6)

    x_4 = np.array([[[[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0],
                       [9.0, 10.0, 11.0, 12.0], [13.0, 14.0, 15.0, 16.0]]]], dtype=np.float32)
    expected_space = np.zeros((1, 4, 2, 2), dtype=np.float32)
    for b in range(1):
        for c in range(1):
            for i in range(4):
                for j in range(4):
                    c_out = c * (r * r) + (i % r) * r + (j % r)
                    expected_space[b, c_out, i // r, j // r] = x_4[b, c, i, j]
    np.testing.assert_allclose(np.array(result.outputs['result_6']), expected_space, rtol=1e-6)

    x_5 = np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32)
    np.testing.assert_allclose(np.array(result.outputs['result_7']), x_5.reshape(2, 2), rtol=1e-6)

    np.testing.assert_allclose(np.array(result.outputs['result_8']),
                               np.squeeze(np.array([[[[1.0, 2.0]]]], dtype=np.float32), axis=2), rtol=1e-6)
    np.testing.assert_allclose(np.array(result.outputs['result_8_1d']),
                               np.squeeze(np.array([[1.0, 2.0]], dtype=np.float32), axis=0), rtol=1e-6)
    np.testing.assert_allclose(np.array(result.outputs['result_8_3d']),
                               np.squeeze(np.array([[[[[1.0, 2.0]]]]], dtype=np.float32), axis=2), rtol=1e-6)
    np.testing.assert_allclose(np.array(result.outputs['result_8_4d']),
                               np.squeeze(np.array([[[[1.0, 2.0]]]], dtype=np.float32), axis=2), rtol=1e-6)
    np.testing.assert_allclose(np.array(result.outputs['result_8_5d']),
                               np.squeeze(np.array([[[[[1.0, 2.0]]]]], dtype=np.float32), axis=2), rtol=1e-6)


def test_transform_unsqueeze_range_split_ops(compiler, runtime):
    """Test unsqueeze, range, constant_of_shape, and split operations."""
    source = """use std::ml;
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
    let data_split_4d = [[[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]];
    let split_sizes_4d = [1];
    let result_9 = std::ml::unsqueeze(x_7, axes_1);
    let result_9_1d = std::ml::unsqueeze(x_7_1d, axes_1_1d);
    let result_9_3d = std::ml::unsqueeze(x_7_3d, axes_1_3d);
    let result_10 = std::ml::range(start, limit, delta);
    let result_11 = std::ml::constant_of_shape(shape_val, value);
    let result_12 = std::ml::split(data_split, split_sizes, 0);
    let result_12_1d = std::ml::split(data_split_1d, split_sizes_1d, 0);
    let result_12_3d = std::ml::split(data_split_3d, split_sizes_3d, 0);
    let result_12_4d = std::ml::split(data_split_4d, split_sizes_4d, 0);
    """
    result = compile_and_execute(source, compiler, runtime)
    assert result.success, f"Execution failed: {result.errors}"

    x_7 = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
    np.testing.assert_allclose(np.array(result.outputs['result_9']),
                               np.expand_dims(x_7, axis=1), rtol=1e-6)
    x_7_1d = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    np.testing.assert_allclose(np.array(result.outputs['result_9_1d']),
                               np.expand_dims(x_7_1d, axis=0), rtol=1e-6)
    x_7_3d = np.array([[[1.0, 2.0], [3.0, 4.0]]], dtype=np.float32)
    np.testing.assert_allclose(np.array(result.outputs['result_9_3d']),
                               np.expand_dims(x_7_3d, axis=1), rtol=1e-6)
    np.testing.assert_allclose(np.array(result.outputs['result_10']),
                               np.arange(0.0, 5.0, 1.0, dtype=np.float32), rtol=1e-6)
    np.testing.assert_allclose(np.array(result.outputs['result_11']),
                               np.full([5], 42.0, dtype=np.float32), rtol=1e-6)

    data_split = np.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]], dtype=np.float32)
    np.testing.assert_allclose(np.array(result.outputs['result_12']), data_split[:2, :], rtol=1e-6)
    data_split_1d = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=np.float32)
    np.testing.assert_allclose(np.array(result.outputs['result_12_1d']), data_split_1d[:3], rtol=1e-6)
    data_split_3d = np.array([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
                               [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]], dtype=np.float32)
    np.testing.assert_allclose(np.array(result.outputs['result_12_3d']),
                               data_split_3d[:1, :, :], rtol=1e-6)
    data_split_4d = np.array([[[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]], dtype=np.float32)
    np.testing.assert_allclose(np.array(result.outputs['result_12_4d']),
                               data_split_4d[:1, :, :, :], rtol=1e-6)


def test_transform_expand_scale_ops(compiler, runtime):
    """Test expand, shape, resize, and upsample operations."""
    source = """use std::ml;
    let data_expand = [[1.0, 2.0]];
    let expand_shape = [3, 2];
    let data_expand_1d = [1.0, 2.0, 3.0];
    let expand_shape_1d = [2, 3];
    let data_expand_3d = [[[1.0, 2.0], [3.0, 4.0]]];
    let expand_shape_3d = [2, 2, 2];
    let data_expand_2d_to_3d = [[1.0, 2.0], [3.0, 4.0]];
    let expand_shape_2d_to_3d = [2, 2, 2];
    let data_expand_3d_to_4d = [[[1.0, 2.0], [3.0, 4.0]]];
    let expand_shape_3d_to_4d = [2, 1, 2, 2];
    let data_shape = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
    let x_resize = [[[[1.0, 2.0], [3.0, 4.0]]]];
    let scales_resize = [2.0, 2.0];
    let x_upsample = [[[[1.0, 2.0], [3.0, 4.0]]]];
    let scales_upsample = [2.0, 2.0];
    let result_13 = std::ml::expand(data_expand, expand_shape);
    let result_13_1d = std::ml::expand(data_expand_1d, expand_shape_1d);
    let result_13_3d = std::ml::expand(data_expand_3d, expand_shape_3d);
    let result_13_2d_to_3d = std::ml::expand(data_expand_2d_to_3d, expand_shape_2d_to_3d);
    let result_13_3d_to_4d = std::ml::expand(data_expand_3d_to_4d, expand_shape_3d_to_4d);
    let result_14 = std::ml::shape(data_shape);
    let result_15 = std::ml::resize(x_resize, scales_resize, "nearest");
    let result_16 = std::ml::upsample(x_upsample, scales_upsample, "nearest");
    """
    result = compile_and_execute(source, compiler, runtime)
    assert result.success, f"Execution failed: {result.errors}"

    np.testing.assert_allclose(
        np.array(result.outputs['result_13']),
        np.broadcast_to(np.array([[1.0, 2.0]], dtype=np.float32), [3, 2]), rtol=1e-6)
    np.testing.assert_allclose(
        np.array(result.outputs['result_13_1d']),
        np.broadcast_to(np.array([1.0, 2.0, 3.0], dtype=np.float32), [2, 3]), rtol=1e-6)
    np.testing.assert_allclose(
        np.array(result.outputs['result_13_3d']),
        np.broadcast_to(np.array([[[1.0, 2.0], [3.0, 4.0]]], dtype=np.float32), [2, 2, 2]), rtol=1e-6)
    np.testing.assert_allclose(
        np.array(result.outputs['result_13_2d_to_3d']),
        np.broadcast_to(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32), [2, 2, 2]), rtol=1e-6)
    np.testing.assert_allclose(
        np.array(result.outputs['result_13_3d_to_4d']),
        np.broadcast_to(np.array([[[1.0, 2.0], [3.0, 4.0]]], dtype=np.float32), [2, 1, 2, 2]), rtol=1e-6)

    data_shape = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    np.testing.assert_allclose(np.array(result.outputs['result_14']),
                               np.array(data_shape.shape, dtype=np.float32), rtol=1e-6)

    x_resize = np.array([[[[1.0, 2.0], [3.0, 4.0]]]], dtype=np.float32)
    expected_resize = np.repeat(np.repeat(x_resize, 2, axis=2), 2, axis=3)
    np.testing.assert_allclose(np.array(result.outputs['result_15']), expected_resize, rtol=1e-6)
    np.testing.assert_allclose(np.array(result.outputs['result_16']), expected_resize, rtol=1e-6)
