#!/usr/bin/env python3
"""
Accuracy tests for std::ml pooling operations against NumPy reference.
Split into smaller test functions for parallel execution.
"""

import numpy as np
from tests.test_utils import compile_and_execute


def test_global_pool_ops(compiler, runtime):
    """Test global_max_pool and global_average_pool across 1D/2D/3D spatial inputs."""
    source = """use std::ml;
    let x = [[[[1.0, 2.0], [3.0, 4.0]]]];
    let x_1d = [[[1.0, 2.0, 3.0, 4.0]]];
    let x_3d = [[[[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]]];
    let result_0 = std::ml::global_max_pool(x);
    let result_0_1d = std::ml::global_max_pool(x_1d);
    let result_0_3d = std::ml::global_max_pool(x_3d);
    let result_3 = std::ml::global_average_pool(x);
    let result_3_1d = std::ml::global_average_pool(x_1d);
    let result_3_3d = std::ml::global_average_pool(x_3d);
    """
    result = compile_and_execute(source, compiler, runtime)
    assert result.success, f"Execution failed: {result.errors}"

    x = np.array([[[[1.0, 2.0], [3.0, 4.0]]]], dtype=np.float32)
    x_1d = np.array([[[1.0, 2.0, 3.0, 4.0]]], dtype=np.float32)
    x_3d = np.array([[[[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]]], dtype=np.float32)

    np.testing.assert_allclose(np.array(result.outputs['result_0']),
                               np.max(x, axis=(-2, -1), keepdims=True), rtol=1e-6)
    np.testing.assert_allclose(np.array(result.outputs['result_0_1d']),
                               np.max(x_1d, axis=-1, keepdims=True), rtol=1e-6)
    np.testing.assert_allclose(np.array(result.outputs['result_0_3d']),
                               np.max(x_3d, axis=(-3, -2, -1), keepdims=True), rtol=1e-6)
    np.testing.assert_allclose(np.array(result.outputs['result_3']),
                               np.mean(x, axis=(-2, -1), keepdims=True), rtol=1e-6)
    np.testing.assert_allclose(np.array(result.outputs['result_3_1d']),
                               np.mean(x_1d, axis=-1, keepdims=True), rtol=1e-6)
    np.testing.assert_allclose(np.array(result.outputs['result_3_3d']),
                               np.mean(x_3d, axis=(-3, -2, -1), keepdims=True), rtol=1e-6)


def test_max_avg_pool_ops(compiler, runtime):
    """Test max_pool and average_pool (1D, 2D, 3D) and max_roi_pool."""
    source = """use std::ml;
    let x = [[[[1.0, 2.0], [3.0, 4.0]]]];
    let x_1d_pool = [[[1.0, 2.0, 3.0, 4.0, 5.0]]];
    let pool_1d = 2;
    let stride_1d = 2;
    let pad_1d = 0;
    let pool_h = 2;
    let pool_w = 2;
    let stride_h = 2;
    let stride_w = 2;
    let pad_h = 0;
    let pad_w = 0;
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
    let x_roi = [[[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]]];
    let rois = [[0, 0, 0, 2, 2]];
    let pooled_shape = [2, 2];
    let spatial_scale = 1.0;
    let result_max_pool_1d = std::ml::max_pool(x_1d_pool, [pool_1d], [stride_1d], [pad_1d]);
    let result_avg_pool_1d = std::ml::average_pool(x_1d_pool, [pool_1d], [stride_1d], [pad_1d]);
    let result_1 = std::ml::max_pool(x, [pool_h, pool_w], [stride_h, stride_w], [pad_h, pad_w]);
    let result_2 = std::ml::average_pool(x, [pool_h, pool_w], [stride_h, stride_w], [pad_h, pad_w]);
    let result_max_pool_3d = std::ml::max_pool(x_3d_pool, [pool_3d_d, pool_3d_h, pool_3d_w], [stride_3d_d, stride_3d_h, stride_3d_w], [pad_3d_d, pad_3d_h, pad_3d_w]);
    let result_avg_pool_3d = std::ml::average_pool(x_3d_pool, [pool_3d_d, pool_3d_h, pool_3d_w], [stride_3d_d, stride_3d_h, stride_3d_w], [pad_3d_d, pad_3d_h, pad_3d_w]);
    let result_5 = std::ml::max_roi_pool(x_roi, rois, pooled_shape, spatial_scale);
    """
    result = compile_and_execute(source, compiler, runtime)
    assert result.success, f"Execution failed: {result.errors}"

    np.testing.assert_allclose(np.array(result.outputs['result_max_pool_1d']),
                               np.array([[[2.0, 4.0]]], dtype=np.float32), rtol=1e-6)
    np.testing.assert_allclose(np.array(result.outputs['result_avg_pool_1d']),
                               np.array([[[1.5, 3.5]]], dtype=np.float32), rtol=1e-5)
    np.testing.assert_allclose(np.array(result.outputs['result_1']),
                               np.array([[[[4.0]]]]), rtol=1e-6)
    np.testing.assert_allclose(np.array(result.outputs['result_2']),
                               np.array([[[[2.5]]]]), rtol=1e-5)
    np.testing.assert_allclose(np.array(result.outputs['result_max_pool_3d']),
                               np.array([[[[[8.0]]]]], dtype=np.float32), rtol=1e-6)
    np.testing.assert_allclose(np.array(result.outputs['result_avg_pool_3d']),
                               np.array([[[[[4.5]]]]], dtype=np.float32), rtol=1e-5)
    x_roi = np.array([[[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]]], dtype=np.float32)
    expected_roi = x_roi[0, 0, 0:2, 0:2].reshape(1, 1, 2, 2)
    np.testing.assert_allclose(np.array(result.outputs['result_5']), expected_roi, rtol=1e-5)


def test_lp_pool_ops(compiler, runtime):
    """Test lp_pool (1D, 2D, 3D)."""
    source = """use std::ml;
    let x = [[[[1.0, 2.0], [3.0, 4.0]]]];
    let x_1d_pool = [[[1.0, 2.0, 3.0, 4.0, 5.0]]];
    let pool_1d = 2;
    let stride_1d = 2;
    let pad_1d = 0;
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
    let result_lp_pool_1d = std::ml::lp_pool(x_1d_pool, [pool_1d], [stride_1d], 2);
    let result_4 = std::ml::lp_pool(x, [2, 2], [1, 1], 2);
    let result_lp_pool_3d = std::ml::lp_pool(x_3d_pool, [pool_3d_d, pool_3d_h, pool_3d_w], [stride_3d_d, stride_3d_h, stride_3d_w], 2);
    """
    result = compile_and_execute(source, compiler, runtime)
    assert result.success, f"Execution failed: {result.errors}"

    x_1d_pool = np.array([[[1.0, 2.0, 3.0, 4.0, 5.0]]], dtype=np.float32)
    x = np.array([[[[1.0, 2.0], [3.0, 4.0]]]], dtype=np.float32)
    x_3d_pool = np.array([[[[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]]], dtype=np.float32)

    expected_lp_1d = np.array([[[np.sqrt(1**2 + 2**2), np.sqrt(3**2 + 4**2)]]], dtype=np.float32)
    np.testing.assert_allclose(np.array(result.outputs['result_lp_pool_1d']), expected_lp_1d, rtol=1e-5)
    np.testing.assert_allclose(np.array(result.outputs['result_4']),
                               np.sqrt(np.sum(x**2, axis=(-2, -1), keepdims=True)), rtol=1e-5)
    np.testing.assert_allclose(
        np.array(result.outputs['result_lp_pool_3d']),
        np.power(np.sum(np.power(np.abs(x_3d_pool), 2), axis=(-3, -2, -1), keepdims=True), 0.5),
        rtol=1e-5)
