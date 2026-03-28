#!/usr/bin/env python3
"""
Accuracy tests for std::ml pooling operations using grouped module fixtures.
"""

import numpy as np
import pytest

from tests.test_utils import compile_and_execute


@pytest.fixture(scope="module")
def global_pool_outputs(module_compiler, module_runtime):
    source = """use std::ml;
    let x_2d = [[[[1.0, 2.0], [3.0, 4.0]]]];
    let x_1d = [[[1.0, 2.0, 3.0, 4.0]]];
    let x_3d = [[[[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]]];
    let max_2d = std::ml::global_max_pool(x_2d);
    let max_1d = std::ml::global_max_pool(x_1d);
    let max_3d = std::ml::global_max_pool(x_3d);
    let avg_2d = std::ml::global_average_pool(x_2d);
    let avg_1d = std::ml::global_average_pool(x_1d);
    let avg_3d = std::ml::global_average_pool(x_3d);
    """
    result = compile_and_execute(source, module_compiler, module_runtime)
    assert result.success, f"Execution failed: {result.errors}"
    return result.outputs


@pytest.fixture(scope="module")
def pool_1d_2d_outputs(module_compiler, module_runtime):
    source = """use std::ml;
    let x_2d = [[[[1.0, 2.0], [3.0, 4.0]]]];
    let x_1d = [[[1.0, 2.0, 3.0, 4.0, 5.0]]];
    let max_1d = std::ml::max_pool(x_1d, [2], [2], [0]);
    let max_2d = std::ml::max_pool(x_2d, [2, 2], [2, 2], [0, 0]);
    let avg_1d = std::ml::average_pool(x_1d, [2], [2], [0]);
    let avg_2d = std::ml::average_pool(x_2d, [2, 2], [2, 2], [0, 0]);
    """
    result = compile_and_execute(source, module_compiler, module_runtime)
    assert result.success, f"Execution failed: {result.errors}"
    return result.outputs


@pytest.fixture(scope="module")
def pool_3d_roi_outputs(module_compiler, module_runtime):
    source = """use std::ml;
    let x_3d = [[[[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]]];
    let max_3d = std::ml::max_pool(x_3d, [2, 2, 2], [1, 1, 1], [0, 0, 0]);
    let avg_3d = std::ml::average_pool(x_3d, [2, 2, 2], [1, 1, 1], [0, 0, 0]);
    let x_roi = [[[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]]];
    let rois = [[0, 0, 0, 2, 2]];
    let pooled_shape = [2, 2];
    let roi = std::ml::max_roi_pool(x_roi, rois, pooled_shape, 1.0);
    """
    result = compile_and_execute(source, module_compiler, module_runtime)
    assert result.success, f"Execution failed: {result.errors}"
    return result.outputs


@pytest.fixture(scope="module")
def lp_pool_outputs(module_compiler, module_runtime):
    source = """use std::ml;
    let x_2d = [[[[1.0, 2.0], [3.0, 4.0]]]];
    let x_1d = [[[1.0, 2.0, 3.0, 4.0, 5.0]]];
    let lp_1d = std::ml::lp_pool(x_1d, [2], [2], 2);
    let lp_2d = std::ml::lp_pool(x_2d, [2, 2], [1, 1], 2);
    let x_3d = [[[[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]]];
    let lp_3d = std::ml::lp_pool(x_3d, [2, 2, 2], [1, 1, 1], 2);
    """
    result = compile_and_execute(source, module_compiler, module_runtime)
    assert result.success, f"Execution failed: {result.errors}"
    return result.outputs


def test_global_max_pool_ops(global_pool_outputs):
    x_2d = np.array([[[[1.0, 2.0], [3.0, 4.0]]]], dtype=np.float32)
    x_1d = np.array([[[1.0, 2.0, 3.0, 4.0]]], dtype=np.float32)
    x_3d = np.array([[[[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]]], dtype=np.float32)
    np.testing.assert_allclose(np.array(global_pool_outputs["max_2d"]), np.max(x_2d, axis=(-2, -1), keepdims=True), rtol=1e-6)
    np.testing.assert_allclose(np.array(global_pool_outputs["max_1d"]), np.max(x_1d, axis=-1, keepdims=True), rtol=1e-6)
    np.testing.assert_allclose(np.array(global_pool_outputs["max_3d"]), np.max(x_3d, axis=(-3, -2, -1), keepdims=True), rtol=1e-6)


def test_global_average_pool_ops(global_pool_outputs):
    x_2d = np.array([[[[1.0, 2.0], [3.0, 4.0]]]], dtype=np.float32)
    x_1d = np.array([[[1.0, 2.0, 3.0, 4.0]]], dtype=np.float32)
    x_3d = np.array([[[[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]]], dtype=np.float32)
    np.testing.assert_allclose(np.array(global_pool_outputs["avg_2d"]), np.mean(x_2d, axis=(-2, -1), keepdims=True), rtol=1e-6)
    np.testing.assert_allclose(np.array(global_pool_outputs["avg_1d"]), np.mean(x_1d, axis=-1, keepdims=True), rtol=1e-6)
    np.testing.assert_allclose(np.array(global_pool_outputs["avg_3d"]), np.mean(x_3d, axis=(-3, -2, -1), keepdims=True), rtol=1e-6)


def test_max_pool_1d_and_2d(pool_1d_2d_outputs):
    np.testing.assert_allclose(np.array(pool_1d_2d_outputs["max_1d"]), np.array([[[2.0, 4.0]]], dtype=np.float32), rtol=1e-6)
    np.testing.assert_allclose(np.array(pool_1d_2d_outputs["max_2d"]), np.array([[[[4.0]]]], dtype=np.float32), rtol=1e-6)


def test_average_pool_1d_and_2d(pool_1d_2d_outputs):
    np.testing.assert_allclose(np.array(pool_1d_2d_outputs["avg_1d"]), np.array([[[1.5, 3.5]]], dtype=np.float32), rtol=1e-5)
    np.testing.assert_allclose(np.array(pool_1d_2d_outputs["avg_2d"]), np.array([[[[2.5]]]], dtype=np.float32), rtol=1e-5)


def test_max_pool_3d(pool_3d_roi_outputs):
    np.testing.assert_allclose(np.array(pool_3d_roi_outputs["max_3d"]), np.array([[[[[8.0]]]]], dtype=np.float32), rtol=1e-6)


def test_average_pool_3d(pool_3d_roi_outputs):
    np.testing.assert_allclose(np.array(pool_3d_roi_outputs["avg_3d"]), np.array([[[[[4.5]]]]], dtype=np.float32), rtol=1e-5)


def test_max_roi_pool(pool_3d_roi_outputs):
    x = np.array([[[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]]], dtype=np.float32)
    expected = x[0, 0, 0:2, 0:2].reshape(1, 1, 2, 2)
    np.testing.assert_allclose(np.array(pool_3d_roi_outputs["roi"]), expected, rtol=1e-5)


def test_lp_pool_1d_and_2d(lp_pool_outputs):
    x_1d = np.array([[[1.0, 2.0, 3.0, 4.0, 5.0]]], dtype=np.float32)
    x_2d = np.array([[[[1.0, 2.0], [3.0, 4.0]]]], dtype=np.float32)
    expected_1d = np.array([[[np.sqrt(1**2 + 2**2), np.sqrt(3**2 + 4**2)]]], dtype=np.float32)
    expected_2d = np.sqrt(np.sum(x_2d**2, axis=(-2, -1), keepdims=True))
    np.testing.assert_allclose(np.array(lp_pool_outputs["lp_1d"]), expected_1d, rtol=1e-5)
    np.testing.assert_allclose(np.array(lp_pool_outputs["lp_2d"]), expected_2d, rtol=1e-5)


def test_lp_pool_3d(lp_pool_outputs):
    x = np.array([[[[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]]], dtype=np.float32)
    expected = np.power(np.sum(np.power(np.abs(x), 2), axis=(-3, -2, -1), keepdims=True), 0.5)
    np.testing.assert_allclose(np.array(lp_pool_outputs["lp_3d"]), expected, rtol=1e-5)
