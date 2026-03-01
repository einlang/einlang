#!/usr/bin/env python3
"""
Accuracy tests for std::ml indexing operations against NumPy reference.
Split into smaller test functions for parallel execution.
"""

import numpy as np
from tests.test_utils import compile_and_execute


def test_slice_gather_ops(compiler, runtime):
    """Test slice, gather, and gather_elements operations."""
    source = """use std::ml;
    let data_1d = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
    let data_2d = [[0.0, 1.0, 2.0, 3.0, 4.0, 5.0]];
    let indices_1d = [3, 1, 0];
    let data = [[10.0, 20.0, 30.0, 40.0]];
    let indices = [[3, 1, 0, 2]];
    let data_1d_gather = [1.0, 2.0, 3.0, 4.0, 5.0];
    let indices_1d_gather = [2, 0, 4];
    let data_3d_gather = [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]];
    let indices_3d_gather = [0, 1];
    let data_1d_gather_elements = [1.0, 2.0, 3.0, 4.0];
    let indices_1d_gather_elements = [2, 0, 3, 1];
    let data_3d_gather_elements = [[[1.0, 2.0], [3.0, 4.0]]];
    let indices_3d_gather_elements = [[[1, 0], [0, 1]]];
    let result_1d_2 = std::ml::slice(data_1d, [1], [5], [0], [2]);
    let result_2d_single_2 = std::ml::slice(data_2d, [1], [5], [1], [2]);
    let result_axis1_3 = std::ml::gather(data_2d, indices_1d, 1);
    let result_4 = std::ml::gather_elements(data, indices, 1);
    let result_gather_1d = std::ml::gather(data_1d_gather, indices_1d_gather, 0);
    let result_gather_3d = std::ml::gather(data_3d_gather, indices_3d_gather, 0);
    let result_gather_elements_1d = std::ml::gather_elements(data_1d_gather_elements, indices_1d_gather_elements, 0);
    let result_gather_elements_3d = std::ml::gather_elements(data_3d_gather_elements, indices_3d_gather_elements, 2);
    """
    result = compile_and_execute(source, compiler, runtime)
    assert result.success, f"Execution failed: {result.errors}"

    data_1d = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    np.testing.assert_allclose(np.array(result.outputs['result_1d_2']), data_1d[1:5:2], rtol=1e-6)
    data_2d = np.array([[0.0, 1.0, 2.0, 3.0, 4.0, 5.0]])
    np.testing.assert_allclose(np.array(result.outputs['result_2d_single_2']),
                               data_2d[:, 1:5:2], rtol=1e-6)
    data_1d_gather = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    np.testing.assert_allclose(np.array(result.outputs['result_gather_1d']),
                               np.take(data_1d_gather, [2, 0, 4], axis=0), rtol=1e-6)
    np.testing.assert_allclose(np.array(result.outputs['result_axis1_3']),
                               np.take(data_2d, [3, 1, 0], axis=1), rtol=1e-6)
    data_3d_gather = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], dtype=np.float32)
    np.testing.assert_allclose(np.array(result.outputs['result_gather_3d']),
                               np.take(data_3d_gather, [0, 1], axis=0), rtol=1e-6)
    data = np.array([[10.0, 20.0, 30.0, 40.0]], dtype=np.float32)
    indices = np.array([[3, 1, 0, 2]], dtype=np.int32)
    np.testing.assert_allclose(np.array(result.outputs['result_4']),
                               np.take_along_axis(data, indices, axis=1), rtol=1e-6)
    data_1d_ge = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    idx_1d_ge = np.array([2, 0, 3, 1], dtype=np.int32)
    expected_1d = np.take_along_axis(data_1d_ge.reshape(1, -1), idx_1d_ge.reshape(1, -1), axis=1).flatten()
    np.testing.assert_allclose(np.array(result.outputs['result_gather_elements_1d']),
                               expected_1d, rtol=1e-6)
    data_3d_ge = np.array([[[1.0, 2.0], [3.0, 4.0]]], dtype=np.float32)
    idx_3d_ge = np.array([[[1, 0], [0, 1]]], dtype=np.int32)
    np.testing.assert_allclose(np.array(result.outputs['result_gather_elements_3d']),
                               np.take_along_axis(data_3d_ge, idx_3d_ge, axis=2), rtol=1e-6)


def test_gather_nd_argmax_ops(compiler, runtime):
    """Test gather_nd, argmax, argmin, and onehot operations."""
    source = """use std::ml;
    let indices_1 = [0, 1, 2];
    let depth = 3;
    let x_argmax = [[1.0, 5.0, 3.0, 2.0]];
    let x_argmin = [[5.0, 1.0, 3.0, 2.0]];
    let data_gather_nd = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
    let indices_gather_nd = [[0, 1], [1, 2]];
    let data_gather_nd_3d = [[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]];
    let indices_gather_nd_3d = [[0, 1, 2], [1, 0, 1]];
    let result_5 = std::ml::onehot(indices_1, depth, [0.0, 1.0]);
    let result_6 = std::ml::argmax(x_argmax);
    let result_7 = std::ml::argmin(x_argmin);
    let result_8 = std::ml::gather_nd(data_gather_nd, indices_gather_nd);
    let result_8_3d = std::ml::gather_nd(data_gather_nd_3d, indices_gather_nd_3d);
    """
    result = compile_and_execute(source, compiler, runtime)
    assert result.success, f"Execution failed: {result.errors}"

    indices = np.array([0, 1, 2], dtype=np.int32)
    np.testing.assert_allclose(np.array(result.outputs['result_5']),
                               np.eye(3, dtype=np.float32)[indices], rtol=1e-6)
    np.testing.assert_array_equal(np.array(result.outputs['result_6']),
                                  np.argmax(np.array([[1.0, 5.0, 3.0, 2.0]]), axis=-1))
    np.testing.assert_array_equal(np.array(result.outputs['result_7']),
                                  np.argmin(np.array([[5.0, 1.0, 3.0, 2.0]]), axis=-1))
    data_nd = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    np.testing.assert_allclose(np.array(result.outputs['result_8']),
                               np.array([data_nd[0, 1], data_nd[1, 2]]), rtol=1e-6)
    data_nd_3d = np.array([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
                             [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]], dtype=np.float32)
    np.testing.assert_allclose(np.array(result.outputs['result_8_3d']),
                               np.array([data_nd_3d[0, 1, 2], data_nd_3d[1, 0, 1]]), rtol=1e-6)


def test_scatter_ops(compiler, runtime):
    """Test scatter, scatter_elements, and scatter_nd operations."""
    source = """use std::ml;
    let data_scatter = [[1.0, 2.0, 3.0]];
    let indices_scatter = [[0, 1, 0]];
    let updates_scatter = [[10.0, 20.0, 30.0]];
    let data_scatter_1d = [1.0, 2.0, 3.0, 4.0];
    let indices_scatter_1d = [2, 0];
    let updates_scatter_1d = [10.0, 20.0];
    let data_scatter_3d = [[[1.0, 2.0], [3.0, 4.0]]];
    let indices_scatter_3d = [[[1, 0], [0, 1]]];
    let updates_scatter_3d = [[[10.0, 20.0], [30.0, 40.0]]];
    let data_scatter_nd = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
    let indices_scatter_nd = [[0, 1], [1, 2]];
    let updates_scatter_nd = [100.0, 200.0];
    let data_scatter_nd_3d = [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]];
    let indices_scatter_nd_3d = [[0, 1, 1], [1, 0, 0]];
    let updates_scatter_nd_3d = [100.0, 200.0];
    let result_9 = std::ml::scatter_elements(data_scatter, indices_scatter, updates_scatter, 1);
    let result_10 = std::ml::scatter(data_scatter, indices_scatter, updates_scatter, 1);
    let result_scatter_1d = std::ml::scatter(data_scatter_1d, indices_scatter_1d, updates_scatter_1d, 0);
    let result_scatter_3d = std::ml::scatter(data_scatter_3d, indices_scatter_3d, updates_scatter_3d, 2);
    let result_11 = std::ml::scatter_nd(data_scatter_nd, indices_scatter_nd, updates_scatter_nd);
    let result_11_3d = std::ml::scatter_nd(data_scatter_nd_3d, indices_scatter_nd_3d, updates_scatter_nd_3d);
    """
    result = compile_and_execute(source, compiler, runtime)
    assert result.success, f"Execution failed: {result.errors}"

    np.testing.assert_allclose(np.array(result.outputs['result_9']),
                               np.array([[30.0, 20.0, 3.0]], dtype=np.float32), rtol=1e-6)
    np.testing.assert_allclose(np.array(result.outputs['result_10']),
                               np.array([[30.0, 20.0, 3.0]], dtype=np.float32), rtol=1e-6)

    expected_1d = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    expected_1d[2] = 10.0
    expected_1d[0] = 20.0
    np.testing.assert_allclose(np.array(result.outputs['result_scatter_1d']), expected_1d, rtol=1e-6)

    expected_3d = np.array([[[1.0, 2.0], [3.0, 4.0]]], dtype=np.float32)
    expected_3d[0, 0, 1] = 10.0
    expected_3d[0, 0, 0] = 20.0
    expected_3d[0, 1, 0] = 30.0
    expected_3d[0, 1, 1] = 40.0
    np.testing.assert_allclose(np.array(result.outputs['result_scatter_3d']), expected_3d, rtol=1e-6)

    expected_nd = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    expected_nd[0, 1] = 100.0
    expected_nd[1, 2] = 200.0
    np.testing.assert_allclose(np.array(result.outputs['result_11']), expected_nd, rtol=1e-6)

    expected_nd_3d = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], dtype=np.float32)
    expected_nd_3d[0, 1, 1] = 100.0
    expected_nd_3d[1, 0, 0] = 200.0
    np.testing.assert_allclose(np.array(result.outputs['result_11_3d']), expected_nd_3d, rtol=1e-6)


def test_nonzero_ops(compiler, runtime):
    """Test nonzero operation across 1D, 2D, and 3D inputs."""
    source = """use std::ml;
    let x_nonzero_1d = [0.0, 1.0, 0.0, 2.0, 0.0];
    let data_nonzero_2d = [[0.0, 1.0, 0.0], [2.0, 0.0, 3.0]];
    let data_nonzero_3d = [[[0.0, 1.0], [2.0, 0.0]], [[0.0, 0.0], [3.0, 4.0]]];
    let result_12_nonzero_1d = std::ml::nonzero(x_nonzero_1d);
    let result_13_nonzero_2d = std::ml::nonzero(data_nonzero_2d);
    let result_14_nonzero_3d = std::ml::nonzero(data_nonzero_3d);
    """
    result = compile_and_execute(source, compiler, runtime)
    assert result.success, f"Execution failed: {result.errors}"

    x_1d = np.array([0.0, 1.0, 0.0, 2.0, 0.0], dtype=np.float32)
    np.testing.assert_array_equal(np.array(result.outputs['result_12_nonzero_1d']),
                                  np.nonzero(x_1d)[0])

    data_2d = np.array([[0.0, 1.0, 0.0], [2.0, 0.0, 3.0]], dtype=np.float32)
    nonzero_2d = np.nonzero(data_2d)
    expected_2d = list(zip(nonzero_2d[0], nonzero_2d[1]))
    actual_2d = result.outputs['result_13_nonzero_2d']
    if isinstance(actual_2d, list):
        actual_tuples = [tuple(item) if isinstance(item, (list, np.ndarray)) else item
                         for item in actual_2d]
        assert len(actual_tuples) == len(expected_2d)
        for a, e in zip(actual_tuples, expected_2d):
            assert a == e

    data_3d = np.array([[[0.0, 1.0], [2.0, 0.0]], [[0.0, 0.0], [3.0, 4.0]]], dtype=np.float32)
    nonzero_3d = np.nonzero(data_3d)
    expected_3d = list(zip(nonzero_3d[0], nonzero_3d[1], nonzero_3d[2]))
    actual_3d = result.outputs['result_14_nonzero_3d']
    if isinstance(actual_3d, list):
        actual_tuples = [tuple(item) if isinstance(item, (list, np.ndarray)) else item
                         for item in actual_3d]
        assert len(actual_tuples) == len(expected_3d)
        for a, e in zip(actual_tuples, expected_3d):
            assert a == e
