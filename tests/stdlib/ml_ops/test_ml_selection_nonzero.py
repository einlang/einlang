#!/usr/bin/env python3
"""
Tests for std::ml::nonzero across ranks. Split from test_ml_selection for smaller modules.
"""

import numpy as np
from tests.test_utils import compile_and_execute


def test_nonzero_all_ranks(compiler, runtime):
    """Test nonzero operation across all supported ranks (1D, 2D, 3D)"""
    source = """use std::ml;
    // 1D
    let x_1d = [0.0, 1.0, 0.0, 2.0, 0.0, 3.0];
    let nonzero_1d = std::ml::nonzero(x_1d);

    // 2D
    let x_2d = [[0.0, 1.0, 0.0], [2.0, 0.0, 3.0], [0.0, 0.0, 0.0]];
    let nonzero_2d = std::ml::nonzero(x_2d);

    // 3D
    let x_3d = [[[0.0, 1.0], [2.0, 0.0]], [[0.0, 0.0], [3.0, 4.0]]];
    let nonzero_3d = std::ml::nonzero(x_3d);
    """
    result = compile_and_execute(source, compiler, runtime)
    assert result.success, f"Execution failed: {result.errors}"

    # Verify 1D
    x_1d = np.array([0.0, 1.0, 0.0, 2.0, 0.0, 3.0], dtype=np.float32)
    expected_1d = np.nonzero(x_1d)[0]
    actual_1d = np.array(result.outputs['nonzero_1d'])
    np.testing.assert_array_equal(actual_1d, expected_1d)

    # Verify 2D
    x_2d = np.array([[0.0, 1.0, 0.0], [2.0, 0.0, 3.0], [0.0, 0.0, 0.0]], dtype=np.float32)
    expected_2d_indices = np.nonzero(x_2d)
    # ONNX format: 2D array [rank, num_nonzero] where each column is an index tuple
    # NumPy nonzero returns (row_indices, col_indices), convert to ONNX format
    expected_2d_onnx = np.column_stack(expected_2d_indices).T  # Shape: [2, num_nonzero]

    # Our implementation returns array of (row, col) pairs — shape (num_nonzero, rank)
    actual_2d = result.outputs['nonzero_2d']
    actual_2d_arr = np.array(actual_2d)
    # Normalize to ONNX format [rank, num_nonzero]
    if actual_2d_arr.ndim == 2 and actual_2d_arr.shape[1] == 2:
        actual_2d_onnx = actual_2d_arr.T
    elif actual_2d_arr.ndim == 2 and actual_2d_arr.shape[0] == 2:
        actual_2d_onnx = actual_2d_arr
    elif actual_2d_arr.ndim == 1:
        num_nonzero = np.count_nonzero(x_2d)
        actual_2d_onnx = actual_2d_arr.reshape(2, -1)
    else:
        raise AssertionError(f"Unexpected shape for nonzero_2d: {actual_2d_arr.shape}")
    sort_idx_expected = np.lexsort(expected_2d_onnx[::-1])
    sort_idx_actual = np.lexsort(actual_2d_onnx[::-1])
    expected_sorted = expected_2d_onnx[:, sort_idx_expected]
    actual_sorted = actual_2d_onnx[:, sort_idx_actual]
    np.testing.assert_array_equal(actual_sorted, expected_sorted)

    # Verify 3D
    x_3d = np.array([[[0.0, 1.0], [2.0, 0.0]], [[0.0, 0.0], [3.0, 4.0]]], dtype=np.float32)
    expected_3d_indices = np.nonzero(x_3d)
    # ONNX format: 2D array [rank, num_nonzero] where each column is an index tuple
    expected_3d_onnx = np.column_stack(expected_3d_indices).T  # Shape: [3, num_nonzero]

    actual_3d = result.outputs['nonzero_3d']
    actual_3d_arr = np.array(actual_3d)
    # Normalize to ONNX format [rank, num_nonzero]
    if actual_3d_arr.ndim == 2 and actual_3d_arr.shape[1] == 3:
        actual_3d_onnx = actual_3d_arr.T
    elif actual_3d_arr.ndim == 2 and actual_3d_arr.shape[0] == 3:
        actual_3d_onnx = actual_3d_arr
    elif actual_3d_arr.ndim == 1:
        num_nonzero_3d = np.count_nonzero(x_3d)
        actual_3d_onnx = actual_3d_arr.reshape(3, -1)
    else:
        raise AssertionError(f"Unexpected shape for nonzero_3d: {actual_3d_arr.shape}")
    sort_idx_expected = np.lexsort(expected_3d_onnx[::-1])
    sort_idx_actual = np.lexsort(actual_3d_onnx[::-1])
    expected_sorted = expected_3d_onnx[:, sort_idx_expected]
    actual_sorted = actual_3d_onnx[:, sort_idx_actual]
    np.testing.assert_array_equal(actual_sorted, expected_sorted)
