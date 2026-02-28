#!/usr/bin/env python3
"""
Tests for std::ml::topk (TopK selection) against NumPy reference.
Split from test_ml_selection for smaller, faster-running modules.
"""

import numpy as np
from tests.test_utils import compile_and_execute


def test_topk_1d(compiler, runtime):
    """
    Test 1D TopK helper function using partition (quickselect).
    Tests the core 1D TopK logic before testing full 2D TopK.

    Note: Since topk_1d_helper is not exported, we test it indirectly through topk
    by using a 2D array with shape [1, N] and axis=1, which effectively tests the 1D helper.
    """
    source = """use std::ml;

    // Test 1D TopK via 2D array with single row (shape [1, N])
    // This effectively tests the 1D helper since topk transposes to innermost and calls the helper
    // ONNX-aligned: topk(X, k, axis) - always returns largest values
    let arr1_2d = [[64.0, 34.0, 25.0, 12.0, 22.0, 11.0, 90.0, 88.0, 45.0, 50.0]];
    let (vals1, idxs1) = std::ml::topk(arr1_2d, 3, 1);

    // Test 1D TopK via 2D - same test (ONNX always returns largest)
    let (vals2, idxs2) = std::ml::topk(arr1_2d, 3, 1);

    // Test with axis=-1 (last axis, same as axis=1 for 2D)
    let (vals3, idxs3) = std::ml::topk(arr1_2d, 3, -1);

    // Test with axis=0 (first axis)
    let (vals4, idxs4) = std::ml::topk(arr1_2d, 1, 0);

    // Test edge case: k = 1
    let (vals5, idxs5) = std::ml::topk(arr1_2d, 1, 1);

    // Test edge case: k = len(arr) (but topk only supports 2D, so we'll test with k = N)
    let (vals6, idxs6) = std::ml::topk(arr1_2d, 10, 1);

    // Test with different array
    let arr2_2d = [[1.0, 5.0, 3.0, 2.0, 4.0]];
    let (vals7, idxs7) = std::ml::topk(arr2_2d, 2, 1);
    """

    result = compile_and_execute(source, compiler, runtime)
    assert result.success, f"Execution failed: {result.errors}"

    arr1 = np.array([64.0, 34.0, 25.0, 12.0, 22.0, 11.0, 90.0, 88.0, 45.0, 50.0], dtype=np.float32)

    # Test 1: top 3 largest (ONNX always returns largest)
    vals1 = np.array(result.outputs['vals1'])
    idxs1 = np.array(result.outputs['idxs1'], dtype=np.int32)
    # For 2D input [1, N], output is [1, k]
    assert vals1.shape == (1, 3), f"Expected shape (1, 3), got {vals1.shape}"
    assert idxs1.shape == (1, 3), f"Expected shape (1, 3), got {idxs1.shape}"
    vals1 = vals1[0]  # Extract the single row
    idxs1 = idxs1[0]  # Extract the single row
    # Verify values are top 3 largest (sorted descending)
    expected_top3 = np.array([90.0, 88.0, 64.0])  # Top 3 largest, sorted descending
    np.testing.assert_allclose(vals1, expected_top3, rtol=1e-6)
    # Verify indices are valid
    assert np.all(idxs1 >= 0) and np.all(idxs1 < len(arr1)), "Indices out of range"
    # Verify values match array at indices
    for i, idx in enumerate(idxs1):
        assert abs(vals1[i] - arr1[idx]) < 1e-6, f"Value mismatch at index {i}: {vals1[i]} != {arr1[idx]}"

    # Test 2: same test (should be same values, sorted descending)
    vals2 = np.array(result.outputs['vals2'])
    idxs2 = np.array(result.outputs['idxs2'], dtype=np.int32)
    assert vals2.shape == (1, 3), f"Expected shape (1, 3), got {vals2.shape}"
    vals2 = vals2[0]  # Extract the single row
    # Values should match exactly (both sorted descending)
    np.testing.assert_allclose(vals2, vals1, rtol=1e-6)

    # Test 3: axis=-1 (same as axis=1 for 2D)
    vals3 = np.array(result.outputs['vals3'])
    idxs3 = np.array(result.outputs['idxs3'], dtype=np.int32)
    assert vals3.shape == (1, 3), f"Expected shape (1, 3), got {vals3.shape}"
    vals3 = vals3[0]  # Extract the single row
    # Should be same as test 1 (axis=-1 is same as axis=1 for 2D, sorted descending)
    np.testing.assert_allclose(vals3, vals1, rtol=1e-6)

    # Test 4: axis=0 (top 1 along first axis - for each column, find top 1 row)
    vals4 = np.array(result.outputs['vals4'])
    idxs4 = np.array(result.outputs['idxs4'], dtype=np.int32)
    assert vals4.shape == (1, 10), f"Expected shape (1, 10), got {vals4.shape}"  # [k, N] = [1, 10]
    vals4 = vals4[0]  # Extract the single row
    # For axis=0 with input [1, 10], we find top 1 row for each of 10 columns
    # Since there's only 1 row, we get all values from that row
    np.testing.assert_allclose(vals4, arr1, rtol=1e-6)

    # Test 5: k = 1 (top 1 largest)
    vals5 = np.array(result.outputs['vals5'])
    idxs5 = np.array(result.outputs['idxs5'], dtype=np.int32)
    assert vals5.shape == (1, 1), f"Expected shape (1, 1), got {vals5.shape}"
    vals5 = vals5[0, 0]  # Extract the single value
    idxs5 = idxs5[0, 0]  # Extract the single index
    assert vals5 == np.max(arr1), f"Expected max value {np.max(arr1)}, got {vals5}"
    assert arr1[idxs5] == np.max(arr1), "Index should point to max value"

    # Test 6: k = len(arr) (all values)
    vals6 = np.array(result.outputs['vals6'])
    idxs6 = np.array(result.outputs['idxs6'], dtype=np.int32)
    assert vals6.shape == (1, 10), f"Expected shape (1, 10), got {vals6.shape}"
    vals6 = vals6[0]  # Extract the single row
    # Should contain all values (sorted descending)
    expected_all_sorted = np.sort(arr1)[::-1]  # Sort descending
    np.testing.assert_allclose(vals6, expected_all_sorted, rtol=1e-6)

    # Test 7: different array (top 2 largest)
    arr2 = np.array([1.0, 5.0, 3.0, 2.0, 4.0], dtype=np.float32)
    vals7 = np.array(result.outputs['vals7'])
    idxs7 = np.array(result.outputs['idxs7'], dtype=np.int32)
    assert vals7.shape == (1, 2), f"Expected shape (1, 2), got {vals7.shape}"
    vals7 = vals7[0]  # Extract the single row
    idxs7 = idxs7[0]  # Extract the single row
    expected_top2 = np.array([5.0, 4.0])  # Top 2 largest, sorted descending
    np.testing.assert_allclose(vals7, expected_top2, rtol=1e-6)
    # Verify values match array at indices
    for i, idx in enumerate(idxs7):
        assert abs(vals7[i] - arr2[idx]) < 1e-6, f"Value mismatch at index {i}: {vals7[i]} != {arr2[idx]}"


def test_topk_2d_axis1_basic(compiler, runtime):
    """
    Test TopK along axis=1 (columns per row) - basic functionality.

    Responsibility: Verify that topk correctly finds top k largest values
    along the last axis (columns) for each row in a 2D tensor.
    Compares against exact NumPy reference computed using argsort.
    """
    source = """use std::ml;
    // Basic 2D case - top 2 along last axis (axis=1)
    // Input: [[1.0, 5.0, 3.0, 2.0], [4.0, 1.0, 6.0, 3.0]]
    // Row 0: top 2 of [1.0, 5.0, 3.0, 2.0] -> [5.0, 3.0] at indices [1, 2]
    // Row 1: top 2 of [4.0, 1.0, 6.0, 3.0] -> [6.0, 4.0] at indices [2, 0]
    let x_2d = [[1.0, 5.0, 3.0, 2.0], [4.0, 1.0, 6.0, 3.0]];
    let (values_2d, indices_2d) = std::ml::topk(x_2d, 2, 1);
    """

    result = compile_and_execute(source, compiler, runtime)
    assert result.success, f"Execution failed: {result.errors}"

    # Compute exact reference using NumPy argsort
    x_2d = np.array([[1.0, 5.0, 3.0, 2.0], [4.0, 1.0, 6.0, 3.0]], dtype=np.float32)
    values_2d_actual = np.array(result.outputs['values_2d'])
    indices_2d_actual = np.array(result.outputs['indices_2d'], dtype=np.int32)

    # Reference: top k along axis=1 using argsort (already sorted descending)
    # argsort(-x) gives indices sorted by descending value, so [:k] gives top k indices in sorted order
    sorted_indices_ref = np.argsort(-x_2d, axis=1, kind='stable')[:, :2]
    # Extract values at sorted indices - they're already in descending order
    values_2d_ref = np.take_along_axis(x_2d, sorted_indices_ref, axis=1)

    # Verify shapes
    assert values_2d_actual.shape == values_2d_ref.shape
    assert indices_2d_actual.shape == sorted_indices_ref.shape

    # Compare with exact reference (both already sorted descending)
    np.testing.assert_allclose(values_2d_actual, values_2d_ref, rtol=1e-6)

    # Verify indices: use vectorized operations to check all indices at once
    # Create index arrays for advanced indexing
    row_indices = np.arange(values_2d_actual.shape[0])[:, None]
    col_indices = indices_2d_actual
    # Extract values at indices using advanced indexing
    values_at_indices = x_2d[row_indices, col_indices]
    # Compare entire arrays
    np.testing.assert_allclose(values_2d_actual, values_at_indices, rtol=1e-6,
        err_msg="Values don't match values at indices")
    # Verify indices are in valid range
    assert np.all((indices_2d_actual >= 0) & (indices_2d_actual < x_2d.shape[1])), \
        "Indices out of range"


def test_topk_2d_axis_neg1(compiler, runtime):
    """
    Test TopK with axis=-1 (negative axis normalization).

    Responsibility: Verify that axis=-1 correctly normalizes to the last axis
    (same as axis=1 for 2D tensors). Should produce identical results to axis=1.
    """
    source = """use std::ml;
    let x_2d = [[1.0, 5.0, 3.0, 2.0], [4.0, 1.0, 6.0, 3.0]];
    let (values_1, indices_1) = std::ml::topk(x_2d, 2, 1);
    let (values_neg1, indices_neg1) = std::ml::topk(x_2d, 2, -1);
    """

    result = compile_and_execute(source, compiler, runtime)
    assert result.success, f"Execution failed: {result.errors}"

    values_1 = np.array(result.outputs['values_1'])
    values_neg1 = np.array(result.outputs['values_neg1'])

    # Compare values (axis=-1 should match axis=1, both sorted descending)
    np.testing.assert_allclose(values_1, values_neg1, rtol=1e-6,
        err_msg="axis=-1 should produce same results as axis=1")


def test_topk_2d_axis0_basic(compiler, runtime):
    """
    Test TopK along axis=0 (rows per column) - basic functionality.

    Responsibility: Verify that topk correctly finds top k largest values
    along the first axis (rows) for each column in a 2D tensor.
    Compares against exact NumPy reference computed using argsort.
    """
    source = """use std::ml;
    // TopK along axis=0 (top k rows per column)
    // Input: [[1.0, 5.0], [4.0, 1.0], [3.0, 6.0]]
    // Column 0: top 2 of [1.0, 4.0, 3.0] -> [4.0, 3.0] at indices [1, 2]
    // Column 1: top 2 of [5.0, 1.0, 6.0] -> [6.0, 5.0] at indices [2, 0]
    let x_axis0 = [[1.0, 5.0], [4.0, 1.0], [3.0, 6.0]];
    let (values_axis0, indices_axis0) = std::ml::topk(x_axis0, 2, 0);
    """

    result = compile_and_execute(source, compiler, runtime)
    assert result.success, f"Execution failed: {result.errors}"

    # Compute exact reference using NumPy argsort
    x_axis0 = np.array([[1.0, 5.0], [4.0, 1.0], [3.0, 6.0]], dtype=np.float32)
    values_axis0_actual = np.array(result.outputs['values_axis0'])
    indices_axis0_actual = np.array(result.outputs['indices_axis0'], dtype=np.int32)

    # Reference: top k along axis=0 using argsort (already sorted descending)
    # argsort(-x) gives indices sorted by descending value, so [:k] gives top k indices in sorted order
    sorted_indices_ref = np.argsort(-x_axis0, axis=0, kind='stable')[:2, :]
    # Extract values at sorted indices - they're already in descending order
    values_axis0_ref = np.take_along_axis(x_axis0, sorted_indices_ref, axis=0)

    # Verify shapes
    assert values_axis0_actual.shape == values_axis0_ref.shape
    assert indices_axis0_actual.shape == sorted_indices_ref.shape

    # Compare with exact reference (both already sorted descending)
    np.testing.assert_allclose(values_axis0_actual, values_axis0_ref, rtol=1e-6)

    # Verify indices: use vectorized operations
    row_indices = indices_axis0_actual
    col_indices = np.arange(values_axis0_actual.shape[1])[None, :]
    values_at_indices = x_axis0[row_indices, col_indices]
    np.testing.assert_allclose(values_axis0_actual, values_at_indices, rtol=1e-6,
        err_msg="Values don't match values at indices")
    assert np.all((indices_axis0_actual >= 0) & (indices_axis0_actual < x_axis0.shape[0])), \
        "Indices out of range"


def test_topk_2d_axis0_k1(compiler, runtime):
    """
    Test TopK along axis=0 with k=1 (max per column).

    Responsibility: Verify that k=1 correctly returns the maximum value
    per column along axis=0. Should match NumPy's argmax behavior.
    """
    source = """use std::ml;
    let x_axis0 = [[1.0, 5.0], [4.0, 1.0], [3.0, 6.0]];
    let (values_axis0_k1, indices_axis0_k1) = std::ml::topk(x_axis0, 1, 0);
    """

    result = compile_and_execute(source, compiler, runtime)
    assert result.success, f"Execution failed: {result.errors}"

    x_axis0 = np.array([[1.0, 5.0], [4.0, 1.0], [3.0, 6.0]], dtype=np.float32)
    values_axis0_k1_actual = np.array(result.outputs['values_axis0_k1'])
    indices_axis0_k1_actual = np.array(result.outputs['indices_axis0_k1'], dtype=np.int32)

    # Reference: top 1 along axis=0 (max per column)
    sorted_indices_ref = np.argsort(-x_axis0, axis=0, kind='stable')[:1, :]
    values_axis0_k1_ref = np.take_along_axis(x_axis0, sorted_indices_ref, axis=0)

    # For k=1, should be exact match
    np.testing.assert_allclose(values_axis0_k1_actual, values_axis0_k1_ref, rtol=1e-6)
    np.testing.assert_array_equal(indices_axis0_k1_actual, sorted_indices_ref)


def test_topk_2d_k_equals_dim(compiler, runtime):
    """
    Test TopK when k equals dimension size (returns all values).

    Responsibility: Verify that when k equals the dimension size along the axis,
    topk returns all values (may be in different order). Should contain all
    original values when sorted.
    """
    source = """use std::ml;
    let x_3col = [[1.0, 5.0, 3.0], [4.0, 1.0, 6.0]];
    let (values_k3, indices_k3) = std::ml::topk(x_3col, 3, 1);
    """

    result = compile_and_execute(source, compiler, runtime)
    assert result.success, f"Execution failed: {result.errors}"

    x_3col = np.array([[1.0, 5.0, 3.0], [4.0, 1.0, 6.0]], dtype=np.float32)
    values_k3_actual = np.array(result.outputs['values_k3'])
    indices_k3_actual = np.array(result.outputs['indices_k3'], dtype=np.int32)

    # Reference: top 3 along axis=1 (all values)
    sorted_indices_ref = np.argsort(-x_3col, axis=1, kind='stable')[:, :3]
    values_k3_ref = np.take_along_axis(x_3col, sorted_indices_ref, axis=1)

    # Compare arrays (should contain all original values when sorted)
    # Both actual and reference are sorted descending per ONNX
    x_3col_sorted = np.sort(x_3col, axis=1)
    values_k3_actual_sorted = np.sort(values_k3_actual, axis=1)
    np.testing.assert_allclose(values_k3_actual_sorted, x_3col_sorted, rtol=1e-6,
        err_msg="Should contain all original values")

    # Verify indices
    row_indices = np.arange(values_k3_actual.shape[0])[:, None]
    col_indices = indices_k3_actual
    values_at_indices = x_3col[row_indices, col_indices]
    np.testing.assert_allclose(values_k3_actual, values_at_indices, rtol=1e-6)
    assert np.all((indices_k3_actual >= 0) & (indices_k3_actual < x_3col.shape[1])), \
        "Indices out of range"


def test_topk_2d_single_row(compiler, runtime):
    """
    Test TopK with single row input (edge case).

    Responsibility: Verify that topk works correctly with single-row 2D tensors.
    Should behave the same as multi-row case but with shape [1, k].
    """
    source = """use std::ml;
    let x_single_row = [[1.0, 5.0, 3.0, 2.0]];
    let (values_single, indices_single) = std::ml::topk(x_single_row, 2, 1);
    """

    result = compile_and_execute(source, compiler, runtime)
    assert result.success, f"Execution failed: {result.errors}"

    x_single_row = np.array([[1.0, 5.0, 3.0, 2.0]], dtype=np.float32)
    values_single_actual = np.array(result.outputs['values_single'])
    indices_single_actual = np.array(result.outputs['indices_single'], dtype=np.int32)

    # Reference: top 2 along axis=1 (already sorted descending)
    # argsort(-x) gives indices sorted by descending value, so [:k] gives top k indices in sorted order
    sorted_indices_ref = np.argsort(-x_single_row, axis=1, kind='stable')[:, :2]
    # Extract values at sorted indices - they're already in descending order
    values_single_ref = np.take_along_axis(x_single_row, sorted_indices_ref, axis=1)

    # Compare with exact reference (both already sorted descending)
    np.testing.assert_allclose(values_single_actual, values_single_ref, rtol=1e-6)

    # Verify indices
    row_indices = np.arange(values_single_actual.shape[0])[:, None]
    col_indices = indices_single_actual
    values_at_indices = x_single_row[row_indices, col_indices]
    np.testing.assert_allclose(values_single_actual, values_at_indices, rtol=1e-6)
    assert np.all((indices_single_actual >= 0) & (indices_single_actual < x_single_row.shape[1])), \
        "Indices out of range"
