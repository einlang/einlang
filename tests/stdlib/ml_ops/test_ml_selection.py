#!/usr/bin/env python3
"""
Comprehensive accuracy tests for std::ml selection operations against ONNX/NumPy reference implementations.
Tests quick_select (via std::array::topk_extract) and prepares for topk implementation.
"""

import pytest
import numpy as np
from tests.test_utils import compile_and_execute, assert_float_close


# ===========================================================================
# Selection Operation Tests
# Clustered tests for efficiency - all selection ops tested together
# Compile/execute errors are failures, not skips
# ===========================================================================

def test_quickselect_partition(compiler, runtime):
    """
    Test quickselect partition function - core of quickselect algorithm.
    
    Note: There's no standalone quick_select function in stdlib.
    The quickselect algorithm is used internally in topk_extract.
    This test demonstrates quickselect behavior by testing the partition
    function which is the core building block of quickselect.
    """
    source = """use std::array;
    
    # Test partition function - core of quickselect
    let arr_partition = [64.0, 34.0, 25.0, 12.0, 22.0, 11.0, 90.0, 88.0, 45.0, 50.0];
    let pivot_partition = 45.0;
    let (smaller, equal, larger) = std::array::partition(arr_partition, pivot_partition);
    
    # Test partition with different pivot
    let pivot_partition2 = 30.0;
    let (smaller2, equal2, larger2) = std::array::partition(arr_partition, pivot_partition2);
    
    # Test partition with edge cases
    let arr_empty = [];
    let (smaller_empty, equal_empty, larger_empty) = std::array::partition(arr_empty, 5.0);
    
    let arr_single = [42.0];
    let (smaller_single, equal_single, larger_single) = std::array::partition(arr_single, 42.0);
    
    # Note: partition only works on 1D arrays. For 2D arrays, flatten first.
    # Test partition with flattened 2D array (1D version)
    let arr_2d_flat = [64.0, 34.0, 25.0, 12.0, 22.0, 11.0, 90.0, 88.0, 45.0];
    let pivot_2d = 30.0;
    let (smaller_2d, equal_2d, larger_2d) = std::array::partition(arr_2d_flat, pivot_2d);
    """
    
    result = compile_and_execute(source, compiler, runtime)
    assert result.success, f"Execution failed: {result.errors}"
    
    # Verify partition
    arr_partition = np.array([64.0, 34.0, 25.0, 12.0, 22.0, 11.0, 90.0, 88.0, 45.0, 50.0], dtype=np.float32)
    pivot = 45.0
    
    smaller_actual = np.array(result.outputs['smaller'])
    equal_actual = np.array(result.outputs['equal'])
    larger_actual = np.array(result.outputs['larger'])
    
    # Verify partition correctness
    expected_smaller = arr_partition[arr_partition < pivot]
    expected_equal = arr_partition[arr_partition == pivot]
    expected_larger = arr_partition[arr_partition > pivot]
    
    np.testing.assert_allclose(np.sort(smaller_actual), np.sort(expected_smaller), rtol=1e-6)
    np.testing.assert_allclose(np.sort(equal_actual), np.sort(expected_equal), rtol=1e-6)
    np.testing.assert_allclose(np.sort(larger_actual), np.sort(expected_larger), rtol=1e-6)
    
    # Verify partition with different pivot
    pivot2 = 30.0
    smaller2_actual = np.array(result.outputs['smaller2'])
    equal2_actual = np.array(result.outputs['equal2'])
    larger2_actual = np.array(result.outputs['larger2'])
    
    expected_smaller2 = arr_partition[arr_partition < pivot2]
    expected_equal2 = arr_partition[arr_partition == pivot2]
    expected_larger2 = arr_partition[arr_partition > pivot2]
    
    np.testing.assert_allclose(np.sort(smaller2_actual), np.sort(expected_smaller2), rtol=1e-6)
    np.testing.assert_allclose(np.sort(equal2_actual), np.sort(expected_equal2), rtol=1e-6)
    np.testing.assert_allclose(np.sort(larger2_actual), np.sort(expected_larger2), rtol=1e-6)
    
    # Verify empty array partition
    smaller_empty_actual = np.array(result.outputs['smaller_empty'])
    equal_empty_actual = np.array(result.outputs['equal_empty'])
    larger_empty_actual = np.array(result.outputs['larger_empty'])
    
    assert len(smaller_empty_actual) == 0, "Empty array should return empty smaller"
    assert len(equal_empty_actual) == 0, "Empty array should return empty equal"
    assert len(larger_empty_actual) == 0, "Empty array should return empty larger"
    
    # Verify single element partition
    smaller_single_actual = np.array(result.outputs['smaller_single'])
    equal_single_actual = np.array(result.outputs['equal_single'])
    larger_single_actual = np.array(result.outputs['larger_single'])
    
    assert len(smaller_single_actual) == 0, "Single element equal to pivot should have empty smaller"
    assert len(equal_single_actual) == 1 and equal_single_actual[0] == 42.0, "Single element equal to pivot should be in equal"
    assert len(larger_single_actual) == 0, "Single element equal to pivot should have empty larger"
    
    # Verify partition on flattened 2D array (partition only works on 1D)
    # This demonstrates that for 2D arrays, you need to flatten first
    arr_2d_flat = np.array([64.0, 34.0, 25.0, 12.0, 22.0, 11.0, 90.0, 88.0, 45.0], dtype=np.float32)
    pivot_2d = 30.0
    
    smaller_2d_actual = np.array(result.outputs['smaller_2d'])
    equal_2d_actual = np.array(result.outputs['equal_2d'])
    larger_2d_actual = np.array(result.outputs['larger_2d'])
    
    expected_smaller_2d = arr_2d_flat[arr_2d_flat < pivot_2d]
    expected_equal_2d = arr_2d_flat[arr_2d_flat == pivot_2d]
    expected_larger_2d = arr_2d_flat[arr_2d_flat > pivot_2d]
    
    np.testing.assert_allclose(np.sort(smaller_2d_actual), np.sort(expected_smaller_2d), rtol=1e-6)
    np.testing.assert_allclose(np.sort(equal_2d_actual), np.sort(expected_equal_2d), rtol=1e-6)
    np.testing.assert_allclose(np.sort(larger_2d_actual), np.sort(expected_larger_2d), rtol=1e-6)


def test_selection_clustered_accuracy(compiler, runtime):
    """Test selection operations - clustered for efficiency"""
    source = """use std::array;
    use std::ml;
    
    # Test data for quick_select (via topk_extract)
    # The quickselect algorithm is used internally in topk_extract
    let arr_1d = [64.0, 34.0, 25.0, 12.0, 22.0, 11.0, 90.0, 88.0, 45.0, 50.0];
    let arr_2d = [[64.0, 34.0, 25.0], [12.0, 22.0, 11.0], [90.0, 88.0, 45.0]];
    
    # Test quick_select via topk_extract - extract top k elements
    # This tests the quickselect algorithm used internally
    let topk_1 = std::array::topk_extract(arr_1d, 1);
    let topk_3 = std::array::topk_extract(arr_1d, 3);
    let topk_5 = std::array::topk_extract(arr_1d, 5);
    let topk_all = std::array::topk_extract(arr_1d, 10);
    
    # Test topk_extract (core algorithm, 1D only)
    # For multi-D arrays or axis-specific topk, use std::ml::topk
    let topk_std_1 = std::array::topk_extract(arr_1d, 1);
    let topk_std_3 = std::array::topk_extract(arr_1d, 3);
    let topk_std_5 = std::array::topk_extract(arr_1d, 5);
    
    # Test with 2D array - flatten first, then use topk_extract
    let arr_2d_flat = [arr_2d[i, j] | i in 0..len(arr_2d), j in 0..len(arr_2d[0])];
    let topk_2d_3 = std::array::topk_extract(arr_2d_flat, 3);
    
    # Test argmax and argmin from std::ml
    let x_argmax = [[1.0, 5.0, 3.0], [2.0, 1.0, 4.0], [3.0, 2.0, 1.0]];
    let x_argmin = [[1.0, 5.0, 3.0], [2.0, 1.0, 4.0], [3.0, 2.0, 1.0]];
    let result_argmax = std::ml::argmax(x_argmax);
    let result_argmin = std::ml::argmin(x_argmin);
    
    # Test nonzero
    let x_nonzero_1d = [0.0, 1.0, 0.0, 2.0, 0.0, 3.0];
    let x_nonzero_2d = [[0.0, 1.0, 0.0], [2.0, 0.0, 3.0], [0.0, 0.0, 0.0]];
    let result_nonzero_1d = std::ml::nonzero(x_nonzero_1d);
    let result_nonzero_2d = std::ml::nonzero(x_nonzero_2d);
    
    # Placeholder for topk - will be implemented later
    # ONNX TopK returns (values, indices) tuple
    # let x_topk = [[1.0, 5.0, 3.0, 2.0], [4.0, 1.0, 6.0, 3.0]];
    # let k_topk = 2;
    # let axis_topk = -1;  # Last axis
    # let largest_topk = 1;  # Largest values
    # let sorted_topk = 1;  # Sorted output
    # let (result_topk_values, result_topk_indices) = std::ml::topk(x_topk, k_topk, axis_topk, largest_topk, sorted_topk);
    """

    result = compile_and_execute(source, compiler, runtime)
    assert result.success, f"Execution failed: {result.errors}"

    # Verify quick_select via topk_extract
    arr_1d = np.array([64.0, 34.0, 25.0, 12.0, 22.0, 11.0, 90.0, 88.0, 45.0, 50.0], dtype=np.float32)
    
    # topk_extract should return top k largest elements (sorted descending)
    # Expected: top 1 = [90.0], top 3 = [90.0, 88.0, 64.0], top 5 = top 5 largest
    topk_1_actual = np.array(result.outputs['topk_1'])
    assert len(topk_1_actual) == 1, f"topk_1 should have length 1, got {len(topk_1_actual)}"
    assert topk_1_actual[0] == 90.0, f"topk_1 should contain 90.0 (max), got {topk_1_actual[0]}"
    
    topk_3_actual = np.array(result.outputs['topk_3'])
    assert len(topk_3_actual) == 3, f"topk_3 should have length 3, got {len(topk_3_actual)}"
    # Should contain the 3 largest: 90, 88, 64 (sorted descending)
    expected_top3 = np.array([90.0, 88.0, 64.0], dtype=np.float32)
    np.testing.assert_allclose(topk_3_actual, expected_top3, rtol=1e-6)
    
    topk_5_actual = np.array(result.outputs['topk_5'])
    assert len(topk_5_actual) == 5, f"topk_5 should have length 5, got {len(topk_5_actual)}"
    expected_top5 = np.array([90.0, 88.0, 64.0, 50.0, 45.0], dtype=np.float32)  # Sorted descending
    np.testing.assert_allclose(topk_5_actual, expected_top5, rtol=1e-6)
    
    topk_all_actual = np.array(result.outputs['topk_all'])
    assert len(topk_all_actual) == 10, f"topk_all should have length 10, got {len(topk_all_actual)}"
    expected_all = np.sort(arr_1d)[::-1]  # Sorted descending
    np.testing.assert_allclose(topk_all_actual, expected_all, rtol=1e-6)
    
    # Verify topk (std::array::topk)
    topk_std_1_actual = np.array(result.outputs['topk_std_1'])
    assert len(topk_std_1_actual) == 1, f"topk_std_1 should have length 1, got {len(topk_std_1_actual)}"
    assert topk_std_1_actual[0] == 90.0, f"topk_std_1 should contain 90.0, got {topk_std_1_actual[0]}"
    
    topk_std_3_actual = np.array(result.outputs['topk_std_3'])
    assert len(topk_std_3_actual) == 3, f"topk_std_3 should have length 3, got {len(topk_std_3_actual)}"
    np.testing.assert_allclose(topk_std_3_actual, expected_top3, rtol=1e-6)
    
    topk_std_5_actual = np.array(result.outputs['topk_std_5'])
    assert len(topk_std_5_actual) == 5, f"topk_std_5 should have length 5, got {len(topk_std_5_actual)}"
    np.testing.assert_allclose(topk_std_5_actual, expected_top5, rtol=1e-6)
    
    # Verify topk with 2D array (should flatten)
    arr_2d = np.array([[64.0, 34.0, 25.0], [12.0, 22.0, 11.0], [90.0, 88.0, 45.0]], dtype=np.float32)
    topk_2d_3_actual = np.array(result.outputs['topk_2d_3'])
    assert len(topk_2d_3_actual) == 3, f"topk_2d_3 should have length 3, got {len(topk_2d_3_actual)}"
    flat_2d = arr_2d.flatten()
    expected_2d_top3 = np.sort(flat_2d)[::-1][:3]  # Top 3 from flattened array, sorted descending
    np.testing.assert_allclose(topk_2d_3_actual, expected_2d_top3, rtol=1e-6)
    
    # Verify argmax
    x_argmax = np.array([[1.0, 5.0, 3.0], [2.0, 1.0, 4.0], [3.0, 2.0, 1.0]], dtype=np.float32)
    result_argmax_actual = np.array(result.outputs['result_argmax'])
    expected_argmax = np.argmax(x_argmax, axis=-1)
    np.testing.assert_array_equal(result_argmax_actual, expected_argmax)
    
    # Verify argmin
    x_argmin = np.array([[1.0, 5.0, 3.0], [2.0, 1.0, 4.0], [3.0, 2.0, 1.0]], dtype=np.float32)
    result_argmin_actual = np.array(result.outputs['result_argmin'])
    expected_argmin = np.argmin(x_argmin, axis=-1)
    np.testing.assert_array_equal(result_argmin_actual, expected_argmin)
    
    # Verify nonzero 1D
    x_nonzero_1d = np.array([0.0, 1.0, 0.0, 2.0, 0.0, 3.0], dtype=np.float32)
    result_nonzero_1d_actual = np.array(result.outputs['result_nonzero_1d'])
    expected_nonzero_1d = np.nonzero(x_nonzero_1d)[0]
    np.testing.assert_array_equal(result_nonzero_1d_actual, expected_nonzero_1d)
    
    # Verify nonzero 2D
    x_nonzero_2d = np.array([[0.0, 1.0, 0.0], [2.0, 0.0, 3.0], [0.0, 0.0, 0.0]], dtype=np.float32)
    result_nonzero_2d_actual = result.outputs['result_nonzero_2d']
    # Note: ONNX NonZero returns 2D array [rank, num_nonzero], but our implementation
    # returns array of tuples. For now, just check that we get the right number of non-zero elements
    nonzero_count = np.count_nonzero(x_nonzero_2d)
    # The result should be an array/list of tuples, so we check its length
    if isinstance(result_nonzero_2d_actual, (list, tuple)):
        assert len(result_nonzero_2d_actual) == nonzero_count, \
            f"nonzero_2d should have {nonzero_count} elements, got {len(result_nonzero_2d_actual)}"
    else:
        # If it's a numpy array, check its shape
        result_arr = np.array(result_nonzero_2d_actual)
        assert result_arr.size > 0, "nonzero_2d should return non-empty result"
    
    # TODO: When topk is implemented, uncomment and verify:
    # x_topk = np.array([[1.0, 5.0, 3.0, 2.0], [4.0, 1.0, 6.0, 3.0]], dtype=np.float32)
    # k_topk = 2
    # axis_topk = -1
    # largest_topk = 1
    # sorted_topk = 1
    # result_topk_values_actual = np.array(result.outputs['result_topk_values'])
    # result_topk_indices_actual = np.array(result.outputs['result_topk_indices'])
    # 
    # # Expected: top 2 values along last axis
    # # Row 0: [1.0, 5.0, 3.0, 2.0] -> top 2: [5.0, 3.0] (indices [1, 2])
    # # Row 1: [4.0, 1.0, 6.0, 3.0] -> top 2: [6.0, 4.0] (indices [2, 0])
    # expected_topk_values = np.array([[5.0, 3.0], [6.0, 4.0]], dtype=np.float32)
    # expected_topk_indices = np.array([[1, 2], [2, 0]], dtype=np.int32)
    # np.testing.assert_allclose(result_topk_values_actual, expected_topk_values, rtol=1e-6)
    # np.testing.assert_array_equal(result_topk_indices_actual, expected_topk_indices)


def test_quickselect_edge_cases(compiler, runtime):
    """Test quickselect edge cases"""
    source = """use std::array;
    
    # Edge case: k = 0
    let arr_empty = [1.0, 2.0, 3.0];
    let topk_0 = std::array::topk_extract(arr_empty, 0);
    
    # Edge case: k >= array length
    let arr_small = [1.0, 2.0];
    let topk_all_small = std::array::topk_extract(arr_small, 5);
    
    # Edge case: single element
    let arr_single = [42.0];
    let topk_single = std::array::topk_extract(arr_single, 1);
    
    # Edge case: all equal elements
    let arr_equal = [5.0, 5.0, 5.0, 5.0];
    let topk_equal = std::array::topk_extract(arr_equal, 2);
    
    # Edge case: already sorted
    let arr_sorted = [1.0, 2.0, 3.0, 4.0, 5.0];
    let topk_sorted = std::array::topk_extract(arr_sorted, 3);
    
    # Edge case: reverse sorted
    let arr_reverse = [5.0, 4.0, 3.0, 2.0, 1.0];
    let topk_reverse = std::array::topk_extract(arr_reverse, 3);
    """

    result = compile_and_execute(source, compiler, runtime)
    assert result.success, f"Execution failed: {result.errors}"

    # k = 0 should return empty array
    topk_0_actual = np.array(result.outputs['topk_0'])
    assert len(topk_0_actual) == 0, f"topk_0 should be empty, got {len(topk_0_actual)}"
    
    # k >= array length should return all elements
    arr_small = np.array([1.0, 2.0], dtype=np.float32)
    topk_all_small_actual = np.array(result.outputs['topk_all_small'])
    assert len(topk_all_small_actual) == len(arr_small), \
        f"topk_all_small should return all elements, got {len(topk_all_small_actual)}"
    expected_all_small = np.sort(arr_small)[::-1]  # Sorted descending
    np.testing.assert_allclose(topk_all_small_actual, expected_all_small, rtol=1e-6)
    
    # Single element
    topk_single_actual = np.array(result.outputs['topk_single'])
    assert len(topk_single_actual) == 1, f"topk_single should have 1 element, got {len(topk_single_actual)}"
    assert topk_single_actual[0] == 42.0, f"topk_single should be 42.0, got {topk_single_actual[0]}"
    
    # All equal elements
    topk_equal_actual = np.array(result.outputs['topk_equal'])
    assert len(topk_equal_actual) == 2, f"topk_equal should have 2 elements, got {len(topk_equal_actual)}"
    np.testing.assert_allclose(topk_equal_actual, [5.0, 5.0], rtol=1e-6)
    
    # Already sorted
    arr_sorted = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    topk_sorted_actual = np.array(result.outputs['topk_sorted'])
    assert len(topk_sorted_actual) == 3, f"topk_sorted should have 3 elements, got {len(topk_sorted_actual)}"
    expected_sorted = np.array([5.0, 4.0, 3.0], dtype=np.float32)  # Sorted descending
    np.testing.assert_allclose(topk_sorted_actual, expected_sorted, rtol=1e-6)
    
    # Reverse sorted
    arr_reverse = np.array([5.0, 4.0, 3.0, 2.0, 1.0], dtype=np.float32)
    topk_reverse_actual = np.array(result.outputs['topk_reverse'])
    assert len(topk_reverse_actual) == 3, f"topk_reverse should have 3 elements, got {len(topk_reverse_actual)}"
    expected_reverse = np.array([5.0, 4.0, 3.0], dtype=np.float32)  # Top 3 largest, sorted descending
    np.testing.assert_allclose(topk_reverse_actual, expected_reverse, rtol=1e-6)


# ===========================================================================
# TopK Preparation Tests
# These tests are prepared for when std::ml::topk is fully implemented
# ===========================================================================

def test_topk_1d(compiler, runtime):
    """
    Test 1D TopK helper function using partition (quickselect).
    Tests the core 1D TopK logic before testing full 2D TopK.
    
    Note: Since topk_1d_helper is not exported, we test it indirectly through topk
    by using a 2D array with shape [1, N] and axis=1, which effectively tests the 1D helper.
    """
    source = """use std::ml;
    
    # Test 1D TopK via 2D array with single row (shape [1, N])
    # This effectively tests the 1D helper since topk transposes to innermost and calls the helper
    # ONNX-aligned: topk(X, k, axis) - always returns largest values
    let arr1_2d = [[64.0, 34.0, 25.0, 12.0, 22.0, 11.0, 90.0, 88.0, 45.0, 50.0]];
    let (vals1, idxs1) = std::ml::topk(arr1_2d, 3, 1);
    
    # Test 1D TopK via 2D - same test (ONNX always returns largest)
    let (vals2, idxs2) = std::ml::topk(arr1_2d, 3, 1);
    
    # Test with axis=-1 (last axis, same as axis=1 for 2D)
    let (vals3, idxs3) = std::ml::topk(arr1_2d, 3, -1);
    
    # Test with axis=0 (first axis)
    let (vals4, idxs4) = std::ml::topk(arr1_2d, 1, 0);
    
    # Test edge case: k = 1
    let (vals5, idxs5) = std::ml::topk(arr1_2d, 1, 1);
    
    # Test edge case: k = len(arr) (but topk only supports 2D, so we'll test with k = N)
    let (vals6, idxs6) = std::ml::topk(arr1_2d, 10, 1);
    
    # Test with different array
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
    # Basic 2D case - top 2 along last axis (axis=1)
    # Input: [[1.0, 5.0, 3.0, 2.0], [4.0, 1.0, 6.0, 3.0]]
    # Row 0: top 2 of [1.0, 5.0, 3.0, 2.0] -> [5.0, 3.0] at indices [1, 2]
    # Row 1: top 2 of [4.0, 1.0, 6.0, 3.0] -> [6.0, 4.0] at indices [2, 0]
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
    # TopK along axis=0 (top k rows per column)
    # Input: [[1.0, 5.0], [4.0, 1.0], [3.0, 6.0]]
    # Column 0: top 2 of [1.0, 4.0, 3.0] -> [4.0, 3.0] at indices [1, 2]
    # Column 1: top 2 of [5.0, 1.0, 6.0] -> [6.0, 5.0] at indices [2, 0]
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


def test_nonzero_all_ranks(compiler, runtime):
    """Test nonzero operation across all supported ranks (1D, 2D, 3D)"""
    source = """use std::ml;
    # 1D
    let x_1d = [0.0, 1.0, 0.0, 2.0, 0.0, 3.0];
    let nonzero_1d = std::ml::nonzero(x_1d);
    
    # 2D
    let x_2d = [[0.0, 1.0, 0.0], [2.0, 0.0, 3.0], [0.0, 0.0, 0.0]];
    let nonzero_2d = std::ml::nonzero(x_2d);
    
    # 3D
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

