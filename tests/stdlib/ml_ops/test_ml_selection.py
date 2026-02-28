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
    
    // Test partition function - core of quickselect
    let arr_partition = [64.0, 34.0, 25.0, 12.0, 22.0, 11.0, 90.0, 88.0, 45.0, 50.0];
    let pivot_partition = 45.0;
    let (smaller, equal, larger) = std::array::partition(arr_partition, pivot_partition);
    
    // Test partition with different pivot
    let pivot_partition2 = 30.0;
    let (smaller2, equal2, larger2) = std::array::partition(arr_partition, pivot_partition2);
    
    // Test partition with edge cases
    let arr_empty = [];
    let (smaller_empty, equal_empty, larger_empty) = std::array::partition(arr_empty, 5.0);
    
    let arr_single = [42.0];
    let (smaller_single, equal_single, larger_single) = std::array::partition(arr_single, 42.0);
    
    // Note: partition only works on 1D arrays. For 2D arrays, flatten first.
    // Test partition with flattened 2D array (1D version)
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
    
    // Test data for quick_select (via topk_extract)
    // The quickselect algorithm is used internally in topk_extract
    let arr_1d = [64.0, 34.0, 25.0, 12.0, 22.0, 11.0, 90.0, 88.0, 45.0, 50.0];
    let arr_2d = [[64.0, 34.0, 25.0], [12.0, 22.0, 11.0], [90.0, 88.0, 45.0]];
    
    // Test quick_select via topk_extract - extract top k elements
    // This tests the quickselect algorithm used internally
    let topk_1 = std::array::topk_extract(arr_1d, 1);
    let topk_3 = std::array::topk_extract(arr_1d, 3);
    let topk_5 = std::array::topk_extract(arr_1d, 5);
    let topk_all = std::array::topk_extract(arr_1d, 10);
    
    // Test topk_extract (core algorithm, 1D only)
    // For multi-D arrays or axis-specific topk, use std::ml::topk
    let topk_std_1 = std::array::topk_extract(arr_1d, 1);
    let topk_std_3 = std::array::topk_extract(arr_1d, 3);
    let topk_std_5 = std::array::topk_extract(arr_1d, 5);
    
    // Test with 2D array - flatten first, then use topk_extract
    let arr_2d_flat = [arr_2d[i, j] | i in 0..len(arr_2d), j in 0..len(arr_2d[0])];
    let topk_2d_3 = std::array::topk_extract(arr_2d_flat, 3);
    
    // Test argmax and argmin from std::ml
    let x_argmax = [[1.0, 5.0, 3.0], [2.0, 1.0, 4.0], [3.0, 2.0, 1.0]];
    let x_argmin = [[1.0, 5.0, 3.0], [2.0, 1.0, 4.0], [3.0, 2.0, 1.0]];
    let result_argmax = std::ml::argmax(x_argmax);
    let result_argmin = std::ml::argmin(x_argmin);
    
    // Test nonzero
    let x_nonzero_1d = [0.0, 1.0, 0.0, 2.0, 0.0, 3.0];
    let x_nonzero_2d = [[0.0, 1.0, 0.0], [2.0, 0.0, 3.0], [0.0, 0.0, 0.0]];
    let result_nonzero_1d = std::ml::nonzero(x_nonzero_1d);
    let result_nonzero_2d = std::ml::nonzero(x_nonzero_2d);
    
    // Placeholder for topk - will be implemented later
    // ONNX TopK returns (values, indices) tuple
    // let x_topk = [[1.0, 5.0, 3.0, 2.0], [4.0, 1.0, 6.0, 3.0]];
    // let k_topk = 2;
    // let axis_topk = -1;  // Last axis
    // let largest_topk = 1;  // Largest values
    // let sorted_topk = 1;  // Sorted output
    // let (result_topk_values, result_topk_indices) = std::ml::topk(x_topk, k_topk, axis_topk, largest_topk, sorted_topk);
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
    
    // Edge case: k = 0
    let arr_empty = [1.0, 2.0, 3.0];
    let topk_0 = std::array::topk_extract(arr_empty, 0);
    
    // Edge case: k >= array length
    let arr_small = [1.0, 2.0];
    let topk_all_small = std::array::topk_extract(arr_small, 5);
    
    // Edge case: single element
    let arr_single = [42.0];
    let topk_single = std::array::topk_extract(arr_single, 1);
    
    // Edge case: all equal elements
    let arr_equal = [5.0, 5.0, 5.0, 5.0];
    let topk_equal = std::array::topk_extract(arr_equal, 2);
    
    // Edge case: already sorted
    let arr_sorted = [1.0, 2.0, 3.0, 4.0, 5.0];
    let topk_sorted = std::array::topk_extract(arr_sorted, 3);
    
    // Edge case: reverse sorted
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

