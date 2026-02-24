#!/usr/bin/env python3
"""
Tests for std::array module functionality with string source examples.
Tests execute and check results to ensure complete array stdlib coverage.
"""

import pytest
import numpy as np
from ..test_utils import compile_and_execute
from einlang.shared.errors import EinlangSourceError


class TestArrayModule:
    """Complete std::array coverage with execution validation"""
    
    def _test_and_execute(self, source: str, compiler, runtime, expected_result=None) -> dict:
        """Execute source and return result for checking"""
        execution_result = compile_and_execute(source, compiler, runtime)
        assert execution_result.success, f"Execution failed: {execution_result.errors}"
        
        result = execution_result.value
        if expected_result is not None:
            assert result == expected_result
        return result
    
    def test_array_functions(self, compiler, runtime):
        """Test array module functions - only testing fully implemented functions"""
        # Combine all test cases into one source to ensure all functions are compiled together
        source = """
# Concatenate
let arr1 = [1, 2]; let other1 = [3, 4]; let x1 = std::array::concatenate(arr1, other1); assert(x1 == [1, 2, 3, 4]);

# Sum
let arr2 = [1, 2, 3, 4, 5]; let x2 = std::array::sum(arr2); assert(x2 == 15);
assert(std::array::sum([42]) == 42);
let arr3 = [-2, 5, -1, 3]; assert(std::array::sum(arr3) == 5);

# Flatten
let matrix1 = [[1, 2], [3, 4]]; let x3 = std::array::flatten(matrix1); assert(x3 == [1, 2, 3, 4]);

# Test len as replacement for shape (shape removed due to complexity)
let arr4 = [1, 2, 3, 4, 5]; assert(len(arr4) == 5);
let matrix2 = [[1, 2], [3, 4]]; assert(len(matrix2) == 2);
        """
        self._test_and_execute(source, compiler, runtime)
    
    def test_reductions(self, compiler, runtime):
        """Test reduction operations on arrays"""
        # Combine all test cases into one source
        source = """
let data1 = [1, 2, 3, 4, 5]; let x1 = sum[i](data1[i]); assert(x1 == 15);
let data2 = [1, 2, 3, 4, 5]; let x2 = max[i](data2[i]); assert(x2 == 5);
let data3 = [1, 2, 3, 4, 5]; let x3 = min[i](data3[i]); assert(x3 == 1);
        """
        self._test_and_execute(source, compiler, runtime)
    
    def test_argmax_argmin(self, compiler, runtime):
        """Test argmax and argmin functions"""
        source = """
use std::array;

# argmax - basic cases
let scores1 = [88, 92, 76, 92, 83];
let idx1 = std::array::argmax(scores1);
assert(idx1 == 1);  # First occurrence of max value 92

let scores2 = [5, 3, 8, 1, 8];
let idx2 = std::array::argmax(scores2);
assert(idx2 == 2);  # First occurrence of max value 8

let single = [42];
assert(std::array::argmax(single) == 0);

# argmin - basic cases
let temps = [25, 18, 30, 18, 22];
let min_idx = std::array::argmin(temps);
assert(min_idx == 1);  # First occurrence of min value 18

let nums = [10, 5, 3, 5, 7];
assert(std::array::argmin(nums) == 2);  # First occurrence of min value 3

# Edge cases - all same values
let all_same = [5, 5, 5, 5];
assert(std::array::argmax(all_same) == 0);  # First index
assert(std::array::argmin(all_same) == 0);  # First index
        """
        self._test_and_execute(source, compiler, runtime)
    
    def test_argmax_all_argmin_all(self, compiler, runtime):
        """Test argmax_all and argmin_all functions"""
        source = """
use std::array;

# argmax_all - returns all indices with max value
let scores = [88, 92, 76, 92, 83];
let all_max = std::array::argmax_all(scores);
assert(len(all_max) == 2);
assert(all_max[0] == 1);
assert(all_max[1] == 3);  # Both indices where value is 92

let nums = [5, 3, 8, 1, 8, 8];
let all_max2 = std::array::argmax_all(nums);
assert(len(all_max2) == 3);
assert(all_max2[0] == 2);
assert(all_max2[1] == 4);
assert(all_max2[2] == 5);  # All indices where value is 8

# argmin_all - returns all indices with min value
let temps = [25, 18, 30, 18, 22];
let all_min = std::array::argmin_all(temps);
assert(len(all_min) == 2);
assert(all_min[0] == 1);
assert(all_min[1] == 3);  # Both indices where value is 18

# Edge case - all same values
let all_same = [5, 5, 5];
let all_max_same = std::array::argmax_all(all_same);
assert(len(all_max_same) == 3);
assert(all_max_same[0] == 0);
assert(all_max_same[1] == 1);
assert(all_max_same[2] == 2);
        """
        self._test_and_execute(source, compiler, runtime)
    
    def test_topk(self, compiler, runtime):
        """Test topk_extract function (core algorithm)"""
        source = """
use std::array;

let scores = [88, 92, 76, 94, 83];
let top3 = std::array::topk_extract(scores, 3);
let all = std::array::topk_extract(scores, 5);
let empty = std::array::topk_extract(scores, 0);
let dups = [5, 8, 5, 8, 3];
let top2 = std::array::topk_extract(dups, 2);
        """
        result = compile_and_execute(source, compiler, runtime)
        assert result.success, f"Execution failed: {result.errors}"
        top3 = np.asarray(result.outputs["top3"])
        all_arr = np.asarray(result.outputs["all"])
        empty_arr = np.asarray(result.outputs["empty"])
        top2_arr = np.asarray(result.outputs["top2"])
        assert len(top3) == 3 and top3[0] == 94 and top3[1] == 92 and top3[2] == 88
        assert len(all_arr) == 5 and all_arr[0] == 94 and all_arr[4] == 76
        assert len(empty_arr) == 0
        assert len(top2_arr) == 2 and top2_arr[0] == 8 and top2_arr[1] == 8
    
    def test_topk_with_indices(self, compiler, runtime):
        """Test topk_with_indices_extract function (core algorithm)"""
        source = """
use std::array;

let scores = [88, 92, 76, 94, 83];
let result = std::array::topk_with_indices_extract(scores, [0, 1, 2, 3, 4], 3);
let top_values = result.0;
let top_indices = result.1;
let all_result = std::array::topk_with_indices_extract(scores, [0, 1, 2, 3, 4], 5);
let all_vals = all_result.0;
let all_idxs = all_result.1;
let empty_result = std::array::topk_with_indices_extract(scores, [0, 1, 2, 3, 4], 0);
let empty_vals = empty_result.0;
let empty_idxs = empty_result.1;
        """
        result = compile_and_execute(source, compiler, runtime)
        assert result.success, f"Execution failed: {result.errors}"
        top_values = np.asarray(result.outputs["top_values"])
        top_indices = np.asarray(result.outputs["top_indices"])
        scores_arr = np.asarray(result.outputs["scores"])
        assert len(top_values) == 3 and len(top_indices) == 3
        for i in range(3):
            assert scores_arr[int(top_indices[i])] == top_values[i]
        all_vals_arr = np.asarray(result.outputs["all_vals"])
        all_idxs_arr = np.asarray(result.outputs["all_idxs"])
        assert len(all_vals_arr) == 5 and len(all_idxs_arr) == 5
        empty_vals_arr = np.asarray(result.outputs["empty_vals"])
        empty_idxs_arr = np.asarray(result.outputs["empty_idxs"])
        assert len(empty_vals_arr) == 0 and len(empty_idxs_arr) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
