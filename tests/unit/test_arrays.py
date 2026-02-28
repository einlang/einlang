#!/usr/bin/env python3
"""
Tests for arrays: array access, comprehensions, ranges, membership.
Tests execute and check results to ensure complete array coverage using system.
"""

import pytest
from tests.test_utils import compile_and_execute


class TestArrays:
    """Complete array coverage with execution validation using system"""
    
    def _test_and_execute(self, source: str, compiler, runtime, expected_result=None) -> dict:
        """Execute source using system and return result for checking"""
        result = compile_and_execute(source, compiler, runtime)
        # Always check exec.result.success unless it is a negative test
        assert result.success, f"Execution failed: {result.errors}"
        
        if expected_result is not None and hasattr(result, 'outputs'):
            variables = result.outputs
            # Check if any variable matches expected result
            for var_name, var_value in variables.items():
                if var_value == expected_result:
                    return variables
        return result.outputs if hasattr(result, 'outputs') else {}
    
    def test_array_access(self, compiler, runtime):
        """Test array indexing"""
        cases = [
            "let arr = [10, 20, 30]; let x = arr[0]; assert(x == 10);",
            "let arr = [1, 2, 3]; let i = 1; let x = arr[i]; assert(x == 2);",
            "let arr = [5, 10, 15]; let i = 1; let x = arr[i + 1]; assert(x == 15);",
        ]
        for source in cases:
            self._test_and_execute(source, compiler, runtime)
    
    def test_array_comprehensions(self, compiler, runtime):
        """Test array comprehensions"""
        cases = [
            "let x = [i * i | i in 0..3]; assert(len(x) == 3);",
            "let x = [i * 2 | i in 1..4]; assert(len(x) == 3);",
            "let data = [1, 2, 3, 4, 5]; let x = [y * 2 | y in data, y > 2]; assert(len(x) == 3);",
        ]
        for source in cases:
            self._test_and_execute(source, compiler, runtime)
    
    def test_ranges(self, compiler, runtime):
        """Test range expressions"""
        cases = [
            "let x = [i | i in 0..5]; assert(len(x) == 5);",
            "let x = [i | i in 1..4]; assert(len(x) == 3);",
            "let squares = [i * i | i in 0..4]; assert(squares[2] == 4);",
        ]
        for source in cases:
            self._test_and_execute(source, compiler, runtime)
    
    def test_membership(self, compiler, runtime):
        """Test membership operator"""
        cases = [
            "let data = [1, 2, 3]; let x = [y | y in data]; assert(len(x) == 3);",
            "let x = [i | i in 0..10, i % 2 == 0]; assert(len(x) == 5);",
            "let nums = [1, 2, 3, 4, 5]; let evens = [x | x in nums, x % 2 == 0]; assert(len(evens) == 2);",
        ]
        for source in cases:
            self._test_and_execute(source, compiler, runtime)
    
    def test_nested_arrays(self, compiler, runtime):
        """Test nested array operations and chained rectangular access M[i][j] == M[i,j]"""
        cases = [
            "let matrix = [[1, 2], [3, 4]]; let x = matrix[0][1]; assert(x == 2);",
            "let nested = [[1, 2, 3], [4, 5, 6]]; let row_sums = [sum[j](row[j]) | row in nested]; assert(len(row_sums) == 2);",
            "let M = [[1, 2, 3], [4, 5, 6]]; assert(M[0][1] == M[0, 1]); assert(M[1][2] == M[1, 2]); assert(M[1][0] == 4);",
        ]
        for source in cases:
            self._test_and_execute(source, compiler, runtime)

    def test_jagged_arrays(self, compiler, runtime):
        """Test jagged (ragged) arrays: inconsistent row lengths, jagged access"""
        cases = [
            "let x = [[1, 2], [3, 4, 5]]; assert(x[0][1] == 2); assert(x[1][2] == 5); assert(len(x) == 2); assert(len(x[0]) == 2); assert(len(x[1]) == 3);",
            "let rows = [[10], [20, 21], [30, 31, 32]]; assert(rows[0][0] == 10); assert(rows[1][1] == 21); assert(rows[2][2] == 32); assert(len(rows) == 3);",
        ]
        for source in cases:
            self._test_and_execute(source, compiler, runtime)
    
    def test_array_with_functions(self, compiler, runtime):
        """Test arrays with function calls"""
        cases = [
            "fn increment(x) { x + 1 } let data = [1, 2, 3]; let x = [increment(y) | y in data]; assert(x[1] == 3);",
            "fn cube(x) { x * x * x } let cubes = [cube(i) | i in 1..4]; assert(cubes[2] == 27);",
        ]
        for source in cases:
            self._test_and_execute(source, compiler, runtime)
    
    def test_tensor_operations(self, compiler, runtime):
        """Test comprehensive tensor operations and rectangular declarations"""
        cases = [
            # Basic tensor operations
            "let input = [1, 2, 3]; let output[i] = input[i] * 2; assert(output[1] == 4);",
            "let data = [[1, 2], [3, 4]]; let bias = [10, 20]; let matrix[i,j] = data[i,j] + bias[j]; assert(matrix[0,1] == 22);",
            "let matrix = [[1, 2, 3], [4, 5, 6]]; let result[i] = sum[j](matrix[i,j]); assert(result[0] == 6);",
            
            # Matrix operations
            "let matrix = [[1, 2], [3, 4]]; let transposed[i,j] = matrix[j,i]; assert(transposed[0,1] == 3);",
            "let input = [1, 2, 3, 4]; let doubled[i] = input[i] * 2; assert(len(doubled) == 4);",
            "let matrix = [[1, 2], [3, 4]]; let transposed[i,j] = matrix[j,i]; assert(len(transposed) == 2);",
            
            # Tensor reductions
            "let data = [2, 4, 6]; let sum_val = sum[i](data[i]); assert(sum_val == 12);",
            "let a = [1, 2]; let b = [3, 4]; let dot_product = sum[i](a[i] * b[i]); assert(dot_product == 11);",
            
            # Tensor comparisons with literals
            "let a = [1, 2, 3]; let b[i] = a[i]; assert(b == a);",
            "let matrix1 = [[1, 2], [3, 4]]; let matrix2[i,j] = matrix1[i,j]; assert(matrix2 == matrix1);",
            "let doubled[i] = [1, 2, 3][i] * 2; assert(doubled == [2, 4, 6]);",
            "let incremented[i] = [10, 20, 30][i] + 1; assert(incremented == [11, 21, 31]);",
            
            # Complex tensor arithmetic
            "let a = [1, 2, 3]; let b = [4, 5, 6]; let sum[i] = a[i] + b[i]; assert(sum == [5, 7, 9]);",
            "let matrix_result[i,j] = [[1, 2], [3, 4]][i,j] * 10; assert(matrix_result == [[10, 20], [30, 40]]);",
        ]
        for source in cases:
            self._test_and_execute(source, compiler, runtime)
    
    def test_matrix_multiplication_patterns(self, compiler, runtime):
        """Test matrix multiplication using Einstein notation"""
        cases = [
            # Simple matrix-vector multiplication
            "let matrix = [[1, 2], [3, 4]]; let vector = [1, 1]; let result[i] = sum[j](matrix[i,j] * vector[j]); assert(result == [3, 7]);",
            
            # Matrix-matrix multiplication pattern
            "let a = [[1, 2], [3, 4]]; let b = [[5, 6], [7, 8]]; let c[i,j] = sum[k](a[i,k] * b[k,j]); assert(c[0,0] == 19);",
            
            # Biased matrix operation (neural network style)
            "let weights = [[1, 1], [1, 1]]; let input = [[2, 3], [4, 5]]; let bias = [1, 2]; let output[i,j] = sum[k](input[i,k] * weights[k,j]) + bias[j]; assert(output[0,0] == 6);",
        ]
        for source in cases:
            self._test_and_execute(source, compiler, runtime)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
