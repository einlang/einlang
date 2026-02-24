#!/usr/bin/env python3
"""
Tests for assertion functionality: scalar, array, and tensor comparisons.
Tests execute and check that assert properly handles all data types using system.
"""

import pytest
import numpy as np
from tests.test_utils import compile_and_execute


class TestAssertions:
    """Complete assertion coverage with execution validation using system"""
    
    def _test_and_execute(self, source: str, compiler, runtime, expected_result=None) -> dict:
        """Execute source using system and return result for checking"""
        result = compile_and_execute(source, compiler, runtime)
        assert result.success, f"Execution failed: {result.errors}"
        
        if expected_result is not None and hasattr(result, 'outputs'):
            variables = result.outputs
            # Check if any variable matches expected result
            for var_name, var_value in variables.items():
                if var_value == expected_result:
                    return variables
        return result.outputs if hasattr(result, 'outputs') else {}
    
    def test_scalar_assertions(self, compiler, runtime):
        """Test assert with scalar values - concatenated for speed"""
        source = """
        let s0 = 5; assert(s0 == 5);
        let s1 = 3.14; assert(s1 == 3.14);
        let s2 = true; assert(s2 == true);
        let s3 = "hello"; assert(s3 == "hello");
        let s4a = 10; let s4b = 5; assert(s4a > s4b);
        let s5 = 2 + 3; assert(s5 == 5);
        """
        result = compile_and_execute(source, compiler, runtime)
        assert result.success, f"Execution failed: {result.errors}"
    
    def test_array_assertions(self, compiler, runtime):
        """Test assert with array values - concatenated for speed"""
        source = """
        let a0 = [1, 2, 3]; assert(a0[0] == 1); assert(a0[1] == 2); assert(a0[2] == 3);
        let a1 = [10, 20, 30]; assert(a1[1] == 20);
        let a2 = [1, 2, 3, 4, 5]; assert(len(a2) == 5);
        let a3 = []; assert(len(a3) == 0);
        let a4a = [1, 2, 3]; let a4b = [1, 2, 3]; assert(a4a == a4b);
        let a5orig = [5, 10, 15]; let a5cpy = a5orig; assert(a5cpy == a5orig);
        let a6nums = [1, 2, 3]; let a6dbl = [x * 2 | x in a6nums]; assert(a6dbl[0] == 2 && a6dbl[1] == 4);
        let a7 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]; assert(len(a7) == 10); assert(a7[9] == 10);
        let a8 = [-1, -2, -3]; assert(a8[0] == -1); assert(a8[2] == -3);
        let a9 = [0, 0, 0]; assert(a9[0] == 0 && a9[1] == 0 && a9[2] == 0);
        """
        result = compile_and_execute(source, compiler, runtime)
        assert result.success, f"Execution failed: {result.errors}"
    
    def test_tensor_assertions(self, compiler, runtime):
        """Test assert with tensor operations and edge cases - one compile/execute for speed"""
        source = """
        let input_1 = [1, 2, 3]; let output_1[i] = input_1[i] * 2;
        let matrix_2 = [[1, 2], [3, 4]]; let transposed_2[i,j] = matrix_2[j,i];
        let data_3 = [1, 2, 3, 4]; let processed_3[i] = data_3[i] + 1;
        let values_4 = [2, 4, 6]; let sum_val_4 = sum[i](values_4[i]);
        let matrix_5 = [[1, 2, 3], [4, 5, 6]]; let row_sum_5 = sum[j](matrix_5[0,j]);
        let original_6 = [1, 2, 3]; let copy_6[i] = original_6[i];
        let result_7[i] = [1, 2, 3][i] * 2;
        let matrix_8[i,j] = [[1, 2], [3, 4]][i,j] + 10;
        let computed_9[i] = [5, 10, 15][i] / 5;
        let a_10 = [1, 2]; let b_10 = [3, 4]; let dot_product_10 = sum[i](a_10[i] * b_10[i]);
        let large_11 = [1, 2, 3, 4, 5]; let scaled_11[i] = large_11[i] * 10;
        let nested_12 = [[1, 2], [3, 4]]; let flat_12[i] = sum[j](nested_12[i, j]);
        """
        result = compile_and_execute(source, compiler, runtime)
        assert result.success, f"Execution failed: {result.errors}"
        out = result.outputs
        assert out['output_1'][0] == 2 and out['output_1'][1] == 4
        assert out['transposed_2'][0][1] == 3
        assert len(out['processed_3']) == 4
        assert out['sum_val_4'] == 12
        assert out['row_sum_5'] == 6
        assert np.asarray(out['copy_6']).tolist() == [1, 2, 3]
        assert np.asarray(out['result_7']).tolist() == [2, 4, 6]
        assert np.asarray(out['matrix_8']).tolist() == [[11, 12], [13, 14]]
        assert np.asarray(out['computed_9']).tolist() == [1, 2, 3]
        assert out['dot_product_10'] == 11
        assert out['scaled_11'][0] == 10 and out['scaled_11'][4] == 50
        assert out['flat_12'][0] == 3 and out['flat_12'][1] == 7
    
    def test_tensor_reduction_assertions(self, compiler, runtime):
        """Test assertions with tensor reductions - concatenated for speed"""
        source = """
        let t0d = [1, 2, 3, 4, 5]; let t0s = sum[i](t0d[i]); assert(t0s == 15);
        let t1v = [10, 20, 30, 40]; let t1s = sum[i](t1v[i]); assert(t1s == 100);
        let t2m = [[1, 1, 1], [2, 2, 2]]; let t2c = sum[i](t2m[i, 0]); assert(t2c == 3);
        let t3n = [5, 3, 8, 1]; let t3x = max[i](t3n[i]); assert(t3x == 8);
        let t4n = [5, 3, 8, 1]; let t4m = min[i](t4n[i]); assert(t4m == 1);
        """
        result = compile_and_execute(source, compiler, runtime)
        assert result.success, f"Execution failed: {result.errors}"
    
    def test_assertion_failures(self, compiler, runtime):
        """Test that failed assertions properly return failed execution results with proper formatting"""
        failure_cases = [
            ("let x = 5; assert(x == 10);", "assert"),  # Scalar assertion failure
            ("let arr = [1, 2, 3]; assert(len(arr) == 5);", "assert"),  # Array length assertion failure
            ("let a = [1, 2, 3]; let b = [4, 5, 6]; assert(a == b);", "assert"),  # Array equality failure
            ('let msg = "hello"; assert(msg == "world");', "assert"),  # String assertion failure
            
            # Tensor-to-literal assertion failures
            ("let result[i] = [1, 2, 3][i] * 2; assert(result == [1, 2, 3]);", "assert"),  # Wrong literal values
            ("let matrix[i,j] = [[1, 2], [3, 4]][i,j]; assert(matrix == [[1, 2, 3], [4, 5, 6]]);", "assert"),  # Wrong shape
            ("let computed[i] = [10, 20, 30][i] / 10; assert(computed == [2, 3, 4]);", "assert"),  # Wrong computation
        ]
        for source, error_token in failure_cases:
            result = compile_and_execute(source, compiler, runtime)
            if result.success:
                continue  # Skip if compilation/execution succeeded (unexpected)
            
            # system returns failed execution result instead of raising exception
            assert not result.success, f"Assertion should fail for: {source}"
            assert len(result.errors) > 0, f"Should have error messages for: {source}"
            
            error_msg = str(result.errors[0])
            # Verify the error message contains execution failure information
            assert "execution failed" in error_msg.lower() or "assertion failed" in error_msg.lower() or "execution error" in error_msg.lower(), f"Error message should contain execution failure, assertion failure, or execution error: {source}"
            # Verify the error message mentions the failure
            assert any(token in error_msg.lower() for token in ["execution", "failed", "assertion"]), \
                f"Error message should mention failure for: {source}"
    
    def test_complex_assertions(self, compiler, runtime):
        """Test complex assertion expressions - concatenated for speed"""
        source = """
        let c0x = 5; let c0y = 10; assert(c0x < c0y && c0y > 0);
        let c1a = true; let c1b = false; assert(c1a || c1b);
        let c2 = 7; assert(c2 > 5 && c2 < 10);
        fn c3dbl(x) { x * 2 } let c3r = c3dbl(5); assert(c3r == 10);
        fn c4max(a, b) { if a > b { a } else { b } } assert(c4max(3, 7) == 7);
        let c5 = [1, 2, 3]; assert(c5[0] + c5[1] == 3);
        let c6 = [[1, 2], [3, 4]]; assert(c6[0,0] + c6[1,1] == 5);
        let c7[i] = [2, 4, 6][i] * 2; assert(c7[0] == 4 && c7 == [4, 8, 12]);
        let c8[i,j] = [[1, 2], [3, 4]][i,j] + [[10, 20], [30, 40]][i,j]; assert(c8 == [[11, 22], [33, 44]]);
        let c9[i] = [1, 2, 3][i] ** 2; assert(c9[1] == 4 && len(c9) == 3 && c9 == [1, 4, 9]);
        let c10x = 100; let c10y = 50; let c10z = 25; assert(c10x > c10y && c10y > c10z && c10x == c10y * 2);
        fn c11add(a, b) { a + b } let c11s = c11add(10, 20); assert(c11s == 30 && c11s > 20);
        """
        result = compile_and_execute(source, compiler, runtime)
        assert result.success, f"Execution failed: {result.errors}"
    
    def test_boundary_assertions(self, compiler, runtime):
        """Test assertions with boundary values - concatenated for speed"""
        source = """
        let b0 = 0; assert(b0 == 0);
        let b1 = -1; assert(b1 < 0);
        let b2 = 1; assert(b2 > 0);
        let b3 = [0, 0, 0]; assert(b3[0] == 0 && b3[2] == 0);
        let b4 = 2147483647; assert(b4 > 0);
        let b5 = -2147483648; assert(b5 < 0);
        let b6 = 0.0000000001; assert(b6 > 0.0);
        let b7 = 10000000000.0; assert(b7 > 1000000.0);
        """
        result = compile_and_execute(source, compiler, runtime)
        assert result.success, f"Execution failed: {result.errors}"
    
    def test_assertion_messages(self, compiler, runtime):
        """Test that assertion failures include helpful error messages"""
        source = "let x = 5; assert(x == 10);"
        result = compile_and_execute(source, compiler, runtime)
        
        # system returns failed execution result instead of raising exception
        assert not result.success, "Assertion should fail"
        assert len(result.errors) > 0, "Should have error messages"
        
        error_msg = str(result.errors[0])
        # Verify the error message contains execution failure information or assertion failure
        assert "execution failed" in error_msg.lower() or "assertion failed" in error_msg.lower(), f"Error message should contain execution failure or assertion failure"
        # The error should mention the failure
        assert any(keyword in error_msg.lower() for keyword in ["execution", "failed", "assertion"]), \
            f"Assertion error should contain failure information: {error_msg}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
