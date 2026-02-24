"""
Minimal failing test cases extracted from terminal failures.

These test cases capture specific failures observed in the test suite:
- Fibonacci assertion failures
- Monomorphization error handling
- Mutual recursion compilation errors
- Index out of bounds errors
- Variable scope issues
"""

import pytest
import numpy as np
from tests.test_utils import compile_and_execute


class TestFibonacciAssertionFailures:
    """Test cases for Fibonacci sequence assertion failures"""
    
    def test_fibonacci_basic_assertions(self, compiler, runtime):
        """Test basic Fibonacci with assertions that are currently failing"""
        source = """
        let fib[0] = 1;
        let fib[1] = 1;
        let fib[n in 2..11] = fib[n-1] + fib[n-2];
        
        assert(fib[0] == 1, "fib[0] should be 1");
        assert(fib[1] == 1, "fib[1] should be 1");
        assert(fib[2] == 2, "fib[2] should be 2");
        """
        
        result = compile_and_execute(source, compiler, runtime)
        
        assert result is not None, "Execution returned None"
        assert result.success, f"Fibonacci test failed: {result.errors}"
        assert 'fib' in result.outputs, "fib variable not found"
        assert result.outputs['fib'][0] == 1, f"fib[0] should be 1, got {result.outputs['fib'][0]}"
        assert result.outputs['fib'][1] == 1, f"fib[1] should be 1, got {result.outputs['fib'][1]}"
        assert result.outputs['fib'][2] == 2, f"fib[2] should be 2, got {result.outputs['fib'][2]}"
    
    def test_fibonacci_array_comparison(self, compiler, runtime):
        """Test Fibonacci sequence with element-wise verification"""
        source = """
        let fib[0] = 1;
        let fib[1] = 1;
        let fib[n in 2..9] = fib[n-1] + fib[n-2];
        
        assert(fib[0] == 1, "fib[0] should be 1");
        assert(fib[1] == 1, "fib[1] should be 1");
        assert(fib[2] == 2, "fib[2] should be 2");
        assert(fib[3] == 3, "fib[3] should be 3");
        assert(fib[4] == 5, "fib[4] should be 5");
        assert(fib[5] == 8, "fib[5] should be 8");
        assert(fib[6] == 13, "fib[6] should be 13");
        assert(fib[7] == 21, "fib[7] should be 21");
        assert(fib[8] == 34, "fib[8] should be 34");
        fib;
        """
        
        result = compile_and_execute(source, compiler, runtime)
        
        assert result is not None, "Execution returned None"
        assert result.success, f"Fibonacci array comparison failed: {result.errors}"
        assert 'fib' in result.outputs, "fib variable not found"
        expected = np.array([1, 1, 2, 3, 5, 8, 13, 21, 34])
        np.testing.assert_array_equal(np.asarray(result.outputs['fib']), expected)
    
    def test_lucas_sequence_assertions(self, compiler, runtime):
        """Test Lucas sequence with assertions that are currently failing"""
        source = """
        let lucas[0] = 2;
        let lucas[1] = 1;
        let lucas[n in 2..9] = lucas[n-1] + lucas[n-2];
        
        let expected = [2, 1, 3, 4, 7, 11, 18, 29, 47];
        assert(lucas == expected, "Lucas sequence should match expected");
        """
        
        result = compile_and_execute(source, compiler, runtime)
        
        assert result is not None, "Execution returned None"
        assert result.success, f"Lucas sequence test failed: {result.errors}"
        assert 'lucas' in result.outputs, "lucas variable not found"
        expected_lucas = [2, 1, 3, 4, 7, 11, 18, 29, 47]
        np.testing.assert_array_equal(result.outputs['lucas'], expected_lucas)


class TestMonomorphizationErrorHandling:
    """Test cases for monomorphization error handling failures"""
    
    def test_type_error_should_be_caught(self, compiler, runtime):
        """Test that type errors are caught during compilation"""
        source = """
        fn add_ints(a: i32, b: i32) -> i32 { a + b }
        let result = add_ints(1, 3.14);
        """
        result = compile_and_execute(source, compiler, runtime)
        assert not result.success, "Should fail due to type mismatch"
        assert len(result.errors) > 0, "Should have error messages"
    
    def test_arity_error_should_be_caught(self, compiler, runtime):
        """Test that arity errors are caught during compilation"""
        source = """
        fn add(a: i32, b: i32) -> i32 { a + b }
        let result = add(1);
        """
        result = compile_and_execute(source, compiler, runtime)
        assert not result.success, "Should fail due to arity mismatch"
        assert len(result.errors) > 0, "Should have error messages"


class TestMutualRecursion:
    """Test cases for mutual recursion failures"""
    
    def test_mutual_recursion_compilation(self, compiler, runtime):
        """Test that mutual recursion compiles and executes correctly"""
        source = """
        fn is_even(n) {
            if n == 0 {
                true
            } else {
                is_odd(n - 1)
            }
        }
        
        fn is_odd(n) {
            if n == 0 {
                false
            } else {
                is_even(n - 1)
            }
        }
        
        let even_4 = is_even(4);
        let odd_4 = is_odd(4);
        let even_5 = is_even(5);
        let odd_5 = is_odd(5);
        """
        
        result = compile_and_execute(source, compiler, runtime)
        
        assert result is not None, "Execution returned None"
        assert result.success, f"Mutual recursion test failed: {result.errors}"
        assert result.outputs['even_4'] == True
        assert result.outputs['odd_4'] == False
        assert result.outputs['even_5'] == False
        assert result.outputs['odd_5'] == True


class TestIndexOutOfBounds:
    """Test cases for index out of bounds errors"""
    
    def test_index_access_beyond_range(self, compiler, runtime):
        """Test accessing array indices beyond declared range"""
        source = """
        let arr[0] = 1;
        let arr[1] = 2;
        let arr[2] = 3;
        
        # Access index 4 which is beyond the range
        let value = arr[4];
        """
        
        result = compile_and_execute(source, compiler, runtime)
        
        # This should either succeed (if system handles out-of-bounds) or fail gracefully
        if not result.success:
            assert len(result.errors) > 0, "Should have error messages for out-of-bounds access"
    
    def test_index_in_recurrence_beyond_range(self, compiler, runtime):
        """Test recurrence relation accessing indices beyond range"""
        source = """
        let fib[0] = 1;
        let fib[1] = 1;
        let fib[n in 2..5] = fib[n-1] + fib[n-2];
        
        # Try to access index beyond range
        let value = fib[10];
        """
        
        result = compile_and_execute(source, compiler, runtime)
        
        # This should either succeed (if system handles out-of-bounds) or fail gracefully
        if not result.success:
            assert len(result.errors) > 0, "Should have error messages for out-of-bounds access"


class TestVariableScope:
    """Test cases for variable scope issues"""
    
    def test_variable_scope_in_functions(self, compiler, runtime):
        """Test variable scope within functions"""
        source = """
        fn test_scope() {
            let x = 10;
            x
        }
        
        let result = test_scope();
        """
        
        result = compile_and_execute(source, compiler, runtime)
        
        assert result is not None, "Execution returned None"
        assert result.success, f"Variable scope test failed: {result.errors}"
        assert result.outputs['result'] == 10


class TestFunctionOverloading:
    """Test cases for function overloading issues"""
    
    def test_function_overloading_basic(self, compiler, runtime):
        """Duplicate fn name in same scope is rejected (Rust-aligned: no redefinition)."""
        source = """
        fn add(a: i32, b: i32) -> i32 { a + b }
        fn add(a: f64, b: f64) -> f64 { a + b }
        
        let int_result = add(1, 2);
        let float_result = add(1.0, 2.0);
        """
        
        result = compile_and_execute(source, compiler, runtime)
        
        assert result is not None, "Execution returned None"
        assert not result.success, "Expected redefinition error (no overloading in same scope)"
        assert result.errors, "Expected at least one error"
        error_str = " ".join(result.errors)
        assert "redefinition" in error_str and "add" in error_str, f"Expected redefinition error, got: {result.errors}"


class TestReductionOperations:
    """Test cases for reduction operation failures"""
    
    def test_reduction_with_variables(self, compiler, runtime):
        """Test reduction operations with variables"""
        source = """
        let arr = [1, 2, 3, 4, 5];
        let sum = sum(arr);
        """
        
        result = compile_and_execute(source, compiler, runtime)
        
        assert result is not None, "Execution returned None"
        assert result.success, f"Reduction operation test failed: {result.errors}"
        assert result.outputs['sum'] == 15


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

