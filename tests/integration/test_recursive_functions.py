"""
Integration tests for recursive functions.

Tests that recursive functions work correctly with monomorphization.
"""

import pytest
from tests.test_utils import compile_and_execute


class TestRecursiveFunctions:
    """Test recursive function support"""
    
    @pytest.fixture(scope="class")
    def compiler(self, class_compiler):
        """Reuse class-scoped compiler"""
        return class_compiler
    
    @pytest.fixture(scope="class")
    def runtime(self, class_runtime):
        """Reuse class-scoped runtime"""
        return class_runtime
    
    def test_simple_countdown(self, compiler, runtime):
        """Test simple recursive countdown"""
        source = """
        fn countdown(n) {
            if n <= 0 {
                0
            } else {
                countdown(n - 1)
            }
        }
        
        let result = countdown(5);
        """
        result = compile_and_execute(source, compiler, runtime)
        
        assert result.success, f"Execution failed: {result.get_errors()}"
        assert result.outputs['result'] == 0
    
    def test_factorial(self, compiler, runtime):
        """Test recursive factorial"""
        source = """
        fn factorial(n) {
            if n <= 1 {
                1
            } else {
                n * factorial(n - 1)
            }
        }
        
        let result = factorial(5);
        """
        result = compile_and_execute(source, compiler, runtime)
        
        assert result.success, f"Execution failed: {result.get_errors()}"
        assert result.outputs['result'] == 120
    
    def test_fibonacci(self, compiler, runtime):
        """Test recursive Fibonacci"""
        source = """
        fn fib(n) {
            if n <= 1 {
                n
            } else {
                fib(n - 1) + fib(n - 2)
            }
        }
        
        let result = fib(8);
        """
        result = compile_and_execute(source, compiler, runtime)
        
        assert result.success, f"Execution failed: {result.get_errors()}"
        assert result.outputs['result'] == 21
    
    def test_quickselect_kth_smallest(self, compiler, runtime):
        """Test recursive quickselect algorithm"""
        source = """
        fn quickselect_kth_smallest(arr, k) {
            if len(arr) == 1 {
                arr[0]
            } else {
                let pivot = arr[0];
                let rest = [arr[i] | i in 1..len(arr)];
                let smaller = [rest[i] | i in 0..len(rest), rest[i] < pivot];
                let larger = [rest[i] | i in 0..len(rest), rest[i] >= pivot];
                let m = len(smaller);
                
                if k < m {
                    quickselect_kth_smallest(smaller, k)
                } else {
                    if k == m {
                        pivot
                    } else {
                        let new_k = k - m - 1;
                        quickselect_kth_smallest(larger, new_k)
                    }
                }
            }
        }
        
        let test_data = [64, 34, 25, 12, 22, 11, 90, 88, 45, 50];
        let smallest_0 = quickselect_kth_smallest(test_data, 0);
        let smallest_4 = quickselect_kth_smallest(test_data, 4);
        """
        result = compile_and_execute(source, compiler, runtime)
        
        assert result.success, f"Execution failed: {result.get_errors()}"
        # Sorted: [11, 12, 22, 25, 34, 45, 50, 64, 88, 90]
        assert result.outputs['smallest_0'] == 11  # minimum
        assert result.outputs['smallest_4'] == 34  # 5th element (0-indexed)
    
    def test_mutual_recursion(self, compiler, runtime):
        """Test mutually recursive functions"""
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
        
        assert result.success, f"Execution failed: {result.get_errors()}"
        assert result.outputs['even_4'] == True
        assert result.outputs['odd_4'] == False
        assert result.outputs['even_5'] == False
        assert result.outputs['odd_5'] == True
    
    def test_sum_recursive(self, compiler, runtime):
        """Test recursive array sum"""
        source = """
        fn sum_recursive(arr) {
            if len(arr) == 0 {
                0
            } else {
                if len(arr) == 1 {
                    arr[0]
                } else {
                    let first = arr[0];
                    let rest = [arr[i] | i in 1..len(arr)];
                    first + sum_recursive(rest)
                }
            }
        }
        
        let numbers = [1, 2, 3, 4, 5];
        let total = sum_recursive(numbers);
        """
        result = compile_and_execute(source, compiler, runtime)
        
        assert result.success, f"Execution failed: {result.get_errors()}"
        assert result.outputs['total'] == 15

