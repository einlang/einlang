"""
End-to-End Integration Tests for Monomorphization

Tests the complete monomorphization pipeline with real compilation and execution:
- Simple programs
- Module integration
- Complex multi-function programs
- Performance validation
"""

import pytest
from tests.test_utils import compile_and_execute


class TestSimplePrograms:
    """Test end-to-end compilation and execution with monomorphization"""
    
    def test_single_function_single_type(self, compiler, runtime):
        """Single generic function with one type"""
        source = """
        fn double(x) { x * 2 }
        let result = double(21);
        """
        result = compile_and_execute(source, compiler, runtime)
        
        assert result.success, f"Execution failed: {result.get_errors()}"
        assert result.outputs['result'] == 42
    
    def test_single_function_multiple_types(self, compiler, runtime):
        """Single generic function with multiple types"""
        source = """
        fn identity(x) { x }
        let a = identity(42);
        let b = identity(3.14);
        let c = identity(true);
        """
        result = compile_and_execute(source, compiler, runtime)
        
        assert result.success, f"Execution failed: {result.get_errors()}"
        assert result.outputs['a'] == 42
        assert abs(result.outputs['b'] - 3.14) < 0.01
        assert result.outputs['c'] == True
    
    def test_multiple_functions_chained(self, compiler, runtime):
        """Multiple functions called in sequence"""
        source = """
        fn double(x) { x * 2 }
        fn add_ten(x) { x + 10 }
        
        let step1 = double(5);
        let step2 = add_ten(step1);
        """
        result = compile_and_execute(source, compiler, runtime)
        
        assert result.success, f"Execution failed: {result.get_errors()}"
        assert result.outputs['step1'] == 10
        assert result.outputs['step2'] == 20
    
    def test_nested_function_calls(self, compiler, runtime):
        """Nested function calls"""
        source = """
        fn double(x) { x * 2 }
        fn triple(x) { x * 3 }
        fn add(x, y) { x + y }
        
        let result = add(double(5), triple(3));
        """
        result = compile_and_execute(source, compiler, runtime)
        
        assert result.success, f"Execution failed: {result.get_errors()}"
        assert result.outputs['result'] == 19  # (5*2) + (3*3) = 10 + 9
    
    def test_conditional_with_generic_functions(self, compiler, runtime):
        """Generic functions used in conditionals"""
        source = """
        fn abs(x) {
            if x < 0 {
                -x
            } else {
                x
            }
        }
        
        let a = abs(-5);
        let b = abs(3);
        """
        result = compile_and_execute(source, compiler, runtime)
        
        assert result.success, f"Execution failed: {result.get_errors()}"
        assert result.outputs['a'] == 5
        assert result.outputs['b'] == 3


class TestArrayOperations:
    """Test monomorphization with array operations"""
    
    @pytest.fixture(scope="class")
    def compiler(self, class_compiler):
        """Reuse class-scoped compiler"""
        return class_compiler
    
    @pytest.fixture(scope="class")
    def runtime(self, class_runtime):
        """Reuse class-scoped runtime"""
        return class_runtime
    
    def test_array_first_element(self, compiler, runtime):
        """Generic function accessing array elements"""
        source = """
        fn first(arr) { arr[0] }
        let result = first([10, 20, 30]);
        """
        result = compile_and_execute(source, compiler, runtime)
        
        assert result.success, f"Execution failed: {result.get_errors()}"
        assert result.outputs['result'] == 10
    
    def test_array_sum(self, compiler, runtime):
        """Generic function accessing array elements"""
        source = """
        fn sum_three(a, b, c) { a + b + c }
        let arr = [1, 2, 3];
        let result = sum_three(arr[0], arr[1], arr[2]);
        """
        result = compile_and_execute(source, compiler, runtime)
        
        assert result.success, f"Execution failed: {result.get_errors()}"
        assert result.outputs['result'] == 6
    
    def test_array_map_pattern(self, compiler, runtime):
        """Multiple calls to generic function with array elements"""
        source = """
        fn double(x) { x * 2 }
        let arr = [1, 2, 3];
        let a = double(arr[0]);
        let b = double(arr[1]);
        let c = double(arr[2]);
        """
        result = compile_and_execute(source, compiler, runtime)
        
        assert result.success, f"Execution failed: {result.get_errors()}"
        assert result.outputs['a'] == 2
        assert result.outputs['b'] == 4
        assert result.outputs['c'] == 6


class TestRecursiveFunctions:
    """Test monomorphization with recursive functions"""
    
    @pytest.fixture(scope="class")
    def compiler(self, class_compiler):
        """Reuse class-scoped compiler"""
        return class_compiler
    
    @pytest.fixture(scope="class")
    def runtime(self, class_runtime):
        """Reuse class-scoped runtime"""
        return class_runtime
    
    def test_factorial(self, compiler, runtime):
        """Recursive factorial function"""
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
        """Recursive Fibonacci function"""
        source = """
        fn fib(n) {
            if n <= 1 {
                n
            } else {
                fib(n - 1) + fib(n - 2)
            }
        }
        
        let result = fib(7);
        """
        result = compile_and_execute(source, compiler, runtime)
        
        assert result.success, f"Execution failed: {result.get_errors()}"
        assert result.outputs['result'] == 13  # 7th Fibonacci number
    
    def test_quickselect_algorithm(self, compiler, runtime):
        """Quickselect algorithm - find kth smallest element"""
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
        
        # Test data: [64, 34, 25, 12, 22, 11, 90, 88, 45, 50]
        # Sorted: [11, 12, 22, 25, 34, 45, 50, 64, 88, 90]
        let data = [64, 34, 25, 12, 22, 11, 90, 88, 45, 50];
        let min_val = quickselect_kth_smallest(data, 0);
        let median = quickselect_kth_smallest(data, 4);
        let max_val = quickselect_kth_smallest(data, 9);
        """
        result = compile_and_execute(source, compiler, runtime)
        
        assert result.success, f"Execution failed: {result.get_errors()}"
        assert result.outputs['min_val'] == 11  # 0th element (minimum)
        assert result.outputs['median'] == 34   # 4th element (5th value in sorted order)
        assert result.outputs['max_val'] == 90  # 9th element (maximum)
    
    def test_quickselect_kth_largest(self, compiler, runtime):
        """Quickselect variant - find kth largest element"""
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
        
        fn quickselect_kth_largest(arr, k) {
            let n = len(arr);
            let smallest_k = n - k;
            quickselect_kth_smallest(arr, smallest_k)
        }
        
        let scores = [88, 92, 76, 94, 83];
        let first = quickselect_kth_largest(scores, 1);
        let second = quickselect_kth_largest(scores, 2);
        let third = quickselect_kth_largest(scores, 3);
        """
        result = compile_and_execute(source, compiler, runtime)
        
        assert result.success, f"Execution failed: {result.get_errors()}"
        assert result.outputs['first'] == 94   # 1st largest
        assert result.outputs['second'] == 92  # 2nd largest
        assert result.outputs['third'] == 88   # 3rd largest


class TestComplexPrograms:
    """Test monomorphization with complex multi-function programs"""
    
    @pytest.fixture(scope="class")
    def compiler(self, class_compiler):
        """Reuse class-scoped compiler"""
        return class_compiler
    
    @pytest.fixture(scope="class")
    def runtime(self, class_runtime):
        """Reuse class-scoped runtime"""
        return class_runtime
    
    def test_mathematical_pipeline(self, compiler, runtime):
        """Complex mathematical computation pipeline"""
        source = """
        fn square(x) { x * x }
        fn add(x, y) { x + y }
        fn sub(x, y) { x - y }
        fn mul(x, y) { x * y }
        
        let a = square(3);
        let b = square(4);
        let c = add(a, b);
        let d = mul(c, 2);
        """
        result = compile_and_execute(source, compiler, runtime)
        
        assert result.success, f"Execution failed: {result.get_errors()}"
        assert result.outputs['a'] == 9   # 3^2
        assert result.outputs['b'] == 16  # 4^2
        assert result.outputs['c'] == 25  # 9 + 16
        assert result.outputs['d'] == 50  # 25 * 2
    
    def test_statistics_functions(self, compiler, runtime):
        """Statistical operations on data"""
        source = """
        fn add_three(a, b, c) { a + b + c }
        fn divide(x, y) { x / y }
        
        let data = [10, 20, 30];
        let total = add_three(data[0], data[1], data[2]);
        let average = divide(total, 3);
        """
        result = compile_and_execute(source, compiler, runtime)
        
        assert result.success, f"Execution failed: {result.get_errors()}"
        assert result.outputs['total'] == 60
        assert result.outputs['average'] == 20
    
    def test_data_transformation_pipeline(self, compiler, runtime):
        """Multi-stage data transformation"""
        source = """
        fn double(x) { x * 2 }
        fn add_ten(x) { x + 10 }
        fn square(x) { x * x }
        
        let n1 = 1;
        let n2 = 2;
        let n3 = 3;
        
        let d1 = double(n1);
        let d2 = double(n2);
        let d3 = double(n3);
        
        let i1 = add_ten(d1);
        let i2 = add_ten(d2);
        let i3 = add_ten(d3);
        
        let s1 = square(i1);
        let s2 = square(i2);
        let s3 = square(i3);
        """
        result = compile_and_execute(source, compiler, runtime)
        
        assert result.success, f"Execution failed: {result.get_errors()}"
        assert result.outputs['s1'] == 144  # ((1*2)+10)^2 = 12^2 = 144
        assert result.outputs['s2'] == 196  # ((2*2)+10)^2 = 14^2 = 196
        assert result.outputs['s3'] == 256  # ((3*2)+10)^2 = 16^2 = 256


class TestMixedTypedUntypedFunctions:
    """Test interaction between typed and untyped functions"""
    
    @pytest.fixture(scope="class")
    def compiler(self, class_compiler):
        """Reuse class-scoped compiler"""
        return class_compiler
    
    @pytest.fixture(scope="class")
    def runtime(self, class_runtime):
        """Reuse class-scoped runtime"""
        return class_runtime
    
    def test_typed_calls_untyped(self, compiler, runtime):
        """Typed function calling untyped function"""
        source = """
        fn double(x) { x * 2 }
        fn process(a: i32) -> i32 { double(a) }
        
        let result = process(21);
        """
        result = compile_and_execute(source, compiler, runtime)
        
        assert result.success, f"Execution failed: {result.get_errors()}"
        assert result.outputs['result'] == 42
    
    def test_untyped_calls_typed(self, compiler, runtime):
        """Untyped function calling typed function"""
        source = """
        fn add_typed(a: i32, b: i32) -> i32 { a + b }
        fn wrapper(x) { x * 2 }
        
        let a = wrapper(10);
        let b = add_typed(5, 15);
        """
        result = compile_and_execute(source, compiler, runtime)
        
        assert result.success, f"Execution failed: {result.get_errors()}"
        assert result.outputs['a'] == 20
        assert result.outputs['b'] == 20
    
    def test_mixed_function_composition(self, compiler, runtime):
        """Complex composition of typed and untyped functions"""
        source = """
        fn generic_double(x) { x * 2 }
        fn generic_square(x) { x * x }
        
        let a = generic_double(5);
        let b = generic_double(10);
        let c = generic_square(4);
        """
        result = compile_and_execute(source, compiler, runtime)
        
        assert result.success, f"Execution failed: {result.get_errors()}"
        assert result.outputs['a'] == 10
        assert result.outputs['b'] == 20
        assert result.outputs['c'] == 16


class TestPerformanceValidation:
    """Validate that monomorphization doesn't break functionality"""
    
    @pytest.fixture(scope="class")
    def compiler(self, class_compiler):
        """Reuse class-scoped compiler"""
        return class_compiler
    
    @pytest.fixture(scope="class")
    def runtime(self, class_runtime):
        """Reuse class-scoped runtime"""
        return class_runtime
    
    def test_many_instantiations(self, compiler, runtime):
        """Function with many type instantiations should work"""
        source = """
        fn identity(x) { x }
        
        let a = identity(1);
        let b = identity(2);
        let c = identity(3);
        let d = identity(4);
        let e = identity(5);
        """
        result = compile_and_execute(source, compiler, runtime)
        
        assert result.success, f"Execution failed: {result.get_errors()}"
        assert result.outputs['a'] == 1
        assert result.outputs['e'] == 5
    
    def test_deeply_nested_calls(self, compiler, runtime):
        """Deeply nested function calls"""
        source = """
        fn inc(x) { x + 1 }
        
        let result = inc(inc(inc(inc(inc(0)))));
        """
        result = compile_and_execute(source, compiler, runtime)
        
        assert result.success, f"Execution failed: {result.get_errors()}"
        assert result.outputs['result'] == 5
    
    def test_wide_function_tree(self, compiler, runtime):
        """Many functions called in parallel"""
        source = """
        fn f1(x) { x + 1 }
        fn f2(x) { x + 2 }
        fn f3(x) { x + 3 }
        fn f4(x) { x + 4 }
        fn f5(x) { x + 5 }
        
        let a = f1(10);
        let b = f2(10);
        let c = f3(10);
        let d = f4(10);
        let e = f5(10);
        """
        result = compile_and_execute(source, compiler, runtime)
        
        assert result.success, f"Execution failed: {result.get_errors()}"
        assert result.outputs['a'] == 11
        assert result.outputs['b'] == 12
        assert result.outputs['c'] == 13
        assert result.outputs['d'] == 14
        assert result.outputs['e'] == 15


class TestErrorHandling:
    """Test that monomorphization handles errors gracefully"""
    
    @pytest.fixture(scope="class")
    def compiler(self, class_compiler):
        """Reuse class-scoped compiler"""
        return class_compiler
    
    def test_type_error_caught(self, compiler):
        """Type errors should still be caught with monomorphization"""
        source = """
        fn add_ints(a: i32, b: i32) -> i32 { a + b }
        let result = add_ints(1, 3.14);
        """
        result = compiler.compile(source, source_file="<test>")
        
        assert not result.success, "Should fail due to type mismatch"
        errors = result.get_errors()
        assert len(errors) > 0, "Should have error messages"
    
    def test_arity_error_caught(self, compiler):
        """Arity errors should still be caught"""
        source = """
        fn add(a: i32, b: i32) -> i32 { a + b }
        let result = add(1);
        """
        result = compiler.compile(source, source_file="<test>")
        
        assert not result.success, "Should fail due to arity mismatch"
        errors = result.get_errors()
        assert len(errors) > 0, "Should have error messages"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

