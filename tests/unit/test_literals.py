#!/usr/bin/env python3
"""
Tests for literal values: integers, floats, booleans, strings, arrays.
Tests execute and check results to ensure complete literal coverage using system.
"""

import pytest
from tests.test_utils import compile_and_execute


class TestLiterals:
    """Complete literal coverage with execution validation using system"""
    
    def _test_and_execute(self, source: str, compiler, runtime, expected_result=None) -> dict:
        """Execute source using system and return result for checking"""
        result = compile_and_execute(source, compiler, runtime)
        assert result.success, f"Execution failed: {result.errors}"
        return result.outputs
    
    def test_integers(self, compiler, runtime):
        """Test integer literals with edge cases and boundary values"""
        cases = [
            ("let x = 0;", "x", 0),
            ("let x = 42;", "x", 42),
            ("let x = -17;", "x", -17),
            ("let x = 123456789;", "x", 123456789),
            ("let x = 2147483647;", "x", 2147483647),
            ("let x = -2147483648;", "x", -2147483648),
            ("let x = 1;", "x", 1),
            ("let x = -1;", "x", -1),
        ]
        for source, var_name, expected_value in cases:
            result = self._test_and_execute(source, compiler, runtime)
            assert var_name in result, f"Variable {var_name} not found in result"
            assert result[var_name] == expected_value, f"Expected {expected_value}, got {result[var_name]}"
    
    def test_integer_arithmetic_boundaries(self, compiler, runtime):
        """Test integer arithmetic with boundary values to ensure proper type handling"""
        cases = [
            "let x = 1000000 + 1000000; assert(x == 2000000);",
            "let x = 2147483647; let y = 0; let z = x + y; assert(z == 2147483647);",
            "let x = -2147483648; let y = 0; let z = x + y; assert(z == -2147483648);",
            "let x = 100000 * 100; assert(x == 10000000);",
        ]
        for source in cases:
            self._test_and_execute(source, compiler, runtime)
    
    def test_floats(self, compiler, runtime):
        """Test float literals with various precisions and scientific notation"""
        cases = [
            ("let x = 0.0;", "x", 0.0),
            ("let x = 3.14159;", "x", 3.14159),
            ("let x = 2.71828;", "x", 2.71828),
            ("let x = 1e-5;", "x", 1e-5),
            ("let x = 1e5;", "x", 1e5),
            ("let x = 1.23e-10;", "x", 1.23e-10),
            ("let x = -3.14159;", "x", -3.14159),
            ("let x = 0.1;", "x", 0.1),
            ("let x = 1.0e-20;", "x", 1.0e-20),
        ]
        for source, var_name, expected_value in cases:
            result = self._test_and_execute(source, compiler, runtime)
            assert var_name in result, f"Variable {var_name} not found in result"
            assert abs(result[var_name] - expected_value) < 1e-5, f"Expected {expected_value}, got {result[var_name]}"
    
    def test_float_precision(self, compiler, runtime):
        """Test floating point precision and rounding behavior"""
        cases = [
            ("let x = 0.1 + 0.2; assert(x > 0.29 && x < 0.31);", True),
            ("let x = 1.0 / 3.0; let y = x * 3.0; assert(y > 0.99 && y < 1.01);", True),
            ("let pi = 3.14159; let circumference = pi * 2.0; assert(circumference > 6.28 && circumference < 6.29);", True),
        ]
        for source, _ in cases:
            self._test_and_execute(source, compiler, runtime)
    
    def test_booleans(self, compiler, runtime):
        """Test boolean literals"""
        cases = [
            ("let x = true;", "x", True),
            ("let x = false;", "x", False),
        ]
        for source, var_name, expected_value in cases:
            result = self._test_and_execute(source, compiler, runtime)
            assert var_name in result, f"Variable {var_name} not found in result"
            assert result[var_name] == expected_value, f"Expected {expected_value}, got {result[var_name]}"
    
    def test_strings(self, compiler, runtime):
        """Test string literals"""
        cases = [
            ('let x = "";', "x", ""),
            ('let x = "hello";', "x", "hello"),
            ('let x = "world";', "x", "world"),
            ('let x = "hello world";', "x", "hello world"),
        ]
        for source, var_name, expected_value in cases:
            result = self._test_and_execute(source, compiler, runtime)
            assert var_name in result, f"Variable {var_name} not found in result"
            assert result[var_name] == expected_value, f"Expected {expected_value}, got {result[var_name]}"
    
    def test_arrays(self, compiler, runtime):
        """Test array literals with various sizes - tensors are np.ndarray at runtime"""
        import numpy as np
        cases = [
            ("let x = [];", "x", []),
            ("let x = [1];", "x", [1]),
            ("let x = [1, 2, 3];", "x", [1, 2, 3]),
            ("let x = [1, 2, 3, 4, 5];", "x", [1, 2, 3, 4, 5]),
            ("let x = [0, 0, 0, 0];", "x", [0, 0, 0, 0]),
            ("let x = [-1, -2, -3];", "x", [-1, -2, -3]),
            ("let x = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000];", "x", 
             [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]),
        ]
        for source, var_name, expected_value in cases:
            result = self._test_and_execute(source, compiler, runtime)
            assert var_name in result, f"Variable {var_name} not found in result"
            np.testing.assert_array_equal(result[var_name], np.array(expected_value), 
                err_msg=f"Expected {expected_value}, got {result[var_name]}")
    
    def test_array_element_access(self, compiler, runtime):
        """Test array element access with boundary cases"""
        import numpy as np
        cases = [
            ("let x = [1, 2, 3]; let y = x[0]; assert(y == 1);", True),
            ("let x = [10, 20, 30, 40]; let y = x[3]; assert(y == 40);", True),
            ("let x = [5, 10, 15]; let first = x[0]; let last = x[2]; assert(first + last == 20);", True),
        ]
        for source, _ in cases:
            self._test_and_execute(source, compiler, runtime)
    
    def test_nested_arrays(self, compiler, runtime):
        """Test nested array literals with various shapes - tensors are np.ndarray at runtime"""
        import numpy as np
        cases = [
            ("let x = [[]];", "x", [[]]),
            ("let x = [[1, 2], [3, 4]];", "x", [[1, 2], [3, 4]]),
            ("let x = [[[1, 2], [3, 4]], [[5, 6], [7, 8]]];", "x", [[[1, 2], [3, 4]], [[5, 6], [7, 8]]]),
            ("let x = [[1], [2], [3]];", "x", [[1], [2], [3]]),
            ("let x = [[10, 20, 30], [40, 50, 60]];", "x", [[10, 20, 30], [40, 50, 60]]),
        ]
        for source, var_name, expected_value in cases:
            result = self._test_and_execute(source, compiler, runtime)
            assert var_name in result, f"Variable {var_name} not found in result"
            np.testing.assert_array_equal(result[var_name], np.array(expected_value),
                err_msg=f"Expected {expected_value}, got {result[var_name]}")
    
    def test_nested_array_access(self, compiler, runtime):
        """Test multi-dimensional array access"""
        cases = [
            ("let matrix = [[1, 2], [3, 4]]; let val = matrix[0, 0]; assert(val == 1);", True),
            ("let matrix = [[1, 2], [3, 4]]; let val = matrix[1, 1]; assert(val == 4);", True),
            ("let matrix = [[10, 20, 30], [40, 50, 60]]; let val = matrix[1, 2]; assert(val == 60);", True),
            ("let tensor = [[[1, 2], [3, 4]], [[5, 6], [7, 8]]]; let val = tensor[1, 1, 1]; assert(val == 8);", True),
        ]
        for source, _ in cases:
            self._test_and_execute(source, compiler, runtime)
    
    def test_mixed_literals(self, compiler, runtime):
        """Test literal types - tensors are np.ndarray at runtime (Rust-aligned strict homogeneity)"""
        import numpy as np
        cases = [
            # Heterogeneous arrays (int+float+bool+string) not supported - Rust-aligned strict typing
            # ("let x = [1, 2.5, true, \"hello\"];", "x", [1, 2.5, True, "hello"], True),  
            ("let x = [[1.0, 2.0], [3.14, 2.71]];", "x", [[1.0, 2.0], [3.14, 2.71]], False),  # homogeneous floats
        ]
        for source, var_name, expected_value, is_mixed in cases:
            result = self._test_and_execute(source, compiler, runtime)
            assert var_name in result, f"Variable {var_name} not found in result"
            # Tensors are np.ndarray at runtime - convert expected to numpy for comparison
            # Use dtype=object for mixed-type arrays to preserve original types
            if is_mixed:
                expected_array = np.array(expected_value, dtype=object)
                np.testing.assert_array_equal(result[var_name], expected_array,
                    err_msg=f"Expected {expected_value}, got {result[var_name]}")
            else:
                # Use allclose for float comparisons to handle float32 precision
                expected_array = np.array(expected_value)
                np.testing.assert_allclose(result[var_name], expected_array, rtol=1e-5,
                    err_msg=f"Expected {expected_value}, got {result[var_name]}")
    
    def test_literal_operations(self, compiler, runtime):
        """Test operations on literals (Rust-style: no implicit int→float conversion)"""
        cases = [
            "let x = 5 + 3; assert(x == 8);",
            "let x = 2.5 * 2.0; assert(x == 5.0);",  # Explicit float literal (Rust-style)
            "let x = true && false; assert(x == false);",
            'let x = "hello" + " " + "world"; assert(x == "hello world");',
        ]
        for source in cases:
            self._test_and_execute(source, compiler, runtime)
        
        # Test array concatenation using stdlib function
        cases = [
            "let arr1 = [1, 2]; let arr2 = [3, 4]; let x = std::array::concatenate(arr1, arr2); assert(x == [1, 2, 3, 4]);",
            "let a = [10]; let b = [20, 30]; let c = std::array::concatenate(a, b); assert(c == [10, 20, 30]);",
        ]
        for source in cases:
            self._test_and_execute(source, compiler, runtime)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
