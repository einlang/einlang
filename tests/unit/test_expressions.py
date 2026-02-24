#!/usr/bin/env python3
"""
Tests for expressions: precedence, parentheses, if expressions, complex patterns.
Tests execute and check results to ensure complete expression coverage using system.
"""

import pytest
from tests.test_utils import compile_and_execute


class TestExpressions:
    """Complete expression coverage with execution validation using system"""
    
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
    
    def test_if_expressions(self, compiler, runtime):
        """Test if expressions - concatenated for speed"""
        source = """
        let e0 = if true { 5 } else { 3 }; assert(e0 == 5);
        let e1 = if false { 1 } else { 0 }; assert(e1 == 0);
        let e2 = if 5 > 3 { 42 } else { 0 }; assert(e2 == 42);
        """
        self._test_and_execute(source, compiler, runtime)
    
    def test_nested_if(self, compiler, runtime):
        """Test nested if expressions"""
        cases = [
            """let a = 5; let b = 3; let c = 8; 
               let x = if a > b { 
                         if a > c { a } else { c } 
                       } else { 
                         if b > c { b } else { c } 
                       }; 
               assert(x == 8);""",
        ]
        for source in cases:
            self._test_and_execute(source, compiler, runtime)
    
    def test_operator_precedence(self, compiler, runtime):
        """Test operator precedence - concatenated for speed"""
        source = """
        let e3 = 2 + 3 * 4; assert(e3 == 14);
        let e4 = 2 * 3 + 4; assert(e4 == 10);
        let e5 = 2 + 3 * 4 - 1; assert(e5 == 13);
        let e6 = (2 + 3) * 4; assert(e6 == 20);
        let e7 = 2 ** 3 ** 2; assert(e7 == 512);
        """
        self._test_and_execute(source, compiler, runtime)
    
    def test_comparison_operators(self, compiler, runtime):
        """Test comparison operators - concatenated for speed"""
        source = """
        let e8 = 5 > 3; assert(e8 == true);
        let e9 = 5 < 3; assert(e9 == false);
        let e10 = 5 >= 5; assert(e10 == true);
        let e11 = 5 <= 3; assert(e11 == false);
        let e12 = 5 == 5; assert(e12 == true);
        let e13 = 5 != 3; assert(e13 == true);
        """
        self._test_and_execute(source, compiler, runtime)
    
    def test_logical_operators(self, compiler, runtime):
        """Test logical operators - one compile/execute for speed"""
        source = """
        let and_tt = true && true;
        let and_tf = true && false;
        let and_ft = false && true;
        let and_ff = false && false;
        let or_tt = true || true;
        let or_tf = true || false;
        let or_ft = false || true;
        let or_ff = false || false;
        let not_t = !true;
        let not_f = !false;
        """
        result = compile_and_execute(source, compiler, runtime)
        assert result.success, f"Execution failed: {result.errors}"
        out = result.outputs
        assert out['and_tt'] is True
        assert out['and_tf'] is False and out['and_ft'] is False and out['and_ff'] is False
        assert out['or_tt'] is True and out['or_tf'] is True and out['or_ft'] is True
        assert out['or_ff'] is False
        assert out['not_t'] is False and out['not_f'] is True
    
    def test_arithmetic_operators(self, compiler, runtime):
        """Test arithmetic operators - concatenated for speed"""
        source = """
        let e14 = 5 + 3; assert(e14 == 8);
        let e15 = 5 - 3; assert(e15 == 2);
        let e16 = 5 * 3; assert(e16 == 15);
        let e17 = 15 / 3; assert(e17 == 5);
        let e18 = 15 % 4; assert(e18 == 3);
        let e19 = 2 ** 3; assert(e19 == 8);
        """
        self._test_and_execute(source, compiler, runtime)
    
    def test_complex_expressions(self, compiler, runtime):
        """Test complex expression combinations - concatenated for speed"""
        source = """
        let e20 = (2 + 3) * (4 - 1); assert(e20 == 15);
        let e21 = 2 ** 3 + 4 * 5; assert(e21 == 28);
        let e22 = (5 > 3) && (4 < 6); assert(e22 == true);
        let e23 = if (2 + 3) > 4 { 10 } else { 0 }; assert(e23 == 10);
        let e24 = 1 + 2 * 3 - 4 / 2; assert(e24 == 5);
        """
        self._test_and_execute(source, compiler, runtime)
    
    def test_function_calls_in_expressions(self, compiler, runtime):
        """Test function calls within expressions - concatenated for speed"""
        source = """
        fn e25add(a, b) { a + b } let e25 = e25add(2, 3) * 2; assert(e25 == 10);
        fn e26sq(x) { x * x } let e26 = e26sq(3) + e26sq(4); assert(e26 == 25);
        fn e27mx(a, b) { if a > b { a } else { b } } let e27 = e27mx(5, 3) + e27mx(2, 8); assert(e27 == 13);
        """
        self._test_and_execute(source, compiler, runtime)
    
    def test_array_expressions(self, compiler, runtime):
        """Test array expressions - concatenated for speed"""
        source = """
        let e28a = [1, 2, 3]; let e28 = e28a[0] * e28a[1] + e28a[2]; assert(e28 == 5);
        let e29 = len([1, 2, 3, 4, 5]) * 2; assert(e29 == 10);
        let e30 = [1, 2, 3][0] + [4, 5, 6][1]; assert(e30 == 6);
        """
        self._test_and_execute(source, compiler, runtime)
    
    def test_mixed_type_expressions(self, compiler, runtime):
        """Test mixed type expressions - concatenated for speed"""
        source = """
        let e31 = 5.0 + 3.14; assert(e31 > 8.13 && e31 < 8.15);
        let e32 = true && false; assert(e32 == false);
        let e33 = 10 > 5; assert(e33 == true);
        """
        self._test_and_execute(source, compiler, runtime)
    
    def test_lambda_expressions(self, compiler, runtime):
        """Test lambda expressions - concatenated for speed"""
        source = """
        let e34f = |x| x * 2; let e34 = e34f(5); assert(e34 == 10);
        let e35g = |a, b| a + b; let e35 = e35g(3, 4); assert(e35 == 7);
        """
        self._test_and_execute(source, compiler, runtime)
    
    def test_advanced_lambda_expressions(self, compiler, runtime):
        """Test advanced lambda expressions - concatenated for speed"""
        source = """
        let e36d = |x| x * 2; let e36 = e36d(5); assert(e36 == 10);
        let e37a = |x, y| x + y; let e37 = e37a(3, 4); assert(e37 == 7);
        let e38s = |x| x * x; let e38 = e38s(4); assert(e38 == 16);
        let e39 = (|x| x + 1)(5); assert(e39 == 6);
        let e40gf = || 5; let e40 = e40gf(); assert(e40 == 5);
        let e41dt = [1, 2, 3]; let e41pr = |arr| [x * 2 | x in arr]; let e41 = e41pr(e41dt); assert(e41 == [2, 4, 6]);
        """
        self._test_and_execute(source, compiler, runtime)
    
    def test_tuple_expressions(self, compiler, runtime):
        """Test tuple expressions and dot notation access"""
        # Tuple expressions actually work!
        cases = [
            "let point = (3, 4); let x = point.0; let y = point.1; assert(x == 3); assert(y == 4);",
        ]
        for source in cases:
            self._test_and_execute(source, compiler, runtime)
    
    def test_match_expressions(self, compiler, runtime):
        """Test pattern matching expressions - concatenated for speed"""
        source = """
        let e43 = match 3 { 1 => "one", 2 => "two", _ => "other" };
        let e44 = match 2 { 1 => "one", 2 => "two", _ => "other" };
        let e45 = match 10 { 0 => "zero", _ => "non-zero" };
        """
        self._test_and_execute(source, compiler, runtime)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
