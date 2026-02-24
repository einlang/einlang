#!/usr/bin/env python3
"""
Tests for declarations: variables, functions, tensors.
Tests execute and check results to ensure complete declaration coverage using system.
"""

import pytest
from tests.test_utils import compile_and_execute


class TestDeclarations:
    """Complete declaration coverage with execution validation using system"""
    
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
    
    def test_variables(self, compiler, runtime):
        """Test variable declarations - concatenated for speed"""
        source = """
        let d0 = 5; assert(d0 == 5);
        let d1 = 42; assert(d1 == 42);
        let d2a = 10; let d2b = d2a; assert(d2b == 10);
        let d3 = 0; assert(d3 == 0);
        let d4 = -100; assert(d4 == -100);
        let d5 = 1000000; assert(d5 == 1000000);
        let d6x = 1; let d6y = 2; let d6z = 3; assert(d6x + d6y + d6z == 6);
        """
        self._test_and_execute(source, compiler, runtime)
    
    def test_typed_variables(self, compiler, runtime):
        """Test typed variable declarations - concatenated for speed"""
        source = """
        let d7: i32 = 42; assert(d7 == 42);
        let d8: f32 = 3.14; assert(d8 == 3.14);
        let d9: bool = true; assert(d9 == true);
        let d10: str = "hello"; assert(d10 == "hello");
        let d11: i32 = 2147483647; assert(d11 > 0);
        let d12: i32 = -2147483648; assert(d12 < 0);
        let d13: f32 = 3.14159; assert(d13 > 3.0 && d13 < 4.0);
        let d14: bool = false; assert(d14 == false);
        """
        self._test_and_execute(source, compiler, runtime)
    
    def test_functions(self, compiler, runtime):
        """Test function definitions - concatenated for speed"""
        source = """
        fn d15get() { 42 } let d15 = d15get(); assert(d15 == 42);
        fn d16add(a, b) { a + b } let d16 = d16add(2, 3); assert(d16 == 5);
        fn d17sq(x: i32) -> i32 { x * x } let d17 = d17sq(4); assert(d17 == 16);
        fn d18mul(a, b, c) { a * b * c } let d18 = d18mul(2, 3, 4); assert(d18 == 24);
        fn d19max(a, b) { if a > b { a } else { b } } assert(d19max(10, 20) == 20);
        fn d20fac(n) { if n <= 1 { 1 } else { n * d20fac(n - 1) } } assert(d20fac(5) == 120);
        fn d21id(x) { x } let d21 = d21id(100); assert(d21 == 100);
        """
        self._test_and_execute(source, compiler, runtime)
    
    def test_function_return_types(self, compiler, runtime):
        """Test functions with explicit return types - concatenated"""
        source = """
        fn d22gi() -> i32 { 42 } let d22 = d22gi(); assert(d22 == 42);
        fn d23gf() -> f32 { 3.14 } let d23 = d23gf(); assert(d23 == 3.14);
        fn d24gb() -> bool { true } let d24 = d24gb(); assert(d24 == true);
        fn d25cmp(a: i32, b: i32) -> i32 { a + b * 2 } assert(d25cmp(5, 10) == 25);
        """
        self._test_and_execute(source, compiler, runtime)
    
    def test_variable_scoping(self, compiler, runtime):
        """Test variable scoping - concatenated for speed"""
        source = """
        let d26x = 10; let d26y = d26x; assert(d26y == 10);
        let d27a = 5; let d27b = d27a * 2; assert(d27b == 10);
        let d28x = 5; fn d28t() { let x = 10; x } let d28r = d28t(); assert(d28r == 10 && d28x == 5);
        fn d29add(x, y) { x + y } let d29x = 1; let d29y = 2; let d29r = d29add(10, 20); assert(d29r == 30 && d29x == 1);
        let d30out = 100; fn d30in() { d30out * 2 } assert(d30in() == 200);
        fn d31nest() { let a = 5; fn inr() { a * 2 } inr() } assert(d31nest() == 10);
        """
        self._test_and_execute(source, compiler, runtime)
    
    def test_tensor_assertions(self, compiler, runtime):
        """Test assert with tensor comparisons - concatenated for speed"""
        source = """
        let ta0a = [1, 2, 3]; let ta0b = [1, 2, 3]; assert(ta0a[0] == ta0b[0] && ta0a[1] == ta0b[1]);
        let ta1 = [[1, 2, 3], [4, 5, 6]]; assert(len(ta1) == 2); assert(len(ta1[0]) == 3);
        let ta2d = [1, 2, 3]; let ta2s[i in 0..3] = ta2d[i] * ta2d[i]; assert(ta2s[1] == 4);
        let ta3m = [[1, 2], [3, 4]]; let ta3d[i in 0..2] = ta3m[i,i]; assert(ta3d[0] == 1 && ta3d[1] == 4);
        let ta4[i] = [1, 2, 3][i] + 5; assert(ta4 == [6, 7, 8]);
        let ta5[i] = [2, 6, 10][i]; assert(ta5 == [2, 6, 10]);
        let ta6[i,j] = [[1, 2], [3, 4]][i,j] * 10; assert(ta6 == [[10, 20], [30, 40]]);
        let ta7v = [1, 2, 3, 4]; let ta7t = sum[i](ta7v[i]); assert(ta7t == 10);
        let ta8m = [[1, 2], [3, 4]]; let ta8r = sum[j](ta8m[0,j]); assert(ta8r == 3);
        let ta9bd = [[[1, 2], [3, 4]], [[5, 6], [7, 8]]]; let ta9r[batch_idx in 0..2, row in 0..2] = sum[col](ta9bd[batch_idx, row, col]); assert(ta9r[0, 0] == 3 && ta9r[0, 1] == 7);
        let ta10b = [2, 4, 6]; let ta10s[i in 0..3] = ta10b[i] / 2; assert(ta10s == [1, 2, 3]);
        let ta11[i in 0..3] = [10, 20, 30][i] - 5; assert(ta11[1] == 15 && ta11 == [5, 15, 25]);
        """
        self._test_and_execute(source, compiler, runtime)
    
    @pytest.mark.skip(reason="function_type annotation not yet resolved from lark Tree to FunctionType")
    def test_advanced_types(self, compiler, runtime):
        """Test advanced type annotations - concatenated for speed"""
        source = """
        let at0: (i32, i32) -> i32 = |a, b| a + b; let at0r = at0(3, 4); assert(at0r == 7);
        let at1: [i32; 3, 3] = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]; assert(at1[1,1] == 5);
        let at2: [f32; *] = [1.0, 2.0, 3.0]; assert(len(at2) == 3);
        """
        self._test_and_execute(source, compiler, runtime)
    
    def test_complex_where_clauses(self, compiler, runtime):
        """Test complex where clause patterns - concatenated for speed"""
        source = """
        let wc0 = [x * 2 | x in 1..10, x % 2 == 0, x > 4]; assert(len(wc0) == 2);
        let wc1d = [1, 2, 3]; let wc1p = [result | x in wc1d, y = x * 2, result = y + 1]; assert(len(wc1p) > 0);
        let wc2m = [[1, 2], [3, 4]]; let wc2s = [sum[j](row[j]) | row in wc2m]; assert(len(wc2s) == len(wc2m));
        """
        self._test_and_execute(source, compiler, runtime)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
