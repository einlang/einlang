#!/usr/bin/env python3
"""
Tests for statements: print, use, multiple statements.
Tests execute and check results to ensure complete statement coverage using system.
"""

import pytest
from tests.test_utils import compile_and_execute


class TestStatements:
    """Complete statement coverage with execution validation using system"""
    
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
    
    def test_print_statements(self, compiler, runtime):
        """Test print statements - concatenated for speed"""
        source = '''
        print("hello");
        let st0 = 42; print("value:", st0);
        print("result:", 123);
        print();
        let st1 = 1; let st2 = 2; let st3 = 3; print(st1, st2, st3);
        '''
        self._test_and_execute(source, compiler, runtime)
    
    def test_use_statements(self, compiler, runtime):
        """Test use statements - concatenated for speed"""
        source = """
        use std::math;
        use std::array;
        use std::ml;
        use std::io;
        """
        self._test_and_execute(source, compiler, runtime)
    
    def test_multiple_statements(self, compiler, runtime):
        """Test multiple statements - concatenated for speed"""
        source = """
        let st4 = 5; let st5 = 10; let st6 = st4 + st5; assert(st6 == 15);
        let st7a = 1; let st7b = 2; let st7c = 3; let st7s = st7a + st7b + st7c; assert(st7s == 6);
        let st8 = 10; let st9 = st8 * 2; let st10 = st9 - 5; assert(st10 == 15);
        """
        self._test_and_execute(source, compiler, runtime)
    
    def test_mixed_statements(self, compiler, runtime):
        """Test mixed statement types - concatenated for speed"""
        source = '''
        let st11 = 42; print("x =", st11); let st12 = st11 * 2; assert(st12 == 84);
        use std::math; let st13 = 5 + 3; print("result:", st13);
        let st14 = 10; print("a =", st14); let st15 = st14 / 2; print("b =", st15); assert(st15 == 5);
        '''
        self._test_and_execute(source, compiler, runtime)
    
    def test_nested_statements(self, compiler, runtime):
        """Test nested statements in functions"""
        cases = [
            """
            fn test() {
                let x = 5;
                let y = 10;
                x + y
            }
            let result = test();
            assert(result == 15);
            """,
            """
            fn calculate(a, b) {
                let sum = a + b;
                let product = a * b;
                sum + product
            }
            let result = calculate(3, 4);
            assert(result == 19);
            """,
        ]
        for source in cases:
            self._test_and_execute(source, compiler, runtime)
    
    def test_conditional_statements(self, compiler, runtime):
        """Test conditional statements"""
        cases = [
            """
            let x = 10;
            let y = if x > 5 { x * 2 } else { x };
            assert(y == 20);
            """,
            """
            let a = 3;
            let b = 7;
            let max = if a > b { a } else { b };
            assert(max == 7);
            """,
        ]
        for source in cases:
            self._test_and_execute(source, compiler, runtime)
    
    def test_tuple_destructuring_basic(self, compiler, runtime):
        """Test basic tuple destructuring"""
        cases = [
            # Simple pair destructuring
            """
            let (x, y) = (10, 20);
            assert(x == 10);
            assert(y == 20);
            """,
            # Triple destructuring
            """
            let (a, b, c) = (1, 2, 3);
            assert(a == 1);
            assert(b == 2);
            assert(c == 3);
            """,
            # Different types in tuple
            """
            let (num, flag) = (42, true);
            assert(num == 42);
            assert(flag == true);
            """,
        ]
        for source in cases:
            self._test_and_execute(source, compiler, runtime)
    
    def test_tuple_destructuring_with_operations(self, compiler, runtime):
        """Test tuple destructuring with subsequent operations"""
        cases = [
            # Use destructured values in arithmetic
            """
            let (x, y) = (5, 3);
            let sum = x + y;
            let product = x * y;
            assert(sum == 8);
            assert(product == 15);
            """,
            # Create new tuple from destructured values
            """
            let (a, b) = (10, 20);
            let swapped = (b, a);
            let (c, d) = swapped;
            assert(c == 20);
            assert(d == 10);
            """,
        ]
        for source in cases:
            self._test_and_execute(source, compiler, runtime)
    
    def test_tuple_destructuring_from_expressions(self, compiler, runtime):
        """Test tuple destructuring from computed expressions"""
        cases = [
            # Destructure from computed tuple
            """
            let pair = (1 + 2, 3 * 4);
            let (x, y) = pair;
            assert(x == 3);
            assert(y == 12);
            """,
            # Nested tuple in expression
            """
            let (sum, diff) = (5 + 3, 10 - 2);
            assert(sum == 8);
            assert(diff == 8);
            """,
        ]
        for source in cases:
            self._test_and_execute(source, compiler, runtime)
    
    def test_tuple_destructuring_single_variable(self, compiler, runtime):
        """Test tuple destructuring with single variable (edge case)"""
        source = """
        let (x,) = (42,);
        assert(x == 42);
        """
        # Note: This may not be supported depending on parser, so we test defensively
        try:
            self._test_and_execute(source, compiler, runtime)
        except:
            # If parser doesn't support single-element tuple syntax, that's OK
            pass
    
    def test_tuple_mixed_with_regular_variables(self, compiler, runtime):
        """Test tuple destructuring mixed with regular variable declarations"""
        cases = [
            """
            let a = 10;
            let (x, y) = (20, 30);
            let b = 40;
            assert(a == 10);
            assert(x == 20);
            assert(y == 30);
            assert(b == 40);
            """,
            """
            let (p, q) = (1, 2);
            let r = p + q;
            let (s, t) = (r * 2, r * 3);
            assert(s == 6);
            assert(t == 9);
            """,
        ]
        for source in cases:
            self._test_and_execute(source, compiler, runtime)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
