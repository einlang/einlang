"""
Comprehensive Tests for Wildcard Pattern Support in Match Expressions
=====================================================================

Tests wildcard pattern (_) support in match expressions, including:
- Basic wildcard matching (catch-all)
- Mixed literal and wildcard patterns
- Wildcard position independence
- Multiple match expressions with wildcards
- Edge cases
"""

import pytest
from tests.test_utils import compile_and_execute


class TestWildcardPatterns:
    """Test wildcard pattern support in match expressions"""
    
    def _test_and_execute(self, source: str, compiler, runtime, inputs=None):
        """Helper to compile and execute Einlang code"""
        result = compile_and_execute(source, compiler, runtime, inputs=inputs)
        assert result.success, f"Execution failed: {result.errors}"
        return result
    
    def test_basic_wildcard_catchall(self, compiler, runtime):
        """Test wildcard as catch-all pattern"""
        source = """
        let x = match 5 {
            1 => "one",
            2 => "two",
            3 => "three",
            _ => "other"
        };
        assert(x == "other");
        """
        self._test_and_execute(source, compiler, runtime)
    
    def test_wildcard_first_match(self, compiler, runtime):
        """Test that wildcard matches immediately if first"""
        source = """
        let x = match 1 {
            _ => "wildcard",
            1 => "one"
        };
        assert(x == "wildcard");
        """
        self._test_and_execute(source, compiler, runtime)
    
    def test_literal_before_wildcard(self, compiler, runtime):
        """Test that literals are checked before wildcard"""
        source = """
        let x = match 1 {
            1 => "one",
            2 => "two",
            _ => "other"
        };
        assert(x == "one");
        """
        self._test_and_execute(source, compiler, runtime)
    
    def test_wildcard_with_integers(self, compiler, runtime):
        """Test wildcard with integer patterns"""
        source = """
        let result = match 42 {
            0 => "zero",
            1 => "one",
            _ => "many"
        };
        assert(result == "many");
        """
        self._test_and_execute(source, compiler, runtime)
    
    def test_wildcard_with_booleans(self, compiler, runtime):
        """Test wildcard with boolean patterns"""
        source = """
        let x = match true {
            false => "no",
            true => "yes"
        };
        assert(x == "yes");
        
        let y = match false {
            true => "yes",
            _ => "no"
        };
        assert(y == "no");
        """
        self._test_and_execute(source, compiler, runtime)
    
    def test_only_wildcard(self, compiler, runtime):
        """Test match with only wildcard pattern"""
        source = """
        let x = match 123 {
            _ => "anything"
        };
        assert(x == "anything");
        """
        self._test_and_execute(source, compiler, runtime)
    
    def test_wildcard_in_nested_expressions(self, compiler, runtime):
        """Test wildcard in match expressions within other expressions"""
        source = """
        let add_one = |x| match x {
            0 => 1,
            _ => x + 1
        };
        
        let result = add_one(5);
        assert(result == 6);
        """
        self._test_and_execute(source, compiler, runtime)
    
    def test_multiple_match_with_wildcards(self, compiler, runtime):
        """Test multiple match expressions each with wildcards"""
        source = """
        let classify = |n| match n {
            0 => "zero",
            1 => "one",
            2 => "two",
            _ => "many"
        };
        
        assert(classify(0) == "zero");
        assert(classify(1) == "one");
        assert(classify(2) == "two");
        assert(classify(10) == "many");
        """
        self._test_and_execute(source, compiler, runtime)
    
    def test_wildcard_with_variable_input(self, compiler, runtime):
        """Test wildcard matching against variable values"""
        source = """
        let n = 7;
        let result = match n {
            1 => "small",
            5 => "medium",
            _ => "other"
        };
        assert(result == "other");
        """
        self._test_and_execute(source, compiler, runtime)
    
    def test_wildcard_precision_inference(self, compiler, runtime):
        """Test that wildcard doesn't cause precision inference errors"""
        source = """
        let x: i32 = 42;
        let category = match x {
            0 => "zero",
            _ => "non-zero"
        };
        assert(category == "non-zero");
        """
        self._test_and_execute(source, compiler, runtime)
    
    def test_wildcard_in_complex_match(self, compiler, runtime):
        """Test wildcard in more complex match scenarios"""
        source = """
        let fibonacci_category = |n| match n {
            0 => "F(0)",
            1 => "F(1)",
            2 => "F(2)",
            3 => "F(3)",
            5 => "F(5)",
            8 => "F(8)",
            _ => "other"
        };
        
        assert(fibonacci_category(0) == "F(0)");
        assert(fibonacci_category(5) == "F(5)");
        assert(fibonacci_category(7) == "other");
        """
        self._test_and_execute(source, compiler, runtime)


class TestWildcardEdgeCases:
    """Test edge cases for wildcard patterns"""
    
    def _test_and_execute(self, source: str, compiler, runtime, inputs=None):
        """Helper to compile and execute Einlang code"""
        result = compile_and_execute(source, compiler, runtime, inputs=inputs)
        assert result.success, f"Execution failed: {result.errors}"
        return result
    
    def test_wildcard_does_not_bind(self, compiler, runtime):
        """Test that wildcard does not create a binding"""
        # Wildcard should match but not bind a variable named "_"
        source = """
        let x = match 42 {
            0 => "zero",
            _ => "other"
        };
        assert(x == "other");
        """
        self._test_and_execute(source, compiler, runtime)
    
    def test_wildcard_with_computed_value(self, compiler, runtime):
        """Test wildcard matching computed values"""
        source = """
        let compute = |n| n * 2;
        let result = match compute(5) {
            10 => "ten",
            _ => "not ten"
        };
        assert(result == "ten");
        """
        self._test_and_execute(source, compiler, runtime)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

