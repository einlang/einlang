#!/usr/bin/env python3
"""
Tests for Match Exhaustiveness Checking Pass

Tests compile-time validation that match expressions cover all cases.
"""

import pytest


class TestExhaustiveness:
    """Test match exhaustiveness checking"""
    
    def test_boolean_exhaustive_both_cases(self, compiler):
        """Test that boolean match with both true and false is exhaustive"""
        source = """
        let x = true;
        let result = match x {
            true => 1,
            false => 0
        };
        """
        
        result = compiler.compile(source, "<test>")
        assert result.success, f"Compilation failed: {result.get_errors()}"
    
    def test_boolean_exhaustive_with_wildcard(self, compiler):
        """Test that boolean match with wildcard is exhaustive"""
        source = """
        let x = true;
        let result = match x {
            true => 1,
            _ => 0
        };
        """
        
        result = compiler.compile(source, "<test>")
        assert result.success, f"Compilation failed: {result.get_errors()}"
    
    def test_boolean_exhaustive_with_identifier(self, compiler):
        """Test that boolean match with identifier pattern is exhaustive"""
        source = """
        let x = true;
        let result = match x {
            true => 1,
            other => 0
        };
        """
        
        result = compiler.compile(source, "<test>")
        assert result.success, f"Compilation failed: {result.get_errors()}"
    
    def test_boolean_non_exhaustive_missing_false(self, compiler):
        """Test that boolean match missing false case is non-exhaustive"""
        source = """
        let x = true;
        let result = match x {
            true => 1
        };
        """
        
        result = compiler.compile(source, "<test>")
        assert not result.success, "Should fail: missing false case"
        errors = result.get_errors()
        assert any("non-exhaustive" in str(err).lower() for err in errors)
        assert any("false" in str(err).lower() for err in errors)
    
    def test_boolean_non_exhaustive_missing_true(self, compiler):
        """Test that boolean match missing true case is non-exhaustive"""
        source = """
        let x = true;
        let result = match x {
            false => 0
        };
        """
        
        result = compiler.compile(source, "<test>")
        assert not result.success, "Should fail: missing true case"
        errors = result.get_errors()
        assert any("non-exhaustive" in str(err).lower() for err in errors)
        assert any("true" in str(err).lower() for err in errors)
    
    def test_integer_non_exhaustive_no_catch_all(self, compiler):
        """Test that integer match without catch-all is non-exhaustive"""
        source = """
        let x = 5;
        let result = match x {
            0 => "zero",
            1 => "one"
        };
        """
        
        result = compiler.compile(source, "<test>")
        assert not result.success, "Should fail: missing catch-all"
        errors = result.get_errors()
        assert any("non-exhaustive" in str(err).lower() for err in errors)
        assert any("other values" in str(err).lower() or "not covered" in str(err).lower() for err in errors)
    
    def test_integer_exhaustive_with_wildcard(self, compiler):
        """Test that integer match with wildcard is exhaustive"""
        source = """
        let x = 5;
        let result = match x {
            0 => "zero",
            1 => "one",
            _ => "other"
        };
        """
        
        result = compiler.compile(source, "<test>")
        assert result.success, f"Compilation failed: {result.get_errors()}"
    
    def test_integer_exhaustive_with_identifier(self, compiler):
        """Test that integer match with identifier pattern is exhaustive"""
        source = """
        let x = 5;
        let result = match x {
            0 => "zero",
            1 => "one",
            n => "other"
        };
        """
        
        result = compiler.compile(source, "<test>")
        assert result.success, f"Compilation failed: {result.get_errors()}"
    
    def test_unreachable_pattern_after_wildcard(self, compiler):
        """Test that patterns after wildcard are unreachable (should compile but warn)"""
        source = """
        let x = 5;
        let result = match x {
            _ => "catch-all",
            0 => "zero"  # Unreachable
        };
        """
        
        result = compiler.compile(source, "<test>")
        # Should compile (exhaustive with catch-all), but pattern after _ is unreachable
        # The pass currently treats this as an error, which is fine for safety
        # In the future, this could be a warning instead
        assert result.success, f"Compilation failed: {result.get_errors()}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

