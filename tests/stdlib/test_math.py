#!/usr/bin/env python3
"""
Tests for std::math module functionality with string source examples.
Tests execute and check results to ensure complete math stdlib coverage.
"""

import pytest
from ..test_utils import compile_and_execute
from einlang.shared.errors import EinlangSourceError


class TestMathModule:
    """Complete std::math coverage with execution validation"""
    
    def _test_and_execute(self, source: str, compiler, runtime, expected_result=None) -> dict:
        """Execute source and return result for checking"""
        execution_result = compile_and_execute(source, compiler, runtime)
        assert execution_result.success, f"Execution failed: {execution_result.errors}"
        
        result = execution_result.value
        if expected_result is not None:
            assert result == expected_result
        return result
    
    def test_basic_functions(self, compiler, runtime):
        """Test basic math functions (OPTIMIZED: combined into single source)"""
        source = """
use std::math;
// Abs, min, max - basic and edge cases
assert(math::abs(-5) == 5);
assert(math::abs(3) == 3);
assert(std::math::abs(0) == 0);
assert(std::math::abs(-0) == 0);
assert(math::min(3, 7) == 3);
assert(math::max(3, 7) == 7);
assert(std::math::min(-1000, 1000) == -1000);
assert(std::math::max(-1000, 1000) == 1000);

// Sqrt - float values (stdlib sqrt expects float for x ** 0.5)
assert(math::sqrt(16.0) == 4.0);
assert(math::sqrt(25.0) == 5.0);

// Sqrt - floating point
let result = std::math::sqrt(0.25);
assert(result > 0.49);
assert(result < 0.51);
        """
        self._test_and_execute(source, compiler, runtime)
    
    def test_trigonometry(self, compiler, runtime):
        """Test trigonometric functions (OPTIMIZED: combined into single source)"""
        source = """
use std::math;
// Basic trig at zero
assert(math::sin(0.0) == 0.0);
assert(math::cos(0.0) == 1.0);
assert(math::tan(0.0) == 0.0);

// Trig with pi
let pi = math::pi();
let result = math::sin(pi/2.0);
assert(result > 0.9);
let angle = pi/4.0;
let sin_val = math::sin(angle);
let cos_val = math::cos(angle);
let diff = math::abs(sin_val - cos_val);
assert(diff < 0.01);
        """
        self._test_and_execute(source, compiler, runtime)
    
    def test_exponential(self, compiler, runtime):
        """Test power functions - concatenated for speed"""
        source = """
        use std::math;
        assert(math::pow(2.0, 3.0) == 8.0);
        assert(math::pow(5.0, 2.0) == 25.0);
        assert(std::math::pow(0, 5) == 0);
        assert(std::math::pow(1, 100) == 1);
        """
        self._test_and_execute(source, compiler, runtime)
    
    def test_constants(self, compiler, runtime):
        """Test mathematical constants - concatenated for speed"""
        source = """
        use std::math;
        let pi = std::math::pi();
        assert(pi > 3.14);
        assert(pi < 3.15);
        """
        self._test_and_execute(source, compiler, runtime)
    


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
