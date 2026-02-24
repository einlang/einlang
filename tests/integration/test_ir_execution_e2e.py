"""
End-to-End Integration Tests for IR Execution Path
===================================================

Tests the complete pipeline: Source → Parser → AST → IR Lowering → IR Backend → Results

This ensures the IR path produces correct results.
"""

import pytest
import numpy as np
from tests.test_utils import compile_and_execute


class TestSimpleExpressions:
    """Test simple expressions through IR path"""
    
    def test_simple_arithmetic(self, compiler, runtime):
        """Test basic arithmetic: 5 + 3"""
        source = "let x = 5 + 3;"
        result = compile_and_execute(source, compiler, runtime)
        assert result.success, f"Execution failed: {result.errors}"
        assert result.outputs['x'] == 8
    
    def test_multiple_operations(self, compiler, runtime):
        """Test: (10 + 5) * 2"""
        source = "let y = (10 + 5) * 2;"
        result = compile_and_execute(source, compiler, runtime)
        assert result.success, f"Execution failed: {result.errors}"
        assert result.outputs['y'] == 30
    
    def test_negative_numbers(self, compiler, runtime):
        """Test: -5 + 10"""
        source = "let z = -5 + 10;"
        result = compile_and_execute(source, compiler, runtime)
        assert result.success, f"Execution failed: {result.errors}"
        assert result.outputs['z'] == 5


class TestVariables:
    """Test variable declarations and references"""
    
    def test_variable_reference(self, compiler, runtime):
        """Test referencing previously declared variables"""
        source = """
        let a = 10;
        let b = a + 5;
        """
        result = compile_and_execute(source, compiler, runtime)
        assert result.success, f"Execution failed: {result.errors}"
        assert result.outputs['a'] == 10
        assert result.outputs['b'] == 15
    
    def test_multiple_variable_chain(self, compiler, runtime):
        """Test chain of variable references"""
        source = """
        let x = 5;
        let y = x * 2;
        let z = y + x;
        """
        result = compile_and_execute(source, compiler, runtime)
        assert result.success, f"Execution failed: {result.errors}"
        assert result.outputs['x'] == 5
        assert result.outputs['y'] == 10
        assert result.outputs['z'] == 15


class TestASTvsIRComparison:
    """Test IR execution path correctness"""
    
    def test_variable_chain_parity(self, compiler, runtime):
        """Ensure complex variable chains match"""
        source = """
        let a = 5;
        let b = a + 3;
        let c = b * 2;
        let d = c - a;
        """
        result = compile_and_execute(source, compiler, runtime)
        assert result.success, f"Execution failed: {result.errors}"
        assert result.outputs['a'] == 5
        assert result.outputs['b'] == 8
        assert result.outputs['c'] == 16
        assert result.outputs['d'] == 11


class TestErrorHandling:
    """Test error handling in IR path"""
    
    def test_undefined_variable_error(self, compiler, runtime):
        """Test that undefined variables are caught"""
        source = "let y = undefined_var;"
        result = compile_and_execute(source, compiler, runtime)
        
        # Undefined variables should be caught - either during compilation or execution
        assert not result.success, "Should fail for undefined variable"
        assert len(result.errors) > 0, "Should have error messages for undefined variable"
        error_str = str(result.errors[0]).lower()
        assert "undefined" in error_str or "not found" in error_str or "unknown" in error_str, \
            f"Error should mention undefined variable, got: {result.errors}"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

