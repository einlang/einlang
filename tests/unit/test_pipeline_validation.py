"""
Tests for Pipeline Type Validation Pass
========================================

Tests compile-time type validation for pipeline expressions.
"""

import pytest


class TestPipelineTypeValidation:
    """Test pipeline type validation"""
    
    def test_standard_pipeline_valid(self, compiler):
        """Test valid standard pipeline"""
        source = """
        let result = 42
            |> |x: i32| x * 2
            |> |x: i32| x + 10;
        """
        context = compiler.compile(source, "<test>")
        assert context.success, f"Compilation failed: {context.get_errors()}"
    
    def test_pipeline_chaining_valid(self, compiler):
        """Test valid pipeline chaining"""
        source = """
        let result = 10
            |> |x: i32| x + 5
            |> |x: i32| x * 2
            |> |x: i32| x - 3;
        """
        context = compiler.compile(source, "<test>")
        assert context.success, f"Compilation failed: {context.get_errors()}"
    
    def test_pipeline_with_lambda_parameter_inference(self, compiler):
        """Test pipeline with lambda parameter type inference"""
        source = """
        let result = 42
            |> |x| x * 2
            |> |x| x + 10;
        """
        context = compiler.compile(source, "<test>")
        # Should work with type inference
        assert context.success, f"Compilation failed: {context.get_errors()}"

