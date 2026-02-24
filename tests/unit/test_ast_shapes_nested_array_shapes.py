"""
Tests for Nested Array Shape Inference

Tests the recursive shape inference for arrays of arbitrary depth (3D+).
"""

import pytest


class TestNestedArrayShapeInference:
    """Test nested array shape inference (3D+ arrays)"""
    
    def test_1d_array_shape(self, compiler):
        """Test shape inference for 1D arrays"""
        source = "let x = [1, 2, 3];"
        result = compiler.compile(source, "<test>")
        assert result.success
    
    def test_2d_array_shape(self, compiler):
        """Test shape inference for 2D arrays"""
        source = "let matrix = [[1, 2], [3, 4]];"
        
        result = compiler.compile(source, "<test>")
        assert result.success
    
    def test_3d_array_shape(self, compiler):
        """Test shape inference for 3D arrays"""
        source = "let tensor3d = [[[1, 2], [3, 4]], [[5, 6], [7, 8]]];"
        
        result = compiler.compile(source, "<test>")
        assert result.success
    
    def test_4d_array_shape(self, compiler):
        """Test shape inference for 4D arrays"""
        source = """
        let tensor4d = [
            [[[1, 2], [3, 4]], [[5, 6], [7, 8]]],
            [[[9, 10], [11, 12]], [[13, 14], [15, 16]]]
        ];
        """
        
        result = compiler.compile(source, "<test>")
        assert result.success
    
    def test_irregular_array_error(self, compiler):
        """Test that irregular arrays (mismatched inner sizes) are rejected."""
        source = """
        let irregular = [[1, 2], [3, 4, 5]];  # Different inner sizes
        """
        
        result = compiler.compile(source, "<test>")
        assert not result.success
        errors = result.get_errors()
        error_str = " ".join(str(e) for e in errors).lower()
        assert "inconsistent shapes" in error_str or "inconsistent" in error_str
    
    def test_mixed_types_error(self, compiler):
        """Test that mixed scalar/array elements are rejected."""
        source = """
        let mixed = [1, [2, 3]];  # Scalar and array mixed
        """
        
        result = compiler.compile(source, "<test>")
        assert not result.success
        errors = result.get_errors()
        error_str = " ".join(str(e) for e in errors).lower()
        assert "inconsistent element types" in error_str or "inconsistent" in error_str
    
    def test_empty_array_shape(self, compiler):
        """Test shape inference for empty arrays"""
        source = "let empty = [];"
        
        result = compiler.compile(source, "<test>")
        assert result.success
    
    def test_nested_empty_array(self, compiler):
        """Test shape inference for nested empty arrays"""
        source = "let nested_empty = [[], []];"
        
        result = compiler.compile(source, "<test>")
        assert result.success
    
    def test_3d_with_different_values(self, compiler):
        """Test 3D array with different numeric values"""
        source = """
        let tensor = [
            [[10, 20], [30, 40]],
            [[50, 60], [70, 80]],
            [[90, 100], [110, 120]]
        ];
        """
        
        result = compiler.compile(source, "<test>")
        assert result.success
    
    def test_irregular_3d_error(self, compiler):
        """Test that irregular 3D arrays are rejected."""
        source = """
        let irregular = [
            [[1, 2], [3, 4]],
            [[5, 6]]  # Different shape!
        ];
        """
        
        result = compiler.compile(source, "<test>")
        assert not result.success
        errors = result.get_errors()
        error_str = " ".join(str(e) for e in errors).lower()
        assert "inconsistent" in error_str



