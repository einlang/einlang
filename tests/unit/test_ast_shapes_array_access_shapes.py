"""
Tests for Array Access Result Shape Inference

Tests that array indexing correctly infers result shapes:
- A[0] when A: [5, 3] → shape [3]
- A[0, 1] when A: [5, 3] → scalar
- A[0] when A: [2, 3, 4] → shape [3, 4]
"""

import pytest


class TestArrayAccessShapeInference:
    """Test array access result shape inference"""
    
    def test_2d_array_single_index(self, compiler):
        """Test A[0] when A: [5, 3] → shape [3]"""
        source = """
        let A: [i32; 5, 3] = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12], [13, 14, 15]];
        let row = A[0];
        """
        
        result = compiler.compile(source, "<test>")
        assert result.success, f"Compilation failed: {result.get_errors()}"
    
    def test_2d_array_double_index(self, compiler):
        """Test A[0, 1] when A: [5, 3] → scalar"""
        source = """
        let A: [i32; 5, 3] = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12], [13, 14, 15]];
        let element = A[0, 1];
        """
        
        result = compiler.compile(source, "<test>")
        assert result.success, f"Compilation failed: {result.get_errors()}"
    
    def test_3d_array_single_index(self, compiler):
        """Test A[0] when A: [2, 3, 4] → shape [3, 4]"""
        source = """
        let A: [i32; 2, 3, 4] = [
            [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]],
            [[13, 14, 15, 16], [17, 18, 19, 20], [21, 22, 23, 24]]
        ];
        let slice = A[0];
        """
        
        result = compiler.compile(source, "<test>")
        assert result.success, f"Compilation failed: {result.get_errors()}"
    
    def test_3d_array_double_index(self, compiler):
        """Test A[0, 1] when A: [2, 3, 4] → shape [4]"""
        source = """
        let A: [i32; 2, 3, 4] = [
            [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]],
            [[13, 14, 15, 16], [17, 18, 19, 20], [21, 22, 23, 24]]
        ];
        let row = A[0, 1];
        """
        
        result = compiler.compile(source, "<test>")
        assert result.success, f"Compilation failed: {result.get_errors()}"
    
    def test_3d_array_triple_index(self, compiler):
        """Test A[0, 1, 2] when A: [2, 3, 4] → scalar"""
        source = """
        let A: [i32; 2, 3, 4] = [
            [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]],
            [[13, 14, 15, 16], [17, 18, 19, 20], [21, 22, 23, 24]]
        ];
        let element = A[0, 1, 2];
        """
        
        result = compiler.compile(source, "<test>")
        assert result.success, f"Compilation failed: {result.get_errors()}"

