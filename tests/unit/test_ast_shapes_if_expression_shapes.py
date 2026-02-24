"""
Tests for If-Expression Shape Unification

Tests that if-expressions validate shape compatibility across branches.
"""

import pytest


class TestIfExpressionShapeUnification:
    """Test if-expression shape unification and validation"""
    
    def test_if_expression_same_shape(self, compiler):
        """Test if-expression with same shapes in both branches"""
        source = """
        let result = if true {
            [[1, 2], [3, 4]]
        } else {
            [[5, 6], [7, 8]]
        };
        """
        
        result = compiler.compile(source, "<test>")
        assert result.success
    
    def test_if_expression_shape_mismatch(self, compiler):
        """Test if-expression with different shapes in branches is allowed (no shape/rank requirement)"""
        source = """
        let result = if true {
            [[1, 2], [3, 4]]
        } else {
            [[5, 6]]
        };
        """
        
        result = compiler.compile(source, "<test>")
        assert result.success
    
    def test_if_expression_nested_arrays(self, compiler):
        """Test if-expression with nested array shapes"""
        source = """
        let result = if true {
            [[[1, 2], [3, 4]], [[5, 6], [7, 8]]]
        } else {
            [[[9, 10], [11, 12]], [[13, 14], [15, 16]]]
        };
        """
        
        result = compiler.compile(source, "<test>")
        assert result.success
    
    def test_if_expression_one_branch(self, compiler):
        """Test if-expression without else branch - should be invalid"""
        source = """
        let result = if true {
            [1, 2, 3]
        };
        """

        result = compiler.compile(source, "<test>")
        # If-expressions without else branches should be invalid when used as expressions
        assert not result.success
        errors = result.get_errors()
        error_str = " ".join(str(e) for e in errors).lower()
        assert "else" in error_str or "invalid" in error_str or "incomplete" in error_str
    
    def test_if_expression_scalar_branches(self, compiler):
        """Test if-expression with scalar branches"""
        source = """
        let result = if true {
            42
        } else {
            24
        };
        """
        
        result = compiler.compile(source, "<test>")
        assert result.success
    
    def test_if_expression_mixed_scalar_array(self, compiler):
        """Test if-expression with scalar in one branch, array in other (allowed; no shape/rank requirement)"""
        source = """
        let result = if true {
            [1, 2, 3]
        } else {
            42
        };
        """
        
        result = compiler.compile(source, "<test>")
        assert result.success
    
    def test_if_expression_1d_arrays(self, compiler):
        """Test if-expression with 1D arrays"""
        source = """
        let result = if true {
            [1, 2, 3]
        } else {
            [4, 5, 6]
        };
        """
        
        result = compiler.compile(source, "<test>")
        assert result.success
    
    def test_if_expression_3d_arrays_same_shape(self, compiler):
        """Test if-expression with 3D arrays of same shape"""
        source = """
        let result = if true {
            [[[1, 2], [3, 4]], [[5, 6], [7, 8]]]
        } else {
            [[[9, 10], [11, 12]], [[13, 14], [15, 16]]]
        };
        """
        
        result = compiler.compile(source, "<test>")
        assert result.success
    
    def test_if_expression_3d_arrays_different_shape(self, compiler):
        """Test if-expression with 3D arrays of different shapes (allowed; no shape/rank requirement)"""
        source = """
        let result = if true {
            [[[1, 2], [3, 4]], [[5, 6], [7, 8]]]
        } else {
            [[[9, 10]], [[11, 12]]]
        };
        """
        
        result = compiler.compile(source, "<test>")
        assert result.success
    
    def test_if_expression_nested_if(self, compiler):
        """Test nested if-expressions"""
        source = """
        let result = if true {
            if false {
                [1, 2, 3]
            } else {
                [4, 5, 6]
            }
        } else {
            [7, 8, 9]
        };
        """
        
        result = compiler.compile(source, "<test>")
        assert result.success

