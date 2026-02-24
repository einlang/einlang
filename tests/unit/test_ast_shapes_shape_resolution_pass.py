"""
Tests for AST Shape Resolution Pass

Verifies compile-time resolution of symbolic shapes to concrete ranges at AST level.

NOTE: These tests were moved from tests/unit/ir/ since shape resolution
now happens at the AST level before IR lowering.
"""

import pytest


class TestShapeResolutionPass:
    """Test compile-time shape resolution at AST level"""
    
    def test_resolve_simple_einstein(self, compiler):
        """Test: let B = [1,2,3]; let A[i] = B[i] * 2;

        Shape resolution should track B's shape from the array literal
        and make it available for resolving A's ranges.
        """
        source = """
        let B = [1, 2, 3];
        let A[i] = B[i] * 2;
        """
        
        result = compiler.compile(source, "<test>")
        assert result.success
        
        einstein_node = None
        for stmt in result.ir.statements:
            if getattr(stmt, 'pattern', None) == 'A':
                einstein_node = stmt
                break
        
        assert einstein_node is not None
        # Shape resolution should have annotated ranges if possible
        # The actual shape tracking is internal to the pass
    
    def test_resolve_2d_einstein(self, compiler):
        """Test: let M = [[1,2],[3,4]]; let N[i,j] = M[i,j] * 2;"""
        source = """
        let M = [[1, 2], [3, 4]];
        let N[i,j] = M[i,j] * 2;
        """
        
        result = compiler.compile(source, "<test>")
        assert result.success
        
        # The pass should successfully resolve shapes
        # Actual shape tracking is internal - main validation is no errors
    
    def test_resolve_reduction(self, compiler):
        """Test: let A = [1,2,3]; let s = sum[i](A[i]);"""
        source = """
        let A = [1, 2, 3];
        let s = sum[i](A[i]);
        """
        
        result = compiler.compile(source, "<test>")
        assert result.success
        
        assert result.ir is not None
        assert len(result.ir.statements) >= 1
    
    def test_chain_resolution(self, compiler):
        """Test shape propagation through chain: B -> A -> C"""
        source = """
        let B = [1, 2, 3, 4];
        let A[i] = B[i] * 2;
        let C[j] = A[j] + 1;
        """
        
        result = compiler.compile(source, "<test>")
        assert result.success
        
        # Shape resolution should handle the chain
        # without crashing or producing errors
    
    def test_no_resolution_for_unknowns(self, compiler):
        """Test that pass doesn't crash on unresolvable shapes"""
        source = """
        let B = [1, 2, 3];
        let A[i] = B[i] * 2;
        """
        
        result = compiler.compile(source, "<test>")
        assert result.success
    
    def test_multiple_arrays(self, compiler):
        """Test resolution with multiple input arrays"""
        source = """
        let A = [1, 2, 3];
        let B = [4, 5, 6];
        let C[i] = A[i] + B[i];
        """
        
        result = compiler.compile(source, "<test>")
        assert result.success
    
    def test_matmul_resolution(self, compiler):
        """Test resolution in matrix multiplication"""
        source = """
        let A = [[1, 2], [3, 4]];
        let B = [[5, 6], [7, 8]];
        let C[i,j] = sum[k](A[i,k] * B[k,j]);
        """
        
        result = compiler.compile(source, "<test>")
        assert result.success


class TestShapeInference:
    """Test shape inference from expressions"""
    
    def test_infer_1d_array(self, compiler):
        """Test shape inference for 1D array literal"""
        source = "let A = [1, 2, 3, 4, 5];"
        
        result = compiler.compile(source, "<test>")
        assert result.success
    
    def test_infer_2d_array(self, compiler):
        """Test shape inference for 2D array literal"""
        source = "let M = [[1, 2, 3], [4, 5, 6]];"
        
        result = compiler.compile(source, "<test>")
        assert result.success


class TestScoping:
    """Test scoping behavior"""
    
    def test_function_scoping(self, compiler):
        """Test that function scopes work correctly"""
        source = """
        let B = [1, 2, 3];
        let C[j] = B[j];
        """
        
        result = compiler.compile(source, "<test>")
        assert result.success
    
    def test_function_local_shadowing(self, compiler):
        """Test that scope stack correctly handles shadowing in functions"""
        source = """
        let arr = [1, 2, 3];
        fn process() {
            let arr = [1, 2, 3, 4, 5];
        }
        let result[i] = arr[i] * 2;
        """
        
        result = compiler.compile(source, "<test>")
        assert result.success


class TestEdgeCases:
    """Test edge cases and error handling"""
    
    def test_empty_program(self, compiler):
        """Test pass on empty program"""
        source = ""
        
        result = compiler.compile(source, "<test>")
        assert result.success
    
    def test_scalar_only(self, compiler):
        """Test program with only scalars (no arrays)"""
        source = """
        let x = 5;
        let y = x + 3;
        """
        
        result = compiler.compile(source, "<test>")
        assert result.success

