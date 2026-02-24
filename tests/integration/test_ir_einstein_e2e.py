"""
End-to-End Integration Tests for Einstein Notation in IR Path
==============================================================

Tests the complete Einstein notation execution pipeline via IR:
Source → Parser → AST → IR Lowering → IR Backend → Results

This ensures Einstein notation (Einlang's core feature!) works correctly
through the IR execution path.
"""

import pytest
import numpy as np
from tests.test_utils import compile_and_execute


class TestSimpleEinstein:
    """Test simple Einstein notation through IR path"""
    
    def test_simple_elementwise(self, compiler, runtime):
        """Test let A[i] = B[i] * 2"""
        source = """
        let B = [1, 2, 3, 4, 5];
        let A[i] = B[i] * 2;
        """
        result = compile_and_execute(source, compiler, runtime)
        assert result.success, f"Execution failed: {result.errors}"
        assert 'A' in result.outputs
        expected = np.array([2, 4, 6, 8, 10])
        np.testing.assert_array_equal(result.outputs['A'], expected)
    
    def test_elementwise_addition(self, compiler, runtime):
        """Test let C[i] = A[i] + B[i]"""
        source = """
        let A = [1, 2, 3];
        let B = [4, 5, 6];
        let C[i] = A[i] + B[i];
        """
        result = compile_and_execute(source, compiler, runtime)
        assert result.success, f"Execution failed: {result.errors}"
        assert 'C' in result.outputs
        expected = np.array([5, 7, 9])
        np.testing.assert_array_equal(result.outputs['C'], expected)
    
    def test_scalar_multiplication(self, compiler, runtime):
        """Test A[i] = B[i] * scalar"""
        source = """
        let B = [10, 20, 30];
        let x = 3;
        let A[i] = B[i] * x;
        """
        result = compile_and_execute(source, compiler, runtime)
        assert result.success, f"Execution failed: {result.errors}"
        assert 'A' in result.outputs
        expected = np.array([30, 60, 90])
        np.testing.assert_array_equal(result.outputs['A'], expected)


class TestMultiDimensionalEinstein:
    """Test multi-dimensional Einstein notation"""
    
    def test_2d_elementwise(self, compiler, runtime):
        """Test let C[i,j] = A[i,j] + B[i,j]"""
        source = """
        let A = [[1, 2], [3, 4]];
        let B = [[5, 6], [7, 8]];
        let C[i,j] = A[i,j] + B[i,j];
        """
        
        result = compile_and_execute(source, compiler, runtime)
        assert result.success, f"Execution failed: {result.errors}"
        assert 'C' in result.outputs
        expected = np.array([[6, 8], [10, 12]])
        np.testing.assert_array_equal(result.outputs['C'], expected)
    
    def test_2d_scaling(self, compiler, runtime):
        """Test let B[i,j] = A[i,j] * 2"""
        source = """
        let A = [[1, 2, 3], [4, 5, 6]];
        let B[i,j] = A[i,j] * 2;
        """
        
        result = compile_and_execute(source, compiler, runtime)
        assert result.success, f"Execution failed: {result.errors}"
        assert 'B' in result.outputs
        expected = np.array([[2, 4, 6], [8, 10, 12]])
        np.testing.assert_array_equal(result.outputs['B'], expected)


class TestBroadcasting:
    """Test broadcasting patterns"""
    
    def test_row_column_broadcast(self, compiler, runtime):
        """Test let C[i,j] = A[i] + B[j]"""
        source = """
        let A = [1, 2, 3];
        let B = [10, 20];
        let C[i,j] = A[i] + B[j];
        """
        
        result = compile_and_execute(source, compiler, runtime)
        assert result.success, f"Execution failed: {result.errors}"
        assert 'C' in result.outputs
        # C[0,0] = A[0] + B[0] = 1 + 10 = 11
        # C[0,1] = A[0] + B[1] = 1 + 20 = 21
        # C[1,0] = A[1] + B[0] = 2 + 10 = 12
        # etc.
        expected = np.array([[11, 21], [12, 22], [13, 23]])
        np.testing.assert_array_equal(result.outputs['C'], expected)


class TestReductions:
    """Test reduction operations"""
    
    def test_sum_reduction(self, compiler, runtime):
        """Test sum[i](A[i])"""
        source = """
        let A = [1, 2, 3, 4, 5];
        let total = sum[i](A[i]);
        """
        
        result = compile_and_execute(source, compiler, runtime)
        assert result.success, f"Execution failed: {result.errors}"
        assert 'total' in result.outputs
        assert result.outputs['total'] == 15
    
    def test_product_reduction(self, compiler, runtime):
        """Test prod[i](A[i])"""
        source = """
        let A = [2, 3, 4];
        let product = prod[i](A[i]);
        """
        
        result = compile_and_execute(source, compiler, runtime)
        assert result.success, f"Execution failed: {result.errors}"
        assert 'product' in result.outputs
        assert result.outputs['product'] == 24
    
    def test_max_reduction(self, compiler, runtime):
        """Test max[i](A[i])"""
        source = """
        let A = [5, 2, 9, 1, 7];
        let maximum = max[i](A[i]);
        """
        
        result = compile_and_execute(source, compiler, runtime)
        assert result.success, f"Execution failed: {result.errors}"
        assert 'maximum' in result.outputs
        assert result.outputs['maximum'] == 9
    
    def test_min_reduction(self, compiler, runtime):
        """Test min[i](A[i])"""
        source = """
        let A = [5, 2, 9, 1, 7];
        let minimum = min[i](A[i]);
        """
        
        result = compile_and_execute(source, compiler, runtime)
        assert result.success, f"Execution failed: {result.errors}"
        assert 'minimum' in result.outputs
        assert result.outputs['minimum'] == 1


class TestMatrixMultiplication:
    """Test matrix multiplication via Einstein notation"""
    
    def test_simple_matmul(self, compiler, runtime):
        """Test let C[i,j] = sum[k](A[i,k] * B[k,j])"""
        source = """
        let A = [[1, 2], [3, 4]];
        let B = [[5, 6], [7, 8]];
        let C[i,j] = sum[k](A[i,k] * B[k,j]);
        """
        
        result = compile_and_execute(source, compiler, runtime)
        assert result.success, f"Execution failed: {result.errors}"
        assert 'C' in result.outputs
        # Matrix multiplication: [[1,2], [3,4]] @ [[5,6], [7,8]]
        # C[0,0] = 1*5 + 2*7 = 19
        # C[0,1] = 1*6 + 2*8 = 22
        # C[1,0] = 3*5 + 4*7 = 43
        # C[1,1] = 3*6 + 4*8 = 50
        expected = np.array([[19, 22], [43, 50]])
        np.testing.assert_array_equal(result.outputs['C'], expected)
    
    def test_rectangular_matmul(self, compiler, runtime):
        """Test matrix multiplication with non-square matrices"""
        source = """
        let A = [[1, 2, 3], [4, 5, 6]];
        let B = [[7, 8], [9, 10], [11, 12]];
        let C[i,j] = sum[k](A[i,k] * B[k,j]);
        """
        
        result = compile_and_execute(source, compiler, runtime)
        assert result.success, f"Execution failed: {result.errors}"
        assert 'C' in result.outputs
        # (2x3) @ (3x2) = (2x2)
        # C[0,0] = 1*7 + 2*9 + 3*11 = 58
        # C[0,1] = 1*8 + 2*10 + 3*12 = 64
        # C[1,0] = 4*7 + 5*9 + 6*11 = 139
        # C[1,1] = 4*8 + 5*10 + 6*12 = 154
        expected = np.array([[58, 64], [139, 154]])
        np.testing.assert_array_equal(result.outputs['C'], expected)


class TestConvenienceAPI:
    """Test Einstein notation via convenience API"""
    
    def test_execute_ir_with_einstein(self, compiler, runtime):
        """Test Einstein notation execution"""
        source = """
        let B = [10, 20, 30];
        let A[i] = B[i] / 10;
        """
        result = compile_and_execute(source, compiler, runtime)
        assert result.success, f"Execution failed: {result.errors}"
        assert 'A' in result.outputs
        np.testing.assert_array_equal(result.outputs['A'], np.array([1.0, 2.0, 3.0]))


class TestComplexExpressions:
    """Test Einstein with complex expressions"""
    
    def test_nested_operations(self, compiler, runtime):
        """Test let A[i] = (B[i] + C[i]) * D[i]"""
        source = """
        let B = [1, 2, 3];
        let C = [4, 5, 6];
        let D = [2, 2, 2];
        let A[i] = (B[i] + C[i]) * D[i];
        """
        
        result = compile_and_execute(source, compiler, runtime)
        assert result.success, f"Execution failed: {result.errors}"
        assert 'A' in result.outputs
        # A[0] = (1+4)*2 = 10, A[1] = (2+5)*2 = 14, A[2] = (3+6)*2 = 18
        expected = np.array([10, 14, 18])
        np.testing.assert_array_equal(result.outputs['A'], expected)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

