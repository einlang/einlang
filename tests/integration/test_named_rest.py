"""
End-to-End Integration Tests for Named Rest Patterns
=====================================================

Tests runtime execution with actual assertions of results.
Only tests that actually execute code and verify results are included.
"""

import pytest
import numpy as np
from tests.test_utils import compile_and_execute


class TestNamedRestPatternRuntime:
    """Test runtime execution of named rest patterns"""
    
    def test_simple_einstein_without_rest(self, compiler, runtime):
        """Test that simple Einstein notation still works (baseline)"""
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
    
    
    def test_matrix_multiplication_runtime(self, compiler, runtime):
        """Test matrix multiplication with runtime assertions"""
        source = """
        let A = [[1, 2], [3, 4]];
        let B = [[5, 6], [7, 8]];
        let C[i, j] = sum[k](A[i, k] * B[k, j]);
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
        
        # Additional assertions
        assert result.outputs['C'].shape == (2, 2)
        # Dtype can be int32, int64, float32, or float64 depending on backend
        assert result.outputs['C'].dtype in [np.int32, np.int64, np.float32, np.float64]
    
    def test_broadcasting_runtime(self, compiler, runtime):
        """Test broadcasting with runtime assertions"""
        source = """
        let A = [1, 2, 3];
        let B = [10, 20];
        let C[i, j] = A[i] + B[j];
        """
        
        result = compile_and_execute(source, compiler, runtime)
        assert result.success, f"Execution failed: {result.errors}"
        assert 'C' in result.outputs
        
        # Broadcasting: C[i,j] = A[i] + B[j]
        # C[0,0] = 1 + 10 = 11
        # C[0,1] = 1 + 20 = 21
        # C[1,0] = 2 + 10 = 12
        # C[1,1] = 2 + 20 = 22
        # C[2,0] = 3 + 10 = 13
        # C[2,1] = 3 + 20 = 23
        expected = np.array([[11, 21], [12, 22], [13, 23]])
        np.testing.assert_array_equal(result.outputs['C'], expected)
        
        # Shape assertions
        assert result.outputs['C'].shape == (3, 2)
    
    def test_rest_pattern_simple_execution(self, compiler, runtime):
        """Test execution with rest patterns - simple case"""
        source = """
        let x = [[1.0, 2.0], [3.0, 4.0]];  // 2x2 matrix
        let result[..batch, j] = x[..batch, j] * 2.0;
        """
        
        result = compile_and_execute(source, compiler, runtime)
        assert result.success, f"Execution failed: {result.errors}"
        assert 'result' in result.outputs
        
        # Expected: result[..batch, j] = x[..batch, j] * 2
        # result[0, 0] = 1.0 * 2 = 2.0
        # result[0, 1] = 2.0 * 2 = 4.0
        # result[1, 0] = 3.0 * 2 = 6.0
        # result[1, 1] = 4.0 * 2 = 8.0
        expected = np.array([[2.0, 4.0], [6.0, 8.0]])
        np.testing.assert_array_equal(result.outputs['result'], expected)
        assert result.outputs['result'].shape == (2, 2)
    
    def test_rest_pattern_with_reduction(self, compiler, runtime):
        """Test execution with rest patterns in reduction"""
        source = """
        let x = [[1.0, 2.0], [3.0, 4.0]];  // 2x2 matrix
        let result[..batch] = sum[j](x[..batch, j]);
        """
        
        result = compile_and_execute(source, compiler, runtime)
        assert result.success, f"Execution failed: {result.errors}"
        assert 'result' in result.outputs
        
        # Expected: result[..batch] = sum[j](x[..batch, j])
        # result[0] = sum[j](x[0, j]) = 1.0 + 2.0 = 3.0
        # result[1] = sum[j](x[1, j]) = 3.0 + 4.0 = 7.0
        expected = np.array([3.0, 7.0])
        np.testing.assert_array_equal(result.outputs['result'], expected)
        assert result.outputs['result'].shape == (2,)
    
    def test_rest_pattern_matrix_multiply(self, compiler, runtime):
        """Test execution with rest patterns in matrix multiplication"""
        source = """
        let x = [[1.0, 2.0], [3.0, 4.0]];  // 2x2 matrix
        let w = [[0.1, 0.2], [0.3, 0.4]];  // 2x2 weight matrix
        let result[..batch, j] = sum[k](x[..batch, k] * w[k, j]);
        """
        
        result = compile_and_execute(source, compiler, runtime)
        assert result.success, f"Execution failed: {result.errors}"
        assert 'result' in result.outputs
        
        # Expected: result[..batch, j] = sum[k](x[..batch, k] * w[k, j])
        # result[0, 0] = x[0, 0]*w[0, 0] + x[0, 1]*w[1, 0] = 1.0*0.1 + 2.0*0.3 = 0.1 + 0.6 = 0.7
        # result[0, 1] = x[0, 0]*w[0, 1] + x[0, 1]*w[1, 1] = 1.0*0.2 + 2.0*0.4 = 0.2 + 0.8 = 1.0
        # result[1, 0] = x[1, 0]*w[0, 0] + x[1, 1]*w[1, 0] = 3.0*0.1 + 4.0*0.3 = 0.3 + 1.2 = 1.5
        # result[1, 1] = x[1, 0]*w[0, 1] + x[1, 1]*w[1, 1] = 3.0*0.2 + 4.0*0.4 = 0.6 + 1.6 = 2.2
        expected = np.array([[0.7, 1.0], [1.5, 2.2]])
        np.testing.assert_array_almost_equal(result.outputs['result'], expected, decimal=5)
        assert result.outputs['result'].shape == (2, 2)
    
    def test_rest_pattern_high_rank_4d(self, compiler, runtime):
        """Test high-rank 4D tensor with 3D rest pattern"""
        source = """
        let x = [[[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], 
                 [[[9.0, 10.0], [11.0, 12.0]], [[13.0, 14.0], [15.0, 16.0]]]];  // Shape: (2, 2, 2, 2)
        let result[..spatial, c] = x[..spatial, c] * 2.0;
        """
        
        result = compile_and_execute(source, compiler, runtime)
        assert result.success, f"Execution failed: {result.errors}"
        assert 'result' in result.outputs
        
        # Expected: result[..spatial, c] = x[..spatial, c] * 2.0
        # All elements should be doubled
        expected = np.array([[[[2.0, 4.0], [6.0, 8.0]], [[10.0, 12.0], [14.0, 16.0]]], 
                           [[[18.0, 20.0], [22.0, 24.0]], [[26.0, 28.0], [30.0, 32.0]]]])
        np.testing.assert_array_equal(result.outputs['result'], expected)
        assert result.outputs['result'].shape == (2, 2, 2, 2)
    
    def test_rest_pattern_3d_tensor(self, compiler, runtime):
        """Test 3D tensor with 2D rest pattern"""
        source = """
        let x = [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]];  // Shape: (2, 2, 2)
        let result[..batch, c] = x[..batch, c] * 3.0;
        """
        
        result = compile_and_execute(source, compiler, runtime)
        assert result.success, f"Execution failed: {result.errors}"
        assert 'result' in result.outputs
        
        # Expected: result[..batch, c] = x[..batch, c] * 3.0
        expected = np.array([[[3.0, 6.0], [9.0, 12.0]], [[15.0, 18.0], [21.0, 24.0]]])
        np.testing.assert_array_equal(result.outputs['result'], expected)
        assert result.outputs['result'].shape == (2, 2, 2)
    
    def test_rest_pattern_reduction_high_rank(self, compiler, runtime):
        """Test reduction with rest pattern on high-rank tensor"""
        source = """
        let x = [[[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], 
                 [[[9.0, 10.0], [11.0, 12.0]], [[13.0, 14.0], [15.0, 16.0]]]];  // Shape: (2, 2, 2, 2)
        let result[..batch] = sum[c](x[..batch, c]);
        """
        
        result = compile_and_execute(source, compiler, runtime)
        assert result.success, f"Execution failed: {result.errors}"
        assert 'result' in result.outputs
        
        # Expected: result[..batch] = sum[c](x[..batch, c])
        # Sum over the last dimension (c)
        # result[0, 0, 0] = sum(c)(x[0, 0, 0, c]) = 1.0 + 2.0 = 3.0
        # result[0, 0, 1] = sum(c)(x[0, 0, 1, c]) = 3.0 + 4.0 = 7.0
        # result[0, 1, 0] = sum(c)(x[0, 1, 0, c]) = 5.0 + 6.0 = 11.0
        # result[0, 1, 1] = sum(c)(x[0, 1, 1, c]) = 7.0 + 8.0 = 15.0
        # result[1, 0, 0] = sum(c)(x[1, 0, 0, c]) = 9.0 + 10.0 = 19.0
        # result[1, 0, 1] = sum(c)(x[1, 0, 1, c]) = 11.0 + 12.0 = 23.0
        # result[1, 1, 0] = sum(c)(x[1, 1, 0, c]) = 13.0 + 14.0 = 27.0
        # result[1, 1, 1] = sum(c)(x[1, 1, 1, c]) = 15.0 + 16.0 = 31.0
        expected = np.array([[[3.0, 7.0], [11.0, 15.0]], [[19.0, 23.0], [27.0, 31.0]]])
        np.testing.assert_array_equal(result.outputs['result'], expected)
        assert result.outputs['result'].shape == (2, 2, 2)
    
    def test_rest_pattern_broadcasting(self, compiler, runtime):
        """Test broadcasting with rest patterns"""
        source = """
        let x = [[1.0, 2.0], [3.0, 4.0]];  // Shape: (2, 2)
        let bias = [10.0, 20.0];  // Shape: (2,)
        let result[..batch, j] = x[..batch, j] + bias[j];
        """
        
        result = compile_and_execute(source, compiler, runtime)
        assert result.success, f"Execution failed: {result.errors}"
        assert 'result' in result.outputs
        
        # Expected: result[..batch, j] = x[..batch, j] + bias[j]
        # result[0, 0] = 1.0 + 10.0 = 11.0
        # result[0, 1] = 2.0 + 20.0 = 22.0
        # result[1, 0] = 3.0 + 10.0 = 13.0
        # result[1, 1] = 4.0 + 20.0 = 24.0
        expected = np.array([[11.0, 22.0], [13.0, 24.0]])
        np.testing.assert_array_equal(result.outputs['result'], expected)
        assert result.outputs['result'].shape == (2, 2)
    
    def test_rest_pattern_complex_expression(self, compiler, runtime):
        """Test complex expression with rest patterns"""
        source = """
        let x = [[1.0, 2.0], [3.0, 4.0]];  // Shape: (2, 2)
        let y = [[5.0, 6.0], [7.0, 8.0]];  // Shape: (2, 2)
        let result[..batch, j] = (x[..batch, j] + y[..batch, j]) * 2.0;
        """
        
        result = compile_and_execute(source, compiler, runtime)
        assert result.success, f"Execution failed: {result.errors}"
        assert 'result' in result.outputs
        
        # Expected: result[..batch, j] = (x[..batch, j] + y[..batch, j]) * 2.0
        # result[0, 0] = (1.0 + 5.0) * 2.0 = 12.0
        # result[0, 1] = (2.0 + 6.0) * 2.0 = 16.0
        # result[1, 0] = (3.0 + 7.0) * 2.0 = 20.0
        # result[1, 1] = (4.0 + 8.0) * 2.0 = 24.0
        expected = np.array([[12.0, 16.0], [20.0, 24.0]])
        np.testing.assert_array_equal(result.outputs['result'], expected)
        assert result.outputs['result'].shape == (2, 2)
    
    def test_rest_pattern_at_beginning(self, compiler, runtime):
        """Test rest pattern at the beginning of index list"""
        source = """
        let x = [[1.0, 2.0], [3.0, 4.0]];  // Shape: (2, 2)
        let result[..batch, j] = x[..batch, j] * 2.0;
        """
        
        result = compile_and_execute(source, compiler, runtime)
        assert result.success, f"Execution failed: {result.errors}"
        assert 'result' in result.outputs
        
        # Expected: result[..batch, j] = x[..batch, j] * 2.0
        # Rest pattern at beginning spans first dimension
        # result[0, 0] = 1.0 * 2 = 2.0
        # result[0, 1] = 2.0 * 2 = 4.0
        # result[1, 0] = 3.0 * 2 = 6.0
        # result[1, 1] = 4.0 * 2 = 8.0
        expected = np.array([[2.0, 4.0], [6.0, 8.0]])
        np.testing.assert_array_equal(result.outputs['result'], expected)
        assert result.outputs['result'].shape == (2, 2)
    
    def test_rest_pattern_in_middle(self, compiler, runtime):
        """Test rest pattern in the middle between explicit indices"""
        source = """
        let x = [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]];  // Shape: (2, 2, 2)
        let result[i, ..batch, j] = x[i, ..batch, j] * 2.0;
        """
        
        result = compile_and_execute(source, compiler, runtime)
        assert result.success, f"Execution failed: {result.errors}"
        assert 'result' in result.outputs
        
        # Expected: result[i, ..batch, j] = x[i, ..batch, j] * 2.0
        # Rest pattern in middle spans middle dimension
        # All elements should be doubled
        expected = np.array([[[2.0, 4.0], [6.0, 8.0]], [[10.0, 12.0], [14.0, 16.0]]])
        np.testing.assert_array_equal(result.outputs['result'], expected)
        assert result.outputs['result'].shape == (2, 2, 2)
    
    def test_rest_pattern_at_end(self, compiler, runtime):
        """Test rest pattern at the end of index list"""
        source = """
        let x = [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]];  // Shape: (2, 2, 2)
        let result[i, ..batch] = x[i, ..batch] * 2.0;
        """
        
        result = compile_and_execute(source, compiler, runtime)
        assert result.success, f"Execution failed: {result.errors}"
        assert 'result' in result.outputs
        
        # Expected: result[i, ..batch] = x[i, ..batch] * 2.0
        # Rest pattern at end spans remaining dimensions
        # All elements should be doubled
        expected = np.array([[[2.0, 4.0], [6.0, 8.0]], [[10.0, 12.0], [14.0, 16.0]]])
        np.testing.assert_array_equal(result.outputs['result'], expected)
        assert result.outputs['result'].shape == (2, 2, 2)
    
    def test_rest_pattern_at_beginning_high_rank(self, compiler, runtime):
        """Test rest pattern at beginning with high-rank tensor"""
        source = """
        let x = [[[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], 
                 [[[9.0, 10.0], [11.0, 12.0]], [[13.0, 14.0], [15.0, 16.0]]]];  // Shape: (2, 2, 2, 2)
        let result[..spatial, c] = x[..spatial, c] * 2.0;
        """
        
        result = compile_and_execute(source, compiler, runtime)
        assert result.success, f"Execution failed: {result.errors}"
        assert 'result' in result.outputs
        
        # Expected: result[..spatial, c] = x[..spatial, c] * 2.0
        # Rest pattern at beginning spans first 3 dimensions
        expected = np.array([[[[2.0, 4.0], [6.0, 8.0]], [[10.0, 12.0], [14.0, 16.0]]], 
                           [[[18.0, 20.0], [22.0, 24.0]], [[26.0, 28.0], [30.0, 32.0]]]])
        np.testing.assert_array_equal(result.outputs['result'], expected)
        assert result.outputs['result'].shape == (2, 2, 2, 2)
    
    def test_rest_pattern_in_middle_high_rank(self, compiler, runtime):
        """Test rest pattern in middle with high-rank tensor"""
        source = """
        let x = [[[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], 
                 [[[9.0, 10.0], [11.0, 12.0]], [[13.0, 14.0], [15.0, 16.0]]]];  // Shape: (2, 2, 2, 2)
        let result[i, ..spatial, c] = x[i, ..spatial, c] * 2.0;
        """
        
        result = compile_and_execute(source, compiler, runtime)
        assert result.success, f"Execution failed: {result.errors}"
        assert 'result' in result.outputs
        
        # Expected: result[i, ..spatial, c] = x[i, ..spatial, c] * 2.0
        # Rest pattern in middle spans middle 2 dimensions
        expected = np.array([[[[2.0, 4.0], [6.0, 8.0]], [[10.0, 12.0], [14.0, 16.0]]], 
                           [[[18.0, 20.0], [22.0, 24.0]], [[26.0, 28.0], [30.0, 32.0]]]])
        np.testing.assert_array_equal(result.outputs['result'], expected)
        assert result.outputs['result'].shape == (2, 2, 2, 2)
    
    def test_rest_pattern_at_end_high_rank(self, compiler, runtime):
        """Test rest pattern at end with high-rank tensor"""
        source = """
        let x = [[[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], 
                 [[[9.0, 10.0], [11.0, 12.0]], [[13.0, 14.0], [15.0, 16.0]]]];  // Shape: (2, 2, 2, 2)
        let result[i, ..spatial] = x[i, ..spatial] * 2.0;
        """
        
        result = compile_and_execute(source, compiler, runtime)
        assert result.success, f"Execution failed: {result.errors}"
        assert 'result' in result.outputs
        
        # Expected: result[i, ..spatial] = x[i, ..spatial] * 2.0
        # Rest pattern at end spans last 3 dimensions
        expected = np.array([[[[2.0, 4.0], [6.0, 8.0]], [[10.0, 12.0], [14.0, 16.0]]], 
                           [[[18.0, 20.0], [22.0, 24.0]], [[26.0, 28.0], [30.0, 32.0]]]])
        np.testing.assert_array_equal(result.outputs['result'], expected)
        assert result.outputs['result'].shape == (2, 2, 2, 2)
    
    def test_outer_product_simple(self, compiler, runtime):
        """Test simple outer product without rest patterns"""
        source = """
        let a = [1.0, 2.0];  // Shape: (2,)
        let b = [3.0, 4.0];  // Shape: (2,)
        let result[i, j] = a[i] * b[j];
        """
        
        result = compile_and_execute(source, compiler, runtime)
        assert result.success, f"Execution failed: {result.errors}"
        assert 'result' in result.outputs
        
        # Expected: result[i, j] = a[i] * b[j]
        # result[0, 0] = 1.0 * 3.0 = 3.0
        # result[0, 1] = 1.0 * 4.0 = 4.0
        # result[1, 0] = 2.0 * 3.0 = 6.0
        # result[1, 1] = 2.0 * 4.0 = 8.0
        expected = np.array([[3.0, 4.0], [6.0, 8.0]])
        np.testing.assert_array_equal(result.outputs['result'], expected)
        assert result.outputs['result'].shape == (2, 2)
    
    def test_outer_product_with_rest_at_beginning(self, compiler, runtime):
        """Test outer product with rest pattern at beginning"""
        source = """
        let a = [[1.0, 2.0], [3.0, 4.0]];  // Shape: (2, 2)
        let b = [5.0, 6.0];  // Shape: (2,)
        let result[..batch, i, j] = a[..batch, i] * b[j];
        """
        
        result = compile_and_execute(source, compiler, runtime)
        assert result.success, f"Execution failed: {result.errors}"
        assert 'result' in result.outputs
        
        # Expected: result[..batch, i, j] = a[..batch, i] * b[j]
        # Rest pattern spans first dimension, then outer product over i and j
        # result[0, 0, 0] = a[0, 0] * b[0] = 1.0 * 5.0 = 5.0
        # result[0, 0, 1] = a[0, 0] * b[1] = 1.0 * 6.0 = 6.0
        # result[0, 1, 0] = a[0, 1] * b[0] = 2.0 * 5.0 = 10.0
        # result[0, 1, 1] = a[0, 1] * b[1] = 2.0 * 6.0 = 12.0
        # result[1, 0, 0] = a[1, 0] * b[0] = 3.0 * 5.0 = 15.0
        # result[1, 0, 1] = a[1, 0] * b[1] = 3.0 * 6.0 = 18.0
        # result[1, 1, 0] = a[1, 1] * b[0] = 4.0 * 5.0 = 20.0
        # result[1, 1, 1] = a[1, 1] * b[1] = 4.0 * 6.0 = 24.0
        expected = np.array([[[5.0, 6.0], [10.0, 12.0]], [[15.0, 18.0], [20.0, 24.0]]])
        np.testing.assert_array_equal(result.outputs['result'], expected)
        assert result.outputs['result'].shape == (2, 2, 2)
    
    def test_outer_product_with_rest_both_vectors(self, compiler, runtime):
        """Test outer product where both vectors have rest patterns"""
        source = """
        let a = [[1.0, 2.0], [3.0, 4.0]];  // Shape: (2, 2)
        let b = [[5.0, 6.0], [7.0, 8.0]];  // Shape: (2, 2)
        let result[..batch_a, i, ..batch_b, j] = a[..batch_a, i] * b[..batch_b, j];
        """
        
        result = compile_and_execute(source, compiler, runtime)
        assert result.success, f"Execution failed: {result.errors}"
        assert 'result' in result.outputs
        
        # Expected: result[..batch_a, i, ..batch_b, j] = a[..batch_a, i] * b[..batch_b, j]
        # This creates a cartesian product of batch dimensions with outer product
        # Shape should be (2, 2, 2, 2) - cartesian product of batch dimensions
        assert result.outputs['result'].shape == (2, 2, 2, 2)
        
        # Verify a few specific elements
        # result[0, 0, 0, 0] = a[0, 0] * b[0, 0] = 1.0 * 5.0 = 5.0
        # result[0, 0, 0, 1] = a[0, 0] * b[0, 1] = 1.0 * 6.0 = 6.0
        # result[0, 1, 1, 0] = a[0, 1] * b[1, 0] = 2.0 * 7.0 = 14.0
        assert result.outputs['result'][0, 0, 0, 0] == 5.0
        assert result.outputs['result'][0, 0, 0, 1] == 6.0
        assert result.outputs['result'][0, 1, 1, 0] == 14.0
    
    def test_outer_product_high_rank(self, compiler, runtime):
        """Test outer product with high-rank tensors and rest patterns"""
        source = """
        let a = [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]];  // Shape: (2, 2, 2)
        let b = [10.0, 20.0];  // Shape: (2,)
        let result[..spatial, i, j] = a[..spatial, i] * b[j];
        """
        
        result = compile_and_execute(source, compiler, runtime)
        assert result.success, f"Execution failed: {result.errors}"
        assert 'result' in result.outputs
        
        # Expected: result[..spatial, i, j] = a[..spatial, i] * b[j]
        # Rest pattern spans first 2 dimensions, then outer product over i and j
        # Shape should be (2, 2, 2, 2)
        assert result.outputs['result'].shape == (2, 2, 2, 2)
        
        # Verify a few specific elements
        # result[0, 0, 0, 0] = a[0, 0, 0] * b[0] = 1.0 * 10.0 = 10.0
        # result[0, 0, 0, 1] = a[0, 0, 0] * b[1] = 1.0 * 20.0 = 20.0
        # result[0, 0, 1, 0] = a[0, 0, 1] * b[0] = 2.0 * 10.0 = 20.0
        assert result.outputs['result'][0, 0, 0, 0] == 10.0
        assert result.outputs['result'][0, 0, 0, 1] == 20.0
        assert result.outputs['result'][0, 0, 1, 0] == 20.0
    
    def test_outer_product_three_way(self, compiler, runtime):
        """Test three-way outer product with rest patterns"""
        source = """
        let a = [[1.0, 2.0], [3.0, 4.0]];  // Shape: (2, 2)
        let b = [5.0, 6.0];  // Shape: (2,)
        let c = [7.0, 8.0];  // Shape: (2,)
        let result[..batch, i, j, k] = a[..batch, i] * b[j] * c[k];
        """
        
        result = compile_and_execute(source, compiler, runtime)
        assert result.success, f"Execution failed: {result.errors}"
        assert 'result' in result.outputs
        
        # Expected: result[..batch, i, j, k] = a[..batch, i] * b[j] * c[k]
        # Rest pattern spans first dimension, then three-way outer product
        # Shape should be (2, 2, 2, 2)
        assert result.outputs['result'].shape == (2, 2, 2, 2)
        
        # Verify a few specific elements
        # result[0, 0, 0, 0] = a[0, 0] * b[0] * c[0] = 1.0 * 5.0 * 7.0 = 35.0
        # result[0, 0, 0, 1] = a[0, 0] * b[0] * c[1] = 1.0 * 5.0 * 8.0 = 40.0
        # result[0, 1, 1, 0] = a[0, 1] * b[1] * c[0] = 2.0 * 6.0 * 7.0 = 84.0
        assert result.outputs['result'][0, 0, 0, 0] == 35.0
        assert result.outputs['result'][0, 0, 0, 1] == 40.0
        assert result.outputs['result'][0, 1, 1, 0] == 84.0
    
    def test_function_with_rest_pattern(self, compiler, runtime):
        """Test function definition with rest patterns - runtime execution"""
        source = """
        fn scale_tensor(x, factor) {
            let result[..batch, j] = x[..batch, j] * factor;
            result
        }
        
        let input = [[1.0, 2.0], [3.0, 4.0]];  // Shape: (2, 2)
        let scaled = scale_tensor(input, 2.5);
        """
        
        result = compile_and_execute(source, compiler, runtime)
        assert result.success, f"Execution failed: {result.errors}"
        assert 'scaled' in result.outputs
        
        # Expected: scaled[..batch, j] = input[..batch, j] * 2.5
        # scaled[0, 0] = 1.0 * 2.5 = 2.5
        # scaled[0, 1] = 2.0 * 2.5 = 5.0
        # scaled[1, 0] = 3.0 * 2.5 = 7.5
        # scaled[1, 1] = 4.0 * 2.5 = 10.0
        expected = np.array([[2.5, 5.0], [7.5, 10.0]])
        np.testing.assert_array_equal(result.outputs['scaled'], expected)
        assert result.outputs['scaled'].shape == (2, 2)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

