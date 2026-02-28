"""
End-to-End Integration Tests for Einstein Notation Concatenation
===============================================================

Tests concatenation using multiple Einstein declarations.

NOTE: Currently, shape inference for multiple declarations only considers
the first declaration's range, not the union of all ranges. This is a known
limitation. The tests are marked as expected to fail until this is fixed.
"""

import pytest
import numpy as np
from tests.test_utils import compile_and_execute


class TestEinsteinConcatenation:
    """Test concatenation using multiple Einstein declarations"""
    
    def test_concat_vectors(self, compiler, runtime):
        """Test concatenating two vectors"""
        source = """
        let A = [1, 2, 3];
        let B = [4, 5, 6, 7];
        
        // Concatenate along first dimension
        let concat[i in 0..3] = A[i];
        let concat[i in 3..7] = B[i - 3];
        """
        exec_result = compile_and_execute(source, compiler, runtime)
        assert exec_result.success, f"Execution failed: {exec_result.errors}"
        assert 'concat' in exec_result.outputs, f"Output keys: {list(exec_result.outputs.keys())}"
        
        result = exec_result.outputs['concat']
        print(f"Result: {result}, shape: {result.shape if hasattr(result, 'shape') else 'no shape'}")
        
        expected = np.array([1, 2, 3, 4, 5, 6, 7])
        np.testing.assert_array_equal(result, expected)
    
    def test_concat_matrices_along_rows(self, compiler, runtime):
        """Test concatenating matrices along rows (first dimension)"""
        source = """
        let A = [[1, 2], [3, 4]];
        let B = [[5, 6], [7, 8], [9, 10]];
        
        // Concatenate along first dimension (rows)
        let concat[i in 0..2, j] = A[i, j];
        let concat[i in 2..5, j] = B[i - 2, j];
        """
        exec_result = compile_and_execute(source, compiler, runtime)
        assert exec_result.success, f"Execution failed: {exec_result.errors}"
        assert 'concat' in exec_result.outputs
        
        expected = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]])
        np.testing.assert_array_equal(exec_result.outputs['concat'], expected)
        assert exec_result.outputs['concat'].shape == (5, 2)
    
    def test_concat_matrices_along_columns(self, compiler, runtime):
        """Test concatenating matrices along columns (second dimension)"""
        source = """
        let A = [[1, 2], [3, 4]];
        let B = [[5, 6, 7], [8, 9, 10]];
        
        // Concatenate along second dimension (columns)
        let concat[i in 0..2, j in 0..2] = A[i, j];
        let concat[i in 0..2, j in 2..5] = B[i, j - 2];
        """
        exec_result = compile_and_execute(source, compiler, runtime)
        assert exec_result.success, f"Execution failed: {exec_result.errors}"
        assert 'concat' in exec_result.outputs
        
        expected = np.array([[1, 2, 5, 6, 7], [3, 4, 8, 9, 10]])
        np.testing.assert_array_equal(exec_result.outputs['concat'], expected)
        assert exec_result.outputs['concat'].shape == (2, 5)
    
    def test_concat_3d_tensors_along_channels(self, compiler, runtime):
        """Test concatenating 3D tensors along channel dimension"""
        source = """
        // Create two 3D tensors: (batch=2, height=2, width=2, channels)
        let A = [[[1, 2], [3, 4]], [[5, 6], [7, 8]]];  // Shape: (2, 2, 2) - single channel
        let B = [[[10, 20], [30, 40]], [[50, 60], [70, 80]]];  // Shape: (2, 2, 2) - single channel
        
        // For 3D case, we'll treat as (batch, spatial, channel) where spatial is flattened
        // Actually, let's do a simpler 2D case first: concatenate along last dimension
        let A_2d = [[1, 2], [3, 4]];  // Shape: (2, 2)
        let B_2d = [[5, 6], [7, 8]];  // Shape: (2, 2)
        
        // Stack along new dimension (add batch dimension)
        // Use i in 0..2 to cover both i=0 and i=1
        let stacked[i in 0..2, j in 0..2, k in 0..2] = A_2d[j, k] where i == 0;
        let stacked[i in 0..2, j in 0..2, k in 0..2] = B_2d[j, k] where i == 1;
        """
        exec_result = compile_and_execute(source, compiler, runtime)
        assert exec_result.success, f"Execution failed: {exec_result.errors}"
        assert 'stacked' in exec_result.outputs
        
        expected = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
        np.testing.assert_array_equal(exec_result.outputs['stacked'], expected)
        assert exec_result.outputs['stacked'].shape == (2, 2, 2)

