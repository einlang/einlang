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


def _assert_successful_outputs(result, expected_arrays=None, expected_scalars=None):
    assert result.success, f"Execution failed: {result.errors}"
    for name, value in (expected_arrays or {}).items():
        assert name in result.outputs
        np.testing.assert_array_equal(result.outputs[name], value)
    for name, value in (expected_scalars or {}).items():
        assert name in result.outputs
        assert result.outputs[name] == value


class TestSimpleEinstein:
    """Test simple Einstein notation through IR path"""

    def test_elementwise_1d_patterns(self, compiler, runtime):
        """Cover the common 1D Einstein elementwise cases in one execution."""
        source = """
        let base = [1, 2, 3, 4, 5];
        let left = [1, 2, 3];
        let right = [4, 5, 6];
        let scaled_input = [10, 20, 30];
        let x = 3;
        let doubled[i] = base[i] * 2;
        let summed[i] = left[i] + right[i];
        let scaled[i] = scaled_input[i] * x;
        """
        result = compile_and_execute(source, compiler, runtime)
        _assert_successful_outputs(
            result,
            expected_arrays={
                "doubled": np.array([2, 4, 6, 8, 10]),
                "summed": np.array([5, 7, 9]),
                "scaled": np.array([30, 60, 90]),
            },
        )


class TestMultiDimensionalEinstein:
    """Test multi-dimensional Einstein notation"""

    def test_multi_dimensional_and_broadcast_patterns(self, compiler, runtime):
        """Cover 2D elementwise, scaling, and broadcast patterns in one execution."""
        source = """
        let A = [[1, 2], [3, 4]];
        let B = [[5, 6], [7, 8]];
        let matrix_sum[i,j] = A[i,j] + B[i,j];
        let scale_src = [[1, 2, 3], [4, 5, 6]];
        let scaled_2d[i,j] = scale_src[i,j] * 2;
        let row = [1, 2, 3];
        let col = [10, 20];
        let broadcast[i,j] = row[i] + col[j];
        """
        result = compile_and_execute(source, compiler, runtime)
        _assert_successful_outputs(
            result,
            expected_arrays={
                "matrix_sum": np.array([[6, 8], [10, 12]]),
                "scaled_2d": np.array([[2, 4, 6], [8, 10, 12]]),
                "broadcast": np.array([[11, 21], [12, 22], [13, 23]]),
            },
        )


class TestReductions:
    """Test reduction operations"""

    def test_scalar_reductions(self, compiler, runtime):
        """Cover the common scalar reduction operators in one execution."""
        source = """
        let sum_values = [1, 2, 3, 4, 5];
        let prod_values = [2, 3, 4];
        let extrema = [5, 2, 9, 1, 7];
        let total = sum[i](sum_values[i]);
        let product = prod[i](prod_values[i]);
        let maximum = max[i](extrema[i]);
        let minimum = min[i](extrema[i]);
        """
        result = compile_and_execute(source, compiler, runtime)
        _assert_successful_outputs(
            result,
            expected_scalars={
                "total": 15,
                "product": 24,
                "maximum": 9,
                "minimum": 1,
            },
        )


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

    def test_einsum_mk_nk_mn_parallel_reduction_dims(self, compiler, runtime):
        """Verify parallel/reduction dims model with large M,N,K so execution must be vectorized (M*N > loop limit)."""
        M, N, K = 64, 64, 128
        source = f"""
        let A[m in 0..{M}, k in 0..{K}] = m * 10 + k;
        let B[n in 0..{N}, k in 0..{K}] = n * 10 + k;
        let C[m in 0..{M}, n in 0..{N}] = sum[k in 0..{K}](A[m, k] * B[n, k]);
        C;
        """
        result = compile_and_execute(source, compiler, runtime)
        assert result.success, f"Execution failed: {result.errors}"
        assert "C" in result.outputs
        C = result.outputs["C"]
        A = (np.arange(M, dtype=np.float32)[:, None] * 10 + np.arange(K, dtype=np.float32)[None, :])
        B = (np.arange(N, dtype=np.float32)[:, None] * 10 + np.arange(K, dtype=np.float32)[None, :])
        expected = np.einsum("mk,nk->mn", A, B)
        np.testing.assert_allclose(C, expected, rtol=1e-5, atol=1e-5)


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


class TestMatmulExecution:
    """Matmul clause execution."""

    def test_matmul_3x2_2x2(self, compiler, runtime):
        """C[i,j] = sum[k](A[i,k] * B[k,j]) gives correct result."""
        source = """
        let A = [[1, 2], [3, 4], [5, 6]];
        let B = [[1, 0], [0, 1]];
        let C[i, j] = sum[k](A[i, k] * B[k, j]);
        """
        result = compile_and_execute(source, compiler, runtime)
        assert result.success, f"Execution failed: {result.errors}"
        C = np.array(result.outputs["C"])
        expected = np.array([[1, 2], [3, 4], [5, 6]])
        np.testing.assert_array_almost_equal(C, expected)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
