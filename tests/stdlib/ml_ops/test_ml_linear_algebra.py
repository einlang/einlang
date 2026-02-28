#!/usr/bin/env python3
"""
Comprehensive accuracy tests for std::ml operations against ONNX/NumPy reference implementations.
Tests all operations added to ml.ein for correctness.
"""

import pytest
import numpy as np
try:
    import scipy.special
except ImportError:
    scipy = None
from tests.test_utils import compile_and_execute, assert_float_close
# ===========================================================================
# Activation Function Tests
# ===========================================================================



# ===========================================================================
# Linear Algebra Operation Tests
# Clustered tests for efficiency - all linear_algebra ops tested together
# Compile/execute errors are failures, not skips
# ===========================================================================

def test_linear_algebra_clustered_accuracy(compiler, runtime):
    """Test linear_algebra operations - clustered for efficiency"""
    source = """use std::ml;
    let a = [[[1.0, 2.0], [3.0, 4.0]]];
    let b = [[[5.0, 6.0], [7.0, 8.0]]];
    let x = [[1.0, 2.0]];
    let weights = [[1.0, 2.0], [3.0, 4.0]];
    let bias = [0.5, 1.0];
    let a_1 = [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]];
    let b_1 = [[[1.0, 0.0], [0.0, 1.0]], [[1.0, 1.0], [1.0, 1.0]]];
    let result_0 = std::ml::matmul(a, b);
    let result_1 = std::ml::linear(x, weights, bias);
    let result_2 = std::ml::batch_matmul(a_1, b_1);
    """

    result = compile_and_execute(source, compiler, runtime)
    assert result.success, f"Execution failed: {result.errors}"

    # Verify each operation


    # Verify matmul - Test MatMul
    a = np.array([[[1.0, 2.0], [3.0, 4.0]]])
    b = np.array([[[5.0, 6.0], [7.0, 8.0]]])
    expected = a @ b
    actual = np.array(result.outputs['result_0'])
    np.testing.assert_allclose(actual, expected, rtol=1e-5)


    # Verify linear - Test Linear layer
    x = np.array([[1.0, 2.0]])
    weights = np.array([[1.0, 2.0], [3.0, 4.0]])
    bias = np.array([0.5, 1.0])
    expected = x @ weights.T + bias
    actual = np.array(result.outputs['result_1'])
    np.testing.assert_allclose(actual, expected, rtol=1e-5)


    # Verify batch_matmul - Test batch_matmul operation
    assert result.success, f"Execution failed: {result.errors}"
    a = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], dtype=np.float32)
    b = np.array([[[1.0, 0.0], [0.0, 1.0]], [[1.0, 1.0], [1.0, 1.0]]], dtype=np.float32)
    expected = np.matmul(a, b)
    actual = np.array(result.outputs['result_2'])
    np.testing.assert_allclose(actual, expected, rtol=1e-6)


def test_gemm_all_ranks(compiler, runtime):
    """Test gemm (General Matrix Multiply) operation"""
    source = """use std::ml;
    // Basic gemm: alpha * A @ B + beta * C
    let A = [[1.0, 2.0], [3.0, 4.0]];
    let B = [[5.0, 6.0], [7.0, 8.0]];
    let C = [[1.0, 1.0], [1.0, 1.0]];
    let result1 = std::ml::gemm(A, B, C, 1.0, 1.0, 0, 0);
    
    // With transposition
    let result2 = std::ml::gemm(A, B, C, 1.0, 0.0, 1, 0);
    let result3 = std::ml::gemm(A, B, C, 2.0, 1.0, 0, 1);
    """
    result = compile_and_execute(source, compiler, runtime)
    assert result.success, f"Execution failed: {result.errors}"
    
    # Verify basic gemm: alpha * A @ B + beta * C
    A = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    B = np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32)
    C = np.array([[1.0, 1.0], [1.0, 1.0]], dtype=np.float32)
    expected1 = 1.0 * (A @ B) + 1.0 * C
    actual1 = np.array(result.outputs['result1'])
    np.testing.assert_allclose(actual1, expected1, rtol=1e-6)
    
    # Verify with transA=1: alpha * A.T @ B + beta * C
    expected2 = 1.0 * (A.T @ B) + 0.0 * C
    actual2 = np.array(result.outputs['result2'])
    np.testing.assert_allclose(actual2, expected2, rtol=1e-6)
    
    # Verify with transB=1: alpha * A @ B.T + beta * C
    expected3 = 2.0 * (A @ B.T) + 1.0 * C
    actual3 = np.array(result.outputs['result3'])
    np.testing.assert_allclose(actual3, expected3, rtol=1e-6)