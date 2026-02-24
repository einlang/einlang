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
from ...test_utils import compile_and_execute, assert_float_close
# Use conftest's compiler/runtime
# ===========================================================================
# Activation Function Tests
# ===========================================================================



# ===========================================================================
# Ml Ex Operation Tests
# Clustered tests for efficiency - all ml_ex ops tested together
# Compile/execute errors are failures, not skips
# ===========================================================================

def test_ml_ex_clustered_accuracy(compiler, runtime):
    """Test ml_ex operations - clustered for efficiency"""
    source = """use std::ml;
    let x = [[[[1.0, 2.0], [3.0, 4.0]]]];
    let scale = 2.0;
    let bias = [0.5];
    let x_1 = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]];
    let x_2 = [1.0, 2.0, 3.0];
    let x_3 = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]];
    let a = [1.0, 2.0, 3.0];
    let b = [4.0, 5.0];
    let a_1 = [[1.0, 2.0], [3.0, 4.0]];
    let b_1 = [[0.0, 5.0], [6.0, 7.0]];
    let x_4 = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]];
    let x_5 = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]];
    let x_6 = [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]];
    let x_7 = [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]];
    let x_8 = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]];
    let x_9 = [[1.0, 2.0], [3.0, 4.0]];
    let a_cosine = [[1.0, 2.0, 3.0]];
    let b_cosine = [[4.0, 5.0, 6.0]];
    let result_0 = std::ml::ml_ex::image_scaler(x, scale, bias);
    let result_cosine = std::ml::cosine_similarity(a_cosine, b_cosine);
    let result_1 = std::ml::ml_ex::eye(3);
    let result_2 = std::ml::ml_ex::diag_extract(x_1);
    let result_3 = std::ml::ml_ex::diag_construct(x_2);
    let result_5 = std::ml::ml_ex::outer(a, b);
    let result_6 = std::ml::ml_ex::kron(a_1, b_1);
    let result_7 = std::ml::ml_ex::tril(x_3, 0);
    let result_8 = std::ml::ml_ex::triu(x_3, 0);
    let result_9 = std::ml::ml_ex::roll(x_6, 2);
    let result_10 = std::ml::ml_ex::flip(x_7);
    let result_11 = std::ml::ml_ex::trace(x_8);
    let result_12 = std::ml::ml_ex::frobenius_norm(x_9);
    """

    result = compile_and_execute(source, compiler, runtime)
    assert result.success, f"Execution failed: {result.errors}"

    # Verify each operation


    # Verify ml_ex - Test ImageScaler
    x = np.array([[[[1.0, 2.0], [3.0, 4.0]]]], dtype=np.float32)  # Using x from source
    scale = 2.0  # Using scale from source
    bias = np.array([0.5], dtype=np.float32)  # Using bias from source
    expected = x * scale + bias.reshape(1, -1, 1, 1)
    actual = np.array(result.outputs['result_0'])
    np.testing.assert_allclose(actual, expected, rtol=1e-6)


    # Verify ml_ex - Test eye (identity matrix) operation
    assert result.success, f"Execution failed: {result.errors}"
    expected = np.eye(3, dtype=np.float32)
    actual = np.array(result.outputs['result_1'])
    np.testing.assert_allclose(actual, expected, rtol=1e-6)


    # Verify ml_ex - Test diag_extract operation
    x_1 = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], dtype=np.float32)  # Using x_1 from source
    expected = np.diag(x_1)  # [1.0, 5.0, 9.0]
    actual = np.array(result.outputs['result_2'])
    np.testing.assert_allclose(actual, expected, rtol=1e-6)


    # Verify ml_ex - Test diag_construct operation
    x_2 = np.array([1.0, 2.0, 3.0], dtype=np.float32)  # Using x_2 from source
    expected = np.diag(x_2)  # [[1, 0, 0], [0, 2, 0], [0, 0, 3]]
    actual = np.array(result.outputs['result_3'])
    np.testing.assert_allclose(actual, expected, rtol=1e-6)


    # Verify ml_ex - Test outer product operation
    assert result.success, f"Execution failed: {result.errors}"
    a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    b = np.array([4.0, 5.0], dtype=np.float32)
    expected = np.outer(a, b)  # [[4, 5], [8, 10], [12, 15]]
    actual = np.array(result.outputs['result_5'])
    np.testing.assert_allclose(actual, expected, rtol=1e-6)


    # Verify ml_ex - Test Kronecker product operation
    a_1 = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)  # Using a_1 and b_1 from source
    b_1 = np.array([[0.0, 5.0], [6.0, 7.0]], dtype=np.float32)
    expected = np.kron(a_1, b_1)
    actual = np.array(result.outputs['result_6'])
    np.testing.assert_allclose(actual, expected, rtol=1e-6)


    # Verify ml_ex - Test lower triangular extraction
    x_3 = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], dtype=np.float32)  # Using x_3 from source
    expected = np.tril(x_3, k=0)
    actual = np.array(result.outputs['result_7'])
    np.testing.assert_allclose(actual, expected, rtol=1e-6)


    # Verify ml_ex - Test upper triangular extraction
    x_3 = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], dtype=np.float32)  # Using x_3 from source
    expected = np.triu(x_3, k=0)
    actual = np.array(result.outputs['result_8'])
    np.testing.assert_allclose(actual, expected, rtol=1e-6)


    # Verify ml_ex - Test roll operation
    x_6 = np.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]], dtype=np.float32)  # Using x_6 from source
    expected = np.roll(x_6, 2, axis=1)
    actual = np.array(result.outputs['result_9'])
    np.testing.assert_allclose(actual, expected, rtol=1e-6)


    # Verify ml_ex - Test flip operation
    x_7 = np.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]], dtype=np.float32)  # Using x_7 from source
    expected = np.flip(x_7, axis=1)
    actual = np.array(result.outputs['result_10'])
    np.testing.assert_allclose(actual, expected, rtol=1e-6)


    # Verify ml_ex - Test trace operation
    x_8 = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], dtype=np.float32)  # Using x_8 from source
    expected = np.trace(x_8)  # Sum of diagonal: 1.0 + 5.0 + 9.0 = 15.0
    actual = np.array(result.outputs['result_11'])
    np.testing.assert_allclose(actual, expected, rtol=1e-6)


    # Verify ml_ex - Test frobenius_norm operation
    x_9 = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)  # Using x_9 from source
    expected = np.linalg.norm(x_9, 'fro')  # sqrt(1^2 + 2^2 + 3^2 + 4^2) = sqrt(30)
    actual = np.array(result.outputs['result_12'])
    np.testing.assert_allclose(actual, expected, rtol=1e-6)


    # Verify cosine_similarity - Test Cosine Similarity (moved from test_ml_utility.py)
    a_cosine = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)  # Using a_cosine from source
    b_cosine = np.array([[4.0, 5.0, 6.0]], dtype=np.float32)  # Using b_cosine from source
    dot_product = np.sum(a_cosine * b_cosine, axis=-1)
    norm_a = np.linalg.norm(a_cosine, axis=-1)
    norm_b = np.linalg.norm(b_cosine, axis=-1)
    expected = dot_product / (norm_a * norm_b)
    actual = np.array(result.outputs['result_cosine'])
    np.testing.assert_allclose(actual, expected, rtol=1e-6)