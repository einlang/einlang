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
# Utility Operation Tests
# Clustered tests for efficiency - all utility ops tested together
# Compile/execute errors are failures, not skips
# ===========================================================================

def test_utility_clustered_accuracy(compiler, runtime):
    """Test utility operations - clustered for efficiency"""
    source = """use std::ml;
    let value = 42.0;
    let x_2 = [[1.0, 2.0, 3.0, 4.0]];
    let x_2_1d = [1.0, 2.0, 3.0, 4.0];
    let x_2_3d = [[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]];
    let x_3 = [[1.0, 2.0, 3.0]];
    let condition = [[1.0, 0.0, 1.0]];
    let x_4 = [[10.0, 20.0, 30.0]];
    let y = [[1.0, 2.0, 3.0]];
    let x_5 = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
    let x_dropout = [[1.0, 2.0, 3.0]];
    let ratio = 0.5;
    let training = 0;
    let x_l2 = [[3.0, 4.0]];
    let epsilon = 1e-7;
    let x_cast = [[1.0, 2.0, 3.0]];
    let to_type = 1;
    let result_1 = std::ml::constant(value);
    let result_6 = std::ml::cumsum(x_2);
    let result_6_1d = std::ml::cumsum(x_2_1d);
    let result_6_3d = std::ml::cumsum(x_2_3d);
    let result_7 = std::ml::identity(x_3);
    let result_8 = std::ml::where(condition, x_4, y);
    let result_9 = std::ml::numel(x_5);
    let result_10 = std::ml::dropout(x_dropout, ratio, training);
    let result_11 = std::ml::l2_normalize(x_l2, epsilon);
    let result_12 = std::ml::cast(x_cast, to_type);
    """

    result = compile_and_execute(source, compiler, runtime)
    assert result.success, f"Execution failed: {result.errors}"

    # Verify each operation

    # Verify constant - Test Constant
    expected = 42.0
    actual = result.outputs['result_1']
    assert_float_close(actual, expected, rel_tol=1e-6)


    # Verify cumsum 1D - Test Cumulative Sum (1D)
    x_2_1d = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    expected_1d = np.cumsum(x_2_1d, axis=0)
    actual_1d = np.array(result.outputs['result_6_1d'])
    np.testing.assert_allclose(actual_1d, expected_1d, rtol=1e-6)
    np.testing.assert_allclose(actual_1d, [1.0, 3.0, 6.0, 10.0], rtol=1e-6)

    # Verify cumsum 2D - Test Cumulative Sum (2D)
    x_2 = np.array([[1.0, 2.0, 3.0, 4.0]])  # Using x_2 from source
    expected = np.cumsum(x_2, axis=-1)
    actual = np.array(result.outputs['result_6'])
    np.testing.assert_allclose(actual, expected, rtol=1e-6)
    np.testing.assert_allclose(actual[0], [1.0, 3.0, 6.0, 10.0], rtol=1e-6)

    # Verify cumsum 3D - Test Cumulative Sum (3D)
    x_2_3d = np.array([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]], dtype=np.float32)
    expected_3d = np.cumsum(x_2_3d, axis=-1)
    actual_3d = np.array(result.outputs['result_6_3d'])
    np.testing.assert_allclose(actual_3d, expected_3d, rtol=1e-6)


    # Verify identity - Test Identity (pass-through)
    x_3 = np.array([[1.0, 2.0, 3.0]])  # Using x_3 from source
    expected = x_3
    actual = np.array(result.outputs['result_7'])
    np.testing.assert_allclose(actual, expected, rtol=1e-6)


    # Verify where - Test Where (conditional select)
    condition = np.array([[1.0, 0.0, 1.0]])  # Using condition from source
    x_4 = np.array([[10.0, 20.0, 30.0]])  # Using x_4 from source
    y = np.array([[1.0, 2.0, 3.0]])  # Using y from source
    expected = np.where(condition, x_4, y)
    actual = np.array(result.outputs['result_8'])
    np.testing.assert_allclose(actual, expected, rtol=1e-6)


    # Verify numel - Test numel operation
    x_5 = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)  # Using x_5 from source
    expected = np.prod(x_5.shape)
    actual = result.outputs['result_9']
    assert_float_close(actual, float(expected), rel_tol=1e-6)


    # Verify dropout - Test dropout operation (inference mode)
    x_dropout = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)  # Using x_dropout from source
    training = 0  # Using training from source (inference mode)
    expected = x_dropout  # In inference mode, dropout returns input unchanged
    actual = np.array(result.outputs['result_10'])
    np.testing.assert_allclose(actual, expected, rtol=1e-6)


    # Verify l2_normalize - Test l2_normalize operation
    x_l2 = np.array([[3.0, 4.0]], dtype=np.float32)  # Using x_l2 from source
    epsilon = 1e-7  # Using epsilon from source
    norm = np.sqrt(np.sum(x_l2**2, axis=-1, keepdims=True) + epsilon)
    expected = x_l2 / norm
    actual = np.array(result.outputs['result_11'])
    np.testing.assert_allclose(actual, expected, rtol=1e-6)


    # Verify cast - Test cast operation (to INT32)
    x_cast = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)  # Using x_cast from source
    to_type = 1  # Using to_type from source (INT32)
    expected = x_cast.astype(np.int32).astype(np.float32)  # Cast to int32 then back to float32
    actual = np.array(result.outputs['result_12'])
    np.testing.assert_allclose(actual, expected, rtol=1e-6)


def test_size_all_ranks(compiler, runtime):
    """Test size operation across all supported ranks (0D, 1D, 2D, 3D)"""
    source = """use std::ml;
    // 0D (scalar)
    let x_0d = 42.0;
    let size_0d = std::ml::size(x_0d);
    
    // 1D
    let x_1d = [1.0, 2.0, 3.0, 4.0];
    let size_1d = std::ml::size(x_1d);
    
    // 2D
    let x_2d = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
    let size_2d = std::ml::size(x_2d);
    
    // 3D
    let x_3d = [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]];
    let size_3d = std::ml::size(x_3d);
    """
    result = compile_and_execute(source, compiler, runtime)
    assert result.success, f"Execution failed: {result.errors}"
    
    # Verify 0D
    x_0d = np.array(42.0, dtype=np.float32)
    expected_0d = x_0d.size
    np.testing.assert_allclose(result.outputs['size_0d'], expected_0d, rtol=1e-6)
    
    # Verify 1D
    x_1d = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    expected_1d = x_1d.size
    np.testing.assert_allclose(result.outputs['size_1d'], expected_1d, rtol=1e-6)
    
    # Verify 2D
    x_2d = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    expected_2d = x_2d.size
    np.testing.assert_allclose(result.outputs['size_2d'], expected_2d, rtol=1e-6)
    
    # Verify 3D
    x_3d = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], dtype=np.float32)
    expected_3d = x_3d.size
    np.testing.assert_allclose(result.outputs['size_3d'], expected_3d, rtol=1e-6)
    