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
# Special Ops Operation Tests
# Clustered tests for efficiency - all special_ops ops tested together
# Compile/execute errors are failures, not skips
# ===========================================================================

def test_erf_all_ranks(compiler, runtime):
    """Test erf operation across all supported ranks (0D, 1D, 2D, 3D)"""
    source = """use std::ml;
    # 0D
    let x_scalar = 0.5;
    let erf_0d = std::ml::erf(x_scalar);
    
    # 1D
    let x_1d = [-1.0, 0.0, 1.0, 0.5, -0.5];
    let erf_1d = std::ml::erf(x_1d);
    
    # 2D
    let x_2d = [[-1.0, 0.0, 1.0], [0.5, -0.5, 2.0]];
    let erf_2d = std::ml::erf(x_2d);
    
    # 3D
    let x_3d = [[[-1.0, 0.0], [1.0, 0.5]], [[-0.5, 2.0], [0.0, -1.0]]];
    let erf_3d = std::ml::erf(x_3d);
    """
    result = compile_and_execute(source, compiler, runtime)
    assert result.success, f"Execution failed: {result.errors}"

    # Verify 0D
    x_scalar = 0.5
    if scipy is not None:
        expected_0d = scipy.special.erf(x_scalar)
    else:
        expected_0d = 0.5204998778130465  # erf(0.5)
    actual_0d = result.outputs['erf_0d']
    np.testing.assert_allclose(actual_0d, expected_0d, rtol=1e-5)

    # Verify 1D
    x_1d = np.array([-1.0, 0.0, 1.0, 0.5, -0.5], dtype=np.float32)
    if scipy is not None:
        expected_1d = np.array([scipy.special.erf(val) for val in x_1d], dtype=np.float32)
    else:
        expected_1d = np.array([-0.84270079, 0.0, 0.84270079, 0.52049988, -0.52049988], dtype=np.float32)
    actual_1d = np.array(result.outputs['erf_1d'])
    np.testing.assert_allclose(actual_1d, expected_1d, rtol=1e-5)

    # Verify 2D
    x_2d = np.array([[-1.0, 0.0, 1.0], [0.5, -0.5, 2.0]], dtype=np.float32)
    if scipy is not None:
        expected_2d = np.array([[scipy.special.erf(val) for val in row] for row in x_2d], dtype=np.float32)
    else:
        expected_2d = np.array([[-0.84270079, 0.0, 0.84270079], [0.52049988, -0.52049988, 0.99532227]], dtype=np.float32)
    actual_2d = np.array(result.outputs['erf_2d'])
    np.testing.assert_allclose(actual_2d, expected_2d, rtol=1e-5)

    # Verify 3D
    x_3d = np.array([[[-1.0, 0.0], [1.0, 0.5]], [[-0.5, 2.0], [0.0, -1.0]]], dtype=np.float32)
    if scipy is not None:
        expected_3d = np.array([[[scipy.special.erf(val) for val in row] for row in plane] for plane in x_3d], dtype=np.float32)
    else:
        expected_3d = np.array([[[-0.84270079, 0.0], [0.84270079, 0.52049988]], [[-0.52049988, 0.99532227], [0.0, -0.84270079]]], dtype=np.float32)
    actual_3d = np.array(result.outputs['erf_3d'])
    np.testing.assert_allclose(actual_3d, expected_3d, rtol=1e-5)


def test_is_nan_is_inf_all_ranks(compiler, runtime):
    """Test is_nan and is_inf operations across all supported ranks (0D, 1D, 2D, 3D)"""
    source = """use std::ml;
    # 0D
    let x_scalar_nan = 0.0/0.0;
    let x_scalar_inf = 1.0/0.0;
    let is_nan_0d = std::ml::is_nan(x_scalar_nan);
    let is_inf_0d = std::ml::is_inf(x_scalar_inf);
    
    # 1D
    let x_1d = [0.0, 1.0, 0.0/0.0, 2.0];
    let x_1d_inf = [0.0, 1.0, 1.0/0.0, 2.0];
    let is_nan_1d = std::ml::is_nan(x_1d);
    let is_inf_1d = std::ml::is_inf(x_1d_inf);
    
    # 2D
    let x_2d = [[0.0, 1.0, 0.0/0.0], [2.0, 3.0, 4.0]];
    let x_2d_inf = [[0.0, 1.0, 1.0/0.0], [2.0, 3.0, 4.0]];
    let is_nan_2d = std::ml::is_nan(x_2d);
    let is_inf_2d = std::ml::is_inf(x_2d_inf);
    
    # 3D
    let x_3d = [[[0.0, 1.0], [0.0/0.0, 2.0]], [[3.0, 4.0], [5.0, 6.0]]];
    let x_3d_inf = [[[0.0, 1.0], [1.0/0.0, 2.0]], [[3.0, 4.0], [5.0, 6.0]]];
    let is_nan_3d = std::ml::is_nan(x_3d);
    let is_inf_3d = std::ml::is_inf(x_3d_inf);
    """
    result = compile_and_execute(source, compiler, runtime)
    assert result.success, f"Execution failed: {result.errors}"

    # Verify 0D
    x_scalar_nan = np.nan
    x_scalar_inf = np.inf
    expected_is_nan_0d = np.isnan(x_scalar_nan).astype(np.float32)
    expected_is_inf_0d = np.isinf(x_scalar_inf).astype(np.float32)
    actual_is_nan_0d = np.array(result.outputs['is_nan_0d'])
    actual_is_inf_0d = np.array(result.outputs['is_inf_0d'])
    np.testing.assert_allclose(actual_is_nan_0d, expected_is_nan_0d, rtol=1e-6)
    np.testing.assert_allclose(actual_is_inf_0d, expected_is_inf_0d, rtol=1e-6)

    # Verify 1D
    x_1d = np.array([0.0, 1.0, np.nan, 2.0], dtype=np.float32)
    x_1d_inf = np.array([0.0, 1.0, np.inf, 2.0], dtype=np.float32)
    expected_is_nan_1d = np.isnan(x_1d).astype(np.float32)
    expected_is_inf_1d = np.isinf(x_1d_inf).astype(np.float32)
    actual_is_nan_1d = np.array(result.outputs['is_nan_1d'])
    actual_is_inf_1d = np.array(result.outputs['is_inf_1d'])
    np.testing.assert_allclose(actual_is_nan_1d, expected_is_nan_1d, rtol=1e-6)
    np.testing.assert_allclose(actual_is_inf_1d, expected_is_inf_1d, rtol=1e-6)

    # Verify 2D
    x_2d = np.array([[0.0, 1.0, np.nan], [2.0, 3.0, 4.0]], dtype=np.float32)
    x_2d_inf = np.array([[0.0, 1.0, np.inf], [2.0, 3.0, 4.0]], dtype=np.float32)
    expected_is_nan_2d = np.isnan(x_2d).astype(np.float32)
    expected_is_inf_2d = np.isinf(x_2d_inf).astype(np.float32)
    actual_is_nan_2d = np.array(result.outputs['is_nan_2d'])
    actual_is_inf_2d = np.array(result.outputs['is_inf_2d'])
    np.testing.assert_allclose(actual_is_nan_2d, expected_is_nan_2d, rtol=1e-6)
    np.testing.assert_allclose(actual_is_inf_2d, expected_is_inf_2d, rtol=1e-6)

    # Verify 3D
    x_3d = np.array([[[0.0, 1.0], [np.nan, 2.0]], [[3.0, 4.0], [5.0, 6.0]]], dtype=np.float32)
    x_3d_inf = np.array([[[0.0, 1.0], [np.inf, 2.0]], [[3.0, 4.0], [5.0, 6.0]]], dtype=np.float32)
    expected_is_nan_3d = np.isnan(x_3d).astype(np.float32)
    expected_is_inf_3d = np.isinf(x_3d_inf).astype(np.float32)
    actual_is_nan_3d = np.array(result.outputs['is_nan_3d'])
    actual_is_inf_3d = np.array(result.outputs['is_inf_3d'])
    np.testing.assert_allclose(actual_is_nan_3d, expected_is_nan_3d, rtol=1e-6)
    np.testing.assert_allclose(actual_is_inf_3d, expected_is_inf_3d, rtol=1e-6)