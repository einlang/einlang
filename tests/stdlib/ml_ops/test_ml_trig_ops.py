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
# Trig Ops Operation Tests
# Clustered tests for efficiency - all trig_ops ops tested together
# Compile/execute errors are failures, not skips
# ===========================================================================

def test_trig_ops_clustered_accuracy(compiler, runtime):
    """Test trig_ops operations - clustered for efficiency"""
    source = """use std::ml;
    let x = [[0.0, 1.5708, 3.14159]];
    let x_1 = [[-1.0, 0.0, 1.0]];
    let y = [[1.0, 0.0, -1.0]];
    let x_2 = [[1.0, 1.0, 1.0]];
    let x_3 = [[0.5, 0.707, 0.866]];
    let x_4 = [[-1.0, 0.0, 1.0]];
    let sin_result_0 = std::ml::sin(x);
    let cos_result_0 = std::ml::cos(x);
    let tan_result_0 = std::ml::tan(x);
    let sinh_result_1 = std::ml::sinh(x_1);
    let cosh_result_1 = std::ml::cosh(x_1);
    let tanh_result_1 = std::ml::tanh(x_1);
    let result_2 = std::ml::atan2(y, x_2);
    let result_3 = std::ml::asin(x_3);
    let result_4 = std::ml::acos(x_3);
    let result_5 = std::ml::atan(x_4);
    let result_6 = std::ml::asinh(x_4);
    let result_7 = std::ml::acosh(x_2);
    let result_8 = std::ml::atanh(x_3);
    """

    result = compile_and_execute(source, compiler, runtime)
    assert result.success, f"Execution failed: {result.errors}"

    # Verify each operation

    # Verify sin - Test sin, cos, tan operations
    # Use f64 reference since einlang computes in f64 for untyped float literals
    x = np.array([[0.0, 1.5708, 3.14159]], dtype=np.float64)
    np.testing.assert_allclose(np.array(result.outputs['sin_result_0']), np.sin(x), rtol=1e-4, atol=1e-6)
    np.testing.assert_allclose(np.array(result.outputs['cos_result_0']), np.cos(x), rtol=1e-4, atol=1e-6)
    np.testing.assert_allclose(np.array(result.outputs['tan_result_0']), np.tan(x), rtol=1e-4, atol=1e-6)


    # Verify sinh - Test sinh, cosh, tanh operations
    x_1 = np.array([[-1.0, 0.0, 1.0]], dtype=np.float32)  # Using x_1 from source
    np.testing.assert_allclose(np.array(result.outputs['sinh_result_1']), np.sinh(x_1), rtol=1e-5)
    np.testing.assert_allclose(np.array(result.outputs['cosh_result_1']), np.cosh(x_1), rtol=1e-5)
    np.testing.assert_allclose(np.array(result.outputs['tanh_result_1']), np.tanh(x_1), rtol=1e-5)


    # Verify atan2 - Test atan2 operation
    y = np.array([[1.0, 0.0, -1.0]], dtype=np.float32)  # Using y from source
    x_2 = np.array([[1.0, 1.0, 1.0]], dtype=np.float32)  # Using x_2 from source
    # atan2(y, x) - note: TSC atan2 takes (y, x) in that order
    expected = np.arctan2(y, x_2)
    actual = np.array(result.outputs['result_2'])
    np.testing.assert_allclose(actual, expected, rtol=1e-5)


    # Verify asin - Test asin operation
    x_3 = np.array([[0.5, 0.707, 0.866]], dtype=np.float32)  # Using x_3 from source (values in [-1, 1])
    expected = np.arcsin(x_3)
    actual = np.array(result.outputs['result_3'])
    np.testing.assert_allclose(actual, expected, rtol=1e-5)


    # Verify acos - Test acos operation
    x_3 = np.array([[0.5, 0.707, 0.866]], dtype=np.float32)  # Using x_3 from source (values in [-1, 1])
    expected = np.arccos(x_3)
    actual = np.array(result.outputs['result_4'])
    np.testing.assert_allclose(actual, expected, rtol=1e-5)


    # Verify atan - Test atan operation
    x_4 = np.array([[-1.0, 0.0, 1.0]], dtype=np.float32)  # Using x_4 from source
    expected = np.arctan(x_4)
    actual = np.array(result.outputs['result_5'])
    np.testing.assert_allclose(actual, expected, rtol=1e-5)


    # Verify asinh - Test asinh operation
    x_4 = np.array([[-1.0, 0.0, 1.0]], dtype=np.float32)  # Using x_4 from source
    expected = np.arcsinh(x_4)
    actual = np.array(result.outputs['result_6'])
    np.testing.assert_allclose(actual, expected, rtol=1e-5)


    # Verify acosh - Test acosh operation
    x_2 = np.array([[1.0, 1.0, 1.0]], dtype=np.float32)  # Using x_2 from source (values >= 1)
    expected = np.arccosh(x_2)
    actual = np.array(result.outputs['result_7'])
    np.testing.assert_allclose(actual, expected, rtol=1e-5)


    # Verify atanh - Test atanh operation
    x_3 = np.array([[0.5, 0.707, 0.866]], dtype=np.float32)  # Using x_3 from source (values in (-1, 1))
    expected = np.arctanh(x_3)
    actual = np.array(result.outputs['result_8'])
    np.testing.assert_allclose(actual, expected, rtol=1e-5)


def test_inverse_trig_all_ranks(compiler, runtime):
    """Test inverse trigonometric functions (asin, acos, atan, asinh, acosh, atanh) across all ranks"""
    
    source = """use std::ml;
    // Test 0D (scalar)
    let x_scalar = 0.5;
    let asin_0d = std::ml::asin(x_scalar);
    let acos_0d = std::ml::acos(x_scalar);
    let atan_0d = std::ml::atan(x_scalar);
    let asinh_0d = std::ml::asinh(x_scalar);
    let acosh_0d = std::ml::acosh(1.5);
    let atanh_0d = std::ml::atanh(x_scalar);
    
    // Test 1D
    let x_1d = [0.0, 0.5, -0.5, 0.707];
    let asin_1d = std::ml::asin(x_1d);
    let acos_1d = std::ml::acos(x_1d);
    let atan_1d = std::ml::atan(x_1d);
    let asinh_1d = std::ml::asinh(x_1d);
    let acosh_1d = std::ml::acosh([1.0, 1.5, 2.0, 3.0]);
    let atanh_1d = std::ml::atanh([0.0, 0.5, -0.5, 0.707]);
    
    // Test 2D
    let x_2d = [[0.0, 0.5], [-0.5, 0.707]];
    let asin_2d = std::ml::asin(x_2d);
    let acos_2d = std::ml::acos(x_2d);
    let atan_2d = std::ml::atan(x_2d);
    let asinh_2d = std::ml::asinh(x_2d);
    let acosh_2d = std::ml::acosh([[1.0, 1.5], [2.0, 3.0]]);
    let atanh_2d = std::ml::atanh([[0.0, 0.5], [-0.5, 0.707]]);
    
    // Test 3D
    let x_3d = [[[0.0, 0.5], [-0.5, 0.707]], [[0.3, -0.3], [0.8, -0.8]]];
    let asin_3d = std::ml::asin(x_3d);
    let acos_3d = std::ml::acos(x_3d);
    let atan_3d = std::ml::atan(x_3d);
    let asinh_3d = std::ml::asinh(x_3d);
    let acosh_3d = std::ml::acosh([[[1.0, 1.5], [2.0, 3.0]], [[1.2, 2.5], [1.8, 4.0]]]);
    let atanh_3d = std::ml::atanh([[[0.0, 0.5], [-0.5, 0.707]], [[0.3, -0.3], [0.8, -0.8]]]);
    """
    
    result = compile_and_execute(source, compiler, runtime)
    assert result.success, f"Test failed: {result.errors}"
    
    # Verify 0D
    x_scalar = 0.5
    np.testing.assert_allclose(np.array(result.outputs['asin_0d']), np.arcsin(x_scalar), rtol=1e-5)
    np.testing.assert_allclose(np.array(result.outputs['acos_0d']), np.arccos(x_scalar), rtol=1e-5)
    np.testing.assert_allclose(np.array(result.outputs['atan_0d']), np.arctan(x_scalar), rtol=1e-5)
    np.testing.assert_allclose(np.array(result.outputs['asinh_0d']), np.arcsinh(x_scalar), rtol=1e-5)
    np.testing.assert_allclose(np.array(result.outputs['acosh_0d']), np.arccosh(1.5), rtol=1e-5)
    np.testing.assert_allclose(np.array(result.outputs['atanh_0d']), np.arctanh(x_scalar), rtol=1e-5)
    
    # Verify 1D
    x_1d = np.array([0.0, 0.5, -0.5, 0.707], dtype=np.float32)
    np.testing.assert_allclose(np.array(result.outputs['asin_1d']), np.arcsin(x_1d), rtol=1e-5)
    np.testing.assert_allclose(np.array(result.outputs['acos_1d']), np.arccos(x_1d), rtol=1e-5)
    np.testing.assert_allclose(np.array(result.outputs['atan_1d']), np.arctan(x_1d), rtol=1e-5)
    np.testing.assert_allclose(np.array(result.outputs['asinh_1d']), np.arcsinh(x_1d), rtol=1e-5)
    acosh_input = np.array([1.0, 1.5, 2.0, 3.0], dtype=np.float32)
    np.testing.assert_allclose(np.array(result.outputs['acosh_1d']), np.arccosh(acosh_input), rtol=1e-5)
    atanh_input = np.array([0.0, 0.5, -0.5, 0.707], dtype=np.float32)
    np.testing.assert_allclose(np.array(result.outputs['atanh_1d']), np.arctanh(atanh_input), rtol=1e-5)
    
    # Verify 2D
    x_2d = np.array([[0.0, 0.5], [-0.5, 0.707]], dtype=np.float32)
    np.testing.assert_allclose(np.array(result.outputs['asin_2d']), np.arcsin(x_2d), rtol=1e-5)
    np.testing.assert_allclose(np.array(result.outputs['acos_2d']), np.arccos(x_2d), rtol=1e-5)
    np.testing.assert_allclose(np.array(result.outputs['atan_2d']), np.arctan(x_2d), rtol=1e-5)
    np.testing.assert_allclose(np.array(result.outputs['asinh_2d']), np.arcsinh(x_2d), rtol=1e-5)
    acosh_2d = np.array([[1.0, 1.5], [2.0, 3.0]], dtype=np.float32)
    np.testing.assert_allclose(np.array(result.outputs['acosh_2d']), np.arccosh(acosh_2d), rtol=1e-5)
    atanh_2d = np.array([[0.0, 0.5], [-0.5, 0.707]], dtype=np.float32)
    np.testing.assert_allclose(np.array(result.outputs['atanh_2d']), np.arctanh(atanh_2d), rtol=1e-5)
    
    # Verify 3D
    x_3d = np.array([[[0.0, 0.5], [-0.5, 0.707]], [[0.3, -0.3], [0.8, -0.8]]], dtype=np.float32)
    np.testing.assert_allclose(np.array(result.outputs['asin_3d']), np.arcsin(x_3d), rtol=1e-5)
    np.testing.assert_allclose(np.array(result.outputs['acos_3d']), np.arccos(x_3d), rtol=1e-5)
    np.testing.assert_allclose(np.array(result.outputs['atan_3d']), np.arctan(x_3d), rtol=1e-5)
    np.testing.assert_allclose(np.array(result.outputs['asinh_3d']), np.arcsinh(x_3d), rtol=1e-5)
    acosh_3d = np.array([[[1.0, 1.5], [2.0, 3.0]], [[1.2, 2.5], [1.8, 4.0]]], dtype=np.float32)
    np.testing.assert_allclose(np.array(result.outputs['acosh_3d']), np.arccosh(acosh_3d), rtol=1e-5)
    atanh_3d = np.array([[[0.0, 0.5], [-0.5, 0.707]], [[0.3, -0.3], [0.8, -0.8]]], dtype=np.float32)
    np.testing.assert_allclose(np.array(result.outputs['atanh_3d']), np.arctanh(atanh_3d), rtol=1e-5)


def test_atan2_all_ranks(compiler, runtime):
    """Test atan2 operation across all supported ranks"""
    
    source = """use std::ml;
    // Test 0D (scalar)
    let y_scalar = 1.0;
    let x_scalar = 1.0;
    let atan2_0d = std::ml::atan2(y_scalar, x_scalar);
    
    // Test 1D
    let y_1d = [1.0, 0.0, -1.0, 1.0];
    let x_1d = [1.0, 1.0, 1.0, 0.0];
    let atan2_1d = std::ml::atan2(y_1d, x_1d);
    
    // Test 2D
    let y_2d = [[1.0, 0.0], [-1.0, 1.0]];
    let x_2d = [[1.0, 1.0], [1.0, 0.0]];
    let atan2_2d = std::ml::atan2(y_2d, x_2d);
    
    // Test 3D
    let y_3d = [[[1.0, 0.0], [-1.0, 1.0]], [[0.0, 1.0], [1.0, -1.0]]];
    let x_3d = [[[1.0, 1.0], [1.0, 0.0]], [[1.0, 0.0], [1.0, 1.0]]];
    let atan2_3d = std::ml::atan2(y_3d, x_3d);
    """
    
    result = compile_and_execute(source, compiler, runtime)
    assert result.success, f"Test failed: {result.errors}"
    
    # Verify 0D
    np.testing.assert_allclose(np.array(result.outputs['atan2_0d']), np.arctan2(1.0, 1.0), rtol=1e-5)
    
    # Verify 1D
    y_1d = np.array([1.0, 0.0, -1.0, 1.0], dtype=np.float32)
    x_1d = np.array([1.0, 1.0, 1.0, 0.0], dtype=np.float32)
    np.testing.assert_allclose(np.array(result.outputs['atan2_1d']), np.arctan2(y_1d, x_1d), rtol=1e-5)
    
    # Verify 2D
    y_2d = np.array([[1.0, 0.0], [-1.0, 1.0]], dtype=np.float32)
    x_2d = np.array([[1.0, 1.0], [1.0, 0.0]], dtype=np.float32)
    np.testing.assert_allclose(np.array(result.outputs['atan2_2d']), np.arctan2(y_2d, x_2d), rtol=1e-5)
    
    # Verify 3D
    y_3d = np.array([[[1.0, 0.0], [-1.0, 1.0]], [[0.0, 1.0], [1.0, -1.0]]], dtype=np.float32)
    x_3d = np.array([[[1.0, 1.0], [1.0, 0.0]], [[1.0, 0.0], [1.0, 1.0]]], dtype=np.float32)
    np.testing.assert_allclose(np.array(result.outputs['atan2_3d']), np.arctan2(y_3d, x_3d), rtol=1e-5)