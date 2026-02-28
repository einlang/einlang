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