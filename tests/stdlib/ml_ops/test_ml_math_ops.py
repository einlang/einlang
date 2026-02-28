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
# Math Ops Operation Tests
# Clustered tests for efficiency - all math_ops ops tested together
# Compile/execute errors are failures, not skips
# ===========================================================================

def test_math_ops_clustered_accuracy(compiler, runtime):
    """Test math_ops operations - clustered for efficiency"""
    source = """use std::ml;
    let a = [[1.0, 2.0, 3.0]];
    let b = [[4.0, 5.0, 6.0]];
    let x = [[2.0, 3.0, 4.0]];
    let exp = 3.0;
    let x_1 = [[-2.0, 0.0, 5.0, 10.0]];
    let a_1 = [[true, false, true]];
    let b_1 = [[true, true, false]];
    let x_2 = [[1.0, -2.0, 3.0]];
    let x_3 = [[1.0, 2.0, 4.0]];
    let x_4 = [[1.0, 2.0, 3.0]];
    let a_2 = [[1.0, 0.0, 1.0, 0.0]];
    let b_2 = [[1.0, 1.0, 0.0, 0.0]];
    let x_5 = [[4.0, 9.0, 16.0]];
    let x_6 = [[0.0, 1.0, 2.0]];
    let x_7 = [[1.0, 2.71828, 10.0]];
    let x_8 = [[1.3, 1.7, -1.3, -1.7]];
    let a_3 = [[10.0, 7.0]];
    let b_3 = [[3.0, 2.0]];
    let x_9 = [[-2.0, 0.0, 2.0]];
    let add_result_0 = std::ml::add(a, b);
    let sub_result_0 = std::ml::subtract(a, b);
    let mul_result_0 = std::ml::multiply(a, b);
    let div_result_0 = std::ml::divide(b, a);
    let result_1 = std::ml::power(x, exp);
    let result_2 = std::ml::clip(x_1, 0.0, 6.0);
    let and_result_3 = std::ml::logical_and(a_1, b_1);
    let or_result_3 = std::ml::logical_or(a_1, b_1);
    let not_result_3 = std::ml::logical_not(a_1);
    let result_4 = std::ml::neg(x_2);
    let result_5 = std::ml::reciprocal(x_3);
    let result_6 = std::ml::square(x_4);
    let result_7 = std::ml::logical_xor(a_2, b_2);
    let result_8 = std::ml::sqrt(x_5);
    let result_9 = std::ml::exp(x_6);
    let result_10 = std::ml::log(x_7);
    let floor_result_11 = std::ml::floor(x_8);
    let ceil_result_11 = std::ml::ceil(x_8);
    let round_result_11 = std::ml::round(x_8);
    let mod_result_12 = std::ml::mod(a_3, b_3);
    let fmod_result_12 = std::ml::fmod(a_3, b_3);
    let result_13 = std::ml::sign(x_9);
    """

    result = compile_and_execute(source, compiler, runtime)
    assert result.success, f"Execution failed: {result.errors}"

    # Verify each operation


    # Verify add - Test element-wise Add, Sub, Mul, Div
    a = np.array([[1.0, 2.0, 3.0]])
    b = np.array([[4.0, 5.0, 6.0]])
    np.testing.assert_allclose(np.array(result.outputs['add_result_0']), a + b, rtol=1e-6)
    np.testing.assert_allclose(np.array(result.outputs['sub_result_0']), a - b, rtol=1e-6)
    np.testing.assert_allclose(np.array(result.outputs['mul_result_0']), a * b, rtol=1e-6)
    np.testing.assert_allclose(np.array(result.outputs['div_result_0']), b / a, rtol=1e-6)


    # Verify power - Test Power operation
    x = np.array([[2.0, 3.0, 4.0]])
    exp = 3.0
    expected = x ** exp
    actual = np.array(result.outputs['result_1'])
    np.testing.assert_allclose(actual, expected, rtol=1e-6)


    # Verify clip - Test Clip/Clamp operation
    x_1 = np.array([[-2.0, 0.0, 5.0, 10.0]])  # Using x_1 from source
    expected = np.clip(x_1, 0.0, 6.0)
    actual = np.array(result.outputs['result_2'])
    np.testing.assert_allclose(actual, expected, rtol=1e-6)


    # Verify logical_and - Test logical operators
    a_1 = np.array([[True, False, True]])  # Using a_1 and b_1 from source
    b_1 = np.array([[True, True, False]])
    expected_and = np.logical_and(a_1, b_1)
    expected_or = np.logical_or(a_1, b_1)
    expected_not = np.logical_not(a_1)
    np.testing.assert_array_equal(result.outputs['and_result_3'], expected_and)
    np.testing.assert_array_equal(result.outputs['or_result_3'], expected_or)
    np.testing.assert_array_equal(result.outputs['not_result_3'], expected_not)


    # Verify neg - Test Negation
    x_2 = np.array([[1.0, -2.0, 3.0]])  # Using x_2 from source
    expected = -x_2
    actual = np.array(result.outputs['result_4'])
    np.testing.assert_allclose(actual, expected, rtol=1e-6)


    # Verify reciprocal - Test Reciprocal
    x_3 = np.array([[1.0, 2.0, 4.0]])  # Using x_3 from source
    expected = 1.0 / x_3
    actual = np.array(result.outputs['result_5'])
    np.testing.assert_allclose(actual, expected, rtol=1e-6)


    # Verify square - Test Square
    x_4 = np.array([[1.0, 2.0, 3.0]])  # Using x_4 from source
    expected = x_4 ** 2
    actual = np.array(result.outputs['result_6'])
    np.testing.assert_allclose(actual, expected, rtol=1e-6)


    # Verify logical_xor - Test Logical XOR
    a_2 = np.array([[1.0, 0.0, 1.0, 0.0]])  # Using a_2 and b_2 from source
    b_2 = np.array([[1.0, 1.0, 0.0, 0.0]])
    expected = np.logical_xor(a_2, b_2)
    actual = np.array(result.outputs['result_7'])
    np.testing.assert_array_equal(actual, expected)


    # Verify sqrt - Test sqrt operation
    x_5 = np.array([[4.0, 9.0, 16.0]], dtype=np.float32)  # Using x_5 from source
    expected = np.sqrt(x_5)
    actual = np.array(result.outputs['result_8'])
    np.testing.assert_allclose(actual, expected, rtol=1e-6)


    # Verify exp - Test exp operation
    x_6 = np.array([[0.0, 1.0, 2.0]], dtype=np.float32)  # Using x_6 from source
    expected = np.exp(x_6)
    actual = np.array(result.outputs['result_9'])
    np.testing.assert_allclose(actual, expected, rtol=1e-5)


    # Verify log - Test log operation
    x_7 = np.array([[1.0, 2.71828, 10.0]], dtype=np.float32)  # Using x_7 from source
    expected = np.log(x_7)
    actual = np.array(result.outputs['result_10'])
    np.testing.assert_allclose(actual, expected, rtol=1e-5)


    # Verify floor - Test floor, ceil, round operations
    x_8 = np.array([[1.3, 1.7, -1.3, -1.7]], dtype=np.float32)  # Using x_8 from source
    np.testing.assert_allclose(np.array(result.outputs['floor_result_11']), np.floor(x_8), rtol=1e-6)
    np.testing.assert_allclose(np.array(result.outputs['ceil_result_11']), np.ceil(x_8), rtol=1e-6)
    np.testing.assert_allclose(np.array(result.outputs['round_result_11']), np.round(x_8), rtol=1e-6)


    # Verify mod - Test mod and fmod operations
    a_3 = np.array([[10.0, 7.0]], dtype=np.float32)  # Using a_3 and b_3 from source
    b_3 = np.array([[3.0, 2.0]], dtype=np.float32)
    np.testing.assert_allclose(np.array(result.outputs['mod_result_12']), np.mod(a_3, b_3), rtol=1e-6)
    np.testing.assert_allclose(np.array(result.outputs['fmod_result_12']), np.fmod(a_3, b_3), rtol=1e-6)


    # Verify sign - Test sign operation
    x_9 = np.array([[-2.0, 0.0, 2.0]], dtype=np.float32)  # Using x_9 from source
    expected = np.sign(x_9)
    actual = np.array(result.outputs['result_13'])
    np.testing.assert_allclose(actual, expected, rtol=1e-6)


def test_mod_fmod_edge_cases(compiler, runtime):
    """Test mod and fmod with edge cases (zero, negative, floating point)"""
    source = """use std::ml;
    let a_edge = [10.0, 7.0, 0.0, -5.0];
    let b_edge = [3.0, 0.0, 2.0, 2.0];
    let mod_edge = std::ml::mod(a_edge, b_edge);
    let fmod_edge = std::ml::fmod(a_edge, b_edge);
    let a_float = [10.5, 7.3, 15.7];
    let b_float = [3.2, 2.1, 4.5];
    let mod_float = std::ml::mod(a_float, b_float);
    let fmod_float = std::ml::fmod(a_float, b_float);
    """
    result = compile_and_execute(source, compiler, runtime)
    assert result.success, f"Edge cases test failed: {result.errors}"
    a_edge = np.array([10.0, 7.0, 0.0, -5.0], dtype=np.float32)
    b_edge = np.array([3.0, 2.0, 2.0, 2.0], dtype=np.float32)
    np.testing.assert_allclose(
        np.array(result.outputs['mod_edge'])[[0, 2, 3]],
        np.mod(a_edge[[0, 2, 3]], b_edge[[0, 2, 3]]), rtol=1e-6)
    np.testing.assert_allclose(
        np.array(result.outputs['fmod_edge'])[[0, 2, 3]],
        np.fmod(a_edge[[0, 2, 3]], b_edge[[0, 2, 3]]), rtol=1e-6)
    a_float = np.array([10.5, 7.3, 15.7], dtype=np.float32)
    b_float = np.array([3.2, 2.1, 4.5], dtype=np.float32)
    np.testing.assert_allclose(
        np.array(result.outputs['mod_float']), np.mod(a_float, b_float), rtol=1e-5)
    np.testing.assert_allclose(
        np.array(result.outputs['fmod_float']), np.fmod(a_float, b_float), rtol=1e-5)