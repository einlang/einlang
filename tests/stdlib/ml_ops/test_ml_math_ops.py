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


def test_logical_ops_all_ranks(compiler, runtime):
    """Test logical operations (logical_and, logical_or, logical_xor, logical_not) across all supported ranks (0D, 1D, 2D, 3D) with boolean tensors"""
    source = """use std::ml;
    // 0D (scalars) - boolean values
    let a_0d = true;
    let b_0d = false;
    let and_0d = std::ml::logical_and(a_0d, b_0d);
    let or_0d = std::ml::logical_or(a_0d, b_0d);
    let xor_0d = std::ml::logical_xor(a_0d, b_0d);
    let not_0d = std::ml::logical_not(a_0d);

    // 1D - boolean arrays
    let a_1d = [true, false, true, false];
    let b_1d = [true, true, false, false];
    let and_1d = std::ml::logical_and(a_1d, b_1d);
    let or_1d = std::ml::logical_or(a_1d, b_1d);
    let xor_1d = std::ml::logical_xor(a_1d, b_1d);
    let not_1d = std::ml::logical_not(a_1d);

    // 2D - boolean arrays
    let a_2d = [[true, false, true], [false, true, false]];
    let b_2d = [[true, true, false], [false, false, true]];
    let and_2d = std::ml::logical_and(a_2d, b_2d);
    let or_2d = std::ml::logical_or(a_2d, b_2d);
    let xor_2d = std::ml::logical_xor(a_2d, b_2d);
    let not_2d = std::ml::logical_not(a_2d);

    // 3D - boolean arrays
    let a_3d = [[[true, false], [true, false]], [[false, true], [false, true]]];
    let b_3d = [[[true, true], [false, false]], [[true, false], [true, false]]];
    let and_3d = std::ml::logical_and(a_3d, b_3d);
    let or_3d = std::ml::logical_or(a_3d, b_3d);
    let xor_3d = std::ml::logical_xor(a_3d, b_3d);
    let not_3d = std::ml::logical_not(a_3d);
    """
    result = compile_and_execute(source, compiler, runtime)
    assert result.success, f"Execution failed: {result.errors}"

    # Verify 0D - boolean scalars
    a_0d = np.array(True, dtype=bool)
    b_0d = np.array(False, dtype=bool)
    expected_and_0d = np.logical_and(a_0d, b_0d)
    expected_or_0d = np.logical_or(a_0d, b_0d)
    expected_xor_0d = np.logical_xor(a_0d, b_0d)
    expected_not_0d = np.logical_not(a_0d)
    np.testing.assert_array_equal(result.outputs['and_0d'], expected_and_0d)
    np.testing.assert_array_equal(result.outputs['or_0d'], expected_or_0d)
    np.testing.assert_array_equal(result.outputs['xor_0d'], expected_xor_0d)
    np.testing.assert_array_equal(result.outputs['not_0d'], expected_not_0d)

    # Verify 1D - boolean arrays
    a_1d = np.array([True, False, True, False], dtype=bool)
    b_1d = np.array([True, True, False, False], dtype=bool)
    expected_and_1d = np.logical_and(a_1d, b_1d)
    expected_or_1d = np.logical_or(a_1d, b_1d)
    expected_xor_1d = np.logical_xor(a_1d, b_1d)
    expected_not_1d = np.logical_not(a_1d)
    np.testing.assert_array_equal(np.array(result.outputs['and_1d'], dtype=bool), expected_and_1d)
    np.testing.assert_array_equal(np.array(result.outputs['or_1d'], dtype=bool), expected_or_1d)
    np.testing.assert_array_equal(np.array(result.outputs['xor_1d'], dtype=bool), expected_xor_1d)
    np.testing.assert_array_equal(np.array(result.outputs['not_1d'], dtype=bool), expected_not_1d)

    # Verify 2D - boolean arrays
    a_2d = np.array([[True, False, True], [False, True, False]], dtype=bool)
    b_2d = np.array([[True, True, False], [False, False, True]], dtype=bool)
    expected_and_2d = np.logical_and(a_2d, b_2d)
    expected_or_2d = np.logical_or(a_2d, b_2d)
    expected_xor_2d = np.logical_xor(a_2d, b_2d)
    expected_not_2d = np.logical_not(a_2d)
    np.testing.assert_array_equal(np.array(result.outputs['and_2d'], dtype=bool), expected_and_2d)
    np.testing.assert_array_equal(np.array(result.outputs['or_2d'], dtype=bool), expected_or_2d)
    np.testing.assert_array_equal(np.array(result.outputs['xor_2d'], dtype=bool), expected_xor_2d)
    np.testing.assert_array_equal(np.array(result.outputs['not_2d'], dtype=bool), expected_not_2d)

    # Verify 3D - boolean arrays
    a_3d = np.array([[[True, False], [True, False]], [[False, True], [False, True]]], dtype=bool)
    b_3d = np.array([[[True, True], [False, False]], [[True, False], [True, False]]], dtype=bool)
    expected_and_3d = np.logical_and(a_3d, b_3d)
    expected_or_3d = np.logical_or(a_3d, b_3d)
    expected_xor_3d = np.logical_xor(a_3d, b_3d)
    expected_not_3d = np.logical_not(a_3d)
    np.testing.assert_array_equal(np.array(result.outputs['and_3d'], dtype=bool), expected_and_3d)
    np.testing.assert_array_equal(np.array(result.outputs['or_3d'], dtype=bool), expected_or_3d)
    np.testing.assert_array_equal(np.array(result.outputs['xor_3d'], dtype=bool), expected_xor_3d)
    np.testing.assert_array_equal(np.array(result.outputs['not_3d'], dtype=bool), expected_not_3d)


def test_rsqrt_all_ranks(compiler, runtime):
    """Test rsqrt (reciprocal square root) operation across all supported ranks (0D, 1D, 2D, 3D)"""
    source = """use std::ml;
    // 0D (scalar)
    let x_0d = 4.0;
    let rsqrt_0d = std::ml::rsqrt(x_0d);
    
    // 1D
    let x_1d = [4.0, 9.0, 16.0, 25.0];
    let rsqrt_1d = std::ml::rsqrt(x_1d);
    
    // 2D
    let x_2d = [[4.0, 9.0, 16.0], [25.0, 36.0, 49.0]];
    let rsqrt_2d = std::ml::rsqrt(x_2d);
    
    // 3D
    let x_3d = [[[4.0, 9.0], [16.0, 25.0]], [[36.0, 49.0], [64.0, 81.0]]];
    let rsqrt_3d = std::ml::rsqrt(x_3d);
    """
    result = compile_and_execute(source, compiler, runtime)
    assert result.success, f"Execution failed: {result.errors}"
    
    # Verify 0D
    x_0d = np.array(4.0, dtype=np.float32)
    expected_0d = 1.0 / np.sqrt(x_0d)
    np.testing.assert_allclose(result.outputs['rsqrt_0d'], expected_0d, rtol=1e-6)
    
    # Verify 1D
    x_1d = np.array([4.0, 9.0, 16.0, 25.0], dtype=np.float32)
    expected_1d = 1.0 / np.sqrt(x_1d)
    np.testing.assert_allclose(np.array(result.outputs['rsqrt_1d']), expected_1d, rtol=1e-6)
    
    # Verify 2D
    x_2d = np.array([[4.0, 9.0, 16.0], [25.0, 36.0, 49.0]], dtype=np.float32)
    expected_2d = 1.0 / np.sqrt(x_2d)
    np.testing.assert_allclose(np.array(result.outputs['rsqrt_2d']), expected_2d, rtol=1e-6)
    
    # Verify 3D
    x_3d = np.array([[[4.0, 9.0], [16.0, 25.0]], [[36.0, 49.0], [64.0, 81.0]]], dtype=np.float32)
    expected_3d = 1.0 / np.sqrt(x_3d)
    np.testing.assert_allclose(np.array(result.outputs['rsqrt_3d']), expected_3d, rtol=1e-6)


def test_mod_fmod_all_ranks(compiler, runtime):
    """Test mod and fmod operations across all supported ranks (0D, 1D, 2D, 3D)"""
    
    source = """use std::ml;
    // Test 0D (scalar)
    let a_scalar = 10.0;
    let b_scalar = 3.0;
    let mod_0d = std::ml::mod(a_scalar, b_scalar);
    let fmod_0d = std::ml::fmod(a_scalar, b_scalar);
    
    // Test 1D
    let a_1d = [10.0, 7.0, 15.0, -10.0];
    let b_1d = [3.0, 2.0, 4.0, 3.0];
    let mod_1d = std::ml::mod(a_1d, b_1d);
    let fmod_1d = std::ml::fmod(a_1d, b_1d);
    
    // Test 2D
    let a_2d = [[10.0, 7.0], [15.0, 20.0]];
    let b_2d = [[3.0, 2.0], [4.0, 6.0]];
    let mod_2d = std::ml::mod(a_2d, b_2d);
    let fmod_2d = std::ml::fmod(a_2d, b_2d);
    
    // Test 3D
    let a_3d = [[[10.0, 7.0], [15.0, 20.0]], [[5.0, 12.0], [8.0, 9.0]]];
    let b_3d = [[[3.0, 2.0], [4.0, 6.0]], [[2.0, 5.0], [3.0, 4.0]]];
    let mod_3d = std::ml::mod(a_3d, b_3d);
    let fmod_3d = std::ml::fmod(a_3d, b_3d);
    """
    
    result = compile_and_execute(source, compiler, runtime)
    assert result.success, f"Test failed: {result.errors}"
    
    # Verify 0D
    a_scalar, b_scalar = 10.0, 3.0
    np.testing.assert_allclose(np.array(result.outputs['mod_0d']), np.mod(a_scalar, b_scalar), rtol=1e-6)
    np.testing.assert_allclose(np.array(result.outputs['fmod_0d']), np.fmod(a_scalar, b_scalar), rtol=1e-6)
    
    # Verify 1D
    a_1d = np.array([10.0, 7.0, 15.0, -10.0], dtype=np.float32)
    b_1d = np.array([3.0, 2.0, 4.0, 3.0], dtype=np.float32)
    np.testing.assert_allclose(np.array(result.outputs['mod_1d']), np.mod(a_1d, b_1d), rtol=1e-6)
    np.testing.assert_allclose(np.array(result.outputs['fmod_1d']), np.fmod(a_1d, b_1d), rtol=1e-6)
    
    # Verify 2D
    a_2d = np.array([[10.0, 7.0], [15.0, 20.0]], dtype=np.float32)
    b_2d = np.array([[3.0, 2.0], [4.0, 6.0]], dtype=np.float32)
    np.testing.assert_allclose(np.array(result.outputs['mod_2d']), np.mod(a_2d, b_2d), rtol=1e-6)
    np.testing.assert_allclose(np.array(result.outputs['fmod_2d']), np.fmod(a_2d, b_2d), rtol=1e-6)
    
    # Verify 3D
    a_3d = np.array([[[10.0, 7.0], [15.0, 20.0]], [[5.0, 12.0], [8.0, 9.0]]], dtype=np.float32)
    b_3d = np.array([[[3.0, 2.0], [4.0, 6.0]], [[2.0, 5.0], [3.0, 4.0]]], dtype=np.float32)
    np.testing.assert_allclose(np.array(result.outputs['mod_3d']), np.mod(a_3d, b_3d), rtol=1e-6)
    np.testing.assert_allclose(np.array(result.outputs['fmod_3d']), np.fmod(a_3d, b_3d), rtol=1e-6)


def test_expm1_log1p_all_ranks(compiler, runtime):
    """Test expm1 and log1p operations across all supported ranks (0D, 1D, 2D, 3D)"""
    
    source = """use std::ml;
    // Test 0D (scalar)
    let x_scalar = 0.5;
    let expm1_0d = std::ml::expm1(x_scalar);
    let log1p_0d = std::ml::log1p(x_scalar);
    
    // Test 1D
    let x_1d = [0.0, 0.5, 1.0, -0.5, 2.0];
    let expm1_1d = std::ml::expm1(x_1d);
    let log1p_1d = std::ml::log1p(x_1d);
    
    // Test 2D
    let x_2d = [[0.0, 0.1, 0.5], [1.0, 2.0, -0.1]];
    let expm1_2d = std::ml::expm1(x_2d);
    let log1p_2d = std::ml::log1p(x_2d);
    
    // Test 3D
    let x_3d = [[[0.0, 0.1], [0.5, 1.0]], [[-0.1, 0.0], [2.0, 3.0]]];
    let expm1_3d = std::ml::expm1(x_3d);
    let log1p_3d = std::ml::log1p(x_3d);
    """
    
    result = compile_and_execute(source, compiler, runtime)
    assert result.success, f"Test failed: {result.errors}"
    
    # Verify 0D
    x_scalar = 0.5
    np.testing.assert_allclose(np.array(result.outputs['expm1_0d']), np.expm1(x_scalar), rtol=1e-6)
    np.testing.assert_allclose(np.array(result.outputs['log1p_0d']), np.log1p(x_scalar), rtol=1e-6)
    
    # Verify 1D
    x_1d = np.array([0.0, 0.5, 1.0, -0.5, 2.0], dtype=np.float32)
    np.testing.assert_allclose(np.array(result.outputs['expm1_1d']), np.expm1(x_1d), rtol=1e-6)
    np.testing.assert_allclose(np.array(result.outputs['log1p_1d']), np.log1p(x_1d), rtol=1e-6)
    
    # Verify 2D
    x_2d = np.array([[0.0, 0.1, 0.5], [1.0, 2.0, -0.1]], dtype=np.float32)
    np.testing.assert_allclose(np.array(result.outputs['expm1_2d']), np.expm1(x_2d), rtol=1e-6)
    np.testing.assert_allclose(np.array(result.outputs['log1p_2d']), np.log1p(x_2d), rtol=1e-6)
    
    # Verify 3D
    x_3d = np.array([[[0.0, 0.1], [0.5, 1.0]], [[-0.1, 0.0], [2.0, 3.0]]], dtype=np.float32)
    np.testing.assert_allclose(np.array(result.outputs['expm1_3d']), np.expm1(x_3d), rtol=1e-6)
    np.testing.assert_allclose(np.array(result.outputs['log1p_3d']), np.log1p(x_3d), rtol=1e-6)


def test_expm1_log1p_numerical_stability(compiler, runtime):
    """Test expm1 and log1p for numerical stability with small values"""
    
    source = """use std::ml;
    // Test with very small values where expm1 and log1p provide better numerical stability
    let x_small = [1e-8, 1e-10, -1e-8, -1e-10];
    let expm1_small = std::ml::expm1(x_small);
    let log1p_small = std::ml::log1p(x_small);
    
    // Test with values near zero
    let x_near_zero = [0.0, 1e-15, -1e-15];
    let expm1_near_zero = std::ml::expm1(x_near_zero);
    let log1p_near_zero = std::ml::log1p(x_near_zero);
    """
    
    result = compile_and_execute(source, compiler, runtime)
    assert result.success, f"Numerical stability test failed: {result.errors}"
    
    # Test small values
    x_small = np.array([1e-8, 1e-10, -1e-8, -1e-10], dtype=np.float32)
    np.testing.assert_allclose(
        np.array(result.outputs['expm1_small']),
        np.expm1(x_small),
        rtol=1e-5  # Slightly relaxed for very small values
    )
    np.testing.assert_allclose(
        np.array(result.outputs['log1p_small']),
        np.log1p(x_small),
        rtol=1e-5
    )
    
    # Test near zero
    x_near_zero = np.array([0.0, 1e-15, -1e-15], dtype=np.float32)
    np.testing.assert_allclose(
        np.array(result.outputs['expm1_near_zero']),
        np.expm1(x_near_zero),
        rtol=1e-5
    )
    np.testing.assert_allclose(
        np.array(result.outputs['log1p_near_zero']),
        np.log1p(x_near_zero),
        rtol=1e-5
    )