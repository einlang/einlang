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
# Comparison Ops Operation Tests
# Clustered tests for efficiency - all comparison_ops ops tested together
# Compile/execute errors are failures, not skips
# ===========================================================================

def test_comparison_ops_clustered_accuracy(compiler, runtime):
    """Test comparison_ops operations - clustered for efficiency (includes all-ranks)"""
    source = """use std::ml;
    let a = [[1.0, 2.0, 3.0]];
    let b = [[2.0, 2.0, 2.0]];
    let greater_0 = std::ml::greater(a, b);
    let less_0 = std::ml::less(a, b);
    let equal_0 = std::ml::equal(a, b);
    let result_1 = std::ml::greater_or_equal(a, b);
    let result_2 = std::ml::less_or_equal(a, b);
    let result_3 = std::ml::not_equal(a, b);
    # All ranks (merged from test_comparison_ops_all_ranks)
    let a_0d = 2.0;
    let b_0d = 1.0;
    let equal_0d = std::ml::equal(a_0d, b_0d);
    let greater_0d = std::ml::greater(a_0d, b_0d);
    let less_0d = std::ml::less(a_0d, b_0d);
    let greater_or_equal_0d = std::ml::greater_or_equal(a_0d, b_0d);
    let less_or_equal_0d = std::ml::less_or_equal(a_0d, b_0d);
    let not_equal_0d = std::ml::not_equal(a_0d, b_0d);
    let a_1d = [1.0, 2.0, 3.0, 2.0];
    let b_1d = [2.0, 2.0, 1.0, 2.0];
    let equal_1d = std::ml::equal(a_1d, b_1d);
    let greater_1d = std::ml::greater(a_1d, b_1d);
    let less_1d = std::ml::less(a_1d, b_1d);
    let greater_or_equal_1d = std::ml::greater_or_equal(a_1d, b_1d);
    let less_or_equal_1d = std::ml::less_or_equal(a_1d, b_1d);
    let not_equal_1d = std::ml::not_equal(a_1d, b_1d);
    let a_2d = [[1.0, 2.0, 3.0], [4.0, 2.0, 1.0]];
    let b_2d = [[2.0, 2.0, 2.0], [3.0, 2.0, 2.0]];
    let equal_2d = std::ml::equal(a_2d, b_2d);
    let greater_2d = std::ml::greater(a_2d, b_2d);
    let less_2d = std::ml::less(a_2d, b_2d);
    let greater_or_equal_2d = std::ml::greater_or_equal(a_2d, b_2d);
    let less_or_equal_2d = std::ml::less_or_equal(a_2d, b_2d);
    let not_equal_2d = std::ml::not_equal(a_2d, b_2d);
    let a_3d = [[[1.0, 2.0], [3.0, 2.0]], [[4.0, 1.0], [2.0, 3.0]]];
    let b_3d = [[[2.0, 2.0], [2.0, 2.0]], [[3.0, 2.0], [2.0, 2.0]]];
    let equal_3d = std::ml::equal(a_3d, b_3d);
    let greater_3d = std::ml::greater(a_3d, b_3d);
    let less_3d = std::ml::less(a_3d, b_3d);
    let greater_or_equal_3d = std::ml::greater_or_equal(a_3d, b_3d);
    let less_or_equal_3d = std::ml::less_or_equal(a_3d, b_3d);
    let not_equal_3d = std::ml::not_equal(a_3d, b_3d);
    """

    result = compile_and_execute(source, compiler, runtime)
    assert result.success, f"Execution failed: {result.errors}"

    # Verify each operation


    # Verify greater - Test comparison operators
    a = np.array([[1.0, 2.0, 3.0]])
    b = np.array([[2.0, 2.0, 2.0]])
    expected = (a > b)
    actual = np.array(result.outputs['greater_0'])
    np.testing.assert_array_equal(actual, expected)


    # Verify less - Test comparison operators
    a = np.array([[1.0, 2.0, 3.0]])
    b = np.array([[2.0, 2.0, 2.0]])
    expected = (a < b)
    actual = np.array(result.outputs['less_0'])
    np.testing.assert_array_equal(actual, expected)


    # Verify equal - Test comparison operators
    a = np.array([[1.0, 2.0, 3.0]])
    b = np.array([[2.0, 2.0, 2.0]])
    expected = (a == b)
    actual = np.array(result.outputs['equal_0'])
    np.testing.assert_array_equal(actual, expected)


    # Verify greater_or_equal - Test GreaterOrEqual
    a = np.array([[1.0, 2.0, 3.0]])
    b = np.array([[2.0, 2.0, 2.0]])
    expected = (a >= b)
    actual = np.array(result.outputs['result_1'])
    np.testing.assert_array_equal(actual, expected)


    # Verify less_or_equal - Test LessOrEqual
    a = np.array([[1.0, 2.0, 3.0]])
    b = np.array([[2.0, 2.0, 2.0]])
    expected = (a <= b)
    actual = np.array(result.outputs['result_2'])
    np.testing.assert_array_equal(actual, expected)


    # Verify not_equal - Test NotEqual
    a = np.array([[1.0, 2.0, 3.0]])
    b = np.array([[2.0, 2.0, 2.0]])
    expected = (a != b)
    actual = np.array(result.outputs['result_3'])
    np.testing.assert_array_equal(actual, expected)

    # All ranks (merged from test_comparison_ops_all_ranks)
    a_0d = np.array(2.0, dtype=np.float32)
    b_0d = np.array(1.0, dtype=np.float32)
    np.testing.assert_array_equal(result.outputs['equal_0d'], (a_0d == b_0d))
    np.testing.assert_array_equal(result.outputs['greater_0d'], (a_0d > b_0d))
    np.testing.assert_array_equal(result.outputs['less_0d'], (a_0d < b_0d))
    np.testing.assert_array_equal(result.outputs['greater_or_equal_0d'], (a_0d >= b_0d))
    np.testing.assert_array_equal(result.outputs['less_or_equal_0d'], (a_0d <= b_0d))
    np.testing.assert_array_equal(result.outputs['not_equal_0d'], (a_0d != b_0d))
    a_1d = np.array([1.0, 2.0, 3.0, 2.0], dtype=np.float32)
    b_1d = np.array([2.0, 2.0, 1.0, 2.0], dtype=np.float32)
    np.testing.assert_array_equal(np.array(result.outputs['equal_1d'], dtype=bool), (a_1d == b_1d))
    np.testing.assert_array_equal(np.array(result.outputs['greater_1d'], dtype=bool), (a_1d > b_1d))
    np.testing.assert_array_equal(np.array(result.outputs['less_1d'], dtype=bool), (a_1d < b_1d))
    np.testing.assert_array_equal(np.array(result.outputs['greater_or_equal_1d'], dtype=bool), (a_1d >= b_1d))
    np.testing.assert_array_equal(np.array(result.outputs['less_or_equal_1d'], dtype=bool), (a_1d <= b_1d))
    np.testing.assert_array_equal(np.array(result.outputs['not_equal_1d'], dtype=bool), (a_1d != b_1d))
    a_2d = np.array([[1.0, 2.0, 3.0], [4.0, 2.0, 1.0]], dtype=np.float32)
    b_2d = np.array([[2.0, 2.0, 2.0], [3.0, 2.0, 2.0]], dtype=np.float32)
    np.testing.assert_array_equal(np.array(result.outputs['equal_2d'], dtype=bool), (a_2d == b_2d))
    np.testing.assert_array_equal(np.array(result.outputs['greater_2d'], dtype=bool), (a_2d > b_2d))
    np.testing.assert_array_equal(np.array(result.outputs['less_2d'], dtype=bool), (a_2d < b_2d))
    np.testing.assert_array_equal(np.array(result.outputs['greater_or_equal_2d'], dtype=bool), (a_2d >= b_2d))
    np.testing.assert_array_equal(np.array(result.outputs['less_or_equal_2d'], dtype=bool), (a_2d <= b_2d))
    np.testing.assert_array_equal(np.array(result.outputs['not_equal_2d'], dtype=bool), (a_2d != b_2d))
    a_3d = np.array([[[1.0, 2.0], [3.0, 2.0]], [[4.0, 1.0], [2.0, 3.0]]], dtype=np.float32)
    b_3d = np.array([[[2.0, 2.0], [2.0, 2.0]], [[3.0, 2.0], [2.0, 2.0]]], dtype=np.float32)
    np.testing.assert_array_equal(np.array(result.outputs['equal_3d'], dtype=bool), (a_3d == b_3d))
    np.testing.assert_array_equal(np.array(result.outputs['greater_3d'], dtype=bool), (a_3d > b_3d))
    np.testing.assert_array_equal(np.array(result.outputs['less_3d'], dtype=bool), (a_3d < b_3d))
    np.testing.assert_array_equal(np.array(result.outputs['greater_or_equal_3d'], dtype=bool), (a_3d >= b_3d))
    np.testing.assert_array_equal(np.array(result.outputs['less_or_equal_3d'], dtype=bool), (a_3d <= b_3d))
    np.testing.assert_array_equal(np.array(result.outputs['not_equal_3d'], dtype=bool), (a_3d != b_3d))


def test_not_all_ranks(compiler, runtime):
    """Test not (unary logical negation) operation across all supported ranks (0D, 1D, 2D, 3D)"""
    source = """use std::ml;
    # 0D (scalar) - boolean
    let x_0d = true;
    let not_0d = std::ml::not(x_0d);
    
    # 1D - boolean array
    let x_1d = [true, false, true, false];
    let not_1d = std::ml::not(x_1d);
    
    # 2D - boolean array
    let x_2d = [[true, false, true], [false, true, false]];
    let not_2d = std::ml::not(x_2d);
    
    # 3D - boolean array
    let x_3d = [[[true, false], [true, false]], [[false, true], [false, true]]];
    let not_3d = std::ml::not(x_3d);
    """
    result = compile_and_execute(source, compiler, runtime)
    assert result.success, f"Execution failed: {result.errors}"
    
    # Verify 0D
    x_0d = np.array(True, dtype=bool)
    expected_0d = np.logical_not(x_0d)
    np.testing.assert_array_equal(result.outputs['not_0d'], expected_0d)
    
    # Verify 1D
    x_1d = np.array([True, False, True, False], dtype=bool)
    expected_1d = np.logical_not(x_1d)
    np.testing.assert_array_equal(np.array(result.outputs['not_1d'], dtype=bool), expected_1d)
    
    # Verify 2D
    x_2d = np.array([[True, False, True], [False, True, False]], dtype=bool)
    expected_2d = np.logical_not(x_2d)
    np.testing.assert_array_equal(np.array(result.outputs['not_2d'], dtype=bool), expected_2d)
    
    # Verify 3D
    x_3d = np.array([[[True, False], [True, False]], [[False, True], [False, True]]], dtype=bool)
    expected_3d = np.logical_not(x_3d)
    np.testing.assert_array_equal(np.array(result.outputs['not_3d'], dtype=bool), expected_3d)