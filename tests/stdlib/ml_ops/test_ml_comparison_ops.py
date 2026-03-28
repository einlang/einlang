#!/usr/bin/env python3
"""
Tests for std::ml comparison ops using one grouped module fixture.
"""

import numpy as np
import pytest

from ...test_utils import compile_and_execute


def _assert_comparison_family(outputs, prefix, left, right):
    np.testing.assert_array_equal(np.array(outputs[f"equal_{prefix}"], dtype=bool), left == right)
    np.testing.assert_array_equal(np.array(outputs[f"greater_{prefix}"], dtype=bool), left > right)
    np.testing.assert_array_equal(np.array(outputs[f"less_{prefix}"], dtype=bool), left < right)
    np.testing.assert_array_equal(np.array(outputs[f"greater_or_equal_{prefix}"], dtype=bool), left >= right)
    np.testing.assert_array_equal(np.array(outputs[f"less_or_equal_{prefix}"], dtype=bool), left <= right)
    np.testing.assert_array_equal(np.array(outputs[f"not_equal_{prefix}"], dtype=bool), left != right)


@pytest.fixture(scope="module")
def comparison_outputs(module_compiler, module_runtime):
    source = """use std::ml;
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
    result = compile_and_execute(source, module_compiler, module_runtime)
    assert result.success, f"Execution failed: {result.errors}"
    return result.outputs


def test_comparison_ops_0d_and_1d(comparison_outputs):
    a_0d = np.array(2.0, dtype=np.float32)
    b_0d = np.array(1.0, dtype=np.float32)
    _assert_comparison_family(comparison_outputs, "0d", a_0d, b_0d)

    a_1d = np.array([1.0, 2.0, 3.0, 2.0], dtype=np.float32)
    b_1d = np.array([2.0, 2.0, 1.0, 2.0], dtype=np.float32)
    _assert_comparison_family(comparison_outputs, "1d", a_1d, b_1d)


def test_comparison_ops_2d(comparison_outputs):
    a_2d = np.array([[1.0, 2.0, 3.0], [4.0, 2.0, 1.0]], dtype=np.float32)
    b_2d = np.array([[2.0, 2.0, 2.0], [3.0, 2.0, 2.0]], dtype=np.float32)
    _assert_comparison_family(comparison_outputs, "2d", a_2d, b_2d)


def test_comparison_ops_3d(comparison_outputs):
    a_3d = np.array([[[1.0, 2.0], [3.0, 2.0]], [[4.0, 1.0], [2.0, 3.0]]], dtype=np.float32)
    b_3d = np.array([[[2.0, 2.0], [2.0, 2.0]], [[3.0, 2.0], [2.0, 2.0]]], dtype=np.float32)
    _assert_comparison_family(comparison_outputs, "3d", a_3d, b_3d)


def test_not_all_ranks(compiler, runtime):
    source = """use std::ml;
    let x_0d = true;
    let not_0d = std::ml::not(x_0d);

    let x_1d = [true, false, true, false];
    let not_1d = std::ml::not(x_1d);

    let x_2d = [[true, false, true], [false, true, false]];
    let not_2d = std::ml::not(x_2d);

    let x_3d = [[[true, false], [true, false]], [[false, true], [false, true]]];
    let not_3d = std::ml::not(x_3d);
    """
    result = compile_and_execute(source, compiler, runtime)
    assert result.success, f"Execution failed: {result.errors}"

    x_0d = np.array(True, dtype=bool)
    x_1d = np.array([True, False, True, False], dtype=bool)
    x_2d = np.array([[True, False, True], [False, True, False]], dtype=bool)
    x_3d = np.array([[[True, False], [True, False]], [[False, True], [False, True]]], dtype=bool)
    np.testing.assert_array_equal(result.outputs["not_0d"], np.logical_not(x_0d))
    np.testing.assert_array_equal(np.array(result.outputs["not_1d"], dtype=bool), np.logical_not(x_1d))
    np.testing.assert_array_equal(np.array(result.outputs["not_2d"], dtype=bool), np.logical_not(x_2d))
    np.testing.assert_array_equal(np.array(result.outputs["not_3d"], dtype=bool), np.logical_not(x_3d))
