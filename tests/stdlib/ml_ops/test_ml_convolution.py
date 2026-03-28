#!/usr/bin/env python3
"""
Tests for std::ml::conv using one module-scoped compile/execute.
"""

import numpy as np
import pytest

from tests.test_utils import compile_and_execute


@pytest.fixture(scope="module")
def conv_outputs(module_compiler, module_runtime):
    source = """use std::ml;
    let x_1d = [[[1.0, 2.0, 3.0, 4.0]]];
    let w_1d = [[[1.0, 0.5]]];
    let b_1d = [0.0];
    let result_1d = std::ml::conv(x_1d, w_1d, b_1d, [1], [0, 0], [1], 1);

    let x_2d = [[[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]]];
    let w_2d = [[[[1.0, 0.0], [0.0, 1.0]]]];
    let b_2d = [0.0];
    let result_2d = std::ml::conv(x_2d, w_2d, b_2d, [1, 1], [0, 0, 0, 0], [1, 1], 1);

    let x_3d = [[[[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]]];
    let w_3d = [[[[[1.0, 0.0], [0.0, 1.0]], [[0.5, 0.5], [0.5, 0.5]]]]];
    let b_3d = [0.0];
    let result_3d = std::ml::conv(x_3d, w_3d, b_3d, [1, 1, 1], [0, 0, 0, 0, 0, 0], [1, 1, 1], 1);
    """
    result = compile_and_execute(source, module_compiler, module_runtime)
    assert result.success, f"Execution failed: {result.errors}"
    return result.outputs


def test_conv_1d(conv_outputs):
    expected = np.array([[[2.0, 3.5, 5.0]]], dtype=np.float32)
    actual = np.array(conv_outputs["result_1d"])
    np.testing.assert_allclose(actual, expected, rtol=1e-5)


def test_conv_2d(conv_outputs):
    x = np.array(
        [[[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]]],
        dtype=np.float32,
    )
    w = np.array([[[[1.0, 0.0], [0.0, 1.0]]]], dtype=np.float32)
    b = np.array([0.0], dtype=np.float32)
    expected = np.zeros((1, 1, 2, 2), dtype=np.float32)
    for i in range(2):
        for j in range(2):
            expected[0, 0, i, j] = np.sum(x[0, 0, i : i + 2, j : j + 2] * w[0, 0]) + b[0]
    actual = np.array(conv_outputs["result_2d"])
    np.testing.assert_allclose(actual, expected, rtol=1e-5)


def test_conv_3d(conv_outputs):
    expected = np.array([[[[[18.0]]]]], dtype=np.float32)
    actual = np.array(conv_outputs["result_3d"])
    np.testing.assert_allclose(actual, expected, rtol=1e-5)
