#!/usr/bin/env python3
"""
Tests for std::ml::conv_transpose using one grouped module fixture.
"""

import numpy as np
import pytest

from tests.test_utils import compile_and_execute


@pytest.fixture(scope="module")
def conv_transpose_outputs(module_compiler, module_runtime):
    source = """use std::ml;
    let x_1d = [[[1.0, 2.0, 3.0]]];
    let w_1d = [[[1.0, 0.5]]];
    let b_1d = [0.0];
    let result_1d = std::ml::conv_transpose(x_1d, w_1d, b_1d, [2], [0], [0]);

    let x_2d = [[[[1.0, 2.0], [3.0, 4.0]]]];
    let w_2d = [[[[1.0, 0.5], [0.5, 1.0]]]];
    let b_2d = [0.0];
    let result_2d = std::ml::conv_transpose(x_2d, w_2d, b_2d, [2, 2], [0, 0], [0, 0]);

    let x_3d = [[[[[1.0, 2.0], [3.0, 4.0]]]]];
    let w_3d = [[[[[1.0, 0.5], [0.5, 1.0]], [[0.5, 0.5], [0.5, 0.5]]]]];
    let b_3d = [0.0];
    let result_3d = std::ml::conv_transpose(x_3d, w_3d, b_3d, [1, 1, 1], [0, 0, 0], [0, 0, 0]);
    """
    result = compile_and_execute(source, module_compiler, module_runtime)
    assert result.success, f"Execution failed: {result.errors}"
    return result.outputs


def test_conv_transpose_1d(conv_transpose_outputs):
    expected = np.array([[[1.0, 0.5, 2.0, 1.0, 3.0, 1.5]]], dtype=np.float32)
    np.testing.assert_allclose(np.array(conv_transpose_outputs["result_1d"]), expected, rtol=1e-5)


def test_conv_transpose_2d(conv_transpose_outputs):
    expected = np.array(
        [[[[1.0, 0.5, 2.0, 1.0],
           [0.5, 1.0, 1.0, 2.0],
           [3.0, 1.5, 4.0, 2.0],
           [1.5, 3.0, 2.0, 4.0]]]],
        dtype=np.float32,
    )
    np.testing.assert_allclose(np.array(conv_transpose_outputs["result_2d"]), expected, rtol=1e-5)


def test_conv_transpose_3d(conv_transpose_outputs):
    x_np = np.array([[[[[1.0, 2.0], [3.0, 4.0]]]]], dtype=np.float32)
    w_np = np.array([[[[[1.0, 0.5], [0.5, 1.0]], [[0.5, 0.5], [0.5, 0.5]]]]], dtype=np.float32)
    expected = np.zeros((1, 1, 2, 3, 3), dtype=np.float32)
    for d_out in range(2):
        for i_out in range(3):
            for j_out in range(3):
                total = 0.0
                for kd in range(2):
                    for kh in range(2):
                        for kw in range(2):
                            d_in = d_out - kd
                            i_in = i_out - kh
                            j_in = j_out - kw
                            if 0 <= d_in < 1 and 0 <= i_in < 2 and 0 <= j_in < 2:
                                total += x_np[0, 0, d_in, i_in, j_in] * w_np[0, 0, kd, kh, kw]
                expected[0, 0, d_out, i_out, j_out] = total
    np.testing.assert_allclose(np.array(conv_transpose_outputs["result_3d"]), expected, rtol=1e-5)
