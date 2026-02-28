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
# Convolution Operation Tests
# Clustered tests for efficiency - all convolution ops tested together
# Compile/execute errors are failures, not skips
# ===========================================================================

def test_conv(compiler, runtime):
    """Test conv operation across all ranks (1D, 2D, 3D)"""
    source = """use std::ml;
    // Conv 1D
    let x_conv_1d = [[[1.0, 2.0, 3.0, 4.0]]];
    let w_conv_1d = [[[1.0, 0.5]]];
    let b_conv_1d = [0.0];
    let result_1d = std::ml::conv(x_conv_1d, w_conv_1d, b_conv_1d, [1], [0], [1]);
    // Conv 2D
    let x_conv_2d = [[[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]]];
    let w_conv_2d = [[[[1.0, 0.0], [0.0, 1.0]]]];
    let b_conv_2d = [0.0];
    let result_2d = std::ml::conv(x_conv_2d, w_conv_2d, b_conv_2d, [1, 1], [0, 0], [1, 1]);
    // Conv 3D
    let x_conv_3d = [[[[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]]];
    let w_conv_3d = [[[[[1.0, 0.0], [0.0, 1.0]], [[0.5, 0.5], [0.5, 0.5]]]]];
    let b_conv_3d = [0.0];
    let result_3d = std::ml::conv(x_conv_3d, w_conv_3d, b_conv_3d, [1, 1, 1], [0, 0, 0], [1, 1, 1]);
    """
    result = compile_and_execute(source, compiler, runtime)
    assert result.success, f"Execution failed: {result.errors}"

    # Verify conv 1D
    x_conv_1d = np.array([[[1.0, 2.0, 3.0, 4.0]]], dtype=np.float32)
    w_conv_1d = np.array([[[1.0, 0.5]]], dtype=np.float32)
    expected_1d = np.array([[[2.0, 3.5, 5.0]]], dtype=np.float32)  # 1*1+2*0.5=2, 2*1+3*0.5=3.5, 3*1+4*0.5=5
    actual_1d = np.array(result.outputs['result_1d'])
    np.testing.assert_allclose(actual_1d, expected_1d, rtol=1e-5)

    # Verify conv 2D
    x_conv_2d = np.array([[[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]]], dtype=np.float32)
    w_conv_2d = np.array([[[[1.0, 0.0], [0.0, 1.0]]]], dtype=np.float32)
    b_conv_2d = np.array([0.0], dtype=np.float32)
    # Manual convolution calculation: 2x2 kernel on 3x3 input with stride 1, pad 0
    expected_2d = np.zeros((1, 1, 2, 2), dtype=np.float32)
    for i in range(2):
        for j in range(2):
            expected_2d[0, 0, i, j] = np.sum(x_conv_2d[0, 0, i:i+2, j:j+2] * w_conv_2d[0, 0]) + b_conv_2d[0]
    actual_2d = np.array(result.outputs['result_2d'])
    np.testing.assert_allclose(actual_2d, expected_2d, rtol=1e-5)

    # Verify conv 3D
    x_conv_3d = np.array([[[[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]]], dtype=np.float32)
    w_conv_3d = np.array([[[[[1.0, 0.0], [0.0, 1.0]], [[0.5, 0.5], [0.5, 0.5]]]]], dtype=np.float32)
    # Front layer: 1*1 + 2*0 + 3*0 + 4*1 = 5
    # Back layer: 5*0.5 + 6*0.5 + 7*0.5 + 8*0.5 = 13
    # Total: 5 + 13 = 18
    expected_3d = np.array([[[[[18.0]]]]], dtype=np.float32)
    actual_3d = np.array(result.outputs['result_3d'])
    np.testing.assert_allclose(actual_3d, expected_3d, rtol=1e-5)
