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
from tests.test_utils import compile_and_execute
# ===========================================================================
# Activation Function Tests
# ===========================================================================



# ===========================================================================
# Loss Functions Operation Tests
# Clustered tests for efficiency - all loss_functions ops tested together
# Compile/execute errors are failures, not skips
# ===========================================================================

def test_loss_functions_clustered_accuracy(compiler, runtime):
    """Test loss_functions operations - clustered for efficiency"""
    source = """use std::ml;
    let pred = [[1.0, 2.0, 3.0]];
    let target = [[1.5, 2.5, 2.5]];
    let target_1 = [[1.5, 3.5, 2.5]];
    let delta = 1.0;
    let pred_1 = [[0.1, 0.5, 0.9]];
    let target_2 = [[0.0, 1.0, 1.0]];
    let target_3 = [[0.0, 0.0, 1.0]];
    let pred_bce = [[0.2, 0.8, 0.3]];
    let target_bce = [[0.0, 1.0, 1.0]];
    let result_0 = std::ml::mse_loss(pred, target);
    let result_1 = std::ml::mae_loss(pred, target);
    let result_4 = std::ml::cross_entropy_loss(pred, target_3);
    let result_5 = std::ml::ml_ex::huber_loss(pred, target, delta);
    let result_6 = std::ml::ml_ex::binary_cross_entropy(pred_bce, target_bce);
    """

    result = compile_and_execute(source, compiler, runtime)
    assert result.success, f"Execution failed: {result.errors}"

    # Verify each operation


    # Verify mse_loss - Test MSE Loss
    pred = np.array([[1.0, 2.0, 3.0]])
    target = np.array([[1.5, 2.5, 2.5]])
    expected = np.mean((pred - target)**2, axis=-1)
    actual = np.array(result.outputs['result_0'])
    np.testing.assert_allclose(actual, expected, rtol=1e-6)


    # Verify mae_loss - Test MAE Loss
    pred = np.array([[1.0, 2.0, 3.0]])
    target = np.array([[1.5, 2.5, 2.5]])
    expected = np.mean(np.abs(pred - target), axis=-1)
    actual = np.array(result.outputs['result_1'])
    np.testing.assert_allclose(actual, expected, rtol=1e-6)


    # Verify cross_entropy_loss - Test Cross Entropy Loss
    pred = np.array([[1.0, 2.0, 3.0]])
    target = np.array([[0.0, 0.0, 1.0]])  # Using target_3 from source
    exp_pred = np.exp(pred - np.max(pred, axis=-1, keepdims=True))
    softmax_pred = exp_pred / np.sum(exp_pred, axis=-1, keepdims=True)
    expected = -np.sum(target * np.log(softmax_pred), axis=-1)
    actual = np.array(result.outputs['result_4'])
    np.testing.assert_allclose(actual, expected, rtol=1e-5)


    # Verify huber_loss - Test Huber Loss
    pred = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)  # Using pred from source
    target = np.array([[1.5, 2.5, 2.5]], dtype=np.float32)  # Using target from source
    delta = 1.0  # Using delta from source
    diff = pred - target
    abs_diff = np.abs(diff)
    expected = np.mean(np.where(abs_diff <= delta, 0.5 * diff**2, delta * (abs_diff - 0.5 * delta)), axis=-1)
    actual = np.array(result.outputs['result_5'])
    np.testing.assert_allclose(actual, expected, rtol=1e-5)


    # Verify binary_cross_entropy - Test Binary Cross Entropy Loss
    pred_bce = np.array([[0.2, 0.8, 0.3]], dtype=np.float32)  # Using pred_bce from source
    target_bce = np.array([[0.0, 1.0, 1.0]], dtype=np.float32)  # Using target_bce from source
    eps = 1e-7
    clipped_pred = np.clip(pred_bce, eps, 1.0 - eps)
    expected = -np.sum(target_bce * np.log(clipped_pred) + (1.0 - target_bce) * np.log(1.0 - clipped_pred), axis=-1)
    actual = np.array(result.outputs['result_6'])
    np.testing.assert_allclose(actual, expected, rtol=1e-5)