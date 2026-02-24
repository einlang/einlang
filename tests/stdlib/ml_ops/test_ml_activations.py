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

# USE_V2: conftest provides compiler/runtime
# Use conftest's compiler/runtime for speed (session/class-scoped)
# ===========================================================================
# Activation Function Tests
# ===========================================================================



# ===========================================================================
# Activations Operation Tests
# Clustered tests for efficiency - all activations ops tested together
# Compile/execute errors are failures, not skips
# ===========================================================================

def test_activations_clustered_accuracy(compiler, runtime):
    """Test activations operations - clustered for efficiency"""
    source = """use std::ml;
    let x = [[-2.0, -1.0, 0.0, 1.0, 2.0]];
    let x_1 = [[1.0, 2.0, 3.0, 4.0]];
    let x_2 = [[-1.0, 0.0, 1.0, 2.0]];
    let alpha = 0.1;
    let alpha_1 = 1.0;
    let x_3 = [[-1.0, 0.0, 1.0]];
    let x_4 = [[-2.0, 0.0, 2.0, 25.0]];
    let x_5 = [[-1.0, 0.0, 3.0, 7.0]];
    let x_6 = [[-1.0, 0.0, 1.0, 2.0]];
    let x_7 = [[-2.0, -0.5, 0.0, 0.5, 2.0]];
    let alpha_2 = 0.25;
    let x_8 = [[-2.0, -1.0, 0.0, 1.0]];
    let alpha_3 = 1.0;
    let x_9 = [[-4.0, -1.0, 0.0, 1.0, 4.0]];
    let x_10 = [[-4.0, -1.0, 0.0, 1.0, 4.0]];
    let x_11 = [[1.0, 2.0, 3.0]];
    let x_12 = [[-2.0, -0.5, 0.0, 0.5, 2.0]];
    let lambda = 1.0;
    let x_13 = [[-2.0, -0.5, 0.0, 0.5, 2.0]];
    let x_14 = [[-1.0, 0.0, 1.0, 2.0]];
    let x_15 = [[-1.0, 0.0, 1.0]];
    let x_16 = [[-1.0, 0.0, 0.5, 1.0, 2.0]];
    let alpha_4 = 0.5;
    let pred = [[1.0, 2.0, 3.0]];
    let target = [[0.0, 0.0, 1.0]];
    let x_17 = [[-1.0, 0.0, 1.0, 2.0]];
    let x_elu_alpha = [[-2.0, -1.0, 0.0, 1.0, 2.0]];
    let alpha_elu_alpha = 0.5;
    let x_gelu_tanh = [[-2.0, -1.0, 0.0, 1.0, 2.0]];
    let result_0 = std::ml::relu(x);
    let result_1 = std::ml::sigmoid(x);
    let result_2 = std::ml::softmax(x);
    let result_3 = std::ml::gelu(x);
    let result_4 = std::ml::leaky_relu(x, alpha);
    let result_5 = std::ml::elu(x_2, alpha);
    let result_6 = std::ml::selu(x);
    let result_7 = std::ml::softplus(x);
    let result_8 = std::ml::relu6(x);
    let result_9 = std::ml::swish(x);
    let result_10 = std::ml::hardtanh(x, -1.0, 1.0);
    let result_11 = std::ml::prelu(x, alpha);
    let result_12 = std::ml::celu(x_8, alpha_3);
    let result_13 = std::ml::softsign(x);
    let result_14 = std::ml::hardswish(x_9);
    let result_15 = std::ml::hardsigmoid(x_10);
    let result_16 = std::ml::log_softmax(x_11);
    let result_17 = std::ml::softshrink(x_12, lambda);
    let result_18 = std::ml::hardshrink(x_13, lambda);
    let result_19 = std::ml::threshold(x_14, 0.5, 0.0);
    let result_20 = std::ml::tanhshrink(x_15);
    let result_21 = std::ml::thresholded_relu(x_16, alpha_4);
    let result_22 = std::ml::softmax_cross_entropy_loss(pred, target);
    let result_23 = std::ml::mish(x_17);
    let result_24 = std::ml::elu_alpha(x_elu_alpha, alpha_elu_alpha);
    let result_25 = std::ml::gelu_tanh(x_gelu_tanh);
    """

    result = compile_and_execute(source, compiler, runtime)
    assert result.success, f"Execution failed: {result.errors}"

    # Verify each operation


    # Verify relu - Test ReLU against NumPy
    x = np.array([[-2.0, -1.0, 0.0, 1.0, 2.0]])
    expected = np.maximum(0, x)
    actual = np.array(result.outputs['result_0'])
    np.testing.assert_allclose(actual, expected, rtol=1e-6)


    # Verify sigmoid - Test Sigmoid against NumPy
    x = np.array([[-2.0, -1.0, 0.0, 1.0, 2.0]])
    expected = 1.0 / (1.0 + np.exp(-x))
    actual = np.array(result.outputs['result_1'])
    np.testing.assert_allclose(actual, expected, rtol=1e-6)


    # Verify softmax - Test Softmax against NumPy (numerically stable version)
    x = np.array([[-2.0, -1.0, 0.0, 1.0, 2.0]])  # Using x from source (result_2 uses x)
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    expected = exp_x / np.sum(exp_x, axis=-1, keepdims=True)
    actual = np.array(result.outputs['result_2'])
    np.testing.assert_allclose(actual, expected, rtol=1e-6)
    assert_float_close(np.sum(actual[0]), 1.0, rel_tol=1e-6)


    # Verify gelu - Test GELU approximation
    x = np.array([[-2.0, -1.0, 0.0, 1.0, 2.0]])  # Using x from source (result_3 uses x)
    sqrt_2_over_pi = np.sqrt(2.0 / np.pi)
    expected = 0.5 * x * (1.0 + np.tanh(sqrt_2_over_pi * (x + 0.044715 * x**3)))
    actual = np.array(result.outputs['result_3'])
    np.testing.assert_allclose(actual, expected, rtol=1e-5)


    # Verify leaky_relu - Test Leaky ReLU
    x = np.array([[-2.0, -1.0, 0.0, 1.0, 2.0]])
    alpha = result.outputs.get('alpha', 0.1)
    expected = np.where(x > 0, x, alpha * x)
    actual = np.array(result.outputs['result_4'])
    np.testing.assert_allclose(actual, expected, rtol=1e-6)


    # Verify elu - Test ELU
    x_2 = np.array([[-1.0, 0.0, 1.0, 2.0]])  # Using x_2 from source (result_5 uses x_2)
    alpha = result.outputs.get('alpha', 0.1)  # Using alpha from source
    expected = np.where(x_2 > 0, x_2, alpha * (np.exp(x_2) - 1.0))
    actual = np.array(result.outputs['result_5'])
    np.testing.assert_allclose(actual, expected, rtol=1e-5)


    # Verify selu - Test SELU
    x = np.array([[-2.0, -1.0, 0.0, 1.0, 2.0]])  # Using x from source (result_6 uses x)
    # SELU constants (not in outputs, use direct values)
    lambda_val = 1.0507009873554804934193349852946
    alpha = 1.6732632423543772848170429916717
    expected = np.where(x > 0, lambda_val * x, lambda_val * alpha * (np.exp(x) - 1.0))
    actual = np.array(result.outputs['result_6'])
    np.testing.assert_allclose(actual, expected, rtol=1e-5)


    # Verify softplus - Test Softplus
    x = np.array([[-2.0, -1.0, 0.0, 1.0, 2.0]])  # Using x from source (result_7 uses x)
    expected = np.log(np.exp(x) + 1)
    actual = np.array(result.outputs['result_7'])
    np.testing.assert_allclose(actual, expected, rtol=1e-6)


    # Verify relu6 - Test ReLU6
    x = np.array([[-2.0, -1.0, 0.0, 1.0, 2.0]])  # Using x from source (result_8 uses x)
    expected = np.minimum(np.maximum(x, 0), 6)
    actual = np.array(result.outputs['result_8'])
    np.testing.assert_allclose(actual, expected, rtol=1e-6)


    # Verify swish - Test Swish/SiLU
    x = np.array([[-2.0, -1.0, 0.0, 1.0, 2.0]])  # Using x from source (result_9 uses x)
    expected = x * (1.0 / (1.0 + np.exp(-x)))
    actual = np.array(result.outputs['result_9'])
    np.testing.assert_allclose(actual, expected, rtol=1e-5)


    # Verify hardtanh - Test HardTanh
    x = np.array([[-2.0, -1.0, 0.0, 1.0, 2.0]])  # Using x from source (result_10 uses x)
    expected = np.clip(x, -1.0, 1.0)
    actual = np.array(result.outputs['result_10'])
    np.testing.assert_allclose(actual, expected, rtol=1e-6)


    # Verify prelu - Test PReLU
    x = np.array([[-2.0, -1.0, 0.0, 1.0, 2.0]])  # Using x from source (result_11 uses x)
    alpha = 0.1  # Using alpha from source (result_11 uses alpha, not alpha_2)
    expected = np.where(x > 0, x, alpha * x)
    actual = np.array(result.outputs['result_11'])
    np.testing.assert_allclose(actual, expected, rtol=1e-6)


    # Verify celu - Test CELU
    x_8 = np.array([[-2.0, -1.0, 0.0, 1.0]])  # Using x_8 from source (result_12 uses x_8)
    alpha_3 = 1.0  # Using alpha_3 from source (result_12 uses alpha_3)
    expected = np.where(x_8 > 0, x_8, alpha_3 * (np.exp(x_8 / alpha_3) - 1.0))
    actual = np.array(result.outputs['result_12'])
    np.testing.assert_allclose(actual, expected, rtol=1e-5)


    # Verify softsign - Test Softsign
    x = np.array([[-2.0, -1.0, 0.0, 1.0, 2.0]])
    expected = x / (1.0 + np.abs(x))
    actual = np.array(result.outputs['result_13'])
    np.testing.assert_allclose(actual, expected, rtol=1e-6)


    # Verify hardswish - Test HardSwish
    x = np.array([[-4.0, -1.0, 0.0, 1.0, 4.0]])
    expected = x * np.clip(x + 3.0, 0.0, 6.0) / 6.0
    actual = np.array(result.outputs['result_14'])
    np.testing.assert_allclose(actual, expected, rtol=1e-5)


    # Verify hardsigmoid - Test HardSigmoid
    x_10 = np.array([[-4.0, -1.0, 0.0, 1.0, 4.0]])  # Using x_10 from source (result_15 uses x_10)
    expected = np.clip(x_10 / 6 + 0.5, 0, 1)
    actual = np.array(result.outputs['result_15'])
    np.testing.assert_allclose(actual, expected, rtol=1e-6)


    # Verify log_softmax - Test Log Softmax
    x_11 = np.array([[1.0, 2.0, 3.0]])  # Using x_11 from source (result_16 uses x_11)
    max_x = np.max(x_11, axis=-1, keepdims=True)
    log_sum_exp = max_x + np.log(np.sum(np.exp(x_11 - max_x), axis=-1, keepdims=True))
    expected = x_11 - log_sum_exp
    actual = np.array(result.outputs['result_16'])
    np.testing.assert_allclose(actual, expected, rtol=1e-5)


    # Verify softshrink - Test Softshrink
    x = np.array([[-2.0, -0.5, 0.0, 0.5, 2.0]])
    lambda_val = result.outputs.get('lambda', 1.0)
    expected = np.where(x > lambda_val, x - lambda_val,
                        np.where(x < -lambda_val, x + lambda_val, 0.0))
    actual = np.array(result.outputs['result_17'])
    np.testing.assert_allclose(actual, expected, rtol=1e-6)


    # Verify hardshrink - Test Hardshrink
    x = np.array([[-2.0, -0.5, 0.0, 0.5, 2.0]])
    lambda_val = result.outputs.get('lambda', 1.0)
    expected = np.where(np.abs(x) > lambda_val, x, 0.0)
    actual = np.array(result.outputs['result_18'])
    np.testing.assert_allclose(actual, expected, rtol=1e-6)


    # Verify threshold - Test Threshold
    x = np.array([[-1.0, 0.0, 1.0, 2.0]])
    expected = np.where(x <= 0.5, 0.0, x)
    actual = np.array(result.outputs['result_19'])
    np.testing.assert_allclose(actual, expected, rtol=1e-6)


    # Verify tanhshrink - Test Tanhshrink
    x_15 = np.array([[-1.0, 0.0, 1.0]])  # Using x_15 from source
    expected = x_15 - np.tanh(x_15)
    actual = np.array(result.outputs['result_20'])
    np.testing.assert_allclose(actual, expected, rtol=1e-6)


    # Verify thresholded_relu - Test Thresholded ReLU
    x_16 = np.array([[-1.0, 0.0, 0.5, 1.0, 2.0]])  # Using x_16 from source (result_21 uses x_16)
    alpha_4 = 0.5  # Using alpha_4 from source (result_21 uses alpha_4)
    expected = np.where(x_16 > alpha_4, x_16, 0.0)
    actual = np.array(result.outputs['result_21'])
    np.testing.assert_allclose(actual, expected, rtol=1e-6)


    # Verify softmax_cross_entropy_loss - Test Softmax Cross Entropy Loss (combined op)
    pred = np.array([[1.0, 2.0, 3.0]])
    target = np.array([[0.0, 0.0, 1.0]])
    max_pred = np.max(pred, axis=-1, keepdims=True)
    log_sum_exp = max_pred + np.log(np.sum(np.exp(pred - max_pred), axis=-1, keepdims=True))
    expected = -np.sum(target * (pred - log_sum_exp), axis=-1)
    actual = np.array(result.outputs['result_22'])
    np.testing.assert_allclose(actual, expected, rtol=1e-5)


    # Verify mish - Test Mish activation
    x_17 = np.array([[-1.0, 0.0, 1.0, 2.0]], dtype=np.float32)  # Using x_17 from source
    softplus = np.log(1.0 + np.exp(x_17))
    expected = x_17 * np.tanh(softplus)
    actual = np.array(result.outputs['result_23'])
    np.testing.assert_allclose(actual, expected, rtol=1e-5)


    # Verify elu_alpha - Test ELU with custom alpha
    x_elu_alpha = np.array([[-2.0, -1.0, 0.0, 1.0, 2.0]], dtype=np.float32)  # Using x_elu_alpha from source
    alpha_elu_alpha = 0.5  # Using alpha_elu_alpha from source
    expected = np.where(x_elu_alpha > 0, x_elu_alpha, alpha_elu_alpha * (np.exp(x_elu_alpha) - 1.0))
    actual = np.array(result.outputs['result_24'])
    np.testing.assert_allclose(actual, expected, rtol=1e-5)


    # Verify gelu_tanh - Test GELU with tanh approximation
    x_gelu_tanh = np.array([[-2.0, -1.0, 0.0, 1.0, 2.0]], dtype=np.float32)  # Using x_gelu_tanh from source
    # GELU tanh: x * 0.5 * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
    sqrt_2_over_pi = 0.7978845608028654
    coeff = 0.044715
    inner = sqrt_2_over_pi * (x_gelu_tanh + coeff * x_gelu_tanh**3)
    expected = x_gelu_tanh * 0.5 * (1.0 + np.tanh(inner))
    actual = np.array(result.outputs['result_25'])
    np.testing.assert_allclose(actual, expected, rtol=1e-5)