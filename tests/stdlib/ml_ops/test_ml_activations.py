#!/usr/bin/env python3
"""
Accuracy tests for std::ml activation functions against NumPy reference.
Split into smaller test functions for parallel execution.
"""

import numpy as np
try:
    import scipy.special
except ImportError:
    scipy = None
from ...test_utils import compile_and_execute, assert_float_close


def test_activations_basic(compiler, runtime):
    """Test relu, sigmoid, softmax, gelu, leaky_relu, elu, selu, softplus, relu6,
    swish, hardtanh, prelu, celu."""
    source = """use std::ml;
    let x = [[-2.0, -1.0, 0.0, 1.0, 2.0]];
    let x_2 = [[-1.0, 0.0, 1.0, 2.0]];
    let alpha = 0.1;
    let x_8 = [[-2.0, -1.0, 0.0, 1.0]];
    let alpha_3 = 1.0;
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
    """
    result = compile_and_execute(source, compiler, runtime)
    assert result.success, f"Execution failed: {result.errors}"

    x = np.array([[-2.0, -1.0, 0.0, 1.0, 2.0]])
    np.testing.assert_allclose(np.array(result.outputs['result_0']), np.maximum(0, x), rtol=1e-6)
    np.testing.assert_allclose(np.array(result.outputs['result_1']),
                               1.0 / (1.0 + np.exp(-x)), rtol=1e-6)
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    softmax_ref = exp_x / np.sum(exp_x, axis=-1, keepdims=True)
    np.testing.assert_allclose(np.array(result.outputs['result_2']), softmax_ref, rtol=1e-6)
    assert_float_close(np.sum(np.array(result.outputs['result_2'])[0]), 1.0, rel_tol=1e-6)
    sqrt_2_over_pi = np.sqrt(2.0 / np.pi)
    gelu_ref = 0.5 * x * (1.0 + np.tanh(sqrt_2_over_pi * (x + 0.044715 * x**3)))
    np.testing.assert_allclose(np.array(result.outputs['result_3']), gelu_ref, rtol=1e-5)
    alpha = 0.1
    np.testing.assert_allclose(np.array(result.outputs['result_4']),
                               np.where(x > 0, x, alpha * x), rtol=1e-6)
    x_2 = np.array([[-1.0, 0.0, 1.0, 2.0]])
    np.testing.assert_allclose(np.array(result.outputs['result_5']),
                               np.where(x_2 > 0, x_2, alpha * (np.exp(x_2) - 1.0)), rtol=1e-5)
    lam = 1.0507009873554804934193349852946
    alp = 1.6732632423543772848170429916717
    np.testing.assert_allclose(np.array(result.outputs['result_6']),
                               np.where(x > 0, lam * x, lam * alp * (np.exp(x) - 1.0)), rtol=1e-5)
    np.testing.assert_allclose(np.array(result.outputs['result_7']),
                               np.log(np.exp(x) + 1), rtol=1e-6)
    np.testing.assert_allclose(np.array(result.outputs['result_8']),
                               np.minimum(np.maximum(x, 0), 6), rtol=1e-6)
    np.testing.assert_allclose(np.array(result.outputs['result_9']),
                               x * (1.0 / (1.0 + np.exp(-x))), rtol=1e-5)
    np.testing.assert_allclose(np.array(result.outputs['result_10']), np.clip(x, -1.0, 1.0), rtol=1e-6)
    np.testing.assert_allclose(np.array(result.outputs['result_11']),
                               np.where(x > 0, x, alpha * x), rtol=1e-6)
    x_8 = np.array([[-2.0, -1.0, 0.0, 1.0]])
    np.testing.assert_allclose(np.array(result.outputs['result_12']),
                               np.where(x_8 > 0, x_8, 1.0 * (np.exp(x_8 / 1.0) - 1.0)), rtol=1e-5)


def test_activations_advanced(compiler, runtime):
    """Test softsign, hardswish, hardsigmoid, log_softmax, softshrink, hardshrink,
    threshold, tanhshrink, thresholded_relu, softmax_cross_entropy_loss, mish,
    elu_alpha, gelu_tanh."""
    source = """use std::ml;
    let x = [[-2.0, -1.0, 0.0, 1.0, 2.0]];
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

    x = np.array([[-2.0, -1.0, 0.0, 1.0, 2.0]])
    np.testing.assert_allclose(np.array(result.outputs['result_13']),
                               x / (1.0 + np.abs(x)), rtol=1e-6)
    x_9 = np.array([[-4.0, -1.0, 0.0, 1.0, 4.0]])
    np.testing.assert_allclose(np.array(result.outputs['result_14']),
                               x_9 * np.clip(x_9 + 3.0, 0.0, 6.0) / 6.0, rtol=1e-5)
    x_10 = np.array([[-4.0, -1.0, 0.0, 1.0, 4.0]])
    np.testing.assert_allclose(np.array(result.outputs['result_15']),
                               np.clip(x_10 / 6 + 0.5, 0, 1), rtol=1e-6)
    x_11 = np.array([[1.0, 2.0, 3.0]])
    max_x = np.max(x_11, axis=-1, keepdims=True)
    log_sum_exp = max_x + np.log(np.sum(np.exp(x_11 - max_x), axis=-1, keepdims=True))
    np.testing.assert_allclose(np.array(result.outputs['result_16']),
                               x_11 - log_sum_exp, rtol=1e-5)
    x_12 = np.array([[-2.0, -0.5, 0.0, 0.5, 2.0]])
    lam = 1.0
    np.testing.assert_allclose(np.array(result.outputs['result_17']),
                               np.where(x_12 > lam, x_12 - lam,
                                        np.where(x_12 < -lam, x_12 + lam, 0.0)), rtol=1e-6)
    np.testing.assert_allclose(np.array(result.outputs['result_18']),
                               np.where(np.abs(x_12) > lam, x_12, 0.0), rtol=1e-6)
    x_14 = np.array([[-1.0, 0.0, 1.0, 2.0]])
    np.testing.assert_allclose(np.array(result.outputs['result_19']),
                               np.where(x_14 <= 0.5, 0.0, x_14), rtol=1e-6)
    x_15 = np.array([[-1.0, 0.0, 1.0]])
    np.testing.assert_allclose(np.array(result.outputs['result_20']),
                               x_15 - np.tanh(x_15), rtol=1e-6)
    x_16 = np.array([[-1.0, 0.0, 0.5, 1.0, 2.0]])
    np.testing.assert_allclose(np.array(result.outputs['result_21']),
                               np.where(x_16 > 0.5, x_16, 0.0), rtol=1e-6)
    pred = np.array([[1.0, 2.0, 3.0]])
    target = np.array([[0.0, 0.0, 1.0]])
    max_pred = np.max(pred, axis=-1, keepdims=True)
    log_sum_exp_pred = max_pred + np.log(np.sum(np.exp(pred - max_pred), axis=-1, keepdims=True))
    np.testing.assert_allclose(np.array(result.outputs['result_22']),
                               -np.sum(target * (pred - log_sum_exp_pred), axis=-1), rtol=1e-5)
    x_17 = np.array([[-1.0, 0.0, 1.0, 2.0]], dtype=np.float32)
    softplus = np.log(1.0 + np.exp(x_17))
    np.testing.assert_allclose(np.array(result.outputs['result_23']),
                               x_17 * np.tanh(softplus), rtol=1e-5)
    x_elu = np.array([[-2.0, -1.0, 0.0, 1.0, 2.0]], dtype=np.float32)
    alpha_elu = 0.5
    np.testing.assert_allclose(np.array(result.outputs['result_24']),
                               np.where(x_elu > 0, x_elu, alpha_elu * (np.exp(x_elu) - 1.0)), rtol=1e-5)
    x_gt = np.array([[-2.0, -1.0, 0.0, 1.0, 2.0]], dtype=np.float32)
    c = 0.044715
    sqrt2pi = 0.7978845608028654
    inner = sqrt2pi * (x_gt + c * x_gt**3)
    np.testing.assert_allclose(np.array(result.outputs['result_25']),
                               x_gt * 0.5 * (1.0 + np.tanh(inner)), rtol=1e-5)
