"""
Tests for ML RNN (recurrent) operations. Split from test_ml_recurrent.
"""

import numpy as np
from tests.test_utils import compile_and_execute


def test_rnn_basic(compiler, runtime):
    """Test ONNX RNN operator with Tanh activation"""
    source = """use std::ml;
    // Simple test: seq_length=2, batch_size=1, input_size=2, hidden_size=2
    let X = [[[1.0, 2.0]], [[3.0, 4.0]]];
    let W = [[0.1, 0.2], [0.3, 0.4]];  // [hidden_size, input_size]
    let R = [[0.5, 0.6], [0.7, 0.8]];  // [hidden_size, hidden_size]
    let B = [0.1, 0.2, 0.3, 0.4];  // [2*hidden_size] = [Wb, Rb]
    let initial_h = [[0.0, 0.0]];
    let hidden_size = 2;
    let direction = 0;
    let activation = "Tanh";

    let (Y, Y_h) = std::ml::rnn(X, W, R, B, initial_h, hidden_size, direction, activation);
    """

    result = compile_and_execute(source, compiler, runtime)
    assert result.success, f"Execution failed: {result.errors}"

    Y = np.array(result.outputs['Y'])
    Y_h = np.array(result.outputs['Y_h'])

    # Check shape
    assert Y.shape == (2, 1, 2), f"Y shape should be (2, 1, 2), got {Y.shape}"
    assert Y_h.shape == (1, 2), f"Y_h shape should be (1, 2), got {Y_h.shape}"

    # Compute reference using NumPy
    X_np = np.array([[[1.0, 2.0]], [[3.0, 4.0]]])
    W_np = np.array([[0.1, 0.2], [0.3, 0.4]])
    R_np = np.array([[0.5, 0.6], [0.7, 0.8]])
    B_np = np.array([0.1, 0.2, 0.3, 0.4])
    Wb_np = B_np[:2]  # [0.1, 0.2]
    Rb_np = B_np[2:]  # [0.3, 0.4]
    initial_h_np = np.array([[0.0, 0.0]])

    # t=0: h_0 = initial_h
    h_prev = initial_h_np[0]  # [0.0, 0.0]

    # t=1: h_1 = tanh(W @ x_1 + R @ h_0 + Wb + Rb)
    x_1 = X_np[1, 0]  # [3.0, 4.0]
    z_1 = W_np @ x_1 + R_np @ h_prev + Wb_np + Rb_np
    h_1 = np.tanh(z_1)

    expected_Y_0 = initial_h_np  # Shape: (1, 2)
    expected_Y_1 = h_1.reshape(1, -1)  # Shape: (1, 2)
    expected_Y_h = h_1.reshape(1, -1)  # Shape: (1, 2)

    np.testing.assert_allclose(Y[0], expected_Y_0, rtol=1e-5, atol=1e-8)
    np.testing.assert_allclose(Y[1], expected_Y_1, rtol=1e-5, atol=1e-8)
    np.testing.assert_allclose(Y_h, expected_Y_h, rtol=1e-5, atol=1e-8)


def test_rnn_without_bias(compiler, runtime):
    """Test ONNX RNN operator without bias - using zero bias instead of empty array"""
    source = """use std::ml;
    // seq_length=2, batch_size=1, input_size=1, hidden_size=1
    let X = [[[1.0]], [[2.0]]];
    let W = [[0.5]];  // [hidden_size, input_size]
    let R = [[0.3]];  // [hidden_size, hidden_size]
    let B = [0.0, 0.0];  // Zero bias (equivalent to no bias for hidden_size=1, needs 2*hidden_size)
    let initial_h = [[0.0]];
    let hidden_size = 1;
    let direction = 0;
    let activation = "Tanh";

    let (Y, Y_h) = std::ml::rnn(X, W, R, B, initial_h, hidden_size, direction, activation);
    """

    result = compile_and_execute(source, compiler, runtime)
    assert result.success, f"Execution failed: {result.errors}"

    Y = np.array(result.outputs['Y'])
    Y_h = np.array(result.outputs['Y_h'])

    # Check shape
    assert Y.shape == (2, 1, 1), f"Y shape should be (2, 1, 1), got {Y.shape}"
    assert Y_h.shape == (1, 1), f"Y_h shape should be (1, 1), got {Y_h.shape}"

    # Compute reference (with zero bias: B = [0.0, 0.0])
    expected_Y_1 = np.tanh(0.5 * 2.0 + 0.0 + 0.0)

    np.testing.assert_allclose(Y[0], np.array([[0.0]]), rtol=1e-5, atol=1e-8)
    np.testing.assert_allclose(Y[1], np.array([[expected_Y_1]]), rtol=1e-5, atol=1e-8)
    np.testing.assert_allclose(Y_h, np.array([[expected_Y_1]]), rtol=1e-5, atol=1e-8)
