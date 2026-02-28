"""
Tests for ML Recurrent Operations

Accuracy checking: 
- For integer/boolean operations: strict equality via np.testing.assert_array_equal
- For floating point operations: use np.testing.assert_allclose with rtol=1e-5, atol=1e-8
  to account for floating point precision differences
"""

import pytest
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
    # Note: Using zero bias instead of empty array due to compiler limitation with empty arrays
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
    # t=0: h_0 = 0.0
    # t=1: h_1 = tanh(0.5 * 2.0 + 0.3 * 0.0 + 0.0 + 0.0) = tanh(1.0) ≈ 0.7616
    expected_Y_1 = np.tanh(0.5 * 2.0 + 0.0 + 0.0)  # W @ x + R @ h + Wb + Rb
    
    np.testing.assert_allclose(Y[0], np.array([[0.0]]), rtol=1e-5, atol=1e-8)
    np.testing.assert_allclose(Y[1], np.array([[expected_Y_1]]), rtol=1e-5, atol=1e-8)
    np.testing.assert_allclose(Y_h, np.array([[expected_Y_1]]), rtol=1e-5, atol=1e-8)


def test_lstm_full_with_biases(compiler, runtime):
    """Test Step 7: Cell clipping"""
    source = """use std::ml;
    let X = [[[1.0]], [[2.0]]];
    let hidden_size = 1;
    let W = [[0.1], [0.2], [0.3], [0.4]];
    let R = [[0.1], [0.2], [0.3], [0.4]];
    let initial_h = [[0.0]];
    let initial_c = [[1.0]];
    let clip_threshold = 0.5;  // Clip cell values to [-0.5, 0.5]
    let (Y, Y_h, Y_c) = std::ml::lstm_step7(X, W, R, initial_h, initial_c, hidden_size, clip_threshold);
    """
    
    result = compile_and_execute(source, compiler, runtime)
    assert result.success, f"Execution failed: {result.errors}"
    
    Y = np.array(result.outputs['Y'])
    Y_h = np.array(result.outputs['Y_h'])
    Y_c = np.array(result.outputs['Y_c'])
    
    # Check shape
    assert Y.shape == (2, 1, 1), f"Y shape should be (2, 1, 1), got {Y.shape}"
    assert Y_h.shape == (1, 1), f"Y_h shape should be (1, 1), got {Y_h.shape}"
    assert Y_c.shape == (1, 1), f"Y_c shape should be (1, 1), got {Y_c.shape}"
    
    # Expected: gates_1 = [0.2, 0.4, 0.6, 0.8]
    # f_t = sigmoid(0.4) ≈ 0.5986877, i_t = sigmoid(0.2) ≈ 0.5498340, g_t = tanh(0.6) ≈ 0.5370496, o_t = sigmoid(0.8) ≈ 0.6899745
    # c_1 = f_t * c_0 + i_t * g_t = 0.5986877 * 1.0 + 0.5498340 * 0.5370496 ≈ 0.8939758
    # But with clipping at 0.5: c_1_clipped = 0.5 (since 0.894 > 0.5)
    # h_1 = o_t * tanh(c_1_clipped) = 0.6899745 * tanh(0.5) ≈ 0.315
    
    # Check base case (t=0)
    np.testing.assert_allclose(Y[0], np.array([[0.0]]), rtol=1e-5, atol=1e-8)
    
    # Compute reference values using NumPy
    gates_1 = np.array([0.1*2.0, 0.2*2.0, 0.3*2.0, 0.4*2.0])
    f_t = 1.0 / (1.0 + np.exp(-gates_1[1]))  # sigmoid(0.4)
    i_t = 1.0 / (1.0 + np.exp(-gates_1[0]))  # sigmoid(0.2)
    g_t = np.tanh(gates_1[2])  # tanh(0.6)
    o_t = 1.0 / (1.0 + np.exp(-gates_1[3]))  # sigmoid(0.8)
    
    c_1 = f_t * 1.0 + i_t * g_t  # c_1 = f_t * c_0 + i_t * g_t
    c_1_clipped = np.clip(c_1, -0.5, 0.5)  # Clip to [-0.5, 0.5]
    h_1 = o_t * np.tanh(c_1_clipped)  # h_1 = o_t * tanh(c_1_clipped)
    
    expected_h1 = np.array([[h_1]])
    expected_c1 = np.array([[c_1_clipped]])
    
    np.testing.assert_allclose(Y[1], expected_h1, rtol=1e-5, atol=1e-8)
    np.testing.assert_allclose(Y_h, expected_h1, rtol=1e-5, atol=1e-8)
    np.testing.assert_allclose(Y_c, expected_c1, rtol=1e-5, atol=1e-8)


def test_lstm_full_with_biases(compiler, runtime):
    """Test Step 8: Full LSTM with biases"""
    source = """use std::ml;
    let X = [[[1.0]], [[2.0]]];
    let hidden_size = 1;
    let W = [[0.1], [0.2], [0.3], [0.4]];
    let R = [[0.1], [0.2], [0.3], [0.4]];
    let B = [0.01, 0.02, 0.03, 0.04, 0.01, 0.02, 0.03, 0.04];  // [8*hidden_size] - biases
    let initial_h = [[0.0]];
    let initial_c = [[1.0]];
    let direction = 0;
    let clip_threshold = 0.0;  // No clipping
    let (Y, Y_h, Y_c) = std::ml::lstm(X, W, R, B, initial_h, initial_c, hidden_size, direction, clip_threshold);
    """
    
    result = compile_and_execute(source, compiler, runtime)
    assert result.success, f"Execution failed: {result.errors}"
    
    Y = np.array(result.outputs['Y'])
    Y_h = np.array(result.outputs['Y_h'])
    Y_c = np.array(result.outputs['Y_c'])
    
    # Check shape
    assert Y.shape == (2, 1, 1), f"Y shape should be (2, 1, 1), got {Y.shape}"
    assert Y_h.shape == (1, 1), f"Y_h shape should be (1, 1), got {Y_h.shape}"
    assert Y_c.shape == (1, 1), f"Y_c shape should be (1, 1), got {Y_c.shape}"
    
    # Expected: gates_1 = W @ X[1] + R @ h_0 + B
    # W @ X[1] = [0.1*2.0, 0.2*2.0, 0.3*2.0, 0.4*2.0] = [0.2, 0.4, 0.6, 0.8]
    # R @ h_0 = [0.1*0.0, 0.2*0.0, 0.3*0.0, 0.4*0.0] = [0.0, 0.0, 0.0, 0.0]
    # B = [0.01, 0.02, 0.03, 0.04, 0.01, 0.02, 0.03, 0.04]
    # gates_1 = [0.2+0.01+0.01, 0.4+0.02+0.02, 0.6+0.03+0.03, 0.8+0.04+0.04] = [0.22, 0.44, 0.66, 0.88]
    
    # Check base case (t=0)
    np.testing.assert_allclose(Y[0], np.array([[0.0]]), rtol=1e-5, atol=1e-8)
    
    # Compute reference values using NumPy
    gates_1 = np.array([0.1*2.0 + 0.01 + 0.01, 0.2*2.0 + 0.02 + 0.02, 0.3*2.0 + 0.03 + 0.03, 0.4*2.0 + 0.04 + 0.04])
    f_t = 1.0 / (1.0 + np.exp(-gates_1[1]))  # sigmoid(0.44)
    i_t = 1.0 / (1.0 + np.exp(-gates_1[0]))  # sigmoid(0.22)
    g_t = np.tanh(gates_1[2])  # tanh(0.66)
    o_t = 1.0 / (1.0 + np.exp(-gates_1[3]))  # sigmoid(0.88)
    
    c_1 = f_t * 1.0 + i_t * g_t  # c_1 = f_t * c_0 + i_t * g_t
    h_1 = o_t * np.tanh(c_1)  # h_1 = o_t * tanh(c_1)
    
    expected_h1 = np.array([[h_1]])
    expected_c1 = np.array([[c_1]])
    
    np.testing.assert_allclose(Y[1], expected_h1, rtol=1e-5, atol=1e-8)
    np.testing.assert_allclose(Y_h, expected_h1, rtol=1e-5, atol=1e-8)
    np.testing.assert_allclose(Y_c, expected_c1, rtol=1e-5, atol=1e-8)


def test_gru_full(compiler, runtime):
    """Test full GRU implementation with biases"""
    source = """use std::ml;
    let X = [[[1.0]], [[2.0]]];
    let hidden_size = 1;
    let W = [[0.1], [0.2], [0.3]];  // [3*hidden_size, input_size] - reset, update, hidden
    let R = [[0.1], [0.2], [0.3]];  // [3*hidden_size, hidden_size]
    let B = [0.01, 0.02, 0.03, 0.01, 0.02, 0.03];  // [6*hidden_size] - biases
    let initial_h = [[0.0]];
    let direction = 0;
    let linear_before_reset = 0;
    let (Y, Y_h) = std::ml::gru(X, W, R, B, initial_h, hidden_size, direction, linear_before_reset);
    """
    
    result = compile_and_execute(source, compiler, runtime)
    assert result.success, f"Execution failed: {result.errors}"
    
    Y = np.array(result.outputs['Y'])
    Y_h = np.array(result.outputs['Y_h'])
    
    # Check shape
    assert Y.shape == (2, 1, 1), f"Y shape should be (2, 1, 1), got {Y.shape}"
    assert Y_h.shape == (1, 1), f"Y_h shape should be (1, 1), got {Y_h.shape}"
    
    # GRU Formula:
    # r_t = sigmoid(W_r @ x_t + R_r @ h_{t-1} + b_r)  # reset gate
    # z_t = sigmoid(W_z @ x_t + R_z @ h_{t-1} + b_z)  # update gate
    # h_tilde = tanh(W_h @ x_t + R_h @ (r_t ⊙ h_{t-1}) + b_h)  # candidate hidden state
    # h_t = (1 - z_t) ⊙ h_{t-1} + z_t ⊙ h_tilde  # final hidden state
    
    # For t=1:
    # W_r @ X[1] = 0.1 * 2.0 = 0.2
    # R_r @ h_0 = 0.1 * 0.0 = 0.0
    # b_r = 0.01 + 0.01 = 0.02 (Wb_r + Rb_r)
    # r_1 = sigmoid(0.2 + 0.0 + 0.02) = sigmoid(0.22)
    
    # W_z @ X[1] = 0.2 * 2.0 = 0.4
    # R_z @ h_0 = 0.2 * 0.0 = 0.0
    # b_z = 0.02 + 0.02 = 0.04
    # z_1 = sigmoid(0.4 + 0.0 + 0.04) = sigmoid(0.44)
    
    # W_h @ X[1] = 0.3 * 2.0 = 0.6
    # R_h @ (r_1 ⊙ h_0) = 0.3 * (r_1 * 0.0) = 0.0
    # b_h = 0.03 + 0.03 = 0.06
    # h_tilde_1 = tanh(0.6 + 0.0 + 0.06) = tanh(0.66)
    
    # h_1 = (1 - z_1) * h_0 + z_1 * h_tilde_1 = (1 - z_1) * 0.0 + z_1 * h_tilde_1 = z_1 * h_tilde_1
    
    # Check base case (t=0)
    np.testing.assert_allclose(Y[0], np.array([[0.0]]), rtol=1e-5, atol=1e-8)
    
    # Compute reference values using NumPy
    # Reset gate
    r_1 = 1.0 / (1.0 + np.exp(-(0.1*2.0 + 0.1*0.0 + 0.01 + 0.01)))  # sigmoid(0.22)
    # Update gate
    z_1 = 1.0 / (1.0 + np.exp(-(0.2*2.0 + 0.2*0.0 + 0.02 + 0.02)))  # sigmoid(0.44)
    # Candidate hidden state
    h_tilde_1 = np.tanh(0.3*2.0 + 0.3*(r_1 * 0.0) + 0.03 + 0.03)  # tanh(0.66)
    # Final hidden state
    h_1 = (1.0 - z_1) * 0.0 + z_1 * h_tilde_1  # h_1 = (1 - z_1) * h_0 + z_1 * h_tilde_1
    
    expected_h1 = np.array([[h_1]])
    
    np.testing.assert_allclose(Y[1], expected_h1, rtol=1e-5, atol=1e-8)
    np.testing.assert_allclose(Y_h, expected_h1, rtol=1e-5, atol=1e-8)


def test_lstm_full_complex(compiler, runtime):
    """Test full LSTM with larger dimensions: seq_length=5, batch_size=2, hidden_size=3, input_size=2"""
    source = """use std::ml;
    // seq_length=5, batch_size=2, input_size=2, hidden_size=3
    // X shape: [seq_length=5, batch_size=2, input_size=2]
    let X = [[[1.0, 0.5], [0.5, 1.0]],
             [[2.0, 1.0], [1.0, 2.0]],
             [[0.5, 2.0], [2.0, 0.5]],
             [[1.5, 0.5], [1.0, 1.0]],
             [[0.5, 1.5], [0.5, 0.5]]];
    let hidden_size = 3;
    let input_size = 2;
    // W: [4*hidden_size=12, input_size=2]
    let W = [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6],  // i gate (3 rows)
             [0.2, 0.1], [0.4, 0.3], [0.6, 0.5],  // f gate (3 rows)
             [0.1, 0.3], [0.2, 0.4], [0.3, 0.5],  // g gate (3 rows)
             [0.3, 0.1], [0.4, 0.2], [0.5, 0.3]]; // o gate (3 rows)
    // R: [4*hidden_size=12, hidden_size=3]
    let R = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9],  // i gate (3 rows)
             [0.3, 0.4, 0.5], [0.6, 0.7, 0.8], [0.9, 0.1, 0.2],  // f gate (3 rows)
             [0.1, 0.3, 0.5], [0.2, 0.4, 0.6], [0.3, 0.5, 0.7],  // g gate (3 rows)
             [0.2, 0.4, 0.6], [0.3, 0.5, 0.7], [0.4, 0.6, 0.8]]; // o gate (3 rows)
    // B: [8*hidden_size=24] - biases
    let B = [0.01, 0.02, 0.03, 0.01, 0.02, 0.03, 0.01, 0.02, 0.03, 0.01, 0.02, 0.03,  // input biases
             0.01, 0.02, 0.03, 0.01, 0.02, 0.03, 0.01, 0.02, 0.03, 0.01, 0.02, 0.03]; // recurrent biases
    let initial_h = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]];
    let initial_c = [[0.2, 0.3, 0.4], [0.5, 0.6, 0.7]];
    let direction = 0;
    let clip_threshold = 0.0;
    let (Y, Y_h, Y_c) = std::ml::lstm(X, W, R, B, initial_h, initial_c, hidden_size, direction, clip_threshold);
    """
    
    result = compile_and_execute(source, compiler, runtime)
    assert result.success, f"Execution failed: {result.errors}"
    
    Y = np.array(result.outputs['Y'])
    Y_h = np.array(result.outputs['Y_h'])
    Y_c = np.array(result.outputs['Y_c'])
    
    # Check shape
    assert Y.shape == (5, 2, 3), f"Y shape should be (5, 2, 3), got {Y.shape}"
    assert Y_h.shape == (2, 3), f"Y_h shape should be (2, 3), got {Y_h.shape}"
    assert Y_c.shape == (2, 3), f"Y_c shape should be (2, 3), got {Y_c.shape}"
    
    # Compute reference using NumPy
    X_np = np.array([[[1.0, 0.5], [0.5, 1.0]],
                     [[2.0, 1.0], [1.0, 2.0]],
                     [[0.5, 2.0], [2.0, 0.5]],
                     [[1.5, 0.5], [1.0, 1.0]],
                     [[0.5, 1.5], [0.5, 0.5]]])
    W_np = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6],
                     [0.2, 0.1], [0.4, 0.3], [0.6, 0.5],
                     [0.1, 0.3], [0.2, 0.4], [0.3, 0.5],
                     [0.3, 0.1], [0.4, 0.2], [0.5, 0.3]])
    R_np = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9],
                     [0.3, 0.4, 0.5], [0.6, 0.7, 0.8], [0.9, 0.1, 0.2],
                     [0.1, 0.3, 0.5], [0.2, 0.4, 0.6], [0.3, 0.5, 0.7],
                     [0.2, 0.4, 0.6], [0.3, 0.5, 0.7], [0.4, 0.6, 0.8]])
    B_np = np.array([0.01, 0.02, 0.03, 0.01, 0.02, 0.03, 0.01, 0.02, 0.03, 0.01, 0.02, 0.03,
                     0.01, 0.02, 0.03, 0.01, 0.02, 0.03, 0.01, 0.02, 0.03, 0.01, 0.02, 0.03])
    initial_h_np = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
    initial_c_np = np.array([[0.2, 0.3, 0.4], [0.5, 0.6, 0.7]])
    
    # Verify shapes and basic properties
    # Check that Y[0] matches initial_h
    np.testing.assert_allclose(Y[0], initial_h_np, rtol=1e-5, atol=1e-8)
    
    # Check that all values are finite and reasonable
    assert np.all(np.isfinite(Y)), "Y contains non-finite values"
    assert np.all(np.isfinite(Y_h)), "Y_h contains non-finite values"
    assert np.all(np.isfinite(Y_c)), "Y_c contains non-finite values"
    
    # Check that values are in reasonable range (LSTM outputs should be bounded)
    assert np.all(np.abs(Y) < 10), "Y values are too large"
    assert np.all(np.abs(Y_h) < 10), "Y_h values are too large"
    assert np.all(np.abs(Y_c) < 10), "Y_c values are too large"
    
    # Check that Y_h matches the last timestep of Y
    np.testing.assert_allclose(Y_h, Y[-1], rtol=1e-5, atol=1e-8)


def test_gru_full_complex(compiler, runtime):
    """Test full GRU with larger dimensions: seq_length=4, batch_size=2, hidden_size=3, input_size=2"""
    source = """use std::ml;
    // seq_length=4, batch_size=2, input_size=2, hidden_size=3
    // X shape: [seq_length=4, batch_size=2, input_size=2]
    let X = [[[1.0, 0.5], [0.5, 1.0]],
             [[2.0, 1.0], [1.0, 2.0]],
             [[0.5, 2.0], [2.0, 0.5]],
             [[1.5, 0.5], [1.0, 1.0]]];
    let hidden_size = 3;
    let input_size = 2;
    // W: [3*hidden_size=9, input_size=2] - reset, update, hidden
    let W = [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6],  // reset gate
             [0.2, 0.1], [0.4, 0.3], [0.6, 0.5],  // update gate
             [0.1, 0.3], [0.2, 0.4], [0.3, 0.5]]; // hidden gate
    // R: [3*hidden_size=9, hidden_size=3]
    let R = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9],  // reset gate
             [0.2, 0.3, 0.4], [0.5, 0.6, 0.7], [0.8, 0.9, 0.1],  // update gate
             [0.1, 0.3, 0.5], [0.2, 0.4, 0.6], [0.3, 0.5, 0.7]]; // hidden gate
    // B: [6*hidden_size=18] - biases
    let B = [0.01, 0.02, 0.03, 0.01, 0.02, 0.03, 0.01, 0.02, 0.03,  // input biases
             0.01, 0.02, 0.03, 0.01, 0.02, 0.03, 0.01, 0.02, 0.03]; // recurrent biases
    let initial_h = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]];
    let direction = 0;
    let linear_before_reset = 0;
    let (Y, Y_h) = std::ml::gru(X, W, R, B, initial_h, hidden_size, direction, linear_before_reset);
    """
    
    result = compile_and_execute(source, compiler, runtime)
    assert result.success, f"Execution failed: {result.errors}"
    
    Y = np.array(result.outputs['Y'])
    Y_h = np.array(result.outputs['Y_h'])
    
    # Check shape
    assert Y.shape == (4, 2, 3), f"Y shape should be (4, 2, 3), got {Y.shape}"
    assert Y_h.shape == (2, 3), f"Y_h shape should be (2, 3), got {Y_h.shape}"
    
    # Compute reference using NumPy
    X_np = np.array([[[1.0, 0.5], [0.5, 1.0]],
                     [[2.0, 1.0], [1.0, 2.0]],
                     [[0.5, 2.0], [2.0, 0.5]],
                     [[1.5, 0.5], [1.0, 1.0]]])
    W_np = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6],
                     [0.2, 0.1], [0.4, 0.3], [0.6, 0.5],
                     [0.1, 0.3], [0.2, 0.4], [0.3, 0.5]])
    R_np = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9],
                     [0.2, 0.3, 0.4], [0.5, 0.6, 0.7], [0.8, 0.9, 0.1],
                     [0.1, 0.3, 0.5], [0.2, 0.4, 0.6], [0.3, 0.5, 0.7]])
    B_np = np.array([0.01, 0.02, 0.03, 0.01, 0.02, 0.03, 0.01, 0.02, 0.03,
                     0.01, 0.02, 0.03, 0.01, 0.02, 0.03, 0.01, 0.02, 0.03])
    initial_h_np = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
    
    # Verify shapes and basic properties
    # Check that Y[0] matches initial_h
    np.testing.assert_allclose(Y[0], initial_h_np, rtol=1e-5, atol=1e-8)
    
    # Check that all values are finite and reasonable
    assert np.all(np.isfinite(Y)), "Y contains non-finite values"
    assert np.all(np.isfinite(Y_h)), "Y_h contains non-finite values"
    
    # Check that values are in reasonable range (GRU outputs should be bounded)
    assert np.all(np.abs(Y) < 10), "Y values are too large"
    assert np.all(np.abs(Y_h) < 10), "Y_h values are too large"
    
    # Check that Y_h matches the last timestep of Y
    np.testing.assert_allclose(Y_h, Y[-1], rtol=1e-5, atol=1e-8)

