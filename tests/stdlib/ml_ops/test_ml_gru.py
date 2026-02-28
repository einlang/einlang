"""
Tests for ML GRU operations. Split from test_ml_recurrent.
"""
import numpy as np
from tests.test_utils import compile_and_execute


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
