"""
Tests for ML LSTM operations. Split from test_ml_recurrent.
"""
import numpy as np
from tests.test_utils import compile_and_execute


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
