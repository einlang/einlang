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
# Reductions Operation Tests
# Clustered tests for efficiency - all reductions ops tested together
# Compile/execute errors are failures, not skips
# ===========================================================================

def test_reductions_clustered_accuracy(compiler, runtime):
    """Test reductions operations - clustered for efficiency"""
    source = """use std::ml;
    let x = [[1.0, 2.0, 3.0, 4.0]];
    let x_1 = [[1.0, 5.0, 3.0, 2.0]];
    let x_2 = [[1.0, 2.0, 3.0]];
    let x_3 = [[3.0, 4.0]];
    let x_4 = [[5.0, 1.0, 3.0, 2.0]];
    let x_5 = [[-1.0, 2.0, -3.0]];
    let x_6 = [[1.0, 2.0, 3.0]];
    let x_7 = [[1.0, 2.0, 3.0]];
    let x_8 = [[2.0, 3.0, 4.0], [1.0, 2.0, 3.0]];
    let result_0 = std::ml::reduce_sum(x);
    let result_1 = std::ml::reduce_mean(x);
    let result_2 = std::ml::reduce_max(x_1);
    let result_3 = std::ml::reduce_sum_square(x_2);
    let result_4 = std::ml::reduce_l2(x_3);
    let result_5 = std::ml::reduce_min(x_4);
    let result_6 = std::ml::reduce_l1(x_5);
    let result_7 = std::ml::reduce_log_sum(x_6);
    let result_8 = std::ml::reduce_log_sum_exp(x_7);
    let result_9 = std::ml::reduce_prod(x_8);
    """

    result = compile_and_execute(source, compiler, runtime)
    assert result.success, f"Execution failed: {result.errors}"

    # Verify each operation


    # Verify reduce_sum - Test ReduceSum
    x = np.array([[1.0, 2.0, 3.0, 4.0]])
    expected = np.sum(x, axis=-1)
    actual = np.array(result.outputs['result_0'])
    np.testing.assert_allclose(actual, expected, rtol=1e-6)


    # Verify reduce_mean - Test ReduceMean
    x = np.array([[1.0, 2.0, 3.0, 4.0]])
    expected = np.mean(x, axis=-1)
    actual = np.array(result.outputs['result_1'])
    np.testing.assert_allclose(actual, expected, rtol=1e-6)


    # Verify reduce_max - Test ReduceMax
    x_1 = np.array([[1.0, 5.0, 3.0, 2.0]])  # Using x_1 from source
    expected = np.max(x_1, axis=-1)
    actual = np.array(result.outputs['result_2'])
    np.testing.assert_allclose(actual, expected, rtol=1e-6)


    # Verify reduce_sum_square - Test ReduceSumSquare
    x_2 = np.array([[1.0, 2.0, 3.0]])  # Using x_2 from source
    expected = np.sum(x_2**2, axis=-1)
    actual = np.array(result.outputs['result_3'])
    np.testing.assert_allclose(actual, expected, rtol=1e-6)


    # Verify reduce_l2 - Test ReduceL2 (L2 norm)
    x_3 = np.array([[3.0, 4.0]])  # Using x_3 from source
    expected = np.sqrt(np.sum(x_3**2, axis=-1))
    actual = np.array(result.outputs['result_4'])
    np.testing.assert_allclose(actual, expected, rtol=1e-6)
    np.testing.assert_allclose(actual[0], 5.0, rtol=1e-6)


    # Verify reduce_min - Test ReduceMin
    x_4 = np.array([[5.0, 1.0, 3.0, 2.0]])  # Using x_4 from source
    expected = np.min(x_4, axis=-1)
    actual = np.array(result.outputs['result_5'])
    np.testing.assert_allclose(actual, expected, rtol=1e-6)


    # Verify reduce_l1 - Test ReduceL1
    x_5 = np.array([[-1.0, 2.0, -3.0]])  # Using x_5 from source
    expected = np.sum(np.abs(x_5), axis=-1)
    actual = np.array(result.outputs['result_6'])
    np.testing.assert_allclose(actual, expected, rtol=1e-6)


    # Verify reduce_log_sum - Test ReduceLogSum
    x_6 = np.array([[1.0, 2.0, 3.0]])  # Using x_6 from source
    expected = np.log(np.sum(x_6, axis=-1))
    actual = np.array(result.outputs['result_7'])
    np.testing.assert_allclose(actual, expected, rtol=1e-6)


    # Verify reduce_log_sum_exp - Test ReduceLogSumExp (numerically stable)
    x_7 = np.array([[1.0, 2.0, 3.0]])  # Using x_7 from source
    max_x = np.max(x_7, axis=-1, keepdims=True)
    expected = np.squeeze(max_x + np.log(np.sum(np.exp(x_7 - max_x), axis=-1, keepdims=True)))
    actual = np.array(result.outputs['result_8'])
    np.testing.assert_allclose(actual, expected, rtol=1e-5)


    # Verify reduce_prod - Test reduce_prod operation
    x_8 = np.array([[2.0, 3.0, 4.0], [1.0, 2.0, 3.0]], dtype=np.float32)  # Using x_8 from source
    expected = np.prod(x_8, axis=1)  # [24.0, 6.0]
    actual = np.array(result.outputs['result_9'])
    np.testing.assert_allclose(actual, expected, rtol=1e-6)