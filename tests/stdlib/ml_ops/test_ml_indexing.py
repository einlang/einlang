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
# Indexing Operation Tests
# Clustered tests for efficiency - all indexing ops tested together
# Compile/execute errors are failures, not skips
# ===========================================================================

def test_indexing_clustered_accuracy(compiler, runtime):
    """Test indexing operations - clustered for efficiency"""
    source = """use std::ml;
    let x = [[1.0, 5.0, 3.0, 2.0]];
    let x_1 = [[5.0, 1.0, 3.0, 2.0]];
    let data_1d = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
    let data_2d = [[0.0, 1.0, 2.0, 3.0, 4.0, 5.0]];
    let data_2d_1 = [[10.0, 20.0, 30.0, 40.0]];
    let indices_1d = [3, 1, 0];
    let data = [[10.0, 20.0, 30.0, 40.0]];
    let indices = [[3, 1, 0, 2]];
    let indices_1 = [0, 1, 2];
    let depth = 3;
    let x_argmax = [[1.0, 5.0, 3.0, 2.0]];
    let x_argmin = [[5.0, 1.0, 3.0, 2.0]];
    let data_gather_nd = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
    let indices_gather_nd = [[0, 1], [1, 2]];
    let data_gather_nd_3d = [[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]];
    let indices_gather_nd_3d = [[0, 1, 2], [1, 0, 1]];
    let data_scatter = [[1.0, 2.0, 3.0]];
    let indices_scatter = [[0, 1, 0]];
    let updates_scatter = [[10.0, 20.0, 30.0]];
    let data_scatter_1d = [1.0, 2.0, 3.0, 4.0];
    let indices_scatter_1d = [2, 0];
    let updates_scatter_1d = [10.0, 20.0];
    let data_scatter_3d = [[[1.0, 2.0], [3.0, 4.0]]];
    let indices_scatter_3d = [[[1, 0], [0, 1]]];
    let updates_scatter_3d = [[[10.0, 20.0], [30.0, 40.0]]];
    let data_scatter_nd = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
    let indices_scatter_nd = [[0, 1], [1, 2]];
    let updates_scatter_nd = [100.0, 200.0];
    let data_scatter_nd_3d = [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]];
    let indices_scatter_nd_3d = [[0, 1, 1], [1, 0, 0]];
    let updates_scatter_nd_3d = [100.0, 200.0];
    let x_nonzero_1d = [0.0, 1.0, 0.0, 2.0, 0.0];
    let data_nonzero_2d = [[0.0, 1.0, 0.0], [2.0, 0.0, 3.0]];
    let data_nonzero_3d = [[[0.0, 1.0], [2.0, 0.0]], [[0.0, 0.0], [3.0, 4.0]]];
    let result_1d_2 = std::ml::slice(data_1d, [1], [5], [0], [2]);
    let result_2d_single_2 = std::ml::slice(data_2d, [1], [5], [1], [2]);
    let result_axis1_3 = std::ml::gather(data_2d, indices_1d, 1);
    let result_4 = std::ml::gather_elements(data, indices, 1);
    let result_5 = std::ml::onehot(indices_1, depth, [0.0, 1.0]);
    let result_6 = std::ml::argmax(x_argmax);
    let result_7 = std::ml::argmin(x_argmin);
    let data_1d_gather = [1.0, 2.0, 3.0, 4.0, 5.0];
    let indices_1d_gather = [2, 0, 4];
    let result_gather_1d = std::ml::gather(data_1d_gather, indices_1d_gather, 0);
    let data_3d_gather = [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]];
    let indices_3d_gather = [0, 1];
    let result_gather_3d = std::ml::gather(data_3d_gather, indices_3d_gather, 0);
    let data_1d_gather_elements = [1.0, 2.0, 3.0, 4.0];
    let indices_1d_gather_elements = [2, 0, 3, 1];
    let result_gather_elements_1d = std::ml::gather_elements(data_1d_gather_elements, indices_1d_gather_elements, 0);
    let data_3d_gather_elements = [[[1.0, 2.0], [3.0, 4.0]]];
    let indices_3d_gather_elements = [[[1, 0], [0, 1]]];
    let result_gather_elements_3d = std::ml::gather_elements(data_3d_gather_elements, indices_3d_gather_elements, 2);
    let result_8 = std::ml::gather_nd(data_gather_nd, indices_gather_nd);
    let result_8_3d = std::ml::gather_nd(data_gather_nd_3d, indices_gather_nd_3d);
    let result_9 = std::ml::scatter_elements(data_scatter, indices_scatter, updates_scatter, 1);
    let result_10 = std::ml::scatter(data_scatter, indices_scatter, updates_scatter, 1);
    let result_scatter_1d = std::ml::scatter(data_scatter_1d, indices_scatter_1d, updates_scatter_1d, 0);
    let result_scatter_3d = std::ml::scatter(data_scatter_3d, indices_scatter_3d, updates_scatter_3d, 2);
    let result_11 = std::ml::scatter_nd(data_scatter_nd, indices_scatter_nd, updates_scatter_nd);
    let result_11_3d = std::ml::scatter_nd(data_scatter_nd_3d, indices_scatter_nd_3d, updates_scatter_nd_3d);
    let result_12_nonzero_1d = std::ml::nonzero(x_nonzero_1d);
    let result_13_nonzero_2d = std::ml::nonzero(data_nonzero_2d);
    let result_14_nonzero_3d = std::ml::nonzero(data_nonzero_3d);
    """

    result = compile_and_execute(source, compiler, runtime)
    assert result.success, f"Execution failed: {result.errors}"

    # Verify each operation


    # Verify slice - Test Slice 1D
    data_1d = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    expected_1d = data_1d[1:5:2]
    actual_1d = np.array(result.outputs['result_1d_2'])
    np.testing.assert_allclose(actual_1d, expected_1d, rtol=1e-6)


    # Verify slice - Test Slice 2D
    data_2d = np.array([[0.0, 1.0, 2.0, 3.0, 4.0, 5.0]])
    expected_2d_single = data_2d[:, 1:5:2]
    actual_2d_single = np.array(result.outputs['result_2d_single_2'])
    np.testing.assert_allclose(actual_2d_single, expected_2d_single, rtol=1e-6)


    # Verify gather 1D - Test ONNX Gather operator (1D)
    data_1d_gather = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    indices_1d_gather = np.array([2, 0, 4], dtype=np.int32)
    expected_1d = np.take(data_1d_gather, indices_1d_gather, axis=0)
    actual_1d = np.array(result.outputs['result_gather_1d'])
    np.testing.assert_allclose(actual_1d, expected_1d, rtol=1e-6)

    # Verify gather 2D - Test ONNX Gather operator with different axes
    data_2d = np.array([[0.0, 1.0, 2.0, 3.0, 4.0, 5.0]])  # Using data_2d from source
    indices_1d = np.array([3, 1, 0])
    expected = np.take(data_2d, indices_1d, axis=1)
    actual = np.array(result.outputs['result_axis1_3'])
    np.testing.assert_allclose(actual, expected, rtol=1e-6)

    # Verify gather 3D - Test ONNX Gather operator (3D)
    data_3d_gather = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], dtype=np.float32)
    indices_3d_gather = np.array([0, 1], dtype=np.int32)
    expected_3d = np.take(data_3d_gather, indices_3d_gather, axis=0)
    actual_3d = np.array(result.outputs['result_gather_3d'])
    np.testing.assert_allclose(actual_3d, expected_3d, rtol=1e-6)


    # Verify gather_elements 1D - Test gather_elements operation (1D)
    data_1d_gather_elements = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    indices_1d_gather_elements = np.array([2, 0, 3, 1], dtype=np.int32)
    expected_1d = np.take_along_axis(data_1d_gather_elements.reshape(1, -1), indices_1d_gather_elements.reshape(1, -1), axis=1).flatten()
    actual_1d = np.array(result.outputs['result_gather_elements_1d'])
    np.testing.assert_allclose(actual_1d, expected_1d, rtol=1e-6)

    # Verify gather_elements 2D - Test gather_elements operation (2D)
    data = np.array([[10.0, 20.0, 30.0, 40.0]], dtype=np.float32)
    indices = np.array([[3, 1, 0, 2]], dtype=np.int32)
    expected = np.take_along_axis(data, indices, axis=1)
    actual = np.array(result.outputs['result_4'])
    np.testing.assert_allclose(actual, expected, rtol=1e-6)

    # Verify gather_elements 3D - Test gather_elements operation (3D)
    data_3d_gather_elements = np.array([[[1.0, 2.0], [3.0, 4.0]]], dtype=np.float32)
    indices_3d_gather_elements = np.array([[[1, 0], [0, 1]]], dtype=np.int32)
    expected_3d = np.take_along_axis(data_3d_gather_elements, indices_3d_gather_elements, axis=2)
    actual_3d = np.array(result.outputs['result_gather_elements_3d'])
    np.testing.assert_allclose(actual_3d, expected_3d, rtol=1e-6)


    # Verify onehot - Test onehot operation
    indices = np.array([0, 1, 2], dtype=np.int32)  # Using indices_1 from source
    depth = result.outputs.get('depth', 3)
    expected = np.eye(int(depth), dtype=np.float32)[indices]
    actual = np.array(result.outputs['result_5'])
    np.testing.assert_allclose(actual, expected, rtol=1e-6)


    # Verify argmax - Test argmax operation
    x_argmax = np.array([[1.0, 5.0, 3.0, 2.0]], dtype=np.float32)  # Using x_argmax from source
    expected = np.argmax(x_argmax, axis=-1)  # Index of max (5.0) is 1
    actual = np.array(result.outputs['result_6'])
    np.testing.assert_array_equal(actual, expected)


    # Verify argmin - Test argmin operation
    x_argmin = np.array([[5.0, 1.0, 3.0, 2.0]], dtype=np.float32)  # Using x_argmin from source
    expected = np.argmin(x_argmin, axis=-1)  # Index of min (1.0) is 1
    actual = np.array(result.outputs['result_7'])
    np.testing.assert_array_equal(actual, expected)


    # Verify gather_nd 2D - Test gather_nd operation (2D)
    data_gather_nd = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)  # Using data_gather_nd from source
    indices_gather_nd = np.array([[0, 1], [1, 2]], dtype=np.int32)  # Using indices_gather_nd from source
    expected = np.array([data_gather_nd[0, 1], data_gather_nd[1, 2]])  # [2.0, 6.0]
    actual = np.array(result.outputs['result_8'])
    np.testing.assert_allclose(actual, expected, rtol=1e-6)

    # Verify gather_nd 3D - Test gather_nd operation (3D)
    data_gather_nd_3d = np.array([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]], dtype=np.float32)
    indices_gather_nd_3d = np.array([[0, 1, 2], [1, 0, 1]], dtype=np.int32)
    expected_3d = np.array([data_gather_nd_3d[0, 1, 2], data_gather_nd_3d[1, 0, 1]])  # [6.0, 8.0]
    actual_3d = np.array(result.outputs['result_8_3d'])
    np.testing.assert_allclose(actual_3d, expected_3d, rtol=1e-6)


    # Verify scatter_elements 2D - Test scatter_elements operation (2D)
    # Note: scatter_elements REPLACES values (last update wins if multiple updates target same index)
    data_scatter = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)  # Using data_scatter from source
    indices_scatter = np.array([[0, 1, 0]], dtype=np.int32)  # Using indices_scatter from source
    updates_scatter = np.array([[10.0, 20.0, 30.0]], dtype=np.float32)  # Using updates_scatter from source
    # Implementation: replaces values, last update wins
    # For index 0: last update is 30.0 (from indices[2]=0, updates[2]=30.0)
    # For index 1: update is 20.0
    # For index 2: no update, keeps original 3.0
    expected = np.array([[30.0, 20.0, 3.0]], dtype=np.float32)
    actual = np.array(result.outputs['result_9'])
    np.testing.assert_allclose(actual, expected, rtol=1e-6)



    # Verify scatter 1D - Test scatter operation (1D)
    data_scatter_1d = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    indices_scatter_1d = np.array([2, 0], dtype=np.int32)
    updates_scatter_1d = np.array([10.0, 20.0], dtype=np.float32)
    expected_1d = data_scatter_1d.copy()
    expected_1d[2] = 10.0
    expected_1d[0] = 20.0
    actual_1d = np.array(result.outputs['result_scatter_1d'])
    np.testing.assert_allclose(actual_1d, expected_1d, rtol=1e-6)

    # Verify scatter 2D - Test scatter operation (2D, delegates to scatter_elements)
    data_scatter = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)  # Using data_scatter from source
    indices_scatter = np.array([[0, 1, 0]], dtype=np.int32)  # Using indices_scatter from source
    updates_scatter = np.array([[10.0, 20.0, 30.0]], dtype=np.float32)  # Using updates_scatter from source
    expected = np.array([[30.0, 20.0, 3.0]], dtype=np.float32)  # Same as scatter_elements (replacement, not accumulation)
    actual = np.array(result.outputs['result_10'])
    np.testing.assert_allclose(actual, expected, rtol=1e-6)

    # Verify scatter 3D - Test scatter operation (3D)
    data_scatter_3d = np.array([[[1.0, 2.0], [3.0, 4.0]]], dtype=np.float32)
    indices_scatter_3d = np.array([[[1, 0], [0, 1]]], dtype=np.int32)
    updates_scatter_3d = np.array([[[10.0, 20.0], [30.0, 40.0]]], dtype=np.float32)
    expected_3d = data_scatter_3d.copy()
    expected_3d[0, 0, 1] = 10.0  # indices[0,0,1]=1, updates[0,0,1]=10.0 -> data[0,0,1]=10.0
    expected_3d[0, 0, 0] = 20.0  # indices[0,0,0]=0, updates[0,0,0]=20.0 -> data[0,0,0]=20.0
    expected_3d[0, 1, 0] = 30.0  # indices[0,1,0]=0, updates[0,1,0]=30.0 -> data[0,1,0]=30.0
    expected_3d[0, 1, 1] = 40.0  # indices[0,1,1]=1, updates[0,1,1]=40.0 -> data[0,1,1]=40.0
    actual_3d = np.array(result.outputs['result_scatter_3d'])
    np.testing.assert_allclose(actual_3d, expected_3d, rtol=1e-6)


    # Verify scatter_nd 2D - Test scatter_nd operation (2D)
    # Note: scatter_nd REPLACES values (not accumulates)
    data_scatter_nd = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)  # Using data_scatter_nd from source
    indices_scatter_nd = np.array([[0, 1], [1, 2]], dtype=np.int32)  # Using indices_scatter_nd from source
    updates_scatter_nd = np.array([100.0, 200.0], dtype=np.float32)  # Using updates_scatter_nd from source
    # Implementation: replaces values directly
    # For [0,1]: replace with 100.0
    # For [1,2]: replace with 200.0
    expected = data_scatter_nd.copy()
    expected[0, 1] = 100.0
    expected[1, 2] = 200.0
    actual = np.array(result.outputs['result_11'])
    np.testing.assert_allclose(actual, expected, rtol=1e-6)

    # Verify scatter_nd 3D - Test scatter_nd operation (3D)
    data_scatter_nd_3d = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], dtype=np.float32)
    indices_scatter_nd_3d = np.array([[0, 1, 1], [1, 0, 0]], dtype=np.int32)
    updates_scatter_nd_3d = np.array([100.0, 200.0], dtype=np.float32)
    expected_3d = data_scatter_nd_3d.copy()
    expected_3d[0, 1, 1] = 100.0
    expected_3d[1, 0, 0] = 200.0
    actual_3d = np.array(result.outputs['result_11_3d'])
    np.testing.assert_allclose(actual_3d, expected_3d, rtol=1e-6)


    # Verify nonzero - Test nonzero operation (1D) - moved from test_ml_utility.py
    x_nonzero_1d = np.array([0.0, 1.0, 0.0, 2.0, 0.0], dtype=np.float32)  # Using x_nonzero_1d from source
    expected_1d = np.nonzero(x_nonzero_1d)[0]  # Get indices of non-zero elements: [1, 3]
    actual_1d = np.array(result.outputs['result_12_nonzero_1d'])
    np.testing.assert_array_equal(actual_1d, expected_1d)


    # Verify nonzero - Test nonzero operation (2D) - moved from test_ml_utility.py
    data_nonzero_2d = np.array([[0.0, 1.0, 0.0], [2.0, 0.0, 3.0]], dtype=np.float32)  # Using data_nonzero_2d from source
    nonzero_2d = np.nonzero(data_nonzero_2d)
    expected_2d = list(zip(nonzero_2d[0], nonzero_2d[1]))  # [(0,1), (1,0), (1,2)]
    actual_2d = result.outputs['result_13_nonzero_2d']
    # Convert to list of tuples for comparison
    if isinstance(actual_2d, list):
        actual_tuples = [tuple(item) if isinstance(item, (list, np.ndarray)) else item for item in actual_2d]
        assert len(actual_tuples) == len(expected_2d), f"Expected {len(expected_2d)} tuples, got {len(actual_tuples)}"
        for actual_tuple, expected_tuple in zip(actual_tuples, expected_2d):
            assert actual_tuple == expected_tuple, f"Expected {expected_tuple}, got {actual_tuple}"


    # Verify nonzero - Test nonzero operation (3D) - moved from test_ml_utility.py
    data_nonzero_3d = np.array([[[0.0, 1.0], [2.0, 0.0]], [[0.0, 0.0], [3.0, 4.0]]], dtype=np.float32)  # Using data_nonzero_3d from source
    nonzero_3d = np.nonzero(data_nonzero_3d)
    expected_3d = list(zip(nonzero_3d[0], nonzero_3d[1], nonzero_3d[2]))  # [(0,0,1), (0,1,0), (1,1,0), (1,1,1)]
    actual_3d = result.outputs['result_14_nonzero_3d']
    # Convert to list of tuples for comparison
    if isinstance(actual_3d, list):
        actual_tuples = [tuple(item) if isinstance(item, (list, np.ndarray)) else item for item in actual_3d]
        assert len(actual_tuples) == len(expected_3d), f"Expected {len(expected_3d)} tuples, got {len(actual_tuples)}"
        for actual_tuple, expected_tuple in zip(actual_tuples, expected_3d):
            assert actual_tuple == expected_tuple, f"Expected {expected_tuple}, got {actual_tuple}"