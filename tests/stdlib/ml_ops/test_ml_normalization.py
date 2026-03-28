#!/usr/bin/env python3
"""
Tests for std::ml normalization ops using one grouped module fixture.
"""

import numpy as np
import pytest

from ...test_utils import compile_and_execute


@pytest.fixture(scope="module")
def normalization_outputs(module_compiler, module_runtime):
    source = """use std::ml;
    let input_2d = [[1.0, 2.0], [3.0, 4.0]];
    let scale_2d = [1.0, 1.0];
    let bias_2d = [0.0, 0.0];
    let mean_2d = [2.0, 3.0];
    let var_2d = [1.0, 1.0];
    let result_2d = std::ml::batch_normalization(input_2d, scale_2d, bias_2d, mean_2d, var_2d, 1e-5);

    let input_3d = [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]];
    let scale_3d = [1.0, 1.0];
    let bias_3d = [0.0, 0.0];
    let mean_3d = [2.0, 3.0];
    let var_3d = [1.0, 1.0];
    let result_3d = std::ml::batch_normalization(input_3d, scale_3d, bias_3d, mean_3d, var_3d, 1e-5);

    let input_4d = [[[[1.0, 2.0], [3.0, 4.0]]]];
    let scale_4d = [1.0];
    let bias_4d = [0.0];
    let mean_4d = [2.0];
    let var_4d = [1.0];
    let result_4d = std::ml::batch_normalization(input_4d, scale_4d, bias_4d, mean_4d, var_4d, 1e-5);

    let input_5d = [[[[[1.0, 2.0], [3.0, 4.0]]]]];
    let scale_5d = [1.0];
    let bias_5d = [0.0];
    let mean_5d = [2.0];
    let var_5d = [1.0];
    let result_5d = std::ml::batch_normalization(input_5d, scale_5d, bias_5d, mean_5d, var_5d, 1e-5);

    let instance_input = [[[[1.0, 2.0], [3.0, 4.0]]]];
    let instance_scale = [1.0];
    let instance_bias = [0.0];
    let instance_result = std::ml::instance_normalization(instance_input, instance_scale, instance_bias, 1e-5);

    let layer_input_2d = [[1.0, 2.0], [3.0, 4.0]];
    let layer_scale_2d = [1.0, 1.0];
    let layer_bias_2d = [0.0, 0.0];
    let layer_result_2d = std::ml::layer_normalization(layer_input_2d, layer_scale_2d, layer_bias_2d, 1e-5, -1);

    let layer_input_3d = [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]];
    let layer_scale_3d = [1.0, 1.0];
    let layer_bias_3d = [0.0, 0.0];
    let layer_result_3d = std::ml::layer_normalization(layer_input_3d, layer_scale_3d, layer_bias_3d, 1e-5, -1);

    let lrn_input = [[[[1.0, 2.0], [3.0, 4.0]]]];
    let lrn_result = std::ml::lrn(lrn_input, 2, 1.0, 0.5, 1.0);

    let lp_input = [[1.0, 2.0, 3.0]];
    let lp_result = std::ml::lp_normalization(lp_input, -1, 2.0);

    let mvn_input = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
    let mvn_result = std::ml::mean_variance_normalization(mvn_input, [1]);
    """
    result = compile_and_execute(source, module_compiler, module_runtime)
    assert result.success, f"Execution failed: {result.errors}"
    return result.outputs


def test_batch_normalization_across_ranks(normalization_outputs):
    input_2d = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    scale_2d = np.array([1.0, 1.0], dtype=np.float32)
    bias_2d = np.array([0.0, 0.0], dtype=np.float32)
    mean_2d = np.array([2.0, 3.0], dtype=np.float32)
    var_2d = np.array([1.0, 1.0], dtype=np.float32)
    expected_2d = (input_2d - mean_2d) / np.sqrt(var_2d + 1e-5) * scale_2d + bias_2d
    np.testing.assert_allclose(np.array(normalization_outputs["result_2d"]), expected_2d, rtol=1e-5)

    input_3d = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], dtype=np.float32)
    scale_3d = np.array([1.0, 1.0], dtype=np.float32)
    bias_3d = np.array([0.0, 0.0], dtype=np.float32)
    mean_3d = np.array([2.0, 3.0], dtype=np.float32)
    var_3d = np.array([1.0, 1.0], dtype=np.float32)
    expected_3d = (input_3d - mean_3d[:, np.newaxis]) / np.sqrt(var_3d[:, np.newaxis] + 1e-5) * scale_3d[:, np.newaxis] + bias_3d[:, np.newaxis]
    np.testing.assert_allclose(np.array(normalization_outputs["result_3d"]), expected_3d, rtol=1e-5)

    input_4d = np.array([[[[1.0, 2.0], [3.0, 4.0]]]], dtype=np.float32)
    scale_4d = np.array([1.0], dtype=np.float32)
    bias_4d = np.array([0.0], dtype=np.float32)
    mean_4d = np.array([2.0], dtype=np.float32)
    var_4d = np.array([1.0], dtype=np.float32)
    expected_4d = (input_4d - mean_4d[:, np.newaxis, np.newaxis]) / np.sqrt(var_4d[:, np.newaxis, np.newaxis] + 1e-5) * scale_4d[:, np.newaxis, np.newaxis] + bias_4d[:, np.newaxis, np.newaxis]
    np.testing.assert_allclose(np.array(normalization_outputs["result_4d"]), expected_4d, rtol=1e-5)

    input_5d = np.array([[[[[1.0, 2.0], [3.0, 4.0]]]]], dtype=np.float32)
    scale_5d = np.array([1.0], dtype=np.float32)
    bias_5d = np.array([0.0], dtype=np.float32)
    mean_5d = np.array([2.0], dtype=np.float32)
    var_5d = np.array([1.0], dtype=np.float32)
    expected_5d = (input_5d - mean_5d[:, np.newaxis, np.newaxis, np.newaxis]) / np.sqrt(var_5d[:, np.newaxis, np.newaxis, np.newaxis] + 1e-5) * scale_5d[:, np.newaxis, np.newaxis, np.newaxis] + bias_5d[:, np.newaxis, np.newaxis, np.newaxis]
    np.testing.assert_allclose(np.array(normalization_outputs["result_5d"]), expected_5d, rtol=1e-5)


def test_instance_normalization(normalization_outputs):
    input_arr = np.array([[[[1.0, 2.0], [3.0, 4.0]]]], dtype=np.float32)
    scale = np.array([1.0], dtype=np.float32)
    bias = np.array([0.0], dtype=np.float32)
    mean = np.mean(input_arr, axis=(2, 3), keepdims=True)
    var = np.var(input_arr, axis=(2, 3), keepdims=True)
    expected = scale.reshape(1, -1, 1, 1) * (input_arr - mean) / np.sqrt(var + 1e-5) + bias.reshape(1, -1, 1, 1)
    np.testing.assert_allclose(np.array(normalization_outputs["instance_result"]), expected, rtol=1e-5)


def test_layer_normalization_across_ranks(normalization_outputs):
    input_2d = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    scale_2d = np.array([1.0, 1.0], dtype=np.float32)
    bias_2d = np.array([0.0, 0.0], dtype=np.float32)
    mean_2d = np.mean(input_2d, axis=-1, keepdims=True)
    var_2d = np.var(input_2d, axis=-1, keepdims=True)
    expected_2d = scale_2d * (input_2d - mean_2d) / np.sqrt(var_2d + 1e-5) + bias_2d
    np.testing.assert_allclose(np.array(normalization_outputs["layer_result_2d"]), expected_2d, rtol=1e-5)

    input_3d = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], dtype=np.float32)
    scale_3d = np.array([1.0, 1.0], dtype=np.float32)
    bias_3d = np.array([0.0, 0.0], dtype=np.float32)
    mean_3d = np.mean(input_3d, axis=-1, keepdims=True)
    var_3d = np.var(input_3d, axis=-1, keepdims=True)
    expected_3d = scale_3d * (input_3d - mean_3d) / np.sqrt(var_3d + 1e-5) + bias_3d
    np.testing.assert_allclose(np.array(normalization_outputs["layer_result_3d"]), expected_3d, rtol=1e-5)


def test_lrn(normalization_outputs):
    input_arr = np.array([[[[1.0, 2.0], [3.0, 4.0]]]], dtype=np.float32)
    expected = input_arr / np.power(1.0 + 1.0 * np.sum(input_arr**2, axis=1, keepdims=True), 0.5)
    np.testing.assert_allclose(np.array(normalization_outputs["lrn_result"]), expected, rtol=1e-4)


def test_lp_normalization(normalization_outputs):
    input_arr = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
    norm = np.power(np.sum(np.power(np.abs(input_arr), 2.0), axis=-1, keepdims=True), 0.5)
    expected = input_arr / norm
    np.testing.assert_allclose(np.array(normalization_outputs["lp_result"]), expected, rtol=1e-5)


def test_mean_variance_normalization(normalization_outputs):
    input_arr = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    mean = np.mean(input_arr, axis=(1,), keepdims=True)
    var = np.var(input_arr, axis=(1,), keepdims=True)
    expected = (input_arr - mean) / np.sqrt(var + 1e-9)
    np.testing.assert_allclose(np.array(normalization_outputs["mvn_result"]), expected, rtol=1e-5)
