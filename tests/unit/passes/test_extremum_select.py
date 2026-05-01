import numpy as np

from einlang.compiler.driver import CompilerDriver
from einlang.runtime.runtime import EinlangRuntime
from tests.test_utils import compile_and_execute


def _run(source: str):
    compiler = CompilerDriver()
    runtime = EinlangRuntime(backend="numpy")
    result = compile_and_execute(source, compiler, runtime, source_file="<test>")
    assert result.success, result.errors if result.errors else result.error
    return result.outputs


def test_runtime_argmax_outputs_remain_valid_indices():
    outputs = _run(
        """
let logits = [[1.0, 3.0, 2.0], [0.0, 1.0, 5.0]];
let ids[i] = argmax[j](logits[i, j]);
let table = [10, 20, 30];
let picked[i in 0..2] = table[ids[i]];
"""
    )

    np.testing.assert_array_equal(np.asarray(outputs["ids"]), np.array([1, 2], dtype=np.int64))
    np.testing.assert_array_equal(np.asarray(outputs["picked"]), np.array([20, 30], dtype=np.int64))


def test_runtime_argmin_outputs_remain_valid_indices():
    outputs = _run(
        """
let logits = [[1.0, -3.0, 2.0], [0.0, 1.0, -5.0]];
let ids[i] = argmin[j](logits[i, j]);
let table = [10, 20, 30];
let picked[i in 0..2] = table[ids[i]];
"""
    )

    np.testing.assert_array_equal(np.asarray(outputs["ids"]), np.array([1, 2], dtype=np.int64))
    np.testing.assert_array_equal(np.asarray(outputs["picked"]), np.array([20, 30], dtype=np.int64))


def test_runtime_scalar_argmax_output_remains_valid_index():
    outputs = _run(
        """
let logits = [1.0, 3.0, 2.0];
let idx = argmax[i](logits[i]);
let labels = [10, 20, 30];
let picked = labels[idx];
"""
    )

    assert int(outputs["idx"]) == 1
    assert int(outputs["picked"]) == 20


def test_max_gradient_routes_to_winning_index():
    outputs = _run(
        """
let x = [1.0, 3.0, 2.0];
let y = max[i](x[i]);
let dy = @y / @x;
"""
    )

    np.testing.assert_array_equal(np.asarray(outputs["dy"]), np.array([0.0, 1.0, 0.0], dtype=np.float32))


def test_min_gradient_routes_to_winning_index():
    outputs = _run(
        """
let x = [1.0, -3.0, 2.0];
let y = min[i](x[i]);
let dy = @y / @x;
"""
    )

    np.testing.assert_array_equal(np.asarray(outputs["dy"]), np.array([0.0, 1.0, 0.0], dtype=np.float32))
