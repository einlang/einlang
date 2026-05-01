import numpy as np

from einlang.compiler.driver import CompilerDriver
from einlang.runtime.runtime import EinlangRuntime
from tests.test_utils import compile_and_execute


def _run(source: str):
    return compile_and_execute(
        source,
        CompilerDriver(),
        EinlangRuntime(backend="numpy"),
    )


def test_argmax_selects_named_coordinate_in_last_position():
    result = _run(
        """
        let logits = [[1.0, 5.0, 3.0], [9.0, 4.0, 8.0]];
        let pred[b in 0..2] = argmax[class](logits[b, class]);
        pred;
        """
    )

    assert result.success, result.errors
    np.testing.assert_array_equal(result.outputs["pred"], np.array([1, 0], dtype=np.int32))


def test_argmax_selects_named_coordinate_in_first_position():
    result = _run(
        """
        let logits = [[1.0, 9.0], [5.0, 4.0], [3.0, 8.0]];
        let pred[b in 0..2] = argmax[class](logits[class, b]);
        pred;
        """
    )

    assert result.success, result.errors
    np.testing.assert_array_equal(result.outputs["pred"], np.array([1, 0], dtype=np.int32))


def test_argmax_selects_named_coordinate_in_middle_position():
    result = _run(
        """
        let logits = [
            [[1.0, 7.0], [5.0, 2.0], [3.0, 9.0]],
            [[9.0, 1.0], [4.0, 8.0], [8.0, 3.0]]
        ];
        let pred[b in 0..2, t in 0..2] = argmax[class](logits[b, class, t]);
        pred;
        """
    )

    assert result.success, result.errors
    np.testing.assert_array_equal(
        result.outputs["pred"],
        np.array([[1, 2], [0, 1]], dtype=np.int32),
    )


def test_argmin_selection_reduction_returns_integer_addresses():
    result = _run(
        """
        let logits = [[1.0, 5.0, 3.0], [9.0, 4.0, 8.0]];
        let pred[b in 0..2] = argmin[class](logits[b, class]);
        pred;
        """
    )

    assert result.success, result.errors
    np.testing.assert_array_equal(result.outputs["pred"], np.array([0, 1], dtype=np.int32))


def test_coordinate_function_call_syntax_survives_lowering_and_execution():
    result = _run(
        """
        fn id_axis[j](x: [f32; ..left, j, ..right])
            -> [f32; ..left, j, ..right]
        {
            x
        }

        let x[class in 0..2] = class as f32;
        let y = id_axis[class](x);
        y;
        """
    )

    assert result.success, result.errors
    np.testing.assert_array_equal(result.outputs["y"], np.array([0.0, 1.0], dtype=np.float32))
