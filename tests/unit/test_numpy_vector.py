import numpy as np

from einlang.backends.numpy_expressions import _safe_oob_ndarray_access


def test_safe_oob_ndarray_access_zero_fills_masked_positions():
    array = np.arange(9, dtype=np.float32).reshape(3, 3)
    rows = np.array([[0, -1], [2, 1]], dtype=np.intp)
    cols = np.array([[1, 2], [3, 0]], dtype=np.intp)

    got = _safe_oob_ndarray_access(array, [rows, cols])

    want = np.array([[1.0, 0.0], [0.0, 3.0]], dtype=np.float32)
    np.testing.assert_array_equal(got, want)


def test_safe_oob_ndarray_access_preserves_trailing_axes():
    array = np.arange(24, dtype=np.float32).reshape(3, 4, 2)
    rows = np.array([0, 2, 4], dtype=np.intp)
    cols = np.array([1, -1, 3], dtype=np.intp)

    got = _safe_oob_ndarray_access(array, [rows, cols])

    want = np.array(
        [
            array[0, 1],
            [0.0, 0.0],
            [0.0, 0.0],
        ],
        dtype=np.float32,
    )
    np.testing.assert_array_equal(got, want)
