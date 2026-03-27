import numpy as np
import pytest

from einlang.backends.numpy_helpers import (
    builtin_array_append,
    builtin_assert,
    builtin_shape,
    builtin_typeof,
)


def test_builtin_array_append_expands_tuple_inputs():
    assert builtin_array_append((1, 2), 3) == [1, 2, 3]


def test_builtin_array_append_preserves_ndarray_dtype():
    result = builtin_array_append(np.array([1, 2], dtype=np.int16), 3)

    assert isinstance(result, np.ndarray)
    assert result.dtype == np.int16
    np.testing.assert_array_equal(result, np.array([1, 2, 3], dtype=np.int16))


def test_builtin_shape_accepts_nested_tuples():
    got = builtin_shape(((1, 2), (3, 4)))

    np.testing.assert_array_equal(got, np.array([2, 2], dtype=int))


def test_builtin_typeof_marks_ragged_sequences_as_array():
    assert builtin_typeof([[1], [2, 3]]) == "array"
    assert builtin_typeof((np.array([1, 2]), np.array([3, 4]))) == "rectangular"


def test_builtin_assert_handles_nested_sequences_of_numpy_scalars():
    builtin_assert([[np.bool_(True)], [np.int64(1)]])

    with pytest.raises(RuntimeError, match="assertion failed: boom"):
        builtin_assert([[np.bool_(True)], [np.int64(0)]], "boom")
