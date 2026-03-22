"""Tensor ``@y/@x`` checks for **sum** reductions (non-scalar y, tensor x).

No ``print`` — compile, run numpy backend, compare ``@r/@M``, ``@c/@M``, ``@y/@x`` to NumPy references.
More reduction ops can follow in this module.

**Scalar quotient vs tensor ``x``:** If the independent variable is a tensor (vector/matrix) and
``@y/@x`` (or ``@s/@x``) lowers to a **rank-0** scalar at runtime, that is usually a bug: it
often means an unintended contraction (e.g. inner product with an all-ones tangent) instead of
the cotangent/Jacobian layout matching ``x``'s shape. Tests here call
``_assert_quotient_not_scalar_rank0`` whenever the primal is a non-scalar tensor.
"""

from __future__ import annotations

import numpy as np
import pytest

from tests.unit.test_autodiff_pass import _REPO_ROOT, _assert_allclose, _compile_run


def _assert_quotient_not_scalar_rank0(arr: np.ndarray, binding: str, primal_desc: str) -> None:
    if arr.ndim == 0:
        raise AssertionError(
            "%s has rank 0 (scalar) but independent variable is a tensor (%s). "
            "For tensor x this is usually wrong: expect a derivative tensor matching x's shape, "
            "not a single contracted float."
            % (binding, primal_desc)
        )


def test_sum_reduction_row_sum_dr_dM() -> None:
    """r[i] = sum[j](M[i,j]); @r/@M has shape M, entries 1 (row-sum Jacobian / cotangent layout)."""
    source = """
let M = [[1.0, 2.0], [3.0, 4.0]];
let r[i] = sum[j](M[i, j]);
let dr_dM = @r / @M;
"""
    _, out = _compile_run(source, root_path=_REPO_ROOT)
    dr_dM = out.get("dr_dM")
    assert dr_dM is not None, "expected dr_dM, got %s" % list(out.keys())
    arr = np.asarray(dr_dM, dtype=np.float64)
    _assert_quotient_not_scalar_rank0(arr, "dr_dM", "M (2x2)")
    assert arr.shape == (2, 2), "dr_dM shape (2,2), got %s" % (arr.shape,)
    _assert_allclose(arr, np.ones((2, 2)), msg="row sum @r/@M")


def test_sum_reduction_column_sum_dc_dM() -> None:
    """c[j] = sum[i](M[i,j]); @c/@M is ones (same shape as M)."""
    source = """
let M = [[1.0, 2.0], [3.0, 4.0]];
let c[j] = sum[i](M[i, j]);
let dc_dM = @c / @M;
"""
    _, out = _compile_run(source, root_path=_REPO_ROOT)
    dc_dM = out.get("dc_dM")
    assert dc_dM is not None, "expected dc_dM, got %s" % list(out.keys())
    arr = np.asarray(dc_dM, dtype=np.float64)
    assert arr.shape == (2, 2), "dc_dM shape (2,2), got %s" % (arr.shape,)
    _assert_allclose(arr, np.ones((2, 2)), msg="column sum @c/@M")


def test_sum_reduction_batched_dy_dx() -> None:
    """y[b,i] = sum[j](x[b,i,j]); @y/@x shape (B,I,J), all 1 (matches AUTODIFF_EINSTEIN_OPS batched_reduction_sum)."""
    source = """
let x = [[[1.0, 2.0], [3.0, 4.0]], [[0.5, 0.5], [0.1, 0.2]]];
let y[b, i] = sum[j](x[b, i, j]);
let dy_dx = @y / @x;
"""
    _, out = _compile_run(source, root_path=_REPO_ROOT)
    dy_dx = out.get("dy_dx")
    assert dy_dx is not None, "expected dy_dx, got %s" % list(out.keys())
    arr = np.asarray(dy_dx, dtype=np.float64)
    _assert_quotient_not_scalar_rank0(arr, "dy_dx", "x (2x2x2)")
    assert arr.shape == (2, 2, 2), "dy_dx shape (2,2,2), got %s" % (arr.shape,)
    _assert_allclose(arr, np.ones((2, 2, 2)), msg="batched sum @y/@x")


def test_sum_reduction_single_row_vector_x() -> None:
    """One batch row: x shape [1,3], y[b] = sum[j](x[b,j]); @y/@x is [1,3] ones."""
    source = """
let x = [[1.0, 2.0, 3.0]];
let y[b] = sum[j](x[b, j]);
let dy_dx = @y / @x;
"""
    _, out = _compile_run(source, root_path=_REPO_ROOT)
    dy_dx = out.get("dy_dx")
    assert dy_dx is not None, "expected dy_dx, got %s" % list(out.keys())
    arr = np.asarray(dy_dx, dtype=np.float64)
    _assert_quotient_not_scalar_rank0(arr, "dy_dx", "x (1x3)")
    assert arr.shape == (1, 3), "dy_dx shape (1,3), got %s" % (arr.shape,)
    _assert_allclose(arr, np.ones((1, 3)), msg="single-row sum @y/@x")


def test_sum_of_vector_x_ds_dx_gradient_shape() -> None:
    """s = sum_i x[i] with vector x — ∂s/∂x is length-n ones (cotangent layout matches x)."""
    source = """
let x = [0.5, 1.0, 2.0];
let s = sum[i](x[i]);
let ds_dx = @s / @x;
"""
    _, out = _compile_run(source, root_path=_REPO_ROOT)
    arr = np.asarray(out.get("ds_dx"), dtype=np.float64)
    _assert_quotient_not_scalar_rank0(arr, "ds_dx", "x vector length 3")
    assert arr.shape == (3,), "ds_dx shape (3,), got %s" % (arr.shape,)
    _assert_allclose(arr, np.ones(3), msg="sum(x) gradient w.r.t. x")
