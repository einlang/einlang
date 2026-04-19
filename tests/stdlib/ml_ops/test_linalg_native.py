#!/usr/bin/env python3
"""Test native linalg (norm, cholesky, solve) without numpy wrappers."""

import numpy as np
import pytest
from tests.test_utils import compile_and_execute, assert_float_close


def test_vec_norm_2(compiler, runtime):
    source = """
    use std::ml::linalg_ops::vec_norm_2;
    let v = [3.0, 4.0];
    let n = vec_norm_2(v);
    """
    result = compile_and_execute(source, compiler, runtime)
    assert result.success, result.errors
    assert_float_close(result.outputs["n"], 5.0)


def test_matrix_norm_frobenius(compiler, runtime):
    source = """
    use std::ml::linalg_ops::matrix_norm_frobenius;
    let A = [[3.0, 4.0], [0.0, 0.0]];
    let n = matrix_norm_frobenius(A);
    """
    result = compile_and_execute(source, compiler, runtime)
    assert result.success, result.errors
    assert_float_close(result.outputs["n"], 5.0)


def test_cholesky(compiler, runtime):
    # Cholesky L for A = [[4,1],[1,3]]. L should be lower triangular, L*L^T = A.
    source = """
    use std::ml::linalg_ops::cholesky;
    let A = [[4.0, 1.0], [1.0, 3.0]];
    let L = cholesky(A);
    """
    result = compile_and_execute(source, compiler, runtime)
    assert result.success, result.errors
    L = np.array(result.outputs["L"])
    A = np.array([[4.0, 1.0], [1.0, 3.0]])
    np.testing.assert_allclose(L @ L.T, A, rtol=1e-5)
    assert L[0, 1] == 0.0, "Cholesky L should be lower triangular"


def test_solve_cholesky(compiler, runtime):
    # Solve A x = b via Cholesky; A = [[4,1],[1,3]], b = [1, 2].
    source = """
    use std::ml::linalg_ops::solve_cholesky;
    let A = [[4.0, 1.0], [1.0, 3.0]];
    let b = [1.0, 2.0];
    let x = solve_cholesky(A, b);
    """
    result = compile_and_execute(source, compiler, runtime)
    assert result.success, result.errors
    x = np.array(result.outputs["x"])
    assert x.shape == (2,), f"expected shape (2,), got {x.shape}"
    A = np.array([[4.0, 1.0], [1.0, 3.0]])
    b = np.array([1.0, 2.0])
    np.testing.assert_allclose(A @ x, b, rtol=1e-5)


def test_solve_triangular_lower_unit(compiler, runtime):
    # Solve L*y = b with L unit lower (L[i,i]=1). Direct call to solve_triangular_lower_unit.
    source = """
    use std::ml::linalg_ops::solve_triangular_lower_unit;
    let L = [[1.0, 0.0], [0.5, 1.0]];
    let b = [1.0, 2.0];
    let y = solve_triangular_lower_unit(L, b);
    """
    result = compile_and_execute(source, compiler, runtime)
    assert result.success, result.errors
    y = np.array(result.outputs["y"])
    L = np.array([[1.0, 0.0], [0.5, 1.0]])
    b = np.array([1.0, 2.0])
    np.testing.assert_allclose(L @ y, b, rtol=1e-5)


def test_lu(compiler, runtime):
    # Doolittle LU: A = L*U, L unit lower, U upper. Check L*U = A.
    source = """
    use std::ml::linalg_ops::lu;
    let A = [[4.0, 1.0], [1.0, 3.0]];
    let (L, U) = lu(A);
    """
    result = compile_and_execute(source, compiler, runtime)
    assert result.success, result.errors
    L = np.array(result.outputs["L"])
    U = np.array(result.outputs["U"])
    A = np.array([[4.0, 1.0], [1.0, 3.0]])
    np.testing.assert_allclose(L @ U, A, rtol=1e-5)
    # L unit lower: diagonal 1, strict upper zero
    np.testing.assert_allclose(np.diag(L), np.ones(2), rtol=1e-5)
    assert L[0, 1] == 0.0, "L should be lower triangular"
    # U upper: strict lower zero
    assert U[1, 0] == 0.0, "U should be upper triangular"


def test_solve_lu(compiler, runtime):
    # Solve A*x = b via LU; A = [[4,1],[1,3]], b = [1, 2].
    source = """
    use std::ml::linalg_ops::solve;
    let A = [[4.0, 1.0], [1.0, 3.0]];
    let b = [1.0, 2.0];
    let x = solve(A, b);
    """
    result = compile_and_execute(source, compiler, runtime)
    assert result.success, result.errors
    x = np.array(result.outputs["x"])
    assert x.shape == (2,), f"expected shape (2,), got {x.shape}"
    A = np.array([[4.0, 1.0], [1.0, 3.0]])
    b = np.array([1.0, 2.0])
    np.testing.assert_allclose(A @ x, b, rtol=1e-5)
