#!/usr/bin/env python3
"""
Tests for std::ml::ml_ex using one grouped module fixture.
"""

import numpy as np
import pytest

from ...test_utils import compile_and_execute


@pytest.fixture(scope="module")
def ml_ex_outputs(module_compiler, module_runtime):
    source = """use std::ml;
    let x = [[[[1.0, 2.0], [3.0, 4.0]]]];
    let scaled = std::ml::ml_ex::image_scaler(x, 2.0, [0.5]);
    let eye = std::ml::ml_ex::eye(3);

    let mat = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]];
    let vec = [1.0, 2.0, 3.0];
    let extracted = std::ml::ml_ex::diag_extract(mat);
    let constructed = std::ml::ml_ex::diag_construct(vec);
    let traced = std::ml::ml_ex::trace(mat);
    let tril = std::ml::ml_ex::tril(mat, 0);
    let triu = std::ml::ml_ex::triu(mat, 0);

    let a = [1.0, 2.0, 3.0];
    let b = [4.0, 5.0];
    let a2 = [[1.0, 2.0], [3.0, 4.0]];
    let b2 = [[0.0, 5.0], [6.0, 7.0]];
    let outer = std::ml::ml_ex::outer(a, b);
    let kron = std::ml::ml_ex::kron(a2, b2);

    let roll_mat = [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]];
    let rolled = std::ml::ml_ex::roll(roll_mat, 2);
    let flipped = std::ml::ml_ex::flip(roll_mat);

    let fro_mat = [[1.0, 2.0], [3.0, 4.0]];
    let cosine_a = [[1.0, 2.0, 3.0]];
    let cosine_b = [[4.0, 5.0, 6.0]];
    let fro = std::ml::ml_ex::frobenius_norm(fro_mat);
    let cosine = std::ml::cosine_similarity(cosine_a, cosine_b);
    """
    result = compile_and_execute(source, module_compiler, module_runtime)
    assert result.success, f"Execution failed: {result.errors}"
    return result.outputs


def test_image_scaler_and_eye(ml_ex_outputs):
    x = np.array([[[[1.0, 2.0], [3.0, 4.0]]]], dtype=np.float32)
    scaled_expected = x * 2.0 + np.array([0.5], dtype=np.float32).reshape(1, -1, 1, 1)
    np.testing.assert_allclose(np.array(ml_ex_outputs["scaled"]), scaled_expected, rtol=1e-6)
    np.testing.assert_allclose(np.array(ml_ex_outputs["eye"]), np.eye(3, dtype=np.float32), rtol=1e-6)


def test_diag_ops_and_trace(ml_ex_outputs):
    mat = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], dtype=np.float32)
    vec = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    np.testing.assert_allclose(np.asarray(ml_ex_outputs["extracted"], dtype=np.float32), np.diag(mat), rtol=1e-6)
    np.testing.assert_allclose(np.asarray(ml_ex_outputs["constructed"], dtype=np.float32), np.diag(vec), rtol=1e-6)
    np.testing.assert_allclose(np.asarray(ml_ex_outputs["traced"], dtype=np.float32), np.trace(mat), rtol=1e-6)


def test_outer_and_kron(ml_ex_outputs):
    a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    b = np.array([4.0, 5.0], dtype=np.float32)
    a2 = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    b2 = np.array([[0.0, 5.0], [6.0, 7.0]], dtype=np.float32)
    np.testing.assert_allclose(np.array(ml_ex_outputs["outer"]), np.outer(a, b), rtol=1e-6)
    np.testing.assert_allclose(np.array(ml_ex_outputs["kron"]), np.kron(a2, b2), rtol=1e-6)


def test_triangular_ops(ml_ex_outputs):
    mat = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], dtype=np.float32)
    np.testing.assert_allclose(np.array(ml_ex_outputs["tril"]), np.tril(mat, k=0), rtol=1e-6)
    np.testing.assert_allclose(np.array(ml_ex_outputs["triu"]), np.triu(mat, k=0), rtol=1e-6)


def test_roll_and_flip(ml_ex_outputs):
    mat = np.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]], dtype=np.float32)
    np.testing.assert_allclose(np.array(ml_ex_outputs["rolled"]), np.roll(mat, 2, axis=1), rtol=1e-6)
    np.testing.assert_allclose(np.array(ml_ex_outputs["flipped"]), np.flip(mat, axis=1), rtol=1e-6)


def test_frobenius_norm_and_cosine_similarity(ml_ex_outputs):
    mat = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    a = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
    b = np.array([[4.0, 5.0, 6.0]], dtype=np.float32)
    expected_cosine = np.sum(a * b, axis=-1) / (np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1))
    np.testing.assert_allclose(np.array(ml_ex_outputs["fro"]), np.linalg.norm(mat, "fro"), rtol=1e-6)
    np.testing.assert_allclose(np.array(ml_ex_outputs["cosine"]), expected_cosine, rtol=1e-6)
