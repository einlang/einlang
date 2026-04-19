#!/usr/bin/env python3
"""
Focused example tests for applications with autodiff sensitivity checks.
"""

import importlib
import os
import sys
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pytest

from tests.test_utils import compile_and_execute, ensure_test_dependency


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


@contextmanager
def _example_runtime_context(example_dir: Path):
    """Run with the example directory as cwd/import root for relative Python helpers."""
    example_dir_str = str(example_dir)
    old_cwd = os.getcwd()
    inserted = False
    if example_dir_str not in sys.path:
        sys.path.insert(0, example_dir_str)
        inserted = True
    importlib.invalidate_caches()
    os.chdir(example_dir_str)
    try:
        yield
    finally:
        os.chdir(old_cwd)
        if inserted:
            try:
                sys.path.remove(example_dir_str)
            except ValueError:
                pass


def test_decay_calibration_autodiff_prints_and_refines_fit(session_compiler, session_runtime):
    """Run the new autodiff decay calibration example and pin its printed refinement values."""
    source = (PROJECT_ROOT / "examples" / "applications" / "decay_calibration_autodiff.ein").read_text(
        encoding="utf-8"
    )
    result = compile_and_execute(
        source,
        session_compiler,
        session_runtime,
        source_file=str(PROJECT_ROOT / "examples" / "applications" / "decay_calibration_autodiff.ein"),
    )
    assert result.success, result.error or result.errors

    actual = np.array(
        [
            float(result.outputs["sse"]),
            float(result.outputs["sse_next"]),
            float(result.outputs["k_next"]),
            float(result.outputs["u0_next"]),
        ],
        dtype=np.float64,
    )

    dt = 0.1
    n_obs = 12
    k_true = 0.08
    u0_true = 2.0
    k = 0.05
    u0 = 1.6
    time = np.arange(n_obs, dtype=np.float64) * dt
    data = u0_true * np.exp(-k_true * time)
    fitted = u0 * np.exp(-k * time)
    residuals = data - fitted
    sse = np.sum(residuals * residuals)
    d_sse_dk = np.sum(2.0 * residuals * (time * fitted))
    d_sse_du0 = np.sum(2.0 * residuals * (-np.exp(-k * time)))
    k_next = k - 0.01 * d_sse_dk
    u0_next = u0 - 0.01 * d_sse_du0
    fitted_next = u0_next * np.exp(-k_next * time)
    sse_next = np.sum((data - fitted_next) ** 2)
    expected = np.array([sse, sse_next, k_next, u0_next], dtype=np.float64)

    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
    assert actual[1] < actual[0], "SSE should decrease after the autodiff update"


def test_diabetes_ridge_sklearn_style_matches_numpy_reference(compiler, runtime):
    """Run the sklearn-style ridge example and pin its fit/eval outputs against NumPy."""
    ensure_test_dependency("sklearn")

    source_file = PROJECT_ROOT / "examples" / "applications" / "diabetes_ridge_sklearn_style.ein"
    source = source_file.read_text(encoding="utf-8")

    with _example_runtime_context(source_file.parent):
        result = compile_and_execute(
            source,
            compiler,
            runtime,
            source_file=str(source_file),
        )

        import diabetes_data

        train_features = diabetes_data.load_train_features()
        train_targets = diabetes_data.load_train_targets()
        test_features = diabetes_data.load_test_features()
        test_targets = diabetes_data.load_test_targets()

    assert result.success, result.error or result.errors

    train_design = np.concatenate(
        [train_features, np.ones((train_features.shape[0], 1), dtype=np.float32)],
        axis=1,
    )
    test_design = np.concatenate(
        [test_features, np.ones((test_features.shape[0], 1), dtype=np.float32)],
        axis=1,
    )
    alpha = np.float32(1.0)
    gram = train_design.T @ train_design
    reg = np.eye(train_design.shape[1], dtype=np.float32)
    reg[-1, -1] = 0.0
    expected_weights = np.linalg.solve(gram + alpha * reg, train_design.T @ train_targets)
    expected_train_preds = train_design @ expected_weights
    expected_test_preds = test_design @ expected_weights

    expected = np.array(
        [
            np.mean((expected_train_preds - train_targets) ** 2, dtype=np.float32),
            np.mean((expected_test_preds - test_targets) ** 2, dtype=np.float32),
            1.0
            - np.sum((expected_train_preds - train_targets) ** 2, dtype=np.float32)
            / np.sum((train_targets - np.mean(train_targets, dtype=np.float32)) ** 2, dtype=np.float32),
            1.0
            - np.sum((expected_test_preds - test_targets) ** 2, dtype=np.float32)
            / np.sum((test_targets - np.mean(test_targets, dtype=np.float32)) ** 2, dtype=np.float32),
            expected_weights[-1],
        ],
        dtype=np.float64,
    )
    actual = np.array(
        [
            float(np.asarray(result.outputs["train_mse"], dtype=np.float64)),
            float(np.asarray(result.outputs["test_mse"], dtype=np.float64)),
            float(np.asarray(result.outputs["train_r2"], dtype=np.float64)),
            float(np.asarray(result.outputs["test_r2"], dtype=np.float64)),
            float(np.asarray(result.outputs["intercept"], dtype=np.float64)),
        ],
        dtype=np.float64,
    )

    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-4)
    np.testing.assert_allclose(
        np.asarray(result.outputs["weights"], dtype=np.float64),
        expected_weights.astype(np.float64),
        rtol=1e-5,
        atol=1e-4,
    )
    assert actual[3] > 0.4, "held-out R^2 should stay above a weak baseline"
