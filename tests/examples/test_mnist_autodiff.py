#!/usr/bin/env python3
"""
Source-level autodiff checks for the MNIST training example.
"""

from __future__ import annotations

import importlib
import importlib.util
from contextlib import contextmanager
import os
from pathlib import Path
import sys

import pytest
import numpy as np
from tests.test_utils import compile_and_execute


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_MNIST_LOSS_REF_PATH = PROJECT_ROOT / "examples" / "mnist" / "compare_train_one_step_numpy.py"


def _load_path_module(module_name: str, module_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _load_loss_ref():
    return _load_path_module("mnist_source_loss_reference", _MNIST_LOSS_REF_PATH)


@contextmanager
def _mnist_example_runtime_context():
    example_dir = PROJECT_ROOT / "examples" / "mnist"
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


def test_mnist_train_one_step_source_loss_matches_numpy_reference_and_decreases(compiler, runtime):
    ref = _load_loss_ref()
    source_file = PROJECT_ROOT / "examples" / "mnist" / "train_one_step.ein"
    source = source_file.read_text(encoding="utf-8")

    result = compile_and_execute(
        source,
        compiler,
        runtime,
        source_file=str(source_file),
    )

    assert result.success, result.error or result.errors

    loss0 = float(np.asarray(result.outputs["loss0_scalar"], dtype=np.float64))
    loss1 = float(np.asarray(result.outputs["loss1_scalar"], dtype=np.float64))
    ref_loss0, ref_loss1 = ref._run_numpy()

    assert loss1 < loss0
    assert np.allclose(loss0, ref_loss0, atol=1e-6, rtol=1e-6)
    assert np.allclose(loss1, ref_loss1, atol=1e-6, rtol=1e-6)


def test_mnist_train_sklearn_digits_multi_epoch_matches_numpy_reference_and_improves(compiler, runtime):
    pytest.importorskip("sklearn")

    source_file = PROJECT_ROOT / "examples" / "mnist" / "train_sklearn_digits.ein"
    source = source_file.read_text(encoding="utf-8")

    with _mnist_example_runtime_context():
        result = compile_and_execute(
            source,
            compiler,
            runtime,
            source_file=str(source_file),
        )
        import mnist_data

    assert result.success, result.error or result.errors

    train_images, train_labels = mnist_data.load_train_images(), mnist_data.load_train_labels()
    test_images, test_labels = mnist_data.load_test_images(), mnist_data.load_test_labels()

    batch_size = 1437
    eval_size = 360
    lr = np.float32(0.2)
    x_train = train_images[:batch_size, 0].reshape(batch_size, 64).astype(np.float32)
    y_train = train_labels[:batch_size].astype(np.float32)
    x_eval = test_images[:eval_size, 0].reshape(eval_size, 64).astype(np.float32)
    y_eval = test_labels[:eval_size].astype(np.float32)
    train_truth = np.argmax(y_train, axis=1)
    eval_truth = np.argmax(y_eval, axis=1)

    theta = np.zeros((65, 10), dtype=np.float32)
    for p in range(65):
        for cls in range(10):
            theta[p, cls] = np.float32(0.0 if p == 64 else 1e-3 * (1.0 + ((p * 3 + cls) % 11)))

    expected_loss = []
    expected_correct = []
    expected_eval_correct = []
    for epoch in range(11):
        logits = x_train @ theta[:64] + theta[64]
        diff = logits - y_train
        expected_loss.append(float(np.mean(diff * diff, dtype=np.float32)))
        expected_correct.append(float(np.sum(np.argmax(logits, axis=1) == train_truth)))

        eval_logits = x_eval @ theta[:64] + theta[64]
        expected_eval_correct.append(float(np.sum(np.argmax(eval_logits, axis=1) == eval_truth)))

        if epoch < 10:
            d_logits = (2.0 * diff / np.float32(batch_size * 10)).astype(np.float32)
            grad = np.zeros_like(theta)
            grad[:64] = x_train.T @ d_logits
            grad[64] = d_logits.sum(axis=0)
            theta = theta - lr * grad

    actual_loss = np.array(
        [float(np.asarray(result.outputs[f"loss{i}"], dtype=np.float64)) for i in range(11)],
        dtype=np.float64,
    )
    actual_correct = np.array(
        [float(np.asarray(result.outputs[f"correct{i}"], dtype=np.float64)) for i in range(11)],
        dtype=np.float64,
    )
    actual_eval_correct = np.array(
        [float(np.asarray(result.outputs[f"eval_correct{i}"], dtype=np.float64)) for i in range(11)],
        dtype=np.float64,
    )

    np.testing.assert_allclose(actual_loss, np.array(expected_loss, dtype=np.float64), rtol=5e-7, atol=2e-8)
    np.testing.assert_allclose(actual_correct, np.array(expected_correct, dtype=np.float64), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(
        actual_eval_correct, np.array(expected_eval_correct, dtype=np.float64), rtol=1e-12, atol=1e-12
    )
    assert np.all(actual_loss[1:] < actual_loss[:-1]), "train loss should decrease each epoch"
    assert actual_correct[-1] > actual_correct[1], "later epochs should beat the first update on train accuracy"
    assert actual_eval_correct[-1] > actual_eval_correct[1], "later epochs should beat the first update on eval accuracy"
    assert actual_eval_correct[-1] > 300.0, "full-split trainer should reach a useful held-out accuracy bar"


def test_mnist_train_sklearn_digits_mlp_one_step_improves_batch_and_eval(compiler, runtime):
    pytest.importorskip("sklearn")

    source_file = PROJECT_ROOT / "examples" / "mnist" / "train_sklearn_digits_mlp.ein"
    source = source_file.read_text(encoding="utf-8")

    with _mnist_example_runtime_context():
        result = compile_and_execute(
            source,
            compiler,
            runtime,
            source_file=str(source_file),
        )

    assert result.success, result.error or result.errors

    actual = np.array(
        [
            float(np.asarray(result.outputs["batch_loss0"], dtype=np.float64)),
            float(np.asarray(result.outputs["batch_loss1"], dtype=np.float64)),
            float(np.asarray(result.outputs["batch_accuracy0"], dtype=np.float64)),
            float(np.asarray(result.outputs["batch_accuracy1"], dtype=np.float64)),
            float(np.asarray(result.outputs["eval_accuracy0"], dtype=np.float64)),
            float(np.asarray(result.outputs["eval_accuracy1"], dtype=np.float64)),
            float(np.asarray(result.outputs["batch_correct0"], dtype=np.float64)),
            float(np.asarray(result.outputs["batch_correct1"], dtype=np.float64)),
            float(np.asarray(result.outputs["eval_correct0"], dtype=np.float64)),
            float(np.asarray(result.outputs["eval_correct1"], dtype=np.float64)),
        ],
        dtype=np.float64,
    )
    expected = np.array(
        [
            0.3485898849088699,
            0.21272937767207623,
            0.125,
            0.3125,
            0.0625,
            0.1875,
            2.0,
            5.0,
            1.0,
            3.0,
        ],
        dtype=np.float64,
    )

    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
    assert actual[1] < actual[0], "batch loss should decrease after the update"
    assert actual[7] > actual[6], "batch correct count should increase after the update"
    assert actual[9] > actual[8], "held-out correct count should improve after the update"


def test_gradient_descent_autodiff_converges_to_quadratic_solution(compiler, runtime):
    source_file = PROJECT_ROOT / "examples" / "gradient_descent_autodiff.ein"
    source = source_file.read_text(encoding="utf-8")

    result = compile_and_execute(
        source,
        compiler,
        runtime,
        source_file=str(source_file),
    )

    assert result.success, result.error or result.errors

    loss_initial = float(np.asarray(result.outputs["loss_initial"], dtype=np.float64))
    loss_final = float(np.asarray(result.outputs["loss_final"], dtype=np.float64))
    x0_final = float(np.asarray(result.outputs["x"]).reshape(20, 2)[19, 0])
    x1_final = float(np.asarray(result.outputs["x"]).reshape(20, 2)[19, 1])

    assert loss_final < loss_initial
    assert loss_initial == np.float64(2.0)
    assert loss_final < 1e-6
    assert np.allclose([x0_final, x1_final], [0.5, 0.5], atol=5e-5, rtol=0.0)
