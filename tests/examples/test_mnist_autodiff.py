#!/usr/bin/env python3
"""
Source-level autodiff checks for the MNIST training example.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

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
