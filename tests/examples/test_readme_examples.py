#!/usr/bin/env python3
"""
README contract tests for runnable top-level examples and entry points.
"""

import importlib
import os
import re
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pytest

from einlang import run as einlang_run
from tests.examples.test_simulation_accuracy import ALL_ACCURACY_EXAMPLES, _assert_accuracy_case
from tests.test_utils import compile_and_execute


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
README_PATH = PROJECT_ROOT / "README.md"


def _extract_concrete_readme_example_paths():
    text = README_PATH.read_text(encoding="utf-8")
    return tuple(sorted(set(re.findall(r"examples/[A-Za-z0-9_./-]+\.ein", text))))


_ACCURACY_CASES_BY_PATH = {}
for case in ALL_ACCURACY_EXAMPLES:
    path = case[0]
    if isinstance(path, str):
        _ACCURACY_CASES_BY_PATH.setdefault(path, []).append(case[1:])


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


def _run_readme_example(compiler, runtime, relative_path: str):
    path = PROJECT_ROOT / relative_path
    source = path.read_text(encoding="utf-8")
    with _example_runtime_context(path.parent):
        return compile_and_execute(source, compiler, runtime, source_file=str(path))


def _run_cli(args, *, stdin_text=None):
    env = dict(os.environ)
    src_path = str(PROJECT_ROOT / "src")
    env["PYTHONPATH"] = (
        src_path if not env.get("PYTHONPATH") else src_path + os.pathsep + env["PYTHONPATH"]
    )
    return subprocess.run(
        [sys.executable, "-m", "einlang", *args],
        input=stdin_text,
        text=True,
        capture_output=True,
        cwd=str(PROJECT_ROOT),
        env=env,
        timeout=300,
    )


def _as_scalar(value):
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return value.item()
        raise AssertionError(f"expected scalar-like value, got shape {value.shape!r}")
    return value


def _assert_success(result):
    assert result.success, result.error or result.errors


def _assert_hello(result):
    _assert_success(result)
    np.testing.assert_array_equal(
        np.asarray(result.outputs["C"]),
        np.array([[19, 22], [43, 50]], dtype=np.int64),
    )


def _assert_autodiff_small(result):
    _assert_success(result)
    expected = {
        "x": 1.0,
        "y": 2.0,
        "z": 3.0,
        "dz_dx": 1.0,
        "dz_dy": 1.0,
        "w": 2.0,
        "dw_dx": 2.0,
        "dw_dy": 1.0,
        "u": -1.0,
        "du_dx": 1.0,
        "du_dy": -1.0,
        "v": 0.5,
        "dv_dx": 0.5,
        "dv_dy": -0.25,
    }
    for name, expected_value in expected.items():
        assert float(_as_scalar(result.outputs[name])) == pytest.approx(expected_value), name


def _assert_autodiff_matmul(result):
    _assert_success(result)
    np.testing.assert_allclose(
        np.asarray(result.outputs["C"], dtype=np.float64),
        np.array([[19.0, 22.0], [43.0, 50.0]], dtype=np.float64),
    )
    np.testing.assert_allclose(
        np.asarray(result.outputs["dC_dA"], dtype=np.float64),
        np.array(
            [
                [
                    [[5.0, 7.0], [0.0, 0.0]],
                    [[6.0, 8.0], [0.0, 0.0]],
                ],
                [
                    [[0.0, 0.0], [5.0, 7.0]],
                    [[0.0, 0.0], [6.0, 8.0]],
                ],
            ],
            dtype=np.float64,
        ),
    )


def _assert_accuracy_registered_example(result, relative_path: str):
    _assert_success(result)
    assert relative_path in _ACCURACY_CASES_BY_PATH, f"missing accuracy registry for {relative_path}"
    for output_key, ref_fn, rtol, atol, first_n in _ACCURACY_CASES_BY_PATH[relative_path]:
        _assert_accuracy_case(result, relative_path, output_key, ref_fn, rtol, atol, first_n)


def _assert_mnist(result):
    _assert_success(result)
    predictions = np.asarray(result.outputs["predictions"]).tolist()
    assert predictions == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], predictions


README_EXAMPLE_VALIDATORS = {
    "examples/autodiff_matmul.ein": _assert_autodiff_matmul,
    "examples/autodiff_small.ein": _assert_autodiff_small,
    "examples/hello.ein": _assert_hello,
    "examples/mnist/main.ein": _assert_mnist,
    "examples/ode/ode_suite.ein": lambda result: _assert_accuracy_registered_example(
        result, "examples/ode/ode_suite.ein"
    ),
    "examples/optimization/optimization_suite.ein": lambda result: _assert_accuracy_registered_example(
        result, "examples/optimization/optimization_suite.ein"
    ),
    "examples/recurrence/recurrence_suite.ein": lambda result: _assert_accuracy_registered_example(
        result, "examples/recurrence/recurrence_suite.ein"
    ),
}


def test_readme_example_registry_matches_concrete_paths():
    assert set(README_EXAMPLE_VALIDATORS) == set(_extract_concrete_readme_example_paths())


@pytest.mark.parametrize("relative_path", README_EXAMPLE_VALIDATORS, ids=lambda path: path)
def test_readme_concrete_examples_execute(compiler, runtime, relative_path):
    result = _run_readme_example(compiler, runtime, relative_path)
    README_EXAMPLE_VALIDATORS[relative_path](result)


def test_readme_cli_inline_source_example():
    result = _run_cli(["-c", "let x = 1+1; print(x);"])
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "2"


def test_readme_cli_stdin_example():
    result = _run_cli(["-"], stdin_text="let x = 2; print(x);")
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "2"


def test_readme_cli_file_example():
    result = _run_cli(["examples/hello.ein"])
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip().splitlines() == ["A * B =", "[[19, 22], [43, 50]]"]


def test_readme_python_api_source_example():
    result = einlang_run(source="let x = 1+1; print(x);")
    _assert_success(result)
    assert int(_as_scalar(result.outputs["x"])) == 2


def test_readme_python_api_file_example():
    result = einlang_run(file=PROJECT_ROOT / "examples" / "hello.ein")
    _assert_hello(result)
