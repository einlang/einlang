"""Numeric checks for ``let q = @num / @den`` for each golden case in ``test_print_at_golden``.

Expected values live in ``_EXPECTED_DY_DX`` (float or nested list).  Values are read from
``ExecutionResult.outputs['q']``.
"""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import pytest

from einlang.compiler.driver import CompilerDriver
from einlang.runtime.runtime import EinlangRuntime

from tests.unit.test_print_at_golden import GOLDEN_PRINT_CASES

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def _short_err(obj: object, limit: int = 600) -> str:
    if obj is None:
        return ""
    s = str(obj)
    if len(s) > limit:
        return s[:limit] + "..."
    return s


@pytest.fixture(scope="module")
def quotient_context(module_compiler, module_runtime) -> Tuple[CompilerDriver, EinlangRuntime]:
    return module_compiler, module_runtime


def _compile_exec_outputs(
    source: str,
    compiler: CompilerDriver,
    runtime: EinlangRuntime,
) -> Tuple[bool, bool, Dict[str, Any], str]:
    result = compiler.compile(source.strip(), source_file="<test>", root_path=_REPO_ROOT)
    if not result.success:
        return False, False, {}, _short_err(result.get_errors())
    exec_result = runtime.execute(result)
    if not exec_result.success:
        err = exec_result.error or exec_result.errors or "exec failed"
        return True, False, {}, _short_err(err)
    return True, True, dict(exec_result.outputs or {}), ""


def _build_quotient_source(label: str, source: str) -> str:
    s = source.strip()
    if label == "product_two_vars":
        return """
let x = 3.0;
let b = 4.0;
let y = x * b;
let q = @y / @x;
"""
    if label == "quotient":
        return """
let x = 3.0;
let b = 4.0;
let y = x / b;
let q = @y / @x;
"""
    if label == "matmul":
        return """
use std::ml;
let A = [[1.0, 2.0], [3.0, 4.0]];
let B = [[5.0, 6.0], [7.0, 8.0]];
let C = std::ml::matmul(A, B);
let q = @C / @A;
"""
    if label == "batch_matmul":
        return """
use std::ml;
let A = [[[1.0, 2.0], [3.0, 4.0]], [[0.5, 0.5], [0.1, 0.2]]];
let B = [[[5.0, 6.0], [7.0, 8.0]], [[1.0, 1.0], [1.0, 1.0]]];
let C = std::ml::batch_matmul(A, B);
let q = @C / @A;
"""
    if label == "mse_loss":
        return """
use std::ml;
let pred = [[1.0, 2.0, 3.0]];
let target = [[1.5, 2.5, 3.5]];
let y = std::ml::mse_loss(pred, target);
let q = @y / @pred;
"""
    if label == "mae_loss":
        return """
use std::ml;
let pred = [[1.0, 2.0, 3.0]];
let target = [[1.5, 2.5, 3.5]];
let y = std::ml::mae_loss(pred, target);
let q = @y / @pred;
"""
    if label == "huber_loss":
        return """
use std::ml;
let pred = [[1.0, 2.0, 3.0]];
let target = [[1.5, 2.5, 3.5]];
let y = std::ml::huber_loss(pred, target, 1.0);
let q = @y / @pred;
"""
    if label == "binary_cross_entropy":
        return """
use std::ml;
let pred = [[0.8, 0.3, 0.9]];
let target = [[1.0, 0.0, 1.0]];
let y = std::ml::binary_cross_entropy(pred, target);
let q = @y / @pred;
"""
    if label == "cosine_similarity":
        return """
use std::ml;
let a = [[1.0, 2.0, 3.0]];
let b = [[4.0, 5.0, 6.0]];
let y = std::ml::cosine_similarity(a, b);
let q = @y / @a;
"""
    m = re.search(r"print\(@(\w+)\)\s*;", s)
    if not m:
        raise ValueError(f"{label}: no print(@ID); found")
    return re.sub(r"print\(@(\w+)\)\s*;", r"let q = @\1 / @x;\n", s, count=1)


_ExpectedValue = Union[float, List[Any]]

_EXPECTED_DY_DX: Dict[str, _ExpectedValue] = {
    "constant": 0.0,
    "identity": 1.0,
    "add": 2.0,
    "sub": 1.0,
    "product": 6.0,
    "product_two_vars": 4.0,
    "quotient": 1.0 / 4.0,
    "power_const": 12.0,
    "power_square": 2.0,
    "neg": -1.0,
    "chain_let": 8.0,
    "exp_scalar": math.exp(1.0),
    "if_else": 1.0,
    "scalar_mul": 2.0,
    "compound": 7.0,
    "call_plus_fn": 2.0,
    "multistatement_callee_g": 2.0,
    "log_scalar": 0.5,
    "sin_scalar": math.cos(1.0),
    "cos_scalar": -math.sin(1.0),
    "tan_scalar": 1.0 / (math.cos(0.5) ** 2),
    "log1p_scalar": 1.0 / 1.5,
    "expm1_scalar": math.exp(0.5),
    "atan_scalar": 1.0 / (1.0 + 0.5**2),
    "asin_scalar": 1.0 / math.sqrt(1.0 - 0.5**2),
    "acos_scalar": -1.0 / math.sqrt(1.0 - 0.5**2),
    "atan2_two_vars": -1.0 / (2.0**2 + 1.0**2),
    "tanh_scalar": 1.0 / (math.cosh(0.5) ** 2),
    "sinh_scalar": math.cosh(0.5),
    "cosh_scalar": math.sinh(0.5),
    "asinh_scalar": 1.0 / math.hypot(1.0, 0.5),
    "acosh_scalar": 1.0 / math.sqrt(2.0**2 - 1.0),
    "atanh_scalar": 1.0 / (1.0 - 0.5**2),
    "erf_scalar": (2.0 / math.sqrt(math.pi)) * math.exp(-1.0),
    "abs_scalar": 1.0,
    "sign_scalar": 0.0,
    "min_scalar": 1.0,
    "max_scalar": 1.0,
    "reciprocal_scalar": -0.25,
    "rsqrt_scalar": -0.5 * (4.0 ** (-1.5)),
    "sqrt_via_pow": 0.25,
    "mod_scalar": 1.0,
    "quotient_chain": 1.0 / 16.0,
    "exp_einstein": [math.exp(1.0), math.exp(2.0), math.exp(3.0)],
    "einstein_square": [2.0, 4.0, 6.0],
    "softmax": [[0.0, 0.0, 0.0]],
    "linear": [[0.7, 0.7]],
    "max_pool": [[[[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]]]],
    "max_pool_relu_arg": [[[[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]]]],
    "mnist_conv2d": [[[[1.0, 1.0, 0.0], [1.0, 2.0, 1.0], [0.0, 1.0, 1.0]]]],
    "softmax_quotient": [0.08192507, 0.24472846, 0.66524094],
    "sum_reduction": [math.exp(1.0), math.exp(2.0), math.exp(3.0)],
    "prod_reduction": [6.0, 3.0, 2.0],
    "reduce_sum": [[1.0, 1.0, 1.0]],
    "reduce_l1": [[1.0, -1.0, 1.0]],
    "reduce_sum_square": [[2.0, 4.0, 6.0]],
    "reduce_mean": [[1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]],
    "log_softmax": [[0.72990826, 0.26581466, -0.9957229]],
    "reduce_l2": [[0.6, 0.8]],
    "reduce_log_sum": [[1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0]],
    "reduce_log_sum_exp": [[0.09003057, 0.24472845, 0.66524096]],
    "matmul": [[11.0, 15.0], [11.0, 15.0]],
    "batch_matmul": [
        [[11.0, 15.0], [11.0, 15.0]],
        [[2.0, 2.0], [2.0, 2.0]],
    ],
    "mse_loss": [[-1.0 / 3.0, -1.0 / 3.0, -1.0 / 3.0]],
    "mae_loss": [[-1.0 / 3.0, -1.0 / 3.0, -1.0 / 3.0]],
    "huber_loss": [[-1.0 / 6.0, -1.0 / 6.0, -1.0 / 6.0]],
    "binary_cross_entropy": [[-0.41666667, 0.47619048, -0.37037037]],
    "cosine_similarity": [[0.05221242, 0.01305311, -0.02610619]],
}


def _flatten(v: Any) -> List[float]:
    if isinstance(v, (int, float, np.integer, np.floating)):
        return [float(v)]
    if isinstance(v, np.ndarray):
        return [float(x) for x in np.ravel(v)]
    out: List[float] = []
    for item in v:
        out.extend(_flatten(item))
    return out


def _approx_equal(got: Any, expected: Any, abs_tol: float = 1e-5) -> bool:
    fg, fe = _flatten(got), _flatten(expected)
    if len(fg) != len(fe):
        return False
    return all(abs(a - b) <= abs_tol for a, b in zip(fg, fe))


@pytest.mark.parametrize(
    "label,orig_source",
    [(row[0], row[1]) for row in GOLDEN_PRINT_CASES],
    ids=[row[0] for row in GOLDEN_PRINT_CASES],
)
def test_quotient_vs_calculus(
    label: str,
    orig_source: str,
    quotient_context: Tuple[CompilerDriver, EinlangRuntime],
) -> None:
    compiler, runtime = quotient_context
    expected = _EXPECTED_DY_DX.get(label)
    if expected is None:
        pytest.fail("missing _EXPECTED_DY_DX for label %r" % label)
    qsrc = _build_quotient_source(label, orig_source)
    c_ok, e_ok, outputs, err = _compile_exec_outputs(qsrc, compiler, runtime)
    assert c_ok, "%s: compile failed: %s" % (label, err)
    assert e_ok, "%s: exec failed: %s" % (label, err)
    got = outputs.get("q")
    assert got is not None, "%s: missing outputs['q'], have %r" % (label, sorted(outputs.keys()))
    assert _approx_equal(got, expected), (
        "%s: @…/@x got %r expected ∂y/∂x ≈ %s" % (label, got, expected)
    )


def test_quotient_golden_cases_partition() -> None:
    labels = {row[0] for row in GOLDEN_PRINT_CASES}
    covered = set(_EXPECTED_DY_DX)
    assert labels == covered, (
        "partition mismatch: extra %s missing %s"
        % (sorted(covered - labels), sorted(labels - covered))
    )
