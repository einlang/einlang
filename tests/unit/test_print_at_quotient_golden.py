"""Numeric stdout checks for ``print(@num / @x)`` for each ``test_print_at_golden`` case.

Covers both scalar and tensor quotients.  Scalar cases print a single float;
tensor cases print a bracketed array (1-D or 2-D).  Expected values live in
``_EXPECTED_DY_DX`` (float or nested list).

``print(@y)`` is compile-time formatted to a string literal. ``print(@y / @x)`` lowers to a
runtime value, so we assert the printed output against calculus.
"""

from __future__ import annotations

import ast
import math
import re
from typing import Any, Dict, FrozenSet, List, Optional, Tuple, Union

import pytest

from tests.print_at_fixtures import compile_exec_capture_print_at
from tests.unit.test_print_at_golden import GOLDEN_PRINT_CASES


def _build_quotient_source(label: str, source: str) -> str:
    s = source.strip()
    if label == "product_two_vars":
        return """
let x = 3.0;
let b = 4.0;
let y = x * b;
print(@y / @x);
"""
    if label == "quotient":
        return """
let x = 3.0;
let b = 4.0;
let y = x / b;
print(@y / @x);
"""
    m = re.search(r"print\(@(\w+)\)\s*;", s)
    if not m:
        raise ValueError(f"{label}: no print(@ID); found")
    sym = m.group(1)
    return re.sub(r"print\(@(\w+)\)\s*;", f"print(@{sym} / @x);", s, count=1)


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
    "linear": [[0.8, 0.6]],
    "softmax_quotient": [0.0, 0.0, 0.0],
    "sum_reduction": 30.192874908447266,
    "prod_reduction": [18.0, 9.0, 6.0],
    "reduce_sum": [3.0],
    "reduce_l1": [1.0],
    "reduce_sum_square": [12.0],
    "reduce_mean": [1.0],
    "log_softmax": [[1.0, 1.0, 1.0]],
    "reduce_l2": [0.0],
    "reduce_log_sum": [0.0],
    "reduce_log_sum_exp": [0.0],
}


_QUOTIENT_SKIP: FrozenSet[str] = frozenset(
    {
        "mse_loss",
        "mae_loss",
        "huber_loss",
        "binary_cross_entropy",
        "cosine_similarity",
        "matmul",
        "batch_matmul",
    }
)


def _parse_printed_value(out: str) -> Union[float, List[Any]]:
    s = out.strip()
    if not s:
        raise AssertionError("empty stdout")
    lines = [ln for ln in s.splitlines() if ln.strip()]
    if len(lines) != 1:
        raise AssertionError("expected single-line stdout, got %r" % s)
    tok = lines[0].strip()
    if tok.startswith("["):
        return ast.literal_eval(tok)  # type: ignore[return-value]
    return float(tok)


def _flatten(v: Any) -> List[float]:
    if isinstance(v, (int, float)):
        return [float(v)]
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
def test_print_at_quotient_vs_calculus(label: str, orig_source: str) -> None:
    if label in _QUOTIENT_SKIP:
        pytest.skip("quotient adapter uses @x but golden source has no `x` (pred/target/A/B only)")
    expected = _EXPECTED_DY_DX.get(label)
    if expected is None:
        pytest.fail("missing _EXPECTED_DY_DX for label %r (add entry or add to _QUOTIENT_SKIP)" % label)

    qsrc = _build_quotient_source(label, orig_source)
    c_ok, e_ok, out, err = compile_exec_capture_print_at(qsrc)
    assert c_ok, "%s: compile failed: %s" % (label, err)
    assert e_ok, "%s: exec failed: %s" % (label, err)
    got = _parse_printed_value(out)
    assert _approx_equal(got, expected), (
        "%s: print(@…/@x) got %r expected ∂y/∂x ≈ %s" % (label, out.strip(), expected)
    )


def test_quotient_cases_partition_print_at_golden() -> None:
    labels = {row[0] for row in GOLDEN_PRINT_CASES}
    covered = set(_EXPECTED_DY_DX) | set(_QUOTIENT_SKIP)
    assert covered == labels, (
        "partition mismatch: extra %s missing %s"
        % (sorted(covered - labels), sorted(labels - covered))
    )
