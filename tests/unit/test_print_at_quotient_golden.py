"""Numeric stdout checks for ``print(@num / @x)`` for each ``test_print_at_golden`` case.

**Scope:** every adapted program keeps **scalar** ``x`` (and scalar ``y`` / quotient), so a
single printed float is expected. Do **not** treat a rank-0 ``@…/@x`` result as correct when
``x`` is a tensor — see ``test_autodiff_tensor_quotient_reductions`` and
``_assert_quotient_not_scalar_rank0`` there.

``print(@y)`` is compile-time formatted to a string literal. ``print(@y / @x)`` lowers to a
runtime value (here a scalar because ``x`` is scalar), so we assert the printed float against
calculus.
"""

from __future__ import annotations

import math
import re
from typing import Dict, FrozenSet, Optional, Tuple

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


# ∂(num)/∂x at the primal values in each golden source (denominator is always scalar ``x``).
_EXPECTED_DY_DX: Dict[str, float] = {
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
}


_QUOTIENT_SKIP: FrozenSet[str] = frozenset(
    {
        "exp_einstein",
        "sum_reduction",
        "softmax_quotient",
        "einstein_square",
        "prod_reduction",
        "reduce_sum",
        "reduce_l1",
        "reduce_sum_square",
        "reduce_mean",
        "linear",
        "mse_loss",
        "mae_loss",
        "huber_loss",
        "binary_cross_entropy",
        "softmax",
        "log_softmax",
        "reduce_l2",
        "reduce_log_sum",
        "reduce_log_sum_exp",
        "cosine_similarity",
        "matmul",
        "batch_matmul",
    }
)


def _parse_printed_float(out: str) -> float:
    s = out.strip()
    if not s:
        raise AssertionError("empty stdout")
    lines = [ln for ln in s.splitlines() if ln.strip()]
    if len(lines) != 1:
        raise AssertionError("expected single line float stdout, got %r" % s)
    return float(lines[0].strip())


@pytest.mark.parametrize(
    "label,orig_source",
    [(row[0], row[1]) for row in GOLDEN_PRINT_CASES],
    ids=[row[0] for row in GOLDEN_PRINT_CASES],
)
def test_print_at_quotient_vs_calculus(label: str, orig_source: str) -> None:
    if label in _QUOTIENT_SKIP:
        pytest.skip("not a scalar ∂y/∂x with scalar x (tensor / reduction / loss layout)")
    expected = _EXPECTED_DY_DX.get(label)
    if expected is None:
        pytest.fail("missing _EXPECTED_DY_DX for label %r (add entry or add to _QUOTIENT_SKIP)" % label)

    qsrc = _build_quotient_source(label, orig_source)
    c_ok, e_ok, out, err = compile_exec_capture_print_at(qsrc)
    assert c_ok, "%s: compile failed: %s" % (label, err)
    assert e_ok, "%s: exec failed: %s" % (label, err)
    got = _parse_printed_float(out)
    assert got == pytest.approx(expected, rel=0.0, abs=1e-5), (
        "%s: print(@…/@x) got %r expected ∂y/∂x ≈ %s" % (label, out.strip(), expected)
    )


def test_quotient_cases_partition_print_at_golden() -> None:
    labels = {row[0] for row in GOLDEN_PRINT_CASES}
    covered = set(_EXPECTED_DY_DX) | set(_QUOTIENT_SKIP)
    assert covered == labels, (
        "partition mismatch: extra %s missing %s"
        % (sorted(covered - labels), sorted(labels - covered))
    )
