"""Exact stdout goldens for ``std::ml`` activation ``print(@…)``."""

from __future__ import annotations
import re
from pathlib import Path
from typing import List, Set, Tuple

import pytest

from tests.print_at_fixtures import compile_capture_rewritten_print_at

ML_ACTIVATION_PRINT_AT_GOLDEN_CASES: List[Tuple[str, str, str]] = [
    (
        'relu',
        '\n\nuse std::ml;\nlet x = 1.0;\nlet y = std::ml::relu(x);\nprint(@y);\n',
        'let @y = {\n    let _@relu_x: f32 = if x > 0 { @x } else { 0.0 };\n    _@relu_x\n};',
    ),
    (
        'sigmoid',
        '\n\nuse std::ml;\nlet x = 0.5;\nlet y = std::ml::sigmoid(x);\nprint(@y);\n',
        'let @y = (0.0 - exp(-x) * -@x) / (1.0 + exp(-x)) ** 2.0;',
    ),
    (
        'leaky_relu',
        '\n\nuse std::ml;\nlet x = 1.0;\nlet y = std::ml::leaky_relu(x, 0.01);\nprint(@y);\n',
        'let @y = {\n    let _@leaky_relu_call: f32 = if x > 0 { @x } else { 0.01 * @x };\n    _@leaky_relu_call\n};',
    ),
    (
        'elu',
        '\n\nuse std::ml;\nlet x = 1.0;\nlet y = std::ml::elu(x, 1.0);\nprint(@y);\n',
        'let @y = {\n    let _@elu_call: f32 = if x > 0 { @x } else { exp(x) * @x };\n    _@elu_call\n};',
    ),
    (
        'swish',
        '\n\nuse std::ml;\nlet x = 1.0;\nlet y = std::ml::swish(x);\nprint(@y);\n',
        'let @y = x * (0.0 - exp(-x) * -@x) / (1.0 + exp(-x)) ** 2.0 + sigmoid(x) * @x;',
    ),
    (
        'softsign',
        '\n\nuse std::ml;\nlet x = 1.0;\nlet y = std::ml::softsign(x);\nprint(@y);\n',
        'let @y = ((1.0 + abs(x)) * @x - x * sign(x) * @x) / (1.0 + abs(x)) ** 2.0;',
    ),
    (
        'hardtanh',
        '\n\nuse std::ml;\nlet x = 0.5;\nlet y = std::ml::hardtanh(x, -1.0, 1.0);\nprint(@y);\n',
        'let @y = {\n    let _@hardtanh_call: f32 = if x < -1.0 { 0.0 } else if x > 1.0 { 0.0 } else { @x };\n    _@hardtanh_call\n};',
    ),
    (
        'relu6',
        '\n\nuse std::ml;\nlet x = 3.0;\nlet y = std::ml::relu6(x);\nprint(@y);\n',
        'let @y = {\n    let _@relu6_x: i32 = if x <= 0 { 0.0 } else if x >= 6.0 { 0.0 } else { @x };\n    _@relu6_x\n};',
    ),
    (
        'prelu',
        '\n\nuse std::ml;\nlet x = 1.0;\nlet y = std::ml::prelu(x, 0.1);\nprint(@y);\n',
        'let @y = {\n    let _@prelu_call: f32 = if x > 0 { @x } else { 0.1 * @x };\n    _@prelu_call\n};',
    ),
    (
        'elu_alpha',
        '\n\nuse std::ml;\nlet x = 1.0;\nlet y = std::ml::elu_alpha(x, 0.5);\nprint(@y);\n',
        'let @y = {\n    let _@elu_alpha_call: f32 = if x > 0 { @x } else { 0.5 * { exp(x) * @x } };\n    _@elu_alpha_call\n};',
    ),
    (
        'celu',
        '\n\nuse std::ml;\nlet x = 1.0;\nlet y = std::ml::celu(x, 1.0);\nprint(@y);\n',
        'let @y = {\n    let _@celu_call: f32 = if x > 0 { @x } else { exp(x / 1.0) * @x / 1.0 ** 2.0 };\n    _@celu_call\n};',
    ),
    (
        'softshrink',
        '\n\nuse std::ml;\nlet x = 2.0;\nlet y = std::ml::softshrink(x, 0.5);\nprint(@y);\n',
        'let @y = {\n    let _@softshrink_call: f32 = if x > 0.5 { @x } else if x < -0.5 { @x } else { 0.0 };\n    _@softshrink_call\n};',
    ),
    (
        'hardshrink',
        '\n\nuse std::ml;\nlet x = 2.0;\nlet y = std::ml::hardshrink(x, 0.5);\nprint(@y);\n',
        'let @y = {\n    let _@hardshrink_call: f32 = {\n        let abs_x: f32 = if x < 0 { -x } else { x };\n        let _@abs_x: f32 = if x < 0 { -@x } else { @x };\n        if abs_x > 0.5 { @x } else { 0.0 }\n    };\n    _@hardshrink_call\n};',
    ),
    (
        'threshold',
        '\n\nuse std::ml;\nlet x = 2.0;\nlet y = std::ml::threshold(x, 0.5, 0.0);\nprint(@y);\n',
        'let @y = {\n    let _@threshold_call: f32 = if x <= 0.5 { 0.0 } else { @x };\n    _@threshold_call\n};',
    ),
    (
        'hardswish',
        '\n\nuse std::ml;\nlet x = 1.0;\nlet y = std::ml::hardswish(x);\nprint(@y);\n',
        'let @y = {\n    let shifted: f32 = x + 3.0;\n    let clamped: f32 = if shifted <= 0 { 0.0 } else if shifted >= 6.0 { 6.0 } else { shifted };\n    let _@shifted: f32 = @x;\n    let _@clamped: f32 = if shifted <= 0 { 0.0 } else if shifted >= 6.0 { 0.0 } else { _@shifted };\n    (x * _@clamped + clamped * @x) / 6.0\n};',
    ),
    (
        'thresholded_relu',
        '\n\nuse std::ml;\nlet x = 2.0;\nlet y = std::ml::thresholded_relu(x, 0.5);\nprint(@y);\n',
        'let @y = {\n    let _@thresholded_relu_call: f32 = if x > 0.5 { @x } else { 0.0 };\n    _@thresholded_relu_call\n};',
    ),
    (
        'selu',
        '\n\nuse std::ml;\nlet x = 1.0;\nlet y = std::ml::selu(x);\nprint(@y);\n',
        'let @y = {\n    let lambda: f32 = 1.0507009873554805;\n    let alpha: f32 = 1.6732632423543772;\n    if x > 0 { lambda * @x } else { lambda * alpha * { exp(x) * @x } }\n};',
    ),
    (
        'hardsigmoid',
        '\n\nuse std::ml;\nlet x = 0.5;\nlet y = std::ml::hardsigmoid(x);\nprint(@y);\n',
        'let @y = {\n    let shifted: f32 = x + 3.0;\n    let _@shifted: f32 = @x;\n    let _@clamped: f32 = if shifted < 0 { 0.0 } else if shifted > 6.0 { 0.0 } else { _@shifted };\n    _@clamped / 6.0\n};',
    ),
    (
        'softplus',
        '\n\nuse std::ml;\nlet x = 1.0;\nlet y = std::ml::softplus(x);\nprint(@y);\n',
        'let @y = {\n    let _@softplus_x: f32 = if x > 20.0 { @x } else { 1.0 / (1.0 + exp(x)) * { exp(x) * @x } };\n    _@softplus_x\n};',
    ),
    (
        'gelu',
        '\n\nuse std::ml;\nlet x = 1.0;\nlet y = std::ml::gelu(x);\nprint(@y);\n',
        'let @y = {\n    let sqrt_2_over_pi: f32 = 0.7978845608028654;\n    let coeff: f32 = 0.044715;\n    let inner: f32 = sqrt_2_over_pi * (x + coeff * x * x * x);\n    let _@inner = sqrt_2_over_pi * (@x + coeff * (x * x * @x + x * 2.0 * x * @x));\n    0.5 * x * if inner >= 0 {\n        let t: f32 = exp(-2.0 * inner);\n        let _@t: f32 = { exp(-2.0 * inner) * -2.0 * _@inner };\n        ((1.0 + t) * (0.0 - _@t) - (1.0 - t) * _@t) / (1.0 + t) ** 2.0\n    } else {\n        let t: f32 = exp(2.0 * inner);\n        let _@t: f32 = { exp(2.0 * inner) * 2.0 * _@inner };\n        ((t + 1.0) * _@t - (t - 1.0) * _@t) / (t + 1.0) ** 2.0\n    } + (1.0 + tanh(inner)) * 0.5 * @x\n};',
    ),
    (
        'mish',
        '\n\nuse std::ml;\nlet x = 1.0;\nlet y = std::ml::mish(x);\nprint(@y);\n',
        'let @y = {\n    let _@mish_x: f32 = {\n        let sp: f32 = softplus(x);\n        let _@sp: f32 = if x > 20.0 { @x } else { 1.0 / (1.0 + exp(x)) * { exp(x) * @x } };\n        x * if sp >= 0 {\n            let t: f32 = exp(-2.0 * sp);\n            let _@t: f32 = { exp(-2.0 * sp) * -2.0 * _@sp };\n            ((1.0 + t) * (0.0 - _@t) - (1.0 - t) * _@t) / (1.0 + t) ** 2.0\n        } else {\n            let t: f32 = exp(2.0 * sp);\n            let _@t: f32 = { exp(2.0 * sp) * 2.0 * _@sp };\n            ((t + 1.0) * _@t - (t - 1.0) * _@t) / (t + 1.0) ** 2.0\n        } + tanh(sp) * @x\n    };\n    _@mish_x\n};',
    ),
    (
        'tanhshrink',
        '\n\nuse std::ml;\nlet x = 0.5;\nlet y = std::ml::tanhshrink(x);\nprint(@y);\n',
        'let @y = {\n    let _@tanhshrink_x: f32 = @x - if x >= 0 {\n        let t: f32 = exp(-2.0 * x);\n        let _@t: f32 = { exp(-2.0 * x) * -2.0 * @x };\n        ((1.0 + t) * (0.0 - _@t) - (1.0 - t) * _@t) / (1.0 + t) ** 2.0\n    } else {\n        let t: f32 = exp(2.0 * x);\n        let _@t: f32 = { exp(2.0 * x) * 2.0 * @x };\n        ((t + 1.0) * _@t - (t - 1.0) * _@t) / (t + 1.0) ** 2.0\n    };\n    _@tanhshrink_x\n};',
    ),
]


def _std_ml_call_names_in_file(path: Path) -> Set[str]:
    text = path.read_text(encoding="utf-8")
    return set(re.findall(r"std::ml::([a-zA-Z_][a-zA-Z0-9_]*)", text))


def test_std_ml_disjoint_between_golden_and_smoke() -> None:
    here = Path(__file__).resolve().parent
    golden = here / "test_print_at_golden.py"
    smoke = here / "test_print_at_ml_smoke.py"
    g_names = _std_ml_call_names_in_file(golden)
    s_names = _std_ml_call_names_in_file(smoke)
    inter = g_names & s_names
    assert not inter, "std::ml overlap between print-at golden and smoke: %r" % (sorted(inter),)


class TestPrintAtMlSmoke:
    @pytest.mark.parametrize(
        "op_name,source,expected",
        [(row[0], row[1], row[2]) for row in ML_ACTIVATION_PRINT_AT_GOLDEN_CASES],
        ids=[row[0] for row in ML_ACTIVATION_PRINT_AT_GOLDEN_CASES],
    )
    def test_ml_activation_golden_stdout(
        self, op_name: str, source: str, expected: str, session_compiler
    ) -> None:
        c_ok, out, err = compile_capture_rewritten_print_at(
            source, compiler=session_compiler
        )
        assert c_ok, "%s: compile failed: %s" % (op_name, err)
        assert out == expected, "%s: got %r, expected %r" % (op_name, out, expected)
