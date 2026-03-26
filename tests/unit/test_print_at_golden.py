"""Exact stdout goldens for ``print(@…)``.

``std::ml`` here is reductions, layers, matmul, softmax/loss-style calls, ``max_pool``;
MNIST-shaped conv is covered as Einstein sum-of-products (``mnist_conv2d``, same math as ``std::ml::conv`` 2D body).
Activation ``print(@…)`` goldens live in ``test_print_at_ml_smoke`` (disjoint ``std::ml`` symbols).
"""

from __future__ import annotations

from typing import List, Tuple

import pytest

from einlang.compiler.driver import CompilerDriver
from einlang.ir.nodes import BindingIR, BlockExpressionIR
from tests.print_at_fixtures import compile_capture_rewritten_print_at

_HUBER_LOSS_PRINT_AT_GOLDEN = """let @y = {
    let _@huber_loss_call: f32 = {
        let nb = len(pred) as f32;
        let nf = len(pred[0]) as f32;
        let diff[batch.0 in 0..targets.shape[0], j in 0..targets.shape[1]] = pred[batch.0, j] - target[batch.0, j];
        let abs_diff[batch.0 in 0..diff.shape[0], j in 0..diff.shape[1]] = abs(diff[batch.0, j]);
        let _@diff[batch.0 in 0..targets.shape[0], j in 0..targets.shape[1]] = @pred[batch.0, j] - @target[batch.0, j];
        let _@abs_diff[batch.0 in 0..diff.shape[0], j in 0..diff.shape[1]] = sign(diff[batch.0, j]) * _@diff[batch.0, j];
        let _@huber_elem[batch.0 in 0..abs_diff.shape[0], j in 0..abs_diff.shape[1]] = if abs_diff[batch.0, j] <= 1.0 { 0.5 * diff[batch.0, j] * _@diff[batch.0, j] + diff[batch.0, j] * 0.5 * _@diff[batch.0, j] } else { _@abs_diff[batch.0, j] };
        let _@row_huber[batch.0 in 0..huber_elem.shape[0]] = sum[j](_@huber_elem[batch.0, j]);
        let _@loss = sum[batch.0](_@row_huber[batch.0]) / (nb * nf);
        _@loss
    };
    _@huber_loss_call
};"""

GOLDEN_PRINT_CASES: List[Tuple[str, str, str]] = [
    (
        "constant",
        """
let x = 3.0;
let y = 5.0;
print(@y);
""",
        "let @y = 0.0;",
    ),
    (
        "identity",
        """
let x = 3.0;
let y = x;
print(@y);
""",
        "let @y = @x;",
    ),
    (
        "add",
        """
let x = 3.0;
let y = x + x;
print(@y);
""",
        "let @y = 2.0 * @x;",
    ),
    (
        "sub",
        """
let x = 3.0;
let y = x - 1.0;
print(@y);
""",
        "let @y = @x;",
    ),
    (
        "product",
        """
let x = 3.0;
let y = x * x;
print(@y);
""",
        "let @y = 2.0 * x * @x;",
    ),
    (
        "product_two_vars",
        """
let a = 3.0;
let b = 4.0;
let y = a * b;
print(@y);
""",
        "let @y = a * @b + b * @a;",
    ),
    (
        "quotient",
        """
let a = 3.0;
let b = 4.0;
let y = a / b;
print(@y);
""",
        "let @y = (b * @a - a * @b) / b ** 2.0;",
    ),
    (
        "power_const",
        """
let x = 2.0;
let y = x ** 3.0;
print(@y);
""",
        "let @y = 3.0 * x ** 2.0 * @x;",
    ),
    (
        "power_square",
        """
let x = 1.0;
let y = x ** 2;
print(@y);
""",
        "let @y = 2.0 * x * @x;",
    ),
    (
        "neg",
        """
let x = 3.0;
let y = -x;
print(@y);
""",
        "let @y = -@x;",
    ),
    (
        "chain_let",
        """
let x = 2.0;
let z = x * x;
let y = z + z;
print(@y);
""",
        "let @z = 2.0 * x * @x;\nlet @y = 2.0 * @z;",
    ),
    (
        "exp_scalar",
        """
let x = 1.0;
let y = std::math::exp(x);
print(@y);
""",
        "let @y = exp(x) * @x;",
    ),
    (
        "exp_einstein",
        """
let x = [1.0, 2.0, 3.0];
let e[i] = std::math::exp(x[i]);
print(@e);
""",
        "let @e[i] = exp(x[i]) * @x[i];",
    ),
    (
        "sum_reduction",
        """
let x = [1.0, 2.0, 3.0];
let e[i] = std::math::exp(x[i]);
let s = sum[k](e[k]);
print(@s);
""",
        "let @e[i] = exp(x[i]) * @x[i];\nlet @s = sum[k](@e[k]);",
    ),
    (
        "softmax_quotient",
        """
let x = [1.0, 2.0, 3.0];
let e[i] = std::math::exp(x[i]);
let s = sum[k](e[k]);
let y[i] = e[i] / s;
print(@y);
""",
        "let @e[i] = exp(x[i]) * @x[i];\n"
        "let @s = sum[k](@e[k]);\n"
        "let @y[i] = (s * @e[i] - e[i] * @s) / s ** 2.0;",
    ),
    (
        "if_else",
        """
let x = 3.0;
let y = if x > 0.0 { x } else { 0.0 };
print(@y);
""",
        "let @y = if x > 0.0 { @x } else { 0.0 };",
    ),
    (
        "scalar_mul",
        """
let x = 3.0;
let y = 2.0 * x;
print(@y);
""",
        "let @y = 2.0 * @x;",
    ),
    (
        "compound",
        """
let x = 3.0;
let y = x * x + x;
print(@y);
""",
        "let @y = 2.0 * x * @x + @x;",
    ),
    (
        "call_plus_fn",
        """
fn f(t) { t + 1.0 }
let x = 1.0;
let y = x + f(x);
print(@y);
""",
        "let @y = 2.0 * @x;",
    ),
    (
        "multistatement_callee_g",
        """
fn g(t) {
    let a = t + 1.0;
    let b = a * 2.0;
    b
}
let x = 3.0;
let y = g(x);
print(@y);
""",
        "let @y = {\n    let _@a = @x;\n    let _@b = 2.0 * _@a;\n    _@b\n};",
    ),
    (
        "log_scalar",
        """
let x = 2.0;
let y = std::math::log(x);
print(@y);
""",
        "let @y = 1.0 / x * @x;",
    ),
    (
        "sin_scalar",
        """
let x = 1.0;
let y = std::math::sin(x);
print(@y);
""",
        "let @y = cos(x) * @x;",
    ),
    (
        "cos_scalar",
        """
let x = 1.0;
let y = std::math::cos(x);
print(@y);
""",
        "let @y = -sin(x) * @x;",
    ),
    (
        "tan_scalar",
        """
let x = 0.5;
let y = std::math::tan(x);
print(@y);
""",
        "let @y = 1.0 / (cos(x) * cos(x)) * @x;",
    ),
    (
        "log1p_scalar",
        """
let x = 0.5;
let y = std::math::log1p(x);
print(@y);
""",
        "let @y = 1.0 / (1.0 + x) * @x;",
    ),
    (
        "expm1_scalar",
        """
let x = 0.5;
let y = std::math::expm1(x);
print(@y);
""",
        "let @y = exp(x) * @x;",
    ),
    (
        "atan_scalar",
        """
let x = 0.5;
let y = std::math::atan(x);
print(@y);
""",
        "let @y = 1.0 / (1.0 + x * x) * @x;",
    ),
    (
        "asin_scalar",
        """
let x = 0.5;
let y = std::math::asin(x);
print(@y);
""",
        "let @y = 1.0 / (1.0 - x * x) ** 0.5 * @x;",
    ),
    (
        "acos_scalar",
        """
let x = 0.5;
let y = std::math::acos(x);
print(@y);
""",
        "let @y = -1.0 / (1.0 - x * x) ** 0.5 * @x;",
    ),
    (
        "atan2_two_vars",
        """
let y = 1.0;
let x = 2.0;
let z = std::math::atan2(y, x);
print(@z);
""",
        "let @z = x / (x * x + y * y) * @y + -y / (x * x + y * y) * @x;",
    ),
    (
        "tanh_scalar",
        """
let x = 0.5;
let y = std::math::tanh(x);
print(@y);
""",
        "let @y = {\n    let _@tanh_x: f32 = if x as f32 >= 0.0 {\n        let t = exp(-2.0 * x);\n        let _@t = { exp(-2.0 * x) * -2.0 * @x };\n        ((1.0 + t) * (0.0 - _@t) - (1.0 - t) * _@t) / (1.0 + t) ** 2.0\n    } else {\n        let t = exp(2.0 * x);\n        let _@t = { exp(2.0 * x) * 2.0 * @x };\n        ((t + 1.0) * _@t - (t - 1.0) * _@t) / (t + 1.0) ** 2.0\n    };\n    _@tanh_x\n};",
    ),
    (
        "sinh_scalar",
        """
let x = 0.5;
let y = std::math::sinh(x);
print(@y);
""",
        "let @y = {\n    let _@sinh_x: f32 = {\n        let ax = abs(x);\n        let _@ax = { sign(x) * @x };\n        if ax < 20.0 { 0.5 * ({ exp(x) * @x } - { exp(-x) * -@x }) } else {\n            let s = if x as f32 >= 0.0 { 1.0 } else { -1.0 };\n            let _@s = if x as f32 >= 0.0 { 0.0 } else { 0.0 };\n            s * 0.5 * { exp(ax) * _@ax } + exp(ax) * 0.5 * _@s\n        }\n    };\n    _@sinh_x\n};",
    ),
    (
        "cosh_scalar",
        """
let x = 0.5;
let y = std::math::cosh(x);
print(@y);
""",
        "let @y = {\n    let _@cosh_x: f32 = {\n        let ax = abs(x);\n        let _@ax = { sign(x) * @x };\n        if ax < 20.0 { 0.5 * ({ exp(x) * @x } + { exp(-x) * -@x }) } else { 0.5 * { exp(ax) * _@ax } }\n    };\n    _@cosh_x\n};",
    ),
    (
        "asinh_scalar",
        """
let x = 0.5;
let y = std::math::asinh(x);
print(@y);
""",
        "let @y = 1.0 / (x + (x * x + 1.0) ** 0.5) * (@x + 0.5 * (x * x + 1.0) ** -0.5 * 2.0 * x * @x);",
    ),
    (
        "acosh_scalar",
        """
let x = 2.0;
let y = std::math::acosh(x);
print(@y);
""",
        "let @y = 1.0 / (x + (x * x - 1.0) ** 0.5) * (@x + 0.5 * (x * x - 1.0) ** -0.5 * 2.0 * x * @x);",
    ),
    (
        "atanh_scalar",
        """
let x = 0.5;
let y = std::math::atanh(x);
print(@y);
""",
        "let @y = 0.5 * { 1.0 / ((1.0 + x) / (1.0 - x)) * ((1.0 - x) * @x - (1.0 + x) * (0.0 - @x)) / (1.0 - x) ** 2.0 };",
    ),
    (
        "erf_scalar",
        """
let x = 1.0;
let y = std::math::erf(x);
print(@y);
""",
        "let @y = 2.0 / sqrt(pi()) * exp(0.0 - x * x) * @x;",
    ),
    (
        "abs_scalar",
        """
let x = 2.0;
let y = std::math::abs(x);
print(@y);
""",
        "let @y = sign(x) * @x;",
    ),
    (
        "sign_scalar",
        """
let x = 2.0;
let y = std::math::sign(x);
print(@y);
""",
        "let @y = {\n    let _@sign_x: f32 = if x as f32 > 0.0 { 0.0 } else if x as f32 < 0.0 { 0.0 } else { 0.0 };\n    _@sign_x\n};",
    ),
    (
        "min_scalar",
        """
let x = 2.0;
let y = std::math::min(x, 5.0);
print(@y);
""",
        "let @y = {\n    let _@min_call: f32 = if x < 5.0 { @x } else { 0.0 };\n    _@min_call\n};",
    ),
    (
        "max_scalar",
        """
let x = 2.0;
let y = std::math::max(x, 0.0);
print(@y);
""",
        "let @y = {\n    let _@max_call: f32 = if x > 0.0 { @x } else { 0.0 };\n    _@max_call\n};",
    ),
    (
        "reciprocal_scalar",
        """
let x = 2.0;
let y = std::math::reciprocal(x);
print(@y);
""",
        "let @y = (0.0 - @x) / x ** 2.0;",
    ),
    (
        "rsqrt_scalar",
        """
let x = 4.0;
let y = std::math::rsqrt(x);
print(@y);
""",
        "let @y = (0.0 - 0.5 * x ** -0.5 * @x) / (x ** 0.5) ** 2.0;",
    ),
    (
        "sqrt_via_pow",
        """
let x = 4.0;
let y = x ** 0.5;
print(@y);
""",
        "let @y = 0.5 * x ** -0.5 * @x;",
    ),
    (
        "mod_scalar",
        """
let x = 7.0;
let y = x % 3.0;
print(@y);
""",
        "let @y = @x;",
    ),
    (
        "quotient_chain",
        """
let x = 3.0;
let y = x / (x + 1.0);
print(@y);
""",
        "let @y = ((x + 1.0) * @x - x * @x) / (x + 1.0) ** 2.0;",
    ),
    (
        "einstein_square",
        """
let x = [1.0, 2.0, 3.0];
let t[i] = x[i] * x[i];
print(@t);
""",
        "let @t[i] = 2.0 * x[i] * @x[i];",
    ),
    (
        "prod_reduction",
        """
let x = [1.0, 2.0, 3.0];
let p = prod[j](x[j]);
print(@p);
""",
        "let @p = sum[j](prod[k](x[k]) / x[j] * @x[j]);",
    ),
    (
        "reduce_sum",
        """
use std::ml;
let x = [[1.0, 2.0, 3.0]];
let y = std::ml::reduce_sum(x);
print(@y);
""",
        "let @y = {\n    let _@reduce_sum_x: [f32; ?] = {\n        let _@result[batch.0 in 0..x.shape[0]] = sum[j](@x[batch.0, j]);\n        _@result\n    };\n    _@reduce_sum_x\n};",
    ),
    (
        "reduce_l1",
        """
use std::ml;
let x = [[1.0, -2.0, 3.0]];
let y = std::ml::reduce_l1(x);
print(@y);
""",
        "let @y = {\n    let _@reduce_l1_x: [f32; *] = {\n        let _@result[batch.0 in 0..x.shape[0]] = sum[j]({ sign(x[batch.0, j]) * @x[batch.0, j] });\n        _@result\n    };\n    _@reduce_l1_x\n};",
    ),
    (
        "reduce_sum_square",
        """
use std::ml;
let x = [[1.0, 2.0, 3.0]];
let y = std::ml::reduce_sum_square(x);
print(@y);
""",
        "let @y = {\n    let _@reduce_sum_square_x: [f32; *] = {\n        let _@result[batch.0 in 0..x.shape[0]] = sum[j](2.0 * x[batch.0, j] * @x[batch.0, j]);\n        _@result\n    };\n    _@reduce_sum_square_x\n};",
    ),
    (
        "reduce_mean",
        """
use std::ml;
let x = [[1.0, 2.0, 3.0]];
let y = std::ml::reduce_mean(x);
print(@y);
""",
        "let @y = {\n    let count = len(x[0]) as f32;\n    let _@sum_val[batch.0 in 0..x.shape[0]] = sum[j](@x[batch.0, j]);\n    let _@mean[batch.0 in 0..sum_val.shape[0]] = _@sum_val[batch.0] / count;\n    _@mean\n};",
    ),
    (
        "linear",
        """
use std::ml;
let x = [[1.0, 2.0]];
let W = [[0.5, 0.3], [0.2, 0.4]];
let b = [0.1, 0.2];
let y = std::ml::linear(x, W, b);
print(@y);
""",
        "let @y = {\n    let _@linear_call: [f32; ?, ?] = {\n        let _@output[batch.0 in 0..x.shape[0], j in 0..bias.shape[0]] = sum[k](x[batch.0, k] * @W[j, k] + W[j, k] * @x[batch.0, k]) + @b[j];\n        _@output\n    };\n    _@linear_call\n};",
    ),
    (
        "mnist_conv2d",
        """
let x = [[[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]]];
let w = [[[[1.0, 0.0], [0.0, 1.0]]]];
let b = [0.0];
let y[b in 0..1, co in 0..1, i in 0..2, j in 0..2] =
    sum[c in 0..1, m in 0..2, n in 0..2](x[b, c, i + m, j + n] * w[co, c, m, n]) + b[co];
print(@y);
""",
        "let @y[b, co, i, j] = sum[c, m, n](x[b, c, i + m, j + n] * @w[co, c, m, n] + w[co, c, m, n] * @x[b, c, i + m, j + n]) + @b[co];",
    ),
    (
        "max_pool",
        """
use std::ml;
let x = [[[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]]];
let y = std::ml::max_pool(x, [2, 2], [2, 2], [0, 0]);
print(@y);
""",
        'let @y = {\n    let _@max_pool_call: [f32; ?, ?, ?, ?] = {\n        let rank = len([2, 2]);\n        if rank == 1 { {\n            let _@output[_ad_0 in 0..?.shape[0], _ad_1 in 0..?.shape[1], _ad_2 in 0..?.shape[2], _ad_3 in 0..?.shape[3]] = sum[batch.0, batch.1, c, i](if i * [2, 2][0] - [0, 0][0] + m == _ad_3 in 0..?.shape[3] if c == _ad_2 in 0..?.shape[2] if batch.1 == _ad_1 in 0..?.shape[1] if batch.0 == _ad_0 in 0..?.shape[0] 1.0 else 0.0 else 0.0 else 0.0 else 0.0 at argmax[m](x[batch.0, batch.1, c, i * [2, 2][0] - [0, 0][0] + m]));\n            _@output\n        } } else if rank == 2 { {\n            let _@output[_ad_0 in 0..?.shape[0], _ad_1 in 0..?.shape[1], _ad_2 in 0..?.shape[2], _ad_3 in 0..?.shape[3]] = sum[batch.0, c, i, j](if j * [2, 2][1] - [0, 0][1] + n == _ad_3 in 0..?.shape[3] if i * [2, 2][0] - [0, 0][0] + m == _ad_2 in 0..?.shape[2] if c == _ad_1 in 0..?.shape[1] if batch.0 == _ad_0 in 0..?.shape[0] 1.0 else 0.0 else 0.0 else 0.0 else 0.0 at argmax[m, n](x[batch.0, c, i * [2, 2][0] - [0, 0][0] + m, j * [2, 2][1] - [0, 0][1] + n]));\n            _@output\n        } } else if rank == 3 { {\n            let _@output[_ad_0 in 0..?.shape[0], _ad_1 in 0..?.shape[1], _ad_2 in 0..?.shape[2], _ad_3 in 0..?.shape[3], _ad_4 in 0..?.shape[4]] = sum[batch.0, c, i, j, k](if k * [2, 2][2] - [0, 0][2] + p == _ad_4 in 0..?.shape[4] if j * [2, 2][1] - [0, 0][1] + n == _ad_3 in 0..?.shape[3] if i * [2, 2][0] - [0, 0][0] + m == _ad_2 in 0..?.shape[2] if c == _ad_1 in 0..?.shape[1] if batch.0 == _ad_0 in 0..?.shape[0] 1.0 else 0.0 else 0.0 else 0.0 else 0.0 else 0.0 at argmax[m, n, p](x[batch.0, c, i * [2, 2][0] - [0, 0][0] + m, j * [2, 2][1] - [0, 0][1] + n, k * [2, 2][2] - [0, 0][2] + p]));\n            _@output\n        } } else { @x }\n    };\n    _@max_pool_call\n};',
    ),
    (
        "mse_loss",
        """
use std::ml;
let pred = [[1.0, 2.0, 3.0]];
let target = [[1.5, 2.5, 3.5]];
let y = std::ml::mse_loss(pred, target);
print(@y);
""",
        "let @y = {\n    let _@mse_loss_call: f32 = {\n        let nb = len(pred) as f32;\n        let nf = len(pred[0]) as f32;\n        let _@row_sse[batch.0 in 0..targets.shape[0]] = sum[j](2.0 * (pred[batch.0, j] - target[batch.0, j]) * (@pred[batch.0, j] - @target[batch.0, j]));\n        let _@loss = sum[batch.0](_@row_sse[batch.0]) / (nb * nf);\n        _@loss\n    };\n    _@mse_loss_call\n};",
    ),
    (
        "mae_loss",
        """
use std::ml;
let pred = [[1.0, 2.0, 3.0]];
let target = [[1.5, 2.5, 3.5]];
let y = std::ml::mae_loss(pred, target);
print(@y);
""",
        "let @y = {\n    let _@mae_loss_call: f32 = {\n        let nb = len(pred) as f32;\n        let nf = len(pred[0]) as f32;\n        let _@row_l1[batch.0 in 0..targets.shape[0]] = sum[j]({ sign(pred[batch.0, j] - target[batch.0, j]) * (@pred[batch.0, j] - @target[batch.0, j]) });\n        let _@loss = sum[batch.0](_@row_l1[batch.0]) / (nb * nf);\n        _@loss\n    };\n    _@mae_loss_call\n};",
    ),
    (
        "huber_loss",
        """
use std::ml;
let pred = [[1.0, 2.0, 3.0]];
let target = [[1.5, 2.5, 3.5]];
let y = std::ml::huber_loss(pred, target, 1.0);
print(@y);
""",
        _HUBER_LOSS_PRINT_AT_GOLDEN,
    ),
    (
        "binary_cross_entropy",
        """
use std::ml;
let pred = [[0.8, 0.3, 0.9]];
let target = [[1.0, 0.0, 1.0]];
let y = std::ml::binary_cross_entropy(pred, target);
print(@y);
""",
        "let @y = {\n    let _@binary_cross_entropy_call: f32 = {\n        let eps = 1e-07;\n        let clipped_pred[batch.0 in 0..predictions.shape[0], j in 0..predictions.shape[1]] = if pred[batch.0, j] < eps { eps } else if pred[batch.0, j] > 1.0 - eps { 1.0 - eps } else { pred[batch.0, j] };\n        let nb = len(pred) as f32;\n        let nf = len(pred[0]) as f32;\n        let _@clipped_pred[batch.0 in 0..predictions.shape[0], j in 0..predictions.shape[1]] = if pred[batch.0, j] < eps { 0.0 } else if pred[batch.0, j] > 1.0 - eps { 0.0 } else { @pred[batch.0, j] };\n        let _@row_bce[batch.0 in 0..clipped_pred.shape[0]] = -sum[j](target[batch.0, j] * { 1.0 / clipped_pred[batch.0, j] * _@clipped_pred[batch.0, j] } + ln(clipped_pred[batch.0, j]) * @target[batch.0, j] + (1.0 - target[batch.0, j]) * { 1.0 / (1.0 - clipped_pred[batch.0, j]) * (0.0 - _@clipped_pred[batch.0, j]) } + ln(1.0 - clipped_pred[batch.0, j]) * (0.0 - @target[batch.0, j]));\n        let _@loss = sum[batch.0](_@row_bce[batch.0]) / (nb * nf);\n        _@loss\n    };\n    _@binary_cross_entropy_call\n};",
    ),
    (
        "softmax",
        """
use std::ml;
let x = [[1.0, 2.0, 3.0]];
let y = std::ml::softmax(x);
print(@y);
""",
        "let @y = {\n    let max_val[batch.0 in 0..x.shape[0]] = max[j](x[batch.0, j]);\n    let shifted[batch.0 in 0..max_val.shape[0], j in 0..x.shape[1]] = x[batch.0, j] - max_val[batch.0];\n    let exp_vals[batch.0 in 0..shifted.shape[0], j in 0..shifted.shape[1]] = exp(shifted[batch.0, j]);\n    let sums[batch.0 in 0..exp_vals.shape[0]] = sum[k](exp_vals[batch.0, k]);\n    let _@max_val[batch.0 in 0..x.shape[0]] = @x[batch.0, j] at argmax[j](x[batch.0, j]);\n    let _@shifted[batch.0 in 0..max_val.shape[0], j in 0..x.shape[1]] = @x[batch.0, j] - _@max_val[batch.0];\n    let _@exp_vals[batch.0 in 0..shifted.shape[0], j in 0..shifted.shape[1]] = exp(shifted[batch.0, j]) * _@shifted[batch.0, j];\n    let _@sums[batch.0 in 0..exp_vals.shape[0]] = sum[k](_@exp_vals[batch.0, k]);\n    let _@output[batch.0 in 0..sums.shape[0], j in 0..exp_vals.shape[1]] = (sums[batch.0] * _@exp_vals[batch.0, j] - exp_vals[batch.0, j] * _@sums[batch.0]) / sums[batch.0] ** 2.0;\n    _@output\n};",
    ),
    (
        "log_softmax",
        """
use std::ml;
let x = [[1.0, 2.0, 3.0]];
let y = std::ml::log_softmax(x);
print(@y);
""",
        "let @y = {\n    let max_val[batch.0 in 0..x.shape[0]] = max[j](x[batch.0, j]);\n    let shifted[batch.0 in 0..max_val.shape[0], j in 0..x.shape[1]] = x[batch.0, j] - max_val[batch.0];\n    let exp_vals[batch.0 in 0..shifted.shape[0], j in 0..shifted.shape[1]] = exp(shifted[batch.0, j]);\n    let sum_exp[batch.0 in 0..exp_vals.shape[0]] = sum[k](exp_vals[batch.0, k]);\n    let _@max_val[batch.0 in 0..x.shape[0]] = @x[batch.0, j] at argmax[j](x[batch.0, j]);\n    let _@shifted[batch.0 in 0..max_val.shape[0], j in 0..x.shape[1]] = @x[batch.0, j] - _@max_val[batch.0];\n    let _@exp_vals[batch.0 in 0..shifted.shape[0], j in 0..shifted.shape[1]] = exp(shifted[batch.0, j]) * _@shifted[batch.0, j];\n    let _@sum_exp[batch.0 in 0..exp_vals.shape[0]] = sum[k](_@exp_vals[batch.0, k]);\n    let _@log_sum[batch.0 in 0..max_val.shape[0]] = { 1.0 / sum_exp[batch.0] * _@sum_exp[batch.0] } + _@max_val[batch.0];\n    let _@output[batch.0 in 0..log_sum.shape[0], j in 0..x.shape[1]] = @x[batch.0, j] - _@log_sum[batch.0];\n    _@output\n};",
    ),
    (
        "reduce_l2",
        """
use std::ml;
let x = [[3.0, 4.0]];
let y = std::ml::reduce_l2(x);
print(@y);
""",
        "let @y = {\n    let sum_squares[batch.0 in 0..x.shape[0]] = sum[j](x[batch.0, j] ** 2.0);\n    let _@sum_squares[batch.0 in 0..x.shape[0]] = sum[j](2.0 * x[batch.0, j] * @x[batch.0, j]);\n    let _@result[batch.0 in 0..sum_squares.shape[0]] = 0.5 * sum_squares[batch.0] ** -0.5 * _@sum_squares[batch.0];\n    _@result\n};",
    ),
    (
        "reduce_log_sum",
        """
use std::ml;
let x = [[1.0, 2.0, 3.0]];
let y = std::ml::reduce_log_sum(x);
print(@y);
""",
        "let @y = {\n    let sum_val[batch.0 in 0..x.shape[0]] = sum[j](x[batch.0, j]);\n    let _@sum_val[batch.0 in 0..x.shape[0]] = sum[j](@x[batch.0, j]);\n    let _@result[batch.0 in 0..sum_val.shape[0]] = 1.0 / sum_val[batch.0] * _@sum_val[batch.0];\n    _@result\n};",
    ),
    (
        "reduce_log_sum_exp",
        """
use std::ml;
let x = [[1.0, 2.0, 3.0]];
let y = std::ml::reduce_log_sum_exp(x);
print(@y);
""",
        "let @y = {\n    let max_val[batch.0 in 0..x.shape[0]] = max[j](x[batch.0, j]);\n    let shifted[batch.0 in 0..max_val.shape[0], j in 0..x.shape[1]] = x[batch.0, j] - max_val[batch.0];\n    let sum_exp[batch.0 in 0..shifted.shape[0]] = sum[j](exp(shifted[batch.0, j]));\n    let _@max_val[batch.0 in 0..x.shape[0]] = @x[batch.0, j] at argmax[j](x[batch.0, j]);\n    let _@shifted[batch.0 in 0..max_val.shape[0], j in 0..x.shape[1]] = @x[batch.0, j] - _@max_val[batch.0];\n    let _@sum_exp[batch.0 in 0..shifted.shape[0]] = sum[j]({ exp(shifted[batch.0, j]) * _@shifted[batch.0, j] });\n    let _@result[batch.0 in 0..sum_exp.shape[0]] = _@max_val[batch.0] + { 1.0 / sum_exp[batch.0] * _@sum_exp[batch.0] };\n    _@result\n};",
    ),
    (
        "cosine_similarity",
        """
use std::ml;
let a = [[1.0, 2.0, 3.0]];
let b = [[4.0, 5.0, 6.0]];
let y = std::ml::cosine_similarity(a, b);
print(@y);
""",
        "let @y = {\n    let _@cosine_similarity_call: [f32; ?] = {\n        let dot_product[batch.0 in 0..b.shape[0]] = sum[j](a[batch.0, j] * b[batch.0, j]);\n        let norm_a_sq[batch.0 in 0..a.shape[0]] = sum[j](a[batch.0, j] * a[batch.0, j]);\n        let norm_b_sq[batch.0 in 0..b.shape[0]] = sum[j](b[batch.0, j] * b[batch.0, j]);\n        let norm_a[batch.0 in 0..norm_a_sq.shape[0]] = sqrt(norm_a_sq[batch.0]);\n        let norm_b[batch.0 in 0..norm_b_sq.shape[0]] = sqrt(norm_b_sq[batch.0]);\n        let _@dot_product[batch.0 in 0..b.shape[0]] = sum[j](a[batch.0, j] * @b[batch.0, j] + b[batch.0, j] * @a[batch.0, j]);\n        let _@norm_a_sq[batch.0 in 0..a.shape[0]] = sum[j](2.0 * a[batch.0, j] * @a[batch.0, j]);\n        let _@norm_b_sq[batch.0 in 0..b.shape[0]] = sum[j](2.0 * b[batch.0, j] * @b[batch.0, j]);\n        let _@norm_a[batch.0 in 0..norm_a_sq.shape[0]] = 0.5 * norm_a_sq[batch.0] ** -0.5 * _@norm_a_sq[batch.0];\n        let _@norm_b[batch.0 in 0..norm_b_sq.shape[0]] = 0.5 * norm_b_sq[batch.0] ** -0.5 * _@norm_b_sq[batch.0];\n        let _@similarity[batch.0 in 0..norm_b.shape[0]] = (norm_a[batch.0] * norm_b[batch.0] * _@dot_product[batch.0] - dot_product[batch.0] * (norm_a[batch.0] * _@norm_b[batch.0] + norm_b[batch.0] * _@norm_a[batch.0])) / (norm_a[batch.0] * norm_b[batch.0]) ** 2.0;\n        _@similarity\n    };\n    _@cosine_similarity_call\n};",
    ),
    (
        "matmul",
        """
use std::ml;
let A = [[1.0, 2.0], [3.0, 4.0]];
let B = [[5.0, 6.0], [7.0, 8.0]];
let C = std::ml::matmul(A, B);
print(@C);
""",
        "let @C = {\n    let _@matmul_call: [f32; ?, ?] = {\n        let _@output[i in 0..a.shape[0], j in 0..b.shape[1]] = sum[k](A[i, k] * @B[k, j] + B[k, j] * @A[i, k]);\n        _@output\n    };\n    _@matmul_call\n};",
    ),
    (
        # Golden string is print(@C); math: @C[b,i,j]=Σ_k(A[b,i,k]@B[b,k,j]+B[b,k,j]@A[b,i,k]).
        "batch_matmul",
        """
use std::ml;
let A = [[[1.0, 2.0], [3.0, 4.0]], [[0.5, 0.5], [0.1, 0.2]]];
let B = [[[5.0, 6.0], [7.0, 8.0]], [[1.0, 1.0], [1.0, 1.0]]];
let C = std::ml::batch_matmul(A, B);
print(@C);
""",
        "let @C = {\n    let _@batch_matmul_call: [f32; ?, ?, ?] = {\n        let _@result[batch.0 in 0..b.shape[0], i in 0..a.shape[1], j in 0..b.shape[2]] = sum[k](A[batch.0, i, k] * @B[batch.0, k, j] + B[batch.0, k, j] * @A[batch.0, i, k]);\n        _@result\n    };\n    _@batch_matmul_call\n};",
    ),
]


class TestPrintAtGolden:
    @pytest.mark.parametrize(
        "label,source,expected",
        [(row[0], row[1], row[2]) for row in GOLDEN_PRINT_CASES],
        ids=[row[0] for row in GOLDEN_PRINT_CASES],
    )
    def test_golden_stdout(
        self,
        label: str,
        source: str,
        expected: str,
        session_compiler,
    ) -> None:
        c_ok, out, err = compile_capture_rewritten_print_at(
            source, compiler=session_compiler
        )
        assert c_ok, "%s: compile failed: %s" % (label, err)
        assert out == expected, "%s: got %r, expected %r" % (label, out, expected)


class TestPrintAtCalleeTangentIr:
    def test_multistatement_callee_tangent_stays_block_not_inlined(self) -> None:
        """Callees with 2+ lets keep ∂y as BlockExpressionIR, not one big expr."""
        compiler = CompilerDriver()
        source = """
fn g(t) {
    let a = t + 1.0;
    let b = a * 2.0;
    b
}
let x = 3.0;
let y = g(x);
print(@y);
"""
        result = compiler.compile(
            source.strip(), source_file="<test>", stop_after_pass="AutodiffPass"
        )
        assert result.success, result.get_errors() or "compile failed"
        d_binding = next(
            (
                b
                for b in (result.ir.bindings or [])
                if isinstance(b, BindingIR) and b.name == "@y"
            ),
            None,
        )
        assert d_binding is not None, "expected @y binding after AutodiffPass"
        assert isinstance(d_binding.expr, BlockExpressionIR), (
            "expected multi-let callee tangent to stay a block, got %s"
            % type(d_binding.expr).__name__
        )
        n_lets = sum(
            1 for s in (d_binding.expr.statements or []) if isinstance(s, BindingIR)
        )
        assert n_lets >= 2, "expected at least 2 ∂ lets in block, got %s" % n_lets


def test_print_at_calculus_catalog_covers_all_goldens() -> None:
    from tests.print_at_calculus_catalog import GOLDEN_CALCULUS
    from tests.unit.test_print_at_ml_smoke import ML_ACTIVATION_PRINT_AT_GOLDEN_CASES

    labels = {row[0] for row in GOLDEN_PRINT_CASES}
    labels |= {row[0] for row in ML_ACTIVATION_PRINT_AT_GOLDEN_CASES}
    missing = sorted(labels - GOLDEN_CALCULUS.keys())
    extra = sorted(set(GOLDEN_CALCULUS.keys()) - labels)
    assert not missing, "GOLDEN_CALCULUS missing: %s" % missing
    assert not extra, "GOLDEN_CALCULUS has unknown keys: %s" % extra
