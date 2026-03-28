"""Compile-time autodiff coverage for every ``std::ml`` export.

This file is intentionally broad and shallow: it uses tiny ``print(@...)`` programs
to ensure the autodiff pass is exercised for every ``std::ml`` export re-exported
from ``stdlib/ml/mod.ein``. Deeper numeric/runtime assertions stay in the existing
autodiff and print-at golden tests.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from einlang.compiler.driver import CompilerDriver
from tests.unit.test_print_at_golden import GOLDEN_PRINT_CASES
from tests.unit.test_print_at_ml_smoke import ML_ACTIVATION_PRINT_AT_GOLDEN_CASES

_REPO_ROOT = Path(__file__).resolve().parents[2]
_ML_MOD = _REPO_ROOT / "stdlib" / "ml" / "mod.ein"

_STD_ML_EXPORTS = [
    "conv",
    "conv_transpose",
    "depthwise_conv",
    "qconv",
    "max_pool",
    "average_pool",
    "global_average_pool",
    "global_max_pool",
    "lp_pool",
    "max_roi_pool",
    "batch_normalization",
    "instance_normalization",
    "layer_normalization",
    "lrn",
    "lp_normalization",
    "mean_variance_normalization",
    "relu",
    "sigmoid",
    "softmax",
    "log_softmax",
    "leaky_relu",
    "elu",
    "elu_alpha",
    "gelu",
    "swish",
    "selu",
    "softplus",
    "hardtanh",
    "relu6",
    "prelu",
    "celu",
    "gelu_tanh",
    "mish",
    "softsign",
    "tanhshrink",
    "softshrink",
    "hardshrink",
    "threshold",
    "hardswish",
    "thresholded_relu",
    "hardsigmoid",
    "linear",
    "gemm",
    "conv2d",
    "add",
    "subtract",
    "multiply",
    "divide",
    "power",
    "square",
    "neg",
    "abs",
    "sign",
    "min",
    "max",
    "reciprocal",
    "sqrt",
    "rsqrt",
    "exp",
    "log",
    "log1p",
    "expm1",
    "floor",
    "ceil",
    "round",
    "clip",
    "mod",
    "fmod",
    "sin",
    "cos",
    "tan",
    "tanh",
    "sinh",
    "cosh",
    "atan2",
    "erf",
    "is_nan",
    "is_inf",
    "einsum",
    "equal",
    "greater",
    "greater_or_equal",
    "less",
    "less_or_equal",
    "not_equal",
    "not",
    "logical_and",
    "logical_or",
    "logical_xor",
    "logical_not",
    "reduce_mean",
    "reduce_sum",
    "reduce_max",
    "reduce_min",
    "reduce_l1",
    "reduce_l2",
    "reduce_sum_square",
    "reduce_log_sum",
    "reduce_log_sum_exp",
    "reduce_prod",
    "matmul",
    "batch_matmul",
    "vec_norm_2",
    "matrix_norm_frobenius",
    "cholesky",
    "solve_triangular_lower",
    "solve_triangular_lower_unit",
    "solve_triangular_upper",
    "solve_cholesky",
    "where",
    "identity",
    "constant",
    "dropout",
    "l2_normalize",
    "numel",
    "size",
    "cast",
    "slice",
    "cumsum",
    "quantize_linear",
    "dequantize_linear",
    "qmatmul",
    "qlinear",
    "image_scaler",
    "eye",
    "diag_extract",
    "diag_construct",
    "trace",
    "frobenius_norm",
    "outer",
    "kron",
    "tril",
    "triu",
    "roll",
    "repeat_interleave",
    "flip",
    "cross_entropy_loss",
    "mse_loss",
    "mae_loss",
    "huber_loss",
    "binary_cross_entropy",
    "softmax_cross_entropy_loss",
    "cosine_similarity",
    "gather",
    "gather_elements",
    "scatter_elements",
    "onehot",
    "gather_nd",
    "scatter",
    "scatter_nd",
    "pad",
    "depth_to_space",
    "space_to_depth",
    "range",
    "constant_of_shape",
    "concat",
    "tile",
    "transpose",
    "flatten",
    "reshape",
    "squeeze",
    "unsqueeze",
    "split",
    "expand",
    "shape",
    "resize",
    "upsample",
    "topk",
    "nonzero",
    "argmax",
    "argmin",
    "rnn",
    "lstm",
    "gru",
    "attention_dummy",
    "multi_head_attention_simple",
    "multi_head_attention",
    "asin",
    "acos",
    "atan",
    "asinh",
    "acosh",
    "atanh",
]


def _exported_std_ml_ops() -> list[str]:
    text = _ML_MOD.read_text(encoding="utf-8")
    ops: list[str] = []
    for match in re.finditer(r"pub use [^{]+\{([^}]*)\};", text, re.S):
        names = [name.strip() for name in match.group(1).replace("\n", " ").split(",")]
        ops.extend(name for name in names if name)
    return list(dict.fromkeys(ops))


def _extract_single_std_ml_call(source: str) -> str | None:
    calls = set(re.findall(r"std::ml::([a-zA-Z_][a-zA-Z0-9_]*)", source))
    if len(calls) != 1:
        return None
    return next(iter(calls))


def _shared_print_at_sources() -> dict[str, str]:
    sources: dict[str, str] = {}
    for _, source, _ in GOLDEN_PRINT_CASES:
        op_name = _extract_single_std_ml_call(source)
        if op_name is not None:
            sources.setdefault(op_name, source.strip())
    for _, source, _ in ML_ACTIVATION_PRINT_AT_GOLDEN_CASES:
        op_name = _extract_single_std_ml_call(source)
        if op_name is not None:
            sources.setdefault(op_name, source.strip())
    return sources


_SHARED_PRINT_AT_SOURCES = _shared_print_at_sources()
_SHARED_PRINT_AT_OPS = frozenset(_SHARED_PRINT_AT_SOURCES)

_UNARY_GROUP_SIZE = 16
_BINARY_GROUP_SIZE = 12
_REDUCTION_GROUP_SIZE = 8


_UNARY_SUCCESS = frozenset(
    {
        "gelu_tanh",
        "square",
        "neg",
        "abs",
        "sign",
        "reciprocal",
        "sqrt",
        "rsqrt",
        "exp",
        "log",
        "log1p",
        "expm1",
        "floor",
        "ceil",
        "round",
        "sin",
        "cos",
        "tan",
        "tanh",
        "sinh",
        "cosh",
        "erf",
        "is_nan",
        "is_inf",
        "identity",
        "asin",
        "acos",
        "atan",
        "asinh",
        "acosh",
        "atanh",
    }
)

_BINARY_SUCCESS = frozenset(
    {
        "add",
        "subtract",
        "multiply",
        "divide",
        "power",
        "min",
        "max",
        "mod",
        "fmod",
        "atan2",
        "equal",
        "greater",
        "greater_or_equal",
        "less",
        "less_or_equal",
        "not_equal",
    }
)

_REDUCTION_SUCCESS = frozenset(
    {
        "reduce_max",
        "reduce_min",
        "reduce_prod",
        "vec_norm_2",
        "matrix_norm_frobenius",
        "trace",
        "frobenius_norm",
    }
)

_EXPECTED_FAILURE = frozenset()

_SPECIAL_CASES: dict[str, tuple[str, bool]] = {
    "clip": (
        "use std::ml; let x = [0.25, 2.75]; let y = std::ml::clip(x, 0.0, 1.0); print(@y);",
        True,
    ),
    "conv": (
        "use std::ml; let x = [[[1.0, 2.0, 3.0]]]; let w = [[[1.0, 0.5]]]; let b = [0.0]; let y = std::ml::conv(x, w, b, [1], [0, 0], [1], 1); print(@y);",
        True,
    ),
    "conv_transpose": (
        "use std::ml; let x = [[[1.0, 2.0]]]; let w = [[[1.0, 0.5]]]; let b = [0.0]; let y = std::ml::conv_transpose(x, w, b, [1], [0], [0]); print(@y);",
        True,
    ),
    "depthwise_conv": (
        "use std::ml; let x = [[[[1.0, 2.0], [3.0, 4.0]]]]; let w = [[[[1.0]]]]; let b = [0.0]; let y = std::ml::depthwise_conv(x, w, b, [1, 1], [0, 0], [1, 1]); print(@y);",
        True,
    ),
    "qconv": (
        "use std::ml; let x_q = [[[1.0, 2.0, 3.0]]]; let w_q = [[[1.0, 0.5]]]; let b = [0.0]; let y = std::ml::qconv(x_q, 1.0, w_q, 1.0, b, [1], [0, 0], [1], 1); print(@y);",
        True,
    ),
    "average_pool": (
        "use std::ml; let x = [[[1.0, 2.0, 3.0, 4.0, 5.0]]]; let y = std::ml::average_pool(x, [2], [2], [0]); let dy_dx = @y / @x;",
        True,
    ),
    "global_average_pool": (
        "use std::ml; let x = [[[1.0, 2.0, 3.0, 4.0]]]; let y = std::ml::global_average_pool(x); let dy_dx = @y / @x;",
        True,
    ),
    "global_max_pool": (
        "use std::ml; let x = [[[[1.0, 2.0], [3.0, 4.0]]]]; let y = std::ml::global_max_pool(x); print(@y);",
        True,
    ),
    "lp_pool": (
        "use std::ml; let x = [[[[1.0, 2.0], [3.0, 4.0]]]]; let y = std::ml::lp_pool(x, [2, 2], [1, 1], 2.0); print(@y);",
        True,
    ),
    "max_roi_pool": (
        "use std::ml; let x = [[[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]]]; let rois = [[0, 0, 0, 2, 2]]; let y = std::ml::max_roi_pool(x, rois, [2, 2], 1.0); let dy_dx = @y / @x;",
        True,
    ),
    "batch_normalization": (
        "use std::ml; let X = [[[[1.0, 2.0], [3.0, 4.0]]]]; let scale = [1.0]; let B = [0.0]; let mean = [0.0]; let var = [1.0]; let y = std::ml::batch_normalization(X, scale, B, mean, var, 1e-5); print(@y);",
        True,
    ),
    "instance_normalization": (
        "use std::ml; let X = [[[[1.0, 2.0], [3.0, 4.0]]]]; let scale = [1.0]; let B = [0.0]; let y = std::ml::instance_normalization(X, scale, B, 1e-5); print(@y);",
        True,
    ),
    "layer_normalization": (
        "use std::ml; let X = [[1.0, 2.0], [3.0, 4.0]]; let scale = [1.0, 1.0]; let B = [0.0, 0.0]; let y = std::ml::layer_normalization(X, scale, B, 1e-5, -1); print(@y);",
        True,
    ),
    "lrn": (
        "use std::ml; let X = [[[[1.0, 2.0], [3.0, 4.0]]]]; let y = std::ml::lrn(X, 1, 0.0001, 0.75, 1.0); print(@y);",
        True,
    ),
    "lp_normalization": (
        "use std::ml; let X = [[1.0, 2.0], [3.0, 4.0]]; let y = std::ml::lp_normalization(X, 1, 2.0); print(@y);",
        True,
    ),
    "mean_variance_normalization": (
        "use std::ml; let X = [[1.0, 2.0], [3.0, 4.0]]; let y = std::ml::mean_variance_normalization(X, [0, 1]); print(@y);",
        True,
    ),
    "gemm": (
        "use std::ml; let A = [[1.0, 2.0], [3.0, 4.0]]; let B = [[5.0, 6.0], [7.0, 8.0]]; let C = [[0.5, 0.5], [0.5, 0.5]]; let y = std::ml::gemm(A, B, C, 1.0, 1.0, 0, 0); print(@y);",
        True,
    ),
    "conv2d": (
        "use std::ml; let input = [[1.0, 2.0], [3.0, 4.0]]; let kernel = [[1.0]]; let bias = [0.5]; let y = std::ml::conv2d(input, kernel, bias, 1, 1, 0, 0, 1, 1); print(@y);",
        True,
    ),
    "cholesky": (
        "use std::ml; let A = [[4.0, 2.0], [2.0, 3.0]]; let y = std::ml::cholesky(A); print(@y);",
        True,
    ),
    "solve_triangular_lower": (
        "use std::ml; let L = [[2.0, 0.0], [1.0, 3.0]]; let b = [4.0, 5.0]; let y = std::ml::solve_triangular_lower(L, b); print(@y);",
        True,
    ),
    "solve_triangular_lower_unit": (
        "use std::ml; let L = [[1.0, 0.0], [1.0, 1.0]]; let b = [4.0, 5.0]; let y = std::ml::solve_triangular_lower_unit(L, b); print(@y);",
        True,
    ),
    "solve_triangular_upper": (
        "use std::ml; let U = [[2.0, 1.0], [0.0, 3.0]]; let x = [4.0, 5.0]; let y = std::ml::solve_triangular_upper(U, x); print(@y);",
        True,
    ),
    "solve_cholesky": (
        "use std::ml; let A = [[4.0, 2.0], [2.0, 3.0]]; let b = [4.0, 5.0]; let y = std::ml::solve_cholesky(A, b); print(@y);",
        True,
    ),
    "where": (
        "use std::ml; let cond = true; let x = [1.0, 2.0]; let z = [3.0, 4.0]; let y = std::ml::where(cond, x, z); print(@y);",
        True,
    ),
    "constant": (
        "use std::ml; let value = [1.0, 2.0]; let y = std::ml::constant(value); print(@y);",
        True,
    ),
    "dropout": (
        "use std::ml; let x = [0.25, 0.75]; let y = std::ml::dropout(x, 0.25, 1); print(@y);",
        True,
    ),
    "l2_normalize": (
        "use std::ml; let X = [[1.0, 2.0], [3.0, 4.0]]; let y = std::ml::l2_normalize(X, 1e-5); print(@y);",
        True,
    ),
    "numel": (
        "use std::ml; let x = [[1.0, 2.0], [3.0, 4.0]]; let y = std::ml::numel(x); print(@y);",
        True,
    ),
    "size": (
        "use std::ml; let x = [[1.0, 2.0], [3.0, 4.0]]; let y = std::ml::size(x); print(@y);",
        True,
    ),
    "cast": (
        "use std::ml; let x = [0.25, 0.75]; let y = std::ml::cast(x, 0); print(@y);",
        True,
    ),
    "slice": (
        "use std::ml; let x = [[1.0, 2.0], [3.0, 4.0]]; let y = std::ml::slice(x, [0], [1], [0], [1]); print(@y);",
        True,
    ),
    "cumsum": (
        "use std::ml; let x = [1.0, 2.0, 3.0]; let y = std::ml::cumsum(x); print(@y);",
        True,
    ),
    "quantize_linear": (
        "use std::ml; let x = [1.0, 2.0]; let y = std::ml::quantize_linear(x, 1.0, 0.0); print(@y);",
        True,
    ),
    "dequantize_linear": (
        "use std::ml; let x = [1.0, 2.0]; let y = std::ml::dequantize_linear(x, 1.0, 0.0); print(@y);",
        True,
    ),
    "qmatmul": (
        "use std::ml; let A_q = [[1.0, 2.0], [3.0, 4.0]]; let B_q = [[5.0, 6.0], [7.0, 8.0]]; let y = std::ml::qmatmul(A_q, 1.0, B_q, 1.0); print(@y);",
        True,
    ),
    "qlinear": (
        "use std::ml; let x_q = [1.0, 2.0]; let W_q = [[1.0, 2.0], [3.0, 4.0]]; let bias = [0.5, -0.5]; let y = std::ml::qlinear(x_q, 1.0, W_q, 1.0, bias); print(@y);",
        True,
    ),
    "image_scaler": (
        "use std::ml; let x = [[1.0, 2.0], [3.0, 4.0]]; let y = std::ml::image_scaler(x, 0.5, 1.0); print(@y);",
        True,
    ),
    "eye": (
        "use std::ml; let n = 2; let y = std::ml::eye(n); print(@y);",
        True,
    ),
    "diag_extract": (
        "use std::ml; let x = [[1.0, 2.0], [3.0, 4.0]]; let y = std::ml::diag_extract(x); print(@y);",
        True,
    ),
    "diag_construct": (
        "use std::ml; let x = [1.0, 2.0]; let y = std::ml::diag_construct(x); print(@y);",
        True,
    ),
    "outer": (
        "use std::ml; let a = [1.0, 2.0]; let b = [3.0, 4.0]; let y = std::ml::outer(a, b); print(@y);",
        True,
    ),
    "kron": (
        "use std::ml; let a = [[1.0, 2.0], [3.0, 4.0]]; let b = [[0.0, 1.0], [1.0, 0.0]]; let y = std::ml::kron(a, b); print(@y);",
        True,
    ),
    "tril": (
        "use std::ml; let x = [[1.0, 2.0], [3.0, 4.0]]; let y = std::ml::tril(x, 0); print(@y);",
        True,
    ),
    "triu": (
        "use std::ml; let x = [[1.0, 2.0], [3.0, 4.0]]; let y = std::ml::triu(x, 0); print(@y);",
        True,
    ),
    "roll": (
        "use std::ml; let x = [1.0, 2.0, 3.0]; let y = std::ml::roll(x, 1); print(@y);",
        True,
    ),
    "repeat_interleave": (
        "use std::ml; let x = [1.0, 2.0]; let y = std::ml::repeat_interleave(x, 2); print(@y);",
        True,
    ),
    "flip": (
        "use std::ml; let x = [1.0, 2.0, 3.0]; let y = std::ml::flip(x); print(@y);",
        True,
    ),
    "cross_entropy_loss": (
        "use std::ml; let predictions = [0.2, 0.8]; let targets = [0.0, 1.0]; let y = std::ml::cross_entropy_loss(predictions, targets); print(@y);",
        True,
    ),
    "softmax_cross_entropy_loss": (
        "use std::ml; let predictions = [0.2, 0.8]; let targets = [0.0, 1.0]; let y = std::ml::softmax_cross_entropy_loss(predictions, targets); print(@y);",
        True,
    ),
    "gather": (
        "use std::ml; let data = [1.0, 2.0, 3.0]; let indices = [2, 0]; let y = std::ml::gather(data, indices, 0); print(@y);",
        True,
    ),
    "gather_elements": (
        "use std::ml; let data = [[1.0, 2.0], [3.0, 4.0]]; let indices = [[1, 0], [0, 1]]; let y = std::ml::gather_elements(data, indices, 1); print(@y);",
        True,
    ),
    "scatter_elements": (
        "use std::ml; let data = [1.0, 2.0, 3.0]; let indices = [1]; let updates = [9.0]; let y = std::ml::scatter_elements(data, indices, updates, 0); print(@y);",
        True,
    ),
    "onehot": (
        "use std::ml; let indices = [0, 2]; let values = [0.0, 1.0]; let y = std::ml::onehot(indices, 3, values); print(@y);",
        True,
    ),
    "gather_nd": (
        "use std::ml; let data = [[1.0, 2.0], [3.0, 4.0]]; let indices = [[0, 1], [1, 0]]; let y = std::ml::gather_nd(data, indices); print(@y);",
        True,
    ),
    "scatter": (
        "use std::ml; let data = [1.0, 2.0, 3.0]; let indices = [1]; let updates = [9.0]; let y = std::ml::scatter(data, indices, updates, 0); print(@y);",
        True,
    ),
    "scatter_nd": (
        "use std::ml; let data = [1.0, 2.0, 3.0]; let indices = [[1]]; let updates = [9.0]; let y = std::ml::scatter_nd(data, indices, updates); print(@y);",
        True,
    ),
    "pad": (
        "use std::ml; let data = [1.0, 2.0]; let y = std::ml::pad(data, [1], 0.0); print(@y);",
        True,
    ),
    "depth_to_space": (
        "use std::ml; let input = [[[[1.0]], [[2.0]], [[3.0]], [[4.0]]]]; let y = std::ml::depth_to_space(input, 2); print(@y);",
        True,
    ),
    "space_to_depth": (
        "use std::ml; let input = [[[[1.0, 2.0], [3.0, 4.0]]]]; let y = std::ml::space_to_depth(input, 2); print(@y);",
        True,
    ),
    "range": (
        "use std::ml; let start = 0.5; let y = std::ml::range(start, 2.5, 1.0); print(@y);",
        True,
    ),
    "constant_of_shape": (
        "use std::ml; let dims = [2]; let value = 1.5; let y = std::ml::constant_of_shape(dims, value); print(@y);",
        True,
    ),
    "concat": (
        "use std::ml; let a = [[1.0], [2.0]]; let b = [[3.0], [4.0]]; let y = std::ml::concat(a, b); print(@y);",
        True,
    ),
    "tile": (
        "use std::ml; let x = [[1.0, 2.0]]; let y = std::ml::tile(x, 2); print(@y);",
        True,
    ),
    "transpose": (
        "use std::ml; let x = [[1.0, 2.0], [3.0, 4.0]]; let y = std::ml::transpose(x); print(@y);",
        True,
    ),
    "flatten": (
        "use std::ml; let x = [[1.0, 2.0], [3.0, 4.0]]; let y = std::ml::flatten(x); print(@y);",
        True,
    ),
    "reshape": (
        "use std::ml; let x = [[1.0, 2.0], [3.0, 4.0]]; let y = std::ml::reshape(x, [4]); print(@y);",
        True,
    ),
    "squeeze": (
        "use std::ml; let x = [[[1.0, 2.0]]]; let y = std::ml::squeeze(x, [0]); print(@y);",
        True,
    ),
    "unsqueeze": (
        "use std::ml; let x = [1.0, 2.0]; let y = std::ml::unsqueeze(x, [0]); print(@y);",
        True,
    ),
    "split": (
        "use std::ml; let x = [[1.0, 2.0], [3.0, 4.0]]; let parts = std::ml::split(x, [1, 1], 1); let dpart_dx = @parts.0 / @x;",
        True,
    ),
    "expand": (
        "use std::ml; let x = [1.0, 2.0]; let y = std::ml::expand(x, [2, 2]); print(@y);",
        True,
    ),
    "resize": (
        'use std::ml; let X = [[[[1.0, 2.0], [3.0, 4.0]]]]; let y = std::ml::resize(X, [2.0, 2.0], "nearest"); let dy_dX = @y / @X;',
        True,
    ),
    "upsample": (
        'use std::ml; let X = [[[[1.0, 2.0], [3.0, 4.0]]]]; let y = std::ml::upsample(X, [2.0, 2.0], "nearest"); let dy_dX = @y / @X;',
        True,
    ),
    "topk": (
        "use std::ml; let X = [[1.0, 3.0, 2.0], [4.0, 6.0, 5.0]]; let pair = std::ml::topk(X, 2, 1); let dvals_dX = @pair.0 / @X;",
        True,
    ),
    "attention_dummy": (
        "use std::ml; let q = [[[1.0, 0.0], [0.0, 1.0]]]; let k = q; let v = q; let y = std::ml::attention_dummy(q, k, v, 0.5); print(@y);",
        True,
    ),
    "multi_head_attention_simple": (
        "use std::ml; let q = [[[1.0, 0.0], [0.0, 1.0]]]; let k = q; let v = q; let y = std::ml::multi_head_attention_simple(q, k, v, 1, 0.5); print(@y);",
        True,
    ),
    "multi_head_attention": (
        "use std::ml; let q = [[[1.0, 0.0], [0.0, 1.0]]]; let k = q; let v = q; let mask = [[1.0, 1.0], [1.0, 1.0]]; let y = std::ml::multi_head_attention(q, k, v, 1, 0.5, mask); print(@y);",
        True,
    ),
    "rnn": (
        'use std::ml; let X = [[[1.0]]]; let W = [[[1.0]]]; let R = [[[1.0]]]; let B = [[0.0, 0.0]]; let initial_h = [[[0.0]]]; let out = std::ml::rnn(X, W, R, B, initial_h, 1, 0, "tanh"); let dY_dX = @out.0 / @X;',
        True,
    ),
    "lstm": (
        "use std::ml; let X = [[[1.0]]]; let W = [[[1.0]], [[1.0]], [[1.0]], [[1.0]]]; let R = [[[1.0]], [[1.0]], [[1.0]], [[1.0]]]; let B = [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]; let initial_h = [[[0.0]]]; let initial_c = [[[0.0]]]; let out = std::ml::lstm(X, W, R, B, initial_h, initial_c, 1, 0, 0.0); let dY_dX = @out.0 / @X;",
        True,
    ),
    "gru": (
        "use std::ml; let X = [[[1.0]]]; let W = [[[1.0]], [[1.0]], [[1.0]]]; let R = [[[1.0]], [[1.0]], [[1.0]]]; let B = [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]; let initial_h = [[[0.0]]]; let out = std::ml::gru(X, W, R, B, initial_h, 1, 0, 0); let dY_dX = @out.0 / @X;",
        True,
    ),
    "einsum": (
        'use std::ml; let A = [[1.0, 2.0], [3.0, 4.0]]; let B = [[5.0, 6.0], [7.0, 8.0]]; let y = std::ml::einsum("ij,jk->ik", (A, B)); let dy_dA = @y / @A;',
        True,
    ),
    "not": (
        "use std::ml; let x = 1.0; let flag = std::ml::not(false); let y = if flag { x } else { x * 2.0 }; let dy_dx = @y / @x;",
        True,
    ),
    "logical_and": (
        "use std::ml; let x = 1.0; let flag = std::ml::logical_and(true, false); let y = if flag { x } else { x * 2.0 }; let dy_dx = @y / @x;",
        True,
    ),
    "logical_or": (
        "use std::ml; let x = 1.0; let flag = std::ml::logical_or(true, false); let y = if flag { x } else { x * 2.0 }; let dy_dx = @y / @x;",
        True,
    ),
    "logical_xor": (
        "use std::ml; let x = 1.0; let flag = std::ml::logical_xor(true, false); let y = if flag { x } else { x * 2.0 }; let dy_dx = @y / @x;",
        True,
    ),
    "logical_not": (
        "use std::ml; let x = 1.0; let flag = std::ml::logical_not(false); let y = if flag { x } else { x * 2.0 }; let dy_dx = @y / @x;",
        True,
    ),
    "shape": (
        "use std::ml; let x = [1.0, 2.0, 3.0]; let dims = std::ml::shape(x); let y = x[dims[0] - 1]; let dy_dx = @y / @x;",
        True,
    ),
    "nonzero": (
        "use std::ml; let x = [1.0, 2.0, 3.0]; let idxs = std::ml::nonzero([0.0, 1.0, 0.0]); let y = x[idxs[0]]; let dy_dx = @y / @x;",
        True,
    ),
    "argmax": (
        "use std::ml; let x = [1.0, 3.0, 2.0]; let idx = std::ml::argmax(x); let y = x[idx]; let dy_dx = @y / @x;",
        True,
    ),
    "argmin": (
        "use std::ml; let x = [1.0, 3.0, 2.0]; let idx = std::ml::argmin(x); let y = x[idx]; let dy_dx = @y / @x;",
        True,
    ),
}


def _cataloged_ops() -> set[str]:
    return (
        set(_SHARED_PRINT_AT_OPS)
        | set(_UNARY_SUCCESS)
        | set(_BINARY_SUCCESS)
        | set(_REDUCTION_SUCCESS)
        | set(_EXPECTED_FAILURE)
        | set(_SPECIAL_CASES)
    )


def _ops_in_std_ml_order(candidates: set[str]) -> list[str]:
    return [op for op in _STD_ML_EXPORTS if op in candidates]


def _chunked(items: list[str], size: int) -> list[tuple[str, ...]]:
    return [tuple(items[i : i + size]) for i in range(0, len(items), size)]


def _build_grouped_generic_source(registry: str, ops: tuple[str, ...]) -> str:
    lines = ["use std::ml;"]
    if registry == "unary":
        lines.append("let x = [0.25, 0.75];")
        input_name = "x"
        call_template = "std::ml::{op}(x)"
    elif registry == "binary":
        lines.append("let a = [0.25, 0.75];")
        lines.append("let b = [1.5, 2.5];")
        input_name = "a"
        call_template = "std::ml::{op}(a, b)"
    elif registry == "reduction":
        lines.append("let x = [[0.25, 0.75], [1.5, 2.5]];")
        input_name = "x"
        call_template = "std::ml::{op}(x)"
    else:
        raise AssertionError("unknown registry %s" % registry)

    for op_name in ops:
        result_name = "y_%s" % op_name
        diff_name = "dy_%s" % op_name
        lines.append("let %s = %s;" % (result_name, call_template.format(op=op_name)))
        lines.append("let %s = @%s;" % (diff_name, result_name))
    return "\n".join(lines)


def _case_for(op_name: str) -> tuple[str, bool]:
    if op_name in _SHARED_PRINT_AT_SOURCES:
        return _SHARED_PRINT_AT_SOURCES[op_name], True
    if op_name in _SPECIAL_CASES:
        return _SPECIAL_CASES[op_name]
    if op_name in _UNARY_SUCCESS:
        return (
            f"use std::ml; let x = [0.25, 0.75]; let y = std::ml::{op_name}(x); print(@y);",
            True,
        )
    if op_name in _BINARY_SUCCESS:
        return (
            f"use std::ml; let a = [0.25, 0.75]; let b = [1.5, 2.5]; let y = std::ml::{op_name}(a, b); print(@y);",
            True,
        )
    if op_name in _REDUCTION_SUCCESS:
        return (
            f"use std::ml; let x = [[0.25, 0.75], [1.5, 2.5]]; let y = std::ml::{op_name}(x); print(@y);",
            True,
        )
    raise AssertionError("missing autodiff case for std::ml::%s" % op_name)


_GROUPED_GENERIC_CASES = (
    [("unary", ops) for ops in _chunked(_ops_in_std_ml_order(set(_UNARY_SUCCESS)), _UNARY_GROUP_SIZE)]
    + [("binary", ops) for ops in _chunked(_ops_in_std_ml_order(set(_BINARY_SUCCESS)), _BINARY_GROUP_SIZE)]
    + [("reduction", ops) for ops in _chunked(_ops_in_std_ml_order(set(_REDUCTION_SUCCESS)), _REDUCTION_GROUP_SIZE)]
)

_ISOLATED_COMPILE_OPS = [
    op_name
    for op_name in _STD_ML_EXPORTS
    if op_name not in _UNARY_SUCCESS
    and op_name not in _BINARY_SUCCESS
    and op_name not in _REDUCTION_SUCCESS
]


def test_std_ml_autodiff_case_registries_are_disjoint() -> None:
    registries = {
        "shared_print_at": set(_SHARED_PRINT_AT_OPS),
        "unary_generic": set(_UNARY_SUCCESS),
        "binary_generic": set(_BINARY_SUCCESS),
        "reduction_generic": set(_REDUCTION_SUCCESS),
        "expected_failure": set(_EXPECTED_FAILURE),
        "special_cases": set(_SPECIAL_CASES),
    }
    overlaps: list[str] = []
    registry_names = list(registries)
    for i, left_name in enumerate(registry_names):
        left = registries[left_name]
        for right_name in registry_names[i + 1 :]:
            inter = sorted(left & registries[right_name])
            if inter:
                overlaps.append(f"{left_name} vs {right_name}: {inter}")
    assert not overlaps, "std::ml autodiff case registries overlap:\n%s" % "\n".join(overlaps)


def test_std_ml_autodiff_catalog_covers_every_export() -> None:
    exported = set(_exported_std_ml_ops())
    cataloged = _cataloged_ops()
    missing = sorted(exported - cataloged)
    extra = sorted(cataloged - exported)
    assert not missing, "missing std::ml autodiff catalog entries: %s" % missing
    assert not extra, "catalog has non-exported std::ml entries: %s" % extra


@pytest.fixture(scope="module")
def ml_autodiff_compiler(module_compiler) -> CompilerDriver:
    return module_compiler


@pytest.mark.parametrize(
    "registry_name,ops",
    _GROUPED_GENERIC_CASES,
    ids=lambda case: case if isinstance(case, str) else None,
)
def test_std_ml_generic_groups_have_autodiff_compile_coverage(
    registry_name: str, ops: tuple[str, ...], ml_autodiff_compiler: CompilerDriver
) -> None:
    source = _build_grouped_generic_source(registry_name, ops)
    result = ml_autodiff_compiler.compile(
        source,
        source_file="<autodiff-std-ml:%s:%s>" % (registry_name, ",".join(ops)),
        root_path=_REPO_ROOT,
    )
    assert result.success, (
        "std::ml %s batch failed for %s: %s"
        % (registry_name, list(ops), result.get_errors())
    )


@pytest.mark.parametrize("op_name", _ISOLATED_COMPILE_OPS, ids=_ISOLATED_COMPILE_OPS)
def test_std_ml_isolated_exports_have_autodiff_compile_coverage(
    op_name: str, ml_autodiff_compiler: CompilerDriver
) -> None:
    source, expect_success = _case_for(op_name)
    result = ml_autodiff_compiler.compile(
        source.strip(),
        source_file="<autodiff-std-ml:%s>" % op_name,
        root_path=_REPO_ROOT,
    )
    errors = result.get_errors()
    if expect_success:
        assert result.success, "std::ml::%s autodiff compile failed: %s" % (op_name, errors)
    else:
        assert not result.success, "std::ml::%s unexpectedly compiled under autodiff" % op_name
