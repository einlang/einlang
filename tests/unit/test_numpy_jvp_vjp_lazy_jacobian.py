from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np


_MODULE_PATH = (
    Path(__file__).resolve().parent.parent.parent
    / "examples"
    / "numpy_jvp_vjp_lazy_jacobian.py"
)
_CONV_REF_PATH = (
    Path(__file__).resolve().parent.parent.parent
    / "examples"
    / "julia_style_conv_vjp_numpy.py"
)


def _load_path_module(module_name: str, module_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _load_module():
    return _load_path_module("numpy_jvp_vjp_lazy_jacobian", _MODULE_PATH)


def _load_conv_ref_module():
    return _load_path_module("julia_style_conv_vjp_numpy", _CONV_REF_PATH)


def _finite_diff_tensor(
    fn,
    primal_inputs,
    tangent_inputs,
    *,
    eps: float = 1e-6,
):
    primals = [np.array(x, dtype=np.float64, copy=True) for x in primal_inputs]
    tangents = [np.array(t, dtype=np.float64, copy=False) for t in tangent_inputs]
    plus = [p + eps * t for p, t in zip(primals, tangents)]
    minus = [p - eps * t for p, t in zip(primals, tangents)]
    return (np.asarray(fn(*plus), dtype=np.float64) - np.asarray(fn(*minus), dtype=np.float64)) / (2.0 * eps)


def test_scalar_jvp_matches_vjp() -> None:
    mod = _load_module()
    x = mod.Tensor.leaf(np.array(3.0), name="x")
    y = x * x + 2.0 * x + 1.0

    dy_dx_jvp = mod.jvp(y, {x: np.array(1.0)})
    dy_dx_vjp = mod.vjp(y, np.array(1.0), x)

    assert np.allclose(dy_dx_jvp, 8.0)
    assert np.allclose(dy_dx_vjp, 8.0)


def test_lazy_softmax_jacobian_matches_jvp_and_vjp_materialization() -> None:
    mod = _load_module()
    x = mod.Tensor.leaf(np.array([0.2, -0.4, 0.7]), name="x")
    y = mod.softmax(x)
    J = mod.jacobian(y, x)

    y_val = np.asarray(y.value, dtype=np.float64)
    ref = np.diag(y_val) - np.outer(y_val, y_val)

    assert np.allclose(J.materialize_via_jvp(), ref)
    assert np.allclose(J.materialize_via_vjp(), ref)
    assert np.allclose(np.asarray(J), ref)
    assert np.allclose(J.column((1,)), ref[:, 1])
    assert np.allclose(J.row((2,)), ref[2, :])
    assert np.allclose(J[2, 1], ref[2, 1])


def test_scalar_slice_alias_projects_back_to_storage() -> None:
    mod = _load_module()
    W = mod.Tensor.leaf(np.array([[1.0, 2.0], [3.0, 4.0]]), name="W")
    e = W[1, 0] * W[1, 0]
    d_e_d_W = mod.jacobian(e, W)

    ref = np.zeros_like(W.value)
    ref[1, 0] = 6.0

    assert np.allclose(np.asarray(d_e_d_W), ref)
    assert np.allclose(d_e_d_W[1, 0], 6.0)


def test_conv_vjp_matches_reference_numpy_oracle() -> None:
    mod = _load_module()
    ref = _load_conv_ref_module()

    x_val = np.array([[[[1.0, -2.0, 0.5], [0.3, 1.2, -0.7], [2.0, -1.0, 0.8]]]], dtype=np.float64)
    w_val = np.array([[[[0.4, -0.1], [0.2, 0.3]]]], dtype=np.float64)
    b_val = np.array([0.15], dtype=np.float64)
    dy = np.array([[[[0.7, -1.1], [0.2, 0.9]]]], dtype=np.float64)

    stride = (1, 1)
    pads = (0, 0, 0, 0)
    pad_begin = (0, 0)
    pad_end = (0, 0)
    dilation = (1, 1)
    group = 1

    x = mod.Tensor.leaf(x_val, name="x")
    w = mod.Tensor.leaf(w_val, name="w")
    b = mod.Tensor.leaf(b_val, name="b")
    y = mod.conv(x, w, b, stride=stride, pad_begin=pad_begin, pad_end=pad_end, dilation=dilation, group=group)

    dx = mod.vjp(y, dy, x)
    dw = mod.vjp(y, dy, w)
    db = mod.vjp(y, dy, b)
    dx_ref, dw_ref, db_ref = ref.conv_vjp_ml(x_val, w_val, dy, stride, pads, dilation, group)

    assert np.allclose(dx, dx_ref)
    assert np.allclose(dw, dw_ref)
    assert np.allclose(db, db_ref)


def test_conv_avg_pool_vjp_matches_chained_reference_pullbacks() -> None:
    mod = _load_module()
    ref = _load_conv_ref_module()

    x_val = np.array([[[[1.0, -2.0, 0.5], [0.3, 1.2, -0.7], [2.0, -1.0, 0.8]]]], dtype=np.float64)
    w_val = np.array([[[[0.4, -0.1], [0.2, 0.3]]]], dtype=np.float64)
    b_val = np.array([0.05], dtype=np.float64)

    conv_stride = (1, 1)
    conv_pads = (0, 0, 0, 0)
    pad_begin = (0, 0)
    pad_end = (0, 0)
    conv_dilation = (1, 1)
    group = 1

    pool_kernel = (2, 2)
    pool_stride = (1, 1)
    pool_pads = (0, 0)

    x = mod.Tensor.leaf(x_val, name="x")
    w = mod.Tensor.leaf(w_val, name="w")
    b = mod.Tensor.leaf(b_val, name="b")
    conv_out = mod.conv(x, w, b, stride=conv_stride, pad_begin=pad_begin, pad_end=pad_end, dilation=conv_dilation, group=group)
    pooled = mod.avg_pool(conv_out, kernel_shape=pool_kernel, strides=pool_stride, pads=pool_pads)

    dy = np.array([[[[1.3, -0.4], [0.6, 0.8]]]], dtype=np.float64)
    dx = mod.vjp(pooled, dy, x)
    dw = mod.vjp(pooled, dy, w)
    db = mod.vjp(pooled, dy, b)

    conv_ref = ref.conv_ml(x_val, w_val, b_val, conv_stride, conv_pads, conv_dilation, group)
    d_conv_ref = ref.average_pool_vjp(conv_ref, dy, pool_kernel, pool_stride, pool_pads)
    dx_ref, dw_ref, db_ref = ref.conv_vjp_ml(x_val, w_val, d_conv_ref, conv_stride, conv_pads, conv_dilation, group)

    assert np.allclose(dx, dx_ref)
    assert np.allclose(dw, dw_ref)
    assert np.allclose(db, db_ref)


def test_conv_max_pool_jvp_matches_finite_difference_directional_derivative() -> None:
    mod = _load_module()

    x_val = np.array([[[[1.0, -2.0, 0.5], [0.3, 1.2, -0.7], [2.0, -1.0, 0.8]]]], dtype=np.float64)
    w_val = np.array([[[[0.4, -0.1], [0.2, 0.3]]]], dtype=np.float64)
    b_val = np.array([0.12], dtype=np.float64)

    dx_val = np.array([[[[0.2, -0.3, 0.1], [0.05, 0.07, -0.11], [0.13, 0.17, -0.19]]]], dtype=np.float64)
    dw_val = np.array([[[[0.03, -0.04], [0.02, 0.05]]]], dtype=np.float64)
    db_val = np.array([0.09], dtype=np.float64)

    conv_stride = (1, 1)
    pad_begin = (0, 0)
    pad_end = (0, 0)
    conv_dilation = (1, 1)
    group = 1

    pool_kernel = (2, 2)
    pool_stride = (1, 1)
    pool_pads = (0, 0)

    x = mod.Tensor.leaf(x_val, name="x")
    w = mod.Tensor.leaf(w_val, name="w")
    b = mod.Tensor.leaf(b_val, name="b")
    conv_out = mod.conv(x, w, b, stride=conv_stride, pad_begin=pad_begin, pad_end=pad_end, dilation=conv_dilation, group=group)
    pooled = mod.max_pool(conv_out, kernel_shape=pool_kernel, strides=pool_stride, pads=pool_pads)

    jvp_val = mod.jvp(pooled, {x: dx_val, w: dw_val, b: db_val})

    def forward_fn(x_arg, w_arg, b_arg):
        conv_y = mod.conv_forward(x_arg, w_arg, b_arg, conv_stride, pad_begin, pad_end, conv_dilation, group)
        return mod.max_pool_forward(conv_y, pool_kernel, pool_stride, pool_pads)

    fd = _finite_diff_tensor(forward_fn, (x_val, w_val, b_val), (dx_val, dw_val, db_val))

    assert np.allclose(jvp_val, fd, atol=1e-6, rtol=1e-6)


def test_symbolic_conv_relu_maxpool_tangent_print_is_operator_level() -> None:
    mod = _load_module()

    x = mod.Tensor.leaf(np.array([[[[1.0, -2.0, 0.5], [0.3, 1.2, -0.7], [2.0, -1.0, 0.8]]]], dtype=np.float64), name="x")
    w = mod.Tensor.leaf(np.array([[[[0.4, -0.1], [0.2, 0.3]]]], dtype=np.float64), name="w")
    b = mod.Tensor.leaf(np.array([0.15], dtype=np.float64), name="b")

    c = mod.conv(x, w, b, stride=(1, 1), pad_begin=(0, 0), pad_end=(0, 0), dilation=(1, 1), group=1).named("c")
    r = mod.relu(c).named("r")
    p = mod.max_pool(r, kernel_shape=(2, 2), strides=(1, 1), pads=(0, 0)).named("p")

    symbolic = mod.symbolic_tangent_program(p)

    assert "let @c =" in symbolic
    assert "conv(@x" in symbolic
    assert "@w" in symbolic
    assert "@b" in symbolic
    assert "let @r =" in symbolic
    assert "if c > 0.0" in symbolic
    assert "let @p =" in symbolic
    assert "select_at_argmax" in symbolic


def test_symbolic_tangent_wrt_x_matches_jacobian_application_view() -> None:
    mod = _load_module()

    x = mod.Tensor.leaf(np.array([[[[1.0, -2.0, 0.5], [0.3, 1.2, -0.7], [2.0, -1.0, 0.8]]]], dtype=np.float64), name="x")
    w = mod.Tensor.leaf(np.array([[[[0.4, -0.1], [0.2, 0.3]]]], dtype=np.float64), name="w")
    b = mod.Tensor.leaf(np.array([0.15], dtype=np.float64), name="b")

    c = mod.conv(x, w, b, stride=(1, 1), pad_begin=(0, 0), pad_end=(0, 0), dilation=(1, 1), group=1).named("c")
    p = mod.max_pool(mod.relu(c), kernel_shape=(2, 2), strides=(1, 1), pads=(0, 0)).named("p")

    symbolic_wrt_x = mod.symbolic_tangent_program(p, wrt=x)
    relation = mod.symbolic_jacobian_application(p, x)

    assert "@x" in symbolic_wrt_x
    assert "@w" not in symbolic_wrt_x
    assert "@b" not in symbolic_wrt_x
    assert relation == "(@p / @x) · @x"
