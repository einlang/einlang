from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np


_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_PROTO_PATH = _REPO_ROOT / "examples" / "numpy_jvp_vjp_lazy_jacobian.py"
_MNIST_GRAD_REF_PATH = _REPO_ROOT / "examples" / "mnist" / "compare_train_one_step_gradients.py"
_MNIST_LOSS_REF_PATH = _REPO_ROOT / "examples" / "mnist" / "compare_train_one_step_numpy.py"


def _load_path_module(module_name: str, module_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _load_proto():
    return _load_path_module("numpy_jvp_vjp_lazy_jacobian_mnist", _PROTO_PATH)


def _load_grad_ref():
    return _load_path_module("mnist_numpy_grad_reference", _MNIST_GRAD_REF_PATH)


def _load_loss_ref():
    return _load_path_module("mnist_numpy_loss_reference", _MNIST_LOSS_REF_PATH)


def _mnist_init_arrays():
    DT = np.float32
    x = np.zeros((1, 1, 8, 8), dtype=DT)
    for h in range(8):
        for w in range(8):
            x[0, 0, h, w] = DT((h * 8 + w + 1) / 64.0)

    y = np.zeros((10,), dtype=DT)
    y[3] = DT(1.0)

    conv1_w = np.zeros((2, 1, 3, 3), dtype=DT)
    for co in range(2):
        for ci in range(1):
            for kh in range(3):
                for kw in range(3):
                    conv1_w[co, ci, kh, kw] = DT(0.02 * (1.0 + co + kh + kw))
    conv1_b = np.full((2,), DT(0.05), dtype=DT)

    conv2_w = np.zeros((2, 2, 3, 3), dtype=DT)
    for co in range(2):
        for ci in range(2):
            for kh in range(3):
                for kw in range(3):
                    conv2_w[co, ci, kh, kw] = DT(0.015 * (1.0 + co + ci + kh + kw))
    conv2_b = np.full((2,), DT(0.03), dtype=DT)

    fc_w = np.zeros((8, 10), dtype=DT)
    for i in range(8):
        for j in range(10):
            fc_w[i, j] = DT(0.01 * (1.0 + i + j))
    fc_b = np.zeros((10,), dtype=DT)

    return {
        "x": x,
        "y": y,
        "conv1_w": conv1_w,
        "conv1_b": conv1_b,
        "conv2_w": conv2_w,
        "conv2_b": conv2_b,
        "fc_w": fc_w,
        "fc_b": fc_b,
    }


def _build_mnist_graph(mod, arrays):
    x = mod.Tensor.leaf(arrays["x"], name="x")
    y = mod.Tensor.leaf(arrays["y"], name="y")
    conv1_w = mod.Tensor.leaf(arrays["conv1_w"], name="conv1_w")
    conv1_b = mod.Tensor.leaf(arrays["conv1_b"], name="conv1_b")
    conv2_w = mod.Tensor.leaf(arrays["conv2_w"], name="conv2_w")
    conv2_b = mod.Tensor.leaf(arrays["conv2_b"], name="conv2_b")
    fc_w = mod.Tensor.leaf(arrays["fc_w"], name="fc_w")
    fc_b = mod.Tensor.leaf(arrays["fc_b"], name="fc_b")

    c1 = mod.conv(x, conv1_w, conv1_b, stride=(1, 1), pad_begin=(1, 1), pad_end=(1, 1), dilation=(1, 1), group=1)
    p1 = mod.max_pool(mod.relu(c1), kernel_shape=(2, 2), strides=(2, 2), pads=(0, 0))
    c2 = mod.conv(p1, conv2_w, conv2_b, stride=(1, 1), pad_begin=(1, 1), pad_end=(1, 1), dilation=(1, 1), group=1)
    p2 = mod.max_pool(mod.relu(c2), kernel_shape=(2, 2), strides=(2, 2), pads=(0, 0))
    flat = p2.reshape(8)
    logits = (flat.reshape(8, 1) * fc_w).sum(axis=0) + fc_b
    diff = logits - y
    loss = (diff * diff).sum()

    leaves = {
        "x": x,
        "y": y,
        "conv1_w": conv1_w,
        "conv1_b": conv1_b,
        "conv2_w": conv2_w,
        "conv2_b": conv2_b,
        "fc_w": fc_w,
        "fc_b": fc_b,
    }
    return loss, leaves


def _run_proto_mnist_gradients():
    mod = _load_proto()
    arrays = _mnist_init_arrays()
    loss, leaves = _build_mnist_graph(mod, arrays)

    grads = {
        "d_loss_d_x": np.asarray(mod.vjp(loss, np.array(1.0), leaves["x"]), dtype=np.float64),
        "d_loss_d_y": np.asarray(mod.vjp(loss, np.array(1.0), leaves["y"]), dtype=np.float64),
        "d_loss_d_conv1_w": np.asarray(mod.vjp(loss, np.array(1.0), leaves["conv1_w"]), dtype=np.float64),
        "d_loss_d_conv1_b": np.asarray(mod.vjp(loss, np.array(1.0), leaves["conv1_b"]), dtype=np.float64),
        "d_loss_d_conv2_w": np.asarray(mod.vjp(loss, np.array(1.0), leaves["conv2_w"]), dtype=np.float64),
        "d_loss_d_conv2_b": np.asarray(mod.vjp(loss, np.array(1.0), leaves["conv2_b"]), dtype=np.float64),
        "d_loss_d_fc_w": np.asarray(mod.vjp(loss, np.array(1.0), leaves["fc_w"]), dtype=np.float64),
        "d_loss_d_fc_b": np.asarray(mod.vjp(loss, np.array(1.0), leaves["fc_b"]), dtype=np.float64),
    }
    return float(np.asarray(loss.value)), grads


def _run_proto_mnist_one_step():
    mod = _load_proto()
    arrays0 = _mnist_init_arrays()
    loss0, leaves0 = _build_mnist_graph(mod, arrays0)

    lr = np.float32(1e-3)
    arrays1 = {
        "x": arrays0["x"].copy(),
        "y": arrays0["y"].copy(),
        "conv1_w": (arrays0["conv1_w"] - lr * np.asarray(mod.vjp(loss0, np.array(1.0), leaves0["conv1_w"]), dtype=np.float32)).astype(np.float32),
        "conv1_b": (arrays0["conv1_b"] - lr * np.asarray(mod.vjp(loss0, np.array(1.0), leaves0["conv1_b"]), dtype=np.float32)).astype(np.float32),
        "conv2_w": (arrays0["conv2_w"] - lr * np.asarray(mod.vjp(loss0, np.array(1.0), leaves0["conv2_w"]), dtype=np.float32)).astype(np.float32),
        "conv2_b": (arrays0["conv2_b"] - lr * np.asarray(mod.vjp(loss0, np.array(1.0), leaves0["conv2_b"]), dtype=np.float32)).astype(np.float32),
        "fc_w": (arrays0["fc_w"] - lr * np.asarray(mod.vjp(loss0, np.array(1.0), leaves0["fc_w"]), dtype=np.float32)).astype(np.float32),
        "fc_b": (arrays0["fc_b"] - lr * np.asarray(mod.vjp(loss0, np.array(1.0), leaves0["fc_b"]), dtype=np.float32)).astype(np.float32),
    }
    loss1, _ = _build_mnist_graph(mod, arrays1)
    return float(np.asarray(loss0.value)), float(np.asarray(loss1.value))


def test_mnist_toy_gradients_match_existing_numpy_reference() -> None:
    ref = _load_grad_ref()
    loss0, proto = _run_proto_mnist_gradients()
    expected = ref._run_numpy_gradients()

    assert loss0 > 0.0
    for key, ref_value in expected.items():
        assert np.allclose(proto[key], ref_value, atol=1e-6, rtol=1e-6), key


def test_mnist_toy_one_step_loss_matches_numpy_reference_and_decreases() -> None:
    ref = _load_loss_ref()
    proto_loss0, proto_loss1 = _run_proto_mnist_one_step()
    ref_loss0, ref_loss1 = ref._run_numpy()

    assert proto_loss1 < proto_loss0
    assert np.allclose(proto_loss0, ref_loss0, atol=1e-6, rtol=1e-6)
    assert np.allclose(proto_loss1, ref_loss1, atol=1e-6, rtol=1e-6)
