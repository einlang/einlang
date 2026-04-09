from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np

from einlang.compiler.driver import CompilerDriver
from einlang.runtime.runtime import EinlangRuntime


_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_CONV_REF_PATH = _REPO_ROOT / "examples" / "julia_style_conv_vjp_numpy.py"


def _load_conv_ref_module():
    spec = importlib.util.spec_from_file_location("julia_style_conv_vjp_numpy_phase1", _CONV_REF_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["julia_style_conv_vjp_numpy_phase1"] = module
    spec.loader.exec_module(module)
    return module


def _compile_and_run(source: str):
    compiler = CompilerDriver()
    runtime = EinlangRuntime(backend="numpy")
    result = compiler.compile(source.strip(), source_file="<test>", root_path=_REPO_ROOT)
    assert result.success, result.get_errors() or "compile failed"
    exec_result = runtime.execute(result)
    assert exec_result.success, getattr(exec_result, "error", None) or exec_result.errors
    return result, exec_result.outputs or {}


def test_symbolic_print_for_conv_relu_max_pool(capsys) -> None:
    source = """
use std::ml::{conv, max_pool, relu};
let x = [[[[1.0, -2.0, 0.5], [0.3, 1.2, -0.7], [2.0, -1.0, 0.8]]]];
let w = [[[[0.4, -0.1], [0.2, 0.3]]]];
let b = [0.15];
let c = conv(x, w, b, [1, 1], [0, 0, 0, 0], [1, 1], 1);
let r = relu(c);
let p = max_pool(r, [2, 2], [1, 1], [0, 0]);
print(@p);
"""
    _compile_and_run(source)
    out = capsys.readouterr().out
    assert "let @c = conv(@x, w, 0.0" in out
    assert "let @r = if c > 0.0 { @c } else { 0.0 };" in out
    assert "let @p = select_at_argmax(r, @r, (2, 2), (1, 1), (0, 0));" in out


def test_scalar_conv_relu_max_pool_quotient_wrt_input_is_available_during_forward() -> None:
    ref = _load_conv_ref_module()
    source = """
use std::ml::{conv, max_pool, relu};
let x = [[[[1.0, -2.0, 0.5], [0.3, 1.2, -0.7], [2.0, -1.0, 0.8]]]];
let w = [[[[0.4, -0.1], [0.2, 0.3]]]];
let b = [0.15];
let c = conv(x, w, b, [1, 1], [0, 0, 0, 0], [1, 1], 1);
let p = max_pool(relu(c), [2, 2], [1, 1], [0, 0]);
let s = p[0, 0, 0, 0];
let g = @s / @x;
let probe = g[0, 0, 1, 1];
"""
    _, outputs = _compile_and_run(source)
    actual_g = np.asarray(outputs["g"], dtype=np.float64)
    actual_probe = float(np.asarray(outputs["probe"]).reshape(-1)[0])

    x = np.array([[[[1.0, -2.0, 0.5], [0.3, 1.2, -0.7], [2.0, -1.0, 0.8]]]], dtype=np.float64)
    w = np.array([[[[0.4, -0.1], [0.2, 0.3]]]], dtype=np.float64)
    b = np.array([0.15], dtype=np.float64)
    c = ref.conv_ml(x, w, b, (1, 1), (0, 0, 0, 0), (1, 1), 1)
    r = ref.relu_forward(c)
    p = ref.max_pool_forward(r, (2, 2), (1, 1), (0, 0))
    dy = np.zeros_like(p)
    dy[0, 0, 0, 0] = 1.0
    dr = ref.max_pool_vjp(r, dy, (2, 2), (1, 1), (0, 0))
    dc = ref.relu_vjp(c, dr)
    expected_g, _, _ = ref.conv_vjp_ml(x, w, dc, (1, 1), (0, 0, 0, 0), (1, 1), 1)

    assert np.allclose(actual_g, expected_g)
    assert np.allclose(actual_probe, expected_g[0, 0, 1, 1])
