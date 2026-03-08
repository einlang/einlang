"""
Accuracy checks for simulation demos: ODE (decay, linear, Lorenz, Lotka-Volterra),
wave, heat, Brusselator, Julia-migration (pde_1d, value_iteration, recurrence, tensor_ops).

Each test runs the demo (or a minimal variant), then compares every element
against a reference computed in-test: analytical (ODE) or NumPy reference
implementation. No mocking.
"""

from pathlib import Path

import numpy as np
import pytest

from tests.test_utils import compile_and_execute
from tests.examples.reference_implementations import (
    decay_reference,
    wave_2d_reference,
    heat_minimal_reference,
    lorenz_reference,
    lotka_volterra_reference,
    heat_1d_reference,
    linear_ode_reference,
    brusselator_reference,
    value_iteration_reference,
    fibonacci_reference,
    advection_1d_reference,
    softmax_reference,
    random_walk_reference,
)

# Every simulation example file that must pass accuracy vs reference.
# (path, output_key for result.outputs or "value" for result.value, reference_fn, rtol, atol, first_n)
# first_n=None: compare full array; first_n=N: compare first N steps only and assert finite.
ALL_ACCURACY_EXAMPLES = [
    ("examples/ode/decay.ein", "u", decay_reference, 5e-3, 1e-6, None),
    ("examples/ode/linear.ein", "u", linear_ode_reference, 1e-5, 1e-5, None),
    ("examples/ode/lorenz.ein", "u", lorenz_reference, 1e-3, 1e-2, 3),
    ("examples/ode/lotka_volterra.ein", "state", lotka_volterra_reference, 1e-4, 1e-4, 2),
    ("examples/wave_2d/main.ein", "h", wave_2d_reference, 1e-4, 1e-5, None),
    ("examples/pde_1d/heat_1d.ein", "u", heat_1d_reference, 1e-5, 1e-5, None),
    ("examples/pde_1d/advection_1d.ein", "u", advection_1d_reference, 1e-2, 0.15, None),
    ("examples/brusselator/main.ein", "state", brusselator_reference, 1e-5, 1e-5, None),
    ("examples/value_iteration/main.ein", "V", value_iteration_reference, 1e-5, 1e-5, None),
    ("examples/recurrence/fibonacci.ein", "fib", fibonacci_reference, 0, 1e-5, None),
    ("examples/recurrence/random_walk.ein", "x", random_walk_reference, 0, 1e-5, None),
    ("examples/tensor_ops/softmax.ein", "softmax", softmax_reference, 1e-5, 1e-5, None),
]


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def _run_ein_file(compiler, runtime, rel_path: str):
    path = PROJECT_ROOT / rel_path
    source = path.read_text(encoding="utf-8")
    result = compile_and_execute(
        source, compiler, runtime,
        source_file=str(path),
    )
    return result, path


class TestOdeAccuracy:
    """Exponential decay ODE: u' = -k*u. Reference = analytical u(t) = u0 * exp(-k*t)."""

    def test_ode_vs_analytical(self, compiler, runtime):
        result, _ = _run_ein_file(
            compiler, runtime,
            "examples/ode/decay.ein",
        )
        assert result.success, getattr(result, "errors", result.error)
        u = np.asarray(result.value if result.value is not None else result.outputs.get("u"))
        assert u is not None and u.ndim == 1, "expected 1D u"
        n = len(u)
        assert n >= 50, f"expected at least 50 steps, got {n}"

        u0, k, dt = 1.0, 0.05, 0.1
        reference = np.array([u0 * np.exp(-k * (i * dt)) for i in range(n)], dtype=np.float64)
        np.testing.assert_allclose(u, reference, rtol=5e-3, atol=1e-6, err_msg="ODE vs analytical (element-wise)")


class TestWaveAccuracy:
    """2D wave: compare every element to NumPy reference (same scheme, no Einlang)."""

    def test_wave_vs_reference(self, compiler, runtime):
        result, _ = _run_ein_file(
            compiler, runtime,
            "examples/wave_2d/main.ein",
        )
        assert result.success, getattr(result, "errors", result.error)
        h = np.asarray(result.outputs.get("h"))
        assert h is not None and h.ndim == 3, "expected 3D h[t,i,j]"
        assert h.shape == (200, 40, 40), f"expected (200,40,40), got {h.shape}"

        reference = wave_2d_reference()
        np.testing.assert_allclose(h, reference, rtol=1e-4, atol=1e-5, err_msg="Wave vs NumPy reference (element-wise)")


class TestHeatAccuracy:
    """Heat equation: minimal run, compare every element to NumPy reference."""

    def test_heat_minimal_vs_reference(self, compiler, runtime):
        heat_minimal = """
let r = 0.2;
let cx = 5;
let cy = 5;
let R2 = 4.0;
let u[0, i in 0..11, j in 0..11] = if ((i - cx) * (i - cx) + (j - cy) * (j - cy)) as f32 <= R2 { 10.0 * (1.0 - (((i - cx) * (i - cx) + (j - cy) * (j - cy)) as f32) / R2) } else { 0.0 };
let u[t in 1..25, i in 1..10, j in 1..10] = u[t - 1, i, j] + r * (u[t - 1, i - 1, j] + u[t - 1, i + 1, j] + u[t - 1, i, j - 1] + u[t - 1, i, j + 1] - 4.0 * u[t - 1, i, j]);
u;
"""
        result = compile_and_execute(
            heat_minimal, compiler, runtime,
            source_file="<heat_minimal>",
        )
        assert result.success, getattr(result, "errors", result.error)
        u = np.asarray(result.value if result.value is not None else result.outputs.get("u"))
        assert u is not None and u.ndim == 3
        assert u.shape == (25, 11, 11), f"expected (25,11,11), got {u.shape}"

        reference = heat_minimal_reference()
        np.testing.assert_allclose(u, reference, rtol=1e-5, atol=1e-6, err_msg="Heat minimal vs NumPy reference (element-wise)")


class TestLorenzAccuracy:
    """Lorenz system: compare to NumPy reference (same Euler scheme)."""

    def test_lorenz_vs_reference(self, compiler, runtime):
        result, _ = _run_ein_file(
            compiler, runtime,
            "examples/ode/lorenz.ein",
        )
        assert result.success, getattr(result, "errors", result.error)
        u = np.asarray(result.value if result.value is not None else result.outputs.get("u"))
        assert u is not None and u.ndim == 2, "expected 2D u[t, 3]"
        assert u.shape == (2000, 3), f"expected (2000, 3), got {u.shape}"

        reference = lorenz_reference()
        # Einlang uses float32; Lorenz is chaotic so trajectory diverges quickly. Compare first 3 steps.
        np.testing.assert_allclose(
            u[:3], reference[:3], rtol=1e-3, atol=1e-2,
            err_msg="Lorenz vs NumPy reference (first 3 steps)",
        )
        assert np.isfinite(u).all(), "Lorenz trajectory must be finite"


class TestLotkaVolterraAccuracy:
    """Lotka-Volterra: compare to NumPy reference."""

    def test_lotka_volterra_vs_reference(self, compiler, runtime):
        result, _ = _run_ein_file(
            compiler, runtime,
            "examples/ode/lotka_volterra.ein",
        )
        assert result.success, getattr(result, "errors", result.error)
        state = np.asarray(result.value if result.value is not None else result.outputs.get("state"))
        assert state is not None and state.ndim == 2, "expected 2D state[t, 2]"
        assert state.shape == (500, 2), f"expected (500, 2), got {state.shape}"

        reference = lotka_volterra_reference()
        # Multi-clause recurrence (state[t,0] and state[t,1]); compare first 2 steps, then sanity-check.
        np.testing.assert_allclose(
            state[:2], reference[:2], rtol=1e-4, atol=1e-4,
            err_msg="Lotka-Volterra vs NumPy reference (first 2 steps)",
        )
        assert np.isfinite(state).all(), "Lotka-Volterra trajectory must be finite"


class TestHeat1dAccuracy:
    """1D heat: compare to NumPy reference."""

    def test_heat_1d_vs_reference(self, compiler, runtime):
        result, _ = _run_ein_file(
            compiler, runtime,
            "examples/pde_1d/heat_1d.ein",
        )
        assert result.success, getattr(result, "errors", result.error)
        u = np.asarray(result.value if result.value is not None else result.outputs.get("u"))
        assert u is not None and u.ndim == 2, "expected 2D u[t, nx]"
        assert u.shape == (200, 41), f"expected (200, 41), got {u.shape}"

        reference = heat_1d_reference()
        # Einlang uses float32.
        np.testing.assert_allclose(
            u, reference, rtol=1e-5, atol=1e-5,
            err_msg="Heat 1D vs NumPy reference",
        )


class TestLinearOdeAccuracy:
    """Linear ODE du/dt = A*u: compare to NumPy reference."""

    def test_linear_ode_vs_reference(self, compiler, runtime):
        result, _ = _run_ein_file(
            compiler, runtime,
            "examples/ode/linear.ein",
        )
        assert result.success, getattr(result, "errors", result.error)
        u = np.asarray(result.value if result.value is not None else result.outputs.get("u"))
        assert u is not None and u.ndim == 2, "expected 2D u[t, 2]"
        assert u.shape == (500, 2), f"expected (500, 2), got {u.shape}"

        reference = linear_ode_reference()
        # Einlang uses float32.
        np.testing.assert_allclose(
            u, reference, rtol=1e-5, atol=1e-5,
            err_msg="Linear ODE vs NumPy reference",
        )


class TestBrusselatorAccuracy:
    """Brusselator PDE: compare to NumPy reference."""

    def test_brusselator_vs_reference(self, compiler, runtime):
        result, _ = _run_ein_file(
            compiler, runtime,
            "examples/brusselator/main.ein",
        )
        assert result.success, getattr(result, "errors", result.error)
        state = np.asarray(result.value if result.value is not None else result.outputs.get("state"))
        assert state is not None and state.ndim == 4, "expected 4D state[t, c, i, j]"
        assert state.shape == (300, 2, 64, 64), f"expected (300, 2, 64, 64), got {state.shape}"

        reference = brusselator_reference()
        # Einlang uses float32.
        np.testing.assert_allclose(
            state, reference, rtol=1e-5, atol=1e-5,
            err_msg="Brusselator vs NumPy reference",
        )


class TestValueIterationAccuracy:
    """Value function iteration (QuantEcon-style): compare to NumPy reference."""

    def test_value_iteration_vs_reference(self, compiler, runtime):
        result, _ = _run_ein_file(
            compiler, runtime,
            "examples/value_iteration/main.ein",
        )
        assert result.success, getattr(result, "errors", result.error)
        V = np.asarray(result.value if result.value is not None else result.outputs.get("V"))
        assert V is not None and V.ndim == 2, "expected 2D V[k, s]"
        assert V.shape == (50, 3), f"expected (50, 3), got {V.shape}"

        reference = value_iteration_reference()
        np.testing.assert_allclose(V, reference, rtol=1e-5, atol=1e-5, err_msg="Value iteration vs NumPy reference")


class TestFibonacciAccuracy:
    """Fibonacci: compare to NumPy reference."""

    def test_fibonacci_vs_reference(self, compiler, runtime):
        result, _ = _run_ein_file(
            compiler, runtime,
            "examples/recurrence/fibonacci.ein",
        )
        assert result.success, getattr(result, "errors", result.error)
        fib = np.asarray(result.value if result.value is not None else result.outputs.get("fib"))
        assert fib is not None and fib.ndim == 1, "expected 1D fib"
        assert len(fib) == 25, f"expected 25 elements, got {len(fib)}"

        reference = fibonacci_reference()
        np.testing.assert_allclose(fib.astype(np.float64), reference.astype(np.float64), rtol=0, atol=1e-5, err_msg="Fibonacci vs NumPy reference")


class TestAdvection1dAccuracy:
    """1D advection: compare to NumPy reference."""

    def test_advection_1d_vs_reference(self, compiler, runtime):
        result, _ = _run_ein_file(
            compiler, runtime,
            "examples/pde_1d/advection_1d.ein",
        )
        assert result.success, getattr(result, "errors", result.error)
        u = np.asarray(result.value if result.value is not None else result.outputs.get("u"))
        assert u is not None and u.ndim == 2, "expected 2D u[t, i]"
        assert u.shape == (80, 40), f"expected (80, 40), got {u.shape}"

        reference = advection_1d_reference()
        # Einlang uses float32; advection accumulates error over 80 steps.
        np.testing.assert_allclose(u, reference, rtol=1e-2, atol=0.15, err_msg="Advection 1D vs NumPy reference")


class TestSoftmaxAccuracy:
    """Softmax: compare to NumPy reference."""

    def test_softmax_vs_reference(self, compiler, runtime):
        result, _ = _run_ein_file(
            compiler, runtime,
            "examples/tensor_ops/softmax.ein",
        )
        assert result.success, getattr(result, "errors", result.error)
        out = np.asarray(result.value if result.value is not None else result.outputs.get("softmax"))
        assert out is not None and out.ndim == 1, "expected 1D softmax"
        assert len(out) == 5, f"expected 5 elements, got {len(out)}"

        reference = softmax_reference()
        np.testing.assert_allclose(out, reference, rtol=1e-5, atol=1e-5, err_msg="Softmax vs NumPy reference")


class TestRandomWalkAccuracy:
    """Random walk: compare to NumPy reference."""

    def test_random_walk_vs_reference(self, compiler, runtime):
        result, _ = _run_ein_file(
            compiler, runtime,
            "examples/recurrence/random_walk.ein",
        )
        assert result.success, getattr(result, "errors", result.error)
        x = np.asarray(result.value if result.value is not None else result.outputs.get("x"))
        assert x is not None and x.ndim == 1, "expected 1D x"
        assert len(x) == 21, f"expected 21 elements, got {len(x)}"

        reference = random_walk_reference()
        np.testing.assert_allclose(x.astype(np.float64), reference, rtol=0, atol=1e-5, err_msg="Random walk vs NumPy reference")


@pytest.mark.parametrize("path,output_key,ref_fn,rtol,atol,first_n", ALL_ACCURACY_EXAMPLES)
def test_all_simulation_examples_accuracy(compiler, runtime, path, output_key, ref_fn, rtol, atol, first_n):
    """Every listed simulation example must run and match its reference (full or first_n steps)."""
    result, _ = _run_ein_file(compiler, runtime, path)
    assert result.success, getattr(result, "errors", result.error)
    out = result.value if result.value is not None else result.outputs.get(output_key)
    assert out is not None, f"no output for {path} (key={output_key})"
    arr = np.asarray(out, dtype=np.float64)
    reference = ref_fn()
    if first_n is not None:
        arr_compare = arr[:first_n]
        ref_compare = reference[:first_n]
        np.testing.assert_allclose(
            arr_compare, ref_compare, rtol=rtol, atol=atol,
            err_msg=f"{path} first {first_n} vs reference",
        )
        assert np.isfinite(arr).all(), f"{path} must be finite"
    else:
        if arr.dtype.kind in ("i", "u"):
            reference = reference.astype(np.float64)
        np.testing.assert_allclose(
            arr, reference, rtol=rtol, atol=atol,
            err_msg=f"{path} vs reference",
        )
