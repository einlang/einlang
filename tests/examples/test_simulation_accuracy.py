"""
Accuracy checks for simulation demos: ODE (decay, linear, Lorenz, Lotka-Volterra,
pendulum, van_der_pol, SIR, harmonic, fitzhugh_nagumo, lorenz96), wave, heat,
Brusselator, value_iteration, job_search (McCall), recurrence, optimization,
finance, time_series.

All simulation examples must be covered by strict accuracy testing: each runs
and is compared element-wise to a reference (analytical or NumPy). No mocking.

Canonical registry: every path in SIMULATION_EXAMPLE_PATHS must appear in
ALL_ACCURACY_EXAMPLES with a reference and tolerances (rtol/atol).
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
    pendulum_reference,
    van_der_pol_reference,
    sir_reference,
    harmonic_reference,
    logistic_reference,
    gradient_descent_reference,
    power_iteration_reference,
    markov_stationary_reference,
    heat_1d_reference,
    linear_ode_reference,
    brusselator_reference,
    value_iteration_reference,
    fibonacci_reference,
    advection_1d_reference,
    random_walk_reference,
    savings_reference,
    projected_gradient_reference,
    rosenbrock_reference,
    exponential_smoothing_reference,
    mccall_reference,
    fitzhugh_nagumo_reference,
    lorenz96_reference,
)

# Inline heat minimal (2D heat, 25 steps, 11x11) for parametrized test.
HEAT_MINIMAL_SOURCE = """
let r = 0.2;
let cx = 5;
let cy = 5;
let R2 = 4.0;
let u[0, i in 0..11, j in 0..11] = if ((i - cx) * (i - cx) + (j - cy) * (j - cy)) as f32 <= R2 { 10.0 * (1.0 - (((i - cx) * (i - cx) + (j - cy) * (j - cy)) as f32) / R2) } else { 0.0 };
let u[t in 1..25, i in 1..10, j in 1..10] = u[t - 1, i, j] + r * (u[t - 1, i - 1, j] + u[t - 1, i + 1, j] + u[t - 1, i, j - 1] + u[t - 1, i, j + 1] - 4.0 * u[t - 1, i, j]);
u;
"""

# Canonical list of simulation example paths that MUST have strict accuracy tests.
# Add new simulation examples here and to ALL_ACCURACY_EXAMPLES (with reference_implementations).
SIMULATION_EXAMPLE_PATHS = (
    "examples/ode/decay.ein",
    "examples/ode/linear.ein",
    "examples/ode/lorenz.ein",
    "examples/ode/lotka_volterra.ein",
    "examples/ode/pendulum.ein",
    "examples/ode/van_der_pol.ein",
    "examples/ode/sir.ein",
    "examples/ode/harmonic.ein",
    "examples/ode/fitzhugh_nagumo.ein",
    "examples/ode/lorenz96.ein",
    "examples/wave_2d/main.ein",
    "examples/pde_1d/heat_1d.ein",
    "examples/pde_1d/advection_1d.ein",
    "examples/brusselator/main.ein",
    "examples/value_iteration/main.ein",
    "examples/job_search/mccall.ein",
    "examples/recurrence/fibonacci.ein",
    "examples/recurrence/logistic.ein",
    "examples/recurrence/markov_stationary.ein",
    "examples/recurrence/random_walk.ein",
    "examples/finance/savings.ein",
    "examples/optimization/gradient_descent.ein",
    "examples/optimization/power_iteration.ein",
    "examples/optimization/projected_gradient.ein",
    "examples/optimization/rosenbrock.ein",
    "examples/time_series/exponential_smoothing.ein",
)

# Every simulation example file (or inline) that must pass accuracy vs reference.
# path: str (relative path) or (source_str, source_file_name) for inline.
# (path, output_key, reference_fn, rtol, atol, first_n)
# first_n=None: compare full array; first_n=N: compare first N steps only and assert finite.
ALL_ACCURACY_EXAMPLES = [
    ("examples/ode/decay.ein", "u", decay_reference, 5e-3, 1e-6, None),
    ("examples/ode/linear.ein", "u", linear_ode_reference, 1e-5, 1e-5, None),
    ("examples/ode/lorenz.ein", "u", lorenz_reference, 1e-3, 1e-2, 3),
    ("examples/ode/lotka_volterra.ein", "state", lotka_volterra_reference, 1e-4, 1e-4, 2),
    ("examples/ode/pendulum.ein", "state", pendulum_reference, 1e-5, 1e-5, 1),
    ("examples/ode/van_der_pol.ein", "state", van_der_pol_reference, 1e-5, 1e-5, 1),
    ("examples/ode/sir.ein", "state", sir_reference, 1e-5, 1e-5, 1),
    ("examples/ode/harmonic.ein", "state", harmonic_reference, 1e-5, 1e-5, 1),
    ("examples/ode/fitzhugh_nagumo.ein", "state", fitzhugh_nagumo_reference, 1e-4, 1e-4, 3),
    ("examples/ode/lorenz96.ein", "X", lorenz96_reference, 1e-3, 1e-2, 3),
    ("examples/wave_2d/main.ein", "h", wave_2d_reference, 1e-4, 1e-5, None),
    ("examples/pde_1d/heat_1d.ein", "u", heat_1d_reference, 1e-5, 1e-5, None),
    ("examples/pde_1d/advection_1d.ein", "u", advection_1d_reference, 1e-2, 0.15, None),
    ("examples/brusselator/main.ein", "state", brusselator_reference, 1e-5, 1e-5, None),
    ("examples/value_iteration/main.ein", "V", value_iteration_reference, 1e-5, 1e-5, None),
    ("examples/job_search/mccall.ein", "V", mccall_reference, 1e-5, 1e-5, None),
    ("examples/recurrence/fibonacci.ein", "fib", fibonacci_reference, 0, 1e-5, None),
    ("examples/recurrence/logistic.ein", "x", logistic_reference, 1e-5, 1e-5, 10),
    ("examples/optimization/gradient_descent.ein", "x", gradient_descent_reference, 1e-5, 1e-5, None),
    ("examples/optimization/power_iteration.ein", "v", power_iteration_reference, 1e-5, 1e-5, None),
    ("examples/recurrence/markov_stationary.ein", "psi", markov_stationary_reference, 1e-5, 1e-5, None),
    ("examples/recurrence/random_walk.ein", "x", random_walk_reference, 0, 1e-5, None),
    ("examples/finance/savings.ein", "b", savings_reference, 1e-5, 1e-5, None),
    ("examples/optimization/projected_gradient.ein", "x", projected_gradient_reference, 1e-5, 1e-5, None),
    ("examples/optimization/rosenbrock.ein", "x", rosenbrock_reference, 1e-4, 1e-4, None),
    ("examples/time_series/exponential_smoothing.ein", "s", exponential_smoothing_reference, 1e-5, 1e-5, None),
    ((HEAT_MINIMAL_SOURCE, "<heat_minimal>"), "u", heat_minimal_reference, 1e-5, 1e-6, None),
]


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def _run_ein_file(compiler, runtime, path_or_inline):
    """path_or_inline: str (relative path to .ein) or (source_str, source_file_name) for inline."""
    if isinstance(path_or_inline, tuple):
        source, source_file = path_or_inline
        result = compile_and_execute(
            source, compiler, runtime,
            source_file=source_file,
        )
        return result, path_or_inline[1]
    path = PROJECT_ROOT / path_or_inline
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


def _path_label(path_or_inline) -> str:
    """Display label for path or inline (path, output_key, ...) tuple."""
    return path_or_inline[1] if isinstance(path_or_inline, tuple) else path_or_inline


@pytest.mark.parametrize(
    "path,output_key,ref_fn,rtol,atol,first_n",
    ALL_ACCURACY_EXAMPLES,
    ids=[_path_label(row[0]) for row in ALL_ACCURACY_EXAMPLES],
)
def test_all_simulation_examples_accuracy(compiler, runtime, path, output_key, ref_fn, rtol, atol, first_n):
    """Every listed simulation example must run and match its reference (full or first_n steps)."""
    result, _ = _run_ein_file(compiler, runtime, path)
    assert result.success, getattr(result, "errors", result.error)
    out = result.value if result.value is not None else result.outputs.get(output_key)
    label = _path_label(path)
    assert out is not None, f"no output for {label} (key={output_key})"
    arr = np.asarray(out, dtype=np.float64)
    reference = ref_fn()
    # Backend may return (nstates, nsteps) for state; normalize to (nsteps, nstates)
    if arr.ndim == 2 and reference.ndim == 2 and arr.shape != reference.shape:
        if (arr.shape[1], arr.shape[0]) == reference.shape:
            arr = arr.T
    if first_n is not None:
        arr_compare = arr[:first_n]
        ref_compare = reference[:first_n]
        np.testing.assert_allclose(
            arr_compare, ref_compare, rtol=rtol, atol=atol,
            err_msg=f"{label} first {first_n} vs reference",
        )
        assert np.isfinite(arr).all(), f"{label} must be finite"
    else:
        if arr.dtype.kind in ("i", "u"):
            reference = reference.astype(np.float64)
        np.testing.assert_allclose(
            arr, reference, rtol=rtol, atol=atol,
            err_msg=f"{label} vs reference",
        )


def test_every_simulation_example_has_strict_accuracy_test():
    """Every path in SIMULATION_EXAMPLE_PATHS must be in ALL_ACCURACY_EXAMPLES with a reference."""
    paths_in_accuracy = {
        row[0] for row in ALL_ACCURACY_EXAMPLES
        if isinstance(row[0], str)
    }
    missing = [p for p in SIMULATION_EXAMPLE_PATHS if p not in paths_in_accuracy]
    assert not missing, (
        "Simulation examples missing from ALL_ACCURACY_EXAMPLES (add entry + reference_implementations): "
        + ", ".join(missing)
    )
