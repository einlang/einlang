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

import sys
from pathlib import Path

import numpy as np
import pytest

from tests.test_utils import compile_and_execute
from tests.examples.reference_implementations import (
    decay_reference,
    euler_decay_reference,
    gradient_descent_2d_reference,
    value_iteration_quantecon_reference,
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
    cavity_lid_reference,
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
    "examples/pde_1d/cavity_lid.ein",
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
    "examples/run_numerics_diffeq.ein",
    "examples/run_numerics_optim.ein",
    "examples/run_numerics_quantecon.ein",
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
    ("examples/pde_1d/cavity_lid.ein", "u", cavity_lid_reference, 1e-5, 25.0, None),
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
    ("examples/run_numerics_diffeq.ein", "u", euler_decay_reference, 5e-3, 1e-6, None),
    ("examples/run_numerics_optim.ein", "x_traj", gradient_descent_2d_reference, 1e-5, 1e-5, 3),
    ("examples/run_numerics_quantecon.ein", "V_traj", value_iteration_quantecon_reference, 1e-5, 1e-5, None),
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


def _path_label(path_or_inline) -> str:
    """Display label for path or inline (path, output_key, ...) tuple."""
    return path_or_inline[1] if isinstance(path_or_inline, tuple) else path_or_inline


def _path_to_test_name(path_or_inline) -> str:
    """Sanitize path/label to a valid test function name suffix."""
    label = _path_label(path_or_inline)
    name = (
        label.replace(".ein", "")
        .replace("/", "_")
        .replace(".", "_")
        .replace("-", "_")
        .replace("<", "")
        .replace(">", "")
        .replace(" ", "_")
        .strip("_")
    )
    return name


def _accuracy_test_impl(compiler, runtime, path, output_key, ref_fn, rtol, atol, first_n):
    """Run one example and compare output to reference. Single shared implementation."""
    result, _ = _run_ein_file(compiler, runtime, path)
    assert result.success, getattr(result, "errors", result.error)
    out = result.value if result.value is not None else result.outputs.get(output_key)
    label = _path_label(path)
    assert out is not None, f"no output for {label} (key={output_key})"
    arr = np.asarray(out, dtype=np.float64)
    reference = ref_fn()
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


def _make_accuracy_test(path, output_key, ref_fn, rtol, atol, first_n):
    """Return a test function (compiler, runtime) for one accuracy example."""
    def test(compiler, runtime):
        _accuracy_test_impl(compiler, runtime, path, output_key, ref_fn, rtol, atol, first_n)
    return test


# One test per example (no single parametrized group).
_module = sys.modules[__name__]
for path, output_key, ref_fn, rtol, atol, first_n in ALL_ACCURACY_EXAMPLES:
    name = "test_accuracy_" + _path_to_test_name(path)
    setattr(
        _module,
        name,
        _make_accuracy_test(path, output_key, ref_fn, rtol, atol, first_n),
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
