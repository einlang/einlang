#!/usr/bin/env python3
"""
Focused example tests for applications with autodiff sensitivity checks.
"""

from pathlib import Path

import numpy as np

from tests.print_at_fixtures import compile_exec_capture_print_at


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def test_decay_calibration_autodiff_prints_and_refines_fit(session_compiler, session_runtime):
    """Run the new autodiff decay calibration example and pin its printed refinement values."""
    source = (PROJECT_ROOT / "examples" / "applications" / "decay_calibration_autodiff.ein").read_text(
        encoding="utf-8"
    )
    c_ok, e_ok, out, err = compile_exec_capture_print_at(
        source, compiler=session_compiler, runtime=session_runtime
    )
    assert c_ok, err
    assert e_ok, err

    lines = out.splitlines()
    assert len(lines) == 4, f"expected 4 printed values, got {len(lines)}: {lines!r}"

    actual = np.array([float(x) for x in lines], dtype=np.float64)

    dt = 0.1
    n_obs = 12
    k_true = 0.08
    u0_true = 2.0
    k = 0.05
    u0 = 1.6
    time = np.arange(n_obs, dtype=np.float64) * dt
    data = u0_true * np.exp(-k_true * time)
    fitted = u0 * np.exp(-k * time)
    residuals = data - fitted
    sse = np.sum(residuals * residuals)
    d_sse_dk = np.sum(2.0 * residuals * (time * fitted))
    d_sse_du0 = np.sum(2.0 * residuals * (-np.exp(-k * time)))
    k_next = k - 0.01 * d_sse_dk
    u0_next = u0 - 0.01 * d_sse_du0
    fitted_next = u0_next * np.exp(-k_next * time)
    sse_next = np.sum((data - fitted_next) ** 2)
    expected = np.array([sse, sse_next, k_next, u0_next], dtype=np.float64)

    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
    assert actual[1] < actual[0], "SSE should decrease after the autodiff update"
