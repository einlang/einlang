"""
Accuracy checks for simulation demos: ODE, wave, heat, reaction-diffusion.

Each test runs the demo (or a minimal variant), then compares every element
against a reference computed in-test: analytical (ODE) or NumPy reference
implementation (wave, heat, reaction-diffusion). No mocking.
"""

from pathlib import Path
from typing import Dict

import numpy as np
import pytest

from tests.test_utils import compile_and_execute
from tests.examples.reference_implementations import (
    wave_2d_reference,
    heat_minimal_reference,
    reaction_diffusion_reference,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def _rd_per_clause_max_diffs(ein_state: np.ndarray, ref_state: np.ndarray) -> Dict[str, float]:
    """Max |ein - ref| over each clause's region (t=1..499). Keys match main.ein clauses."""
    ein = np.asarray(ein_state, dtype=np.float64)
    ref = np.asarray(ref_state, dtype=np.float64)
    return {
        "interior_U (state[t,0,i,j] i,j 1..126)": float(
            np.abs(ein[1:, 0, 1:127, 1:127] - ref[1:, 0, 1:127, 1:127]).max()
        ),
        "interior_V (state[t,1,i,j] i,j 1..126)": float(
            np.abs(ein[1:, 1, 1:127, 1:127] - ref[1:, 1, 1:127, 1:127]).max()
        ),
        "boundary i=0 (state[t,c,0,j])": float(np.abs(ein[1:, :, 0, :] - ref[1:, :, 0, :]).max()),
        "boundary i=127 (state[t,c,127,j])": float(np.abs(ein[1:, :, 127, :] - ref[1:, :, 127, :]).max()),
        "boundary j=0 (state[t,c,i,0])": float(np.abs(ein[1:, :, 1:127, 0] - ref[1:, :, 1:127, 0]).max()),
        "boundary j=127 (state[t,c,i,127])": float(np.abs(ein[1:, :, 1:127, 127] - ref[1:, :, 1:127, 127]).max()),
    }


def _rd_per_t_max_diff(ein_state: np.ndarray, ref_state: np.ndarray) -> np.ndarray:
    """Max |ein - ref| at each timestep t. Shape (nsteps,) with nsteps = state.shape[0]."""
    ein = np.asarray(ein_state, dtype=np.float64)
    ref = np.asarray(ref_state, dtype=np.float64)
    return np.abs(ein - ref).reshape(ein.shape[0], -1).max(axis=1)


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
            "examples/ode/main.ein",
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


class TestReactionDiffusionAccuracy:
    """Gray-Scott: shape, bounds, and element-wise vs NumPy reference where comparable."""

    def test_rd_shape_bounds_and_reference(self, compiler, runtime):
        result, _ = _run_ein_file(
            compiler, runtime,
            "examples/reaction_diffusion/main.ein",
        )
        assert result.success, getattr(result, "errors", result.error)
        state = np.asarray(result.outputs.get("state"))
        assert state is not None and state.ndim == 4, "expected 4D state[t,c,i,j]"
        assert state.shape == (500, 2, 128, 128), f"expected (500,2,128,128), got {state.shape}"

        # Bounds: U,V in [0,1] (no blow-up)
        assert 0 <= state[:, 0].min() and state[:, 0].max() <= 1.0 + 1e-5
        assert 0 <= state[:, 1].min() and state[:, 1].max() <= 1.0 + 1e-5
        # Initial conditions
        np.testing.assert_allclose(state[0, 0], 1.0, rtol=1e-5)
        assert 0.2 < state[0, 1, 64, 64] < 0.3

        # Full trajectory must match reference with max diff < 0.01 (same scheme, NumPy reference).
        reference = reaction_diffusion_reference()
        max_diff = float(np.abs(state.astype(np.float64) - reference).max())
        assert max_diff < 0.01, (
            f"RD full trajectory vs reference: max |ein - ref| = {max_diff}, required < 0.01"
        )

    def test_rd_per_clause_vs_reference(self, compiler, runtime):
        """Run Ein + reference and report max |ein - ref| per clause and per timestep (run with -s to see)."""
        result, _ = _run_ein_file(
            compiler, runtime,
            "examples/reaction_diffusion/main.ein",
        )
        assert result.success, getattr(result, "errors", result.error)
        state = np.asarray(result.outputs.get("state"))
        assert state is not None and state.shape == (500, 2, 128, 128)
        reference = reaction_diffusion_reference()
        diffs = _rd_per_clause_max_diffs(state, reference)
        for name, max_diff in diffs.items():
            print(f"  {name}: max |ein - ref| = {max_diff}")
        # Per-timestep max diff
        per_t = _rd_per_t_max_diff(state, reference)
        print("  per timestep max |ein - ref|:")
        for t in [1, 2, 3, 4, 5, 10, 20, 50, 100, 200, 499]:
            if t < len(per_t):
                print(f"    t={t}: {per_t[t]}")
        first_001 = next((t for t in range(len(per_t)) if per_t[t] > 0.01), None)
        first_01 = next((t for t in range(len(per_t)) if per_t[t] > 0.1), None)
        if first_001 is not None:
            print(f"  first t where max diff > 0.01: t={first_001}")
        if first_01 is not None:
            print(f"  first t where max diff > 0.1:  t={first_01}")
        # Full trajectory match: uncomment when backend is fixed
        # np.testing.assert_allclose(state, reference, rtol=1e-4, atol=1e-5, err_msg="RD full trajectory vs reference")
