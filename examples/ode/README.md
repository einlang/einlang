# ODE: exponential decay

**Numerical ODE time-stepping** — the same use case as [Julia’s DifferentialEquations.jl](https://docs.sciml.ai/DiffEqDocs/stable/). One state, recurrence over time; no spatial dimension.

## ODE

- **Equation:** du/dt = −k·u  
- **Initial:** u(0) = u0  
- **Euler step:** u[t] = u[t−1] + dt·(−k·u[t−1]) = u[t−1]·(1 − k·dt)

Parameters: k = 0.05, dt = 0.1, 50 steps. Output: trajectory `u[0..50]` (decay from 1 toward 0).

## What this demo shows

- **Recurrence over time** for a scalar ODE (one index `t`).
- **No PDE/stencil** — just the discrete time step, like a minimal DiffEq example.
- Einlang checks shapes at compile time; you write the update once and the backend runs it.

## Run

From repo root or from this directory:

```bash
python3 -m einlang examples/ode/main.ein
# or: python3 -m einlang main.ein  (from examples/ode)
```

**Profile** (per-clause time and path):

```bash
EINLANG_PROFILE_STATEMENTS=1 EINLANG_DEBUG_VECTORIZE=1 python3 -m einlang examples/ode/main.ein
```

For more steps (e.g. 500), increase the range in `main.ein` (e.g. `t in 1..500`). Max steps is `config.DEFAULT_EINSTEIN_LOOP_MAX` (5000).

## Accuracy

Trajectory is checked against the analytical solution u(t) = u0·exp(−k·t) in the test suite:

```bash
python3 -m pytest tests/examples/test_simulation_accuracy.py::TestOdeAccuracy -v
```

## Files

| File       | Description                          |
|------------|--------------------------------------|
| `main.ein` | Exponential decay ODE (recurrence)   |
| `README.md`| This file                            |

## Julia parallel

[Julia demos → Einlang](../../docs/JULIA_DEMOS.md): this example is our counterpart to **DifferentialEquations.jl** ODE time-stepping (explicit Euler). Same idea; we use Einstein notation and recurrence instead of a solver API.
