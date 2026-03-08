# Lotka–Volterra (predator–prey)

**Julia source:** Standard [DifferentialEquations.jl](https://docs.sciml.ai/DiffEqDocs/stable/) and [QuantEcon](https://julia.quantecon.org/) style ODE. Two species: prey u, predator v; du/dt = au − buv, dv/dt = −cv + duv.

## Equations

- **State:** (u, v); discrete u[t], v[t] or single state vector u[t, i] with i ∈ {0,1}.
- **Euler:** one recurrence over t; RHS uses u and v from t−1.

## What this demo shows

- Same recurrence pattern as [ode/](../ode/) with **two coupled scalars** (or a 2-vector). Minimal, well-known example.
- Good side-by-side with a Julia script: same math, shape-safe indices here.

## Run

From repo root:

```bash
python3 -m einlang examples/lotka_volterra/main.ein
```

Output: `state` shape (500, 2) — prey and predator over time.

## Julia parallel

[Julia demos → Einlang](../../docs/JULIA_DEMOS.md): same ODE time-stepping as DifferentialEquations.jl; also appears in economics/ecology tutorials.
