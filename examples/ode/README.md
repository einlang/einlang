---
layout: default
title: ODE examples
---

# ODE: DifferentialEquations.jl–style time-stepping

**Numerical ODE time-stepping** — aligned with [Julia’s DifferentialEquations.jl](https://docs.sciml.ai/DiffEqDocs/stable/). Suite plus standalone: decay, linear, harmonic, pendulum, Van der Pol, Lotka–Volterra (in `ode_suite.ein`); Lorenz, SIR, FitzHugh–Nagumo, Lorenz 96 (separate files). Recurrence over time; no spatial dimension (no PDE/stencil).

## Files

| File | Model | State |
|------|--------|------|
| **ode_suite.ein** | Decay, linear, harmonic, pendulum, Van der Pol, Lotka–Volterra (six in one) | u_decay, u_linear, state_harmonic, state_pendulum, state_van_der_pol, state_lotka |
| `lorenz.ein` | Lorenz (chaotic 3D) | u[t, 0..2] = (x,y,z) |
| `sir.ein` | SIR epidemic model | state[t, 0..2] = (S, I, R) |
| `fitzhugh_nagumo.ein` | FitzHugh–Nagumo (neural/oscillator) | state[t, 0..1] = (v, w) |
| `lorenz96.ein` | Lorenz 96 (chaotic N-dimensional) | X[t, i] |

All use explicit Euler; same recurrence pattern, different RHS. QuantEcon.jl also uses this style for linear ODEs and Lotka–Volterra.

## Run

From repo root:

```bash
python3 -m einlang examples/ode/ode_suite.ein
python3 -m einlang examples/ode/lorenz.ein
python3 -m einlang examples/ode/sir.ein
python3 -m einlang examples/ode/fitzhugh_nagumo.ein
python3 -m einlang examples/ode/lorenz96.ein
```

## Accuracy

Tests compare each to analytical or NumPy reference:

```bash
python3 -m pytest tests/examples/test_simulation_accuracy.py -k "Ode or Lorenz or Lotka or Linear" -v
```

## Julia parallel

[Julia demos → Einlang](../../docs/JULIA_DEMOS.md): this folder is our **DifferentialEquations.jl** counterpart (explicit Euler, same equations). Like [DiffEqDocs](https://docs.sciml.ai/DiffEqDocs/stable/) we lead with the problem (equations in each `.ein`), then the code; each file has a **Julia equivalent** in comments. We use Einstein notation and recurrence instead of a solver API.
