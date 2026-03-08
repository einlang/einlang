# ODE: DifferentialEquations.jl–style time-stepping

**Numerical ODE time-stepping** — aligned with [Julia’s DifferentialEquations.jl](https://docs.sciml.ai/DiffEqDocs/stable/). One folder, four models: scalar decay, linear system, Lorenz, Lotka–Volterra. Recurrence over time; no spatial dimension (no PDE/stencil).

## Files (one per model)

| File | Model | State |
|------|--------|------|
| `decay.ein` | Exponential decay du/dt = −k·u | Scalar u[t] |
| `linear.ein` | Linear system du/dt = A·u | Vector u[t, i] |
| `lorenz.ein` | Lorenz (chaotic 3D) | u[t, 0..2] = (x,y,z) |
| `lotka_volterra.ein` | Predator–prey | state[t, 0]=u, state[t, 1]=v |

All use explicit Euler; same recurrence pattern, different RHS. QuantEcon.jl also uses this style for linear ODEs and Lotka–Volterra.

## Run

From repo root:

```bash
python3 -m einlang examples/ode/decay.ein       # decay
python3 -m einlang examples/ode/linear.ein      # linear A·u
python3 -m einlang examples/ode/lorenz.ein      # Lorenz
python3 -m einlang examples/ode/lotka_volterra.ein
```

## Accuracy

Tests compare each to analytical or NumPy reference:

```bash
python3 -m pytest tests/examples/test_simulation_accuracy.py -k "Ode or Lorenz or Lotka or Linear" -v
```

## Julia parallel

[Julia demos → Einlang](../../docs/JULIA_DEMOS.md): this folder is our **DifferentialEquations.jl** counterpart (explicit Euler, same equations). We use Einstein notation and recurrence instead of a solver API.
