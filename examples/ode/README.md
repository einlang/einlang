# ODE: DifferentialEquations.jl–style time-stepping

**Numerical ODE time-stepping** — aligned with [Julia’s DifferentialEquations.jl](https://docs.sciml.ai/DiffEqDocs/stable/). One folder, ten models: scalar decay, linear system, Lorenz, Lotka–Volterra, **pendulum** ([Classical Physics](https://docs.sciml.ai/DiffEqDocs/stable/examples/classical_physics/)), **van der Pol** ([SciML Benchmarks](https://docs.sciml.ai/SciMLBenchmarksOutput/stable/StiffODE/VanDerPol/)), SIR, harmonic, Fitzhugh–Nagumo, Lorenz 96. Recurrence over time; no spatial dimension (no PDE/stencil).

## Files (one per model)

| File | Model | State |
|------|--------|------|
| `decay.ein` | Exponential decay du/dt = −k·u | Scalar u[t] |
| `linear.ein` | Linear system du/dt = A·u | Vector u[t, i] |
| `lorenz.ein` | Lorenz (chaotic 3D) | u[t, 0..2] = (x,y,z) |
| `lotka_volterra.ein` | Predator–prey | state[t, 0]=u, state[t, 1]=v |
| `pendulum.ein` | Simple pendulum (Classical Physics) | state[t, 0]=θ, state[t, 1]=ω |
| `van_der_pol.ein` | Van der Pol oscillator | state[t, 0]=x, state[t, 1]=y |
| `sir.ein` | SIR epidemic model | state[t, 0..2] = (S, I, R) |
| `harmonic.ein` | Simple harmonic oscillator | state[t, 0]=x, state[t, 1]=v |
| `fitzhugh_nagumo.ein` | Fitzhugh–Nagumo (neural/oscillator) | state[t, 0..1] = (v, w) |
| `lorenz96.ein` | Lorenz 96 (chaotic N-dimensional) | X[t, i] |

All use explicit Euler; same recurrence pattern, different RHS. QuantEcon.jl also uses this style for linear ODEs and Lotka–Volterra.

## Run

From repo root:

```bash
python3 -m einlang examples/ode/decay.ein
python3 -m einlang examples/ode/linear.ein
python3 -m einlang examples/ode/lorenz.ein
python3 -m einlang examples/ode/lotka_volterra.ein
python3 -m einlang examples/ode/pendulum.ein
python3 -m einlang examples/ode/van_der_pol.ein
python3 -m einlang examples/ode/sir.ein
python3 -m einlang examples/ode/harmonic.ein
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
