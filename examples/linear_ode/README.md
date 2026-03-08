# Linear ODE: du/dt = A·u

**Julia source:** [DifferentialEquations.jl](https://docs.sciml.ai/DiffEqDocs/stable/) linear ODEs, [QuantEcon.jl](https://quantecon.org/quantecon-jl/) linear algebra. Vector u, matrix A; du/dt = A·u. Discrete: u[t] = u[t−1] + dt·A·u[t−1] or use a small matrix–vector multiply per step.

## Equations

- **State:** u[t, i]; **matrix** A[i, j]. Update: u[t, i] = u[t−1, i] + dt·sum[j](A[i, j] * u[t−1, j]).
- Recurrence over t; one contraction (matrix–vector) per step. Shapes of A and u checked at compile time.

## What this demo shows

- **Matrix–vector product inside recurrence:** sum[j](A[i,j]*u[t−1,j]). Same as [ode/](../ode/) but with vector state and explicit A.
- Einlang strength: indices i, j are first-class; no stringly-typed einsum; shape errors at compile time.

## Run

From repo root:

```bash
python3 -m einlang examples/linear_ode/main.ein
```

Output: `u` shape (500, 2) — trajectory of du/dt = A·u.

## Julia parallel

[Julia demos → Einlang](../../docs/JULIA_DEMOS.md): same as DifferentialEquations.jl / QuantEcon.jl linear ODEs; we express the RHS as Einstein sum.
