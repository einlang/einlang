# Lorenz system

**Julia source:** [DifferentialEquations.jl](https://docs.sciml.ai/DiffEqDocs/stable/) ODE examples. Classic 3D chaotic ODE: dx/dt = σ(y−x), dy/dt = x(ρ−z)−y, dz/dt = xy−βz.

## Equations

- **State:** u = (x, y, z); discrete u[t, i] with i ∈ {0,1,2}.
- **Euler (or small step):** u[t,i] from u[t−1,:] using the Lorenz RHS. Recurrence over t; shape-checked vector state.

## What this demo shows

- Recurrence over time for a **small vector** state (3 components), like [ode/](../ode/) but multi-dimensional.
- No PDE/stencil; pure ODE time-stepping. Good “minimal chaos” showcase vs Julia.

## Run

From repo root:

```bash
python3 -m einlang examples/lorenz/main.ein
```

Output: 3D trajectory `u` shape (2000, 3) — x, y, z.

## Julia parallel

[Julia demos → Einlang](../../docs/JULIA_DEMOS.md): same class as DifferentialEquations.jl ODEs; we use Einstein notation and recurrence.
