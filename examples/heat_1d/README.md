# 1D heat equation

**Julia source:** [MethodOfLines.jl](https://docs.sciml.ai/MethodOfLines/stable/), 1D PDE tutorials. ∂u/∂t = α ∂²u/∂x²; discrete: recurrence in t, 3-point stencil in x (i−1, i, i+1).

## Equations

- **State:** u[t, i] for time t and grid index i.
- **Update:** u[t,i] = u[t−1,i] + dt·α·(u[t−1,i−1] − 2u[t−1,i] + u[t−1,i+1])/dx². Where clause or index algebra for the stencil.

## What this demo shows

- **One space dimension** (index i); simpler than [wave_2d](../wave_2d/) or [reaction_diffusion](../reaction_diffusion/). Good “first PDE” for Einlang.
- Where clause / stencil indexing; compile-time shape checking on u[t, i].

## Run

From repo root:

```bash
python3 -m einlang examples/heat_1d/main.ein
python3 examples/heat_1d/run_heat_1d.py --html heat_1d.html
```

Output: `u` shape (200, 41). The runner writes a static heatmap HTML (time vs space).

## Julia parallel

[Julia demos → Einlang](../../docs/JULIA_DEMOS.md): same discrete-PDE style as MethodOfLines.jl; we write the discrete Laplacian in Einstein notation.
