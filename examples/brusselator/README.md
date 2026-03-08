# Brusselator (reaction–diffusion)

**Julia source:** [SciML showcase: Brusselator](https://docs.sciml.ai/Overview/stable/showcase/brusselator/). Reaction–diffusion PDE aligned with this Julia demo.

## Equations

Two species X, Y with diffusion and nonlinear reaction (Brusselator kinetics). Discrete: recurrence in time, stencil in space (e.g. 5-point Laplacian in 2D).

## What this demo shows

- 4D state (time, channel, i, j), recurrence + stencil (Laplacian + reaction terms).
- Einlang vs Julia: same numerical scheme; compile-time shape checking and Einstein notation here.

## Run

From repo root:

```bash
python3 -m einlang examples/brusselator/main.ein
python3 examples/brusselator/run_brusselator.py --html brusselator.html
```

Output: `state` shape (300, 2, 64, 64). The runner writes an HTML animation of the V field.

## Julia parallel

[Julia demos → Einlang](../../docs/JULIA_DEMOS.md): this folder is the direct SciML Brusselator counterpart.
