# Brusselator (reaction–diffusion)

**Julia source:** [SciML showcase: Brusselator](https://docs.sciml.ai/Overview/stable/showcase/brusselator/). Second reaction–diffusion PDE alongside [Gray–Scott](../reaction_diffusion/); same family, different reaction terms.

## Equations

Two species X, Y with diffusion and nonlinear reaction (Brusselator kinetics). Discrete: recurrence in time, stencil in space (e.g. 5-point Laplacian in 2D).

## What this demo shows

- Same pattern as [reaction_diffusion/](../reaction_diffusion/): 4D state (time, channel, i, j), recurrence + stencil.
- Einlang vs Julia: same numerical scheme; compile-time shape checking and Einstein notation here.

## Run

From repo root:

```bash
python3 -m einlang examples/brusselator/main.ein
python3 examples/brusselator/run_brusselator.py --html brusselator.html
```

Output: `state` shape (300, 2, 64, 64). The runner writes an HTML animation of the V field.

## Julia parallel

[Julia demos → Einlang](../../docs/JULIA_DEMOS.md): Brusselator is the SciML showcase reaction–diffusion; we have Gray–Scott in [reaction_diffusion/](../reaction_diffusion/). This folder is the direct Brusselator counterpart.
