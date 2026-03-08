# Gray-Scott reaction-diffusion

**Two coupled PDEs** with nonlinear reaction terms; produces spots, stripes, or labyrinth patterns depending on parameters. More involved than a single heat or wave equation: two concentration fields (U, V), Laplacian diffusion for each, and coupling via the reaction term UV².

## Equations

- **U:** ∂U/∂t = D_u ∇²U − UV² + f(1−U)  
- **V:** ∂V/∂t = D_v ∇²V + UV² − (f+k)V  

Explicit time step; 5-point Laplacian in 2D. Parameters: D_u, D_v (diffusion), f (feed), k (kill).

## What this demo shows

- **Single 4D state** `state[t, c, i, j]`: c=0 is U, c=1 is V (both updated in one recurrence so they stay coupled).
- **Recurrence in time** (t); **vectorized** in channel (c), i, j.
- **Conditional in the body** (`if c == 0 { U update } else { V update }`).
- **Nonlinear terms** UV² in both equations.

## Run

From repo root:

```bash
python3 examples/reaction_diffusion/run_rd.py
python3 examples/reaction_diffusion/run_rd.py --html rd.html
```

No weights or extra data. Output: HTML animation of the V field (pattern is most visible in V).

## Files

| File | Description |
|------|-------------|
| `main.ein` | Gray-Scott in Einlang (4D state, recurrence, if on channel) |
| `run_rd.py` | Runs Einlang, writes HTML animation |
| `README.md` | This file |
