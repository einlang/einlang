---
layout: default
title: Wave 2D
---

# 2D wave equation (acoustic / linear wave)

**Real physics simulation**: waves on a 2D domain (e.g. drum membrane, water surface). Same complexity tier as the heat equation but with a **two-level recurrence** (reads both `h[t-1]` and `h[t-2]`), showcasing Einlang’s recurrence and vectorization.

## PDE

- **Equation:** ∂²h/∂t² = c² ∇²h  
- **Discretization:** leapfrog in time, 5-point Laplacian in space  
- **Update:** `h^{n+1}[i,j] = 2*h^n[i,j] - h^{n-1}[i,j] + r * Laplacian(h^n)` with `r = c² dt²/dx²`  
- **Stability:** r ≤ 1 for 2D; demo uses r = 0.5  

## What this demo shows

- **Recurrence over time** with two history levels (`h[t-1]`, `h[t-2]`).
- **Vectorized in space** (i, j) via the same hybrid path as the heat equation.
- **Initial condition:** circular bump (Gaussian-like) at the center; zero initial velocity so `h[1] = h[0]`.

## Run

From repo root:

```bash
python3 examples/wave_2d/run_wave.py
python3 examples/wave_2d/run_wave.py --html wave.html
```

No extra setup or weights. Output: HTML animation (or `wave.html`) of the wave field over time.

**Profile** (per-clause time and vectorized/hybrid/scalar path): from repo root or from this dir:

```bash
EINLANG_PROFILE_STATEMENTS=1 EINLANG_DEBUG_VECTORIZE=1 python3 -m einlang main.ein
```

(From repo root use `python3 -m einlang examples/wave_2d/main.ein`.)

## Accuracy

Initial condition (h[1] = h[0]) and shape are checked in the test suite:

```bash
python3 -m pytest tests/examples/test_simulation_accuracy.py::TestWaveAccuracy -v
```

## Files

| File | Description |
|------|-------------|
| `main.ein` | Wave equation in Einlang (recurrence + stencil) |
| `run_wave.py` | Runs Einlang, writes HTML animation (no matplotlib required) |
| `README.md` | This file |
