---
layout: default
title: Kalman filter
---

# Kalman filter (constant-velocity model)

**Real application:** Track position and velocity from noisy position-only measurements (e.g. radar, GPS, vision). Migrated from a **223-line NumPy application** (`numpy_reference.py`) that implements the same algorithm.

## Model

- **State** \(x = [\text{position}, \text{velocity}]'\).
- **Dynamics:** \(x_{k+1} = F x_k + w_k\), with \(F = \begin{bmatrix} 1 & \Delta t \\ 0 & 1 \end{bmatrix}\) (constant velocity).
- **Measurement:** \(z_k = H x_k + v_k\), \(H = [1, 0]\) (position only).
- **Predict:** \(x_{\text{pred}} = F x\), \(P_{\text{pred}} = F P F' + Q\).
- **Update:** innovation \(y = z - H x_{\text{pred}}\), gain \(K = P_{\text{pred}} H' / (H P_{\text{pred}} H' + R)\), then \(x = x_{\text{pred}} + K y\), \(P = (I - K H) P_{\text{pred}}\).

## Run

From repo root:

```bash
# Einlang (uses inline synthetic measurements; no Python needed)
python3 -m einlang examples/applications/kalman_filter/main.ein

# NumPy reference (generates data, runs filter, saves x_traj, P_traj, innovations)
python3 examples/applications/kalman_filter/numpy_reference.py
```

To compare Einlang vs NumPy on the same data: run the NumPy script first (creates `kalman_measurements.npy` in repo root), then in `main.ein` replace the synthetic `z` with `use std::io::load_npy; let z = load_npy("kalman_measurements.npy") as [f32; 100];` and run from repo root.

## Files

| File | Role |
|------|------|
| `main.ein` | Einlang implementation: recurrence over predict/update, outputs final state and P diag. |
| `numpy_reference.py` | NumPy application (>200 lines): CV model, data generation, filter loop, RMSE/log-likelihood, two scenarios. |

## References

- [FilterPy](https://github.com/rlabbe/filterpy), [Kalman and Bayesian Filters in Python](https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python)
- [Kalman filter (Wikipedia)](https://en.wikipedia.org/wiki/Kalman_filter)
