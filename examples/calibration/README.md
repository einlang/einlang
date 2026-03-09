# Calibration: fit model to data (real application)

**Parameter estimation** — given synthetic "observations" from an exponential decay process, find the decay rate by grid search over candidate parameters and choosing the one that minimizes sum of squared errors. Same pattern as SciML/Optim.jl calibration and real applications (fit ODE/SDE to data).

In production you would load data (e.g. `load_npy` or from a pipeline); here we generate data from the true model so the best-fit parameter is known.

## Run

```bash
python3 -m einlang examples/calibration/decay_fit.ein
```

Output: `loss[ki]` for each grid point; minimum is at the true parameter (0.05). Optionally extend to gradient-based fitting or other models (SIR, etc.).

## Files

| File | What it does |
|------|--------------|
| `decay_fit.ein` | Grid search: for each candidate decay rate `k`, simulate decay trajectory, compute SSE vs synthetic data; print loss per candidate |

## Julia / real-app parallel

- [Optim.jl](https://github.com/JuliaNLSolvers/Optim.jl), [SciML parameter estimation](https://docs.sciml.ai/Overview/stable/showcase/parameter_estimation/): fit model parameters to data.
- [JuliaHub case studies](https://juliahub.com/case-studies): calibration and inverse problems in production.
