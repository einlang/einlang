
# Parameter estimation & workflow applications

**Calibrate then use** (fit model to data, then simulate/forecast), **steady-state risk** (Markov), and **sensor fusion** (Kalman filter).

## Run

```bash
python3 -m einlang examples/applications/decay_calibration.ein
python3 -m einlang examples/applications/markov_credit.ein
python3 -m einlang examples/applications/kalman_filter/main.ein
```

## Files

| Directory / File | Pattern | What it does |
|------------------|---------|--------------|
| `decay_calibration.ein` | **Parameter estimation** | Fit exponential decay (u = u0·e^{-kt}) to synthetic observations via log-linear least squares (`std::numerics::optim`), then simulate forward with RK4 (`std::numerics::ode`). Same workflow as SciML/Optim.jl calibration. |
| `markov_credit.ein` | **Steady-state / risk** | Stationary distribution of a 3-state credit-rating transition model (Good/Fair/Poor). Converts the recurrence pattern from [recurrence/recurrence_suite.ein](https://github.com/einlang/einlang/blob/main/examples/recurrence/recurrence_suite.ein) into a real-world risk/portfolio application. |
| **[kalman_filter/](https://github.com/einlang/einlang/tree/main/examples/applications/kalman_filter)** | **Sensor fusion** | Discrete-time Kalman filter (constant-velocity model): predict/update over noisy position measurements. Migrated from a 223-line NumPy application; see [kalman_filter/README](https://github.com/einlang/einlang/blob/main/examples/applications/kalman_filter/README.md). |

## Julia / real-app parallel

- **Calibration:** [SciML parameter estimation](https://docs.sciml.ai/Overview/stable/showcase/parameter_estimation/), Optim.jl or GLM for fit; DifferentialEquations.jl for ODE.
- **Markov / credit:** [QuantEcon finite Markov](https://julia.quantecon.org/introduction_dynamics/finite_markov.html); transition matrices in Basel/IFRS 9–style risk models.
