# Einlang Examples

**First time?** [Try it in 30 seconds](../README.md#try-it) or read [Getting started](../docs/GETTING_STARTED.md). **Want one capability?** [Examples by feature](../README.md#examples).

Examples are grouped **by domain** below. For a step-by-step path (basics → demos → MNIST → ViT → Whisper), see [Learning path](#learning-path).

---

## By domain

### Scientific simulation (ODEs & PDEs)

Time-stepping and spatial stencils; no weights. Each `.ein` has a **Julia equivalent** in comments. See [Julia demos → Einlang](../docs/JULIA_DEMOS.md).

| Directory | What it does | Run |
|-----------|--------------|-----|
| [`ode/`](ode/) | ODEs (decay, linear, Lorenz, Lotka–Volterra, pendulum, van der Pol, SIR, harmonic, fitzhugh_nagumo, lorenz96) | `python3 -m einlang examples/ode/decay.ein`; `sir.ein`; `fitzhugh_nagumo.ein`; `lorenz96.ein`; … |
| [`pde_1d/`](pde_1d/) | 1D heat, advection; [run_heat_1d.py](pde_1d/run_heat_1d.py) for HTML | `python3 -m einlang examples/pde_1d/heat_1d.ein`; `python3 examples/pde_1d/run_heat_1d.py` |
| [`wave_2d/`](wave_2d/) | 2D acoustic wave | `python3 -m einlang examples/wave_2d/main.ein`; `python3 examples/wave_2d/run_wave.py` |
| [`brusselator/`](brusselator/) | Reaction–diffusion (SciML Brusselator) | `python3 -m einlang examples/brusselator/main.ein`; `python3 examples/brusselator/run_brusselator.py` |

### Discrete dynamics & recurrence

Recurrences (base case + step); Markov chains, chaos.

| Directory | What it does | Run |
|-----------|--------------|-----|
| [`recurrence/`](recurrence/) | fibonacci, random_walk, markov_stationary, logistic map | `python3 -m einlang examples/recurrence/fibonacci.ein`; `markov_stationary.ein`; … |

### Finance

| Directory | What it does | Run |
|-----------|--------------|-----|
| [`finance/`](finance/) | Savings / compound interest projection | `python3 -m einlang examples/finance/savings.ein` |

### Economics & optimization

| Directory | What it does | Run |
|-----------|--------------|-----|
| [`value_iteration/`](value_iteration/) | Bellman value iteration (QuantEcon.jl-style) | `python3 -m einlang examples/value_iteration/main.ein` |
| [`job_search/`](job_search/) | McCall job search (QuantEcon): value function iteration, reservation wage | `python3 -m einlang examples/job_search/mccall.ein` |
| [`optimization/`](optimization/) | Gradient descent, power iteration, projected gradient, Rosenbrock (Optim.jl/SciML) | `python3 -m einlang examples/optimization/gradient_descent.ein`; `rosenbrock.ein`; … |
| **[`numerics/`](numerics/)** | **Reusable numerics:** DiffEq (Euler decay), Optim (gradient descent 2D), QuantEcon (value iteration) | `python3 -m einlang examples/run_numerics.ein`; see [numerics/README.md](numerics/README.md) |

### Time series

| Directory | What it does | Run |
|-----------|--------------|-----|
| [`time_series/`](time_series/) | Exponential smoothing (forecasting; StateSpaceModels/TimeSeries-style) | `python3 -m einlang examples/time_series/exponential_smoothing.ein` |

### Real applications (calibration, scenarios)

Multi-step workflows that mirror production use: fit model to data, run one model over many scenarios. See [Learning from Julia: real applications](../docs/LEARNING_FROM_JULIA.md).

| Directory | What it does | Run |
|-----------|--------------|-----|
| [`calibration/`](calibration/) | Fit decay parameter to synthetic data (grid search over k; SSE loss) | `python3 -m einlang examples/calibration/decay_fit.ein` |
| [`applications/`](applications/) | Savings over multiple interest-rate scenarios (sensitivity / scenario analysis) | `python3 -m einlang examples/applications/savings_scenarios.ein` |

### Computer vision

| Directory | What it does | Run |
|-----------|--------------|-----|
| [`mnist/`](mnist/) | CNN digit recognition | `python3 -m einlang examples/mnist/main.ein` |
| [`mnist_quantized/`](mnist_quantized/) | Same CNN, int8 quantization | `python3 -m einlang examples/mnist_quantized/main.ein` |
| [`deit_tiny/`](deit_tiny/) | Vision Transformer (ImageNet) | `python3 -m einlang examples/deit_tiny/main.ein` |

### Speech & sequence

| Directory | What it does | Run |
|-----------|--------------|-----|
| [`whisper_tiny/`](whisper_tiny/) | Speech-to-text (encoder–decoder, BPE) | `python3 -m einlang examples/whisper_tiny/main.ein` (run `download_weights.py` first) |

### Language & basics

Learn the language: variables, functions, Einstein notation, tensors.

| Directory | What you'll learn | Run |
|-----------|-------------------|-----|
| [`basics/`](basics/) | Variables, arithmetic, functions, data processing | `python3 -m einlang examples/basics/variables_demo.ein` |
| [`demos/`](demos/) | Matrices, tensors, imports, overloading | `python3 -m einlang examples/demos/matrix_operations.ein` |
| [`units/`](units/) | 60+ unit-style examples (every feature) | Lookup by feature; see [units/README.md](units/README.md) |

---

## Learning path

If you prefer a linear path:

| Step | Domain | Run |
|------|--------|-----|
| 1 | Language | [basics/](basics/), [demos/](demos/) |
| 2 | Simulation (one ODE) | `python3 -m einlang examples/ode/decay.ein` |
| 3 | Computer vision | [mnist/](mnist/) → [mnist_quantized/](mnist_quantized/) → [deit_tiny/](deit_tiny/) |
| 4 | Speech | [whisper_tiny/](whisper_tiny/) |

---

## Running

From repository root:

```bash
# Simulation (no weights)
python3 -m einlang examples/ode/decay.ein
python3 examples/pde_1d/run_heat_1d.py
python3 examples/wave_2d/run_wave.py

# Recurrence, finance, economics, optimization, real applications
python3 -m einlang examples/recurrence/fibonacci.ein
python3 -m einlang examples/finance/savings.ein
python3 -m einlang examples/job_search/mccall.ein
python3 -m einlang examples/optimization/power_iteration.ein
python3 -m einlang examples/value_iteration/main.ein
python3 -m einlang examples/run_numerics.ein
python3 -m einlang examples/calibration/decay_fit.ein
python3 -m einlang examples/applications/savings_scenarios.ein

# Language
python3 -m einlang examples/basics/variables_demo.ein
python3 -m einlang examples/demos/matrix_operations.ein

# Models (need weights; see Setup below)
python3 -m einlang examples/mnist/main.ein
python3 -m einlang examples/whisper_tiny/main.ein
```

**Tests (including simulation accuracy):**

```bash
python3 -m pytest tests/examples/
```

Every simulation example is [accuracy-tested](tests/examples/test_simulation_accuracy.py) against NumPy or analytical references and has a **Julia equivalent** in the `.ein` file. See [Julia demos → Einlang](../docs/JULIA_DEMOS.md).

---

## Setup (weights and samples)

- **whisper_tiny:** Weights in repo (≤10 MiB); after clone run `git lfs pull`. Samples via `download_weights.py`.
- **mnist, mnist_quantized, deit_tiny:** Weights/samples not in repo; see each example’s README (e.g. `download_weights.py`, `prepare_weights.py`).

---

**Reference:** [Language reference](../docs/reference.md) · [Standard library](../docs/stdlib.md). **Doc index:** [docs/README.md](../docs/README.md).
