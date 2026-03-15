# Einlang Examples

**First time?** [Try it in 30 seconds](https://github.com/einlang/einlang/blob/main/README.md#try-it) or read [Getting started](https://github.com/einlang/einlang/blob/main/docs/GETTING_STARTED.md). **Want one capability?** [Examples by feature](https://github.com/einlang/einlang/blob/main/README.md#examples).

Examples are grouped **by domain** below. For a step-by-step path (basics → demos → MNIST → ViT → Whisper), see [Learning path](#learning-path).

---

## By domain

### Scientific simulation (ODEs & PDEs)

Time-stepping and spatial stencils; no weights. Each `.ein` has a **Julia equivalent** in comments. See [Julia demos → Einlang](https://github.com/einlang/einlang/blob/main/docs/JULIA_DEMOS.md).

| Directory | What it does | Run |
|-----------|--------------|-----|
| [`ode/`](https://github.com/einlang/einlang/tree/main/examples/ode) | ODEs: suite (decay, linear, harmonic, pendulum, van der Pol, Lotka–Volterra) + lorenz, sir, fitzhugh_nagumo, lorenz96 | `python3 -m einlang examples/ode/ode_suite.ein`; `lorenz.ein`; `sir.ein`; … |
| [`pde_1d/`](https://github.com/einlang/einlang/tree/main/examples/pde_1d) | 1D heat, advection; [run_heat_1d.py](https://github.com/einlang/einlang/blob/main/examples/pde_1d/run_heat_1d.py) for HTML | `python3 -m einlang examples/pde_1d/heat_1d.ein`; `python3 examples/pde_1d/run_heat_1d.py` |
| [`wave_2d/`](https://github.com/einlang/einlang/tree/main/examples/wave_2d) | 2D acoustic wave | `python3 -m einlang examples/wave_2d/main.ein`; `python3 examples/wave_2d/run_wave.py` |
| [`brusselator/`](https://github.com/einlang/einlang/tree/main/examples/brusselator) | Reaction–diffusion (SciML Brusselator) | `python3 -m einlang examples/brusselator/main.ein`; `python3 examples/brusselator/run_brusselator.py` |

### Discrete dynamics & recurrence

Recurrences (base case + step); Markov chains, chaos.

| Directory | What it does | Run |
|-----------|--------------|-----|
| [`recurrence/`](https://github.com/einlang/einlang/tree/main/examples/recurrence) | Suite: fibonacci, logistic, markov_stationary, random_walk | `python3 -m einlang examples/recurrence/recurrence_suite.ein` |

### Finance

| Directory | What it does | Run |
|-----------|--------------|-----|
| [`finance/`](https://github.com/einlang/einlang/tree/main/examples/finance) | Savings / compound interest projection | `python3 -m einlang examples/finance/savings.ein` |

### Economics & optimization

| Directory | What it does | Run |
|-----------|--------------|-----|
| **Autodiff** (root) | Compiler autodiff: `@expr`, `@a / @b` (derivatives); scalars, matmul, user fn, loss gradient; chain-through-bindings example (may show 0 until supported) | `python3 examples/run_autodiff_examples.py` or run `autodiff_small.ein`, `autodiff_matmul.ein`, `autodiff_chain.ein`, `autodiff_user_fn.ein`, `autodiff_loss.ein` |
| [`value_iteration/`](https://github.com/einlang/einlang/tree/main/examples/value_iteration) | Value iteration (Bellman); **policy iteration** (Howard: evaluate + improve) | `python3 -m einlang examples/value_iteration/main.ein` · `policy_iteration.ein` |
| [`job_search/`](https://github.com/einlang/einlang/tree/main/examples/job_search) | McCall job search (QuantEcon): value function iteration, reservation wage | `python3 -m einlang examples/job_search/mccall.ein` |
| [`optimization/`](https://github.com/einlang/einlang/tree/main/examples/optimization) | Suite: gradient descent, power iteration, projected gradient, Rosenbrock (Optim.jl/SciML) | `python3 -m einlang examples/optimization/optimization_suite.ein` |
| **[`numerics/`](https://github.com/einlang/einlang/tree/main/examples/numerics)** | **Reusable numerics:** DiffEq (Euler decay), Optim (gradient descent 2D), QuantEcon (value iteration) | `python3 -m einlang examples/run_numerics.ein`; see [numerics/README](https://github.com/einlang/einlang/blob/main/examples/numerics/README.md) |

### Time series

| Directory | What it does | Run |
|-----------|--------------|-----|
| [`time_series/`](https://github.com/einlang/einlang/tree/main/examples/time_series) | Exponential smoothing (forecasting; StateSpaceModels/TimeSeries-style) | `python3 -m einlang examples/time_series/exponential_smoothing.ein` |

### Parameter estimation & scenario workflows

Multi-step patterns: fit model to data then use it, or run one model over many parameter sets (sensitivity/scenario analysis). See [Learning from Julia: real applications](https://github.com/einlang/einlang/blob/main/docs/LEARNING_FROM_JULIA.md).

| Directory | What it does | Run |
|-----------|--------------|-----|
| [`applications/`](https://github.com/einlang/einlang/tree/main/examples/applications) | **Calibrate then forecast** (decay_calibration); **steady-state risk** (markov_credit); **Kalman filter** (kalman_filter: track position/velocity from noisy measurements, migrated from 223-line NumPy app). | `python3 -m einlang examples/applications/decay_calibration.ein` · `markov_credit.ein` · `examples/applications/kalman_filter/main.ein` |

### Computer vision

| Directory | What it does | Run |
|-----------|--------------|-----|
| [`mnist/`](https://github.com/einlang/einlang/tree/main/examples/mnist) | CNN digit recognition | `python3 -m einlang examples/mnist/main.ein` |
| [`mnist_quantized/`](https://github.com/einlang/einlang/tree/main/examples/mnist_quantized) | Same CNN, int8 quantization | `python3 -m einlang examples/mnist_quantized/main.ein` |
| [`deit_tiny/`](https://github.com/einlang/einlang/tree/main/examples/deit_tiny) | Vision Transformer (ImageNet) | `python3 -m einlang examples/deit_tiny/main.ein` |

### Speech & sequence

| Directory | What it does | Run |
|-----------|--------------|-----|
| [`whisper_tiny/`](https://github.com/einlang/einlang/tree/main/examples/whisper_tiny) | Speech-to-text (encoder–decoder, BPE) | `python3 -m einlang examples/whisper_tiny/main.ein` (run `download_weights.py` first) |

### Language & basics

Learn the language: variables, functions, Einstein notation, tensors.

| Directory | What you'll learn | Run |
|-----------|-------------------|-----|
| [`basics/`](https://github.com/einlang/einlang/tree/main/examples/basics) | Variables, arithmetic, functions, data processing | `python3 -m einlang examples/basics/variables_demo.ein` |
| [`demos/`](https://github.com/einlang/einlang/tree/main/examples/demos) | Matrices, tensors, imports, overloading | `python3 -m einlang examples/demos/matrix_operations.ein` |
| [`units/`](https://github.com/einlang/einlang/tree/main/examples/units) | 60+ unit-style examples (every feature) | Lookup by feature; see [units/README](https://github.com/einlang/einlang/blob/main/examples/units/README.md) |

---

## Learning path

If you prefer a linear path:

| Step | Domain | Run |
|------|--------|-----|
| 1 | Language | [basics/](https://github.com/einlang/einlang/tree/main/examples/basics), [demos/](https://github.com/einlang/einlang/tree/main/examples/demos) |
| 2 | Simulation (ODE suite) | `python3 -m einlang examples/ode/ode_suite.ein` |
| 3 | Computer vision | [mnist/](https://github.com/einlang/einlang/tree/main/examples/mnist) → [mnist_quantized/](https://github.com/einlang/einlang/tree/main/examples/mnist_quantized) → [deit_tiny/](https://github.com/einlang/einlang/tree/main/examples/deit_tiny) |
| 4 | Speech | [whisper_tiny/](https://github.com/einlang/einlang/tree/main/examples/whisper_tiny) |

---

## Running

From repository root:

```bash
# Simulation (no weights)
python3 -m einlang examples/ode/ode_suite.ein
python3 examples/pde_1d/run_heat_1d.py
python3 examples/wave_2d/run_wave.py

# Recurrence, finance, economics, optimization, workflows
python3 -m einlang examples/recurrence/recurrence_suite.ein
python3 -m einlang examples/finance/savings.ein
python3 -m einlang examples/job_search/mccall.ein
python3 -m einlang examples/optimization/optimization_suite.ein
python3 -m einlang examples/value_iteration/main.ein
python3 -m einlang examples/run_numerics.ein
python3 -m einlang examples/applications/decay_calibration.ein
python3 -m einlang examples/applications/markov_credit.ein
python3 -m einlang examples/applications/kalman_filter/main.ein

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

Every simulation example is [accuracy-tested](https://github.com/einlang/einlang/blob/main/tests/examples/test_simulation_accuracy.py) against NumPy or analytical references and has a **Julia equivalent** in the `.ein` file. See [Julia demos → Einlang](https://github.com/einlang/einlang/blob/main/docs/JULIA_DEMOS.md).

---

## Setup (weights and samples)

- **whisper_tiny:** Weights in repo (≤10 MiB); after clone run `git lfs pull`. Samples via `download_weights.py`.
- **mnist, mnist_quantized, deit_tiny:** Weights/samples not in repo; see each example’s README (e.g. `download_weights.py`, `prepare_weights.py`).

---

**Reference:** [Language reference](https://github.com/einlang/einlang/blob/main/docs/reference.md) · [Standard library](https://github.com/einlang/einlang/blob/main/docs/stdlib.md). **Doc index:** [docs/README](https://github.com/einlang/einlang/blob/main/docs/README.md).
