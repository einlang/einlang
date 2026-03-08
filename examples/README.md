# Einlang Examples

**First time?** [Try it in 30 seconds](../README.md#try-it) or read [Getting started](../docs/GETTING_STARTED.md). **Want one capability?** [Examples by feature](../README.md#examples).

Examples are grouped **by domain** below. For a step-by-step path (basics → demos → MNIST → ViT → Whisper), see [Learning path](#learning-path).

---

## By domain

### Scientific simulation (ODEs & PDEs)

Time-stepping and spatial stencils; no weights. Each `.ein` has a **Julia equivalent** in comments. See [Julia demos → Einlang](../docs/JULIA_DEMOS.md).

| Directory | What it does | Run |
|-----------|--------------|-----|
| [`ode/`](ode/) | ODEs (decay, linear, Lorenz, Lotka–Volterra, pendulum, van der Pol, SIR, harmonic) | `python3 -m einlang examples/ode/decay.ein`; `sir.ein`; `harmonic.ein`; … |
| [`pde_1d/`](pde_1d/) | 1D heat, advection; [run_heat_1d.py](pde_1d/run_heat_1d.py) for HTML | `python3 -m einlang examples/pde_1d/heat_1d.ein`; `python3 examples/pde_1d/run_heat_1d.py` |
| [`wave_2d/`](wave_2d/) | 2D acoustic wave | `python3 -m einlang examples/wave_2d/main.ein`; `python3 examples/wave_2d/run_wave.py` |
| [`brusselator/`](brusselator/) | Reaction–diffusion (SciML Brusselator) | `python3 -m einlang examples/brusselator/main.ein`; `python3 examples/brusselator/run_brusselator.py` |

### Discrete dynamics & recurrence

Recurrences (base case + step); Markov chains, chaos, optimization, linear algebra.

| Directory | What it does | Run |
|-----------|--------------|-----|
| [`recurrence/`](recurrence/) | fibonacci, random_walk, markov_stationary, logistic map, gradient_descent, power_iteration | `python3 -m einlang examples/recurrence/fibonacci.ein`; `power_iteration.ein`; … |

### Economics & optimization

| Directory | What it does | Run |
|-----------|--------------|-----|
| [`value_iteration/`](value_iteration/) | Bellman value iteration (QuantEcon.jl-style) | `python3 -m einlang examples/value_iteration/main.ein` |

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

# Recurrence & optimization
python3 -m einlang examples/recurrence/power_iteration.ein
python3 -m einlang examples/value_iteration/main.ein

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
