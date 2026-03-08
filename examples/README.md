# Einlang Examples

**First time?** [Try it in 30 seconds](../README.md#try-it) or read [Getting started](../docs/GETTING_STARTED.md). **Want one capability?** [Examples by feature](../README.md#examples).

This page is the full **learning path**: basics → demos → MNIST → quantized CNN → ViT → Whisper. Each step builds on the last.

## Learning path

### Part 1: Learn the Language

| # | Directory | What you'll learn | Key concepts |
|---|-----------|-------------------|--------------|
| 1 | [`basics/`](basics/) | Variables, arithmetic, functions, data processing | `let`, `fn`, `assert`, `sum[i]`, comprehensions |
| 2 | [`demos/`](demos/) | Matrices, tensors, imports, overloading, CLI | Einstein notation, `use`, `mod`, NCHW tensors, `in` operator |

### Part 2: Real Models

| # | Directory | What you'll learn | Key concepts |
|---|-----------|-------------------|--------------|
| 3 | [`mnist/`](mnist/) | First neural network — CNN digit recognition | `conv`, `relu`, `max_pool`, Python interop, weight loading |
| 4 | [`mnist_quantized/`](mnist_quantized/) | Int8 weight quantization on the same CNN | `dequantize_linear`, symmetric quantization, per-tensor scale |
| 5 | [`deit_tiny/`](deit_tiny/) | Vision Transformer — ImageNet classification | Multi-head attention, softmax, GELU, LayerNorm, 12-block transformer |
| 6 | [`whisper_tiny/`](whisper_tiny/) | Speech-to-text — encoder/decoder with autoregressive loop | 1D conv, cross-attention, causal mask, recurrence, BPE decoding |

### Simulations (ODEs, PDEs; no weights)

| Directory | What it does | Key concepts |
|-----------|----------------|--------------|
| [`ode/`](ode/) | ODEs aligned with Julia DiffEq: decay, linear, Lorenz, Lotka–Volterra | Recurrence over time; scalar/vector state; one folder, four `.ein` files |
| [`wave_2d/`](wave_2d/) | 2D acoustic wave — drum-like propagation | Two-level recurrence (h[t-1], h[t-2]), vectorized stencil, `exp` |
| [`brusselator/`](brusselator/) | Reaction–diffusion (SciML Brusselator) | 4D state, recurrence, Laplacian + reaction terms |
| (heat) | 2D heat diffusion — see [heat_animation.py](heat_animation.py) | One-level recurrence, circular initial condition, HTML animation |

### Julia migration (one folder per Julia source)

Each folder maps to one Julia demo or ecosystem. See [Julia demos → Einlang](../docs/JULIA_DEMOS.md).

| Directory | Julia source | Run |
|-----------|--------------|-----|
| [`ode/`](ode/) | DifferentialEquations.jl (decay, linear, Lorenz, Lotka–Volterra) | `python3 -m einlang examples/ode/decay.ein`; `linear.ein`; `lorenz.ein`; `lotka_volterra.ein` |
| [`brusselator/`](brusselator/) | SciML Brusselator | `python3 -m einlang examples/brusselator/main.ein`; `python3 examples/brusselator/run_brusselator.py` |
| [`pde_1d/`](pde_1d/) | MethodOfLines.jl 1D PDE (heat, advection) | `python3 -m einlang examples/pde_1d/heat_1d.ein`; `python3 examples/pde_1d/run_heat_1d.py` |
| [`value_iteration/`](value_iteration/) | QuantEcon.jl Bellman value iteration | `python3 -m einlang examples/value_iteration/main.ein` |

### More small examples (grouped)

| Directory | Contents | Run |
|-----------|----------|-----|
| [`recurrence/`](recurrence/) | fibonacci.ein, random_walk.ein | `python3 -m einlang examples/recurrence/fibonacci.ein` |
| [`pde_1d/`](pde_1d/) | heat_1d.ein, advection_1d.ein (+ run_heat_1d.py) | `python3 -m einlang examples/pde_1d/heat_1d.ein` or `advection_1d.ein` |
| [`tensor_ops/`](tensor_ops/) | softmax.ein | `python3 -m einlang examples/tensor_ops/softmax.ein` |

### Reference

| Directory | Description |
|-----------|-------------|
| [`units/`](units/) | 60+ unit tests covering every language feature — useful as a lookup table |

**Reference:** [Language reference](../docs/reference.md) · [Standard library](../docs/stdlib.md). **Doc index (by audience):** [docs/README.md](../docs/README.md).

## Real-world use cases

In the spirit of [Julia’s demos](https://julialang.org/); we map their use cases to ours in [Julia demos → Einlang](../docs/JULIA_DEMOS.md). One language for simulation and ML, with compile-time shape checking and Einstein notation.

| Domain | What you do | Example |
|--------|-------------|---------|
| **Scientific simulation** | ODE time-stepping (Julia DiffEq-style); PDEs: heat, acoustic wave, reaction–diffusion (Brusselator). Recurrence over time; vectorized stencils in space. | [ode/](ode/), [heat_animation.py](heat_animation.py), [wave_2d/](wave_2d/), [brusselator/](brusselator/) |
| **Computer vision** | CNNs (MNIST), int8-quantized inference, Vision Transformer (ImageNet). Convs, attention, LayerNorm, GELU. | [mnist/](mnist/), [mnist_quantized/](mnist_quantized/), [deit_tiny/](deit_tiny/) |
| **Speech & sequence** | Speech-to-text: encoder–decoder, cross-attention, causal self-attention, autoregressive decoding, BPE. | [whisper_tiny/](whisper_tiny/) |

Same language and shape checks from small scripts to full models; no stringly-typed einsum.

## Setup (weights and samples)

**whisper_tiny** weights are in the repo (files ≤10 MiB committed; >10 MiB on Git LFS). After clone run `git lfs pull` to fetch the large file. Samples (e.g. `samples/jfk.npy`) are obtained via `download_weights.py`.

Weights and samples for **mnist**, **mnist_quantized**, and **deit_tiny** are not in the repo. Obtain them via each example's setup (see the example's README: e.g. `download_weights.py` for deit_tiny, `prepare_weights.py` for mnist_quantized; mnist may require pre-trained weight files in `weights/` and PGM images in `samples/`).

## Running

All examples run from the repository root:

```bash
# Part 1
python3 -m einlang examples/basics/variables_demo.ein
python3 -m einlang examples/demos/matrix_operations.ein

# Simulations (ODE + PDEs; no weights)
python3 -m einlang examples/ode/decay.ein       # ODE: exponential decay
python3 examples/heat_animation.py            # heat diffusion → heat.html
python3 examples/wave_2d/run_wave.py           # 2D wave equation → wave.html

# Part 2
python3 -m einlang examples/mnist/main.ein
python3 -m einlang examples/mnist_quantized/main.ein
python3 -m einlang examples/deit_tiny/main.ein
python3 -m einlang examples/whisper_tiny/main.ein  # run download_weights.py first
```

Unit tests (including **simulation accuracy** for every simulation example):

```bash
python3 -m pytest tests/examples/
```

**Accuracy:** Every simulation example is compared against Julia (each `.ein` has a **Julia equivalent** in comments, aligned with DifferentialEquations.jl, QuantEcon.jl, MethodOfLines.jl, or SciML) and is accuracy-tested against NumPy or analytical references. Covered: ODE (decay, linear, Lorenz, Lotka–Volterra), wave_2d, pde_1d (heat_1d, advection_1d), Brusselator, value_iteration, recurrence (fibonacci, random_walk), tensor_ops (softmax). See [tests/examples/test_simulation_accuracy.py](../tests/examples/test_simulation_accuracy.py) and [Julia demos → Einlang](../docs/JULIA_DEMOS.md).
