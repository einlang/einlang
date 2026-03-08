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
| [`ode/`](ode/) | ODE: exponential decay (Julia DiffEq-style time-stepping) | Recurrence over time, scalar state |
| [`wave_2d/`](wave_2d/) | 2D acoustic wave — drum-like propagation | Two-level recurrence (h[t-1], h[t-2]), vectorized stencil, `exp` |
| [`reaction_diffusion/`](reaction_diffusion/) | Gray–Scott reaction–diffusion (spots/stripes) | 4D state, recurrence, channel conditional |
| (heat) | 2D heat diffusion — see [heat_animation.py](heat_animation.py) | One-level recurrence, circular initial condition, HTML animation |

### Julia migration (one folder per set)

Each set maps to a well-known Julia demo; one folder per set. See [Julia demos → Einlang](../docs/JULIA_DEMOS.md).

| Directory | Julia source | Run |
|-----------|--------------|-----|
| [`brusselator/`](brusselator/) | SciML Brusselator | `python3 -m einlang examples/brusselator/main.ein`; `python3 examples/brusselator/run_brusselator.py` |
| [`lorenz/`](lorenz/) | DiffEq Lorenz system | `python3 -m einlang examples/lorenz/main.ein` |
| [`lotka_volterra/`](lotka_volterra/) | DiffEq / QuantEcon predator–prey | `python3 -m einlang examples/lotka_volterra/main.ein` |
| [`heat_1d/`](heat_1d/) | MethodOfLines 1D PDE | `python3 -m einlang examples/heat_1d/main.ein`; `python3 examples/heat_1d/run_heat_1d.py` |
| [`linear_ode/`](linear_ode/) | Linear ODE du/dt = A·u | `python3 -m einlang examples/linear_ode/main.ein` |

### Reference

| Directory | Description |
|-----------|-------------|
| [`units/`](units/) | 60+ unit tests covering every language feature — useful as a lookup table |

**Reference:** [Language reference](../docs/reference.md) · [Standard library](../docs/stdlib.md). **Doc index (by audience):** [docs/README.md](../docs/README.md).

## Real-world use cases

In the spirit of [Julia’s demos](https://julialang.org/); we map their use cases to ours in [Julia demos → Einlang](../docs/JULIA_DEMOS.md). One language for simulation and ML, with compile-time shape checking and Einstein notation.

| Domain | What you do | Example |
|--------|-------------|---------|
| **Scientific simulation** | ODE time-stepping (Julia DiffEq-style); PDEs: heat, acoustic wave, reaction–diffusion. Recurrence over time; vectorized stencils in space. | [ode/](ode/), [heat_animation.py](heat_animation.py), [wave_2d/](wave_2d/), [reaction_diffusion/](reaction_diffusion/) |
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
python3 -m einlang examples/ode/main.ein       # ODE: exponential decay
python3 examples/heat_animation.py            # heat diffusion → heat.html
python3 examples/wave_2d/run_wave.py           # 2D wave equation → wave.html

# Part 2
python3 -m einlang examples/mnist/main.ein
python3 -m einlang examples/mnist_quantized/main.ein
python3 -m einlang examples/deit_tiny/main.ein
python3 -m einlang examples/whisper_tiny/main.ein  # run download_weights.py first
```

Unit tests (including **simulation accuracy** for ODE, wave, heat, reaction-diffusion):

```bash
python3 -m pytest tests/examples/
```

Accuracy checks compare ODE to analytical solution; wave/heat/RD to initial conditions and invariants. See [tests/examples/test_simulation_accuracy.py](../tests/examples/test_simulation_accuracy.py).
