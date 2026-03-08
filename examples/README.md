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

### Simulations (PDEs, no weights)

| Directory | What it does | Key concepts |
|-----------|----------------|--------------|
| [`wave_2d/`](wave_2d/) | 2D acoustic wave — drum-like propagation | Two-level recurrence (h[t-1], h[t-2]), vectorized stencil, `exp` |
| (heat) | 2D heat diffusion — see [heat_animation.py](heat_animation.py) | One-level recurrence, circular initial condition, HTML animation |

### Reference

| Directory | Description |
|-----------|-------------|
| [`units/`](units/) | 60+ unit tests covering every language feature — useful as a lookup table |

**Reference:** [Language reference](../docs/reference.md) · [Standard library](../docs/stdlib.md). **Doc index (by audience):** [docs/README.md](../docs/README.md).

## Running

All examples run from the repository root:

```bash
# Part 1
python3 -m einlang examples/basics/variables_demo.ein
python3 -m einlang examples/demos/matrix_operations.ein

# Simulations (no weights; output HTML)
python3 examples/heat_animation.py              # heat diffusion → heat.html
python3 examples/wave_2d/run_wave.py            # 2D wave equation → wave.html

# Part 2
python3 -m einlang examples/mnist/main.ein
python3 -m einlang examples/mnist_quantized/main.ein
python3 -m einlang examples/deit_tiny/main.ein
python3 -m einlang examples/whisper_tiny/main.ein  # run download_weights.py first
```

Unit tests:

```bash
python3 -m pytest tests/examples/
```
