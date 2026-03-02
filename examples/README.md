# Einlang Examples

These examples are organized as a guided path — start at the top and work your way down. Each section builds on concepts introduced in the previous one.

## Learning Path

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

### Reference

| Directory | Description |
|-----------|-------------|
| [`units/`](units/) | 60+ unit tests covering every language feature — useful as a lookup table |

## Running

All examples run from the repository root:

```bash
# Part 1
python3 -m einlang examples/basics/variables_demo.ein
python3 -m einlang examples/demos/matrix_operations.ein

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
