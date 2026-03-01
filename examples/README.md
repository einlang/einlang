# Einlang Examples

## Demos

| Example | Description | Key Features |
|---------|-------------|--------------|
| [`mnist/`](mnist/) | Handwritten digit recognition (CNN) | Conv2D, MaxPool, ReLU, FC layers, PGM image loading |
| [`deit_tiny/`](deit_tiny/) | ImageNet classification (Vision Transformer) | Einstein notation, multi-head attention, LayerNorm, GELU, 12-block transformer |

## Tutorials

| Directory | Description |
|-----------|-------------|
| [`basics/`](basics/) | Introduction to variables, functions, math, and data processing |
| [`demos/`](demos/) | Intermediate examples covering arrays, comprehensions, overloading, and CLI usage |
| [`units/`](units/) | Unit tests exercising Einstein notation, windowed ops, scans, tensor operations, and more |

## Running

All examples run from the repository root:

```bash
python3 -m einlang examples/mnist/main.ein
python3 -m einlang examples/deit_tiny/main.ein
```

Unit tests:

```bash
python3 -m pytest tests/examples/
```
