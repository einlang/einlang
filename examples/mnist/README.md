---
layout: default
title: MNIST
---

# 3 — MNIST Handwritten Digit Recognition

> **Previous**: [`demos/`](../demos/) · **Next**: [`mnist_quantized/`](../mnist_quantized/)

Your first real neural network in Einlang — a convolutional neural network that classifies 28x28 handwritten digits.

## Architecture

```
Input (1x28x28) → Conv(5x5, pad=2, 8 filters) → ReLU → MaxPool(2x2)
               → Conv(5x5, pad=2, 16 filters) → ReLU → MaxPool(3x3)
               → Flatten(256) → FC(256→10) → argmax
```

Achieves 10/10 on the bundled PGM samples.

## What's new here

- **`use std::ml::{conv, relu, max_pool}`** — stdlib ops for convolution, activation, and pooling. In [demos/](../demos/) you wrote convolutions by hand; here you use the library.
- **Python interop** — `python::numpy::load(path)` and `python::pgm_io::load(path)` call Python functions from Einlang to load `.npy` weight files and PGM images.
- **Shape casts** — `load_npy("weights/conv1_w.npy") as [f32; 8, 1, 5, 5]` tells Einlang the exact tensor shape so that downstream operations type-check.
- **Index arithmetic for flattening** — `let flat[k in 0..256] = pool2[0, k/16, (k/4)%4, k%4]` reshapes a 4D tensor into a 1D vector using integer math on indices.

## Files

| File | Description |
|------|-------------|
| `main.ein` | Model definition and inference loop |
| `pgm_io.py` | PGM image loader (called via `python::pgm_io::load`) |
| `samples/*.pgm` | 28x28 grayscale images of digits 0-9 |
| `weights/` | Pre-trained NumPy weight files (conv1, conv2, fc) |

## Usage

```bash
python3 -m einlang examples/mnist/main.ein
```

To run on a single digit:

```bash
EINLANG_MNIST_INPUT=samples/7.pgm python3 -m einlang examples/mnist/main.ein
```

## How it works

Weights are loaded from `.npy` files and sample images from PGM files using Python interop. The `infer` function runs a forward pass through two convolutional layers with ReLU and max-pooling, then a fully-connected layer, returning the predicted digit via `argmax`. The demo classifies all ten bundled samples and asserts the predictions match.

Once you're comfortable with this, [mnist_quantized/](../mnist_quantized/) takes the exact same network and shows how to run it with int8 weights.
