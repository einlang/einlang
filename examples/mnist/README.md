
# 3 — MNIST Handwritten Digit Recognition

> **Previous**: [`demos/`](https://github.com/einlang/einlang/tree/main/examples/demos) · **Next**: [`mnist_quantized/`](https://github.com/einlang/einlang/tree/main/examples/mnist_quantized)

An ONNX-style CNN in Einlang for classifying 28x28 handwritten digits.

## Architecture

```
Input (1x1x28x28)
→ Conv(8, 5x5) + ReLU + MaxPool(2x2)
→ Conv(16, 5x5) + ReLU + MaxPool(3x3)
→ Flatten(256) → Linear(10) → argmax
```

Achieves 10/10 on the bundled PGM samples.

## What's new here

- **CNN ops** — `conv`, `relu`, `max_pool`, and flattening in pure Einlang.
- **`load_npy`** — `main.ein` loads CNN and classifier weights for inference.
- **Python interop** — `python::data_loader::load_images()` calls Python functions from Einlang to load PGM images and labels.

## Files

| File | Description |
|------|-------------|
| `main.ein` | Load CNN weights, bootstrap float weights if missing, and run inference |
| `data_loader.py` | MNIST data/weight bootstrap helpers (called via `python::data_loader::*`) |
| `pgm_io.py` | PGM image loader (used by quantized example) |
| `samples/*.pgm` | 28x28 grayscale images of digits 0-9 |
| `weights/*.npy` | Float inference weights materialized locally |

## Usage

Inference (requires trained weights):

```bash
python3 -m einlang examples/mnist/main.ein
```

## How it works

**`main.ein`** — Ensures float CNN and classifier weights exist locally, materializing them from `examples/mnist_quantized/weights/` when needed, then computes logits for all 10 sample images and asserts predictions match expected labels.

Once you're comfortable with this, [mnist_quantized/](https://github.com/einlang/einlang/tree/main/examples/mnist_quantized) takes the exact same network and shows how to run it with int8 weights.
