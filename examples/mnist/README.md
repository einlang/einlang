
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
- **Sklearn-backed training samples** — `train_sklearn_digits.ein` and `train_sklearn_digits_mlp.ein` pull the `load_digits` split through `python::mnist_data::*`, while keeping the actual training step in Einlang.

## Files

| File | Description |
|------|-------------|
| `main.ein` | Load CNN weights, bootstrap float weights if missing, and run inference |
| `data_loader.py` | MNIST data/weight bootstrap helpers (called via `python::data_loader::*`) |
| `mnist_data.py` | scikit-learn digits loader used by the sklearn-backed training sample |
| `train_sklearn_digits.ein` | Full-split multi-epoch linear training example on sklearn digits, with autodiff updates in pure Einlang |
| `train_sklearn_digits_mlp.ein` | One-step 2-layer MLP training example on sklearn digits, with autodiff in pure Einlang |
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

The sklearn-backed training samples cover two model shapes:

```text
train_sklearn_digits.ein
full 1437/360 sklearn split
8x8 image -> flatten(64) -> Dense(10) -> argmax
10 autodiff updates

train_sklearn_digits_mlp.ein
8x8 image -> flatten(64) -> Dense(12) -> ReLU -> Dense(10) -> argmax
```

`train_sklearn_digits.ein` is the better example if you want to see accuracy build over several epochs on separate train/test batches. `train_sklearn_digits_mlp.ein` stays intentionally smaller and shows the hidden-layer version with a single autodiff update.

Both need `scikit-learn` available in the Python environment because `mnist_data.py` uses `sklearn.datasets.load_digits`.
