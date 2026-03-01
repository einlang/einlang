# MNIST Handwritten Digit Recognition

A convolutional neural network that classifies handwritten digits, implemented in Einlang.

## Architecture

```
Input (1x28x28) → Conv(5x5, pad=2, 8 filters) → ReLU → MaxPool(2x2)
               → Conv(5x5, pad=2, 16 filters) → ReLU → MaxPool(3x3)
               → Flatten(256) → FC(256→10) → argmax
```

Achieves 10/10 on the bundled PGM samples.

## Files

| File | Description |
|------|-------------|
| `main.ein` | Model definition and inference loop |
| `pgm_io.py` | PGM image loader (called from Einlang via `python::pgm_io::load`) |
| `samples/*.pgm` | 28x28 grayscale images of digits 0–9 |
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

Weights are loaded from `.npy` files and sample images from PGM files using Python interop (`python::numpy::load`, `python::pgm_io::load`). The `infer` function runs a forward pass through two convolutional layers with ReLU and max-pooling, then a fully-connected layer, returning the predicted digit via `argmax`. The demo classifies all ten bundled samples and asserts the predictions match.
