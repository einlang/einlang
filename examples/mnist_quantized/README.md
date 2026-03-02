# 4 — Quantized MNIST Digit Recognition

> **Previous**: [`mnist/`](../mnist/) · **Next**: [`deit_tiny/`](../deit_tiny/)

Same CNN as [mnist/](../mnist/), but weights and activations are **int8** and the heavy work (conv and linear) runs as **i8×i8→i32** in the stdlib ops `qconv` and `qlinear`, then rescales to float. This matches the ONNX/PyTorch pattern for real quantization speedup.

## Architecture

```
Input (1x1x28x28) → quantize → qconv(5x5, pad=2, 8ch) → ReLU → MaxPool(2x2)
    → quantize → qconv(5x5, pad=2, 16ch) → ReLU → MaxPool(3x3) → flatten
    → quantize → qlinear(256→10) → argmax
```

Weights are stored as int8; activations are quantized at each layer boundary. Convolution and linear use integer inner loops and a single rescale to float, so compute can be faster than the dequantize-then-float path.

## What's new here

- **`qconv(X_q, x_scale, W_q, w_scale, B, strides, pads, dilations, group)`** — 2D conv with i8×i8→i32 accumulation, then `acc * (x_scale * w_scale) + B`. No full dequant of weights.
- **`qlinear(x_q, x_scale, W_q, w_scale, bias)`** — Linear layer with the same integer pattern; weights layout `[in, out]`.
- **`quantize_linear(x, scale, zero_point)`** — Quantize activations to i8 range for the next layer.
- **Activation scales** — `prepare_weights.py` runs a float forward on the sample images and writes `act1_s.npy` and `flat_s.npy` so requantization between layers uses calibrated scales.

## Files

| File | Description |
|------|-------------|
| `main.ein` | Model using `qconv`, `qlinear`, and `quantize_linear` |
| `prepare_weights.py` | Quantizes weights to int8 and calibrates activation scales |
| `weights/*_q.npy` | Int8 quantized weight tensors |
| `weights/*_s.npy` | Per-tensor float32 scale factors (weights + act1, flat) |
| `weights/*_b.npy` | Float32 biases |
| `samples/` | Symlink to `mnist/samples/` (10 PGM digit images) |

## Usage

```bash
python3 -m einlang examples/mnist_quantized/main.ein
```

Expected output:

```
[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
```

## Regenerating quantized weights and scales

```bash
cd examples/mnist_quantized
python3 prepare_weights.py
```

Requires `../mnist/weights/` and (for calibration) `samples/` and `pgm_io`. If calibration is skipped, default activation scales are used.

## How it works

Weights are quantized with symmetric per-tensor int8 (`scale = max|w|/127`, zero_point 0). Activation scales for the first conv input and for the layer outputs (after pool) are computed by running the float model on the sample images. At inference, `main.ein` quantizes the input, runs `qconv`/`qlinear` with i8×i8→i32 and a single f32 rescale, requantizes activations between layers with `quantize_linear`, and gets logits in float for `argmax`.

Next: [deit_tiny/](../deit_tiny/) implements a Vision Transformer with multi-head attention and 12 transformer blocks.
