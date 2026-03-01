# DeiT-Tiny ImageNet Classification

A Vision Transformer (DeiT-Tiny) that classifies ImageNet images, implemented entirely in Einlang using `stdlib` operations and Einstein notation.

## Architecture

```
Input (1x3x224x224) → PatchEmbed(16x16 conv, 192-d) → CLS token + Position embed
    → 12x Transformer Block:
        LayerNorm → QKV(192→576) → 3-head Attention(64-d) → Proj(192→192) + Residual
        LayerNorm → MLP(192→768, GELU, 768→192) + Residual
    → LayerNorm → CLS head(192→1000) → argmax
```

embed_dim=192, heads=3, mlp_ratio=4, seq_len=197 (196 patches + CLS token).

## Files

| File | Description |
|------|-------------|
| `main.ein` | Full model definition, ImageNet labels, and inference loop |
| `samples/*.npy` | Three preprocessed 224x224 ImageNet images |
| `weights/` | Pre-trained DeiT-Tiny weight files (20 `.npy` files) |

## Usage

```bash
python3 -m einlang examples/deit_tiny/main.ein
```

Expected output:

```
['Egyptian Mau', 'Golden Retriever', 'strawberry']
```

Runs in ~15–30 seconds on CPU depending on hardware.

## How it works

Weights are loaded from `.npy` files via Python interop (`python::numpy::load`). The `infer` function implements the full DeiT-Tiny forward pass using Einlang's Einstein notation for all tensor operations — matrix multiplications, multi-head attention with softmax, GELU activation, and layer normalization from `stdlib`. The demo classifies three bundled ImageNet samples and asserts the predicted class names match expected labels.
