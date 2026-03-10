
# 5 — DeiT-Tiny ImageNet Classification

> **Previous**: [`mnist_quantized/`](https://github.com/einlang/einlang/tree/main/examples/mnist_quantized) · **Next**: [`whisper_tiny/`](https://github.com/einlang/einlang/tree/main/examples/whisper_tiny)

A Vision Transformer (DeiT-Tiny) that classifies 224x224 ImageNet images into 1000 classes. This is a significant step up from MNIST — the model has ~5M parameters and showcases Einstein notation at scale.

## Architecture

```
Input (1x3x224x224) → PatchEmbed(16x16 conv, 192-d) → CLS token + Position embed
    → 12x Transformer Block:
        LayerNorm → QKV(192→576) → 3-head Attention(64-d) → Proj(192→192) + Residual
        LayerNorm → MLP(192→768, GELU, 768→192) + Residual
    → LayerNorm → CLS head(192→1000) → argmax
```

embed_dim=192, heads=3, mlp_ratio=4, seq_len=197 (196 patches + CLS token).

## What's new here

Compared to the CNN examples, this introduces the transformer architecture:

- **Multi-head self-attention** — queries, keys, and values are computed via Einstein notation (`sum[d in 0..192](ln1[0, s, d] * blk_qkv_w[L, d, k])`), split into heads, and softmax is implemented manually using `exp`, `max`, and `sum` reductions.
- **`layer_normalization`** and **`gelu_tanh`** — stdlib ops for LayerNorm and GELU activation, used inside each transformer block.
- **Residual connections** — `let res1[...] = x[...] + proj[...]` implements skip connections that are critical for training deep networks.
- **Parameterized blocks** — `transformer_block(x, L)` takes a block index `L` to index into stacked weight arrays (`blk_qkv_w[L, ...]`), running the same code for all 12 blocks.
- **Patch embedding** — the 224x224 image is split into 14x14 = 196 patches of 16x16 pixels using a single conv, then a CLS token is prepended.

## Files

| File | Description |
|------|-------------|
| `main.ein` | Full model definition, 1000 ImageNet labels, and inference loop |
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

Runs in ~15-30 seconds on CPU depending on hardware.

**Profile** (per-clause time and vectorized/hybrid/scalar path): run from repo root with env set so paths resolve:

```bash
cd examples/deit_tiny && EINLANG_PROFILE_STATEMENTS=1 EINLANG_DEBUG_VECTORIZE=1 python3 -m einlang main.ein
```

(Use `PYTHONPATH=../../src` if not running from repo root.)

## How it works

Weights are loaded from `.npy` files via Python interop. The `infer` function implements the full DeiT-Tiny forward pass: patch embedding via conv, CLS token prepend, positional encoding, 12 transformer blocks (each with multi-head self-attention, layer norm, and MLP with GELU), final normalization, and a linear head that projects the CLS token to 1000 class logits.

The next and final model example, [whisper_tiny/](https://github.com/einlang/einlang/tree/main/examples/whisper_tiny), adds an encoder-decoder architecture with cross-attention and autoregressive token generation.
