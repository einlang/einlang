# Whisper-tiny detailed profile summary

Run with:
```bash
cd /path/to/einlang
PYTHONPATH=src python3 -m einlang examples/whisper_tiny/main.ein \
  --profile-functions --profile-statements --profile-blocks --profile-reductions --profile-lines 15 --debug-vectorize \
  2> whisper_profile.txt
```

## Per-function total time (s)

| Function | Time (s) |
|----------|----------|
| decoder_block | 85.81 |
| encode | 11.39 |
| encoder_block | 11.00 |
| layer_normalization_... | 9.32 |
| gelu_f32 | 1.18 |
| tanh_f32 | 0.57 |
| exp_f32 | 0.47 |
| load_npy | 0.31 |
| resolve_relative_to_script_str | 0.04 |
| sqrt_f32 | 0.04 |
| script_dir | 0.01 |

## Top-level statements (main.ein)

| Stmt | Line | Name | Time (s) |
|------|------|------|----------|
| 54 | L224 | enc_out | 11.39 |
| 56 | L227 | tokens | 92.28 |
| 57 | L246 | text | 0.11 |

- **enc_out** = `encode(mel)` (one encoder pass).
- **tokens** = autoregressive decode loop (t in 4..40): emb + 4×decoder_block + ln + logits + argmax per step.
- **text** = decode tokens to string.

## Einstein clause summary (within tokens block)

- `tokens = blockexpression (40,)`: **44.06s** reported as [scalar] — this is the outer recurrence (36 steps); each step runs the block.
- Inner clauses (emb, decoder_block, layer_norm, logits) are vectorized or matmul; the **emb** clause uses slice-vectorize (if p < t).

## Vectorize counts (full run)

```
[vectorize] Einstein clauses: 13811 vectorized, 4 scalar, 0 hybrid, 0 call-scalar (total 13815)
```

- **4 scalar**: the recurrence steps for `tokens` (or similar outer scalar loops).
- **13811 vectorized**: includes the inner emb (slice-vectorized over p in 0..t) and all encoder/decoder einsum/matmul/vectorized clauses.

## Line buckets (L0–L15)

- During encode: L0–L15 ≈ 10.76s.
- During decode (tokens stmt): L0–L15 ≈ 166.63s (dominated by decoder_block and block body).

## Reductions

- Encoder/decoder: reductions reported as `[reduction] vectorized L86`, `matmul`, `einsum` etc.
- Logits: `logits = sum(final_ln MUL dec_tok) (51865,): 0.068s [einsum]` per step.

## Where time goes

1. **decoder_block** (~85.8s total): 36 steps × ~2.4s per block (self-attn, cross-attn, MLP, layer norm).
2. **encode** (~11.4s): one encoder pass (6 encoder_block + conv + ln).
3. **layer_normalization** (~9.3s): used inside decoder blocks.
4. **tokens** stmt (~92.3s): almost all of it is the 36 decode steps (emb + 4×decoder_block + ln + logits per step).

The inner **emb** clause (`if p < t then ... else 0`) is slice-vectorized, so it no longer runs as 40×384 scalar iterations per step.

---

## Cost of each statement inside `tokens` (per decode step, L229–244)

There are **36 steps** (t = 4..40). Approximate cost **per step** and **total over 36 steps**:

| Statement (main.ein) | Per step | Total (36 steps) | Notes |
|----------------------|----------|------------------|--------|
| **emb** (L230–232)   | ~0.001s | ~0.04s           | `emb[p,d] = if p < t then ... else 0` — vectorized (slice over p in 0..t). |
| **d0** = decoder_block(emb, enc_out, 0) | ~0.24–0.35s | ~9–13s  | First decoder block. |
| **d1** = decoder_block(d0, enc_out, 1) | ~0.22–0.40s | ~8–14s  | Second decoder block. |
| **d2** = decoder_block(d1, enc_out, 2) | ~0.21–0.32s | ~8–12s  | Third decoder block. |
| **d3** = decoder_block(d2, enc_out, 3) | ~0.21–0.40s | ~8–14s  | Fourth decoder block. |
| **final_ln** = layer_normalization(d3, ...) (L239) | ~0.01–0.03s | ~0.4–1s | One layer_norm per step. |
| **logits** = sum(...) (L241–242) | ~0.055–0.085s | ~2.2s | Einsum (384 × 51865). |
| **argmax(logits)** (L243) | &lt;0.01s | &lt;0.4s | Scalar/cheap. |

**Decoder blocks dominate:** total `decoder_block` time is **85.81s** over the run; almost all of that is from these four calls per step (36 × 4 = 144 calls). So **~0.6s per decoder_block call** on average. **logits** is next (~2.2s total). **emb** and **final_ln** are small after vectorization.

---

## Cost inside one `decoder_block` (per call, ~0.25–0.40s)

Structure: **ln1 → self-attn → ln2 → cross-attn → ln3 → MLP**. Approximate cost per clause/group (from profile, one typical block):

| Part | Clause / operation | Shape / role | Time (approx) |
|------|--------------------|--------------|----------------|
| **ln1** | layer_norm(x, …) | (40, 384) | ~0.01–0.04s |
| **Self-attn** | Q, K, V | (40, 384) matmuls | ~0.01–0.02s |
| | score = Qh·Kh / 8 + mask | (6, 40, 40) einsum | ~0.002–0.014s |
| | softmax, sa_ctx, sa_proj, res1 | (6,40,40), (6,40,64), (40,384) | ~0.005–0.02s |
| **ln2** | layer_norm(res1, …) | (40, 384) | ~0.01–0.02s |
| **Cross-attn** | Qc | (40, 384) | ~0.002–0.006s |
| | **Kc = enc · W_k** | **(1500, 384)** matmul | **~0.028–0.045s** |
| | **Vc = enc · W_v** | **(1500, 384)** matmul | **~0.030–0.039s** |
| | **cscore = Qch·Kch / 8** | **(6, 40, 1500)** einsum | **~0.024–0.026s** |
| | cscore_exp, ca, ca_ctx, ca_proj, res2 | softmax + (6,40,1500)·Vch, (40,384) | ~0.01–0.02s |
| **ln3** | layer_norm(res2, …) | (40, 384) | ~0.01–0.02s |
| **MLP** | fc1 = ln3 · W1 + b1 | (40, 1536) matmul | ~0.005–0.010s |
| | gelu(fc1) | (40, 1536) | ~0.002–0.004s |
| | fc2 = act · W2 + b2 | (40, 384) matmul | ~0.005–0.011s |
| | out = res2 + fc2 | (40, 384) | ~0.001s |

**Summary inside decoder_block:** **Cross-attention** dominates (~0.10–0.15s per block): **Kc** and **Vc** (enc 1500×384) and **cscore** (6×40×1500) are the heaviest. Self-attention (40×384, 6×40×40) and MLP (40×1536, 40×384) are smaller. The three layer norms are a few percent each.
