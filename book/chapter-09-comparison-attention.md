---
layout: book
title: "Chapter 9 · Comparison: Attention"
---

# Chapter 9 · Comparison: Attention

> "The attention mechanism is the only thing standing between your model and the void. You should probably know which coordinates it's attending over."
>
> — A reviewer

*Comparisons · Self-attention, cross-attention, and multi-query attention in two notations*

---

You are writing an encoder-decoder model. During development, source and target happen to have the same sequence length—64 tokens. Self-attention works. Cross-attention works. The code for both is a single positional function: `attention(Q, K, V, mask)`. The shapes match. You ship.

Six weeks later, a configuration change sets source length to 128, target to 64. The code still runs—same shapes at the `attention` call, broadcasting absorbs the mismatch. But your model now attends from every target position to every source position *twice*, silently. The BLEU score drops two points. You spend three days tracing the drop to a transposed mask that was broadcasting along the wrong axis. The Square Matrix Test, first encountered with softmax in Chapter 3, returns with a vengeance.

Three numbers govern attention: the number of query heads, the number of key-value heads, and the sequence length. When any two coincide, positional code for one variant becomes textually identical to positional code for another. Only the coordinate names distinguish them.

---

## Scaled Dot-Product Attention: The Skeleton

The core operation:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d}}\right)V$$

In einlang, the coordinate names tell the story:

```rust
fn attention[seq_q, seq_k, head, d](
    Q: [f32; ..batch, head, seq_q, d],
    K: [f32; ..batch, head, seq_k, d],
    V: [f32; ..batch, head, seq_k, d]
) -> [f32; ..batch, head, seq_q, d]
{
    let scores[..batch, head, seq_q, seq_k] =
        sum[d](Q[..batch, head, seq_q, d] * K[..batch, head, seq_k, d]) / (d ** 0.5);
    let weights[..batch, head, seq_q, seq_k] = softmax[seq_k](scores[..batch, head, seq_q, seq_k]);
    sum[seq_k](weights[..batch, head, seq_q, seq_k] * V[..batch, head, seq_k, d])
}
```

Three coordinates do all the work: `seq_q` (which queries we're attending *from*), `seq_k` (which keys we're attending *to*), and `d` (the inner dimension that gets contracted). `softmax[seq_k]` normalizes over the key sequence—each query position produces a distribution over key positions.

---

## Self-Attention vs. Cross-Attention

Self-attention uses the same sequence for queries and keys. Cross-attention uses different sequences—queries from the decoder, keys from the encoder.

**PyTorch (self-attention):**

```python
def self_attention(Q, K, V):
    d = Q.shape[-1]
    scores = torch.matmul(Q, K.transpose(-2, -1)) / (d ** 0.5)
    weights = torch.softmax(scores, dim=-1)
    return torch.matmul(weights, V)
```

**PyTorch (cross-attention):**

```python
def cross_attention(Q, K, V):
    d = Q.shape[-1]
    scores = torch.matmul(Q, K.transpose(-2, -1)) / (d ** 0.5)
    weights = torch.softmax(scores, dim=-1)
    return torch.matmul(weights, V)
```

They are identical. The code does not distinguish between self-attention and cross-attention because the distinction—whether `seq_q == seq_k`—is not recorded anywhere in the source. It is recorded in the shapes of the tensors passed at runtime.

**Einlang:**

```rust
// Self-attention: same coordinate for queries and keys
fn self_attention[seq, head, d](Q: [f32; ..b, head, seq, d], K: [f32; ..b, head, seq, d], V: [f32; ..b, head, seq, d])
    -> [f32; ..b, head, seq, d]
{
    attention[seq_q=seq, seq_k=seq, head, d](Q, K, V)
}

// Cross-attention: different coordinates for queries and keys
fn cross_attention[seq_q, seq_k, head, d](Q: [f32; ..b, head, seq_q, d], K: [f32; ..b, head, seq_k, d], V: [f32; ..b, head, seq_k, d])
    -> [f32; ..b, head, seq_q, d]
{
    attention[seq_q, seq_k, head, d](Q, K, V)
}
```

The distinction is in the type signatures. Self-attention uses `seq` for both queries and keys. Cross-attention uses `seq_q` and `seq_k`—two different coordinate names, potentially with different domain sizes. A reader can tell which is which without checking whether the tensors happen to have the same shape.

---

## The Square Matrix Test for Attention

When `seq_q == seq_k` and `head == some_other_dimension`, the attention matrix is square. The positional code for self-attention, cross-attention, and a transposed variant are numerically identical. Consider this bug:

```python
# Intended: cross-attention from decoder (seq_len=32) to encoder (seq_len=100)
# Bug: accidentally used the same tensor for Q and K
Q = decoder_hidden   # shape (batch, head, 32, d)
K = decoder_hidden   # bug: should be encoder_hidden, shape (batch, head, 100, d)
V = decoder_hidden
output = cross_attention(Q, K, V)  # silently becomes self-attention
```

If `decoder_hidden` and `encoder_hidden` happen to have the same sequence length during development (both 32, or both padded to the same length), this bug is invisible. The shapes match. The loss descends. The model learns—just not what you intended.

In einlang, `cross_attention[seq_q, seq_k, ...](Q, K, V)` with `Q` and `K` both bound to `decoder_hidden` would trigger a coordinate mismatch if `decoder_hidden` has `seq_q` but not `seq_k` as its declared coordinate. If both tensors carry both coordinates (because they were declared with different names), the mismatch is caught at the call site.

---

## Multi-Query Attention (MQA)

MQA uses multiple query heads but only one key-value head, broadcasting the KV head across query heads. This is a performance optimization that changes the coordinate structure:

**PyTorch:**

```python
def mqa_attention(Q, K, V):
    # Q: (batch, head_q, seq_q, d)
    # K: (batch, 1, seq_k, d)      -- single KV head
    # V: (batch, 1, seq_k, d)
    d = Q.shape[-1]
    scores = torch.matmul(Q, K.transpose(-2, -1)) / (d ** 0.5)
    weights = torch.softmax(scores, dim=-1)
    return torch.matmul(weights, V)
```

The code is identical to standard attention. The only difference is that `K` has shape `(batch, 1, seq_k, d)` instead of `(batch, head, seq_k, d)`. The `1` broadcasts silently over all query heads. If someone changes the KV projection to output `head_kv` heads instead of `1`, the code still runs—it just produces a different attention pattern. The `1` is a positional convention, not a checked fact.

**Einlang:**

```rust
fn mqa_attention[head_q, head_kv, seq_q, seq_k, d](
    Q: [f32; ..b, head_q, seq_q, d],
    K: [f32; ..b, head_kv, seq_k, d],
    V: [f32; ..b, head_kv, seq_k, d]
) -> [f32; ..b, head_q, seq_q, d]
{
    let scores[..b, head_q, head_kv, seq_q, seq_k] =
        sum[d](Q[..b, head_q, seq_q, d] * K[..b, head_kv, seq_k, d]) / (d ** 0.5);
    let scores_merged[..b, head_q, seq_q, seq_k] = mean[head_kv](scores[..b, head_q, head_kv, seq_q, seq_k]);
    let weights[..b, head_q, seq_q, seq_k] = softmax[seq_k](scores_merged[..b, head_q, seq_q, seq_k]);
    sum[seq_k](weights[..b, head_q, seq_q, seq_k] * V[..b, head_kv, seq_k, d])
}
```

`head_q` and `head_kv` are different coordinates. The function signature declares that queries have `head_q` heads and keys have `head_kv` heads. When called as MQA, `head_kv` has size 1—but it's a named coordinate, not a silent `1` buried in the shape. If a refactoring changes the KV head count, the coordinate name `head_kv` remains, and it is verified.

The key insight: `head_kv` is a coordinate whose domain happens to be size 1 in the MQA case. It is not a broadcasting hack. It is a structural fact, visible in the type.

---

## What the Comparison Reveals

Attention is the Square Matrix Test at scale. Almost every dimension can be square with some other dimension in some configuration. `seq_q` can equal `seq_k`. `head_q` can equal `head_kv`. `d` can equal any of them. When dimensions coincide, positional code for different architectures becomes textually identical.

In einlang, the coordinate names persist through the coincidence. `self_attention[seq, ...]` and `cross_attention[seq_q, seq_k, ...]` are different functions with different signatures, even when `seq_q == seq_k` at runtime. The difference is in the source code, where a reader can see it and static analysis can check it.

In the next chapter, we complete the comparison trilogy with physical simulation—where the coordinates represent temperature, pressure, and velocity fields, and confusing them means solving the wrong physics.
