---
layout: book
title: "Chapter 10 · Comparison: Attention"
---

# Chapter 10 · Comparison: Attention

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

In Einlang, the coordinate names tell the story:

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

They are identical. Stop and look at that. Two pieces of code—self-attention and cross-attention—with different semantics, different gradient behavior, different architectural implications. Textually identical.

Where is the difference? In the code? No—the code for both is `torch.matmul(Q, K.transpose(-2, -1))`. In the tensor shapes at runtime? Yes—`seq_q` equals `seq_k` in one case and differs in the other. But shapes are runtime values, not source-level facts. The notation tells you nothing about which case you're in. You must trace the shapes backward through the forward pass to know—and after a refactoring, you must do it again.

The code does not distinguish between self-attention and cross-attention because the distinction—whether `seq_q == seq_k`—is not recorded anywhere in the source. It is recorded in the shapes of the tensors passed at runtime.

**Einlang:**

```rust
// Self-attention: same coordinate for queries and keys
fn self_attention[seq, head, d](Q: [f32; ..b, head, seq, d], K: [f32; ..b, head, seq, d], V: [f32; ..b, head, seq, d])
    -> [f32; ..b, head, seq, d]
{
    attention[seq, seq, head, d](Q, K, V)
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

In Einlang, `cross_attention[seq_q, seq_k, ...](Q, K, V)` with `Q` and `K` both bound to `decoder_hidden` would trigger a coordinate mismatch if `decoder_hidden` has `seq_q` but not `seq_k` as its declared coordinate. If both tensors carry both coordinates (because they were declared with different names), the mismatch is caught at the call site.

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

## Grouped-Query Attention (GQA): The Middle Ground

Between MHA (`head_q == head_kv`) and MQA (`head_kv == 1`) lies GQA: `head_kv` is a small number, say 4, that divides `head_q`. Each KV head is shared by a group of query heads. This is a coordinate *grouping* problem—structurally identical to GroupNorm from Chapter 8.

**PyTorch (GQA):**

```python
def gqa_attention(Q, K, V, num_kv_heads):
    # Q: (batch, head_q, seq_q, d)
    # K: (batch, num_kv_heads, seq_k, d)
    # V: (batch, num_kv_heads, seq_k, d)
    head_q = Q.shape[1]
    # Repeat KV heads: (batch, num_kv_heads, seq_k, d) → (batch, head_q, seq_k, d)
    repeat_factor = head_q // num_kv_heads
    K = K.unsqueeze(2).expand(-1, -1, repeat_factor, -1, -1).reshape(Q.shape)
    V = V.unsqueeze(2).expand(-1, -1, repeat_factor, -1, -1).reshape(Q.shape)
    # ... identical to standard attention from here
    d = Q.shape[-1]
    scores = torch.matmul(Q, K.transpose(-2, -1)) / (d ** 0.5)
    weights = torch.softmax(scores, dim=-1)
    return torch.matmul(weights, V)
```

The grouping logic—unsqueeze, expand, reshape—is spread across two lines. The fact that `head_q` is grouped into `(num_kv_heads, repeat_factor)` is encoded in a reshape chain that the reader must reverse-engineer. If the grouping factor changes, the reshape must be updated. If the layout changes (e.g., `head_q` moves from position 1 to position 2), the unsqueeze and expand must be re-aligned.

**Einlang (GQA):**

```rust
fn gqa_attention[head_group, head_kv, seq_q, seq_k, d](
    Q: [f32; ..b, head_group, head_kv, seq_q, d],
    K: [f32; ..b, head_kv, seq_k, d],
    V: [f32; ..b, head_kv, seq_k, d]
) -> [f32; ..b, head_group, head_kv, seq_q, d]
{
    let scores[..b, head_group, head_kv, seq_q, seq_k] =
        sum[d](Q[..b, head_group, head_kv, seq_q, d]
             * K[..b, head_kv, seq_k, d]) / (d ** 0.5);
    let weights[..b, head_group, head_kv, seq_q, seq_k] =
        softmax[seq_k](scores[..b, head_group, head_kv, seq_q, seq_k]);
    sum[seq_k](weights[..b, head_group, head_kv, seq_q, seq_k]
             * V[..b, head_kv, seq_k, d])
}
```

`head_group` and `head_kv` are separate coordinates from the start. No reshape. No expand. No unsqueeze. `K` and `V` are indexed by `head_kv` alone—they broadcast over `head_group` because they omit it. The broadcast is visible in the indexing pattern: `K[..b, head_kv, seq_k, d]` has no `head_group`, while `Q` has both.

Now compare the three variants side by side:

| Variant | Query heads | KV heads | Coordinate structure |
|---|---|---|---|
| MHA | `head` | `head` | `head` shared by Q, K, V |
| GQA | `head_group × head_kv` | `head_kv` | `head_kv` shared; `head_group` only on Q |
| MQA | `head_q` | `head_kv` (size 1) | `head_kv` on K, V; `head_q` only on Q |

In the Einlang signatures, the difference between the three variants is visible in which coordinates appear on which parameters. In the PyTorch implementations, the difference is buried in reshape chains and the value of `num_kv_heads`. The coordinate names make the architecture visible. The positional code makes it deducible—after counting dimensions and tracing reshapes.

---

## The Attention Coordinate Audit

Every attention variant can be audited with four questions. Ask them of any positional attention code you encounter:

1. **Which coordinate does `softmax` normalize over?** In `softmax(scores, dim=-1)`, the answer is "whatever is last." In `softmax[seq_k](scores)`, the answer is `seq_k`.
2. **Which coordinate distinguishes queries from keys?** In MHA, it's the same (`seq`). In cross-attention, it's different (`seq_q` vs `seq_k`). In positional code, this distinction is in the tensor shapes at runtime. In named code, it's in the function signature.
3. **Which coordinate groups query heads with KV heads?** In GQA, `head_group` groups query heads over a shared KV head. In MQA, `head_kv` has size 1. In MHA, there's no grouping—`head` is the same coordinate on Q and K. The grouping structure is invisible in the positional `matmul`; visible in the named index patterns.
4. **Does the backward pass know what to sum over?** The gradient of attention sums over `seq_q` for `dK` and `dV`, over `seq_k` for `dQ`, and over the head grouping for the KV projection. In positional autodiff, these sums happen silently. In named coordinates, they follow from the coordinate sets—same set-subtraction rule from Chapter 7 applied to attention.

You don't need Einlang to ask these questions. You need to know that they are the right questions. And the right questions are only visible when the notation has a place for the answers.

Stop now and look at the last attention implementation you wrote. Can you answer all four questions from the code alone?

---

## The KV-Cache Audit

Autoregressive generation uses a KV-cache: keys and values from previous time steps are stored and reused. The cache introduces a new coordinate relationship: `seq_past` (cached) and `seq_new` (current) must be concatenated into a single `seq_k` for the attention computation.

```python
# Positional KV-cache
K_full = torch.cat([K_cache, K_new], dim=seq_dim)  # which axis is seq_dim?
V_full = torch.cat([V_cache, V_new], dim=seq_dim)
output = attention(Q_new, K_full, V_full)
```

The `dim` argument to `torch.cat` is a position number. If `K_cache` has shape `(batch, head, past_len, d)` and `K_new` has shape `(batch, head, 1, d)`, then `seq_dim` is `2`. But if the layout is `(batch, past_len, head, d)`, `seq_dim` is `1`. The integer shifts with the layout. Change the layout, audit every `cat` call.

In Einlang, the concatenation axis is named:

```rust
let K_full[..batch, head, seq_k, d] = concat[seq_k](
    K_cache[..batch, head, seq_past, d],
    K_new[..batch, head, seq_new, d]
);
let output[..batch, head, seq_q, d] = attention[head, seq_q, seq_k, d](Q_new, K_full, V_full);
```

`concat[seq_k]` names the concatenation axis. The coordinate `seq_k` absorbs both `seq_past` and `seq_new` into a single coordinate. If the layout changes, the coordinate name doesn't. The `cat` happens over `seq_k` regardless of position.

The audit questions for a KV-cache:
1. Which coordinate does `concat` operate over? (`seq_k`)
2. Which coordinate does the attention reduce over? (`seq_k`—the same coordinate)
3. Does the cached `seq_k` range differ from the new `seq_k` range? (They are different domains, now merged)

The coordinate names make the cache structure visible. The positional `dim=seq_dim` records a position. The named `concat[seq_k]` records an identity.

---

如果 `seq_q == seq_k`，这是自注意力还是交叉注意力？代码能回答吗？

In PyTorch, the answer is no—self-attention and cross-attention have identical code. The distinction is in the runtime shapes of the tensors passed at the call site, not in the source. In Einlang, `self_attention[seq, ...]` and `cross_attention[seq_q, seq_k, ...]` are different signatures. The distinction is in the brackets. The code answers the question before it runs.

---

## Flash Attention: The Coordinate Structure Survives Optimization

Flash Attention is a memory-efficient exact attention algorithm that fuses the QK^T matmul, softmax, and PV matmul into a single tiled kernel. It dramatically reduces memory usage by recomputing the softmax statistics in the backward pass rather than storing the full attention matrix. From the user's perspective, the function signature is identical to standard attention. The coordinate structure is unchanged.

This is a demonstration of the principle from Chapter 14: lowering is strategy-independent. The same coordinate structure maps to different execution strategies. Flash Attention is a lowering strategy—a choice of how to execute the computation, not what computation to execute. The coordinate names `seq_q`, `seq_k`, `head`, `d` are identical whether the lowering chooses the standard attention kernel or the Flash Attention kernel.

In a positional API, Flash Attention is a drop-in replacement: replace `attention(Q, K, V)` with `flash_attention(Q, K, V)`. The shapes are the same. The coordinate structure is the same—but only implicitly, in the shapes. In an Einlang API, the coordinate contract is the same—`fn attention[seq_q, seq_k, head, d](...)` for both. The lowering strategy (`standard` vs `flash`) is an annotation, not a signature change:

```rust
#[strategy(flash)]
let output[..b, head, seq_q, d] = attention[head, seq_q, seq_k, d](Q, K, V);
```

The coordinate names don't change. The contract doesn't change. Only the execution strategy changes. This is the separation that the compiler's lowering pass enables: coordinate contracts define what is computed. Lowering strategies define how it is computed. The names belong to the first. The optimizations belong to the second. They are orthogonal.

When a new attention variant appears—Flash Attention 2, Flash Attention 3, a sparse attention kernel, a sliding-window attention—the coordinate contract remains the same. The lowering strategy changes. The names survive the optimization. The reader of the code still sees `attention[head, seq_q, seq_k, d](Q, K, V)` and knows what coordinates are involved, regardless of which kernel executes underneath.

This is the parting lesson of the comparison chapters. The coordinate structure is the invariant. The execution strategy is the variable. Named coordinates record the invariant. Positional notation records neither—it defers both to runtime. The difference is whether the invariant survives the next optimization.


---

*Read the last attention implementation you wrote. Can you distinguish self-attention from cross-attention from the code alone—without checking tensor shapes at runtime? If the answer is no, the distinction lives in your head. That is the gap.*

---

### Stop and Think: The Attention Audit

Open the last attention implementation you wrote. It might be self-attention, cross-attention, multi-head, multi-query, or grouped-query. For each attention variant in your code, answer these four questions from the Coordinate Audit in Section 6:

1. **Which coordinate does `softmax` normalize over?** In positional code, this is `dim=-1` or `dim=seq_dim`. Can you name the coordinate without checking the tensor shape at runtime? If the answer is "the last dimension" — that's a position, not an identity.

2. **Which coordinate distinguishes queries from keys?** In self-attention, it's the same coordinate. In cross-attention, they're different. Can you tell which variant you have from the code alone? If the code is `attention(Q, K, V)` for both — the distinction is in the tensor shapes, not in the source.

3. **Which coordinate groups query heads with KV heads?** In GQA, query heads are grouped with KV heads. In MQA, KV heads = 1. In MHA, they're the same. Can you tell which architecture you have from the code? If the grouping is `repeat_factor = head_q // num_kv_heads` — the grouping is a runtime calculation, not a source-level fact.

4. **Does the backward pass know what to sum over?** The gradient of attention needs to sum over `seq_q` for `dK`/`dV`, over `seq_k` for `dQ`, and over the head grouping for the KV projection. Are these sums explicit in the backward code? Or are they generated by autograd from positional matmuls?

You don't need Einlang to ask these questions. You need to know they're the right questions. And they're only the right questions when the notation has a place for the answers.

Now do one more thing. Write the Einlang signature for your attention function — just the signature, no body. `fn my_attention[seq_q, seq_k, head, d](Q: ..., K: ..., V: ...) -> ...`. Does the signature tell a reader:
- Whether it's self-attention (`seq_q` = `seq_k`) or cross-attention (different)?
- Whether it's MHA (same `head` on Q, K, V), GQA (`head_group, head_kv`), or MQA (`head_q, head_kv`)?
- What `softmax` normalizes over?

If the signature answers all three, you've discovered why the coordinate names matter for attention. If it doesn't, the distinction lives in your head — and in the tensor shapes at runtime.

In the next chapter, we complete the comparison trilogy with physical simulation—where the coordinates represent temperature, pressure, and velocity fields, and confusing them means solving the wrong physics.
