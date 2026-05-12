---
layout: book
title: "Chapter 5 · Blocks and Skeletons"
---

# Chapter 5 · Blocks and Skeletons

> "If you have a procedure with ten parameters, you probably missed some."
>
> — Alan Perlis

*Combinations · Pack polymorphism and normalization skeletons*

---

Primitives let you say simple things. Combinations let you compose simple things into complex ones. Abstraction lets you name the complex things and use them as if they were primitive.

A coordinate pattern wrapped in a function signature produces a reusable block. Several such blocks reveal a common skeleton, dressed in different coordinate names.

---

## The Anatomy of a Coordinate-Aware Function

Here is the complete form:

```rust
fn normalize[coord](x: [f32; ..left, coord, ..right])
    -> [f32; ..left, coord, ..right]
{
    let m[..left, ..right] = max[coord](x[..left, coord, ..right]);
    let centered[..left, coord, ..right] = x[..left, coord, ..right] - m[..left, ..right];
    let scale[..left, ..right] = sum[coord](centered[..left, coord, ..right] ** 2.0);
    centered[..left, coord, ..right] / (scale[..left, ..right] ** 0.5 + 1e-5)
}
```

Four things to notice.

**First, the coordinate parameter `coord`.** It appears in brackets after the function name and in the type signatures. It is not a value—you cannot do arithmetic on it. It is a coordinate identity. When the function body uses `max[coord]` and `sum[coord]`, it is verified that `coord` is the same parameter declared in the signature.

**Second, the rest packs `..left` and `..right`.** They stand for whatever coordinates surround `coord` in the actual argument. If the caller passes `x[b, t, f]` and writes `normalize[f](x)`, then `..left` binds to `[b, t]` and `..right` binds to nothing. If the caller passes `x[b, h, f, d]` with `normalize[f](x)`, then `..left` binds to `[b, h]` and `..right` binds to `[d]`. The function body is polymorphic over the surrounding structure.

**Third, coordinate flow.** Inside the function body, the coordinate `coord` is in scope. It can be used in reductions (`max[coord]`, `sum[coord]`), in indexing, and implicitly in the output shape. The packs `..left` and `..right` flow from the input signature to the output signature.

**Fourth, the return type annotation.** The `-> [f32; ..left, coord, ..right]` tells the reader which coordinates survive. If the function body accidentally consumed `coord` without reconstructing it, or dropped a pack, a coordinate mismatch is reported.

---

## Packs and Polymorphism

Packs are what make coordinate-aware functions reusable across different tensor ranks. There are three patterns:

**Leading packs** (`..left` or `..batch`): absorb dimensions that come before the coordinate of interest. Used when the function treats all leading dimensions uniformly.

**Trailing packs** (`..right` or `..rest`): absorb dimensions that come after. Used when the function operates on a coordinate and doesn't care what follows.

**Named spatial packs** (`..spatial`): absorb spatial dimensions as a group. A function that reshapes spatial coordinates can treat them as a unit:

```rust
fn move_channel[channel, ..spatial](x: [f32; channel, ..spatial])
    -> [f32; ..spatial, channel]
{
    x[..spatial, channel]
}
```

When a pack is ambiguous at the call site, the caller disambiguates by grouping:

```rust
id_axes[(height, width)](x)
```

Pack parameters make coordinate-aware functions rank-polymorphic: the same function works on 2D, 3D, 4D, or higher-dimensional tensors, as long as the coordinate of interest exists somewhere in the shape.

---

## Selection Reductions

A normal reduction returns a combined value. A selection reduction returns an address:

```rust
let pred[b] = argmax[class](logits[b, class]);
```

`max[class](logits[b, class])` returns the maximum *value* for each `b`. `argmax[class](logits[b, class])` returns the index within the `class` domain where that maximum occurs. The result has the surviving coordinates—`[b]`—plus an implicit address domain: the integer stored at each `b` is understood to be an address in the `class` coordinate space.

This is more than a convenience. The return value of `argmax[class]` is not just an integer tensor. It is an integer tensor whose values are addresses in the `class` domain. Later code can use this address to index back into the `class` dimension of another tensor.

---

## One Skeleton, Four Normalizations

Here are four normalization implementations. Read them. Find what they share.

**LayerNorm:**
```python
def layer_norm(x, gamma, beta, eps=1e-5):
    mean = x.mean(dim=-1, keepdim=True)
    var = ((x - mean) ** 2).mean(dim=-1, keepdim=True)
    return (x - mean) / (var + eps).sqrt() * gamma + beta
```

**RMSNorm:**
```python
def rms_norm(x, gamma, eps=1e-5):
    rms = (x ** 2).mean(dim=-1, keepdim=True).sqrt()
    return x / (rms + eps) * gamma
```

**GroupNorm:**
```python
def group_norm(x, num_groups, gamma, beta, eps=1e-5):
    N, C, H, W = x.shape
    x = x.reshape(N, num_groups, C // num_groups, H, W)
    mean = x.mean(dim=(2, 3, 4), keepdim=True)
    var = x.var(dim=(2, 3, 4), keepdim=True)
    x = (x - mean) / (var + eps).sqrt()
    return x.reshape(N, C, H, W) * gamma + beta
```

**InstanceNorm:**
```python
def instance_norm(x, gamma, beta, eps=1e-5):
    N, C, H, W = x.shape
    mean = x.mean(dim=(2, 3), keepdim=True)
    var = x.var(dim=(2, 3), keepdim=True)
    return (x - mean) / (var + eps).sqrt() * gamma + beta
```

Look at the code. The common structure is visible across all four functions. Can you see the specific sequence of operations they all perform? What varies between them?

Here is what they share:

1. **Compute a statistic** (mean, rms, or both) by reducing over one or more dimensions.
2. **Broadcast the statistic back** over the reduced dimensions (via `keepdim=True`).
3. **Apply the statistic elementwise** (subtract and divide).
4. **Scale and shift** with learned parameters that broadcast over the non-feature dimensions.

Every one of these functions does: reduce, broadcast, elementwise, scale. The difference is only *which* dimensions are reduced and *which* parameters broadcast over *which* remaining dimensions.

But look at the code again. Can you *see* the skeleton? In LayerNorm, it's `x.mean(dim=-1, keepdim=True)`. In GroupNorm, it's `x.mean(dim=(2,3,4), keepdim=True)`. In InstanceNorm, it's `x.mean(dim=(2,3), keepdim=True)`. The `dim` arguments are different integers. The `keepdim=True` flag is the same. The `* gamma + beta` ending is the same.

The skeleton IS there—but it's encoded as shape arithmetic. `dim=-1` means one thing in LayerNorm ("the last dimension") and a completely different set of integers in GroupNorm ("dimensions 2, 3, and 4"). The skeleton is visible to a human who understands the dimension layout. It is invisible to a compiler. And it changes when the layout changes.

Now here is the same skeleton in Einlang:

| Function | Reduction coords | Broadcast params | Survivors |
|---|---|---|---|
| Softmax | `q` (max), `k` (sum) | none | `..batch`, `j` |
| LayerNorm | `f` (mean ×2) | `gamma[f]`, `beta[f]` | `..batch`, `f` |
| RMSNorm | `f` (mean) | `gamma[f]` | `..batch`, `f` |
| GroupNorm | `c_in_group`, `..spatial` | `gamma[g, c_in_group]`, `beta[g, c_in_group]` | `..batch`, `g`, `c_in_group`, `..spatial` |
| InstanceNorm | `..spatial` | `gamma[c]`, `beta[c]` | `..batch`, `c`, `..spatial` |

The skeleton is visible in the table because the coordinates are named. Each column says *what*, not *where*. The reduction column names the consumed coordinates. The broadcast column names the parameters and their coordinate sets. The survivors column names what's left.

In a positional API, all four collapse to a single `dim` argument whose meaning shifts with the surrounding layout. `LayerNorm` and `RMSNorm` both use `dim=-1`—but normalize different statistics. `GroupNorm` uses three reduction dimensions buried in a `reshape` chain. The skeleton is invisible.

In a named-coordinate API, the skeleton is a template you can check. The reduction bracket names the consumed coordinates. The indexing pattern names the survivors. The broadcast parameters name the omission. A reviewer can verify that the broadcast coordinate in LayerNorm matches the broadcast coordinate in the gradient without reconstructing both from positional offsets.

This is abstraction: recognizing a pattern, naming it, and reusing it. The pattern is "normalize with named coordinates." Each instance fills in the specific coordinates. The skeleton is constant.

The discovery exercise—comparing four implementations, finding their shared structure—is what you do every time you read unfamiliar tensor code. Names carry the structure. Positions hide it.

Now let's put this claim to the test. Here is a real GroupNorm implementation in PyTorch:

```python
def group_norm(x, num_groups, gamma, beta, eps=1e-5):
    N, C, H, W = x.shape
    x = x.reshape(N, num_groups, C // num_groups, H, W)
    mean = x.mean(dim=(2, 3, 4), keepdim=True)
    var = x.var(dim=(2, 3, 4), keepdim=True)
    x = (x - mean) / (var + eps).sqrt()
    x = x.reshape(N, C, H, W)
    return x * gamma + beta
```

Stop and read this carefully. Ask yourself: which dimensions are being reduced by `dim=(2, 3, 4)`? What do positions 2, 3, and 4 correspond to? You need to trace backward through the `reshape`—position 2 is `C // num_groups` (channels per group), position 3 is `H`, position 4 is `W`. But this reasoning depends on the reshape chain. If the reshape changes, the `dim` tuple must change with it. If someone adds a temporal dimension before the spatial ones, the tuple shifts silently.

Now compare the Einlang version:

```rust
fn group_norm[g, c_in_group, ..spatial](x: [f32; ..batch, g, c_in_group, ..spatial],
    gamma: [f32; g, c_in_group], beta: [f32; g, c_in_group])
    -> [f32; ..batch, g, c_in_group, ..spatial]
{
    let m[..batch, g] = mean[c_in_group, ..spatial](x[..batch, g, c_in_group, ..spatial]);
    let v[..batch, g] = mean[c_in_group, ..spatial](
        (x[..batch, g, c_in_group, ..spatial] - m[..batch, g]) ** 2.0
    );
    let y[..batch, g, c_in_group, ..spatial] =
        (x[..batch, g, c_in_group, ..spatial] - m[..batch, g])
        / (v[..batch, g] + 1e-5) ** 0.5;
    y[..batch, g, c_in_group, ..spatial] * gamma[g, c_in_group] + beta[g, c_in_group]
}
```

The reduced coordinates are named: `c_in_group` and `..spatial`. No reshape needed. No positional arithmetic needed. If a temporal dimension is added, `..spatial` absorbs it—the reduction bracket stays the same. If `num_groups` changes, the coordinate `g` handles it—its domain just has a different size.

Here is the deeper point. In a positional API, "feature" is the last axis in a 2D tensor—`dim=-1`. But in a 4D tensor, what was one axis may span multiple actual dimensions: channels per group, height, width. Positional code requires a reshape-permute-reshape dance to group and ungroup those. Named coordinates handle this naturally: one semantic coordinate (`c_in_group`) plus a spatial pack (`..spatial`) together cover what `dim=(2,3,4)` covers in the positional version. The names don't change when the layout changes.

This is what packs buy you. `..spatial` absorbs however many spatial dimensions exist. The same `GroupNorm` skeleton works whether spatial covers one axis or three. `mean[c_in_group, ..spatial]` says exactly what is consumed—no reshape chain to reverse-engineer.

Now one more question. Suppose you encounter a new normalization variant—say, normalize only over the spatial dimensions, keeping the channel-group dimension intact. What would you change?

Think about it. In the Einlang version, you change one thing: remove `c_in_group` from the reduction bracket. `mean[..spatial](...)` instead of `mean[c_in_group, ..spatial](...)`. The skeleton is unchanged. The coordinate names carry the design decision.

In the PyTorch version, you'd change `dim=(2, 3, 4)` to `dim=(3, 4)`—but only if the reshape hasn't changed the position of the spatial dimensions. If someone added a temporal axis between `c_in_group` and `H`, the tuple would need to shift to `dim=(4, 5)`. The fragility is not in the concept—it is in the notation's inability to record *which* dimensions are spatial.

You have already been thinking in terms of the skeleton. The four-normalizations table at the beginning of this section was not an abstract taxonomy—it was a description of the mental model you were already using. Named coordinates make that mental model executable.

---

This is what packs buy you. `..spatial` absorbs however many spatial dimensions there are. The same `GroupNorm` skeleton works whether spatial covers one axis or three.

---

## Skeletons Compose

The normalization skeleton and the attention skeleton compose. A Transformer block is LayerNorm, then attention, then another LayerNorm, then a feedforward. In a positional implementation, the norm dimensions and attention dimensions share the `dim=-1` convention—until one of them shouldn't.

Here is a complete Transformer block skeleton in Einlang:

```rust
fn transformer_block[head, seq, d, d_ff](
    x: [f32; ..batch, seq, d],
    W_q: [f32; head, d, d_k],
    W_k: [f32; head, d, d_k],
    W_v: [f32; head, d, d_v],
    W_o: [f32; head, d_v, d],
    W_1: [f32; d, d_ff],
    W_2: [f32; d_ff, d],
    gamma1: [f32; d], beta1: [f32; d],
    gamma2: [f32; d], beta2: [f32; d]
) -> [f32; ..batch, seq, d]
{
    // LayerNorm 1
    let norm1[..batch, seq, d] = layer_norm[d](x[..batch, seq, d], gamma1[d], beta1[d]);
    
    // Multi-head attention
    let attn_out[..batch, seq, d] = attention[head, seq, seq, d](
        norm1[..batch, seq, d], norm1[..batch, seq, d], norm1[..batch, seq, d],
        W_q[head, d, d_k], W_k[head, d, d_k], W_v[head, d, d_v], W_o[head, d_v, d]
    );
    
    // Residual connection
    let res1[..batch, seq, d] = x[..batch, seq, d] + attn_out[..batch, seq, d];
    
    // LayerNorm 2
    let norm2[..batch, seq, d] = layer_norm[d](res1[..batch, seq, d], gamma2[d], beta2[d]);
    
    // Feedforward
    let ff[..batch, seq, d_ff] = relu(sum[d](norm2[..batch, seq, d] * W_1[d, d_ff]));
    let ff_out[..batch, seq, d] = sum[d_ff](ff[..batch, seq, d_ff] * W_2[d_ff, d]);
    
    // Residual connection
    res1[..batch, seq, d] + ff_out[..batch, seq, d]
}
```

Every coordinate that is consumed is named in a bracket. Every coordinate that survives is named in the output pattern. The `d` coordinate is consumed in two reductions (`layer_norm[d]` and `attention[..., d]`) and reconstructed each time. The `head` coordinate appears on the attention weights but not on the input `x`—it splits the feature dimension without changing the data layout.

Now ask: if you wanted to change this to a cross-attention block where queries come from one sequence and keys/values from another, what would you change? In a positional implementation, the code wouldn't change at all—the same `attention(Q, K, V)` call works for both. The difference is only in which tensors you pass. In the Einlang version, you'd change the signature: the first `norm1` gets coordinate `seq_q`, the second and third get coordinate `seq_k`. The code change is a coordinate name swap. The reader sees the architectural decision in the type signature.

Skeletons compose because coordinate contracts compose. The output coordinates of one function become the input coordinates of the next. The compiler traces the flow. You trace the meaning.

---

## Spot the Skeleton

Here are four Einlang function signatures. Three implement normalization variants. One doesn't. Can you spot the odd one out?

```rust
fn A[j](x: [f32; ..b, j]) -> [f32; ..b, j]
fn B[coord](x: [f32; ..b, coord]) -> [f32; ..b]
fn C[f](x: [f32; ..b, f]) -> [f32; ..b, f]
fn D[t](x: [f32; ..b, t]) -> [f32; ..b, t]
```

Look at the return types and notice the pattern yourself before reading on.

Here is the pattern: Function B is the odd one out. Its return type is `[f32; ..b]`—the coordinate `coord` is missing. It was consumed and not reconstructed. Functions A, C, and D all return `[f32; ..b, <coordinate>]`—the coordinate survives. B is a reduction function (like `sum[coord]`). A, C, and D are normalization functions that preserve the coordinate.

Now the deeper question: **why** is this distinction visible in the type signature? Because the skeleton is more than "reduce then broadcast." The skeleton is "reduce, then broadcast back to **reconstruct the consumed coordinate in the output.**" A pure reduction consumes and doesn't reconstruct—the coordinate disappears from the return type. A normalization consumes and reconstructs—the coordinate reappears. The difference between "gone forever" and "gone and returned" is the difference between a reduction and a normalization. The return type records it.

The skeleton is visible in the type signature. The reduction bracket in the body (`max[coord]`, `mean[f]`, `sum[j]`) tells you what is consumed. The return type tells you whether it was reconstructed. A reader can distinguish a normalization from a reduction without reading the body—the coordinate flow is in the signature.

This is the abstraction layer. The function signature says: "I operate on coordinate `j`. I preserve `j` in the output. Everything else passes through." The body fills in the specific computation. The signature is the contract. The body is the implementation. And the contract is checkable.

---

## Derive InstanceNorm

You've seen the table. Now derive one entry yourself. InstanceNorm normalizes each sample's each channel independently over the spatial dimensions. In 2D: for each `(N, C)`, compute mean and variance over `(H, W)`.

Look at InstanceNorm. The coordinates it reduces over and the coordinates that survive are visible in the operation's signature. Which coordinates does it consume? Which survive? What parameters broadcast? The answers are in the brackets. Use `..spatial` for the spatial dimensions and `c` for channel.

Here is the answer:

```rust
fn instance_norm[c, ..spatial](x: [f32; ..batch, c, ..spatial],
                                gamma: [f32; c],
                                beta: [f32; c])
    -> [f32; ..batch, c, ..spatial]
{
    let m[..batch, c] = mean[..spatial](x[..batch, c, ..spatial]);
    let v[..batch, c] = mean[..spatial]((x[..batch, c, ..spatial] - m[..batch, c]) ** 2.0);
    let y[..batch, c, ..spatial] =
        (x[..batch, c, ..spatial] - m[..batch, c]) / (v[..batch, c] + 1e-5) ** 0.5;
    y[..batch, c, ..spatial] * gamma[c] + beta[c]
}
```

The reduced coordinates are `..spatial`. The surviving coordinates are `..batch`, `c`, and `..spatial`—the spatial coordinates are consumed for the statistics but preserved in the output (the output still has spatial dimensions). The broadcast parameters are `gamma[c]` and `beta[c]`, which broadcast over `..batch` and `..spatial`.

Notice the pattern in the answer. The reduced coordinates are `..spatial`. The surviving coordinates are `..batch`, `c`, and `..spatial`—the spatial coordinates are consumed for the statistics but preserved in the output (the output still has spatial dimensions). The broadcast parameters are `gamma[c]` and `beta[c]`, which broadcast over `..batch` and `..spatial`.

Three things to observe:
- `..spatial` absorbs however many spatial dimensions there are.
- `..spatial` is placed in the reduction bracket because it's consumed for the statistics.
- `..spatial` is kept in the return type because the output is not a scalar—it's a tensor with spatial dimensions.

These three observations together capture the skeleton. The coordinate names carry the design: `mean[..spatial]` says "I am consuming the spatial dimensions." The return type `[f32; ..batch, c, ..spatial]` says "the spatial dimensions survive." The contradiction resolves: `..spatial` is consumed in the reduction but reconstructed in the output—the signature guarantees it.

Now consider: what if InstanceNorm should normalize over `c` as well? You'd change the reduction bracket to `[c, ..spatial]`. One change. The skeleton is the same. The coordinate name carries the design decision.

---

## Coordinate Facts Flow

Coordinate facts flow through pointwise operations automatically. If `x` carries the fact that it has coordinates `[b, class]`, then `x ** 2.0` also has coordinates `[b, class]`. You don't need to re-declare them. Coordinate facts are preserved through arithmetic, through function calls that return tensors, through `if` expressions. The only operations that change coordinate facts are those that explicitly manipulate coordinates: reductions (which consume), rectangular declarations (which introduce), and coordinate-aware function calls (which thread them through signatures).

This flow is the foundation of abstraction. When you wrap a computation in a coordinate-aware function, the coordinate facts flow from the caller, through the function body, and out to the result. The function doesn't need to re-derive what the caller already knew. The facts propagate.

This is a stronger property than type inference. Type inference deduces that `x ** 2.0` has the same type as `x` (both `f32`). Coordinate flow deduces that `x ** 2.0` has the same coordinate structure as `x` (both `[b, class]`). The coordinate structure is not inferred from runtime shapes—it is propagated from declarations. If `x` is declared as `x[b, class]`, every expression built from `x` carries `(b, class)` unless an operation explicitly removes a coordinate.

The difference between type inference and coordinate flow is that type inference is standard in every typed language, while coordinate flow is absent from every major tensor framework. A PyTorch tensor carries shape information at runtime—`(32, 64)`—but no coordinate identities. The identities are lost the moment the tensor leaves the data loader. In Einlang, the identities propagate through every operation, every function call, every intermediate binding. They are never inferred from shapes. They are propagated from declarations. The source is always the declaration. The flow is always forward.

The pattern is visible across three cases:

1. `let y = x + 1.0;` — coordinates flow through: `y` has the same coordinates as `x`.
2. `let y = sum[j](x[i, j]);` — coordinate `j` is consumed: `y` has `[i]`.
3. `let y[i, j] = x[j, i];` — coordinates are rearranged: `y` has `[i, j]` but from `x[j, i]`.

The third case is worth pausing on. `y[i, j] = x[j, i]` is a transpose. The declaration bracket says `y` has coordinates `(i, j)`. The body references `x[j, i]`—the same names, swapped positions. `i` and `j` are preserved. Their order is changed. This is the coordinate-aware way to write a transpose: not `x.transpose(0, 1)`, but `y[i, j] = x[j, i]`. The names carry the permutation. No position counting needed.

---

Coordinates also have a fourth role: time. A coordinate that doesn't just sit there, but flows.

---

## The Skeleton Pattern, Seen in Signatures

The four-normalizations table revealed that LayerNorm, RMSNorm, GroupNorm, and InstanceNorm share a skeleton. But how do you discover the skeleton in the first place? Not by reading a table. By writing the functions and noticing what changes.

Notice the skeletons already visible in these signatures—no body needed, just the coordinate names and reduction brackets:

```rust
fn softmax[j](x: [f32; ..b, j]) -> [f32; ..b, j];
fn layer_norm[f](x: [f32; ..b, f], gamma: [f32; f], beta: [f32; f]) -> [f32; ..b, f];
fn instance_norm[..s, c](x: [f32; ..b, c, ..s], gamma: [f32; c], beta: [f32; c]) -> [f32; ..b, c, ..s];
```

Look at only the signatures. Can you tell which one reduces over which coordinate? In `softmax[j]`, the coordinate parameter `j` tells you: normalize over `j`. In `layer_norm[f]`, `f` tells you: normalize over `f`. In `instance_norm[..s, c]`, both `..s` and `c` are in the signature—but which one is reduced? The answer depends on which coordinate is placed in the reduction bracket in the body. The signature alone can't tell you—it can only tell you which coordinates exist. The body tells you which ones are consumed.

Now compare to the positional versions:

```python
def softmax(logits, dim=-1): ...
def layer_norm(x, normalized_shape, ...): ...
def instance_norm(x, ...): ...
```

The positional signatures tell you even less. `softmax` has `dim`—a positional hint. `layer_norm` has `normalized_shape`—a shape hint, not a coordinate hint. `instance_norm` has no hint at all—you must read the body to know which dimensions are reduced.

What is happening when you read these signatures: you are asking "which coordinates are needed for this computation?" The ones in the brackets are the answer. The ones not in the brackets pass through. The signature is the skeleton's outline.

---

*The next time you write a function that takes a tensor and returns a tensor, write its coordinate signature in a comment before the body. Which coordinates survive? Which are consumed? Which broadcast? If you can't answer from the code alone, the signature is the place to start.*

---

### Stop and Think: Find the Skeletons in Your Code

You've seen the skeleton in LayerNorm, RMSNorm, GroupNorm, and InstanceNorm. But skeletons aren't limited to normalization. Every operation that follows a reduce-broadcast-elementwise pattern is a skeleton.

Every `mean(`, `sum(`, or `max(` followed by `keepdim=True` in a codebase is a reduction-statistic-broadcast pattern. Which coordinate is reduced? Which coordinate is the statistic broadcast over? If the `dim` argument is an integer, can the coordinate be named? If the answer is "it's whatever dimension is at that position," the skeleton's identity depends on layout.

Every `* gamma + beta` or `* scale + shift` is a broadcast-parameter suffix of a normalization skeleton. Which coordinates do `gamma` and `beta` broadcast over? Can they be named? If not, the broadcast is implicit.

Any two functions that share a skeleton—normalizing, pooling, computing a statistic—have Einlang signatures. Even without using Einlang, writing `fn name[consumed](x: [..batch, consumed]) -> [..batch, consumed]` as a comment reveals the skeleton. If the signatures match, the functions share a skeleton. If they don't, they serve different purposes and should have different signatures.

A normalization variant that normalizes over `batch` instead of `feature` changes `layer_norm[f]` to `layer_norm[batch]`. The signature change documents the architectural decision. A colleague reading the signature understands what is normalized over. In a positional API, changing `dim=-1` to `dim=0` silently shifts the coordinate—and hopes no other code depends on the output shape.

Every skeleton's forward pass is a reduce-broadcast-elementwise pattern. Every skeleton's backward pass is the Inversion Rule applied to that pattern. When you see the skeleton forward, you can predict its backward. The coordinate names are the thread connecting the two directions.
