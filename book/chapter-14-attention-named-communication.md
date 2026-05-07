---
layout: book
title: "Attention as Named Communication"
---

# Attention as Named Communication

> "Programs must be written for people to read, and only incidentally for
> machines to execute."
>
> — Harold Abelson and Gerald Jay Sussman, preface to the first edition of
> *Structure and Interpretation of Computer Programs* (1985)

You are staring at a bug. The Transformer trains fine on short sequences but
diverges silently on long ones. Loss curves look normal. Shapes check out.
But somewhere in the attention block, the values being gathered are from the
wrong positions—the softmax weights point at key position `j`, but the gather
reads from query position `i`. In matrix-land, that's a transpose three layers
up. In coordinate-land, it's a one-letter mistake: `V[i]` instead of `V[j]`.

The tempting story is that attention is just matrix multiplication with a
softmax in the middle. That story is useful for kernels, but too small for
debugging: it forgets who asks, who answers, and which value comes home.

## Scores: Who Asks, Who Answers

```rust
let scores[b, i, j] =
    sum[d](Q[b, i, d] * K[b, j, d]) * scale;
```

Three coordinate roles, three jobs. `i` is the query position—the position doing
the asking. `j` is the key position—the position being compared against. `d` is
the feature coordinate consumed by the dot product. The result keeps `b`, `i`,
and `j`: for each batch item, every query position receives a compatibility
score for every key position.

Read one score:

```text
scores[b, 4, 9]
```

This is how compatible query position `4` is with key position `9` in batch
item `b`. The feature coordinate `d` is gone—it was the common language these
two positions used to compare themselves. The score does not contain a value
vector. It is only a relation between two positions.

The difference from `Q K^T` is not correctness. `Q K^T` is correct. The
difference is that `i` names the asker and `j` names the answerer. When `i` and
`j` share the same extent (self-attention on a sequence of length `N`), a shape
checker sees two axes of size `N` and calls them compatible. The source names
say which one is being updated and which one is being listened to.

## Weights: Which Coordinate Is Normalized

Between scores and values sits the softmax:

```rust
let weights[b, i, j] = softmax[j](scores[b, i, j]);
```

For each fixed `b` and `i`, the weights over all key positions `j` sum to one.
The key coordinate is the one being normalized. The query coordinate is not
reduced—it identifies which row of the attention table is being normalized.

Now break it deliberately:

```rust
let bad_weights[b, i, j] = softmax[i](scores[b, i, j]);
```

This normalizes over queries instead of keys. For each key position `j`, the
query positions distribute. The shapes are identical: both produce `[b, i, j]`.
A shape checker sees no error. A named-coordinate checker objects: the model
claimed to normalize over the coordinate the output was supposed to preserve as
the asking position.

## Gather: Which Coordinate Is Read

After normalization, the output gathers values:

```rust
let output[b, i, d] =
    sum[j](weights[b, i, j] * V[b, j, d]);
```

`j` is consumed. `i` survives as the position being updated. The feature
coordinate `d` returns, now as the value feature being carried home.

Read one point:

```text
output[b, 4, d]
```

The value vector for query position `4`. It is built by scanning all key/value
positions `j`, weighting each `V[b, j, d]` by how strongly query `4` attended
to position `j`. The sentence has grammar: query asks, key answers, value is
carried home.

Now break it:

```rust
let broken[b, i, d] =
    sum[j](weights[b, i, j] * V[b, i, d]);
```

The gather reads `V[b, i, d]` instead of `V[b, j, d]`. Every term in the sum
carries the same value vector—the one at the query position. The weights no
longer choose among positions. The communication pattern collapses. Shape:
identical. Names: caught.

```
   Attention as Communication Protocol

   Scores: i asks, j answers
   +--------------------------------------------------+
   | scores[b,i,j] = sum[d](Q[b,i,d] * K[b,j,d])     |
   | i: query position (survives)                     |
   | j: key position (survives)                       |
   | d: feature (consumed in dot product)             |
   +--------------------------------------------------+
              |
              v
   Weights: softmax over j (the key axis)
   +--------------------------------------------------+
   | weights[b,i,j] = softmax[j](scores[b,i,j])      |
   | j normalized: for each i, sum_j weights = 1     |
   +--------------------------------------------------+
              |
              v
   Gather: V values brought to i
   +--------------------------------------------------+
   | output[b,i,d] = sum[j](weights[b,i,j] * V[b,j,d])|
   | j consumed, i survives, d returns               |
   | i asks --> j answers --> V[j,d] carried home     |
   +--------------------------------------------------+

   Attention = named communication, not matrix multiplication.
```

We are not making attention prettier. We are making the communication graph
checkable — by the compiler, by the reviewer, by the person debugging at 3 AM.
The shapes work regardless. The question is not whether attention works. It is
whether the notation names the communication protocol or buries it in a
transpose.

## Masking: Direction in One Line

Causal attention adds a mask—query `i` may listen only to keys at or before
itself:

```rust
let causal_scores[b, i, j] =
    if j <= i { scores[b, i, j] } else { -infinity };
```

The mask is not just a boolean matrix of the right shape. Its meaning depends
entirely on which coordinate is query and which is key. Swap them and the
direction of time reverses:

```text
j <= i   past and present keys are visible
i <= j   future keys become visible instead
```

Both produce the same shape. One is causal, the other is anticipatory. The
difference lives in the coordinate role, and a named source makes it local.

## Multi-Head: One More Fixed Coordinate

Multi-head attention splits the feature dimension into a head coordinate:

```rust
let Q_h[b, h, i, d] = ...;   // reshape: feature → (head, head_dim)
let K_h[b, h, j, d] = ...;
let V_h[b, h, j, d] = ...;

let scores[b, h, i, j] = sum[d](Q_h[b, h, i, d] * K_h[b, h, j, d]);
let weights[b, h, i, j] = softmax[j](scores[b, h, i, j]);
let out[b, h, i, d] = sum[j](weights[b, h, i, j] * V_h[b, h, j, d]);
```

`h` survives through every step. Heads do not communicate inside the attention
pattern. Each head has its own query, its own keys, its own values, its own
attention weights. The absence of `sum[h]` is the contract.

After attention, a projection may mix heads:

```rust
let mixed[b, i, feature] =
    sum[h, d](out[b, h, i, d] * W_proj[feature, h, d]);
```

That is a separate operation with its own coordinate relation—`h` is consumed
here, not during attention. A reader can now ask local questions:

```text
Does softmax normalize over key position j, separately for each head h?
Does the output preserve query position i?
Does the final projection mix heads, or only repack them?
```

These are model questions. They become easier to ask when the code gives names
to the roles.

A shape checker can tell you that two tensors have compatible ranks. It cannot
tell you that heads are isolated, that queries never mix, that `h` survives
the softmax but dies in the output projection. Those are architectural
promises. When the notation has a place for them, they become compiler checks.
When the notation has no place for them, they become oral tradition — passed
from author to reviewer, remembered for six months, and then forgotten.

## A Concrete Trace

Set concrete dimensions for maximum visibility. Two batch items, three token
positions, four model features, two heads, head dimension of three:

```text
batch = 2,  t = 3,  d_model = 4,  h = 2,  d_head = 3
```

Input `x[b, t, d_model]` with actual numbers:

```text
x[0, 0, :] = [1.0, 0.0, 1.0, 0.0]    (batch 0, token 0)
x[0, 1, :] = [0.0, 1.0, 0.0, 1.0]    (batch 0, token 1)
x[0, 2, :] = [1.0, 1.0, 0.0, 0.0]    (batch 0, token 2)
x[1, 0, :] = [0.0, 0.0, 1.0, 1.0]    (batch 1, token 0)
x[1, 1, :] = [1.0, 0.0, 0.0, 1.0]    (batch 1, token 1)
x[1, 2, :] = [0.0, 1.0, 1.0, 0.0]    (batch 1, token 2)
```

**Step 1 — Project to Q, K, V.** Each head gets its own query, key, and value
through a learned projection. `d_model = 4` goes to `h * d_head = 2 * 3 = 6`:

```rust
let Q[b, h, i, d_head] = sum[d_model](x[b, i, d_model] * W_q[h, d_head, d_model]);
let K[b, h, j, d_head] = sum[d_model](x[b, j, d_model] * W_k[h, d_head, d_model]);
let V[b, h, j, d_head] = sum[d_model](x[b, j, d_model] * W_v[h, d_head, d_model]);
```

Coordinate audit: `d_model` consumed, `b, h, i/j, d_head` survive. Each head
has independent Q, K, V — `h` appears in all three.

Give W_q concrete values for head 0:

```text
W_q[0, :, :]:  row 0 = [ 0.5,  0.0,  0.0,  0.5]
               row 1 = [ 0.0,  0.5, -0.5,  0.0]
               row 2 = [-0.5,  0.0,  0.5,  0.0]
```

Trace `Q[0, 0, 0, :]` (batch 0, token 0, head 0):

```text
Q[0,0,0,0] = 1.0*0.5 + 0.0*0.0 + 1.0*0.0 + 0.0*0.5 = 0.5
Q[0,0,0,1] = 1.0*0.0 + 0.0*0.5 + 1.0*(-0.5) + 0.0*0.0 = -0.5
Q[0,0,0,2] = 1.0*(-0.5) + 0.0*0.0 + 1.0*0.5 + 0.0*0.0 = 0.0
```

So `Q[0, 0, 0, :] = [0.5, -0.5, 0.0]`. For this trace, use the same projection
for K and V (simplified for visibility):

```text
Q[0, 0, :, :] (head 0):
  token 0: [ 0.5, -0.5,  0.0]
  token 1: [ 0.5,  0.5,  0.0]
  token 2: [ 0.0,  0.0,  0.5]
K and V are the same.
```

**Step 2 — Compute scores.** For each head, each query position asks each key
position how compatible they are. The scale factor is `1 / sqrt(3) ≈ 0.577`:

```rust
let scores[b, h, i, j] = sum[d_head](Q[b, h, i, d_head] * K[b, h, j, d_head]) * 0.577;
```

Coordinate audit: `d_head` consumed inside the sum. `b, h, i, j` survive.

Trace the full 3×3 score matrix for batch 0, head 0:

```text
scores[0,0,0,0] = (0.5*0.5 + (-0.5)*(-0.5) + 0.0*0.0) * 0.577 = 0.5 * 0.577 = 0.289
scores[0,0,0,1] = (0.5*0.5 + (-0.5)*0.5 + 0.0*0.0) * 0.577 = 0.0
scores[0,0,0,2] = (0.5*0.0 + (-0.5)*0.0 + 0.0*0.5) * 0.577 = 0.0

scores[0,0,1,0] = (0.5*0.5 + 0.5*(-0.5) + 0.0*0.0) * 0.577 = 0.0
scores[0,0,1,1] = (0.5*0.5 + 0.5*0.5 + 0.0*0.0) * 0.577 = 0.5 * 0.577 = 0.289
scores[0,0,1,2] = (0.5*0.0 + 0.5*0.0 + 0.0*0.5) * 0.577 = 0.0

scores[0,0,2,0] = (0.0*0.5 + 0.0*(-0.5) + 0.5*0.0) * 0.577 = 0.0
scores[0,0,2,1] = (0.0*0.5 + 0.0*0.5 + 0.5*0.0) * 0.577 = 0.0
scores[0,0,2,2] = (0.0*0.0 + 0.0*0.0 + 0.5*0.5) * 0.577 = 0.25 * 0.577 = 0.144
```

The 3×3 score matrix:

```text
          j=0    j=1    j=2
i=0  [  0.289  0.000  0.000  ]
i=1  [  0.000  0.289  0.000  ]
i=2  [  0.000  0.000  0.144  ]
```

The diagonal dominates because Q and K share the projection — each token is
most compatible with itself. `i` names the asker (rows), `j` names the
answerer (columns).

**Step 3 — Softmax over key positions.** For each query position, normalize
attention weights over all key positions:

```rust
let weights[b, h, i, j] = softmax[j](scores[b, h, i, j]);
```

For row `i=0` (scores `[0.289, 0.0, 0.0]`):

```text
exp(0.289)=1.335, exp(0.0)=1.0, exp(0.0)=1.0, sum=3.335
weights[0,0,0,:] = [0.400, 0.300, 0.300]   (sum = 1.0)
```

For row `i=1`: `[0.300, 0.400, 0.300]`. For row `i=2` (scores `[0, 0, 0.144]`):

```text
exp(0)=1.0, exp(0)=1.0, exp(0.144)=1.155, sum=3.155
weights[0,0,2,:] = [0.317, 0.317, 0.366]   (sum = 1.0)
```

Every row sums to 1. The diagonal is emphasized but attention spreads to all
positions.

**Step 4 — Gather values.** For each query position, collect value vectors
from all key positions, weighted by attention:

```rust
let out[b, h, i, d_head] = sum[j](weights[b, h, i, j] * V[b, h, j, d_head]);
```

Coordinate audit: `j` consumed by the sum. `b, h, i, d_head` survive.

Trace `out[0, 0, 1, :]` (batch 0, head 0, query position 1):

```text
out[0,0,1,0] = 0.300*0.5 + 0.400*0.5 + 0.300*0.0 = 0.150 + 0.200 + 0.000 = 0.350
out[0,0,1,1] = 0.300*(-0.5) + 0.400*0.5 + 0.300*0.0 = -0.150 + 0.200 + 0.000 = 0.050
out[0,0,1,2] = 0.300*0.0 + 0.400*0.0 + 0.300*0.5 = 0.000 + 0.000 + 0.150 = 0.150
```

`out[0,0,1,:] = [0.350, 0.050, 0.150]`. Query position 1 has gathered a
weighted blend: 30% from token 0's value, 40% from token 1's value, 30% from
token 2's value.

```
   Value Gather: Query i collects from all key positions j

   weights[0,0,1,:]         V[0,0,:,:]                   contribution
   ─────────────────────────────────────────────────────────────────
   j=0: 0.300          ×    [ 0.5, -0.5,  0.0]     =    [ 0.150, -0.150,  0.000]
   j=1: 0.400          ×    [ 0.5,  0.5,  0.0]     =    [ 0.200,  0.200,  0.000]
   j=2: 0.300          ×    [ 0.0,  0.0,  0.5]     =    [ 0.000,  0.000,  0.150]
                                                          ─────────────────────
   out[0,0,1,:] = sum over j:                            [ 0.350,  0.050,  0.150]
```

**Step 5 — Merge heads and project output.** After attention, merge heads back
into the model dimension. This is the only place heads mix:

```rust
let output[b, i, d_model] =
    sum[h, d_head](out[b, h, i, d_head] * W_out[d_model, h, d_head]);
```

Coordinate audit: `h` and `d_head` consumed. `b, i, d_model` survive. Inside
the attention mechanism, heads were isolated. The output projection is where
they combine.

**The full audit table** of shape-compatible mistakes:

```text
Mistake                             Shape      Still Runs?   Named Check
────────────────────────────────────────────────────────────────────────────
softmax[i] instead of softmax[j]    [b,h,i,j]  yes            role violation
V[b,h,i,d] instead of V[b,h,j,d]    [b,h,i,d]  yes            j absent from V read
Q[b,h,j,d] * K[b,h,i,d] (swap)      [b,h,i,j]  yes            role swap (i_count==j_count)
j <= i → i <= j (causal mask)       [b,h,i,j]  yes            time direction reversed
sum[h] in attention scores          [b,h,i,j]  yes            head leak
omit h from W_q but not from K/V    [b,h,i,j]  yes            head sharing contract
```

Every mistake compiles. Every mistake produces a loss curve that descends.
Every mistake is invisible to a shape checker. Every mistake is visible to a
named-coordinate audit.

Now commit the most dangerous mistake: replace `V[b, h, j, d_head]` with
`V[b, h, i, d_head]` in the gather. Recompute `out[0, 0, 1, :]`:

```text
out_broken[0,0,1,0] = V[0,0,1,0] * (0.300+0.400+0.300) = 0.5 * 1.0 = 0.5
out_broken[0,0,1,1] = V[0,0,1,1] * 1.0 = 0.5 * 1.0 = 0.5
out_broken[0,0,1,2] = V[0,0,1,2] * 1.0 = 0.0 * 1.0 = 0.0
```

The broken output is exactly `V[0, 0, 1, :] = [0.5, 0.5, 0.0]` — the value at
the query position, unchanged by attention. The sum over `j` was vacuous
because the term did not read `j`. The shapes are identical: both are vectors
of length 3. The loss descends either way. The difference is that one version
attends and the other copies.

The compiler check: under `sum[j]`, the value `V` must be indexed by `j`.
Otherwise the sum over `j` is vacuous — the term does not read the coordinate
being reduced. A named-coordinate compiler can report: "term under `sum[j]`
does not read coordinate `j`. Did you mean `V[b, h, j, d_head]`?"

## What the Positional Version Hides

A typical PyTorch attention forward pass—reshape, transpose, matmul, softmax,
transpose, reshape—produces correct shapes. But seven conventions are invisible
in the code:

1. The input axis order `[batch, time, feature]` is a convention, not a contract
2. The reshape `.reshape(B, T, H, dh)` splits *feature* into *head* and *within-head*—but the reshape itself does not say which axis was feature
3. The transpose `.transpose(1, 2)` is driven by matmul layout, not by meaning
4. `k.transpose(-2, -1)` swaps the last two axes—but which axes those *are* depends on the entire preceding chain
5. `softmax(dim=-1)` normalizes over the last axis, which the reader must remember is the key position
6. The final transpose-reshape chain `.transpose(1, 2).reshape(B, T, D)` assumes the split is exactly reversible
7. If `Q` and `K` arrive in different axis orders, the shapes might still align by accident while the meaning is scrambled

None of these are bugs by themselves. A vigilant programmer can hold them all.
The argument is not that PyTorch attention is broken. It is that the code relies
on the reader—and the writer, six months later—to remember which number means
which role at each operation.

Halide's central insight was that image processing pipelines hide two kinds of
decisions in the same code: what to compute, and how to schedule it. Named
coordinates make a neighboring separation: the dataflow (which values combine)
is visible in both positional and named notation, but the axis contract (which
axes carry meaning, which are consumed, which survive) is visible only in the
named version. When a positional program reshapes, transposes, splits, and
flattens, the dataflow survives—the arithmetic is correct—but the axis contract
is reduced to a shape tuple. The shape says how many. It does not say which of
them were `class` and which were `time`.

This is the book's thesis, tested on the hardest operation in the standard
architecture. Notation determines what you can notice. When the notation names
the axis contract, a reviewer can ask whether the contract is correct. When the
notation buries the contract in a shape tuple, the reviewer can only ask whether
the shapes multiply. Both questions matter. Only the first catches the bug where
the shapes multiply correctly and the communication is wrong.

## What These Ideas Share

The attention example combines every earlier chapter:

```text
sum[d]         consumed feature coordinate (dot product)
softmax[j]     coordinate normalization (distribution over keys)
sum[j]         consumed position coordinate (value gather)
b              batch that survives everywhere
i              position being updated
h              head identity, fixed within attention
```

The operation is not simple, but the style of reading is the same as for dot
product or broadcasting. That is the promise of a small notation: the questions
do not change when the example grows.

The practical test is simple: after reading an attention implementation, can
you identify the query coordinate, the key coordinate, the value feature
coordinate, the head coordinate, and the batch coordinate? If the answer
requires a diagram outside the code, the source is asking memory to do too much.

The vocabulary is small. `i` asks, `j` answers, `d` carries, `h` separates.
Every chapter has used the same reading discipline — which coordinates survive,
which are consumed, which are omitted. The operation has grown from a dot
product to a Transformer block, but the questions have not changed. That is
the promise of a small notation: the reading discipline scales because the
coordinate roles scale, even when the formulas do not.

<div class="pause" markdown="1">
**Pause.** Open your own attention implementation — or the one you copied from
a library. Find the line that computes the attention output. Answer three
questions: (1) Which coordinate is `i` (the asker) and which is `j` (the
answerer)? (2) Does the gather read `V` at index `i` or index `j`? (3) If
`query_len == key_len`, would a swap of `i` and `j` change the shape? If the
answer to (3) is no, you have a silent bug waiting to happen. The
communication protocol — i asks, j answers, d carries — is the contract. If
the source does not name the participants, the contract is a convention. And
conventions rot.
</div>

## Try It

Take the attention score formula:

```rust
let scores[b, h, i, j] = sum[d](Q[b, h, i, d] * K[b, h, j, d]);
```

The consumption coordinate is `d` — it disappears into the sum. The two
surviving position coordinates are `i` (the asker) and `j` (the answerer). Now
swap `i` and `j` in the query and key:

```rust
let swapped[b, h, i, j] = sum[d](Q[b, h, j, d] * K[b, h, i, d]);
```

The result still has shape `[b, h, i, j]`. The numbers are transposed but
nonzero, and the loss descends. But the communication pattern has changed: query
position `i` now reads key position `i`'s query, and key position `j` reads
query position `j`'s key. The roles of asker and answerer are swapped across the
diagonal. The coordinate condition that makes this swap invisible to shape checks
is `i_count == j_count` — which is exactly the condition that holds in
self-attention, the pattern used in every Transformer ever deployed.

Now consider three attention variants. Each changes exactly one coordinate
contract, not the entire formula.

Variant A is cross-attention. Queries come from the decoder, keys and values
from the encoder. The query sequence length differs from the key sequence
length, so `i` and `j` have different extents:

```rust
let scores[b, h, i_dec, j_enc] =
    sum[d](Q_dec[b, h, i_dec, d] * K_enc[b, h, j_enc, d]);
let out[b, h, i_dec, d] =
    sum[j_enc](weights[b, h, i_dec, j_enc] * V_enc[b, h, j_enc, d]);
```

In positional notation, the cross-attention formula is identical to the
self-attention formula — the only difference is that `i_count != j_count` at
runtime. In named notation, `i_dec` and `j_enc` differ in the source. A reader
can see that cross-attention is being used without tracing the shapes of `Q_dec`
and `K_enc` back to their origins.

Variant B uses asymmetric features. Queries and keys use `d_qk` for comparison;
values use `d_v` for the carried information. The two feature coordinates have
different extents:

```rust
let Q[b, h, i, d_qk] = ...;
let K[b, h, j, d_qk] = ...;
let V[b, h, j, d_v] = ...;
let scores[b, h, i, j] = sum[d_qk](Q[b, h, i, d_qk] * K[b, h, j, d_qk]);
let out[b, h, i, d_v] = sum[j](weights[b, h, i, j] * V[b, h, j, d_v]);
```

In positional code, a reader must remember that `d_qk` and `d_v` are different
extents despite both being the "feature" dimension. A projection somewhere
changed the size. The named version makes the two feature coordinates visible
and distinct.

Variant C shares key-value features. Keys and values share the same feature
coordinate `d` but the weight matrices project them differently:

```rust
let K[b, h, j, d] = sum[d_in](x[b, j, d_in] * W_k[d, d_in]);
let V[b, h, j, d] = sum[d_in](x[b, j, d_in] * W_v[d, d_in]);
```

Here the error of writing `V[b, h, j, d_qk]` when the value actually uses `d_v`
would be caught by the compiler before the first forward pass. In a positional
implementation, Variant A would crash if `i_count != j_count` in a
shape-sensitive operation. Variant B would crash if `d_qk != d_v` in a
shape-sensitive operation. Variant C would silently run because `d` is the same
extent for keys and values, and the weight matrices map `d_in → d` with
compatible shapes. The named version catches the role confusion before it
becomes a runtime crash or a silent wrong answer.

Next, review a pull request that adds multi-query attention: all heads share a
single set of keys and values but keep separate queries. The author writes the
positional code correctly, but you ask them to write it with named coordinates
to make the sharing contract visible. Standard multi-head attention looks like
this:

```rust
let scores[b, h, i, j] = sum[d](Q[b, h, i, d] * K[b, h, j, d]);
let weights[b, h, i, j] = softmax[j](scores[b, h, i, j]);
let out[b, h, i, d] = sum[j](weights[b, h, i, j] * V[b, h, j, d]);
```

Every coordinate is indexed by `h`. Each head has its own Q, K, V. In
multi-query attention, `K` and `V` omit the `h` coordinate — they are shared
across heads:

```rust
let K_shared[b, j, d] = sum[d_in](x[b, j, d_in] * W_k_shared[d, d_in]);
let V_shared[b, j, d] = sum[d_in](x[b, j, d_in] * W_v_shared[d, d_in]);

let scores[b, h, i, j] = sum[d](Q[b, h, i, d] * K_shared[b, j, d]);
let weights[b, h, i, j] = softmax[j](scores[b, h, i, j]);
let out[b, h, i, d] = sum[j](weights[b, h, i, j] * V_shared[b, j, d]);
```

The omission of `h` from `K_shared` and `V_shared` is the contract. Every head
reads the same keys and values. `h` appears in `Q` but not in `K_shared` or
`V_shared`. Under the broadcasting rule from Chapter 4, `K_shared[b, j, d]` is
reused for every `h` — that is not a bug, it is the architecture.

But what if the author accidentally writes `K_shared` with an `h` coordinate?

```rust
// BUG: K_shared was supposed to omit h, but someone added it back
let K_bug[b, h, j, d] = ...;
let V_bug[b, h, j, d] = ...;
```

The attention formula still works. The shapes are identical to standard
multi-head attention. But every head now has its own separate keys and values.
The "multi-query" label in the architecture description is false. The parameter
count is inflated. The memory savings never materialize. Nothing in the
positional code reports that `K` was supposed to omit the head coordinate. The
coordinate audit that distinguishes the two versions is stark:

```text
Multi-head:    Q[b, h, i, d], K[b, h, j, d], V[b, h, j, d]  → h is present in all three
Multi-query:   Q[b, h, i, d], K[b, j, d], V[b, j, d]        → h is absent from K and V
```

The presence or absence of `h` in the coordinate set of `K` is the only
source-level fact that says which architecture was intended. A shape checker can
verify that the tensor has the right rank, but it cannot verify which
coordinates are supposed to be missing — named coordinates make the sharing
contract visible in the coordinate set.

The same attention formula

```rust
let out[b, h, i, d] = sum[j](weights[b, h, i, j] * V[b, h, j, d]);
```

produces standard multi-head attention when `V` owns `h` and multi-query
attention when `V` omits `h`. The formula does not change. The contract changes.
The contract lives in the coordinate signature of the operand, not in the body
of the attention computation. Named coordinates move the contract from the
architecture diagram into the source where it can be checked.

## A Full Encoder Block

The Transformer encoder chains attention with a feed-forward network, each
followed by residual addition and layer normalization. Write the coordinate
audit for every sublayer:

```text
Input: x[b, t, d_model]

1. Multi-head attention:
   Q/K/V projection: d_model consumed, h and d_head created
   Scores: d_head consumed, i and j survive
   Softmax: j normalized
   Gather: j consumed, d_head survives
   Output projection: h and d_head consumed, d_model returns
   Result: attn_out[b, t, d_model]

2. Add & Norm:
   y[b, t, d_model] = layer_norm[d_model](x[b, t, d_model] + attn_out[b, t, d_model])
   d_model consumed by mean/std statistics, survives as output coordinate.

3. FFN:
   z[b, t, d_ff] = sum[d_model](y[b, t, d_model] * W1[d_ff, d_model]) + bias1[d_ff]
   act[b, t, d_ff] = relu(z[b, t, d_ff])
   out_ffn[b, t, d_model] = sum[d_ff](act[b, t, d_ff] * W2[d_model, d_ff]) + bias2[d_model]
   d_ff is intermediate, consumed by the second projection. W2 fans out across
   [b, t] — its gradient sums over both batch and time.

4. Add & Norm: same as step 2.
```

For the FFN weight W1, the gradient sums over `[b, t]` because `W1[d_ff,
d_model]` is shared across all batch items and all token positions. One weight
cell influences every `[b, t]` position. The coordinate audit is the same
pattern applied four times in sequence. The names don't change. The rules don't
change. Only the number of coordinates changes.

**Line to keep:** attention is not matrix multiplication; it is a communication
protocol between named roles — `i` asks, `j` answers, `h` separates, `d_head`
carries — and every role deserves a name in the source.

### Where This Leads

Attention is a communication protocol. `i` asks, `j` answers, `d` carries the
vocabulary, `h` selects the channel. The shapes work either way — `Q K^T V` is
correct regardless of what you call the axes. But the protocol lives in the names.
When you trace a bug through an attention block, you are not asking whether the
shapes multiply. You are asking which position gathered from which other position.
The notation either gives you a name to check or asks you to deduce it from a
transpose three layers up.

A conversation where nobody has a name still works. But when something goes wrong,
you cannot ask "who said that?" You can only point at a position and hope.

Chapter 15 draws the line: what should a notation refuse to hide? After fourteen
chapters of watching the Hiding Law play out — in reshape, in broadcast, in
gradient, in recurrence, in attention — the question turns inward. What does it
mean for a notation to have integrity? When should a coordinate name, once
introduced, be impossible to erase without the programmer writing the deletion
explicitly?