---
layout: book
title: "Chapter 16: Dynamic Routing and Low-Rank Communication"
---

# Dynamic Routing and Low-Rank Communication

Standard attention was a dense communication story. A query position `i`
compared itself with every key position `j`, then gathered values from those
same `j` positions. The graph was large, but it was fixed.

Modern model code often hides a deeper kind of structure. Linear attention
compresses the communication graph through a feature bottleneck. Mixture of
Experts routes each token through a data-dependent expert. One changes the
shape of communication statically; the other changes it dynamically.

Both examples ask the same question as the rest of the book:

```text
Which fact will later reasoning need to recover?
```

For linear attention, the missing fact is where the bottleneck lives. For MoE,
the missing fact is how content chooses a route, and what shape that route
creates for capacity, load balance, and gradients.

The examples in this chapter use two levels of notation. When a block is
marked as source, it follows the language's ordinary shape: imports are
explicit, functions return their final expression, `if` is an expression, and
index domains appear inline as `i in 0..n`. When routing becomes too heavy to
write at every call site, the right move is still not new syntax. It is an
Einlang standard-library function with a visible reference definition, plus
optional compiler recognition for faster lowering.

For the feature map used below, a scalar helper can be written in ordinary
Einlang:

```rust
use std::math::exp;

pub fn phi(x: f32) -> f32 {
    (if x > 0.0 { x } else { exp(x) - 1.0 }) + 1.0
}
```

The parentheses are not the idea; the important part is that `exp` is imported
and the scalar branch is hidden without hiding any tensor coordinate.

## Dense Communication First

The dense attention pattern can be written as:

```text
scores[b, h, i, j] = sum[d](Q[b, h, i, d] * K[b, h, j, d])
weights[b, h, i, j] = softmax_over[j](scores[b, h, i, j])
out[b, h, i, v] = sum[j](weights[b, h, i, j] * V[b, h, j, v])
```

The roles are visible. Batch `b` and head `h` stay fixed. Query position `i`
asks. Key position `j` answers and is later consumed. Feature coordinate `d`
is consumed while computing scores. Value feature `v` survives in the output.

This graph is expensive because the score table has both `i` and `j`. Linear
attention tries to avoid materializing that table. MoE tries to avoid sending
every token through every expert. Neither move is only an implementation
detail. Each one changes the communication contract.

## Linear Attention Hides a Bottleneck

A common sketch of linear attention is:

```text
softmax(Q K^T) V  becomes  phi(Q) (phi(K)^T V)
```

The compact formula is useful, but it hides the coordinate story. Introduce a
feature map `phi` with a bottleneck coordinate `r`:

```rust
let q_projected[b in 0..batch, h in 0..heads, i in 0..q_len, r in 0..rank] =
    sum[d in 0..feature](Q[b, h, i, d] * W_q_phi[d, r]);

let k_projected[b in 0..batch, h in 0..heads, j in 0..kv_len, r in 0..rank] =
    sum[d in 0..feature](K[b, h, j, d] * W_k_phi[d, r]);

let Q_phi[b in 0..batch, h in 0..heads, i in 0..q_len, r in 0..rank] =
    phi(q_projected[b, h, i, r]);

let K_phi[b in 0..batch, h in 0..heads, j in 0..kv_len, r in 0..rank] =
    phi(k_projected[b, h, j, r]);
```

The exact feature map is a numerical design choice. The source-level fact is
that `r` is now a communication coordinate. It is not the query position and
not the key position. It is the space through which keys and values are
summarized before queries read them.

The key-value summary is:

```rust
let KV[b in 0..batch, h in 0..heads, r in 0..rank, v in 0..value_dim] =
    sum[j in 0..kv_len](K_phi[b, h, j, r] * V[b, h, j, v]);
```

The output then reads that summary:

```rust
let out[b in 0..batch, h in 0..heads, i in 0..q_len, v in 0..value_dim] =
    sum[r in 0..rank](Q_phi[b, h, i, r] * KV[b, h, r, v]);
```

The dense `i, j` table has disappeared. But it did not disappear for free. The
direct path from query `i` to key `j` now passes through `r`:

```text
i -> r -> j
```

That is the bottleneck. If `r` is too small, communication is compressed too
aggressively. If `r` is large, the approximation may be richer but more
expensive. A shape-only implementation may call this a feature dimension. The
coordinate reading says more: `r` is the rank of the communication plan.

## What `phi` Does Not Change

The feature map is often the first place where notation blurs roles. The map
changes feature values, but it should not silently change batch, head, or
position roles:

```text
Q[b, h, i, d]  ->  Q_phi[b, h, i, r]
K[b, h, j, d]  ->  K_phi[b, h, j, r]
```

The position coordinates survive. The old feature coordinate `d` is transformed
into the bottleneck coordinate `r`. The map is local in `b`, `h`, and position.
If a feature map accidentally mixes query positions, or shares random features
across the wrong head, the resulting tensor may still have a plausible shape.
The coordinate contract says where such mixing is allowed.

This is the same distinction as before. A function may hide scalar mechanics.
It should not hide which coordinate it transforms.

Put into a single function, the non-causal version can keep the syntax
discipline explicit:

```rust
use std::math::exp;

fn phi(x: f32) -> f32 {
    (if x > 0.0 { x } else { exp(x) - 1.0 }) + 1.0
}

pub fn linear_attention(
    Q: [f32; ?, ?, ?],
    K: [f32; ?, ?, ?],
    V: [f32; ?, ?, ?],
    W_phi: [f32; ?, ?]
) -> [f32; ?, ?, ?] {
    let batch = len(Q);
    let q_len = len(Q[0]);
    let kv_len = len(K[0]);
    let feature = len(Q[0][0]);
    let rank = len(W_phi[0]);
    let value_dim = len(V[0][0]);

    let q_projected[b in 0..batch, i in 0..q_len, r in 0..rank] =
        sum[d in 0..feature](Q[b, i, d] * W_phi[d, r]);

    let k_projected[b in 0..batch, j in 0..kv_len, r in 0..rank] =
        sum[d in 0..feature](K[b, j, d] * W_phi[d, r]);

    let Q_phi[b in 0..batch, i in 0..q_len, r in 0..rank] =
        phi(q_projected[b, i, r]);

    let K_phi[b in 0..batch, j in 0..kv_len, r in 0..rank] =
        phi(k_projected[b, j, r]);

    let KV[b in 0..batch, r in 0..rank, v in 0..value_dim] =
        sum[j in 0..kv_len](K_phi[b, j, r] * V[b, j, v]);

    let C[b in 0..batch, i in 0..q_len, v in 0..value_dim] =
        sum[r in 0..rank](Q_phi[b, i, r] * KV[b, r, v]);

    C
}
```

This is still a sketch of a particular kernel choice, not a standard-library
promise. Its point is that the communication rank `r` is an explicit coordinate
rather than a hidden implementation parameter.

## Causal Linear Attention Is a Recurrence

Causal masking breaks the simple global summary. A query at time `t` may read
only keys at positions `j <= t`. The key-value summary becomes time-indexed:

```rust
let KV_prefix[b in 0..batch, h in 0..heads, t in 0..seq_len, r in 0..rank, v in 0..value_dim] =
    sum[j in 0..t + 1](K_phi[b, h, j, r] * V[b, h, j, v]);
```

The output is:

```text
out[b, h, t, v] =
    sum[r](Q_phi[b, h, t, r] * KV_prefix[b, h, t, r, v])
```

This is no longer just a matrix multiplication reordered for speed. It is a
prefix recurrence over time. The same object can be written with an explicit
state:

```rust
let state[b in 0..batch, h in 0..heads, 0, r in 0..rank, v in 0..value_dim] =
    K_phi[b, h, 0, r] * V[b, h, 0, v];

let state[b in 0..batch, h in 0..heads, t in 1..seq_len, r in 0..rank, v in 0..value_dim] =
    state[b, h, t - 1, r, v] +
    K_phi[b, h, t, r] * V[b, h, t, v];
```

Now the connection to recurrence is visible. Causal linear attention carries a
state whose coordinates are `[r, v]` for each batch and head. The time
coordinate moves forward, and the state accumulates key-value information.
That fact matters for storage, streaming inference, and reverse-mode
differentiation. If it is hidden inside a scan helper, later analysis has to
recover it.

The inclusive base case above means `state[..., 0, ...]` already contains the
contribution from time zero. If a library chooses an empty-prefix convention
instead, that boundary choice should be visible in the recurrence range rather
than buried in a scan helper.

## Gradients Through the Bottleneck

The bottleneck coordinate `r` is also where sensitivities mix. In the forward
pass, all key positions contribute to `KV[b, h, r, v]`:

```text
KV[b, h, r, v] = sum[j](K_phi[b, h, j, r] * V[b, h, j, v])
```

In the backward pass, a single `K_phi[b, h, j, r]` receives sensitivity through
every value feature `v` that used the same `r`, and every query position `i`
that later read the summary. The dense attention path has not vanished; it has
been routed through a lower-dimensional address.

That is the practical diagnostic:

```text
Where did i and j stop talking directly?
Which coordinate replaced the direct communication path?
Which gradients are forced to meet at that coordinate?
```

For linear attention, the answer is `r`.

## MoE Hides a Dynamic Coordinate

Mixture of Experts changes the problem. The communication graph is not only
compressed; it is chosen by the data.

A gate computes scores for each token and expert:

```rust
let gate_score[b in 0..batch, t in 0..seq_len, e in 0..num_experts] =
    sum[d in 0..feature](x[b, t, d] * W_gate[d, e]);
```

A router chooses one or more experts:

```text
chosen[b, t] = top1[e](gate_score[b, t, e])
```

The coordinate `e` is not just another static axis in a dense tensor. For each
token `[b, t]`, the program chooses a route. The result `chosen[b, t]` is a
content-dependent label. It says which expert this token will visit.

That makes MoE different from the earlier chapters. Coordinates such as `i`,
`j`, and `k` ranged over static domains. The expert domain is static, but the
expert identity used by a token is selected by a function of the token. The
compiler can still reason about the role only if the route is named:

```text
route[b, t] = chosen expert for token [b, t]
```

Without that name, the routing decision becomes a scatter implementation
detail. With the name, later code can ask which tokens went where.

For top-one routing, a syntax-faithful dense sketch can use `argmax` and keep
the hard selection visible:

```rust
use std::array::argmax;

let gate_max[b in 0..batch, t in 0..seq_len] =
    max[e in 0..num_experts](gate_score[b, t, e]);

let gate_exp[b in 0..batch, t in 0..seq_len, e in 0..num_experts] =
    exp(gate_score[b, t, e] - gate_max[b, t]);

let gate_sum[b in 0..batch, t in 0..seq_len] =
    sum[e in 0..num_experts](gate_exp[b, t, e]);

let gate_prob[b in 0..batch, t in 0..seq_len, e in 0..num_experts] =
    gate_exp[b, t, e] / gate_sum[b, t];

let route[b in 0..batch, t in 0..seq_len] =
    argmax(gate_prob[b, t]);

let hard_onehot[b in 0..batch, t in 0..seq_len, e in 0..num_experts] =
    if e == route[b, t] { 1.0 } else { 0.0 };
```

Top-k routing, sorting, and capacity assignment are stronger operations than
this dense sketch. They should be exposed as library functions with stated
coordinate contracts, not smuggled in as a fake slice syntax. The important
constraint is that the library function should itself have an Einlang reference
definition. A backend may later recognize and lower it to a scatter, prefix
count, or fused routing kernel, but the source-level contract should not depend
on an unexplained operation outside the language.

## Capacity Slots Are Hidden Storage

Efficient MoE implementations usually batch tokens by expert. That requires a
static shape. The dynamic route is turned into an expert coordinate and an
expert-local capacity slot:

```text
slot[b, t] = position of token [b, t] inside route[b, t]
keep[b, t] = slot[b, t] < capacity
```

The dispatched tensor has coordinates:

```text
expert_input[e, c, d]
```

where `e` is the expert and `c` is the capacity slot inside that expert. A
token writes to:

```text
expert_input[route[b, t], slot[b, t], d] = x[b, t, d]
```

This is not merely a layout trick. The capacity coordinate `c` is a compiler
or runtime allocation of space inside each expert. If too many tokens choose
the same expert, some tokens may not fit. The mask `keep[b, t]` is therefore a
semantic witness:

```text
keep[b, t] tells whether token [b, t] survived routing
```

A program that drops overflow tokens without naming `keep` has hidden a fact
that debugging will almost certainly need. Load imbalance, accuracy loss, and
training instability all become harder to explain when the discarded
coordinate has no name.

Current Einlang syntax can describe a top-one capacity router as an ordinary
function. The policy choices are visible: ties follow `argmax`, tokens are
ordered lexicographically by `[b, t]`, and overflow is represented by `keep`.

```rust
use std::array::argmax;

pub fn route_top1_with_capacity(gate_prob: [f32; ?, ?, ?], capacity: i32) {
    let batch = len(gate_prob);
    let seq_len = len(gate_prob[0]);

    let route[b in 0..batch, t in 0..seq_len] =
        argmax(gate_prob[b, t]);

    let slot[b in 0..batch, t in 0..seq_len] =
        sum[bb in 0..batch, tt in 0..seq_len](
            if (bb < b || (bb == b && tt < t)) &&
               route[bb, tt] == route[b, t] {
                1
            } else {
                0
            }
        );

    let keep[b in 0..batch, t in 0..seq_len] =
        slot[b, t] < capacity;

    (route, slot, keep)
}
```

This is not the fastest possible router. That is fine. It is the reference
meaning. A production standard library can ship this definition and let a
compiler pass replace the recognized pattern with an efficient implementation,
while preserving the visible `route`, `slot`, and `keep` contract.

## Expert Computation and Return

Once tokens are dispatched, each expert runs its own transformation:

```text
expert_output[e, c, o] =
    expert_ffn[e](expert_input[e, c, d])
```

The result is then gathered back to token coordinates:

```text
y[b, t, o] =
    if keep[b, t] {
        expert_output[route[b, t], slot[b, t], o]
    } else {
        dropped_value[b, t, o]
    }
```

The round trip is a coordinate map with a dynamic address. Token `[b, t]`
becomes `(route[b, t], slot[b, t])`, is processed, and returns to `[b, t]`.
The expert coordinate is not a dense communication axis like attention's `j`.
It is a routing label chosen per token.

This is why MoE needs names. Without `route`, `slot`, and `keep`, the program
looks like scatter, gather, reshape, and mask manipulation. With the names, it
becomes a dynamic communication graph.

## Load Balance Is a Coordinate Reduction

MoE implementations usually add an auxiliary loss so tokens do not all choose
the same expert. The loss is easy to get subtly wrong because it is a statistic
over token and expert coordinates.

One readable coordinate form is:

```text
fraction[e] =
    mean[b, t](one_hot(route[b, t], e))

mean_prob[e] =
    mean[b, t](gate_prob[b, t, e])

balance_loss =
    mean[e](fraction[e] * mean_prob[e])
```

Using only reductions and the hard mask from the top-one sketch, the same
statistics can be made more explicit:

```rust
let tokens_per_expert[e in 0..num_experts] =
    sum[b in 0..batch, t in 0..seq_len](hard_onehot[b, t, e]);

let total_tokens = (batch * seq_len) as f32;

let fraction[e in 0..num_experts] =
    tokens_per_expert[e] / total_tokens;

let mean_prob[e in 0..num_experts] =
    sum[b in 0..batch, t in 0..seq_len](gate_prob[b, t, e]) / total_tokens;

let balance_loss =
    sum[e in 0..num_experts](fraction[e] * mean_prob[e]) * (num_experts as f32);
```

The first statistic counts hard assignments. The second statistic measures the
gate's soft preference. Both are addressed by expert `e`, and both reduce over
token coordinates `[b, t]`. The final loss reduces over `e`.

A positional implementation may compute the same values with `one_hot`, `sum`,
and `mean`. The coordinate form states what the statistics mean. It also makes
two common mistakes visible: reducing over experts when the formula meant to
reduce over tokens, or using pre-softmax logits where the balancing rule
intended probabilities.

## The Discrete Route and the Gradient

Routing introduces a hard question for autodiff. The forward pass uses a
discrete choice:

```text
route[b, t] = top1[e](gate_score[b, t, e])
```

That choice is not an ordinary differentiable coordinate map. Some systems use
a straight-through estimator, treating the forward route as discrete while
letting backward sensitivity flow as if a softer gate had been used.

The important point for this book is not to endorse one estimator. It is to
name the fiction. The current `@fn` form can attach a tangent rule to ordinary
functions, which is enough for many smooth helper functions. A hard forward
choice with a soft surrogate backward path is a stronger contract: it should be
named at the source level, either as a standard-library function with a stated
autodiff rule or, in a richer language, as an explicit custom pullback. A source
form could make the contract visible:

```text
route[b, t] = ste_top1[e](gate_prob[b, t, e])
```

Read that as a warning label. In the forward pass, `e` is consumed to produce a
discrete route. In the backward pass, the implementation uses a surrogate rule
that sends sensitivity through `gate_prob[b, t, e]`. That is not the same as a
smooth coordinate transformation. It is an estimator with a coordinate
contract, and the source should not pretend otherwise.

This is much clearer than hiding the rule inside a detach expression. The
notation says where the discrete decision happens and where the gradient is
pretending to pass.

## Two Stress Tests, One Principle

Linear attention and MoE look different. Linear attention compresses a dense
communication graph through `r`. MoE routes tokens through a dynamic expert
label and a capacity slot. But both extend the same discipline:

```text
name the coordinate where communication changes shape
```

For linear attention, name the bottleneck coordinate and the prefix state. For
MoE, name the route, the slot, the keep mask, and the expert statistics. Those
names are not surface documentation. They are the facts later analysis will
need for gradients, storage, load balance, streaming, and debugging.

The book began with a reshape whose shape was right but whose reason was
hidden. It ends here with more modern failures: an approximation whose
bottleneck is hidden, and a routing decision whose topology is hidden. The
same test still works:

```text
Would a future reader or compiler need to recover this coordinate fact?
```

If yes, it deserves a place in the source.

## Beyond Tensor Code

The same question appears outside neural networks. A SQL query optimizer has
relations, keys, join columns, filters, and projections. If a query plan hides
which column is the join key and which column is merely carried along, later
optimization can still be fast but much harder to explain. A good plan can
change join order, choose indexes, or push filters down. It should not erase
the semantic role of the key that makes the join correct.

A map projection has a similar split. Latitude and longitude can be packed into
pixels, tiles, or geohashes. The storage coordinate may change many times, but
the source facts still matter: which direction is north, which axis wraps, what
distance is preserved, and where distortion is introduced. A coordinate system
does not prevent all mistakes. It makes the cost of each projection visible.

These examples are not tensor programs, but they rhyme with the same rule. The
implementation may choose a plan, a layout, an index, or a tile. The source
should preserve the facts that make those choices meaningful.

## Try It

For linear attention, reduce the bottleneck rank `r` in your mental model and
ask which query-key conversations must now share an address. For MoE, set the
capacity too low and name the first token that becomes `keep = false`. The
point is not to compute a number; it is to identify the coordinate where the
communication plan changed shape.

**Line to keep:** name the coordinate where communication changes shape.
