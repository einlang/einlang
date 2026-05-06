---
layout: book
title: "Local Derivatives, Global Shape"
---

# Local Derivatives, Global Shape

You ran the training script overnight. In the morning, the loss sits at 2.3.
Flat. Eight hours, no movement. You check the shapes. Correct. You check the
learning rate. Reasonable. You check for dead ReLUs. None. You add gradient
clipping. Nothing changes.

On the third afternoon you find it. The weight gradient:

```rust
let dW[out, in] = sum[in](G[batch, in] * x[batch, out]);  // bug
```

That `sum[in]` should have been `sum[batch]`. One bracket. Every dimension
check passed. Every number was nonzero. The model descended a loss surface
that had nothing to do with the problem you meant to solve.

Chapters 7 and 8 gave you the scalar rule for pullbacks. A local derivative is
a single number: `x[batch, in]` for the linear layer, `B[k, j]` for one side of
a matmul. Those chapters kept the coordinate structure simple—one parameter, one
equation. This chapter adds the complication that makes real programs dangerous:
multiple terms, broadcast biases, and shared parameters that fan out across
batch and time. The local derivative is still a scalar. The gradient shape may
carry four sum coordinates. The mismatch between "the scalar is simple" and
"the gradient has four sums" is where PyTorch users learn to fear
`retain_graph=True`.

## The Bug That Shape Checks Miss

Build a tiny MLP layer:

```rust
let z[batch, out] =
    sum[in](x[batch, in] * W[out, in])
    + bias[out];
```

Now write the gradient for `W`:

```rust
let dW[out, in] = sum[batch](G[batch, out] * x[batch, in]);
```

A reader who sees `sum[batch]` for the first time might hesitate. Doesn't `W`
have only two coordinates? Why does its gradient involve `batch`, which `W`
doesn't even own? The answer: `W[out, in]` was used once per batch example.
One cell of `W` contributed to `z[0, out]`, `z[1, out]`, ..., `z[31, out]`
simultaneously. When sensitivity arrives at those 32 output cells, all 32
routes lead back to the same `W[out, in]`. The pullback must sum them.

This sum is invisible in a framework that computes `dW = x.T @ G`. The
dimensions work: `x` is `[batch, in]`, `G` is `[batch, out]`, the product gives
`[in, out]` (or `[out, in]` after a transpose you forgot). The code runs. The
loss curve descends. But if `batch` sum were replaced with `in` sum—if someone
wrote `sum[in](G[batch, in] * x[batch, out])`—the shapes would still match.
Every dimension check would pass. The model would silently learn nonsense.

The bug is not a wrong number. The bug is a wrong coordinate in a sum bracket.

## One Cell, All the Routes

Take the MLP layer with concrete sizes: `batch = 4`, `in = 3`, `out = 2`. Fix
one weight cell: `W[1, 2]`. Where does this cell appear in the forward pass?

```text
z[0, 1] reads W[1, 2] through x[0, 2]
z[1, 1] reads W[1, 2] through x[1, 2]
z[2, 1] reads W[1, 2] through x[2, 2]
z[3, 1] reads W[1, 2] through x[3, 2]
```

Four output cells, one parameter cell. The pullback:

```text
dW[1, 2] = G[0, 1] * x[0, 2]
         + G[1, 1] * x[1, 2]
         + G[2, 1] * x[2, 2]
         + G[3, 1] * x[3, 2]
```

That is `sum[batch](G[batch, 1] * x[batch, 2])`. Now lock `W[0, 1]`:

```text
z[0, 0] reads W[0, 1] through x[0, 1]
z[1, 0] reads W[0, 1] through x[1, 1]
z[2, 0] reads W[0, 1] through x[2, 1]
z[3, 0] reads W[0, 1] through x[3, 1]

dW[0, 1] = sum[batch](G[batch, 0] * x[batch, 1])
```

The same `sum[batch]` appears. It appears for every `[out, in]` pair because
every weight cell is shared across the entire batch. The local derivative at
each route is the corresponding `x` value, which is different for each batch
example. But the reduction coordinate is always `batch`.

## The Scalar Rule and the Shape Rule Are Different Things

Separate the two layers of the calculation:

```text
Layer 1 (scalar): d(loss)/d(W[out, in]) via one (batch, out) route
  = G[batch, out] * x[batch, in]

Layer 2 (collect): d(loss)/d(W[out, in]) summing all routes
  = sum[batch](G[batch, out] * x[batch, in])
```

Layer 1 is local calculus. It answers "for this input cell, through this output
cell, what number flows backward?" The answer is always a product of the
incoming sensitivity and a local derivative. In the linear layer, the local
derivative is `x[batch, in]`. In a squared operation `y = z * z` it would be
`2 * z`. In a sigmoid it would be `sigmoid(z) * (1 - sigmoid(z))`.

Layer 2 is shape reasoning. It answers "which output cells did this input cell
touch, and how do I sum the routes back?" The answer depends entirely on which
coordinates the input value was shared across in the forward program.

A programmer who confuses these two layers writes code that is numerically
active—every value is nonzero—but coordinate-wrong. A programmer who separates
them can debug a gradient by asking "which sum bracket did I forget?" rather
than "are the dimensions compatible?"

The distinction between local calculus and global shape is not a property of
the chain rule. The chain rule is always local — a product of two numbers
at a time. The global shape — which sums appear where — is determined by
which coordinates each value was shared across in the forward program. The
forward pass made sharing decisions. The backward pass collects the
consequences. The notation decides whether those sharing decisions are visible
in the source or must be reconstructed from the arithmetic.

## The Bias Pattern

In the same MLP layer, `bias[out]` is shared across `batch` but not across
`out`. Each `bias[0]` contributes to every `z[batch, 0]`. Each `bias[1]`
contributes to every `z[batch, 1]`. The pullback:

```rust
let dbias[out] = sum[batch](G[batch, out]);
```

No `in` coordinate appears. No `x` multiplier. The local derivative of `z` with
respect to `bias` is `1`, so the per-route contribution is just `G[batch, out]`.
The reduction coordinate is `batch`, the same as the weight pullback.

Now modify the layer to broadcast differently:

```rust
let z[batch, feature] =
    scale[feature] * x[batch, feature]
    + bias[1];  // shared across all features
```

The bias has no coordinates—it is a scalar broadcast to every `[batch, feature]`
cell. Its pullback:

```rust
let dbias = sum[batch, feature](G[batch, feature]);
```

Two reduction coordinates. The local derivative is still `1`. The shape rule
says "sum everything, because this scalar touched everything."

A reader might object: "But I never intended to share bias across features."
The gradient does not judge intent. It invoices what the forward program
declared. If `bias[1]` appeared in the forward expression, the gradient will
sum across every coordinate that `bias` did not own but that appeared in the
expression. The gradient is the shape invoice.

An invoice is only useful if you can read it. A positional gradient delivers
an invoice that says "summed axis 0, kept axes 1 and 2." The reader must know
which coordinate lived at axis 0 when the forward expression was written. A
named gradient delivers an invoice that says "summed batch, kept out and in."
The names are the invoice line items. The numbers are just the quantities.

## Debug Checklist

For a suspicious gradient, read the forward program in four mechanical passes:

```text
1. Which operation supplies the local scalar derivative?
   → The elementwise function at the forward expression's core.

2. Which coordinates does the denominator value own?
   → The coordinates declared on the parameter or variable being differentiated.

3. Which output coordinates did one denominator cell influence?
   → Every coordinate in the output shape that does not appear on the denominator.

4. Which of those routes must be summed back together?
   → All of them. Every coordinate the denominator does not own becomes a sum.
```

Apply it to the weight gradient `dW` above:

```text
1. z = sum[in](x * W), local derivative is x[batch, in]
2. W owns [out, in]
3. One W[out, in] influences all batch examples → batch is the fan-out coordinate
4. dW[out, in] = sum[batch](...)
```

Apply it to a convolutional weight `K[out_chan, in_chan, kh, kw]` applied to
input over `[batch, out_chan, h, w]`:

```text
1. Local derivative is the input patch
2. K owns [out_chan, in_chan, kh, kw]
3. One weight cell influences all batch, all spatial positions → fan-out: batch, h, w
4. dK = sum[batch, h, w](...)
```

Three reductions. The shape rule does not care how the weight was used—conv2d,
depthwise, grouped. It only cares which coordinates the weight does not own but
the output does.

## Reverse Diagnosis

When a gradient has already gone wrong, reverse the checklist:

```text
- Gradient has an unexpected sum → find the forward term that used the
  parameter across that coordinate.
- Gradient is missing a coordinate → the denominator owns that coordinate in
  the forward expression; check whether you intended it to.
- Broadcast parameter has surprising magnitude → check which forward
  coordinates it was shared across; the sum may be correct but the sharing may
  be wrong.
- Reduction coordinate appears in the gradient → the parameter address
  preserved that coordinate; the sum should not include it.
```

The blunt version: if you never named the coordinate in the denominator, the
gradient sum over that coordinate is the invoice for the omission.

## Why Scalar Tests Are Not Enough

A scalar test confirms `d(z^2)/dz = 2z`. It does not confirm that a bias was
shared across the intended coordinate. A numerical gradient check with finite
differences can verify a single `[batch, out]` entry, but it cannot verify that
the `sum[batch]` in the weight gradient collected exactly the batch dimension
and not the feature dimension by accident.

Numerical correctness and coordinate correctness are orthogonal. A gradient can
be numerically exact at every entry and still answer the wrong question because
the sum brackets are wrong. The shapes alone cannot detect this. The coordinate
names can.

The distinction between local calculus and global shape is not a property of the
mathematics. It is an artifact of the notation. The chain rule is always local —
a product of two numbers at a time. The global shape — which sums appear where —
is determined by which coordinates each value was shared across in the forward
program. When coordinates are named, the global shape is a consequence of set
subtraction: output coordinates minus denominator coordinates equals path
coordinates. When coordinates are positional integers, the same fact requires
tracing influence through every operation in the chain. The notation determines
what you can notice, and in the gradient case, it determines whether you see the
shape or must deduce it from numbers after the fact.

Numerical correctness and coordinate correctness are orthogonal properties of
a gradient program. The Hiding Law says: do not hide a fact that later
reasoning must recover. In the gradient case, the hidden fact is which
coordinate the value was shared across. The later reasoning is the pullback
sum. When the notation hides the sharing, the pullback must guess the axis —
and the guess can be numerically correct at every entry while still answering
the wrong question. A bug that lives in the coordinate names is a bug that
lives outside the reach of finite-difference checks.

## Try It

Broadcasts create the hardest gradient bugs because the forward pass looks
innocent. This exercise makes the invisible visible. Design a small layer with
two broadcasts:

```text
let y[time, batch, feature] =
    scale[time] * x[time, batch, feature]
    + bias[feature];
```

Use the four-pass checklist to derive `@loss/@scale` and `@loss/@bias`.
For each gradient, ask: what is the local scalar derivative (for `scale`:
`x[time, batch, feature]`; for `bias`: `1`), what coordinates does each
denominator own (`scale[time]`; `bias[feature]`), what output coordinates did
one denominator cell influence (every coordinate the denominator does not own:
for `scale[time]`, that is `batch` and `feature`; for `bias[feature]`, that is
`time` and `batch`), and what are the reduction coordinates
(`d_scale[time] = sum[batch, feature](...)`; `d_bias[feature] = sum[time, batch](...)`).

Verify: `scale[3]` is shared across all batch items and all features. Its
gradient receives sensitivity from every `[batch, feature]` pair at time step 3.
That is `sum[batch, feature]`. The coordinate accounting says it before any
numbers are computed.

Now trace a single parameter's gradient through a Transformer encoder. At each
step, write the coordinate audit. Which surviving coordinates are the gradient
address? Which are paths being reduced?

```text
Input: x[b, t, d]

1. Attention: scores[b, h, t_q, t_k] = sum[d](Q[b, h, t_q, d] * K[b, h, t_k, d])
   Gradient of Q: dQ[b, h, t_q, d] = sum[t_k](d_scores[b, h, t_q, t_k] * K[b, h, t_k, d])
   Surviving: {b, h, t_q, d}. Path (reduced): {t_k}.

2. Add+Norm: y[b, t, d] = layer_norm[d](x[b, t, d] + attn_out[b, t, d])
   The norm consumes d to compute statistics, but d survives in the output.
   Gradient carries d through — d is both consumed (by mean/std) and survived
   (as output coordinate).

3. FFN: z[b, t, d_out] = sum[d_in](y[b, t, d_in] * W[d_out, d_in]) + bias[d_out]
   Gradient of W: dW[d_out, d_in] = sum[b, t](d_z[b, t, d_out] * y[b, t, d_in])
   Surviving: {d_out, d_in}. Path: {b, t} — shared across entire batch and sequence.

4. Add+Norm again: same pattern as step 2.
```

Which gradient carries the most path coordinates to reduce? The FFN weight `W`
— it fans out across `batch` and `time` simultaneously, so its gradient must sum
over both. A single `W` cell influences every `[b, t]` position. The gradient is
a massive reduction that a positional autodiff system executes correctly but
silently. The coordinate audit makes the reduction visible as `sum[b, t]`.
Composing pullbacks is coordinate accounting at scale: each layer adds its own
survivors and paths, and the audit scales without new rules.

Next, derive the full gradient for this layer with batch norm, distinguishing
training mode from inference mode:

```text
// Training mode: mean and variance are computed per batch
fn batch_norm_train[feature](x: [f32; batch, feature]) -> [f32; batch, feature] {
    let mu[feature] = mean[batch](x[batch, feature]);
    let var[feature] = mean[batch]((x[batch, feature] - mu[feature]) ** 2);
    let y[batch, feature] =
        (x[batch, feature] - mu[feature]) / sqrt(var[feature] + eps)
        * gamma[feature] + beta[feature];
    y
}

// Inference mode: mean and variance are fixed (running statistics)
fn batch_norm_infer[feature](x: [f32; batch, feature]) -> [f32; batch, feature] {
    let y[batch, feature] =
        (x[batch, feature] - running_mean[feature]) / sqrt(running_var[feature] + eps)
        * gamma[feature] + beta[feature];
    y
}
```

For training mode, derive `@loss/@gamma` and `@loss/@beta`. Show that
`@loss/@gamma` sums over `batch` because `gamma[feature]` is shared across the
batch. This is the same pattern as the simple layer above.

For inference mode, `running_mean[feature]` and `running_var[feature]` are
constants. They receive no gradient. The coordinate audit changes: the mean and
variance are no longer functions of `x`. This means fewer intermediate terms in
the pullback, but the surviving coordinates are the same.

When the model switches from training to inference, the *shapes* of all gradients
remain identical. But the *paths* through intermediate values change — `mu` and
`var` are no longer computed from `x`, so the gradient of `x` has fewer terms.
A shape-only analysis sees no difference. A coordinate audit shows which
intermediate values (`mu`, `var`) were consumed and whether they carry gradient
paths.

**Line to keep:** scalar rules are local calculus; coordinate structure tells
where the value lands.

### Where This Leads

Part II is now complete. We have learned that hiding has consequences. Part I
taught us to notice when a coordinate role is omitted. Part II taught us that
the omission does not stay local — it propagates into the gradient. A forward
term that omits `batch` produces a gradient that sums over `batch`. A reduction
that consumes `class` produces a pullback that fans out over `class`. Every
sharing decision in the forward pass becomes a sum in the backward pass. The
notation either names the sharing, or the gradient inherits the silence.

But Part II had a limitation we did not name until now: all our programs were
static. A value depended on other values, but never on *earlier versions of
itself*. The gradient traced sensitivity backward through a fixed graph. We
never asked what happens when the graph itself has a direction — when `h[t]`
reads `h[t-1]`, and the backward pass must run in reverse time, and the compiler
must decide which time steps to store and which to recompute.

Part III introduces time as a coordinate with inherent direction. The audit
questions — survive, consume, omit — still apply. But now a new question joins
them: which values depend on which earlier values? A loop hides the answer in
mutable state. A recurrence names the index and makes the dependency an edge in
the source. That edge determines what the compiler can optimize, what the
autodiff engine can reverse, and what the reviewer can verify without tracing
into the loop body.

Chapter 10 begins with the simplest case: a time axis, a sliding window, and the
question of whether the future is allowed to read the past — or the past is
allowed to read the future.