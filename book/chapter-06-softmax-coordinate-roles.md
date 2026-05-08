---
layout: book
title: "Softmax Has Three Coordinate Roles"
---

# Softmax Has Three Coordinate Roles

> "Information is the resolution of uncertainty."
>
> — Claude Shannon, "A Mathematical Theory of Communication" (1948)

Softmax. You have written it a hundred times. Each time, you passed it one
number: `dim`. That single number hid three distinct coordinate jobs behind it.
This chapter names all three—the stability scan, the normalization denominator,
and the surviving output—and in doing so, catches the bug that shipped for six
weeks in production.

## The Bug That Trained for Six Weeks

It is month three of the project. You own the classifier. The model shipped
behind an API, and every Tuesday you check the dashboard: accuracy, latency,
calibration error. The first two are fine. The third has been drifting upward
since the sprint-17 data pipeline refactor. No one connected the two events,
because the refactor was "just" a layout change—`[batch, class]` became `[class,
batch]`—and every line of model code still runs.

```python
# Sprint 16: logits shape [batch=128, class=10]
probs = torch.softmax(logits, dim=-1)
# dim=-1 is class. Each row sums to 1.0. Correct.

# Sprint 17: data pipeline transposed. logits shape [class=10, batch=128]
probs = torch.softmax(logits, dim=-1)
# dim=-1 is now batch. Each row sums to 1.0. Still correct-looking.
```

The shapes are fine. The normalization invariant (rows sum to 1) holds in both
cases. The loss—cross-entropy—goes down in both cases. Cross-entropy only cares
that `probs[correct_class]` is high. Whether the normalization competes against
other classes or other examples, a gradient still flows toward the correct
answer.

At 2 AM you finally run the calibration report by hand on a single batch and
realize: the probabilities sum to 1.0 across examples, not across classes. The
number 0.7 has meant something different for six weeks. No shape checker caught
it. No gradient check caught it. The notation never recorded which axis was the
distribution.

But calibration error asks a different question: does `probs[class]=0.7` mean
the model is actually correct 70% of the time? After sprint 17, the answer is
no. The model's confidence scores are now computed relative to other examples in
the batch, not relative to other classes for the same example. The number `0.7`
has a different meaning. The shape is right. The meaning is wrong.

Here is the concrete difference with three examples and three classes:

```text
Logits (correct layout [batch, class]):
         cat  dog  fish
  img1: [2.0, 1.0, 0.1]  →  softmax over class → [0.66, 0.24, 0.10]
  img2: [0.5, 2.0, 0.3]  →  softmax over class → [0.14, 0.63, 0.23]
  img3: [0.1, 0.2, 2.0]  →  softmax over class → [0.10, 0.12, 0.78]

Logits (swapped layout [class, batch]):
  img1: [2.0, 1.0, 0.1]  →  softmax over batch → [0.66, 0.24, 0.10]
  img2: [0.5, 2.0, 0.3]  →  softmax over batch → [0.14, 0.63, 0.23]
  img3: [0.1, 0.2, 2.0]  →  softmax over batch → [0.10, 0.12, 0.78]
```

When `batch_size == num_classes`, the numbers are identical—each cell happens to
be the softmax of its column instead of its row, but the column and row have the
same length. The probabilities sum to 1 either way. The loss curve is identical.
The bug is mathematically invisible for square inputs.

Now write softmax with named coordinates:

```rust
// Correct: normalize each example across classes
let probs[batch, class] = softmax[class](logits[batch, class]);

// Bug: normalize across batch for each class
let probs[class, batch] = softmax[batch](logits[class, batch]);
```

The second line is syntactically valid. But a reader who sees `softmax[batch]`
next to a result declared `[class, batch]` can ask: am I normalizing the
examples against each other? The coordinate inside the bracket names the
competition. `softmax[class]` says "classes compete." `softmax[batch]` says
"examples compete." The distinction lives in the source, not in a comment that
drifts out of date.

## One Axis, Three Jobs

Here is softmax, written as the stable formula you learned but broken into its
coordinate parts:

```rust
let probs[b, j] = softmax[j](x[b, j]);
```

One line. Three jobs. The coordinate `b` (batch) is fixed—each example is its
own normalization problem. The coordinate `j` (feature/class) is the one being
normalized. Inside the implementation, there is a third role: the coordinate
that scans all features to build the denominator.

Write the same formula with all roles explicit:

```rust
let m[b]        = max[q](x[b, q]);           // (1) scan for maximum
let e[b, k]     = exp(x[b, k] - m[b]);       // (2) stabilize, exponentiate
let z[b]        = sum[k](e[b, k]);            // (3) build denominator
let probs[b, j] = e[b, j] / z[b];             // (4) normalize each output
```

The input `x[b, j]` appears once. But the output `probs[b, j]` depends on
every `x[b, k]` through the denominator and every `x[b, q]` through the
maximum. Three distinct scopes, all over the same feature range:

| Role | Name | Scope | Consumed? |
|------|------|-------|-----------|
| Stabilizing reference | `q` | `max[q]` | Yes, leaves |
| Denominator scan | `k` | `sum[k]` | Yes, leaves |
| Output coordinate | `j` | `probs[b, j]` | No, survives |

The input and output have the same shape `[b, j]`. A shape-only story says
"nothing changed." A coordinate story says "every `j`-output was computed by
inspecting every `k`-input in the same row, stabilized by every `q`-input."

```
   Softmax: One Axis, Three Jobs

   Input x[b, *] across feature range j for one batch b:
   +------+------+------+------+
   | x[0] | x[1] | x[2] | x[3] |  j = 0, 1, 2, 3
   +------+------+------+------+
      |      |      |      |
      +------+------+------+
             |
   +---------v---------+
   | max[q](x) -> m[b] |  q: scan for max reference, consumed
   +-------------------+
             |
   +---------v---------+
   | exp(x - m) -> e   |  e[b, k] for every k in feature range
   +-------------------+
             |
   +---------v---------+
   | sum[k](e) -> z[b] |  k: denominator scan, consumed
   +-------------------+
             |
   +---------v---------+
   | e[b,j] / z[b]     |  -> probs[b, j]  j: output survivor
   +-------------------+

   q, k, j all range over the same feature axis.
   dim=-1 sees one; coordinates name three distinct scopes.
   Each role maps to a distinct gradient term (Chapter 7).
```

In PyTorch, all three roles collapse into a single argument:

```python
probs = torch.softmax(logits, dim=-1)
```

In JAX, the same:

```python
probs = jax.nn.softmax(logits, axis=-1)
```

Both produce correct numbers. Neither distinguishes `q` (the stability scan),
`k` (the denominator sum), and `j` (the output survivor). The integer `-1` is
three different coordinate jobs, flattened into one positional convention. The
gradient that flows backward through `dim=-1` carries three distinct terms to
three distinct targets. The autodiff engine handles this correctly — it traces
through `max`, `exp`, `sum`, and `div` separately. But the reader who writes
`dim=-1` and the reviewer who reads `dim=-1` have no place to verify that the
max, the sum, and the output are all operating on the same set of positions.
The coordinate names `q`, `k`, `j` give each role its own receipt.
```

Pick one cell of the gradient. `@loss / @logits[2, 5]`. The index `5` appears in
three places in the forward formula: as `q` (the query position in the
numerator), as `k` (the key position in the sum), and as `j` (the output
position that survives). Each appearance produces a different term in the
pullback. `q=5` contributes through the max selection — a sparse term, nonzero
only when `5` was the maximum. `k=5` contributes through the denominator sum —
a dense term, every `j` position's denominator includes this `k`. `j=5`
contributes directly — the output at position `5` receives gradient from the
loss. Three gradient paths through the same integer `5`. A single `dim=-1`
collapses them into one number. Three letters — `q`, `k`, `j` — leave them
separate on the page.

## Why Three Roles and Not One

You might ask: if `q`, `k`, and `j` all range over the same set of class
indices, why give them three different names? Why not use `j` everywhere and
let the reader infer the scope from context?

Because scope changes the gradient.

When you write `max[q](x[b, q])`, you are making a claim: the result `m[b]`
does not depend on which `q` achieved the maximum—only on the maximum value
itself. The gradient of `m[b]` with respect to `x[b, q]` is 1 for the argmax
index and 0 for all others. The coordinate `q` is consumed.

When you write `sum[k](e[b, k])`, you are making a different claim: the result
`z[b]` depends on every `k` equally. The gradient of `z[b]` with respect to
`e[b, k]` is 1 for every `k`. The coordinate `k` is consumed, but the gradient
pattern is completely different from the `max` consumption.

When you write `probs[b, j] = e[b, j] / z[b]`, you are making a third claim:
the output has a coordinate `j` that survives. Its gradient with respect to
`e[b, j]` is `1/z[b]` for the matching `j` and zero for non-matching because
`e[b, j]` only appears in the numerator for that specific `j`.

If you used `j` for all three roles, the source would look like this:

```rust
// DO NOT WRITE THIS — j has three different gradient behaviors
let m[b]       = max[j](x[b, j]);
let e[b, j]    = exp(x[b, j] - m[b]);
let z[b]       = sum[j](e[b, j]);
let probs[b, j] = e[b, j] / z[b];
```

Lines 1 and 3 both say `[j]` but `j` means "consumed by max" in line 1 and
"consumed by sum" in line 3. Line 2 and line 4 both say `e[b, j]` but the `j`
in line 2 is "the coordinate being exponentiated" while the `j` in line 4 is
"the output survivor." The compiler can disambiguate these, but the reader
cannot do it at a glance.

Three names—`q`, `k`, `j`—make the three gradient behaviors visually distinct.
`q` is consumed by max (sparse gradient). `k` is consumed by sum (dense
gradient). `j` survives (diagonal gradient in the numerator, dense through the
denominator). The letters are not cosmetic. They are gradient contracts.

A positional API calls all three `dim=-1` and relies on the reader to know
which gradient pattern applies at each step. The named form surfaces the
distinction at the source level.

This is the thesis at its most practical. Three different gradient
contracts, all hidden behind the same integer. When a reader sees
`dim=-1` in a softmax implementation, they must reconstruct which of the
three roles that integer refers to at each line. When they see `q`, `k`,
or `j`, the gradient contract is visible in the letter. The notation does
not make softmax simpler. It makes the complexity inspectable.

## What This Means for Gradients

Because `probs[b, j]` depends on every `x[b, k]`, the softmax Jacobian is not
diagonal. A small change to `x[b, 1]` can change `probs[b, 0]`, `probs[b, 1]`,
and `probs[b, 3]`. All three shift because the denominator shifted.

Make this concrete for one batch member with three classes:

```text
x = [2.0, 1.0, 0.1]

Step 1: m = max(x) = 2.0
Step 2: e = exp(x - m) = [exp(0), exp(-1), exp(-1.9)]
        = [1.0, 0.368, 0.150]
Step 3: z = sum(e) = 1.0 + 0.368 + 0.150 = 1.518
Step 4: probs = e / z = [0.659, 0.242, 0.099]

Now perturb x[1] by +0.01: x = [2.0, 1.01, 0.1]
Step 1: m = max(x) = 2.0  (unchanged — max still at index 0)
Step 2: e = [1.0, exp(-0.99), exp(-1.9)]
        = [1.0, 0.372, 0.150]
Step 3: z = 1.522
Step 4: probs = [0.657, 0.244, 0.099]

probs[0] changed: 0.659 → 0.657  (even though x[0] was not perturbed!)
probs[1] changed: 0.242 → 0.244  (expected — x[1] was perturbed)
probs[2] changed: 0.099 → 0.099  (almost unchanged — but not exactly)
```

`probs[0]` shifted because `z` changed, and `z` depends on every `x[k]`. This
is the off-diagonal Jacobian: `@probs[b,0] / @x[b,1]` is not zero. It is
`-probs[0] * probs[1]`, a quantity that exists purely because the denominator
sums over all `k`.

A sigmoid is different:

```rust
let s[b, j] = 1 / (1 + exp(-x[b, j]));
```

Here `s[b, j]` reads only `x[b, j]`. A change to `x[b, 1]` changes `s[b, 1]`
and nothing else. Same input shape, same output shape. Completely different
dependency graph. The Jacobian of sigmoid IS diagonal.

The coordinate names reveal the difference at the source level. In the softmax
formula, `k` and `q` appear because other positions in the row are consulted.
In the sigmoid formula, only `j` appears. The scopes are the contract.

Now consider the batch. Does `probs[b, j]` depend on `x[b2, k]` for `b2 != b`?

```rust
let probs[b, j] = softmax[j](x[b, j]);
```

`softmax[j]` normalizes within the surviving coordinates—`b` is a survivor, so
the normalization is per-batch-member. The bracket says `[j]`, not `[b, j]`. A
positional API buries this guarantee inside a `dim` argument; the named form
puts it in the bracket where the reader can verify it.

Later chapters will need this distinction. When Chapter 7 traces a gradient
back through softmax, the pullback will contain terms summed over `j` and terms
that reference the full row. The three roles you name here become three paths
in the backward pass.

## The Softmax Audit: Four Questions Before You Write `dim`

Before you type `softmax(logits, dim=-1)`, answer four questions. If you cannot
answer them from the code alone, the `dim` is hiding something.

| Question | Positional answer | Named answer |
|---|---|---|
| Which coordinate is being normalized? | `dim=-1` (whatever that is today) | `softmax[class]` — the bracket names it |
| Which coordinates define independent problems? | Everything not `dim=-1` | The survivors on the left: `probs[batch, class]` |
| Does the normalized coordinate survive? | Yes, but invisible—output has same shape as input | `class` on the left of `probs` confirms survival |
| Is the Jacobian diagonal? | Depends on whether `dim` is also scanned elsewhere | `k` in `sum[k]` reveals the off-diagonal scan |

Three coordinate relationships, one integer argument. `dim=-1` is not wrong. It is
silent about which of the three you meant — and the three have different
gradient contracts, different Jacobian structures, and different silent failure
modes. The coordinate `j` survives into `probs[b, j]`. The coordinate `k` is consumed
by `sum[k]`. The coordinate `q` is consumed by `max[q]`. Three different
gradient patterns emerge from three different scope assignments. A single `dim`
collapses all three.

The audit table is not a checklist for ceremony. It is a tool for
noticing what the positional notation hides. A reviewer who can answer
all four questions from the code alone is reading a program that states
its coordinate contracts. A reviewer who must answer from memory is
reading a program that buries them. The difference is not a matter of
style. When the softmax bug from the opening shipped to production, it
shipped because the answer to question one -- "which coordinate is being
normalized?" -- had silently changed from `class` to `batch`, and
nothing in the source said so.

## The Three-Role Template

This pattern repeats across many normalization functions. The template is:

```text
1. Name the surviving coordinate (the one on the left)
2. Name the reduction coordinate (the one inside sum/max/mean)
3. Name the broadcast parameters (the ones that omit survivors)
```

Apply it to LayerNorm:

```rust
// LayerNorm over feature coordinate f
let mu[b, t]       = mean[f](x[b, t, f]);           // (1) scan f
let sigma2[b, t]   = mean[f]((x[b, t, f] - mu[b, t]) ** 2);  // (2) scan f
let normed[b, t, f] = (x[b, t, f] - mu[b, t]) / sqrt(sigma2[b, t] + eps);
let y[b, t, f]     = normed[b, t, f] * gamma[f] + beta[f];  // (3) broadcast params
```

Surviving coordinates: `b`, `t`, `f`. Reduction coordinate: `f` (local to
`mean[...]`). Broadcast parameters: `gamma[f]`, `beta[f]` (omit `b` and `t`).

The same `f` plays both survivor and local roles. It survives into the output
because each feature gets a normalized value. It is local to `mean[f]` because
computing the mean requires scanning all features. The compiler distinguishes
the two scopes automatically. The source makes the distinction visible.

Now apply the same template to GroupNorm:

```rust
// GroupNorm: channels split into groups
// g = group index, c_in_group = channel within group
let mu[b, g, i, j]        = mean[c_in_group, i2, j2](
    x[b, g, c_in_group, i + i2, j + j2]
);
let sigma2[b, g, i, j]    = mean[c_in_group, i2, j2](
    (x[b, g, c_in_group, i + i2, j + j2] - mu[b, g, i, j]) ** 2
);
let y[b, g, c_in_group, i, j] = (x[b, g, c_in_group, i, j] - mu[b, g, i, j])
    / sqrt(sigma2[b, g, i, j] + eps) * gamma[g, c_in_group] + beta[g, c_in_group];
```

The reduction consumes `c_in_group`, `i2`, `j2`. The broadcast parameters
`gamma`, `beta` omit `b`, `i`, `j`. The pattern is identical to softmax: name
the survivors, name the locals, name the broadcasters. Scale to any number of
coordinates without new rules.

RMSNorm is the simplest case—no mean centering, just rescaling by the root mean
square:

```rust
// RMSNorm over feature coordinate f
let rms[b, t]    = sqrt(mean[f](x[b, t, f] ** 2));
let normed[b, t, f] = x[b, t, f] / (rms[b, t] + eps);
let y[b, t, f]   = normed[b, t, f] * gamma[f];
```

Survivors: `b`, `t`, `f`. Local: `f` (consumed by `mean[f]`). Broadcast: `gamma[f]`
(omits `b`, `t`). The same `f` in three scopes: the `f` being squared, the `f`
being averaged away, and the `f` that survives. RMSNorm is softmax stripped to
its bare structure—a single reduction coordinate playing two local roles, with
one broadcast parameter.

Here is the comparison across four normalization functions:

| Function | Reduction coords | Broadcast coords | Survivors |
|---|---|---|---|
| Softmax | `q` (max), `k` (sum) | `m[b]` (broadcast `j`) | `b`, `j` |
| LayerNorm | `f` (mean ×2) | `gamma[f]`, `beta[f]` | `b`, `t`, `f` |
| RMSNorm | `f` (mean) | `gamma[f]` | `b`, `t`, `f` |
| GroupNorm | `c_in_group`, `i2`, `j2` | `gamma[g, c_in_group]`, `beta[g, c_in_group]` | `b`, `g`, `c_in_group`, `i`, `j` |

Every normalization follows the same coordinate skeleton: reduce to get
statistics, broadcast statistics back, apply elementwise. The skeleton is
invisible in positional notation because `dim` looks the same whether the
reduction is for mean, for variance, or for softmax denominator. The names make
the skeleton a template you can check.

This is the Hiding Law at scale. A single normalization function may
involve four different coordinate relationships. In a positional API, all
four collapse to a single `dim` argument whose meaning shifts with the
surrounding layout. In a coordinate API, each relationship gets a name --
survivor, local, broadcast -- and the names follow the coordinates through
the entire function. The difference is not verbosity. It is whether a
reviewer can verify that the broadcast coordinate in LayerNorm matches the
broadcast coordinate in the gradient without reconstructing both from
positional offsets.

Three relationships folded into one integer. The integer fits in a register. The
relationships require a sentence. A notation that has only room for the integer
has already decided what you do not need to know — and made that decision before
you arrived.

## The Square Matrix Test

Here is a concrete test for any reduction-based formula:

```text
1. Set all coordinate extents equal (make everything square).
2. Swap two coordinates in the input.
3. Ask: does the formula still compile?
4. Ask: does the formula still mean the same thing?
```

Apply it to softmax:

```rust
// Original: softmax over class
let probs[batch, class] = softmax[class](logits[batch, class]);

// Square case: batch_size = num_classes = 128
// Swap input: logits is [class, batch]
let probs[batch, class] = softmax[class](logits[class, batch]);
```

The formula compiles. But `class` is now the *row* coordinate of `logits`, not
the *column* coordinate. The softmax scans across rows. A positional API would
compile `torch.softmax(logits, dim=1)` and silently normalize along whatever
axis happens to be in position 1 after the swap.

The named version does not prevent the swap from happening. But it makes the
consequence visible: the coordinate inside `softmax[...]` no longer matches the
role the reader expects for `class` in the input layout. The reader can see
the change and ask the question.

## Softmax and Broadcasting Together

Chapter 4 showed broadcasting. Chapter 5 showed reduction. Softmax uses both
in one formula:

```rust
let probs[b, j] = softmax[j](x[b, j]);
```

The broadcast: `m[b]` (the maximum) is reused for every `j` in the subtraction
`x[b, j] - m[b]`. The coordinate `j` is absent from `m[b]`. The reduction:
`sum[k](e[b, k])` consumes `k` and leaves `z[b]`. The broadcast and the
reduction operate over the same feature range but in different scopes.

Now ask: what happens if you add a constant to every logit?

```text
x' = x + c   (c added to every class score for a given batch member)
m' = max(x') = max(x) + c
e' = exp(x' - m') = exp(x - m) = e
z' = sum(e') = sum(e) = z
probs' = e' / z' = e / z = probs
```

The probabilities are unchanged. This is the softmax invariance: shifting all
logits by the same constant does not change the output. In coordinate terms:

```rust
// This is an identity:
let probs[b, j] = softmax[j](x[b, j] + offset[b]);
//                                      ^^^^^^^^ omits j — broadcast
// Softmax consumes j internally, so any term that omits j disappears.
```

The `offset[b]` omits `j`, so it is broadcast across all `j` inside the softmax.
The softmax consumes `j` via `max[q]` (the shift is absorbed into `m[b]`), so
the offset has no effect on the output. The invariance is a direct consequence
of the coordinate structure: a term that is independent of the normalized
coordinate cannot affect the normalized output.

This is why the chapter belongs here—at the midpoint of the book. Softmax
forces you to use everything from Chapters 2 through 5 at once. Named axes.
Coordinate maps (the offsets in the subtraction). Broadcasting (maximum reuse).
Reduction (denominator). And now the extra demand: distinguishing the same
coordinate in different scopes. An invariance that falls out of the coordinate
structure without a separate proof.

> If you can read softmax fluently, you can read anything in this book.

### Part I Stage Report

Part I has been about a single question: what can the compiler *not* see?
Chapter 1 showed that a reshape drops the coordinate story — the shape
trace survives, the role claim does not. Chapter 2 showed that axis
positions do not capture roles: two axes can swap and the shape looks
identical. Chapter 3 showed that transpose, flatten, and depth\_to\_space
are coordinate maps whose meaning evaporates when written as integer
permutations. Chapter 4 showed that broadcasting hides omission — a
missing coordinate is the program's way of saying "this term does not
depend on this axis," and a notation that buries the omission buries the
fact a reviewer needs. Chapter 5 showed that reduction consumes a
coordinate and leaves the survivors to carry forward, and that the
ledger of survivors and locals is the contract every later line depends
on. This chapter — softmax — demanded all of those tools at once: named
axes, broadcasting, reduction, and the new demand of distinguishing the
same coordinate in three different scopes.

Three tools now live in your audit kit: survive, consume, omit. Three
questions you can ask of any tensor line, in any framework. The compiler
sees shapes. You now see roles. But noticing is only the surface. Part I
did not touch what happens *because* of the hiding. When a coordinate
role is hidden in the forward pass, the gradient inherits the silence.
The autodiff engine becomes a reader that cannot read. Part II turns the
audit onto automatic differentiation. The tools do not change. The
direction does. A forward expression already knows which input cells
influence which output cells. The backward pass is just collecting
sensitivity along those routes — and if the forward notation named the
routes, the gradient is mechanical. If it hid them, the gradient is a
guess.

> **Pause.** Open your own codebase and find a softmax call. Any softmax call.
> Now answer three questions from the code alone: (1) Which coordinate is being
> normalized? (2) Which coordinates survive? (3) Which coordinate appears in
> both a `sum[...]` and as a survivor? If you cannot answer all three from the
> source without checking upstream variable names or documentation, the `dim`
> argument is hiding something. Write the answers down. The three roles — `q`,
> `k`, `j` — are the map. Chapter 7 will run them backward through the chain
> rule.

## Try It

You have used softmax a thousand times. You may have never audited it. Let's fix
that. Take three softmax calls from common model code:

```python
# 1. Classifier
probs = torch.softmax(logits, dim=-1)   # logits: [batch, class]

# 2. Attention weights
attn = torch.softmax(scores, dim=-1)     # scores: [batch, head, query, key]

# 3. Sequence model
probs = torch.softmax(logits, dim=1)     # logits: [batch, time, vocab]
```

For each, name the normalized coordinate and the surviving coordinates. Write
the Einlang equivalent. The classifier normalizes `class` and keeps `batch`.
The attention normalizes `key` and keeps `batch`, `head`, `query`. The sequence
model normalizes `vocab` and keeps `batch`, `time`. Three different bracketed
coordinates, all hidden behind `dim=-1` or `dim=1`.

Now take the softmax from your own attention implementation. Which coordinate is
being normalized? Which coordinates survive? Which coordinate appears in BOTH a
`sum[...]` and as a survivor? That coordinate is consumed to compute the
denominator but survives because each position gets its own normalized output.
If your attention is cross-attention, write the coordinates to show that `q`
(query position) and `k` (key position) are different roles even when their
extents match. With `num_heads == seq_len`, a refactor that swaps head and key
dimensions produces identical shape tuples — the bracket character is the only
thing that differs between correct and broken.

Consider a common bug: `m = x.max()` used as the stability shift instead of
`max[q](x[b, q])`. When `batch_size > 1`, the global max shifts every batch
item by a potentially different mixed constant. The coordinate-level invariant
catches it: "For any `offset[b]` that omits `j`, `softmax[j](x[b, j] +
offset[b])` must equal `softmax[j](x[b, j])`." A scalar `m` omits both `b` and
`j`, violating the invariant. The bug is a one-coordinate omission error.

Finally, look at `log_softmax` — the numerically stable version that returns
log-probabilities:

```text
fn log_softmax[j](x: [f32; b, j]) -> [f32; b, j] {
    let m[b] = max[q](x[b, q]);
    let lse[b] = log(sum[k](exp(x[b, k] - m[b]))) + m[b];
    let y[b, j] = x[b, j] - lse[b];
    y
}
```

Its Jacobian differs from softmax's:

```text
d(log_softmax[b,j]) / d(x[b,i]) = [i==j] - softmax[b,i]
d(softmax[b,j])     / d(x[b,i]) = softmax[b,j] * ([i==j] - softmax[b,i])
```

The difference is explainable purely through coordinate scopes. `log_softmax`
separates into `x[b, j]` (gradient is `1` at `j`, `0` elsewhere) and `lse[b]`
(gradient is `-softmax[b, i]` at every `i`). Softmax has an extra `softmax[b,
j] * ...` factor because division makes the denominator's influence
multiplicative. The log turns division into subtraction — the output no longer
scales with the denominator. `j` survives, `k` is consumed by `sum[k]`, `q` is
consumed by `max[q]`. Three scopes over one feature dimension. The `[i==j]` term
comes from the `j` path. The `-softmax[b, i]` term comes from the `lse[b]` path.
The `softmax[b, j] * ...` factor appears only when denominator and output
interact multiplicatively. Three letters — `q`, `k`, `j` — carry the entire
Jacobian story.

**Line to keep:** every tensor line is a small audit of which coordinates
survive, which are consumed, and which are silently omitted.

### Where This Leads

Part I taught you to notice when a coordinate role is hidden. A reshape drops the
coordinate story. A broadcast omits a name without stating the omission. A
`dim=-1` consumes whatever axis happens to be last. Six chapters, one reading
discipline: survive, consume, omit.

But noticing is only the surface. The question Part I did not touch is what
happens *because* of the hiding. When the source omits a coordinate role, the
gradient inherits the silence. The autodiff engine becomes a reader that cannot
read. Part II turns the audit onto automatic differentiation.
A forward expression already knows which input cells influence which output
cells. The backward pass is just collecting sensitivity along those routes. If
the forward notation names the sharing, the gradient is mechanical. If it hides
the sharing, the gradient is guesswork — and guesswork with correct shapes is
the hardest bug to find.

Softmax gave us three coordinate roles — `q`, `k`, `j` — in four lines of code.
When sensitivity flows backward through them, those three roles become three
gradient terms with three different structures. The `q` consumed by `max`
produces a sparse Jacobian. The `k` consumed by `sum` produces a dense one. The
`j` that survives creates a diagonal term and an off-diagonal term through the
denominator. The letters you learned to read here will appear, unchanged, in the
pullback formulas of Chapter 7.

You have now seen all three mechanisms of the language in single-expression form.
The axis name and its audit are the primitive. Reduction, broadcast, and `where`
are the means of combination. The coordinate-aware function is the means of
abstraction. Part I built the vocabulary. Part II will run it backward through
the chain rule. The primitive will not change. Survive, consume, omit will still
be the three questions. Only the direction reverses.

Chapter 7 begins the backward pass. Bring the three roles with you. They are
the map — but now we will watch them survive a transformation that Part I never
tested: the journey from forward pass to backward pass, where hidden facts become
silent bugs.