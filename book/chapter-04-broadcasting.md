---
layout: book
title: "What Does Broadcasting Hide?"
---

# What Does Broadcasting Hide?

> "Whereof one cannot speak, thereof one must be silent."
>
> — Ludwig Wittgenstein, *Tractatus Logico-Philosophicus* (1922), Proposition 7

When you borrow a friend's car, you don't ask which friend. The car works the
same regardless. Broadcasting is the tensor equivalent: when a term omits a
coordinate, it claims independence from that coordinate. The claim is often
correct. When it is wrong, it is wrong silently—the car belongs to someone
else and you do not notice until the police show up.

The frightening broadcasting bug is not a crash. It is a model that trains
while sharing a value across the wrong role.

## The Bug That Trained

Start with a story. Not a fabricated one—a real category of bug that has
shipped to production.

You are training an image classifier. Your model normalizes activations with a
running mean. The running mean should be per-feature: every pixel channel gets
its own normalization statistic, computed over the batch.

```python
# PyTorch — can you spot the bug?
x = torch.randn(64, 128)          # [batch=64, feature=128]
running_mean = torch.zeros(128)   # [feature=128]
x_norm = x - running_mean          # broadcasting: (64,128) - (128,)
```

The code runs. The shapes align. `running_mean` has shape `(128,)`, which
broadcasts to `(64, 128)` by prepending a singleton dimension. Every batch
item sees the same feature-specific mean. Loss decreases. Metrics improve. You
ship.

Three weeks later, a colleague refactors the data pipeline. The new loader
produces tensors of shape `[feature, batch]` instead of `[batch, feature]`.
The line `x - running_mean` still runs. `running_mean` still has shape
`(128,)`. Broadcasting prepends the singleton to the *first* axis, which is
now `feature`. The subtraction is now:

```text
x[feature, batch] - running_mean[feature] → per-feature subtraction? No.
```

Wait. If `x` is `[128, 64]` and `running_mean` is `[128]`, broadcasting
matches on the *last* axis. So `running_mean[128]` broadcasts to `[128, 64]`
by *appending* a singleton, not prepending. The subtraction is still
per-feature.

But what if `x` had shape `[feature=128, batch=64]` and the *intent* was to
normalize per-batch? `running_mean` of shape `[64]` would broadcast to `[128,
64]` by appending a singleton—along the feature axis. Every feature in a
given batch item gets the same correction. That might even look like it's
working. The loss still decreases.

No error. No crash. Just a model that learned around a normalization applied
to the wrong coordinate.

This is the frightening broadcasting bug. Not a shape mismatch. A role
mismatch that happens to have the right shape.

Pause and ask: what would have caught this bug at the moment the refactor
landed?

## What Broadcasting Actually Means

Before fixing it, understand what broadcasting does. Take the simplest case:

```python
a = torch.randn(16, 1, 64)   # [batch=16, singleton=1, feature=64]
b = torch.randn(1, 32, 64)   # [singleton=1, time=32, feature=64]
c = a + b                     # result: [16, 32, 64]
```

Most tensor programmers can read that after a moment. The second dimension of
`a` expands from 1 to 32. The first dimension of `b` expands from 1 to 16.

Now ask a sharper question. When you write `c = a + b`, what makes the
expansion *correct*? The PyTorch answer is: the singleton dimensions align, so
the shapes are compatible. The semantic answer is different: `a` is
independent of `time`, and `b` is independent of `batch`. Independence is why
reuse is valid. But broadcasting only checks shape compatibility. It does not
check independence. It *cannot* check independence, because the source never
stated it.

The contract broadcasting enforces:

```text
shape rule:    singleton can expand to match
semantic rule: (not checked) value is independent of expanded coordinate
```

The first rule is automatic. The second rule lives in your head. When they
disagree, you get the bug that trained.

This is not an argument against broadcasting. Broadcasting is one of the
reasons array code is pleasant. The argument is that the source should have a
way to state the semantic rule when it matters.

The Hiding Law does not forbid implicit behavior. It forbids hiding a fact
that later reasoning must recover. Broadcasting is fine when the
independence claim is obvious. It becomes dangerous when the independence
claim is wrong -- and the notation provides no place to check it. The
missing feature is not a runtime check. It is a syntax for stating what
the runtime would need to check.

## Visible Absence

Here is the same addition, with the independence claim made explicit:

```rust
let c[batch, time, feature] = a[batch, feature] + b[time, feature];
```

Read the two terms separately. `a[batch, feature]` mentions `batch` and
`feature`. It does not mention `time`. `b[time, feature]` mentions `time` and
`feature`. It does not mention `batch`.

The absence is the claim. If a term does not mention `time`, it cannot
possibly care which `time` you picked. The compiler can see this directly:

```text
a[batch, feature]  → uses batch, feature;  absent: time
b[time, feature]   → uses time, feature;   absent: batch
```

No singleton dimension. No alignment rule to remember. No question about which
axis prepends and which appends. The term either mentions a coordinate or it
doesn't. If it doesn't, it's constant along that coordinate.

Two small axioms capture the entire mechanism:

```text
1. If a term does not mention a result coordinate,
   that term is independent of the coordinate.

2. Independence means every value along that coordinate
   receives the same term. A backend may implement this as
   reuse, broadcasting, or repetition.
```

This is the middle path. The source does not expand every repeated cell. But
it also does not pretend the repetition was discovered from a singleton
dimension after the fact. The claim is in the expression.

## The Bias Test

Now return to the bug. A bias vector can be a feature bias or a batch bias:

```python
# PyTorch: both are legal, both have the same shape
y = x + feature_bias   # feature_bias shape: (128,)
y = x + batch_bias     # batch_bias shape: (64,) — wait, different shape
```

Shape alone *can* distinguish them if `feature != batch`. But what if they're
equal?

```python
x = torch.randn(128, 128)          # [feature=128, batch=128]
feature_bias = torch.zeros(128)    # [128]
batch_bias = torch.zeros(128)      # [128] — same shape
y1 = x + feature_bias              # legal, per-feature bias
y2 = x + batch_bias                # also legal, broadcasting matches on last axis
```

Both run. Both produce a `[128, 128]` result. The difference is invisible to
the shape checker. In the PyTorch version, `feature_bias` and `batch_bias` are
distinguished only by your intention, not by the code.

This is the thesis in a single example. The two biases have the same
shape. The two additions produce the same output shape. Every automated
check passes. The only place the two programs differ is in the meaning of
the axis -- and the positional notation has no vocabulary for meaning. It
has vocabulary for extent and stride. The coordinate notation adds one
word: the name of the coordinate the bias actually inspects.

The Einlang version forces the decision to appear where the addition is
written:

```rust
// Feature bias: same bias value for every batch item
let y[b, f] = x[b, f] + feature_bias[f];

// Batch bias: same offset for every feature in a given example
let y[b, f] = x[b, f] + batch_bias[b];
```

Read two concrete cells from each:

```text
Feature bias:
  y[3, 5] = x[3, 5] + feature_bias[5]
  y[7, 5] = x[7, 5] + feature_bias[5]   ← batch changed, bias address unchanged

Batch bias:
  y[3, 5] = x[3, 5] + batch_bias[3]
  y[3, 9] = x[3, 9] + batch_bias[3]     ← feature changed, bias address unchanged
```

The difference is visible at the use site. If someone accidentally writes:

```rust
let y[b, f] = x[b, f] + feature_bias[b];  // feature_bias indexed by batch
```

A reader can ask: why is the feature bias addressed by `b`? Even if
`feature_count == batch_count`, the role is wrong. The notation gives the
question a place to live.

Now the diagnostic for the refactor scenario becomes concrete. If the data
loader changes the coordinate order, the lines that mention coordinates by
name must change with it. The lines that used position silently are the ones
that break.

```
   Broadcasting as Coordinate Omission

   Correct: y[b,f] = x[b,f] + bias[f]    bias omits b
   +-----+-----+-----+-----+
   |     | f=0 | f=1 | f=2 |
   +-----+-----+-----+-----+
   | b=0 | +b0 | +b1 | +b2 |  bias[f] reused for every b
   +-----+-----+-----+-----+
   | b=1 | +b0 | +b1 | +b2 |  same bias, same f indices
   +-----+-----+-----+-----+

   Wrong: bias indexed by b when model expects per-feature bias
   +-----+-----+-----+-----+
   |     | f=0 | f=1 | f=2 |
   +-----+-----+-----+-----+
   | b=0 |b[0] |b[0] |b[0] |  bias varies with b, constant
   +-----+-----+-----+-----+  across f -- opposite of intent
   | b=1 |b[1] |b[1] |b[1] |
   +-----+-----+-----+-----+

   Same shape, opposite meaning. The absence is the contract.
```

## Reduction Shows the Same Absence

Broadcasting is not a separate mechanism. It is one side of a coin. The other
side is reduction.

```rust
let y[b, f] = x[b, f] + bias[f];    // broadcasting: bias omits b
let total[b] = sum[f](x[b, f]);      // reduction: sum consumes f
```

Read them as a pair. In the first line, `bias[f]` omits `b`, so it is reused
for every batch item. In the second line, `f` is introduced by `sum` and
consumed, so it does not survive into `total`. For a concrete row:

```text
total[3] = x[3, 0] + x[3, 1] + ... + x[3, n-1]
```

The coordinate `f` walks across the row, fills the sum, and leaves. Only
`batch` remains on the result.

This pairing matters in the gradient chapters. A value broadcast in the
forward pass receives a summed gradient in the backward pass:

```rust
// Forward: bias[f] is reused across b
let y[b, f] = x[b, f] + bias[f];

// Backward: bias[f] collects sensitivity from every b
let dbias[f] = sum[b](dy[b, f]);
```

The omitted coordinate in the forward expression becomes a reduced coordinate
in the backward expression. This is not a special trick of biases. It is the
coordinate accounting forced by reuse: a value reused along `b` receives
sensitivity from every `b`.

So the concrete question for any broadcast is not "did the library expand a
singleton dimension?" The better question is:

```text
Which coordinate does this term not mention?
```

That question tells you three things at once: where the value is constant,
where mistakes can hide, and which coordinate will later be collected if a
derivative flows backward.

This forward-backward duality connects directly to Chapter 3's coordinate map
analysis. A coordinate map like transpose or flatten is reversible: information
moves without being consumed. Broadcasting is different. A value is reused
without being copied. The backward pass reveals the reuse by summing sensitivity
over every position that shared the value — but only if the notation recorded
which coordinate was shared. If the source never states it, the backward pass
does not know which coordinate to sum over. The omission in the forward pass
becomes a missing fact in the backward pass. The notation must carry it
forward.

## What the Compiler Can Actually Check

When absence is visible, the compiler has more to work with. Take three terms:

```text
x[b, f]           → uses b, f
feature_bias[f]   → uses f;    absent: b
batch_bias[b]     → uses b;    absent: f
```

The compiler can now separate three questions. First, are the addressed values
rectangular and indexable? Second, do the index ranges agree with the array
extents? Third, which result coordinates are absent from each term?

The third question is the semantic one. If `feature_bias[f]` appears in a
result addressed by `[b, f]`, the absent coordinate is `b`—the value is
constant along batch. A compiler that tracks this can report the fact. A
compiler that only sees shapes can report that a singleton expanded, but it
cannot report *why*.

This changes the quality of error messages. Compare:

```text
Shape-based:  "operands could not be broadcast: (64,128) vs (64,)"
Coordinate:   "feature_bias[b] is indexed by batch, but declared as feature bias"
```

The first tells you what went wrong with the layout. The second tells you what
went wrong with the model. When you are debugging at 3 AM, the second message
is the one that saves an hour.

## Where Clauses: When a Coordinate Is Local

Broadcasting is not the only place where a coordinate can be constrained
without becoming part of the result. A `where` clause introduces local
bindings:

```rust
let output[i, j] = activated
    where z = sum[k](input[i, k] * weight[k, j]) + bias[j],
          activated = if z > 0.0 { z } else { 0.0 };
```

Here `z` and `activated` are not output coordinates. They are local facts used
while computing each `[i, j]` cell. The rule is the same as broadcasting: a
local name does not mention all result coordinates, so it is independent of
the ones it omits.

A different `where` clause acts as a filter:

```rust
let upper[i, j] = matrix[i, j] where i <= j;
```

This still defines a rectangular family addressed by `[i, j]`, but the guard
controls which cells receive the expression value. The guard `i <= j` is not a
binding—it does not introduce a reusable value—but it affects coverage and
execution.

The compiler must distinguish these cases to reason about shape and range
correctly. A binding like `activated = ... z ...` depends on the earlier
binding `z`. A guard like `i <= j` does not introduce a value but constrains
the domain. The indexed style gives both a place in the source, and the
compiler can keep output shape, local computation, and filtered coverage in
separate buckets.

This is the broader point. A `where` clause is not syntactic sugar. It
is a scoping mechanism that lets a coordinate be local to a computation
without becoming part of the result family. Broadcasting achieves the
same thing by omission: a term that does not mention a coordinate makes
that coordinate local to the broadcast. The two mechanisms -- `where`
and omission -- are different syntax for the same semantic fact. A
notation that can state both can check both. A notation that can state
neither must rely on the reader to infer the scope of every intermediate
value.

## The Audit

You now have a concrete test for any tensor expression. Given a term that
appears inside a larger result, ask:

```text
1. Which coordinates does this term use?
2. Which result coordinates does this term omit?
3. Would the omission still be correct if the omitted coordinate
   had a different role with the same length?
```

Apply it to the simplest case:

```text
x[b, f] + bias[f]

Term bias[f]:
  uses: f
  omits: b
  test: if we swapped b's role from "batch" to "feature",
        would bias[f] still be correct? No — it doesn't mention b at all.
```

Now apply it to the bug from the beginning of the chapter:

```python
x = torch.randn(128, 128)          # [feature=128, batch=128] after refactor
running_mean = torch.zeros(128)    # intended as per-feature, shape happens to be 128
x_norm = x - running_mean          # broadcasting matches on last axis
```

The term `running_mean` omits the *first* axis. After the refactor, the first
axis is `feature`. So `running_mean` is being reused across features—the
opposite of the intent. But if `feature_count == batch_count`, the shapes are
identical, and no error is raised. The test catches it because the omission is
now across the wrong role.

## Summary

Broadcasting is a shape-compatibility rule dressed as a semantic claim. The
shape rule says a singleton can expand to match. The semantic rule—the one
broadcasting cannot check—says the value must be independent of the expanded
coordinate. When those two rules disagree, the code runs but the model learns
the wrong thing.

Where Chapter 2 introduced the role audit (which coordinates survive, which are
consumed, which are omitted), and Chapter 3 showed how coordinate maps preserve
information without destroying it, this chapter showed that omission is
different from both. A coordinate omitted from a term is not destroyed—it is
simply not inspected. The term is constant along that coordinate. The backward
pass reveals this by summing sensitivity over every position that shared the
value.

Three ideas will carry forward:

1. **Absence is a claim.** When `bias[f]` appears in `y[b, f] = x[b, f] +
   bias[f]`, the absence of `b` from `bias` is the claim of independence. The
   compiler can inspect that claim directly, without recovering it from
   singleton dimensions.
2. **Omission and reduction are dual.** A value reused along `b` in the forward
   pass collects a sum over `b` in the backward pass. This pairing is not a
   special case of biases—it is the coordinate accounting forced by reuse. Read
   Chapter 7 with this duality in mind.
3. **Error messages should name roles, not extents.** "feature_bias is indexed
   by batch" is more useful than "operands could not be broadcast: (64,128) vs
   (64,)". The role name tells you what went wrong with the model; the extent
   only tells you what went wrong with the layout.

The broadcast audit—"which coordinate does this term not mention?"—will return
in Chapters 7–9 when we trace gradient denominators back to their forward-pass
origins. A gradient that sums over `b` is the backward echo of a forward term
that omitted `b`.

### Where This Leads

Broadcasting is not a layout trick. It is a claim: the term that omits a
coordinate asserts, for every cell it produces, that the omitted coordinate's
value does not matter. A bias vector does not depend on batch identity — the
claim is usually right. When it is wrong, the program runs, the shapes match,
the loss descends, and the model learns a pattern that conflates two roles.

When the notation names the omission, the claim is visible in the source — a
blank where the coordinate should be. When the notation hides the omission, the
claim is a convention, and conventions survive only in the reader's memory.
This is the Hiding Law in its positive form: do not just avoid hiding. Give the
fact a place to live. The absence of a coordinate is a fact. The bracket that
omits it but leaves the name visible is the place.

But absence is only half the story. Some coordinates are not merely absent
from a term—they are *consumed* by an operation and removed from the result
entirely. A sum over `k` makes `k` disappear. Where does it go? And what
happens when you need it back?

Chapter 5 answers that question: the index that leaves.

<div class="pause" markdown="1">
**Pause.** Find the last three places in your codebase where you used
broadcasting — a bias added to a tensor, a mean subtracted, a scalar multiplied
across a batch. For each one, write down which coordinate the broadcast value
is independent of. Now ask: if the data pipeline swapped two axes of equal
size, would the broadcast still apply to the right coordinate? If the answer is
"the shape would still be compatible," you have found a silent bug waiting to
happen. Write the coordinate name down. Even in a comment. Even on a sticky
note. The act of naming it is the first half of the audit.
</div>

## Try It

Recall from Chapter 3 the coordinate map: `result[b, c, i, j] = input[b, c *
(r*r) + (i%r)*r + (j%r), i/r, j/r]`. Audit it for broadcasting using this
chapter's vocabulary. For the input term on the right-hand side, list the
coordinates it uses and the coordinates it omits:

```text
Term: input[b, c * (r*r) + (i%r)*r + (j%r), i/r, j/r]
  Uses:  b, c, r, i, j (all five appear, whether as themselves or in arithmetic)
  Omits: none—every result coordinate appears in the index expression
```

The input term uses every coordinate. There is no broadcasting in this map. It
is a pure coordinate map (Chapter 3), not a broadcast expression. Here is the
insight: if a reviewer mistakenly thought `b` was absent from the input term,
they would be claiming the input value is independent of batch—that every batch
item shares the same input. But the coordinate equation shows `b` appearing
directly in the input index. The equation itself corrects the mistake. That is
the difference between a shape tuple and a coordinate relation: the relation can
prove whether broadcasting is happening.

Create a toy tensor where `time_count == feature_count == 64`. Write a bias
that a positional framework could broadcast in either role:

```text
let y[t, f] = x[t, f] + bias[f];   // per-feature bias: same bias for every time
let y[t, f] = x[t, f] + bias[t];   // per-time bias: same bias for every feature
```

Which one does your model contract permit? If you cannot answer from the code
alone, the notation has already done its job: it forced the question to appear.
Now break it deliberately. Set `time_count = feature_count = 128`. Write
`bias[f]` where the model contract expects `bias[t]`. Both produce shape `[128,
128]`. A shape-only checker sees `(128) + (128, 128)` broadcasting legally—it
cannot distinguish `bias[f]` from `bias[t]`. A coordinate-aware checker sees
`bias` indexed by `f` in a result that also carries `t`, and could report
something like:

```text
broadcast check: bias[f] omits coordinate t.
  t has extent 128 and role 'time'.
  bias is indexed by 'feature'. It is constant across time.
  If constant-across-time is not intended, check the bias coordinate.
```

The error message names the role (feature), the omitted coordinate (time), and
the semantic claim (constant across time). A shape-only message would say
"broadcast from (128,) to (128,128) successful"—which is true and useless.

Implement batch norm from scratch using only named coordinates and the primitive
`fold_over`. The `fold_over` primitive applies an associative binary operation
over a named coordinate:

```text
fn fold_over[coord](op: (a, a) -> a, init: a, x: [a; ..left, coord, ..right])
    -> [a; ..left, ..right]
```

Use it to build batch normalization over the feature coordinate:

```text
fn batch_norm[feature](x: [f32; batch, feature]) -> [f32; batch, feature] {
    // Step 1: mean over batch using fold_over
    let sum_feature[feature] = fold_over[batch](add, 0.0, x[batch, feature]);
    let mean[feature] = sum_feature[feature] / batch_count;

    // Step 2: variance over batch
    let sq_diff[batch, feature] = (x[batch, feature] - mean[feature]) ** 2;
    let var_sum[feature] = fold_over[batch](add, 0.0, sq_diff[batch, feature]);
    let var[feature] = var_sum[feature] / batch_count;

    // Step 3: normalize
    let y[batch, feature] =
        (x[batch, feature] - mean[feature]) / sqrt(var[feature] + eps);
    y
}
```

The insight: `fold_over` with `op=add` over `batch` gives the sum over batch. In
the backward pass (describe, do not implement), the gradient of `fold_over` over
`batch` reverses the broadcasting: `init` receives gradient summed over `batch`,
and each element along `batch` receives a contribution from `add`'s partial
derivative. The forward broadcast becomes a backward reduction. This is the
broadcast-reduce duality from this chapter, generalized.

Now break it. What if your `batch` coordinate is actually `time`? You trained on
a time series where each example is one time step, but you accidentally labeled
the coordinate `batch`. The code runs. The shapes match. `fold_over[batch](add,
...)` computes running statistics that mix time and batch—each "mean" is now a
mix of different examples at different times. The model silently trains on
statistics that conflate two semantically different axes. `fold_over[batch](...)`
and `fold_over[time](...)` differ in one character, but produce shapes that are
identical when `batch_count == time_count`. No shape check catches this. Only
the coordinate name in the bracket can. `fold_over` generalizes the
broadcast-reduce duality into a single coordinate function. The bracket says
which coordinate is being folded. Change the bracket, change the semantics. The
shape stays the same.

**Line to keep:** if a term does not mention `j`, it cannot care which `j` you
chose.