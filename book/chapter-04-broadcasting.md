---
layout: book
title: "Chapter 4: What Does Broadcasting Hide?"
---

# What Does Broadcasting Hide?

The frightening broadcasting bug is not a crash. It is a model that trains
while sharing a value across the wrong role.

Broadcasting is one of the great conveniences of array programming:

```python
a = torch.randn(16, 1, 64)
b = torch.randn(1, 32, 64)
c = a + b
```

The result has shape `(16, 32, 64)`. The second dimension of `a` expands. The
first dimension of `b` expands. Most tensor programmers can read that after a
moment.

But the pause is interesting. For `c[3, 5, :]`, which part of `a` is used?
Which part of `b`?

```text
a[3, 0, :]
b[0, 5, :]
```

The answer is not difficult. That is part of the charm of broadcasting. But it
is not written in the expression `a + b`; it is recovered from a convention
about singleton dimensions and shape alignment. Broadcasting keeps the code
short by moving the coordinate story into a rule the reader already knows.

That trade is often worth it. It is also an excellent place for a quiet mistake
to hide.

The quiet mistake can be expensive. Suppose a training job computes a
per-feature running mean, but a refactor leaves the mean addressed by `batch`
because `batch_count == feature_count` in the experiment that shipped. The
subtraction broadcasts legally. The loss decreases. The model learns around a
normalization that is shared across the wrong role. Nothing in the shape tuple
has to scream. The first hard symptom may be a deployment set whose batch size
or feature count changes, or a model whose calibration is wrong for reasons no
unit test caught.

Broadcasting is not foolish; it is one of the reasons array code is pleasant.
The sharper question is whether a value is truly independent of a coordinate,
or whether a singleton dimension merely made the wrong reuse look legal.

## Implicit Expansion or Visible Absence

Broadcasting offers a tempting design: let singleton dimensions expand
silently, and trust shape rules to reject impossible cases. This is concise,
and for many programs it is exactly what the programmer wants. The weakness is
that the source does not say which coordinate was absent by design.

A stricter design could forbid broadcasting entirely. That would make every
reuse explicit, and it would also make ordinary tensor formulas unpleasantly
heavy. The useful middle ground is to make absence visible in the coordinate
story without forcing every expanded element to be written.

At this point it is reasonable to object: is visible absence just another
implicit behavior? The difference is where the fact lives. Implicit
broadcasting asks the reader to infer reuse from shape layout. A missing
coordinate in an indexed term declares the reuse in the expression itself.
One is accidental expansion; the other is stated independence.

Einlang's notation follows that middle path. A term that lacks `j` is not an
accident of layout; it is a value that does not depend on `j`. The design
principle is the same as before: if the omitted coordinate matters to later
checking or differentiation, the source should not hide it inside a singleton
dimension.

Every singleton axis should raise a small question: is this value truly
independent of that coordinate, or did we merely arrange memory so the library
would accept it? A feature bias is independent of batch. A per-example offset
is not. Both can be stored as vectors. The coordinate expression is where the
distinction becomes hard to miss.

This is also a function-boundary issue. A helper such as:

```text
let y[b, feature] = add_bias[feature](x[b, feature], bias[feature])
```

is acceptable only if the bracketed coordinate states the reuse contract:
`bias[feature]` is reused across every other coordinate of `x`. Without that
coordinate argument, the helper has hidden the same fact broadcasting usually
hides.

## Omitted Coordinates

An index-oriented version can make the omission explicit without inventing a
singleton slot:

```text
a depends on i and k
b depends on j and k
c[i, j, k] = a[i, k] + b[j, k]
```

In the expression for `c`, the term `a[i, k]` depends on `i` and `k`, but not
on `j`. The term `b[j, k]` depends on `j` and `k`, but not on `i`. So for one
concrete output coordinate:

```text
c[3, 5, k] = a[3, k] + b[5, k]
```

There is no hidden singleton axis to remember. Broadcasting becomes a property
of the expression's free coordinates. If a term does not mention `j`, it cannot
possibly care which `j` you picked.

The rule can be stated as two small axioms:

```text
1. If a term does not mention a result coordinate, that term is independent of
   the coordinate.
2. Independence means every value along that coordinate receives the same term;
   a backend may implement this as reuse, broadcasting, or repetition.
```

This is the whole middle path. The source does not expand every repeated cell,
but it also does not pretend the repetition was discovered from a singleton
dimension after the fact.

## Bias Has a Job

Broadcasting bugs are usually not rank bugs. They are role bugs. A tensor may
have a shape that permits expansion while still expanding along the wrong
meaning.

Consider:

```python
y = x + bias
```

If `x` has logical shape `[batch, feature]`, then a feature bias is:

```rust
let y[b, f] = x[b, f] + bias[f];
```

A batch bias is a different program:

```rust
let y[b, f] = x[b, f] + bias[b];
```

Both can produce a two-dimensional result. Only one matches the intended model.
The indexed form forces the decision to appear where the addition is written.

Read the two concrete points:

```text
y[3, 5] = x[3, 5] + bias[5]   feature bias
y[3, 5] = x[3, 5] + bias[3]   batch bias
```

A bias vector is more than "a vector." It is a vector with a job. The index
tells you what job it is doing.

This is also why broadcasting is a semantic choice, not a rank trick. If
`bias` has length `64`, it could match a feature coordinate of
length `64`, a time coordinate of length `64`, or a class coordinate of length
`64`. The size alone cannot choose among those stories:

```text
x[b, feature] + bias[feature]
x[b, time]    + bias[time]
x[b, class]   + bias[class]
```

The notation asks the addition to say which reuse it means.

This separates logical absence from physical layout. A backend may still store
`bias[feature]` as a tensor with a singleton batch dimension because that
layout is convenient for a kernel. The source-level claim is different:
`bias` does not depend on `b`. Treating those as separate facts gives the
compiler freedom to choose a layout without pretending that a singleton axis
has semantic content.

## Reduction Shows the Same Absence

The reduction operators in `stdlib/ml/reduction_ops.ein` show the same
discipline in a small production example:

```rust
let result[..batch] = sum[j](x[..batch, j]);
```

This is `reduce_sum`. The ordinary description would be "reduce over the last
axis." The indexed description says more. The rest pattern `..batch` survives.
The local coordinate `j` is introduced by `sum` and consumed inside the
reduction body. The output keeps the batch-shaped part and loses the feature
coordinate.

For a concrete row:

```text
result[b] = x[b, 0] + x[b, 1] + ... + x[b, n - 1]
```

The coordinate `j` walks across the row, fills the sum, and leaves. Only the
batch label remains on the result.

This is also the first place where reduction and broadcasting meet. A value
that no longer has `j` may later be used in an expression that does have `j`.
If that happens, it will be invariant along `j` because `j` was consumed.

## Softmax Reuses the Pattern

`softmax` in `stdlib/ml/activations.ein` is a denser example:

```rust
let output[..batch, j] =
    exp(x[..batch, j] - max[q](x[..batch, q]))
    / sum[k](exp(x[..batch, k] - max[q](x[..batch, q])));
```

There are three coordinate roles in this one expression. The coordinate `j` is
the feature being returned. The coordinate `k` is local to the denominator. The
coordinate `q` is local to the maximum used for numerical stability. The batch
tail stays untouched through all of it.

That separation matters. If all three roles were described only as "the last
axis," the reader would have to keep their scopes in memory. Here the scopes
are written into the formula:

```text
j  the feature we are computing
k  the feature axis scanned by the denominator
q  the feature axis scanned by the maximum
```

The names are small, but they make the normalization legible. Anonymous axes
invite accidental swaps; named local coordinates make the swaps easier to see.

## What Does an Omitted Coordinate Mean?

The softmax expression is a useful place to notice how much is carried by
absence. Its three appearances of `x` mention different local coordinates:

```text
x[..batch, j]
x[..batch, k]
x[..batch, q]
```

The whole chapter is in that difference. Broadcasting is not a separate
mechanism. It is what happens when a value does not depend on a coordinate that
the surrounding expression still has. The important question is not only which
coordinate is written, but which coordinate a term is allowed to ignore.

## The Pair With Reduction

Broadcasting and reduction are paired ideas. Broadcasting is what happens when
a coordinate is absent from a term but present in the surrounding result.
Reduction is what happens when a coordinate is present locally but absent from
the surrounding result.

Put them side by side:

```rust
let y[b, f] = x[b, f] + bias[f];
let total[b] = sum[f](x[b, f]);
```

In the first line, `bias[f]` omits `b`, so it is reused for every batch item.
In the second line, `f` is introduced by `sum` and consumed, so it does not
survive into `total`.

This pairing matters in the gradient chapters. A value broadcast in the forward
pass usually receives a summed gradient in the backward pass. A
coordinate consumed in the forward pass often reappears in the shape of a
parameter gradient. The source notation makes those relationships easier to
predict because it shows which coordinates were absent and which were local.

Every expression can now be read with two questions instead of one. Which
coordinates are present? Which coordinates are deliberately absent? The second
question is what broadcasting usually hides, and it is often the question that
finds the bug.

## The Wrong Shared Parameter

Take a matrix of activations and two possible biases:

```text
x[b, f] = ...
feature_bias[f] = ...
batch_bias[b] = ...
```

Adding the feature bias is:

```rust
let y[b, f] = x[b, f] + feature_bias[f];
```

The term `feature_bias[f]` omits `b`, so the same feature bias is used for
every example. Read two cells:

```text
y[0, 3] = x[0, 3] + feature_bias[3]
y[7, 3] = x[7, 3] + feature_bias[3]
```

The coordinate `b` changed, but the bias address did not. That is the precise
meaning of the broadcast.

Adding the batch bias is different:

```rust
let y[b, f] = x[b, f] + batch_bias[b];
```

Now `batch_bias[b]` omits `f`, so every feature in the same example receives
the same offset:

```text
y[7, 0] = x[7, 0] + batch_bias[7]
y[7, 3] = x[7, 3] + batch_bias[7]
```

Both programs produce a tensor addressed by `[b, f]`. Both are valid if the
model wants that behavior. They are not interchangeable. A positional
broadcasting system may encode the difference as a vector of shape `[F]`
versus a vector reshaped to `[B, 1]`. That works, but the role is implicit in
where the singleton dimension was inserted. The named form says the role at
the use site.

This is where the example stops being a reminder about broadcasting and starts
being a real failure mode. If `batch_count` and `feature_count` are both `128`,
the wrong vector can have the right length. A shape-only error message may not
appear at all. A visible-coordinate reading still catches the mistake because
the suspicious line is local:

```text
feature_bias[b]
```

That address says the feature bias is being indexed by examples. Even if the
lengths match, the role does not.

The same audit explains the reverse direction in gradients. If a scalar loss
depends on:

```rust
let y[b, f] = x[b, f] + feature_bias[f];
```

then the gradient of `feature_bias[f]` must collect all batch routes:

```rust
let dfeature_bias[f] = sum[b](dy[b, f]);
```

The omitted coordinate in the forward expression becomes a reduced coordinate
in the backward expression. That is not a special trick of biases. It is the
coordinate accounting forced by reuse. A value reused along `b` receives
sensitivity from every `b`.

So the concrete question for any broadcast is not "did the library expand a
singleton dimension?" The better question is:

```text
Which coordinate does this term not mention?
```

That question tells you where the value is constant, where mistakes can hide,
and which coordinate will later be collected if a derivative flows backward.

## What Static Checking Can Actually See

In the explicit form, broadcasting is not an afterthought. The compiler sees
which indices a term uses:

```text
x[b, f]             uses b and f
feature_bias[f]     uses f
batch_bias[b]       uses b
```

That lets the checker separate three questions. First, are the addressed
values rectangular and indexable? Second, do the index ranges agree with the
array extents? Third, which result coordinates are absent from each term?

The third question is the semantic one. If `feature_bias[f]` appears in a
result addressed by `[b, f]`, the absent coordinate is `b`; the value is
constant along batch. If `batch_bias[b]` appears, the absent coordinate is
`f`; the value is constant along features. This is exactly the fact a
positional broadcasting rule reconstructs from singleton dimensions.

Making absence visible also improves explanations. A good error or review
comment can say "this term does not depend on `b`" rather than "axis 0 was
broadcast." The first sentence names the model role. The second names a layout
position. Both may be true, but the role sentence is the one that helps find a
semantic bug.

Explicit absence deserves a place beside explicit presence. A coordinate can
matter precisely because one term does not mention it.

## Where Clauses as Classified Facts

Broadcasting is not the only place where a coordinate may be constrained
without becoming part of the result. A `where` clause can introduce a local
binding or a guard:

```rust
let output[i, j] = activated
    where z = sum[k](input[i, k] * weight[k, j]) + bias[j],
          activated = if z > 0.0 { z } else { 0.0 };
```

Here `z` and `activated` are not output coordinates. They are local facts used
while computing each `[i, j]` cell. A different `where` clause can act as a
filter:

```rust
let upper[i, j] = matrix[i, j] where i <= j;
```

That line still defines a rectangular family addressed by `[i, j]`, but the
guard controls which cells receive the expression value and which cells receive
the default.

The compiler has to distinguish those cases before range and shape reasoning
can stay sane. Constraint classification separates binding-like constraints
from guard-like constraints and orders bindings by dependency. A binding such
as `activated = ... z ...` depends on the earlier binding `z`; a guard such as
`i <= j` does not introduce a reusable value, but it does affect coverage and
execution.

Here the indexed style grows beyond axis names. The source can also classify
local facts: derived values, predicates, and guards. The compiler can then keep
output shape, local computation, and filtered coverage in separate buckets
instead of treating every `where` item as a string of condition text.

The small audit for broadcasting is `x[b, f] + bias[f]`: list the coordinates
each term depends on, then change only the intended role of `bias` while
keeping the rank plausible. If the expression still looks harmless, the reuse
was only shape-compatible, not semantic.

## Try It

Create a toy tensor where `time_count == feature_count`. Write a bias that a
positional framework could broadcast in either role. Then write the named
version twice, once as `bias[feature]` and once as `bias[time]`, and ask which
one the model contract permits. The useful error message should name the role,
not only the extent.

**Line to keep:** if a term does not mention `j`, it cannot care which `j` you
chose.
