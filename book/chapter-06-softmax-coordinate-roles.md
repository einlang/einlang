---
layout: book
title: "Chapter 6: Softmax Has Three Coordinate Roles"
---

# Softmax Has Three Coordinate Roles

Softmax is often described as an operation along an axis. That is true, but it
is not quite enough. In a stable implementation, the same logical feature axis
appears in several different roles.

The compact mathematical slogan is:

```text
softmax(x)_j = exp(x_j) / sum_k exp(x_k)
```

The stable implementation then adds a maximum, often written as if it were
only a numerical trick. The trap is to think there is only one index because
there is only one logical axis. Computationally, the returned feature, the
denominator scan, and the stabilizing maximum have different scopes.

The standard library writes the batched case as:

```rust
let output[..batch, j] =
    exp(x[..batch, j] - max[q](x[..batch, q]))
    / sum[k](exp(x[..batch, k] - max[q](x[..batch, q])));
```

The formula is longer than `softmax(x, axis=-1)`, but it exposes the structure
that the compact call hides.

Once that structure is stated, it can become a coordinate function:

```rust
let output[..batch, j] = softmax[j](x[..batch, j]);
```

This call is compact, but it is not positional. The bracketed `j` says which
coordinate is normalized, while the result still carries `j`. The standard
library can hide the stable maximum and denominator scans only because the
coordinate contract remains visible at the boundary.

Softmax has one logical feature axis, but several coordinate jobs: the
feature being returned, the feature scanned by the denominator, and the feature
scanned by the maximum. Since the output has the same shape as the input, it is
easy to mistake softmax for an elementwise operation. Naming the scopes keeps
that mistake from becoming invisible.

## One Axis Name or Several Scopes

A tempting notation would use one feature name everywhere. After all, softmax
normalizes over one axis. But the expression has several local jobs: one
feature is being returned, another local coordinate scans the denominator, and
another may scan the maximum used for numerical stability.

Using one name for all of those jobs makes the formula look simpler while
making the scopes harder to see. Using distinct local names makes the formula
slightly longer, but it reveals which occurrence is an address of the result
and which occurrences are local scans.

Einlang's choice is to make scope visible. A coordinate name is not only a
label for an axis; it is also a statement about where that axis is live.

Compute just one probability by hand. To produce `output[b, 7]`, the formula
needs the score at feature `7`, the maximum over all features in the same row,
and the denominator over all features in the same row. Calling all of those
positions "the softmax axis" is true but too coarse. The single output value
already contains three scopes.

## The Three Roles

The coordinate `j` is the feature being returned:

```text
output[..batch, j]
```

The coordinate `k` is the feature coordinate scanned by the denominator:

```text
sum[k](...)
```

The coordinate `q` is the feature coordinate scanned by the maximum:

```text
max[q](...)
```

All three range over the same conceptual axis, but they have different scopes.
That is why they deserve different names. If a formula reused one letter for
all three jobs, the reader would have to infer the scopes from punctuation
alone.

This is also why `j`, `k`, and `q` should not be collapsed simply because they
range over the same feature extent. They have different binding sites. The
output coordinate `j` is an address of the returned value. The denominator
coordinate `k` is a local scan. The maximum coordinate `q` is another local
scan whose result is shared across `j`. In reverse mode, those scopes induce
different routes of sensitivity. Naming them separately is not decoration; it
is respect for the derivative graph that will later be built from the formula.

## Why Stability Adds a Role

The maximum is not there to change the mathematical result. It is there to keep
the exponentials numerically well behaved:

```text
exp(x[j] - max[q](x[q]))
```

Subtracting the maximum shifts every feature by the same amount for a fixed
batch item. The softmax probabilities stay the same, but the largest exponent
becomes `exp(0)` instead of possibly overflowing.

That numerical trick introduces another local coordinate. The coordinate `q`
does not choose the output feature. It scans all features to find the shift
used by every output feature in the same row. In compact API code, this role is
usually implicit. In the indexed expression, it is visible.

This is a recurring theme: practical numerical code often has extra structure
that the mathematical slogan omits. "Softmax over the last axis" is the slogan.
The stable formula has an output coordinate, a denominator coordinate, and a
maximum coordinate.

That extra structure has a visible consequence. The stabilizing maximum is
shared by every returned feature in the same batch row:

```text
output[b, 0] uses max[q](x[b, q])
output[b, 7] uses max[q](x[b, q])
output[b, 12] uses max[q](x[b, q])
```

The maximum does not depend on `j`. It is broadcast across the feature being
returned, and the notation shows why.

## Coordinate Reading

For one batch item and one feature:

```text
output[b, 7]
```

the numerator uses `x[b, 7]`. The denominator scans every `x[b, k]`. The
stabilizing maximum scans every `x[b, q]`. The batch coordinate is fixed
throughout.

The operation is not "broadcasting plus a reduction plus some exponentials" in
the abstract. It is a set of coordinate relationships:

```text
j  choose the feature being produced
k  collect all features for the denominator
q  collect all features for the stabilizing maximum
```

## Why This Helps

Softmax bugs often come from normalizing over the wrong axis. In anonymous
shape code, the difference between "over features" and "over batch" may be one
integer argument.

With visible coordinates, the wrong program looks different:

```text
bad[b, j] =
    exp(x[b, j]) / sum[bb](exp(x[bb, j]))
```

This normalizes across batch for each feature. Maybe a program wants that. Most
classifiers do not. The coordinate `bb` makes the mistake visible.

## The Shape of the Result

Softmax preserves the feature coordinate. It normalizes values along that
coordinate, but it does not remove it:

```text
input   x[..batch, j]
output  output[..batch, j]
```

This distinguishes softmax from a reduction such as:

```rust
let total[..batch] = sum[j](x[..batch, j]);
```

Both expressions scan features. Only one keeps the feature coordinate in the
result. In softmax, each feature receives a normalized value. In `sum`, all
features contribute to one value and then disappear.

That distinction is easy to blur when both operations are described as acting
"over an axis." The names make the difference mechanical: if `j` appears on
the left, `j` survives. If `j` appears only inside `sum[j]`, `j` leaves.

## What Gradients Will Notice

Softmax is not elementwise in the feature coordinate. The value at `output[j]`
depends on all `x[k]` in the same row through the denominator and the maximum.
A derivative request must respect that coupling.

The visible coordinates warn the reader before calculus begins. The expression
itself says that one output feature reaches across the whole feature row. That
is why a softmax Jacobian has interactions among features rather than a purely
diagonal elementwise shape.

## Which Coordinates Stay Fixed?

The real softmax line turns on one question: which coordinates remain fixed
while the denominator is computed? The answer is `..batch`. The feature
coordinate is being scanned locally. That one distinction is the difference
between a per-example normalization and a cross-example one, and it is exactly
the kind of distinction a shape tuple cannot explain by itself.

## Why Softmax Belongs Here

Softmax belongs here because it combines the previous two ideas. It broadcasts
the stabilizing maximum across the feature coordinate being returned, and it
reduces the denominator across that same feature role. The coordinate is
conceptually one axis, but operationally it appears in several scopes.

That makes softmax a good stress test for the reading method. A simple slogan
does not carry enough detail:

```text
softmax over features
```

The real expression asks the reader to distinguish the feature being produced,
the features being summed, and the features being scanned for the maximum. If
those roles are confused, the formula may still have plausible shapes while
doing the wrong normalization.

A production tensor language needs to handle this kind of ordinary complexity.
Softmax is important here because one mathematical axis plays multiple local
roles inside a stable numerical formula.

Carry this forward: when one dimension appears more than once in an
explanation, ask whether those appearances have the same scope. If they do not,
give the scopes distinct names. That small act prevents many shape-correct but
meaning-wrong programs.

The same reading works for normalization layers. In LayerNorm, the feature
coordinates used to compute the mean and variance are local to the reduction,
while batch and often time-like coordinates survive. The learned scale and
shift parameters omit the surviving example coordinates and are therefore
broadcast. In GroupNorm, a channel coordinate is split into group and
within-group roles; the reduction scans only the within-group feature and
spatial coordinates for each group. The exercise is the same as softmax: name
the coordinate being returned, name the coordinates being scanned, and name the
parameters whose addresses are being reused.

Softmax also shows why visible dimensions cannot stop at shape checking. The
input and output have the same shape, so a pure shape story would say almost
nothing happened. But a great deal happened: every output feature was coupled
to every input feature in the same row, and the row was normalized into a
distribution.

The coordinate names reveal that hidden work. `j` survives, but it survives
after being compared against a denominator built from `k` and a stabilizing
maximum built from `q`. Same shape, different dependency structure. That is a
pattern the later chapters will reuse for gradients and attention.

## Same Shape, Different Dependency Graph

Use one batch row with three logits:

```text
x[b, 0] = 2.0
x[b, 1] = 1.0
x[b, 2] = 0.0
```

A stable softmax can be read as three coordinate stages:

```rust
let m[b] = max[q](x[b, q]);
let e[b, k] = exp(x[b, k] - m[b]);
let z[b] = sum[k](e[b, k]);
let y[b, j] = e[b, j] / z[b];
```

The maximum uses `q` because it scans the row to find a stabilizing reference.
For this row, `m[b] = 2.0`. The exponentials are then:

```text
e[b, 0] = exp(0.0)
e[b, 1] = exp(-1.0)
e[b, 2] = exp(-2.0)
```

The denominator uses `k` because it scans the row again to build the normalizer:

```text
z[b] = exp(0.0) + exp(-1.0) + exp(-2.0)
```

Finally `j` names the output feature being returned:

```text
y[b, 0] = e[b, 0] / z[b]
y[b, 1] = e[b, 1] / z[b]
y[b, 2] = e[b, 2] / z[b]
```

The three output cells are different, but they share the same `m[b]` and
`z[b]`. That is the coupling shape of softmax. The result still has coordinate
`[b, j]`, yet each `y[b, j]` depends on every input `x[b, k]` through the
denominator and on every input `x[b, q]` through the maximum. The hard part is
not computing a three-element softmax; it is remembering that same shape does
not imply same dependency graph.

This is exactly why reusing one name for every role is misleading. If the
formula wrote `j` for the output coordinate, the denominator coordinate, and
the maximum coordinate, the scopes would blur. The expression would look as if
one coordinate were doing one job. In reality, one logical feature axis is
visited in several local scopes. The names `q`, `k`, and `j` do not describe
different physical axes; they describe different binding sites over the same
range.

A useful bug check is to ask which coordinates remain fixed while the row is
normalized. Batch `b` is fixed. The distribution is over features. If the
formula accidentally scans `b` instead:

```rust
let z[j] = sum[b](e[b, j]);
```

then the program normalizes each feature across examples. That can produce the
same output shape `[b, j]`, but it answers a different question. It makes
examples compete with one another instead of making features compete inside an
example. The coordinate names make that difference visible at the denominator,
where the mistake is born.

The derivative story also begins here. Because each `y[b, j]` depends on all
`x[b, k]` in the same row, the pullback through softmax is not purely
elementwise. A change to `x[b, 1]` can change `y[b, 0]`, `y[b, 1]`, and
`y[b, 2]`. The output shape hides that coupling; the coordinate reading exposes
it.

## What Same Shape Hides

Softmax is a useful stress test because it preserves rank and extents:

```text
input   x[b, j]
output  y[b, j]
```

A pure shape story might describe this as an elementwise operation from a
matrix to a matrix. That would be false. The coordinate `j` survives into the
answer, but the value at one `j` was computed using all feature positions in
the row. The dependency structure is row-global even though the output address
is local.

That distinction matters to optimization and differentiation. A backend may
fuse the maximum, exponentials, sum, and division into one stable kernel, but
it must preserve the fact that the normalization scope is per `b` and over
feature positions. A differentiator must preserve the same coupling when
sensitivities flow backward. A test that checks only output shape would miss
the central contract.

So the chapter's practical advice is: whenever an operation returns the same
shape it received, do not assume it was coordinatewise. Ask which other
coordinates each output cell inspected before it returned.

A final concrete check is to compare softmax with a true elementwise sigmoid:

```rust
let s[b, j] = 1 / (1 + exp(-x[b, j]));
```

Here `s[b, j]` reads only `x[b, j]`. No `k` or `q` scans the row. The output
shape matches the input shape, just as softmax does, but the dependency
structure is different. A change to `x[b, 1]` changes `s[b, 1]` only; it can
change every `y[b, j]` in a softmax row. Same shape, different graph. That is
why this chapter spent so much time on scopes instead of only on extents.

Once the scopes are clear, the stable formula no longer looks like an
implementation trick. It becomes a sequence of coordinate claims: choose a row
maximum, exponentiate each feature against it, sum the row, and return one
normalized coordinate.

For one cell `output[b, j]`, the numerator and denominator tell different
stories. Name every coordinate they read; the surviving coordinate and the
normalizing coordinate are not doing the same job.

## Try It

Analyze LayerNorm the same way. Name the coordinate being returned, the
coordinates scanned to compute the mean, the coordinates scanned to compute the
variance, and the coordinates owned by the learned scale and shift. You should
find several roles even before writing a derivative.

**Line to keep:** practical structure omitted by a mathematical slogan is
exactly the structure source code must be able to carry.
