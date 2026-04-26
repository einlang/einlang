---
layout: book
title: "Chapter 9: Local Derivatives, Global Shape"
---

# Local Derivatives, Global Shape

Not every derivative announces itself with a matrix multiply. Many are local
elementwise facts:

```rust
let y[i] = x[i] * x[i];
```

For a scalar loss depending on `y`, the gradient with respect to `x` is:

```rust
let dx[i] = dy[i] * 2 * x[i];
```

No coordinate is reduced. The derivative at `x[i]` depends on the sensitivity
at `y[i]` and the primal value `x[i]`. The coordinate `i` simply survives.

This is the other side of the gradient story. Reductions create summed
pullbacks. Elementwise operations preserve coordinates.

Keep the main duality close:

```text
Forward broadcast  <->  Backward reduce
Forward reduce     <->  Backward broadcast-like fan-out
```

Most surprising gradient shapes are one of these two lines with a real
coordinate name filled in.

The scalar rule is only half the answer. Local calculus tells us the value to
multiply by; the coordinate structure tells us where that value lands and which
shared or omitted routes must be collected on the way back.

## Scalar Rule or Shaped Rule

A derivative table can tell us that the derivative of `x * x` is `2 * x`. That
scalar fact is true, but it is not yet a tensor rule. A tensor language still
has to answer where that local derivative lives.

One design would keep local calculus rules separate from shape reasoning and
ask the autodiff engine to combine them later. Another design writes local
rules inside coordinate structure from the beginning. The second design makes a
simple distinction visible: some derivatives stay at the same address, while
others must collect influence from many addresses.

Einlang's visible indices give local calculus a place to sit. The scalar rule
does not disappear; it becomes one part of a shaped rule.

Implementation order matters here. Shape analysis and type inference run
before autodiff, so the differentiator is not guessing the rank of `x`, `bias`,
or `W` from a runtime trace. It receives IR whose ranges, shapes, and scalar
element types have already been checked as far as the source permits.

Give the same scalar derivative table to two programs:

```rust
let a[i] = x[i] * x[i];
let b[i] = x[0] * x[0];
```

The local derivative of squaring is the same, but the address story is not. In
the first line, each output reads its matching input. In the second, every
output reads the same input cell. A scalar table alone cannot tell the
difference.

## Elementwise Means Same Address

For an elementwise operation, each output coordinate depends on the matching
input coordinate:

```rust
let y[i] = relu(x[i]);
```

Ignoring the nondifferentiable point at zero for the moment, the pullback has
the same coordinate:

```text
dx[i] = dy[i] * if x[i] > 0 { 1 } else { 0 }
```

The derivative is local. The value at `x[7]` affects `y[7]`, not `y[3]` or
`y[12]`. The gradient does not need a reduction because no output coordinate
fan-out occurred in the forward expression.

This is a useful baseline. When a gradient expression contains a `sum`, ask
what forward dependency caused one input coordinate to influence multiple
output coordinates. When it does not contain a `sum`, the forward relationship
was probably coordinatewise.

## Broadcasting in a Derivative

Now add a scalar bias:

```rust
let y[i] = x[i] + b;
```

The gradient with respect to `x` keeps `i`:

```rust
let dx[i] = dy[i];
```

The gradient with respect to `b` is scalar, so `i` must be collected:

```rust
let db = sum[i](dy[i]);
```

The forward expression did not mention `i` in `b`; therefore `b` was broadcast
across `i`. The backward expression reverses that broadcast by reducing over
`i`.

This is a useful sanity check. Broadcast in the forward direction usually means
sum in the reverse direction.

The accumulation is clearest with three outputs:

```text
y[0] = x[0] + b
y[1] = x[1] + b
y[2] = x[2] + b
```

The scalar derivative of each line with respect to `b` is `1`. The global
gradient is not three separate scalars; it is one scalar receiving three
routes of sensitivity. The coordinate `i` explains why they meet.

## Why Broadcast Reverses to Sum

The scalar `b` contributes to every output coordinate:

```text
y[0] = x[0] + b
y[1] = x[1] + b
y[2] = x[2] + b
```

If the loss changes through all three outputs, the sensitivity to `b` is the
sum of those routes. The forward pass reused one value many times. The reverse
pass collects many sensitivities back into one value.

This is not a special case bolted onto autodiff. It follows from coordinates.
The forward term `b` omitted `i`, so it was invariant along `i`. The backward
question `@loss / @b` has no `i` coordinate, so the `i` contributions must be
reduced.

## A Shaped Bias

For a feature bias:

```rust
let y[b, f] = x[b, f] + bias[f];
```

the bias gradient keeps `f` and reduces over `b`:

```rust
let dbias[f] = sum[b](dy[b, f]);
```

For a batch bias:

```rust
let y[b, f] = x[b, f] + bias[b];
```

the bias gradient keeps `b` and reduces over `f`:

```rust
let dbias[b] = sum[f](dy[b, f]);
```

Same rank, different meaning. The coordinate names decide the shape of the
gradient.

## Local Nonlinearities in a Larger Program

Most neural-network layers mix both patterns. A dense layer creates a
contraction:

```rust
let z[b, out] = sum[in](x[b, in] * W[out, in]) + bias[out];
```

An activation then preserves the coordinates:

```rust
let y[b, out] = relu(z[b, out]);
```

The pullback through `relu` is local in `[b, out]`. The pullback through the
matrix multiply reduces over whichever coordinates do not belong to the
requested parameter. The pullback through `bias[out]` reduces over `b`.

The global gradient is therefore assembled from local facts:

```text
relu      preserve [b, out]
bias      collect b, keep out
weights   collect b, keep out and in
input     collect out, keep b and in
```

The compiler may implement this through a graph, a tape, or symbolic
transformation. The reader's model can stay simpler: local operations preserve
addresses; broadcasts collect omitted coordinates; contractions collect the
coordinates that no longer survive.

## When a Local Rule Becomes Global

An operation can look elementwise while still hiding a global shape effect. If
one term omits an output coordinate, the forward pass broadcasts it. The
gradient will later have to collect that omitted coordinate. The local
calculus rule may be simple, but the coordinate story decides where the
sensitivity accumulates.

## Local Rule, Global Shape

The gradient story separates two forces. Local calculus decides the scalar
derivative at one coordinate. Coordinate structure decides how that local fact
is shaped, broadcast, or reduced across the whole tensor.

For:

```rust
let y[i] = x[i] * x[i];
```

the scalar derivative is `2 * x[i]`, and the coordinate structure is
one-to-one. For:

```rust
let y[i] = x[i] + b;
```

the scalar derivative with respect to `b` is `1`, but the coordinate structure
says that one scalar `b` influenced every `i`. Therefore the global gradient is
not simply `1`; it is `sum[i](dy[i])`.

This distinction prevents a common misunderstanding. Gradients are not only
calculus facts, and they are not only graph facts. They are calculus facts
placed inside coordinate structure. Visible dimensions give that structure a
source-level form.

This also explains why scalar examples can be misleading. A scalar derivative
like:

```text
d(x * x) / dx = 2 * x
```

teaches the local calculus rule but hides the shape question. Tensor programs
need both. The local rule says what happens at one coordinate. The coordinate
structure says how many such coordinates exist and whether they interact.

For an elementwise square, each coordinate is independent. For softmax, one
output coordinate depends on many input coordinates. For a broadcast bias, many
output coordinates depend on one input coordinate. These are different global
shapes built from simple local derivative facts.

The rule of thumb is simple: first identify the local derivative, then ask how
coordinates carry that derivative across the whole expression.

That rule is deliberately procedural. It gives a concrete check before
trusting a framework result. Pick one coordinate. Ask which output
coordinates it affects. Then generalize the answer back into an indexed
formula. If that hand reading disagrees with the gradient shape, the bug is
usually in the coordinate story, not in calculus.

## Local Rule Inside a Shared-Parameter Layer

Consider a layer that combines a matrix multiply, a feature bias, and a square.
The scalar derivative is easy; the pressure comes from deciding which
coordinates a shared parameter must collect:

```rust
let z[b, out] = sum[in](x[b, in] * W[out, in]) + bias[out];
let y[b, out] = z[b, out] * z[b, out];
let loss = sum[b, out](y[b, out]);
```

The local derivative of the square is:

```text
dy_dz[b, out] = 2 * z[b, out]
```

Because `y[b, out]` depends only on `z[b, out]`, that part is elementwise. No
coordinate is reduced. The incoming sensitivity from `loss` is `1` at every
`[b, out]`, so the cotangent at `z` is:

```text
G[b, out] = 2 * z[b, out]
```

Now the shaped rules diverge. The bias is addressed only by `out`, so it
receives all batch contributions:

```rust
let dbias[out] = sum[b](G[b, out]);
```

The weight is addressed by `[out, in]`. Each weight cell is used for every
batch example with the corresponding input feature:

```rust
let dW[out, in] = sum[b](G[b, out] * x[b, in]);
```

The input is addressed by `[b, in]`. Each input cell contributes to every
output feature through the corresponding column of `W`:

```rust
let dx[b, in] = sum[out](G[b, out] * W[out, in]);
```

This one layer contains three different global shapes built from one local
fact. The square preserves `[b, out]`. The bias gradient reduces `b`. The
weight gradient reduces `b` but keeps `out` and `in`. The input gradient
reduces `out` but keeps `b` and `in`.

A scalar derivative table can tell you that the derivative of `z * z` is
`2 * z`. It cannot tell you whether `dbias` should sum over `b`, whether `dW`
should sum over `in`, or whether `dx` should keep `out`. Those answers come
from the coordinate structure around the local rule.

This is a useful debugging procedure for real models. When a gradient has the
wrong magnitude, do not start by suspecting calculus. Ask whether a broadcast
caused an unintended sum. A bias shared across batch should receive a batch
sum; a bias accidentally shared across features will receive a feature sum. A
parameter used across time will collect time contributions. If the sharing was
intended, the sum is right. If the sharing was accidental, the gradient exposes
the mistake.

The implemented compiler order supports this reading. Shape and type analysis
have already determined the rectangular ranks and element types before the
autodiff pass expands requests. The differentiator is therefore composing
local rules inside a checked coordinate environment. That is the difference
between "the derivative formula exists somewhere" and "the derivative formula
has a source-level shape that can be inspected."

## Debug Checklist for Non-Scalar Gradients

For a suspicious gradient, read the forward program in four passes:

```text
1. Which operation supplies the local scalar derivative?
2. Which coordinates does the denominator value own?
3. Which output coordinates did one denominator cell influence?
4. Which of those routes must be summed back together?
```

In the mixed layer above, the square answers the first question. The parameter
`W[out, in]` answers the second. One weight cell influences all batch examples
for its `out` and `in` pair, so the third answer includes `b`. The fourth
answer is therefore `sum[b]`.

This checklist is intentionally mechanical. It keeps local calculus from
pretending to be the whole story, and it keeps shape reasoning from becoming a
separate mystery. The derivative is a scalar rule placed into a coordinate
network. If either part is read incorrectly, the gradient may have a plausible
rank while still answering the wrong question.

For an autodiff shape error, use the same checklist in reverse:

```text
If a gradient has an unexpected sum, find the forward term that omitted that
coordinate.
If a gradient is missing a coordinate, ask whether the denominator value owns
that coordinate.
If a broadcasted parameter has a surprising magnitude, check which forward
coordinate it was shared across.
If a reduction axis appears in a gradient, ask which input address preserved it.
```

The blunt version is this: if you never named the axis in the forward program,
the backward error is often the invoice.

The checklist also explains why small scalar tests are not enough. A scalar
test can confirm the derivative of `z * z`, but it cannot confirm that a bias
was shared over the intended coordinate. For tensor programs, numerical
correctness and coordinate correctness meet; neither one replaces the other.

That is why the chapter keeps returning to address questions. The derivative
value matters, but so does the coordinate where that value lands and the route
by which it arrived there.

Both facts must remain visible.

A shared parameter such as `W[out, in]` is the simplest place to see the issue.
A local derivative may be computed at each `b`, but the parameter has no `b`
coordinate. The gradient must say where that omitted coordinate went.

## Try It

Add a bias `bias[out]` to an activation `z[b, out]`, then compute a scalar
loss. Before deriving anything, predict whether `@loss / @bias` keeps `b`.
Then write the pullback. The common mistake is to preserve the coordinate that
was only present because of broadcasting.

**Line to keep:** forward omission becomes backward collection.
