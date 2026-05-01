---
layout: book
title: "Chapter 7: What Is a Gradient?"
---

# What Is a Gradient?

Automatic differentiation is often introduced with a runtime graph. The forward
pass records operations. The backward pass walks those operations in reverse
and accumulates sensitivities.

That model is useful. It is also easy to confuse two things:

```text
execution history
derivative structure
```

Execution history says what happened. Derivative structure says where a small
change can go. Einlang's bet is modest: when the forward program exposes its
indices, some of the backward program is already sitting in the source.

The easy mistake is to collapse those two ideas. A program may execute
`a`, then `b`, then `c`; that order does not by itself say which coordinates of
`a` can influence which coordinates of `c`. Conversely, a derivative route can
be sparse even when the execution history looked dense. History records events;
the derivative records possible influence.

The failure to watch for is a gradient with a plausible numeric story but no
longer the shape of the question it was asked to answer. A tape can remember
what ran. It does not, by itself, make the derivative question's address
obvious. The notation `@target / @source` keeps both sides of that question in
view before the rewrite begins.

Before the notation, use a physical picture. Imagine `x[j]` as one stone under
your feet and `y[i]` as a height measured somewhere on a hill. A derivative
asks a local sensitivity question: if this one stone moves a little, which
height changes, and by how much? The notation `@y[i] / @x[j]` is that
one-to-one question with addresses attached. Reverse mode usually avoids
materializing every such question, but the address discipline is still there.

## Tape or Transformation

The usual implementation story for reverse-mode differentiation begins with a
record of execution. That record is valuable. It tells the system which
operations occurred and which intermediate values may be needed later. But a
tape-first explanation can make one mistake too easy: treating the history of
execution as if it were the source of the derivative's shape.

Another design is to treat differentiation as a source transformation wherever
the source has enough structure. If an expression says that `W[i, j]`
contributes to `y[i]` through a sum over `j`, the derivative does not need to
discover the coordinates of `W` by replaying the program. They are already in
the binding.

Execution graphs are still useful. Einlang changes what the language hands to
the differentiator before execution begins. The source can already say which
coordinates the derivative must keep and which ones it must collect.

In the implementation, `@` is not a wrapper around an external autodiff API.
The compiler parses differential and quotient requests, runs range, shape, and
type analysis first, then `AutodiffPass` expands those requests into ordinary
Einlang IR before Einstein lowering. `AutodiffRequestLoweringPass` clears the
remaining request form, and an autodiff leak check rejects any leftover `@`
artifact. So when the book calls autodiff a compile-time transformation, this
is not a metaphor; it is where the pass runs.

If an autodiff tape records only that a matrix multiply happened, it still
needs a rule to recover which coordinates of the inputs line up with which
coordinates of the output. If the source already says
`sum[j](W[i, j] * x[j])`, that recovery step becomes a reading of the formula.
The tape may still decide when values are available; the indices say where
sensitivity can go.

## A Small Linear Map

Take a linear map:

```rust
let y[i] = sum[j](W[i, j] * x[j]);
```

The output coordinate is `i`. The reduction coordinate is `j`. The parameter
`W` is addressed by both. The input `x` is addressed only by `j`.

Now suppose a scalar `loss` depends on `y`, and ask for the sensitivity of that
loss with respect to `W`:

```rust
let dloss_dW = @loss / @W;
```

The local pattern is not mysterious:

```rust
let dloss_dW[i, j] = dloss_dy[i] * x[j];
```

To see why, perturb one cell of `W`. If `W[p, q]` changes by a small amount,
which output cells of `y` notice? Only `y[p]` notices, because `p` selects the
row. Inside that row, the changed coefficient multiplies `x[q]`, because `q`
selects the input coordinate. So the local response is:

```text
change in y[p] = change in W[p, q] * x[q]
```

The loss sees that change through the sensitivity already sitting at
`dloss_dy[p]`. Rename `p, q` back to `i, j`, and the formula above follows.

The gradient wears the same coordinate address as the thing it is about. Since
`W` lives at `[i, j]`, its loss gradient also has a value at `[i, j]`. The
output sensitivity `dloss_dy[i]` follows the output coordinate, and the input
factor `x[j]` follows the consumed coordinate from the forward expression.

This is the chain rule, but with the coordinate structure left on the page.

If the full Jacobian were written for the linear map, its entries would be:

```text
@y[p] / @W[i, j] = x[j] when p = i, otherwise 0
@y[p] / @x[j]    = W[p, j]
```

The first object has coordinates `[p, i, j]`, but most entries are zero because
one parameter row only affects the matching output row. The pullback formulas
avoid materializing that sparse object by composing it immediately with the
incoming sensitivity.

One cell is enough to check the rule. For `W[4, 9]`, only `y[4]` notices the
change, and it notices it through `x[9]`. So:

```text
dloss_dW[4, 9] = dloss_dy[4] * x[9]
```

The general formula is just this cell reading with the names restored.

## Changing the Question Changes the Shape

Now ask for the sensitivity with respect to `x` instead:

```rust
let dloss_dx[j] = sum[i](dloss_dy[i] * W[i, j]);
```

The result has only the coordinate `j`, because `x` has only the coordinate
`j`. The coordinate `i` appears in the path from `loss` through `y`, but it is
not an address of `x`, so it must be summed away.

This is a useful habit: ask where the sensitivity is addressed.

```text
W[i, j]  receives a grid of sensitivities
x[j]     receives a vector of sensitivities
```

The derivative request is not an oracle that happens to return the right
shape. It is a source-level question about named bindings. The denominator of
the derivative request decides the coordinate shape of the answer.

That one sentence is the practical rule for the autodiff section. A gradient
with respect to a scalar has no free coordinates. A gradient with respect to a
vector has the vector's coordinate. A gradient with respect to a matrix has the
matrix's two coordinates. Other coordinates may appear while the derivative is
being computed, but they are path coordinates, not addresses of the answer. If
they do not belong to the value named in the denominator, they must eventually
be reduced, broadcast, or otherwise accounted for by the chain rule.

This is also why reverse-mode systems usually propagate a cotangent, not a
full Jacobian. The incoming sensitivity `dloss_dy[i]` already has the
coordinates of `y`; the pullback only has to translate that family into the
coordinates of the requested input. For the linear map above, the translation
to `x[j]` collects over `i`:

```text
incoming   dloss_dy[i]
requested  x[j]
bridge     W[i, j]
result     sum[i](dloss_dy[i] * W[i, j])
```

The coordinate view therefore explains both the mathematics and the
implementation shape. A compiler can avoid ever constructing the object with
coordinates `[i, j]` for every pair of output and input positions unless some
program explicitly asks for it.

## Matmul Makes the Shape Visible

The linear algebra standard library gives the derivative story a clean forward
surface. Matrix multiplication is written as:

```rust
let output[i, j] = sum[k](a[i, k] * b[k, j]);
```

This is from `stdlib/ml/linalg_ops.ein`. The important part here is not that
matrix multiplication exists. It is that the compiler sees the contraction
structure directly. The coordinates `i` and `j` survive. The coordinate `k` is
local to the `sum`.

That one line already foreshadows the shapes of several derivative questions:

```text
@loss / @output  has coordinates [i, j]
@loss / @a       has coordinates [i, k]
@loss / @b       has coordinates [k, j]
```

The chain rule supplies the missing reductions. For example, a gradient with
respect to `a[i, k]` must collect contributions over all `j`, because `a[i, k]`
helps produce every `output[i, j]`:

```text
dloss_da[i, k] = sum[j](dloss_doutput[i, j] * b[k, j])
```

The exact lowering may be optimized, fused, or delegated to a backend. The
coordinate story is already available before that choice.

## Batch Should Stay Batch

The batched version keeps the same idea while allowing arbitrary batch-shaped
prefixes:

```rust
let result[..batch, i, j] =
    sum[k](a[..batch, i, k] * b[..batch, k, j]);
```

Here `..batch` is not part of the contraction. It is carried through. The
coordinate `k` does the local work and disappears. The coordinates `i`, `j`,
and the entire batch prefix survive.

For one batch member, the formula is ordinary matrix multiplication:

```text
result[batch..., i, j] =
    sum[k](a[batch..., i, k] * b[batch..., k, j])
```

The derivative engine does not need to rediscover that batch structure is
carried unchanged. The source has separated batch structure from contraction
structure.

This is exactly the kind of fact a coordinate function must preserve if
`matmul` becomes a library boundary:

```rust
fn matmul[i, j, k](a: [f32; ..batch, i, k],
                  b: [f32; ..batch, k, j])
    -> [f32; ..batch, i, j]
```

The function may lower to a tuned kernel, but the contract still says that
`k` is consumed and `..batch` is not. Autodiff can ask for the pullback against
that contract rather than guessing from a positional call.

## Graphs Are History; Gradients Are Structure

Execution graphs remain useful implementation devices. A real compiler may
store intermediate values, choose reverse mode, fuse kernels, or keep some
derivative expressions lazy.

Visible coordinates shift the center of gravity. If the source already names
the coordinates, differentiation can begin as symbolic transformation over
index expressions. The compiler does not have to discover from a tape that `W`
has an `[i, j]` structure. The program already said so.

## What Shape Does the Question Ask For?

Read this line again:

```rust
let output[i, j] = sum[k](a[i, k] * b[k, j]);
```

Before thinking about any autodiff engine, the coordinate shapes of
`@loss / @a` and `@loss / @b` are already suggested by the denominator. The
interesting consequence is that part of the backward pass is not discovered
after execution. It is latent in the coordinate structure of the forward
program.

## The Limit of the Claim

Visible indices do not make all of automatic differentiation simple. Real
programs include conditionals, nonlinearities, library boundaries, mutation in
the host language, numerical stability tricks, and storage choices. A compiler
still has serious work to do.

The claim is narrower and more useful: visible indices make the shape of many
derivative questions predictable before execution. They let the reader see why
`@loss / @W` has the coordinates of `W`, why `@loss / @x` has the coordinates
of `x`, and why some coordinates must be reduced along the way.

This matters because gradient bugs are often shape bugs with semantic roots. A
parameter is shared across a coordinate that should have been separate. A bias
is broadcast across batch and then receives an unexpectedly large summed
gradient. A reduction removes a coordinate, and the backward pass has to put
that influence somewhere. These are not only implementation details of an
autodiff engine. They are coordinate facts about the program.

The derivative question can now be read like the earlier tensor questions.
When asking how one named value changes with respect to another, the
denominator tells which coordinates the answer must preserve. The chain rule
still supplies the computation, but the source already supplies much of the
shape.

## One Parameter Cell, Many Routes

Use a two-output linear map with three input features. The local reading is
short, but the point is fan-out: one denominator cell can influence many
numerator cells, and those routes have to meet again at the gradient address.

```rust
let y[i] = sum[j](W[i, j] * x[j]);
let loss = sum[i](y[i] * y[i]);
```

The forward coordinates say that `W[i, j]` contributes to exactly one output
coordinate `y[i]`, and within that output it is paired with `x[j]`. Now pick
one parameter, `W[1, 2]`. Perturbing it by a small amount changes only:

```text
y[1]
```

and the amount of change is scaled by:

```text
x[2]
```

The loss has local derivative `2 * y[1]` with respect to `y[1]`, so the one
cell reading gives:

```text
dloss_dW[1, 2] = 2 * y[1] * x[2]
```

Restoring the coordinate names yields:

```rust
let dloss_dW[i, j] = 2 * y[i] * x[j];
```

No runtime tape was needed to guess the address of the answer. The denominator
`@W` already says that the result must be addressed by `[i, j]`.

Now ask for `@loss / @x`. Pick `x[2]`. That input participates in every output
row:

```text
y[0] reads W[0, 2] * x[2]
y[1] reads W[1, 2] * x[2]
```

So the sensitivity to `x[2]` must collect contributions over `i`:

```text
dloss_dx[2] =
    2 * y[0] * W[0, 2] +
    2 * y[1] * W[1, 2]
```

The indexed rule is:

```rust
let dloss_dx[j] = sum[i](2 * y[i] * W[i, j]);
```

The reduction over `i` is not an optimization detail. It is the coordinate
fact that `x[j]` fans out into all output coordinates.

This example also explains why binding names matter in Einlang. The request:

```rust
let dloss_dW = @loss / @W;
```

names two existing bindings. The compiler can look up the source expression
that defines `loss`, the binding that defines `W`, and the coordinate facts
already attached by range and shape analysis. That is a different contract
from asking an external library to differentiate an anonymous callback. The
request lives inside the same source graph as the indexed formulas.

The detailed rule is simple enough to test by hand: choose one denominator
cell, list every numerator cell that can depend on it, multiply by the local
derivative along each route, and sum routes that meet at the same denominator
address. The compiler automates that accounting, but the coordinate names make
the accounting intelligible.

That intelligibility is the point: the generated derivative is not a separate
world from the source program. The difficulty is not the scalar derivative of
`x * x`; it is route accounting through a shaped expression where some
coordinates fan out and others remain local.

## Why Autodiff Waits for Shape Facts

The autodiff pass sits after range analysis, shape analysis, and type
inference. That order is not administrative. It changes what the differentiator
is allowed to assume.

For:

```rust
let C[i, j] = sum[k](A[i, k] * B[k, j]);
let dC_dA = @C / @A;
```

the differentiator should not have to rediscover that `k` is a valid reduction
coordinate, that the two uses of `k` agree in extent, or that `A` owns
coordinates `[i, k]`. Those facts are already compiler facts by the time
autodiff runs. The derivative request can therefore be expanded as a
transformation over checked IR rather than as a runtime investigation of a
tape.

Rest-pattern preprocessing matters to gradients for the same reason. In a
batched matmul:

```rust
let result[..batch, i, j] =
    sum[k](a[..batch, i, k] * b[..batch, k, j]);
```

the prefix is not a contraction coordinate. If the input rank is known, the
prefix has already been expanded into concrete batch coordinates before the
later passes need to reason about it. The gradient with respect to `a` should
preserve those batch coordinates and reduce over `j`, not treat `..batch` as a
single mysterious object.

So autodiff is a source transformation, but it is not an isolated one. It
depends on the earlier inference passes to supply domains for indices, shapes
for values, scalar types for arithmetic, and explicit slots for rest prefixes.
By the time the derivative is rewritten, the coordinate bookkeeping has
already been checked.

Before computing any derivative values, write down the address of the question
`@C / @A` for `C[i, j]` and `A[i, k]`. The target contributes one set of
coordinates; the value being differentiated with respect to contributes
another. The derivative question owns both.

## Try It

Derive a linear layer by hand:

```rust
let y[b, out] = sum[in](x[b, in] * W[out, in]) + bias[out];
```

Mark the roles of `b`, `in`, and `out`, then write the coordinate shape of
`@loss / @W` and `@loss / @bias` for a scalar loss. The value rule is ordinary
calculus; the interesting part is which coordinates must be collected.

**Line to keep:** a gradient is not magic; it is a readable communication
protocol in the source language.
