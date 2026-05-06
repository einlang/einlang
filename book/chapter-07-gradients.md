---
layout: book
title: "What Is a Gradient?"
---

# What Is a Gradient?

> "To every action there is always opposed an equal reaction."
>
> — Isaac Newton, *Principia Mathematica* (1687)

The forward pass computes a value. The backward pass computes its reaction:
sensitivity flowing in reverse through every operation, equal in magnitude,
opposite in direction. The coordinate names that guided the forward pass must
survive the reaction, or the gradient finds the wrong target.

Part I taught us to notice when a coordinate role is hidden. Part II asks what
happens because of it. A forward term that omits `batch` produces a gradient
that sums over `batch`. A reduction that consumes `class` produces a pullback
that fans out over `class`. Every sharing decision in the forward pass becomes a
collection operation in the backward pass. The notation either names the
sharing, or the gradient inherits the silence — and a silent gradient bug is the
hardest bug to find.

You trained a neural network for three days. The loss went down. You deployed.

Then a colleague asks: "Did the gradient flow through the attention mask
correctly?"

You used `loss.backward()`. The numbers looked fine. The loss curve was smooth.
But `backward()` doesn't prove the gradient went through the right
coordinates—only that it went through *some* coordinates that produced a
nonzero result. A gradient that leaks across the batch dimension can still
produce a perfectly smooth loss curve. It's just optimizing the wrong question.

This chapter shows how names make gradient shapes predictable before a single
number is computed.

## One Cell, One Route

Start with the simplest case. A linear map:

```rust
let y[i] = sum[j](W[i, j] * x[j]);
```

The output coordinate is `i`. The reduction coordinate is `j`. The parameter
`W` is addressed by both.

Now ask: if some scalar `loss` depends on `y`, what is the sensitivity of `loss`
with respect to `W`?

```rust
let dloss_dW = @loss / @W;
```

Pick one cell of `W`. Say `W[4, 9]`. Which output cells notice a small change
to this one cell?

Only `y[4]` notices. Because `i=4` selects the row. Inside that row, the
changed coefficient multiplies `x[9]`. So:

```text
change in y[4] = change in W[4, 9] * x[9]
```

The loss sees that change through the sensitivity already sitting at
`dloss_dy[4]`. The full formula is:

```rust
let dloss_dW[i, j] = dloss_dy[i] * x[j];
```

The gradient wears the same coordinate address as the thing it is about. `W`
lives at `[i, j]`, so its gradient also lives at `[i, j]`. The output
sensitivity `dloss_dy[i]` follows the output coordinate. The input factor
`x[j]` follows the consumed coordinate from the forward expression.

This is the chain rule with the coordinate structure left on the page. No
`backward()` call needed to guess the shape. The names have already answered.

```
   Gradient Address Diagram

   Forward: y[i] = sum[j](W[i,j] * x[j])
   +-----+-----+-----+     +--------+
   | W   | j=0 | j=1 |     | y      |
   +-----+-----+-----+     +--------+
   | i=0 | W00 | W01 | --> | y[0]   |  dloss/dy[0] arrvies here
   +-----+-----+-----+     +--------+
   | i=1 | W10 | W11 | --> | y[1]   |  dloss/dy[1] arrives here
   +-----+-----+-----+     +--------+

   Gradient w.r.t. W:
   +-----+-----+-----+
   | dW  | j=0 | j=1 |
   +-----+-----+-----+
   | i=0 |dy[0]|dy[0]|  dW[i,j] = dy[i] * x[j]
   |     |*x[0]|*x[1]|
   +-----+-----+-----+
   | i=1 |dy[1]|dy[1]|
   |     |*x[0]|*x[1]|
   +-----+-----+-----+

   The gradient address [i,j] IS the coordinate contract.
   Sensitivity flows to W's address through the output cell.
```

## Change the Question, Change the Shape

Now ask for sensitivity with respect to `x` instead:

```rust
let dloss_dx = @loss / @x;
```

Pick one cell of `x`. Say `x[2]`. Which output cells notice? This time: every
single one. `x[2]` participates in `y[0]` (through `W[0, 2]`), in `y[1]`
(through `W[1, 2]`), and so on.

So the sensitivity to `x[2]` must collect contributions over `i`:

```text
dloss_dx[2] =
    dloss_dy[0] * W[0, 2] +
    dloss_dy[1] * W[1, 2] +
    ...
```

The indexed rule is:

```rust
let dloss_dx[j] = sum[i](dloss_dy[i] * W[i, j]);
```

The result has only coordinate `j`, because `x` has only coordinate `j`. The
coordinate `i` appears in the path from `loss` through `y`, but `i` is not an
address of `x`. It must be summed away.

Contrast the two derivative shapes:

```text
@loss/@W  → [i, j]     W's coordinates, both survive
@loss/@x  → [j]        x's coordinate, i is consumed
```

The denominator determines the shape of the answer. Not the tape. Not the
runtime. The binding name on the left of `@loss / @W`.

This is the first deepening of the book. Part I taught us to read the coordinate
audit forward — which names survive, which are consumed, which are omitted. Now
we see the same audit run backward. The gradient denominator names the address
where sensitivity collects. The path coordinates — those in the output but not
in the denominator — name the sums. Every omission in the forward pass becomes a
sum in the backward pass. The audit is the same. The direction is reversed.

## The Inversion Rule

Chapter 4 showed broadcasting. Chapter 5 showed reduction. Gradients reveal
that these two are inverses.

```rust
// Forward: bias[j] is reused across every i
let y[i, j] = x[i, j] + bias[j];

// Backward: bias[j] collects sensitivity from every i
let dbias[j] = sum[i](dy[i, j]);
```

A value broadcast along a coordinate receives a gradient summed over that
same coordinate. This is not a special case for biases. It is the coordinate
statement of the chain rule.

Every forward broadcast becomes a backward reduction. The omitted coordinate
in the forward term becomes the consumed coordinate in the backward gradient.

```text
Forward pattern:   term omits k  →  term is independent of k
Backward pattern:  term omitted k  →  gradient sums over k
```

Read your model. Find a broadcast. The backward sum is already written in the
coordinate structure—you just haven't named the reduction yet.

The inversion is not a new rule. It is the combination principle, read backward.
Every forward combination leaves a trace in the audit — a consumed coordinate, an
omitted one. The backward pass reads that trace and inverts the operation.
Broadcast becomes sum. Sum becomes broadcast. The primitive — survive, consume,
omit — is unchanged. Only the direction reversed.

The inversion rule is the Hiding Law applied to axes. The forward broadcast
is a statement: this term does not depend on `k`. The backward reduction is
the invoice for that decision: to collect sensitivity correctly, you must sum
over `k`. If the notation does not name `k` explicitly, the broadcast and the
reduction become invisible to each other — and the reader must guess which
axis to sum. The guess is often right. When it is wrong, the shapes still
match, the loss still descends, and the bug leaves no trace a shape checker
can find.

## The Denomininator Rule

Here is the practical rule for the entire autodiff section. Write this down:

```text
The denominator of @loss / @X determines the coordinate shape
of the answer. A gradient with respect to a scalar has no free
coordinates. A gradient with respect to a vector has the vector's
coordinate. A gradient with respect to a matrix has the matrix's
two coordinates.
```

Other coordinates may appear while the derivative is being computed. They are
path coordinates, not addresses of the answer. If they do not belong to the
value named in the denominator, they must eventually be reduced or absorbed.

For the linear map:

```text
dloss_dW[i, j] = dloss_dy[i] * x[j]     // i, j survive; no path coordinates
dloss_dx[j]    = sum[i](dloss_dy[i] * W[i, j])  // j survives; i is path, sum it away
```

The formula that has no sum has no fan-out to collect. The formula that has a
sum has a path coordinate to collect. The coordinate names tell you which is
which before calculus begins.

This is the core question of Part II in miniature. The autodiff engine traces
backward through the computation graph, but it does not reason about the
computation graph — it replays it. The coordinate names do the reasoning in
advance, classifying every index as an answer coordinate or a path coordinate.
When the names are present, the classification is a set subtraction. When the
names are absent, the classification is a debugging session — and the debugger
must reconstruct from shapes what the source could have stated in names.

## Matmul: Three Coordinates, Two Gradients

The linear map is one parameter and one input. Matrix multiplication has two
inputs and one contraction coordinate. The forward expression:

```rust
let C[i, j] = sum[k](A[i, k] * B[k, j]);
```

Coordinate ledger:

```text
Survivors: i, j (appear in C[i, j])
Local:     k    (introduced by sum[k], consumed)
```

Before running any numbers, the denominator rule gives the shapes:

```text
@loss/@C  → [i, j]     C's coordinates
@loss/@A  → [i, k]     A's coordinates
@loss/@B  → [k, j]     B's coordinates
```

Now fill in the formulas. For `@loss/@A`:

```rust
// A[i, k] helps produce every C[i, j] for all j
// The path coordinate is j → must be summed
let dloss_dA[i, k] = sum[j](dloss_dC[i, j] * B[k, j]);
```

For `@loss/@B`:

```rust
// B[k, j] helps produce every C[i, j] for all i
// The path coordinate is i → must be summed
let dloss_dB[k, j] = sum[i](dloss_dC[i, j] * A[i, k]);
```

The consumed coordinate `k` reappears in both gradients. One gradient sums over
`j`. The other sums over `i`. The coordinate names decide which sum goes where.
A positional API encodes this as `axis=1` versus `axis=0` and hopes the reader
can reverse-engineer the intention from the number.

The named version gives the reader a place to verify:

```text
A[i, k] contains i and k
C[i, j] contains i and j
Missing from A: j → sum over j → dloss_dA[i, k] = sum[j](...)
```

This is not advanced calculus. It is set subtraction on coordinate names.

When the notation records only dimension counts, the reader must check: does
this sum consume the contraction coordinate or the batch coordinate? The
shapes cannot answer — a `3` in position 1 and a `3` in position 2 are the
same integer. The names can answer, because `j` and `batch` are different
facts. The difference between a bug and a derivation is whether the notation
holds a place for the answer.

## The Batch Prefix Survives

Add a batch dimension:

```rust
let C[batch, i, j] = sum[k](A[batch, i, k] * B[batch, k, j]);
```

Same story. The batch coordinate is not part of the contraction. It survives
untouched into every derivative:

```rust
let dloss_dA[batch, i, k] = sum[j](dloss_dC[batch, i, j] * B[batch, k, j]);
let dloss_dB[batch, k, j] = sum[i](dloss_dC[batch, i, j] * A[batch, i, k]);
```

The differentiation does not need to "discover" that batch is carried. The
source separated batch from contraction before autodiff ran.

## What the Compiler Knows

By the time the autodiff pass runs, the compiler already knows:

```text
- The coordinate domains of every binding (range analysis)
- The shape of every value (shape analysis)
- The scalar type of every arithmentic operation (type inference)
```

So when the compiler encounters:

```rust
let dC_dA = @C / @A;
```

it does not guess the answer shape. It reads the coordinate set of `A` from the
compiler's own bookkeeping. It reads the coordinate set of `C`. It computes the
difference: path coordinates = coordinates of `C` ∖ coordinates of `A`. It
inserts a sum over every path coordinate. The formula falls out mechanically.

This is the difference between differentiation as a tape replay and
differentiation as a source transformation. The tape records what happened. The
source already states where sensitivity can go. With named coordinates, the
source states more.

## Try It

The fastest way to understand a gradient is to trace one cell. Pick `W[4, 9]` in
the linear layer from this chapter:

```text
let y[i] = sum[j](W[i, j] * x[j]);
```

List every output cell that feels a change to this weight. For each output cell
`y[i]`, write the route that sensitivity takes: `W[4, 9]` appears in `y[4]`
(through `x[9]`), and nowhere else. So `dloss/dW[4, 9] = dloss/dy[4] * x[9]`.

Now do the reverse: pick `x[2]`. Which output cells feel a change to this input?
Every `y[i]` for all `i`, because `x[2]` participates in every output through
`W[i, 2]`. So `dloss/dx[2] = sum[i](dloss/dy[i] * W[i, 2])`. The reduction over
`i` is the path coordinate — it is not part of `x`'s address so it must be summed
away.

This is the core gradient reflex: one denominator cell, trace its output
influence, collect the routes. The reduction coordinate is always the one the
denominator does not own.

A colleague writes the weight gradient for `y[i] = sum[j](W[i, j] * x[j])` as:

```python
dW = dy.unsqueeze(-1) * x.unsqueeze(-2)
```

They produce shape `[i, j]`. Does this match `dW[i, j] = dy[i] * x[j]`? Write
the coordinate audit:

```text
Forward: y[i] = sum[j](W[i, j] * x[j])

W owns: {i, j}
y owns: {i}
Path: y's coords \ W's coords = {i} \ {i, j} = {} (empty! no sum needed)

dW[i, j] = dy[i] * x[j]
```

Wait — the path is empty. No sum is needed. Why? Because every output coordinate
appears in W's address. The answer: `dW[i, j]` correctly receives `dy[i]` (the
output sensitivity at the row where W was used) multiplied by `x[j]` (the input
that W scaled). There is no fan-out — each `W[i, j]` affects exactly one output
cell `y[i]`.

Now repeat for `dx`. `x` owns `{j}`. `y` owns `{i}`. Path `= {i} \ {j} = {i}`.
So `dx[j] = sum[i](dy[i] * W[i, j])`. The positional code for this would be
`dx = dy @ W` — but verify that the coordinate audit and the shape both agree.

Which version would a positional autodiff system get right? Which one risks
confusion? The `dW` formula has no sum, so shape alignment alone suffices. The
`dx` formula has a sum — over a coordinate that a positional system may number
differently after upstream reshapes.

Next, consider a more complex expression:

```text
let y[b, i, j] = sum[k](A[b, i, k] * B[b, k, j]) + C[i, j];
```

Derive `dA`, `dB`, and `dC` by hand. Use the coordinate accounting method, not
calculus memorization.

Read each operand's coordinates from the forward expression:

```text
A[b, i, k]  → owns {b, i, k}
B[b, k, j]  → owns {b, k, j}
C[i, j]     → owns {i, j}
```

Read `y`'s coordinates: `{b, i, j}`.

Compute path coordinates and write pullbacks:

```text
dA[b, i, k]: path = {b, i, j} \ {b, i, k} = {j}
  dA[b, i, k] = sum[j](dy[b, i, j] * B[b, k, j])

dB[b, k, j]: path = {b, i, j} \ {b, k, j} = {i}
  dB[b, k, j] = sum[i](dy[b, i, j] * A[b, i, k])

dC[i, j]: path = {b, i, j} \ {i, j} = {b}
  dC[i, j] = sum[b](dy[b, i, j])
```

Here is the key insight: `dC[i, j]` receives sensitivity from *every* batch
item. `C` is a parameter shared across the batch — it does not own `b` in the
forward expression, so the gradient *must* sum over `b`. This sum is invisible in
a positional formula `dC = dy.sum(dim=0)` — the reader must know that `dim=0` is
batch. But the coordinate accounting derives it: `{b, i, j} \ {i, j} = {b}`.
No memorization. No guessing which axis is batch. Just set subtraction on
coordinate names.

Gradient shapes come from coordinate accounting, not from memorization. The
denominator's coordinates and the output's coordinates, compared as sets, give
you every reduction. No calculus required.

**Line to keep:** a gradient is not magic; it is a readable communication
protocol in the source language.

### Where This Leads

Part II turns the coordinate audit from Part I onto a new target: automatic
differentiation. In Part I we asked which coordinates survive and which are
consumed. In Part II we ask a deeper question: if a small change to one cell
propagates through the computation, which output cells feel it? The answer
depends entirely on which coordinates that cell was shared across in the forward
program. A gradient is not magic — it is the forward program's coordinate
sharing decisions, collected and summed back. The denominator's coordinates and
the output's coordinates, compared as sets, give you every reduction before a
single number is computed.

We have seen the simplest case: a scalar loss, a linear map, and a single
denominator. The gradient shape follows from coordinate names. But linear maps
are easy. In Chapter 8, the forward expression is matrix multiplication — two
inputs, three coordinates, one contraction — and the derivative question asks
for both sides. You will see exactly how `k` reappears in each gradient, why
`@loss/@A` reduces over `j` while `@loss/@B` reduces over `i`, and how the
pullback structure falls out of the forward formula.