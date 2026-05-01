---
layout: book
title: "Chapter 5: The Index That Leaves"
---

# The Index That Leaves

Reduction is what happens when a coordinate does its job and leaves.

In:

```rust
let dot = sum[i](u[i] * v[i]);
```

the coordinate `i` exists inside the `sum`. It chooses matching elements of
`u` and `v`. After the sum, it is gone. The result has no coordinate `i`
because the point of the reduction was to consume it.

That is a small idea. It is also the source of many tensor programs.

The important word is "leaves." A reduction is not just an operator with an
axis parameter. It is a local coordinate with a scope, a domain, and a reason
for disappearing from the result.

There is a family resemblance here to linear logic and linear type systems. A
linear variable is not casually reusable after it has been consumed. A
reduction coordinate is not a value variable, but the discipline has the same
flavor: `k` is introduced for local work, used to align terms, and then cannot
be read as a coordinate of the result. The notation makes that resource
boundary explicit.

A reduction should make its bargain explicit: some coordinates survive, one or
more coordinates are local to the `sum`, and the result no longer has them.
Square cases are the place where reducing the wrong one can stay quiet.

One reliable way to read a reduction is to circle the coordinate inside the
reducer and then try to use it after the reducer has returned. The attempt
should feel wrong in the same way that reading a local variable outside its
scope feels wrong. A reduction is an arithmetic operation, but it is also a
scope boundary.

## Axis Argument or Local Name

The common interface for reduction is an axis argument:

```text
sum(x, axis=-1)
```

This is compact, but it turns the reduced coordinate into a position. If
previous operations have moved axes around, the reader must reconstruct which role
`-1` names. A language can add comments or shape annotations, but the reduction
itself still consumes a number rather than a role.

Einlang chooses a local name:

```text
sum[j](x[..batch, j])
```

Now the consumed coordinate is introduced exactly where it leaves. The name
`j` does not survive into the result, and that absence is the point. The
syntax is not longer for ceremony; it is longer because later gradient and
shape reasoning will need to know which coordinate was consumed.

Imagine pausing a debugger after the reduction. The value `result[..batch]`
can tell you which batch member it belongs to, but it cannot tell you which
`j` contributed which amount. That is not lost metadata by accident. It is the
operation's meaning. Naming `j` locally makes the loss deliberate.

## Survivors and Locals

A reduction expression has two kinds of coordinates. Some are free in the
result. Some are local to the reducer. The free coordinates describe the family
being produced. The local coordinates describe the work done for each member of
that family.

Consider:

```rust
let row_total[row] = sum[col](A[row, col]);
```

For each `row`, the expression scans every `col`. The result has one value per
row because `row` remains free. The coordinate `col` never escapes the `sum`.

Now switch the names:

```rust
let col_total[col] = sum[row](A[row, col]);
```

The shape may look similar if the matrix is square, but the meaning is
different. The first program totals rows. The second totals columns. Axis
numbers can express the same distinction, but names make the distinction live
inside the formula.

## Matrix Multiplication

Matrix multiplication is the same move with two surviving coordinates:

```rust
let C[i, j] = sum[k](A[i, k] * B[k, j]);
```

The coordinates `i` and `j` survive because they appear on the left. The
coordinate `k` is local to the sum. For a concrete output point:

```text
C[2, 5] = sum[k](A[2, k] * B[k, 5])
```

That single coordinate reading is the whole operation. Row `2` of `A` meets
column `5` of `B`; `k` walks across their shared positions and disappears.

The lifecycle of `k` is worth writing once:

```text
k is introduced by sum[k]
k indexes A[i, k] and B[k, j]
k is consumed by the reduction
k is absent from C[i, j]
k can reappear in a gradient address such as dA[i, k] or dB[k, j]
```

Forward reduction removes `k` from the result. A later derivative request may
bring `k` back, not because reduction failed, but because the question has
changed from "what is `C`?" to "which input cell receives sensitivity?"

If `A` and `B` are square, a mistaken reduction can still produce a square
result. For example, reducing over `i` instead of `k` is not a small spelling
variation; it asks a different set of values to meet. The local name is what
lets a reviewer say "the inner coordinate is supposed to leave here."

## Why the Name Matters

Many libraries express reduction by an axis number:

```python
row_sum = A.sum(axis=1)
```

That is concise, but the meaning of `axis=1` depends on the surrounding shape
contract. If the axes move, the number may still be valid while the intention
is no longer valid.

With named coordinates:

```rust
let row_sum[row] = sum[col](A[row, col]);
```

the consumed coordinate is `col`, not "whatever currently sits at position 1."
The output keeps `row` because `row` is the coordinate left free.

## Multiple Indices Can Leave Together

Convolution uses the same pattern with more local coordinates. A two-dimensional
windowed sum can be written schematically as:

```text
out[b, co, i, j] =
    sum[ci, kh, kw](
        x[b, ci, i + kh, j + kw] * kernel[co, ci, kh, kw]
    )
```

The output keeps batch `b`, output channel `co`, and spatial position `i, j`.
The local coordinates `ci`, `kh`, and `kw` leave. They describe the input
channel and the two coordinates inside the kernel window.

This is a larger example, but no new reading rule is needed. Circle the
coordinates on the left. Those survive. Circle the coordinates introduced by
`sum`. Those are local. Everything else is a relationship between the two.

The standard library convolution code has more details: padding, dilation,
groups, and bounds. Those details matter for implementation, but the center is
still a reduction over local coordinates that do not survive into the output.

## A Reducer in the Library

The standard library's `matmul` is exactly this:

```rust
let output[i, j] = sum[k](a[i, k] * b[k, j]);
```

The line is short because the idea is short. Two coordinates survive. One
coordinate leaves. Any later optimization can recognize the familiar matrix
multiply pattern, but the source remains a formula rather than a call with
implicit axis meaning.

## What a Reducer Promises

A reducer also carries an algebraic promise. `sum` adds values and has identity
`0`. `prod` multiplies values and has identity `1`. `max` and `min` choose
extreme values. These promises influence what optimizations are legal. A
compiler may reorder some sums under assumptions about arithmetic. It may use a
specialized matrix multiply kernel when it recognizes the contraction pattern.
It may lower a `max` differently from a `sum`.

Some operations need a second result if the consumed coordinate still matters.
`max[col](A[row, col])` returns the largest value in each row; it does not by
itself preserve which `col` won. If later code needs the winning coordinate,
the program should ask for that coordinate explicitly:

```rust
let winner[row] = argmax[col](A[row, col]);
```

This is the first selection-shaped coordinate function. It consumes `col` like
a reduction, but the result is not the maximum value. It is an integer address
in the `col` domain. This is another design boundary: reduction can consume a
coordinate, but a language should not smuggle the consumed coordinate back in
unless the source requested it.

Once that boundary exists, a user-defined function can preserve it:

```rust
fn top1[class](x: [f32; ..left, class, ..right])
    -> [i32; ..left, ..right]
{
    argmax[class](x[..left, class, ..right])
}
```

The caller writes `top1[class](logits)`. The function body may hide the local
selection loop, but the signature still says that `class` is consumed and that
all surrounding coordinates survive. That is the reusable form of "the index
that leaves."

This same pattern will recur: the operation may be compact, but the consumed
coordinate remains visible in brackets. That is what keeps `argmax[col]` from
being just another positional helper.

The source does not have to prescribe the schedule. It only has to state which
coordinate is local and which reduction is being performed. That is enough for
the program to be read as a formula and for later passes to choose an
implementation.

## What Did the Result Forget?

In this expression:

```rust
let col_norm[j] = sqrt(sum[i](A[i, j] * A[i, j]));
```

the surviving coordinate is `j`; the coordinate `i` leaves through the sum.
The deeper point is that the result is not a compressed copy of `A`. It is a
new object whose addresses are columns. The row coordinate mattered while the
norm was computed, but the finished value has no place to store row-specific
information.

## The Shape of the Work

A reduction changes not only the result shape, but also the shape of the work.
For:

```rust
let C[i, j] = sum[k](A[i, k] * B[k, j]);
```

there is one reduction over `k` for every pair `[i, j]`. The surviving
coordinates define a family of independent reduction problems. This is why the
formula can later become a loop nest, a tiled kernel, or a call to a matrix
multiply library. The source has not fixed the schedule, but it has fixed the
work structure.

A reader can separate three layers:

```text
result family   C[i, j]
local work      sum[k](...)
implementation  loop, kernel, vectorized primitive, or backend call
```

The result family and local work belong in the source model. Implementation
belongs to lowering. Mixing those layers too early makes programs harder to
reason about. Hiding the consumed coordinate makes them harder to check.

Omitted coordinates and consumed coordinates are dual ways for a dimension to
leave the surface of an expression. Broadcasting reuses a value along a
coordinate it never mentioned. Reduction uses a coordinate locally and then
removes it from the answer. Together, they explain much of tensor shape
behavior without reaching for axis-number folklore.

There is also a useful semantic test: after a reduction, you should not expect
to recover the individual pieces. A row sum does not remember which column
contributed which amount. A dot product does not remember which coordinate
made the vectors align strongly. A maximum may remember a value, but unless the
program explicitly asks for an index as well, it does not preserve the whole
input row.

That loss is not a flaw. It is the purpose of the operation. Reduction is how a
program says "this local coordinate mattered while computing the value, but it
is not part of the answer." Once you read reductions that way, many tensor
programs become less mysterious: the result shape is simply the list of
coordinates that were not eaten.

## A Lowering Sketch

The reduction:

```rust
let r[i] = sum[j](A[i, j]);
```

can lower to a loop shape like:

```text
for i in range(rows):
    acc = 0
    for j in range(cols):
        acc += A[i, j]
    r[i] = acc
```

The loop nest contains both `i` and `j`, but only `i` appears in the output
address. That is the operational version of the source rule. The accumulator
exists because `j` is local. The final store uses only survivor coordinates.

This sketch also shows why the source should keep reducer names explicit.
Changing the order of loops may be legal for a simple sum, but changing which
coordinate is local is not the same transformation. A compiler can optimize
execution only after it knows the semantic boundary between survivor and local
work.

## Reduction Direction in a Square Matrix

Start with a square-matrix case where the wrong reduction can keep a plausible
shape:

```text
A[0, 0] = 1   A[0, 1] = 2   A[0, 2] = 3
A[1, 0] = 4   A[1, 1] = 5   A[1, 2] = 6
```

The row sum is:

```rust
let r[i] = sum[j](A[i, j]);
```

The coordinate `i` survives because it appears on the left. The coordinate
`j` is local to the reduction. Expanding the two output cells gives:

```text
r[0] = A[0, 0] + A[0, 1] + A[0, 2]
r[1] = A[1, 0] + A[1, 1] + A[1, 2]
```

So `r = [6, 15]`. The result did not "drop axis 1" in a mysterious way. It
kept the row coordinate and consumed the column coordinate.

The column sum uses the same data but a different local coordinate:

```rust
let c[j] = sum[i](A[i, j]);
```

Now the expansion is:

```text
c[0] = A[0, 0] + A[1, 0]
c[1] = A[0, 1] + A[1, 1]
c[2] = A[0, 2] + A[1, 2]
```

The result is `[5, 7, 9]`. Again, the shape follows the survivor. The
coordinate on the left is the one still available after the reduction.

This example is small, but it captures the compiler rule. If `j` appears as the
second index of `A[i, j]`, range analysis can infer that `j` ranges over the
second extent of `A`. If the same `j` also indexed another array with a
different extent, the compiler would have a shape agreement question before
runtime. The local name is therefore not only explanatory. It is how the
compiler ties together all uses of the same reduction variable.

A common bug is to reduce the wrong coordinate in a square matrix, where shape
does not help:

```rust
let maybe_rows[i] = sum[j](A[j, i]);
```

If `A` is `3 x 3`, this program is shape-compatible with a row sum, but it is
actually a column sum addressed with the name `i`. The visible reading catches
the contradiction: the surviving coordinate `i` appears in the second position
of `A`, so it is not the row coordinate of the input. A positional API would
often describe both as `axis=0` or `axis=1` depending on surrounding layout.
The indexed form lets the reader inspect the exact address relation.

Reductions also make the cost of the computation visible. For each surviving
coordinate, the implementation must iterate over every local coordinate. The
shape of `r` is two cells, but the work includes six reads and four additions.
That distinction matters for lowering: the output family is addressed by
survivors; the loop nest also contains locals. A reducer promises both a result
shape and a work shape, and the index that leaves is the key to both.

The source reducer draws that boundary before any loop order is chosen.

## Range Inference Is the First Proof

A reduction variable is useful only if the compiler knows where it ranges. In:

```rust
let C[i, j] = sum[k](A[i, k] * B[k, j]);
```

the source does not explicitly write `k in 0..K`. Range analysis recovers that
fact from uses of `k`. The access `A[i, k]` says that `k` ranges over axis `1`
of `A`; the access `B[k, j]` says that the same `k` ranges over axis `0` of
`B`. Those two inferred ranges must agree. If `A` is `[m, 64]` and `B` is
`[32, n]`, the compiler has a shape mismatch before execution.

Explicit ranges participate in the same system:

```rust
let head[t in 0..10] = x[t];
let tail[t in 10..20] = x[t];
```

Here the range does not have to be inferred from an indexed operand; the source
states it. But shape analysis still needs the range fact to know how many
cells the declaration covers and how literal-index clauses or adjacent ranges
combine.

Offset expressions add pressure:

```rust
let y[t in 1..T] = x[t - 1] + x[t];
```

The binder says `t` starts at `1`. That makes `t - 1` legal at the first
iteration. If the range started at `0`, the same expression would try to read
before the beginning of `x`. Range analysis has to run before shape analysis
because output extents and index safety depend on the domain of the coordinate,
not only on the rank of the arrays being indexed.

Range inference is therefore the first proof the compiler attempts. It proves
that each local coordinate has a domain, that repeated uses of the same
coordinate agree on that domain, and that derived index expressions can be
checked against known extents. Only then can the compiler ask what shape the
result family has.

Matrix multiply leaves a clean final test: mark the two surviving coordinates
and the one consumed coordinate, then swap the consumed one with a survivor.
Some shapes may still line up. The mathematical claim will not.

## Try It

In a traditional framework, write `sum(axis=n)` and then imagine later code
referring to that consumed axis as if it still existed. Now write the same
program in Einlang and make the mistake explicit: try to use the reduced
coordinate on the result. The useful failure is not numerical; it is the
compile-time discovery that a local coordinate has left scope.

**Line to keep:** a coordinate does its work, then disappears from the result.
