---
layout: book
title: "Chapter 8: Matrix Multiplication Teaches the Pullback"
---

# Matrix Multiplication Teaches the Pullback

The forward equation is:

```rust
let C[i, j] = sum[k](A[i, k] * B[k, j]);
```

This line already contains most of the backward pass. The coordinate `i`
belongs to rows of `A` and rows of `C`. The coordinate `j` belongs to columns of `B` and
columns of `C`. The coordinate `k` is the shared inner coordinate.

Before differentiating anything, read the dependency of one output cell:

```text
C[2, 5] = sum[k](A[2, k] * B[k, 5])
```

Only row `2` of `A` and column `5` of `B` affect this output. That observation
is small, but it contains the pullback. If a later loss changes because
`C[2, 5]` changed, that sensitivity can flow backward only to the `A[2, k]`
values and the `B[k, 5]` values that participated in the sum.

This is the gentler way to meet the pullback: not by materializing a giant
Jacobian, but by asking where one output cell could have come from. The zeros
in the dense object are already visible as omitted coordinate routes.

The pullback is easiest to read by holding one input cell fixed and asking
which output cells could have noticed it. That address question removes most
of the imaginary giant Jacobian before it is ever built.

For matrix multiplication, that reading becomes a five-step procedure:

```text
1. Choose one input cell, such as A[i, k].
2. List every output cell that directly reads it: C[i, j] for all j.
3. Attach the incoming sensitivity at each such output: G[i, j].
4. Multiply by the local derivative along that route: B[k, j].
5. Sum the routes that meet back at the input address: sum[j](...).
```

The result is `dA[i, k] = sum[j](G[i, j] * B[k, j])`. The same five steps
derive `dB` by holding a cell of `B` fixed.

The symmetry is worth keeping in sight:

```text
forward consumes k        backward reopens routes through output coordinates
forward preserves i, j    backward preserves the input cell's coordinates
forward broadcasts bias   backward collects the omitted coordinate
```

The pullback is not a new kind of mystery. It is the same coordinate accounting
run in the opposite direction.

## Jacobian Object or Indexed Rule

One possible design for derivatives is to treat every derivative as a large
Jacobian. That is mathematically uniform, and usually the wrong object to
compute. The useful computation is almost never the materialized Jacobian; it
is the way sensitivities are pulled back through coordinates.

Another design is to keep only high-level derivative rules such as "the
gradient of matrix multiplication uses transposes." That is efficient, but it
can make the transpose feel like a spell rather than a consequence.

The indexed rule sits between those extremes. It does not build the giant
Jacobian, and it does not ask the reader to accept the transpose rule on
faith. It shows which coordinates stay and which are collected.

The same is true if matrix multiplication is later exposed as a coordinate
function:

```text
let C[i, j] = matmul[k](A[i, k], B[k, j])
```

The call can be short because `k` names the consumed coordinate. The function
may lower to BLAS, but the pullback still knows which routes to reopen because
the contraction coordinate crossed the boundary.

It is still useful to imagine the full Jacobian once. Each cell `C[i, j]`
would have a derivative with respect to each cell `A[p, q]`. Most of those
entries are zero, because `C[i, j]` only reads row `i` of `A`. The indexed
pullback is the sparse story without ever materializing the sparse object.

Suppose a scalar loss depends on `C`, and let `G[i, j]` be the sensitivity
`@loss / @C[i, j]`. What shape should `@loss / @A` have? It must have the shape
of `A`, so its coordinates are `[i, k]`.

The only omitted coordinate from the output sensitivity is `j`, so `j` must be
collected:

```rust
let dA[i, k] = sum[j](G[i, j] * B[k, j]);
```

Likewise, `@loss / @B` has coordinates `[k, j]`, so `i` must be collected:

```rust
let dB[k, j] = sum[i](A[i, k] * G[i, j]);
```

## The Pattern

This is not a coincidence. The derivative with respect to a value keeps the
coordinates of that value. Any output coordinate that helped carry sensitivity
but is not part of the requested value becomes local to a reduction.

That sentence is worth reading slowly:

```text
the requested value decides what survives
the chain rule decides what gets summed
```

In implementation terms, a compiler may lower these equations to transposed
matrix multiplies. In source terms, the pullback is an index transformation.

## One Cell of the Pullback

Focus on `A[2, 3]`. Which output cells did it influence?

In the forward equation:

```text
C[i, j] = sum[k](A[i, k] * B[k, j])
```

the value `A[2, 3]` contributes when `i = 2` and `k = 3`. The coordinate `j`
is still free, so `A[2, 3]` contributes to every `C[2, j]`. Therefore its
gradient must collect over `j`:

```text
dA[2, 3] = sum[j](G[2, j] * B[3, j])
```

This is the whole pullback in one coordinate. The formula for `dA[i, k]` is
not an arbitrary transposed matmul trick. It is the result of asking which
output coordinates one input coordinate helped produce.

Now focus on `B[3, 5]`. It contributes when `k = 3` and `j = 5`; the free
coordinate is `i`. So:

```text
dB[3, 5] = sum[i](A[i, 3] * G[i, 5])
```

The two pullbacks are mirror images because the forward expression uses `A`
and `B` in different coordinate roles.

## Batched Pullbacks

For batched matrix multiplication:

```text
C[..batch, i, j] =
    sum[k](A[..batch, i, k] * B[..batch, k, j])
```

the batch prefix survives in all three gradients:

```text
dA[..batch, i, k]
dB[..batch, k, j]
dC[..batch, i, j]
```

The batch coordinates do not participate in the contraction. They identify
which member of the family is being multiplied. This is why naming the prefix
is more than a convenience: it prevents the batch structure from being confused
with the algebraic structure.

For a single batch member:

```text
dA[batch..., i, k] =
    sum[j](G[batch..., i, j] * B[batch..., k, j])
```

The batch prefix stays fixed while `j` is collected. A bug that accidentally
reduces over a batch coordinate would mean that examples in the batch are
sharing gradient contributions. That may be intended in some reductions, but
it is not the pullback of ordinary batched matrix multiplication. Visible
batch coordinates make that distinction reviewable.

The prefix is not special-cased by a matmul rule hidden in a library. It is
present in the indexed equation, so the same rule that preserves `i` and `k`
also preserves every coordinate in `..batch`.

## What the Compiler Learns

A compiler does not need to materialize a giant Jacobian to use this structure.
It can recognize the index pattern and lower the pullback to efficient matrix
multiplications. The important point is earlier than that
optimization: the source already states enough coordinate structure to derive
the shape of the pullback.

There are two separable compiler jobs here. The first is semantic: derive an
indexed pullback that preserves the right coordinates and reduces the rest.
The second is operational: recognize that the resulting reductions can be
implemented as ordinary matrix multiplications, perhaps with transposed views
or fused kernels. If those jobs are blurred, the transpose formulas look like
primitive magic. If they are separated, the transposes are just one lowering of
the coordinate rule.

This is the difference between asking "what operations happened?" and asking
"which coordinates connect the value I care about to the value I am
differentiating with respect to?" The latter question is visible in the
formula.

## What Must the Gradient Be Addressed Like?

Before any implementation strategy appears, the request `@loss / @A` has a
coordinate obligation: the answer must be addressed like `A`. That observation
changes the feel of the pullback. The gradient is not an opaque artifact
returned by an engine; it is a value whose coordinates were named before the
engine began.

## Pullback as Coordinate Accounting

Matrix multiplication is the right example here because it sits at the crossing
of three earlier ideas. It has surviving coordinates, a consumed coordinate,
and a derivative that must decide which coordinates to preserve. Nothing about
the pullback is arbitrary once those roles are visible.

The forward pass says:

```text
i survives for A and C
j survives for B and C
k is shared and consumed
```

The backward pass then asks a family of coordinate questions. For `A`, preserve
`i` and `k`, collect `j`. For `B`, preserve `k` and `j`, collect `i`. For the
output sensitivity, preserve `i` and `j`.

This is the first place where the earlier pieces start to feel like one
system. The same reading habits used for reshape, broadcasting, and reduction
now explain a central autodiff rule. The reward is fewer special cases to
carry separately.

One symmetry will return in the next chapter:

```text
forward broadcast  -> backward reduce
forward reduce     -> backward broadcast-like fan-out
```

The arrows are not slogans. They are coordinate accounting. A value reused
across a coordinate receives many routes of sensitivity back. A coordinate
summed away in the forward pass often reappears as a route along which an
upstream sensitivity is distributed.

A final check is to compare the pullback equations to the familiar transposes:

```text
dA = G * B^T
dB = A^T * G
```

Those formulas are compact and useful, but the indexed version explains why
the transposes appear. `dA` needs `[i, k]`, so `B` must be read as `[k, j]`
while `j` is summed. `dB` needs `[k, j]`, so `A` must be read as `[i, k]` while
`i` is summed. The transpose is not arbitrary; it is a consequence of which
coordinates must line up in the reduction.

That is the level of explanation visible-index notation offers: not replacing
familiar linear algebra, but making its coordinate reasons available in source.

## Pullback Without Materializing the Jacobian

Let `A` be `2 x 3`, `B` be `3 x 2`, and `C` be `2 x 2`. The small sizes are
not the lesson; they are a way to see why the real object is not the dense
Jacobian:

```rust
let C[i, j] = sum[k](A[i, k] * B[k, j]);
```

Expand one output row:

```text
C[0, 0] = A[0, 0] * B[0, 0] +
          A[0, 1] * B[1, 0] +
          A[0, 2] * B[2, 0]

C[0, 1] = A[0, 0] * B[0, 1] +
          A[0, 1] * B[1, 1] +
          A[0, 2] * B[2, 1]
```

Now ask where `A[0, 1]` appears. It appears in both `C[0, 0]` and `C[0, 1]`,
but not in row `1` of `C`. If an incoming cotangent is named `G[i, j]`, the
gradient cell for `A[0, 1]` is therefore:

```text
dA[0, 1] = G[0, 0] * B[1, 0] +
           G[0, 1] * B[1, 1]
```

Restoring names:

```rust
let dA[i, k] = sum[j](G[i, j] * B[k, j]);
```

The reduction coordinate is `j` because `A[i, k]` contributes to all output
columns `j` for the same row `i`.

Now do the same for `B[1, 0]`. It appears in:

```text
C[0, 0] through A[0, 1]
C[1, 0] through A[1, 1]
```

So:

```text
dB[1, 0] = G[0, 0] * A[0, 1] +
           G[1, 0] * A[1, 1]
```

and in names:

```rust
let dB[k, j] = sum[i](G[i, j] * A[i, k]);
```

This table is the long form of the familiar transposes. For `dA`, the bridge
uses `B[k, j]`; for `dB`, the bridge uses `A[i, k]`. The transpose notation
compresses that alignment, but the indexed form explains it.

The compiler-facing value of this reading is that no dense four-dimensional
Jacobian has to be the primary object. A literal Jacobian from `C[i, j]` to
`A[p, q]` would have coordinates `[i, j, p, q]`, mostly filled with zero
except when `p = i` and `q` matches the contraction coordinate. The pullback
rule avoids materializing that object by immediately composing the incoming
cotangent with the local dependency. The source indices show the same
sparsity: most coordinates simply do not line up.

The mental test is compact and strict. For any proposed matmul gradient,
choose one input cell. List the output cells that read it. If the proposed
formula sums exactly those cells and preserves exactly the input cell's
coordinates, the pullback shape is right. If it preserves an output coordinate
that the input does not have, or forgets one that the input does have, the
formula is wrong even if the resulting array happens to have a plausible
shape.

## Why the Giant Object Disappears

The full derivative of `C` with respect to `A` can be imagined as:

```text
dC_dA[i, j, p, q]
```

Most entries are zero. `C[i, j]` depends on `A[p, q]` only when `p = i`; when
that holds, the local factor is `B[q, j]`. The pullback form composes this
sparse relation with `G[i, j]` immediately:

```text
dA[p, q] = sum[i, j](G[i, j] * dC_dA[i, j, p, q])
```

After applying the equality `p = i`, the unnecessary coordinate disappears:

```text
dA[p, q] = sum[j](G[p, j] * B[q, j])
```

This is exactly the earlier rule with names changed back to `[i, k]`. The
indexed notation lets the reader see why the dense Jacobian would be wasteful
and why the pullback has the smaller shape. It is not a shortcut that loses
meaning. It is coordinate accounting that removes zeros before they become an
implementation burden.

The same accounting explains batching. If a batch prefix is present:

```rust
let C[b, i, j] = sum[k](A[b, i, k] * B[b, k, j]);
```

then the pullbacks keep `b` rather than summing it:

```text
dA[b, i, k] = sum[j](G[b, i, j] * B[b, k, j])
dB[b, k, j] = sum[i](G[b, i, j] * A[b, i, k])
```

Batch identifies independent matrix products. It is not a contraction
coordinate. If a gradient formula sums over `b` here, it has changed the
meaning from per-example sensitivity to a shared-parameter accumulation.

That accumulation may be correct for shared parameters in a training loss, but
it should be an explicit consequence of the surrounding loss reduction, not an
accidental property of the matmul pullback itself.

Hold one cell `A[p, q]` fixed and ask which cells of `C` can depend on it. One
coordinate is forced equal; another remains to be summed in the pullback. That
small address question is the whole trick.

## Try It

Given `C[i, j] = sum[k](A[i, k] * B[k, j])`, run the five-step procedure twice:
first with one fixed cell `A[i, k]`, then with one fixed cell `B[k, j]`. Write
the two gradient expressions and mark which coordinate was consumed in each
pullback.

**Line to keep:** hold one input cell fixed and ask which output cells can feel
it.
