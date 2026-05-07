---
layout: book
title: "Matrix Multiplication Teaches the Pullback"
---

# Matrix Multiplication Teaches the Pullback

> "The purpose of abstraction is not to be vague, but to create a new semantic
> level in which one can be absolutely precise."
>
> — Edsger W. Dijkstra, "The Humble Programmer" (1972)

Why does the gradient of a matrix multiplication transpose the other matrix?
Most practitioners memorize this fact. Few can derive it from the forward
expression in thirty seconds. The reason is not calculus. It is coordinate
accounting.

Chapter 7 showed a linear map with one input and one output path. Matrix
multiplication has two inputs and three coordinates. The inner coordinate `k` is
consumed in the forward pass. The outer coordinates `i` and `j` survive. When
sensitivity flows backward, it splits into two pullbacks, each reducing over a
different outer coordinate. The transpose that practitioners memorize is a
consequence of which coordinate name appears where.

## The Bug That Memorizes the Transpose

Your colleague is implementing a transformer from scratch. The attention layer
works. The FFN works. The loss decreases. But the model is not learning—every
few hundred steps, the loss spikes and resettles at a slightly worse value.

After three days of staring, you find it at 11 PM:

```python
# Forward: C = A @ B      (correct)
C = A @ B

# Backward for A (intended: dA = G @ B.T)
dA = G @ B    # BUG: missing transpose on B
```

`dA` has the right shape `[i, k]`. `G` is `[i, j]`, `B` is `[k, j]`. The
multiplication `G[:, :] @ B[:, :]` would normally fail, but `j == k` in this
layer (the FFN has a square weight matrix). So `G @ B` runs. The numbers are
nonzero. The gradients flow. The optimizer takes steps. Everything looks correct.

But `G @ B` computes `sum[j](G[i, j] * B[j, k])`—it reads `B` transposed. The
gradient for `A` now depends on the wrong slice of `B`. Meanwhile `dB = A.T @ G`
is correct—but `dA` is wrong, so the model's learning about `A` is corrupted. The
loss goes down because `dB` still pushes `B` in a direction that compensates.
The two weight matrices are jointly adapting to a bug.

A shape checker sees `[i, k]` and reports success. A coordinate reader sees:

```rust
// Forward
let C[i, j] = sum[k](A[i, k] * B[k, j]);

// Correct pullback for A
let dA[i, k] = sum[j](G[i, j] * B[k, j]);

// Bug: sums over the wrong coordinate
let dA[i, k] = sum[j](G[i, j] * B[j, k]);
//                               ^^^^ should be B[k, j]
```

`B[k, j]` uses the local coordinate `k` paired with the route coordinate `j`.
The bug, `B[j, k]`, swaps them—`j` now indexes the wrong dimension of `B`.
When `j == k`, the swap is invisible to shape checks. When `j != k`, the shapes
are compatible only accidentally. The coordinate names make the swap visible
because `B[k, j]` and `B[j, k]` are different addresses.

## The Forward Expression Already Knows

Write the forward expression:

```rust
let C[i, j] = sum[k](A[i, k] * B[k, j]);
```

Now read one output cell:

```text
C[2, 5] = sum[k](A[2, k] * B[k, 5])
```

Only row `2` of `A` and column `5` of `B` affect this output. If a later loss
changes because `C[2, 5]` changed, that sensitivity can flow backward only to
`A[2, k]` values and `B[k, 5]` values. Every other cell of `A` and `B` lies on
a route that did not participate.

This is the whole trick. Hold one input cell still. Ask which output cells read
it. Sum the incoming sensitivities along those routes. The rest of the chapter
applies this to every cell.

We are not tracing gradients for the exercise. We are asking whether the
coordinate names we chose in the forward pass survive the backward pass. The
forward expression declared that `A[i, k]` meets `B[k, j]` at `k`. The
pullback must recover that same `k` and distinguish it from `i` and `j`. If
the notation hides the name `k` behind a dimension integer, the pullback
cannot verify the alignment — it can only hope the integer is still in the
right position.

## The Five-Step Procedure

Lock one input cell and follow its forward influence:

```text
1. Choose one input cell (e.g., A[i, k])
2. List every output cell that directly reads it: C[i, j] for all j
3. Attach the incoming sensitivity at each such output: G[i, j]
4. Multiply by the local derivative along each route: B[k, j]
5. Sum the routes that meet back at the input address: sum[j](...)
```

Apply it to `A[i, k]`:

```rust
// A[i, k] contributes to every C[i, j] for all j
// The free coordinate j becomes the reduction coordinate
let dA[i, k] = sum[j](G[i, j] * B[k, j]);
```

Apply it to `B[k, j]`:

```rust
// B[k, j] contributes to every C[i, j] for all i
// The free coordinate i becomes the reduction coordinate
let dB[k, j] = sum[i](G[i, j] * A[i, k]);
```

The procedure produces two formulas. They are mirror images because `A` and `B`
sit in different coordinate roles in the forward expression. `A` owns `i` and
`k`. `B` owns `k` and `j`. The transpose that practitioners memorize is not a
rule of calculus — it is the coordinate names, read backward.

## Verify With Concrete Indices

Take a small case: `A` is `2 × 3`, `B` is `3 × 2`, `C` is `2 × 2`. Write out
row 0 of `C`:

```text
C[0, 0] = A[0, 0] * B[0, 0] + A[0, 1] * B[1, 0] + A[0, 2] * B[2, 0]
C[0, 1] = A[0, 0] * B[0, 1] + A[0, 1] * B[1, 1] + A[0, 2] * B[2, 1]
```

Where does `A[0, 1]` appear? In both `C[0, 0]` and `C[0, 1]`, but nowhere in
row `1` of `C`. The pullback collects the incoming sensitivities at those two
output cells:

```text
dA[0, 1] = G[0, 0] * B[1, 0] + G[0, 1] * B[1, 1]
```

This is exactly `sum[j](G[0, j] * B[1, j])`. The concrete indices verify what
the five-step procedure derives.

Now trace `B[1, 0]`:

```text
B[1, 0] appears in:
  C[0, 0] through A[0, 1]
  C[1, 0] through A[1, 1]

dB[1, 0] = G[0, 0] * A[0, 1] + G[1, 0] * A[1, 1]
```

This is `sum[i](G[i, 0] * A[i, 1])`. Same logic, different sum coordinate.

```
   Pullback Fan-Out: 2x3x2 matmul (i=2, k=3, j=2)

   A[0,1] fan-out:              B[1,0] fan-out:
   C[0,0] = ... + A[0,1]*B[1,0] C[0,0] = ... + A[0,1]*B[1,0]
   C[0,1] = ... + A[0,1]*B[1,1] C[1,0] = ... + A[1,1]*B[1,0]
       ^                           ^
       |                           |
   dA[0,1] = sum[j](G[0,j]*    dB[1,0] = sum[i](G[i,0]*
             B[1,j])                       A[i,1])

   +-----------+                   +-----------+
   | A pulls j |                   | B pulls i |
   +-----------+                   +-----------+
   i,k survive, j consumed     k,j survive, i consumed

   "Hold one input cell, ask which outputs feel it."
```

## The Transpose Is Not Magic

The compact linear algebra formulas are:

```text
dA = G @ B^T
dB = A^T @ G
```

Engineers memorize these. The indexed version explains why the transposes
appear. `dA` needs coordinates `[i, k]`. `G` has `[i, j]`. `B` has `[k, j]`.
To multiply them, `B` must be read as `[k, j]` → the second coordinate aligns
with `j` for the reduction. That alignment is what the transpose notation
records, but the transpose is a consequence, not an axiom.

A reader who remembers `dA = G @ B^T` as a rule can still write `dA = G @ B`
if `B` happens to be square. The dimensions match. The code runs. The answer is
wrong because the coordinates are wrong, but the shape check passes. The
indexed form prevents this: `sum[j](G[i, j] * B[k, j])` has `j` paired with `j`
by name, and no accidental transposition can fake that alignment.

The transpose is not an axiom of calculus. It is a consequence of coordinate
alignment — and consequences are only reliable when their premises are visible.
The memorized rule `dA = G @ B^T` works when the reader remembers which
coordinate `B^T` is transposing. The indexed rule `sum[j](G[i, j] * B[k, j])`
works when the reader can read. The notation determines whether the transpose
is a fact you recall or a fact you verify.

Julia's Zygote computes the same pullback. `gradient(() -> sum(A * B), A)` traces
`A * B`, records the tape, and plays it backward. The pullback it produces is
`G * transpose(B)` — mathematically identical to `sum[j](G[i,j] * B[k,j])`. But
the `j` that was summed, and the `k` that survived, are properties of the tape
execution, not facts in the source. If `B` is square, you cannot tell from the
Zygote output whether the pullback transposed the right matrix or the wrong one.
The numbers match either way. The coordinate names would have told you.

## Batched Pullbacks

Add a batch prefix:

```rust
let C[batch, i, j] = sum[k](A[batch, i, k] * B[batch, k, j]);
```

The batch coordinate is not part of the contraction. It survives in all three
gradients:

```rust
let dA[batch, i, k] = sum[j](G[batch, i, j] * B[batch, k, j]);
let dB[batch, k, j] = sum[i](G[batch, i, j] * A[batch, i, k]);
```

A gradient formula that accidentally sums over `batch` would share sensitivity
across examples. The loss curve would still go down—the numbers are nonzero,
the shapes are plausible—but each example would receive an average of the
batch's gradient rather than its own. The bug is invisible to dimensional
analysis. It is visible in the coordinate source because `batch` appears in
the reduction brackets only if the programmer put it there.

## The Giant Jacobian That Never Materializes

A literal Jacobian from `C` to `A` would have four coordinates: `[i, j, p, q]`
— output row, output column, input row, input column. Most entries are zero,
and the zeros follow a pattern you already know. `C[i, j]` depends on `A[p, q]`
only when the row indices match: `p = i`. When that holds, the local derivative
is `B[q, j]` — the matching column from the other matrix.

The pullback never builds the giant four-dimensional object. It reads the index
relation `p = i` from the coordinate names and compresses immediately:

```text
dA[p, q] = sum[i, j](G[i, j] * dC_dA[i, j, p, q])
         = sum[j](G[p, j] * B[q, j])    // after p = i
```

The unnecessary coordinate disappears in one step because the names already
know which entries are zero. The notation does not build the giant Jacobian and
then compress it. It skips to the compressed form.

## What the Compiler Sees

When a coordinate-aware function boundary exists:

```rust
fn matmul[i, j, k](a: [f32; ..batch, i, k],
                   b: [f32; ..batch, k, j])
    -> [f32; ..batch, i, j]
```

the compiler knows three things before differentiation begins:

```text
1. The consumed coordinate is k
2. The output coordinates are i, j
3. The prefix coordinates (..batch) are carried through unchanged
```

The pullback for `a` preserves `i` and `k`, collects `j`. The pullback for `b`
preserves `k` and `j`, collects `i`. The prefix survives in both. This is not
a special-case rule for matmul. It is the same coordinate accounting from
Chapter 7 applied to a function with a named contraction.

The compiler can lower these equations to transposed matrix multiplies and
fuse kernels for the surrounding operations. The lowering is an implementation
choice. The coordinate roles are already decided in the source.

## Try It

The pullback formula looks abstract until you trace one cell. Let's fix that.

Trace the pullback for a concrete case. Let `A` be `2 × 3`, `B` be `3 × 2`,
and `C = A @ B` be `2 × 2`. Write the forward expansion for every output cell.
Circle every occurrence of `A[0, 1]` — it appears in both `C[0, 0]` and
`C[0, 1]`. Sum the routes that include incoming sensitivity `G` to get
`dA[0, 1] = G[0, 0] * B[1, 0] + G[0, 1] * B[1, 1]`. This is
`sum[j](G[0, j] * B[1, j])`. Now circle `B[2, 0]` and trace its fan-out to
get `dB[2, 0] = sum[i](G[i, 0] * A[i, 2])`. The reduction is over `j` for
`dA`, over `i` for `dB`. The path-coordinate rule predicts this mechanically.

Now try a batched bilinear form with two contractions:

```text
let y[b, i] = sum[j, k](A[b, i, k] * B[b, k, j] * x[b, j]);
```

Three inputs, two consumed coordinates. For `dA`, A's coords are `{b, i, k}`,
y's are `{b, i}`, so the path is `{j}` — sum over `j`:
`dA[b, i, k] = sum[j](dy[b, i] * B[b, k, j] * x[b, j])`. For `dB`, B's coords
are `{b, k, j}`, path is `{i}`: `dB[b, k, j] = sum[i](dy[b, i] * A[b, i, k] *
x[b, j])`. For `dx`, x's coords are `{b, j}`, but x also contracts over `k`:
`dx[b, j] = sum[i, k](dy[b, i] * A[b, i, k] * B[b, k, j])`. When `i_count ==
k_count == j_count`, all gradient shapes match the forward shapes, but the
reduction coordinates differ. A positional system may transpose the wrong
matrix while still producing the right output shape.

The classic bug is `dA = G @ B` instead of `G @ B.T`. Both produce shape
`[i, k]` when matrices are square. Write the coordinate formulas:

```text
// Correct: dA[i, k] = sum[j](G[i, j] * B[k, j])
//   j aligns G[i, j] with B[k, j] by name

// Bug:     dA_wrong[i, k] = sum[j](G[i, j] * B[j, k])
//   j is misaligned — B[j, k] reads the wrong axis
```

For `dA[2, 3]`, the correct version reads `G[2, j] * B[3, j]` — row 2 of G,
row 3 of B, same column `j` in both. The bug reads `G[2, j] * B[j, 3]` — row
2 of G, column 3 of B, completely different cells. When `B` is symmetric, the
two versions produce identical numbers. The loss descends. The model trains for
a week before anyone notices. The coordinate formula `sum[j](G[i, j] * B[k, j])`
prevents this entirely because `j` aligns the two terms by name — the alignment
is a visible fact, not a convention to remember.

**Line to keep:** the sum coordinate in a pullback is never arbitrary. It is
the coordinate the operand does not own.

## Why the Sum Coordinate Flips

Why does `dA` sum over `j` while `dB` sums over `i`? The answer is not in the
gradient. It is in the forward expression.

```rust
let C[i, j] = sum[k](A[i, k] * B[k, j]);
```

Look at `A`'s coordinates: `{i, k}`. Look at `C`'s coordinates: `{i, j}`. The
output has `j`; `A` does not. So each `A[i, k]` fans out to every `C[i, j]` for
all `j`. To collect the sensitivity back to `A[i, k]`, you must sum over `j`.

Look at `B`'s coordinates: `{k, j}`. Look at `C`'s coordinates: `{i, j}`. The
output has `i`; `B` does not. So each `B[k, j]` fans out to every `C[i, j]` for
all `i`. To collect the sensitivity back to `B[k, j]`, you must sum over `i`.

The rule is mechanical:

```text
For each operand X:
  1. Read X's coordinates from the forward expression.
  2. Read C's coordinates (the output).
  3. The path coordinates = C's coordinates ∖ X's coordinates.
  4. The pullback sums over the path coordinates.
```

For `A`:
```text
A's coords: {i, k}
C's coords: {i, j}
Path: {j}  ←  the sum coordinate in dA
```

For `B`:
```text
B's coords: {k, j}
C's coords: {i, j}
Path: {i}  ←  the sum coordinate in dB
```

The sum coordinate is never arbitrary. It is always the coordinate the operand
does NOT own but the output DOES. This rule generalizes from matmul to any
reduction-based operation with named coordinates. You do not need to memorize
which pullback transposes which input. You read the coordinates and the path
falls out.

This is the Hiding Law in its most mechanical form. The forward pass knows
which coordinate was consumed and which survived. Later reasoning — the
pullback, the optimizer, the person debugging — must recover that same fact
to sum in the right place. When the notation records coordinates by name,
recovery is set subtraction. When the notation records only dimension
integers, recovery is archaeology. The mathematics is the same either way.
The effort of verification is not.

A positional API buries this rule under shapes: "if C is `[m, n]` and A is
`[m, k]`, the missing dimension is `n`, so sum over it." But the missing
dimension is not a number—it is a role. When the numbers coincide by accident
(square matrices), the number says nothing about the role. The coordinate set
difference `{i, j} ∖ {i, k} = {j}` always says what the path is, even when
`|j| == |i| == |k|`.

### Where This Leads

The transpose that practitioners memorize is coordinate alignment, not an axiom of
calculus. In the forward pass, `A[i, k]` and `B[k, j]` meet at `k`. In the
backward pass, sensitivity to `A` flows through `j`, sensitivity to `B` flows
through `i`. The path-coordinate rule — output coordinates minus operand
coordinates — gives you the sum coordinate mechanically.

A memorized rule works until the shapes get complicated — until there are five
coordinates, two contractions, and a batch prefix. A coordinate rule works as
long as the source names what it consumes. The rule does not get harder with more
coordinates. Set subtraction is set subtraction with three names or with ten. The
memorized rule rots when the architecture changes. The coordinate rule is the
same audit in a larger ledger.

But what happens when the forward expression has multiple terms — a bias,
a broadcast, a shared parameter that fans out across batch and time? The local
derivative is still a scalar. The gradient may carry four sum coordinates. How
does a one-line scalar rule produce a gradient with four reductions? Chapter 9
asks: how does the local become global?