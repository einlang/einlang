---
layout: book
title: "The Index That Leaves"
---

# The Index That Leaves

> "Nothing is lost, nothing is created, everything is transformed."
>
> — Antoine Lavoisier, *Traité Élémentaire de Chimie* (1789)

Chapter 4 showed what happens when a term omits a coordinate. The value is
reused along that dimension. Broadcasting.

This chapter shows what happens when a coordinate is deliberately *consumed*.
The opposite move. A value that walks across the coordinate, collecting it cell
by cell, until nothing remains.

You have written this move a thousand times. You may have never named it.

## The Bug That Log-Sum-Exp Hid

It is 2 AM. You are staring at a training curve that is not converging the way
it should. The loss descends. The shapes all check out. But the validation
metric — calibration error — has been drifting upward for six weeks, and no one
can explain why.

The program computes a log-sum-exp over the `class` axis:

```python
# Original: logits shape [batch=100, class=10]
lse = torch.logsumexp(logits, dim=-1)   # [batch]
loss = lse.mean()
```

Six weeks ago, a colleague refactored the data pipeline. The new loader
produces `[class, batch]` instead of `[batch, class]`. The shapes are now
`[10, 100]`. The line `logsumexp(logits, dim=-1)` still runs. But now `dim=-1`
points to `batch`, not `class`.

The output shape is `[10]`. The `.mean()` produces a scalar either way. The
loss goes down. The shapes are valid at every step. And yet: the model is now
computing the log-sum-exp over batch items for each class, not over classes
for each batch item. Every class receives a summary of all examples.

You will not find this bug by checking shapes. `dim=-1` is always legal. You
will find it when you trace a single miscalibrated prediction back through six
layers of a deployed model and ask: which coordinate did `dim=-1` consume?

Named reductions make the answer visible without the trace:

```rust
// Correct: consume class, keep batch
let lse[b] = sum[c](logits[b, c]);

// Bug: consume batch, keep class  (still runs when extents square up)
let lse[c] = sum[b](logits[c, b]);
```

`c` inside `sum[...]` means "I am consuming class." `b` on the left means
"batch survives." If the data pipeline swaps axes, `logits[c, b]` still
addresses the right cells — but the reader can SEE that `sum[c]` is consuming
the row coordinate. The question has a place to live.

## The Sum That Eats a Dimension

The mathematical operation is the log-sum-exp:

$$\text{lse}_b = \log \sum_c \exp(x_{b,c})$$

The sum runs over $c$. The result has no $c$ — the coordinate was consumed.
In a positional API, this is written by counting:

```python
lse = torch.logsumexp(logits, dim=-1)   # result: [batch]
```

The result is one-dimensional. The `class` axis disappeared. Where did it go?

It was consumed. The sum walked across every class, accumulated the exponentials,
and left nothing behind. The result has no memory of individual classes.

Now write it in Einlang:

```rust
let lse[b] = sum[c](logits[b, c]);
```

Read this line three times, because it says more than a `dim=-1` ever could.

First read: the result has coordinate `b`. Each batch member gets one number.

Second read: `c` is introduced by `sum[c]`. It is local to the reduction. It
does not survive into `lse`.

Third read: the input is addressed by `logits[b, c]`. Both coordinates are
used inside the sum. Only `b` escapes.

This is the entire idea. A reduction introduces a local coordinate, uses it to
align terms, and then consumes it. The result shape is whatever survived—the
coordinates you did not put inside `sum[...]`.

We are doing more than describing what a reduction computes. We are stating
which coordinate the reduction consumed -- and that fact has no default.
In a positional API, `dim=-1` consumes whatever axis happens to be last.
In a coordinate API, `sum[class]` consumes `class` regardless of where
`class` sits in the rank. The first is a command that changes meaning when
the layout changes. The second is a claim that survives a transpose.

Notice that the range of `c` is never stated. The compiler infers it from the
shape of `logits` — `c` must cover whatever extent `logits` carries at that
position. You can write `sum[c in 0..num_classes]` when explicitness matters,
or `sum[c]` when the shape already answers. The compiler treats both the same.
Explicit and implicit ranges unify into one representation. The notation does
not care which you wrote.

## The Bug That Hides in Square Matrices

Here is a bug that a `dim=-1` API will not catch.

```python
# PyTorch: compute row sums
row_sums = A.sum(dim=1)        # [rows]
col_sums = A.sum(dim=0)        # [cols]

# Refactored code: axes were transposed somewhere upstream
A = A.T
row_sums = A.sum(dim=1)        # still runs, but now this is a column sum
```

If `A` is `[rows=3, cols=5]`, the shapes are different enough that you might
notice. But what if `A` is square?

```python
A = torch.randn(128, 128)

# Intended: sum over the second axis (which was originally "class")
# After a silent transpose, the second axis has changed roles
loss = some_loss(A.sum(dim=1))   # runs fine, wrong meaning
```

No crash. No shape error. The model trains. The metric drifts. You will not
find this bug until you trace a single incorrect prediction back through six
layers of a deployed model.

With named reductions, the bug is visible at the write site:

```rust
// Intended: sum over class
let per_row[batch] = sum[class](A[batch, class]);

// After a silent transpose in the data pipeline:
// A now has shape [class, batch]
let per_row[batch] = sum[class](A[class, batch]);  
//  ↑ 'class' is now row coordinate of A, not column
// The index positions of A have changed, and the reader can see it
```

The coordinates `batch` and `class` appear in both the result declaration and
the input addresses. If the input layout changes, the mismatch is visible at
the coordinate level, not buried under a `dim=1` that happens to stay
syntactically valid.

## Survivors and Locals: A Two-Column Ledger

Every reduction creates two sets of coordinates. Here is the ledger for a
matrix multiply:

```rust
let C[i, j] = sum[k](A[i, k] * B[k, j]);
```

| Kind | Coordinates | Where they appear |
|------|------------|-------------------|
| Survivor | `i`, `j` | On the left (`C[i, j]`), in input terms (`A[i,*]`, `B[*,j]`), and on the right-hand side of any later assignment |
| Local | `k` | Introduced by `sum[k]`, used inside the reduction body (`A[*,k]`, `B[k,*]`), absent from `C` |

For a concrete cell:

```text
C[2, 5] = A[2, 0]*B[0, 5] + A[2, 1]*B[1, 5] + ... + A[2, n-1]*B[n-1, 5]
```

The fixed `2` and `5` are survivors. The walking `k` is local. The sum fills
in every `k`-step and then discards the walker. The result has no `k`-shaped
slot because no single `k`-value owns the cell.

Now re-read the same layout for a row sum and a column sum:

```rust
let r[i] = sum[j](A[i, j]);   // row sum: i survives, j consumed
let c[j] = sum[i](A[i, j]);   // col sum: j survives, i consumed
```

The survivor is always the coordinate on the left. The local is always the
coordinate inside `sum[...]`. No positional reasoning required.

Here is the concrete expansion for a `2 × 3` matrix:

```text
A = [[1, 2, 3],
     [4, 5, 6]]

r[i] = sum[j](A[i, j])
→ r[0] = 1 + 2 + 3 = 6
→ r[1] = 4 + 5 + 6 = 15

c[j] = sum[i](A[i, j])
→ c[0] = 1 + 4 = 5
→ c[1] = 2 + 5 = 7
→ c[2] = 3 + 6 = 9
```

The same matrix. One local coordinate swapped. Two different answers. A
positional API can express this as `axis=0` versus `axis=1`, but only if the
reader tracks which axis is which. The named form makes the swap a visible
edit: change `sum[j]` to `sum[i]`, and now the survivor is `j` instead of `i`.

```
   Reduction: One Coordinate Leaves

   Row sums r[i] = sum[j](A[i,j]) — walk across j, keep i:

        A[i,j]
   +----+----+----+----+
   |    | j=0| j=1| j=2|
   +----+----+----+----+
   |i=0 | 1  | 2  | 3  | → r[0] = 1 + 2 + 3 = 6
   +----+----+----+----+
   |i=1 | 4  | 5  | 6  | → r[1] = 4 + 5 + 6 = 15
   +----+----+----+----+

   Column sums c[j] = sum[i](A[i,j]) — walk down i, keep j:

        A[i,j]
   +----+----+----+----+
   |    | j=0| j=1| j=2|
   +----+----+----+----+
   |i=0 | 1  | 2  | 3  |
   +----+----+----+----+
   |i=1 | 4  | 5  | 6  |
   +----+----+----+----+
        |    |    |
        v    v    v
      c[0]  c[1]  c[2]
      =5    =7    =9

   One bracket says which coordinate leaves.
```

The question is not whether the sum is correct. The question is whether
the reader, three months from now, can tell which coordinate was consumed
without reconstructing the author's intention from a variable name. A
positional API says `axis=0` and hopes the reader remembers what axis 0
was. A coordinate API says `sum[i]` and the answer is the bracket itself.
The difference is not convenience. It is whether the consumed coordinate
survives the reader's limited memory.

## The Lifecycle of a Local Coordinate

A local coordinate has a clearly bounded lifecycle. Write it once, and every
subsequent operation can assume it:

```text
Phase 1: Introduction
  k is introduced by sum[k] — its scope begins here

Phase 2: Use
  k aligns terms inside the reduction body
  (A[i, k] and B[k, j] are both addressed with k)

Phase 3: Consumption
  k is consumed when the sum completes — it leaves scope

Phase 4: Absence
  k does not appear in the result type
  let C[i, j] = ...  — no k in the output address

Phase 5: Possible return
  k may reappear in a gradient expression
  let dA[i, k] = sum[j](dC[i, j] * B[k, j])  — k is back, but now
  it's a survivor in a different reduction context
```

Phase 5 is the one that surprises people. A coordinate consumed in the forward
pass can reappear in the backward pass. This is not a violation of scoping. The
backward pass is a different expression with a different set of survivors. The
forward local coordinate `k` was consumed for the forward answer. When you ask
"which input cell receives sensitivity?", the question reintroduces `k` as a
survivor in the gradient.

Here is a concrete trace to make Phase 5 tangible. Start with named values:

```text
A = [[2, 3, 1],    B = [[1, 0],
     [0, 1, 4]]         [2, 1],
                        [0, 3]]

C = A @ B = [[2·1+3·2+1·0,  2·0+3·1+1·3],
             [0·1+1·2+4·0,  0·0+1·1+4·3]]

  = [[8,  6],
     [2, 13]]
```

Now trace the gradient of `A`. In the forward pass, `A[0, 1] = 3` contributed
to `C[0, 0]` (multiplied by `B[1, 0] = 2`) and to `C[0, 1]` (multiplied by
`B[1, 1] = 1`). Two routes. In the backward pass, if the incoming sensitivity
at `C` is `G`, then:

```text
dA[0, 1] = G[0, 0] * B[1, 0] + G[0, 1] * B[1, 1]
         = G[0, 0] * 2       + G[0, 1] * 1
```

This is `sum[j](G[0, j] * B[1, j])`. Exactly Phase 5: `k` (which was consumed
as `sum[k]` in `C = A @ B`) has returned — but now as the second coordinate of
`dA`, a survivor in a DIFFERENT reduction context. The reduction coordinate in
the gradient is `j`, not `k`. `k` is the address being answered. `j` is the
path being summed.

The lifecycle is not a loophole. It is the chain rule with coordinates named.
A coordinate consumed in one expression returns in another because a different
question is being asked. The first question was "what is the aggregate?" The
second question is "which input cell receives sensitivity?" The coordinates
that answer each question differ, and the names track which is which.

## When Reduction Goes Wrong

Three real categories of reduction bugs. For each: the wrong code, the right
code, and the diagnostic question that catches it.

**1. Wrong coordinate consumed.** The most common pattern. `dim=-1` points to
whatever axis happens to be last.

```text
Wrong:  logsumexp(logits, dim=-1)   -- dim=-1 silently changes meaning after transpose
Right:  sum[class](logits[b, class]) -- class in the bracket follows the role

Diagnostic: "After a transpose two lines up, what does dim=-1 point to now?"
```

**2. Square matrix silent swap.** When all extents are equal, row sums and
column sums produce the same output shape.

```text
Wrong:  A.sum(dim=0) on a transposed A   -- shape [128], meaning: column sum named as row sum
Right:  sum[i](A[i, j])                  -- i is consumed, j survives -- identities visible

Diagnostic: "Is the consumed coordinate the row or the column? If you can't
answer from the code alone, the notation has hidden something."
```

**3. Batch leakage through scope error.** A reduction that accidentally
includes `batch` in its scope.

```text
Wrong:  sum[b, k](A[b, i, k] * B[b, k, j])   -- batch consumed! all examples averaged
Right:  sum[k](A[b, i, k] * B[b, k, j])       -- only k consumed

Diagnostic: "Circle the coordinates inside sum[...]. Is 'batch' in there? If
yes, you're averaging across examples."
```

The loss curve goes down in all three cases. The shapes are valid in all three
cases. The diagnostic question — "which coordinate was consumed?" — is the only
thing that separates the right computation from the wrong one.

This is the Hiding Law applied to reduction. The consumed coordinate is a
fact that later reasoning must recover -- gradient computation needs to
know which coordinate was summed, the reviewer needs to know what the
result means, and the optimizer needs to know which axis to parallelize
over. If the source buries that fact in a `dim=-1` whose meaning drifts
with the layout, every downstream reader pays the cost of reconstructing it.
If the source writes `sum[class]`, the fact is visible to every tool that
reads the source, including the compiler that will later emit the backward
pass.

## Not All Reducers Return a Scalar

`sum`, `prod`, `max`, and `min` all return a value per survivor cell. But not
every reducer does. Some return a coordinate:

```rust
let winner[batch] = argmax[class](logits[batch, class]);
```

`argmax[class]` consumes `class` and returns, for each batch member, the
integer index of the winning class. The result is not the maximum value. It is
an address within the consumed domain.

The survivor is still `batch`. The local is still `class`. The return type
differs (`i32` instead of the input element type), but the scoping rule is
identical.

This is a design boundary worth marking. A reduction consumes a coordinate. If
the program later needs the consumed coordinate's identity, it must ask for it
explicitly—with `argmax`, `argmin`, or a user-defined selection function. The
language should not smuggle the consumed coordinate back as hidden metadata.
The source that wants the winning index should write `argmax`. The source that
wants the winning value should write `max`. The distinction is visible.

You can wrap this pattern:

```rust
fn top1[class](x: [f32; ..left, class, ..right]) -> [i32; ..left, ..right] {
    argmax[class](x[..left, class, ..right])
}
```

The caller writes `top1[class](logits)`. The function signature says `class` is
consumed and all surrounding coordinates survive. The body is one line. The
contract is in the brackets, not in a comment.

## The Shape of the Work, Not Just the Shape of the Result

A reduction tells you two shapes: the output shape (survivors) and the work
shape (survivors × locals).

```rust
let C[i, j] = sum[k](A[i, k] * B[k, j]);
```

The output `C` has shape `[i, j]`. There is one reduction for every pair of `i`
and `j`. Each reduction walks across `k`. If `i` has 64 values, `j` has 32, and
`k` has 128, then:

```text
Output cells:   64 × 32 = 2,048
Work per cell:  128 multiplications + 127 additions
Total work:     2,048 × 128 = 262,144 multiplications
```

The numbers matter. But the structure matters more. The survivors define a
family of independent reduction problems. The local coordinate defines the work
done for each family member. This is why the formula can lower to a loop nest,
a tiled kernel, or a BLAS call without changing its meaning. The source fixes
the work structure. The backend chooses the schedule.

Compare two sources that differ only in which coordinate is local:

```rust
let r[i] = sum[j](A[i, j]);   // one reduction per row
let c[j] = sum[i](A[i, j]);   // one reduction per column
```

Same input matrix. Same reducer (`sum`). Different work shapes. The source
makes the choice explicit. A positional API hides the choice inside a number
and hopes the reader knows what the number means.

## Convolution: Multiple Locals, Same Rule

A convolution introduces more local coordinates, but the rule does not change:

```rust
let out[b, co, i, j] = sum[ci, kh, kw](
    x[b, ci, i + kh, j + kw] * kernel[co, ci, kh, kw]
);
```

Survivors: `b`, `co`, `i`, `j`. Locals: `ci`, `kh`, `kw`.

Three coordinates leave together. They describe the input channel and the two
spatial offsets inside the kernel window. The result has no memory of which
input channel or which kernel offset contributed most. It only has the
aggregate.

The rule for reading this expression: circle the coordinates on the left.
Those survive. Circle the coordinates inside `sum[...]`. Those are local.
Everything else—the offsets `i + kh`, the kernel address `co, ci, kh, kw`—is
a relationship between the two sets. The rule scales to any number of
coordinates without requiring new concepts.

## The Inversion: Broadcasting's Opposite

Chapter 4 introduced this pairing:

```rust
let y[b, f] = x[b, f] + bias[f];     // broadcast: bias omits b
let total[b] = sum[f](x[b, f]);       // reduce: sum consumes f
```

The two lines are inverses. Omission keeps a coordinate absent. Reduction
forces a coordinate to leave. Together, they explain most of tensor shape
behavior.

In the forward pass, a broadcast spreads a value along an omitted coordinate.
In the backward pass, the gradient of that broadcast is a sum over the same
coordinate:

```rust
// Forward: bias[f] is reused across every b
let y[b, f] = x[b, f] + bias[f];

// Backward: bias[f] collects sensitivity from every b
let dbias[f] = sum[b](dy[b, f]);
```

The omitted coordinate in the forward term becomes the consumed coordinate in
the backward gradient. This is not a special case for biases. It is the
coordinate-level statement of the chain rule: a value that is reused along `b`
receives sensitivity from every `b`.

Chapter 8 will prove this with the matrix multiply pullback. For now, the
heuristic is enough: if you broadcast along a coordinate, expect that
coordinate to become a reduction in the backward pass.

This symmetry is not a coincidence we exploit. It is a structural fact
the notation makes visible. When a positional program writes `bias +
x`, the compiler sees two tensors with compatible shapes. When a
coordinate program writes `bias[f] + x[b, f]`, the compiler sees that
`b` is absent from the first term -- and it knows, without being told,
that the backward pass will sum over `b`. The forward omission is the
backward reduction. One bracket carries both facts.

## Practical Reading Rule

```text
1. Read the result declaration:   let name[survivors] = ...
2. Read the reducer:               sum[local](...)
3. Read the body:                  which coordinates address each term
4. Confirm:                        local ∉ survivors
5. Confirm:                        every survivor appears on the left
```

For `let C[i, j] = sum[k](A[i, k] * B[k, j])`:

```text
survivors       = {i, j}
local           = {k}
k in survivors  = false    ✓
{i, j} on left  = true     ✓
```

That is the entire typecheck for a reduction. No axis numbers. No positional
alignment. The names do the work.

Now apply the same rule to a common mistake:

```rust
let r[i] = sum[j](A[i, j]);    // correct: row sum
let r[i] = sum[j](A[j, i]);    // plausible, but this is a column sum named r[i]
```

The second line is shape-legal if `A` is square. The source claims the survivor
is `i`, but `i` indexes the columns of `A`. A careful reader catches the
contradiction. A positional API describes both as valid reductions with
different `axis` arguments; the names give the contradiction a place to live.

Now scale the same rule to three coordinates. This is a batched weighted sum—a
reduction that appears inside every attention head:

```rust
let context[b, i, d] = sum[j](weights[b, i, j] * values[b, j, d]);
```

Apply the five-step reading:

```text
1. Result declaration:  context[b, i, d]     → survivors = {b, i, d}
2. Reducer:             sum[j]               → local = {j}
3. Body:                weights[b,i,j] reads {b,i,j}
                        values[b,j,d] reads  {b,j,d}
4. Confirm:             j ∉ {b,i,d}          ✓
5. Confirm:             {b,i,d} on left      ✓
```

The rule did not grow. Five steps, regardless of rank. The local `j` is consumed
whether the tensor has two coordinates or twelve.

Now the diagnostic. What if the data pipeline swapped the sequence axes?

```rust
// Intended: j is the key position (source of values)
let context[b, i, d] = sum[j](weights[b, i, j] * values[b, j, d]);

// Bug: j accidentally indexes the query position
let context[b, i, d] = sum[j](weights[b, i, j] * values[b, i, d]);
//                                                         ^^^ should be j
```

When `query_len == key_len`, `values[b, i, d]` and `values[b, j, d]` have
identical shapes. The sum still runs. `context` still has shape `[b, i, d]`. The
loss descends. But the gather reads from the query position instead of the key
position—the model is attending to itself, not to the source. The five-step
reading catches it at step 3: `values[b, i, d]` is missing the local coordinate
`j`. The local was introduced but never used by that term.

This is the whole discipline. Read the left for survivors. Read `sum[...]` for
locals. Confirm every local is absent from the left. Confirm every survivor
appears on the left. The number of coordinates does not change the rule.

## Try It

The best way to learn the consumed-coordinate instinct is to audit code you
already trust. Take three common reductions from your own codebase:

```python
# PyTorch
x.mean(dim=0)          # x has shape [batch, feature]
A.sum(dim=1)           # A has shape [height, width]
torch.max(logits, -1)  # logits has shape [batch, class]
```

For each, name the survivor and the consumed coordinate. Write the Einlang
equivalent. Then pick one concrete output cell—say `mean_result[3]`—and trace
which input cells were visited to compute it. The coordinate ledger tells you
the work shape (survivors × locals). `mean(dim=0)` on `[batch, feature]` has
one reduction per feature, each walking over batch. `sum(dim=1)` on `[height,
width]` has one reduction per row, each walking over width. These are different
work shapes that share the same positional signature `dim=...`. The ledger
distinguishes them.

Now go deeper. Take one reduction from your own code—a `.sum`, `.mean`, or
`.max` call. Ask what coordinate role is being consumed. Not "axis 0" or
"dim -1"—the role. Is it `class`, `channel`, `head`, `time`, `expert`? If a
data pipeline refactor transposed two axes upstream, would this line still run?
Would the shapes still match? Rewrite the reduction in Einlang, giving the
consumed coordinate a name that matches its role.

The hard case: a colleague writes `A.sum(dim=1)` on a square matrix where the
first axis is `source` and the second is `target`. They intended a row sum.
After a transpose, `A` becomes `[target, source]`, and `dim=1` now silently
computes a column sum:

```text
let outgoing[source] = sum[target](A[source, target]);    // before transpose
let outgoing[target] = sum[source](A[target, source]);    // after — still runs, wrong survivor
```

The names reveal that `outgoing` is now indexed by the wrong role. A shape
checker sees `[128] → [128]` and is satisfied. A coordinate checker sees the
survivor changed from `source` to `target`. This is not a shape error. It is a
role error.

Finally, consider a coordinate function `reduce_over[coord]` that accepts an
associative binary operation and a tensor, reducing away the named coordinate:

```text
fn reduce_over[coord](op: (a, a) -> a, init: a, x: [a; ..left, coord, ..right])
    -> [a; ..left, ..right]
```

Use it to implement `row_sum` and `col_sum` from the same input `A[i, j]`:

```text
let row_sum[i] = reduce_over[j](add, 0, A[i, j]);
let col_sum[j] = reduce_over[i](add, 0, A[i, j]);
```

When `A` is square, both calls produce a 1-D result of the same extent. A
shape-only API sees two calls with different `axis=` arguments but identical
output shape. It cannot tell you which is the row sum without a comment. The
rest packs `..left` and `..right` absorb all other coordinates. A single call
`reduce_over[class](logits[batch, class])` produces `[batch]`, while
`reduce_over[d](scores[b, h, q, k, d])` produces `[b, h, q, k]`—the function
adapts to any rank. The bracket says what leaves. Everything else stays.

**Line to keep:** the shape of the result is the list of coordinates that were
not eaten.

### Where This Leads

Broadcasting and reduction are inverses. One omits a coordinate; the other
consumes one. Positional notation treats them differently — broadcasting is
implicit, reduction demands an explicit `dim` argument. Named coordinates make
the symmetry obvious: an omitted coordinate and a consumed coordinate are the
same kind of fact, stated with the same brackets.

You now have both halves of coordinate disappearance. Together they explain how
tensors change shape.

But a reduction consumes only one set of coordinates. Softmax needs three.

The softmax function is not one reduction. It is three. First, a `max` scan over
`class` that finds the stability constant—consuming `class` into a scalar per
batch member. Second, an exp over `class` that produces normalized
values—keeping `class` alive. Third, a `sum` over `class` that computes the
denominator—consuming `class` again, but in a different scope than the `max`.
The output has `class` as a survivor, but only because the division cancels the
denominator's consumption.

Three different roles. All three range over the same domain—the same set of
class indices. A positional API flattens all three into `dim=-1` and hopes the
reader cannot tell the difference. But the reader can. The `q` used for the
stability scan is not the `k` that defines the numerator. The `k` that defines
the numerator is not the `j` that survives into the output. Three letters, one
axis, three different fates.

Chapter 6 names them. Bring the ledger from this chapter—the survivor/local
distinction—and watch what happens when a single operation needs the distinction
three times over.