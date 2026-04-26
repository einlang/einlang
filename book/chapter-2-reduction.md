---
layout: book
title: "Chapter 2: Reduction"
---

# Chapter 2: Reduction

The previous chapter restored indices to tensor programs. This chapter studies
what happens when an index is deliberately consumed. Reduction is where local
coordinate-wise definitions become dot products, matrix multiplication,
pooling, and the first real examples of compiler-visible algebraic structure.
This transformation from families to summaries is not merely a convenience—it's
a fundamental operation that enables composition at different scales. By making
reduction explicit, we create a language where aggregation patterns are as
visible and analyzable as the element-wise operations that precede them,
enabling optimizations that span from single operations to entire algorithms.

## 2.1 Sums, Dot Products, and Matrix Multiplication

Reduction is the first operation in the book that changes the dimension of a
value by consuming an index. Indexed binding creates axes; reduction removes
them in a controlled way. This controlled consumption is what enables the
compiler to track not just what computation happens, but how complexity is
reduced. The explicit reduction syntax ensures that aggregation patterns
remain visible in the source, enabling analyses that would be impossible with
implicit summation conventions.

Formulas:

```text
alpha = sum_i u_i v_i
C_{i,j} = sum_k A_{i,k} B_{k,j}
```

Einlang:

```rust
let alpha = sum[i](u[i] * v[i]);
let C[i, j] = sum[k](A[i, k] * B[k, j]);
```

The reduction index is a bound variable. It exists inside `sum[k](...)` and
then disappears. Output indices survive; reduction indices are consumed. This
scoping discipline ensures that reduction operations are local and
composable, preventing the kind of global state accumulation that complicates
analysis in other systems.

In matrix multiplication, `k` indexes axis 1 of `A` and axis 0 of `B`. The
compiler checks that those dimensions agree. The result shape comes from the
surviving axes: rows from `A`, columns from `B`.

Nested reductions can often be lowered as a single multi-axis reduction:

```rust
let total = sum[i](sum[j](A[i, j]));
```

The source says two reductions, but both `i` and `j` are consumed axes. The
compiler is free to choose a fused implementation when the backend supports it.

### Surviving and Consumed Indices

The main rule is:

```text
output indices survive
reduction indices are consumed
```

Compare these definitions:

```rust
let row_sum[i] = sum[j](A[i, j]);
let col_sum[j] = sum[i](A[i, j]);
let total = sum[i](row_sum[i]);
```

In the first line, `i` survives and `j` is consumed. In the second line, `j`
survives and `i` is consumed. In the third line, `i` is consumed and no output
index remains, so the result is scalar.

This is the same kind of discipline as lexical scope in ordinary programming.
A reduction introduces a local name, uses it inside a body, and prevents it from
escaping. The difference is that the local name ranges over an axis.

### Matrix Multiplication Without a Special Case

Matrix multiplication often appears as a primitive operation, but its structure
is fully visible in index notation. This visibility is not merely pedagogical—it's
what enables the compiler to optimize matrix operations without being told
they're "special." When the structure is explicit, fusion, tiling, and memory
layout optimizations become systematic rather than ad-hoc.

```rust
let C[i, j] = sum[k](A[i, k] * B[k, j]);
```

The row index `i` comes from `A`; the column index `j` comes from `B`; the inner
index `k` connects them and is consumed. The familiar shape rule emerges
naturally from the index relationships:

```text
(M, K) times (K, N) gives (M, N)
```

is not a memorized property of a special operator. It is what remains after
`k` has been consumed. This derivation from first principles enables the
compiler to check compatibility automatically and generate appropriate code
for different matrix sizes and layouts.

## 2.2 Beyond Sum: Max, Min, Product, and Pooling

> **Think More.** Once reduction is a pattern, `sum` is only one possible answer
> to the question "what should survive from this axis?" Max pooling, products,
> and masked reductions all make different promises. Which promises are algebraic
> enough for optimization, and which ones require the compiler to be more
> conservative? Where should user-defined reducers sit between flexibility and
> analyzability?

A reduction is a pattern, not a synonym for addition. The syntax names an index
domain and a way of combining the values encountered over that domain.

Formulas:

```text
m = max_i v_i
O_{n,c,h,w} = max_{kh,kw} I_{n,c,h*s+kh,w*s+kw}
```

Einlang:

```rust
let m = max[i](v[i]);

let O[n, c, h, w] = max[kh in 0..K, kw in 0..K](
    I[n, c, h * stride + kh, w * stride + kw]
);
```

Einlang reduction syntax currently includes `sum`, `max`, `min`, and `prod`.
Mean is usually expressed as a sum divided by a count, or by using a standard
library operation when reducing a whole tensor is the right abstraction.

Pooling shows why bounded index arithmetic matters. The compiler sees the
domains of `kh` and `kw`; it also sees how the input coordinates are derived
from output indices and kernel offsets. That gives range analysis enough
structure to infer or check the legal output space.

### Identity Values and Guards

Every reduction has an identity value:

```text
sum  -> 0
prod -> 1
max  -> negative infinity
min  -> positive infinity
```

This matters for guarded reductions:

```rust
let positive_sum = sum[i](x[i]) where x[i] > 0.0;
```

The guard filters the reduction domain. Points that do not pass the guard do
not contribute to the sum. Operationally, the identity value fills the skipped
positions. Conceptually, the formula has said "sum over the positive entries"
without introducing a separate loop or temporary array.

### Pooling as Local Choice

Max pooling is a good example because it is not linear. There is no sum of
products to hide behind. The source says: for each output point, inspect a local
window and keep the largest value.

```rust
let O[n, c, h, w] = max[kh in 0..K, kw in 0..K](
    I[n, c, h * stride + kh, w * stride + kw]
);
```

The output indices choose the window origin. The reduction indices choose an
offset inside the window. The index expression combines origin and offset
directly in the input access. This is a complete description of pooling's
geometry.

The compiler facts are also visible. `kh` and `kw` are bounded. The input
coordinates are affine expressions over known indices. The input access must
remain inside `I`. If a shape is known, some of these facts can be checked
statically; otherwise they become runtime obligations.

## 2.3 Why Reduction Is Explicit

> **Think More.** Implicit summation is compact because readers infer intent
> from convention. Einlang chooses explicit reduction because the disappearance
> of an axis is a semantic event. Is the extra syntax worth it when it marks the
> exact point where shape, storage, and derivative structure change?

Einstein notation often uses repeated subscripts to imply summation. Einlang
does not. It writes the reduction point:

```rust
let C[i, j] = sum[k](A[i, k] * B[k, j]);
```

This is a deliberate trade. Repetition alone is compact, but it hides a major
semantic event: an axis is consumed. In Einlang, `sum[k]` is visible to the
reader, the shape checker, the lowering pass, and autodiff.

Explicit reduction also gives optimization a handle. If two reductions can be
fused, the compiler sees the reduction operators and their bound indices
directly. It does not need to recover them from an encoded string.

### Why This Matters for Later Chapters

Reduction is the first source form that changes rank by binding an axis and
then removing it. Autodiff needs to understand that event: the derivative of a
sum distributes over the summed terms, and the derivative of a matrix
multiplication follows the contracted index. Recurrence analysis also benefits
from explicit axes, because reductions can live inside recurrent bodies without
confusing the recurrence index with a local reduction index.

### A Complete Worked Reading

This fragment computes a row-normalized distribution:

```rust
let row_max[i] = max[j](scores[i, j]);
let shifted[i, j] = scores[i, j] - row_max[i];
let denom[i] = sum[j](exp(shifted[i, j]));
let probs[i, j] = exp(shifted[i, j]) / denom[i];
```

The first line consumes `j` and leaves `i`, producing one maximum per row. The
second line uses `row_max[i]`; because it has no `j`, it is constant across each
row. The third line consumes `j` again and leaves one denominator per row. The
fourth line uses that denominator across `j`.

This reading combines reduction and broadcasting before broadcasting has even
needed its own mechanism. The missing index already tells the story.

## 2.4 Reduction Algorithms and Patterns

> **Think More.** A reduction is both a mathematical operation and a scheduling
> opportunity. Sequential accumulation, tree reduction, and SIMD lowering may
> compute the same result only when the operation's laws permit it. Which facts
> should surface in the language?

Beyond basic reductions, explicit reduction syntax enables direct expression
of many algorithmic patterns that are often hidden behind library calls.

### Statistical Aggregations

Computing multiple statistics in a single pass:

```rust
let data[i in 0..100] = normal_random();  // hypothetical random data

// Multiple reductions over the same data
let count = sum[i](1);
let total = sum[i](data[i]);
let mean = total / count;
let variance = sum[i]((data[i] - mean) * (data[i] - mean)) / count;
let std_dev = sqrt(variance);
```

This demonstrates how reductions compose: each statistical measure is a
reduction pattern that can be analyzed and optimized independently.

### Argmax and Argmin with Values

Finding both position and value of extrema:

```rust
let values[i in 0..5] = [3.2, 7.1, 2.8, 9.4, 1.5][i];

// Find maximum value and its index
let max_val = max[i](values[i]);
let argmax = sum[i](i * (values[i] == max_val ? 1 : 0));
```

The argmax uses a clever reduction pattern: multiply each index by whether it
holds the maximum, then sum. This works because only one position should equal
the maximum (assuming no ties).

### Weighted Reductions

Combining values with importance weights:

```rust
let values[i in 0..4] = [1.0, 2.0, 3.0, 4.0][i];
let weights[i in 0..4] = [0.1, 0.3, 0.4, 0.2][i];

// Weighted average
let weighted_sum = sum[i](values[i] * weights[i]);
let total_weight = sum[i](weights[i]);
let weighted_avg = weighted_sum / total_weight;
```

This pattern extends to more complex weighting schemes and can be optimized
using fused operations.

### Conditional Reductions with Masks

Reducing only over certain elements:

```rust
let matrix[i in 0..3, j in 0..3] = i * 3 + j;
let mask[i in 0..3, j in 0..3] = (i + j) % 2 == 0;

// Sum only even-positioned elements
let masked_sum = sum[i, j](matrix[i, j]) where mask[i, j];
let masked_count = sum[i, j](1) where mask[i, j];
let masked_avg = masked_sum / masked_count;
```

The `where` clause creates conditional reductions, enabling sparse operations
and selective aggregations.

### Implementation Models for Reductions

Different compilation strategies for reductions:

**Sequential Accumulation**: Simple loop that accumulates results one element
at a time. Easy to implement but limited parallelism.

**Tree Reduction**: Divide the data into chunks, reduce each chunk in
parallel, then combine results in a tree structure. Enables high parallelism
but requires more complex coordination.

**SIMD/Vector Reduction**: Use vector instructions to reduce multiple elements
simultaneously. Highly efficient for numerical data but limited to
associative operations.

**GPU Kernel Reduction**: Generate specialized GPU kernels that exploit
massive parallelism. Best for large datasets but adds compilation complexity.

**Symbolic Reduction**: Manipulate reduction expressions algebraically before
evaluation, enabling optimizations like constant folding or operation fusion.

Each model has different trade-offs in performance, memory usage, and
applicability to different reduction operations.

The next part uses the same index discipline to explain broadcasting and axis
transformations.

## 2.5 Reduction as Algebra, Not Just Syntax

Reduction is often introduced in programming as "the thing that combines many
values into one." That description is serviceable, but it is too thin to
support the design choices made in Einlang. In this language, reduction is not
simply a library helper that happens to iterate over a dimension. It is an
algebraic event in the source. An axis that used to exist in the expression is
consumed, and the way in which it is consumed matters to both meaning and
optimization.

To see why, compare two superficially similar fragments:

```rust
let total = sum[i](x[i]);
let biggest = max[i](x[i]);
```

Both definitions remove the index `i`. Both produce a scalar if `x` is a vector.
But they do not promise the same algebra. `sum` is associative and has identity
`0`. `max` is idempotent in a way `sum` is not, and its identity behaves like
negative infinity rather than zero. Those facts shape what the compiler may
legally reorder, fuse, parallelize, or guard.

This is why a serious reduction story must go beyond "iterate and combine." The
combining behavior is part of the source meaning. When the reduction operator is
explicit, the implementation gains a handle on the algebraic promises the user
has made. That handle is one of the main reasons the notation is valuable.

### Consumption Is a Semantic Change

The disappearance of an index is easy to underestimate. In ordinary API-based
array code, reduction may look like just another method call, but in structured
tensor notation it is a real shape transformation. Before reduction, the result
can vary along that axis. After reduction, it cannot.

Take this pair:

```rust
let score[i] = u[i] * v[i];
let total = sum[i](score[i]);
```

The first line preserves `i`; the second consumes it. The distinction is not
merely notational. Before the sum, each point has an address in the output
family. After the sum, the family has collapsed into one scalar. If we later ask
for a derivative, a schedule, or a printout, that change in structural status
matters. A consumed axis is not lying dormant in the result. It is gone.

That is why implicit summation conventions are such an uneasy fit for a language
concerned with compiler-visible structure. A repeated index in a formula may be
compact for a mathematician, but it hides the fact that a structural
transition has occurred. Einlang spends a few extra characters on `sum[k]`
because those characters mark the event precisely where it happens.

### Reduction and Meaningful Names

Named intermediate results become especially valuable around reductions.
Consider:

```rust
let energy[i] = x[i] * x[i];
let norm2 = sum[i](energy[i]);
```

The reduction result could have been written inline, but naming `energy` makes
the program easier to talk about. We can print it, inspect it, differentiate
through it, or attach a conceptual reading to it: "pointwise squared energy
before aggregation." In larger programs, naming the non-reduced or partially
reduced stages creates a ladder of meanings. The program ceases to be a single
opaque reduction and becomes a series of related structural claims.

This is one of the book's recurring themes: names and indices together create
places where later analysis can attach. Reduction does not interrupt that story.
It sharpens it. The programmer is now saying not only "this value exists" but
also "this axis disappears here, under this algebraic rule."

### Associativity, Parallelism, and Caution

A reduction operator is attractive to implementers because it often opens a door
to parallel execution. A long sum can be split into chunks, reduced in parallel,
and combined. A long maximum can be computed through a tournament tree. But the
legitimacy of such strategies depends on algebraic properties, not just on the
presence of brackets.

For exact arithmetic, associativity is clear. For floating-point arithmetic, it
is subtler. The source says `sum`, but floating-point addition is not truly
associative in machine behavior. That fact does not invalidate reductions; it
simply reminds us that source algebra and machine algebra are related but not
identical. A serious compiler or runtime may choose different accumulation
orders for speed, reproducibility, or numerical stability. By naming the
reduction explicitly, the language makes that trade visible rather than burying
it inside a hidden kernel call.

### Locality of Meaning

Another benefit of explicit reduction is local readability. A reader should not
have to scan an entire expression to discover whether the result is a scalar, a
row vector, or a matrix. With reduction syntax, the consumed indices are stated
where they are introduced:

```rust
let row_sum[i] = sum[j](A[i, j]);
let col_sum[j] = sum[i](A[i, j]);
```

The difference between row-wise and column-wise aggregation becomes immediately
visible. No external documentation is needed. The source itself says which axis
survives and which axis disappears.

## 2.6 Numerical Meaning and Stable Aggregation

A chapter on reduction would be incomplete if it talked only about shape and not
about numerical behavior. Aggregation is where many tiny local values accumulate
into one decision, one probability, one loss, or one statistic. That makes
reduction a structural operation and a numerical one at the same time.

The softmax example hints at this:

```rust
let row_max[i] = max[j](scores[i, j]);
let shifted[i, j] = scores[i, j] - row_max[i];
let denom[i] = sum[j](exp(shifted[i, j]));
let probs[i, j] = exp(shifted[i, j]) / denom[i];
```

The subtraction by `row_max[i]` is not cosmetic. It is a numerically motivated
rearrangement that keeps the exponentials in a safer range. Notice what makes
this example so readable: the stability trick remains structurally visible. We
do not lose the row-wise denominator inside a library call. We can see one
reduction producing a maximum, another producing a sum, and a broadcast-like
reuse of the rowwise results.

### Reduction Reveals Where Stability Lives

A surprising amount of numerical method design can be phrased as deciding where
reductions should happen and in what transformed space. Log-sum-exp is a classic
example. Mean and variance calculations may use centered forms to avoid
cancellation. Norms may square before summing and take roots afterward. These
are not merely arithmetic curiosities. They are choices about the shape of an
aggregation pipeline.

Because Einlang exposes the pipeline directly, it gives the author a place to
spell out the stable version rather than hiding it inside a special primitive.
That matters for pedagogy, for verification, and for experimentation. A reader
can inspect the source and see not only that a stable trick exists, but how it
interacts with axes and reductions.

### Reproducibility Versus Throughput

Once reductions become large, another practical question appears: should a
system prioritize consistent answers across runs and hardware, or maximum
throughput? Different evaluation orders in floating-point arithmetic can lead to
slightly different final sums. Some applications care deeply about that; others
care primarily about speed.

An explicit reduction syntax does not solve that policy question by itself, but
it helps isolate it. The programmer has clearly marked the operations whose
evaluation order matters. A future system can expose policies such as "reproducible
sum," "fast parallel sum," or "compensated sum" without changing the fact that
the source named a reduction over a specific index domain.

### Masked and Sparse Aggregation

Numerical work also often requires selective aggregation. One may need to sum
only valid tokens in a sequence, ignore padding, or accumulate only positive
entries. The `where` form makes that structure source-visible:

```rust
let valid_mass = sum[t](weights[t]) where mask[t];
```

This is more precise than first constructing a filtered array in a host
language. The domain restriction remains attached to the reduction itself.
Backend implementations may lower it as a masked loop, a predicated vector
instruction, or a sparse kernel, but the source meaning stays stable: the axis
is reduced only where the guard admits points.

## 2.7 Lowering, Fusion, and Why Explicit Reduction Pays Off

From an implementation point of view, reduction is where the promise of
structure-visible syntax starts paying rent. Elementwise operations are useful,
but many systems can already optimize them reasonably well. Reductions are more
interesting because they involve both data movement and algebraic combination.
When a language marks them explicitly, the compiler can reason in a much more
principled way about lowering choices.

Consider these two lines:

```rust
let prod[i, j, k] = A[i, k] * B[k, j];
let C[i, j] = sum[k](prod[i, j, k]);
```

A naive implementation might materialize `prod` in full and then reduce it. A
better implementation may fuse the multiplication into the reduction and never
store the full rank-3 intermediate. The key observation is that fusion is not a
black-box optimization guessed after the fact. It is justified by the explicit
structure of the source. The compiler can see that `k` is local to the
aggregation and that `prod` has no independent life outside the reduced use.

### Reduction as a Boundary for Storage

Reductions also define meaningful boundaries for storage planning. An
unreduced family may need to remain addressable at many points. A reduced
result no longer carries the consumed axis, so the runtime can often keep only a
partial accumulator while traversing that domain. That is true whether the final
output is scalar, vector, or matrix.

For

```rust
let row_sum[i] = sum[j](A[i, j]);
```

the compiler may traverse one row at a time, maintain a scalar accumulator for
that row, and write the result once. The source already suggests that strategy
because the surviving and consumed axes are visible. A library call like
`A.sum(axis=1)` may do something similar, but it does not expose the
same structural relation to the rest of the language.

### Reduction in the Larger Language

It is also worth noticing that reduction is the first place where several of the
book's major ideas begin to intersect. Names matter because intermediate values
may be worth preserving or differentiating. Axes matter because some survive and
some disappear. Guards matter because domains may be filtered. Later, autodiff
will care because reduced expressions distribute sensitivities in specific ways.
Recurrence will care because a reduction may live inside a temporal update body.

That is why this chapter sits at a hinge point. Before reduction, we mainly
learned how to describe tensor-shaped families. After reduction, we can begin to
describe whole algorithms that summarize, compare, normalize, and contract those
families. The language becomes capable not only of representing structure, but
of transforming structure in ways that remain legible.

## 2.8 Reading Larger Programs Through Reductions

Once a reader learns to notice reductions, larger tensor programs become easier
to decompose. Scan a definition and ask three questions:

- Which axes are introduced?
- Which axes are carried through to the result?
- Which axes disappear inside reductions?

In an attention block, reductions build scores, normalize over keys, and combine
values. In a convolution, reductions over channel and kernel-offset axes combine
local evidence. In a loss function, reductions over batch or token axes turn
many local errors into a smaller objective. The operators differ, but the
reading habit is the same: reductions mark the places where a program consumes
one coordinate domain to produce another.

## 2.9 Discussion: Reduction as a Modeling Choice

Reduction is not only a way to compute fewer numbers. It is a way to say which
distinctions remain visible in the result. In matrix multiplication, summing
over `k` says that the shared feature axis explains how each output cell is
formed, but does not itself survive as an output address. In pooling, reducing
over a window says that the window contributes to a selected representative. In
a loss, reducing over examples says that many local discrepancies are being
turned into one objective.

This is why reduction benefits from source-level notation. The reader can see
what is being collapsed, what remains visible afterward, and which algebraic
operation governs the collapse. The compiler can use the same facts when it
checks shapes, lowers loops, reasons about identities, or decides whether a
reduction can be fused or reordered.

## 2.10 A Practical Heuristic

If a dense tensor expression is hard to read, find the reductions first and ask
what each one is allowed to forget. That question often reveals the purpose of
the expression faster than reading operators from left to right. It also helps
separate local arithmetic from the structural step where many values become
fewer values.

## Summary

Reduction consumes index domains explicitly. That makes shape changes visible
and gives the compiler a clear object to analyze.

- A reduction index is lexically scoped, creating local transformations that
  compose cleanly;
- All reads using that index must agree on its range, enabling static
  verification of compatibility;
- The result shape is determined by indices not consumed by the reduction,
  making dimensional changes predictable;
- Guarded reductions use reduction identities for skipped points, enabling
  conditional aggregation without complexity;
- Explicit reductions are easier to check, differentiate, and optimize,
  turning what could be opaque operations into transparent transformations.

As we move forward, reduction becomes the bridge between elementwise tensor
definitions and higher-level algorithms that summarize, normalize, compare, and
contract data.
