---
layout: book
title: "Chapter 3: Broadcasting and Transformations"
---

# Chapter 3: Broadcasting and Transformations

Not every tensor operation creates or consumes an axis. Some operations align
axes, repeat values conceptually, reorder dimensions, or derive one coordinate
from another. This chapter treats those transformations as index relationships
rather than as hidden array-manipulation rituals. By making these operations
explicit in the index structure, we enable the compiler to optimize memory
access patterns, avoid unnecessary copies, and generate efficient code for
different architectures. The result is a programming model where
transformations are not afterthoughts, but first-class operations that can be
composed, analyzed, and optimized alongside the computations they support.

## 3.1 Broadcasting by Missing Indices

Broadcasting reveals the power of index relationships in enabling composition
across different tensor shapes. What appears as a simple convenience becomes,
in index notation, a fundamental operation that expresses non-dependence
between dimensions.

Formula:

```text
R_{i,j} = v_i + M_{i,j}
```

Einlang:

```rust
let R[i, j] = v[i] + M[i, j];
```

Inside an indexed declaration, a tensor term that lacks an output index is
invariant along that missing axis. Here `v[i]` varies with `i` but not with `j`,
so it is reused across columns. `M[i, j]` varies with both axes. This explicit
indexing makes broadcasting a consequence of the language's structure rather
than a special-case operation, enabling the compiler to reason about when and
how values should be replicated.

This is the clearest form of broadcasting: the index names determine what is
replicated. If `v` has length `n` and `M` has shape `(n, m)`, then `R` has shape
`(n, m)`. If the shared `i` dimension disagrees, the compiler reports a shape
mismatch. The beauty of this approach is that broadcasting becomes a natural
consequence of the index relationships, not an ad-hoc mechanism bolted onto the
language.

Plain operators also support scalar broadcasting and same-rank tensor
elementwise operations. For different-rank tensor combinations, explicit
indexed declarations are the idiomatic way to state which axes line up.

### Broadcasting Without a Separate Ritual

Many array systems explain broadcasting as an algorithm over trailing
dimensions. Einlang's indexed declarations allow a more local reading:

```rust
let R[i, j] = row[i] + col[j] + bias;
```

`row[i]` depends on `i` and is invariant along `j`. `col[j]` depends on `j` and
is invariant along `i`. `bias` has no tensor index, so it is invariant along
both output axes. Nothing else needs to be said.

This is useful because it keeps the explanation attached to the formula. The
reader can see which term varies along which axis. The compiler can infer the
same fact from the presence or absence of index names.

### A Worked Reading: Affine Layers

A dense layer with a bias can be written:

```rust
let y[b, o] = sum[i](x[b, i] * W[i, o]) + bias[o];
```

There are three different index roles in one line. `b` is a batch axis that
survives. `o` is an output-feature axis that survives. `i` is an input-feature
axis that is consumed by the sum.

The bias term `bias[o]` depends on `o` but not on `b`, so it is shared across
the batch. There is no separate `broadcast_to` operation in the source. The
missing `b` is the broadcast.

This line also shows why the formula is more informative than a call such as
`linear(x, W, bias)`. The call names a concept. The indexed form names the axes
inside the concept.

## 3.2 Transpose and Axis Reordering

> **Think More.** A transpose can be implemented as data movement, a view, or
> even eliminated by later fusion. At the source level, it is only a different
> ordering of named axes. How much should a language expose about layout, and how
> much should it leave as a backend decision? A discussion can split here between
> the mathematical view of tensors and the systems view of memory. By abstracting
> transformations as coordinate mappings, we enable flexible composition where
> the same tensor can be viewed through different lenses without fixing the
> underlying representation. Implementation models range from eager transposition
> (immediate data rearrangement) to lazy transposition (deferred until access),
> each with different memory and performance trade-offs. Programming paradigms
> differ: some treat transposition as a data operation, others as a metadata
> operation, affecting how algorithms compose transformations.

Formula:

```text
(A^T)_{j,i} = A_{i,j}
```

Einlang:

```rust
let AT[j, i] = A[i, j];
```

Axis reordering is index reordering. The left side declares the result order;
the right side states how the result reads the input.

For a 3D tensor:

```rust
let U[i, k, j] = T[i, j, k];
let V[k, j, i] = U[i, k, j];
```

the two declarations are two permutations. A backend may compose them and avoid
unnecessary data movement when it can represent a view or fuse the consumer.

### Permutations as Axis Maps

Think of a transpose as a map from output coordinates to input coordinates:

```text
output point AT[j, i] reads input point A[i, j]
```

For a three-axis tensor:

```rust
let P[k, i, j] = T[i, j, k];
```

the output coordinate order is `(k, i, j)`, while the input is read in order
`(i, j, k)`. The declaration describes a coordinate transform. Whether the
backend materializes a new tensor, creates a view, or fuses the transform into a
consumer is a later decision.

## 3.3 Named Rest: Variable Numbers of Axes

> **Think More.** The rest pattern says that some axes should be preserved even
> though the formula does not care about their individual names. This is shape
> polymorphism by pattern, not by a separate type calculus. What kinds of generic
> tensor programs become possible if "the rest of the shape" can be named? What
> mistakes also become easier to make when a pattern hides several axes behind
> one word? This pattern enables polymorphic operations that work across varying
> tensor ranks, building on the index abstraction to create reusable components
> that maintain type safety through consistent naming. Implementation models
> differ: some systems expand rest patterns early, while others preserve them
> for later analysis. Programming paradigms range from explicit axis naming
> (verbose but clear) to rest patterns (concise but potentially confusing), where
> the choice affects both expressiveness and error detection.

Formula:

```text
sum over all axes except the last one
```

Einlang:

```rust
let sums[..batch] = sum[k](x[..batch, k]);
let grand = sum[..batch](sums[..batch]);
```

`..batch` is a named rest pattern. It matches a contiguous sequence of axes. In
`x[..batch, k]`, the final axis is named `k`; all earlier axes are matched by
`..batch`.

A batched matrix multiply can be written:

```rust
fn batch_matmul(A: [f32; *], B: [f32; *]) -> [f32; *] {
    let C[..batch, i, j] = sum[k](A[..batch, i, k] * B[..batch, k, j]);
    C
}
```

At a call site, monomorphization and rest-pattern preprocessing specialize the
definition for the concrete rank. This is shape polymorphism through index
patterns, not a separate type-level language.

### Why Rest Is Named

The name in `..batch` is not decorative. It lets the same unknown sequence of
axes appear in several places:

```rust
let y[..batch, j] = x[..batch, j] + bias[j];
```

Both occurrences of `..batch` must match the same axis sequence. That is what
lets the compiler know that the batch axes of `y` and `x` line up. If a rest
pattern appears in the output but cannot be determined from an input access, the
program is under-specified and should be rejected.

Rest patterns are therefore not "variadic arrays" in the vague sense. They are
named pieces of an index list, and the name gives the compiler a consistency
condition.

## 3.4 Index Arithmetic and Convolution

> **Think More.** Convolution becomes less mysterious when it is read as
> coordinate arithmetic plus reduction. But this raises a design question: should
> the language provide a special `conv` primitive, or should it let the compiler
> recognize the indexed pattern and choose an efficient lowering? This question
> opens a broader discussion about whether domain knowledge belongs in syntax,
> libraries, or compiler intelligence. Implementation models range from primitive
> convolution operations (efficient but limited) to pattern recognition
> (flexible but complex), each affecting how domain-specific optimizations are
> implemented. Programming paradigms differ: some embed domain operations in the
> language, others keep them in libraries, affecting both performance and
> extensibility.
> libraries, or compiler pattern recognition. By expressing convolution through
> index arithmetic, we build complex operations from the same primitives used
> for simpler transformations, enabling a unified approach to tensor
> computations.

Formula:

```text
O_{n,oc,oh,ow} =
  sum_{ic,kh,kw} I_{n,ic,oh+kh,ow+kw} K_{oc,ic,kh,kw}
```

Einlang:

```rust
let O[n, oc, oh, ow] = sum[ic, kh, kw](
    I[n, ic, oh + kh, ow + kw] * K[oc, ic, kh, kw]
);
```

Convolution is index arithmetic plus reduction. The input coordinate expression
keeps the relation between output coordinates and input coordinates visible
without introducing extra names.

Because the compiler can see the index equations, it can reason about legal
iteration spaces and avoid out-of-bounds reads. Backends may lower this pattern
as scalar loops, vectorized NumPy operations, or a specialized kernel when one
is available. The source stays close to the formula.

### A Small Interpreter for Index Arithmetic

For one output point `(n, oc, oh, ow)`, the convolution definition becomes:

```text
sum over ic, kh, kw:
    I[n, ic, oh + kh, ow + kw] * K[oc, ic, kh, kw]
```

No additional output axis is introduced by `oh + kh` or `ow + kw`; these are
ordinary index expressions inside the input access. Stride, dilation, padding
conventions, and window placement can be expressed by changing those index
expressions while leaving the reduction structure recognizable.

### Transformations as Delayed Decisions

A recurring theme in this part is that the source program states coordinate
relationships, not storage operations. A transpose need not imply an immediate
copy. A broadcast need not imply allocating a larger tensor. A convolution need
not imply a particular lowering strategy.

The source gives the compiler these facts:

```text
which axes exist
which terms depend on which axes
which coordinates are derived from others
which axes are consumed by reductions
```

The backend then chooses a representation. This separation is one of the main
reasons to keep index structure explicit. It delays low-level choices until the
compiler has enough context to make them well.

## 3.5 Transformation Algorithms and Patterns

> **Think More.** Transformations like broadcasting, transposition, and
> reshaping form the connective tissue of tensor algorithms, enabling operations
> to work across different shapes and layouts. When transformations are explicit
> in the index structure, compilers can optimize memory access patterns, avoid
> unnecessary copies, and generate efficient code. Implementation models range
> from view-based transformations (no data movement) to copy-based
> transformations (explicit rearrangement), each with different performance and
> memory characteristics. Programming paradigms differ: some languages treat
> transformations as metadata (cheap views), while others treat them as data
> movement (explicit operations), affecting how algorithms are structured and
> composed.

Explicit transformation patterns enable sophisticated algorithms that would be
complex or impossible with implicit approaches.

### Batch Processing with Broadcasting

Applying operations across batches of data:

```rust
let batch_data[batch in 0..32, feature in 0..128] = input_data[batch, feature];
let weights[feature in 0..128, output in 0..64] = trained_weights[feature, output];

// Broadcast weights across batch dimension
let weighted[batch, feature, output] = batch_data[batch, feature] * weights[feature, output];
let activations[batch, output] = sum[feature](weighted[batch, feature, output]);
```

Broadcasting enables efficient batch processing without explicit loops.

### Transpose-Based Algorithms

Matrix operations using transposition:

```rust
let A[i in 0..3, j in 0..4] = i * 4 + j;
let B[j in 0..4, i in 0..3] = A[i, j];  // Transpose

// Matrix-vector multiplication using transpose
let x[j in 0..4] = [1, 2, 3, 4][j];
let result[i in 0..3] = sum[j](B[j, i] * x[j]);
```

Transposition enables different algorithmic approaches to the same computation.

### Convolution Patterns

Multi-dimensional convolution using index transformations:

```rust
let input[h in 0..28, w in 0..28] = image_data[h, w];
let kernel[kh in 0..3, kw in 0..3] = conv_weights[kh, kw];

// Convolution with stride 1
let output[oh in 0..26, ow in 0..26] = sum[kh, kw](
    input[oh + kh, ow + kw] * kernel[kh, kw]
);
```

Index arithmetic expresses spatial relationships directly.

### Reshaping for Different Views

Treating data with different dimensional interpretations:

```rust
let flat[i in 0..12] = sequential_data[i];

// View as 3x4 matrix
let matrix[row in 0..3, col in 0..4] = flat[row * 4 + col];

// View as 2x2x3 tensor
let tensor[a in 0..2, b in 0..2, c in 0..3] = flat[a * 6 + b * 3 + c];
```

Reshaping enables algorithms to work with different data interpretations.

### Implementation Models for Transformations

**View-Based Model**: Transformations create lightweight views without copying
data. Efficient for memory but may complicate optimization.

**Copy-Based Model**: Transformations explicitly rearrange data. Simpler for
some optimizations but increases memory usage.

**Lazy Evaluation Model**: Defer transformations until data is actually
accessed. Enables fusion optimizations but can hide performance costs.

**Compile-Time Model**: Analyze transformation patterns and generate optimized
code. Best performance but requires sophisticated analysis.

**Runtime Model**: Perform transformations dynamically based on data shapes.
Flexible but may incur runtime overhead.

Each model affects how algorithms are written and optimized.

## 3.6 Views, Layout, and Why Transformations Matter to Implementers

Broadcasting and axis reordering are sometimes dismissed as bookkeeping. That
reaction is understandable if one meets them only as awkward preparatory calls
before "the real computation." But in tensor work, bookkeeping is often where
the real meaning of a program hides. A transformation answers questions such as:
which axes are supposed to align, which values are conceptually repeated, which
coordinates are merely being renamed, and which layout choices can be delayed
until the backend knows enough to make them well?

Einlang's contribution is to make these questions explicit in the source rather
than scattering them across convenience APIs. The payoff is not only aesthetic.
It is practical. Once a program states transformations as coordinate
relationships, a compiler can distinguish three very different cases that many
library interfaces blur together:

- a value that truly needs to be materialized in a different arrangement;
- a value that can be represented as a view over existing storage;
- a value whose transformation can disappear entirely because a later consumer
  reads it in the transformed pattern already.

### Broadcasting as a Statement of Invariance

The most important idea behind broadcasting in this chapter is not "make shapes
match." It is "state clearly which axes a term does not depend on." That is a
deeper and more general claim. If a term lacks an output index, it is invariant
along that dimension. The compiler can then treat repetition as conceptual
rather than immediately allocating a larger array.

This is easier to appreciate if we look at a bias term in a batched model:

```rust
let y[b, o] = sum[i](x[b, i] * W[i, o]) + bias[o];
```

The source is not asking to allocate a giant matrix full of repeated bias
values. It is asserting that `bias[o]` varies with `o` but not with `b`. That is
an invariant statement about the expression. The implementation may realize it
with scalar reuse, register reuse, vector lanes, or some more elaborate kernel
strategy, but the language never had to pretend that a physical expansion was
the real idea.

### Layout Is a Backend Concern, but Not an Invisible One

A language can decide not to expose concrete memory layout in source while still
being extremely aware that layout exists. That is Einlang's position. The source
need not talk in row-major or column-major terms every time it transposes a
tensor, but a backend absolutely cares whether a given access pattern is
contiguous, strided, tiled, or cache-friendly.

The explicit axis notation helps because it separates two concerns cleanly. The
source says which coordinate transform is intended:

```rust
let AT[j, i] = A[i, j];
```

The backend decides whether that transform should be represented by changed
strides, delayed indexing arithmetic, or an eager physical copy. If a later
consumer immediately reads `AT[j, i]` in a way that cancels the transpose, the
copy may vanish. If a later kernel expects a particular layout, materialization
may be worthwhile. The crucial point is that the source and the storage plan are
allowed to be different layers of description.

### The Cost of Premature Layout Commitment

Many high-performance systems make layout choices early because those choices
can produce excellent kernel performance. The danger is that early commitment
can also force later transformations to become expensive. If every transpose is
treated as a concrete data movement too soon, programs accumulate avoidable
copies. If every broadcast is turned into a real expansion, memory pressure
grows for no semantic reason.

Structure-visible transformations give an implementation more room. The compiler
can hold onto the coordinate relationship and wait to see how it composes with
later operations. This is one of the quiet strengths of the notation. The
surface program remains close to the formula while the backend still has time to
make intelligent systems choices.

## 3.7 Shape Polymorphism, Reuse, and the Value of Named Rest Patterns

The rest-pattern syntax in this chapter is easy to underestimate because it is
compact. A token like `..batch` can look like a small convenience layered atop
the "real" tensor language. In practice it carries a major idea: some tensor
equations should be reusable across different ranks without losing structural
meaning.

When we write

```rust
let y[..batch, j] = x[..batch, j] + bias[j];
```

we are not just avoiding a longer list of axis names. We are saying that the
equation is indifferent to the internal identity of a whole contiguous block of
leading axes, so long as the same block appears consistently on both sides.
Those leading axes might represent a simple batch dimension, or batch plus
sequence, or batch plus heads plus time, depending on the caller. The named rest
pattern captures the shared structural fact without overcommitting to one rank.

### Generic Programs Need Visible Contracts

Generic tensor code is often written in one of two unsatisfying ways. Either it
is too specific, naming every axis and forcing callers to reshape data into that
template, or it is too vague, hiding structure inside ad hoc host-language shape
manipulation. Named rest patterns offer a middle path. They say, in effect:
"there is a sequence of axes here, and I do not care about each axis
individually, but I do care that the same sequence flows through the equation."

That is a real contract. It is not merely a shortcut. Because the contract is
named, the compiler can enforce that multiple occurrences refer to the same
axis sequence. This is what makes shape polymorphism intelligible rather than
mystical. The source is still talking about structure; it is simply abstracting
over part of the structure in a disciplined way.

### Reuse Without Losing Readability

One common failure mode in generic numerical code is that reuse destroys local
readability. A helper becomes "general" only because it stops naming what it is
doing clearly. Rest patterns resist that failure mode. A function such as

```rust
fn normalize_last(x: [f32; *]) -> [f32; *] {
    let denom[..batch] = sum[k](x[..batch, k]);
    let y[..batch, k] = x[..batch, k] / denom[..batch];
    y
}
```

remains readable. The reader can see immediately that all leading axes pass
through, that the last axis is reduced for the denominator, and that the result
reuses the same leading axes. The abstraction has not erased the geometry of the
operation.

## 3.8 Transformation Families as a Way of Thinking

Perhaps the deepest lesson of this chapter is that transformations are not a
grab-bag of special operations. They are a family of related ideas about how
coordinates determine meaning. Broadcasting says a term ignores an axis.
Transpose says the result names the same coordinates in a different order.
Convolution says one coordinate is derived from another through arithmetic.
Rest patterns say a whole group of axes may move through an equation as a named
block. These are all facets of one underlying viewpoint: tensor programs are
coordinate programs.

Once that viewpoint clicks, many library rituals look different. A mysterious
chain of `unsqueeze`, `transpose`, `reshape`, and `expand` calls in another
system can often be re-read as a sequence of straightforward coordinate claims.
The point of Einlang is not to eliminate all need for transformation. It is to
make transformation legible enough that the reader can still understand the
computation as a whole.

### Why This Matters for Humans

A reader trying to maintain or review a tensor program rarely struggles because
they cannot execute loops in their head. They struggle because the connection
between the intended mathematics and the written code has become obscure.
Transformation-heavy code is notorious for this. One dimension is inserted only
to satisfy a later broadcast. Another is swapped only because a kernel expects a
particular convention. A third disappears into a helper function whose name
conceals the exact axis logic.

When the program instead says

```rust
let scores[b, h, q, k] = sum[d](Q[b, h, q, d] * K[b, h, k, d]);
```

or

```rust
let AT[j, i] = A[i, j];
```

the reader can recover the geometry locally. That does not make the whole
system trivial, but it does keep each step accountable. The source earns the
right to be dense because its density carries visible structure rather than
compressed accidents of API use.

### Why This Matters for Compilers

Compilers benefit for parallel reasons. A transform represented as a named
relation among indices can be composed, simplified, or canceled with other
relations. A transform represented only by opaque runtime calls is much harder
to reason about globally. Once again the language is doing something larger than
syntax. It is choosing a representation of intent that survives long enough for
tooling to act on it.

This becomes especially important when transformations interact with reduction
and autodiff. A transposed value may be reduced along a particular axis. A
broadcasted bias may contribute to a gradient summed over a batch. A convolution
pattern may later be recognized by a lowering pass. All of these depend on the
fact that coordinate relationships were not erased too early.

### The Chapter's Wider Argument

The wider argument of this chapter is therefore simple but far-reaching:
transformation should be treated as meaning, not cleanup. Once we accept that,
it becomes natural to give broadcasting, permutation, and coordinate arithmetic
a direct place in the language. The result is not merely prettier code. It is a
programming model in which data shape, coordinate logic, and implementation
strategy can be discussed in one vocabulary.

## 3.9 A Reader's Checklist for Transformation-Heavy Code

When a tensor program becomes difficult to follow, the problem is often not the
arithmetic. It is that the reader has lost track of how coordinates are being
reinterpreted. A useful discipline is to annotate transformation-heavy code with
four simple questions. Which axes survive from input to output? Which axes are
newly introduced? Which axes are merely reordered or renamed? Which terms are
constant along some output direction and therefore only conceptually repeated?

This checklist sounds modest, but it scales surprisingly well. It clarifies why
a bias broadcasts across a batch, why a transpose can disappear into a later
consumer, why a convolution window is not a new axis in the output, and why a
named rest pattern is stronger than a vague "any leading dimensions are okay"
promise. In effect, it turns transformations from a pile of tactics into a
stable reading method.

That stability is what makes this chapter more than a catalog of array tricks.
It teaches a habit of interpretation. Once that habit is learned, many
apparently different tensor rewrites collapse into one family of coordinate
arguments. That is the wider payoff of treating transformations as first-class
source structure.

## 3.10 Transformations as the Grammar of Tensor Programs

Elementwise arithmetic tells us what happens at a point. Transformations tell us
how points relate across families. In that sense, broadcasting, permutation,
rest patterns, and index arithmetic together form much of the grammar of tensor
programs. They explain how one family of values is reinterpreted, aligned, or
projected into another family before any heavy arithmetic even begins.

This is why so much tensor code lives or dies by the clarity of its
transformations. When those relationships are explicit, the program stays close
to its mathematics. When they are obscured by convenience calls and hidden shape
manipulation, the arithmetic may still be correct, but the program becomes hard
to trust. Einlang's transformation syntax therefore earns its place by doing
more than saving keystrokes. It gives shape logic a readable grammar of its own.

## 3.11 Why Shape-Sensitive Readers Care

Readers who work seriously with tensor systems quickly learn that many bugs,
performance surprises, and interpretive mistakes come not from the local
arithmetic but from the silent handling of shape. A tensor program can compute
the right scalar formula in the wrong geometric arrangement and still mislead
its reader for a long time. Broadcasting and transformation syntax matter
because they put shape logic where it can be discussed openly rather than
rediscovered from side effects.

That openness is valuable across audiences. The beginner gains a clearer map of
which axes are doing what. The practitioner gains a more trustworthy description
of how data is being aligned and reused. The implementer gains visible
relationships that may later support views, fusion, or layout-aware lowering.
The same explicitness serves all three because shape is not an incidental
property of tensor programs. It is one of their main sources of meaning.

For that reason, transformation syntax should not be treated as auxiliary
ornament around "real" tensor computation. In many systems it is the part that
most determines whether the computation can be trusted, explained, or optimized.
Einlang gives it a first-class place because shape logic deserves to live in the
open.

That is why this chapter continues to matter even when the later mathematics
becomes more advanced. Broadcasting, permutation, and coordinate arithmetic are
not introductory housekeeping concepts we quickly outgrow. They are the durable
means by which tensor programs keep their geometry intelligible.

Whenever a tensor program feels confusing, it is often worth returning to the
transformations first. Very often the arithmetic is innocent and the real source
of difficulty is hidden geometry. A language that surfaces that geometry is
therefore doing more than offering convenience. It is protecting the reader's
ability to understand the program at all.

That protection is one reason transformation syntax belongs in the conceptual
center of tensor programming. Geometry is not an accessory to the computation.
It is one of the things the computation is about.

### A Final Rule of Thumb

When two tensor programs compute the same arithmetic but differ in how clearly
their axes line up, survive, or move, prefer the one whose transformation logic
you can explain aloud. In practice, that is usually the one that will age
better, teach better, and optimize better too.

That interpretive habit is one of the most transferable skills the chapter
offers to anyone who reads tensor code seriously.

It helps with design, debugging, and communication alike. Once a reader starts
to see transformations as public statements about coordinate meaning, many
otherwise mysterious tensor programs become discussable in ordinary language.
That is a real gain, and it is part of why this material deserves space in the
book rather than a quick mention.

It also gives the chapter a durable afterlife: long after specific APIs change,
the underlying geometry remains readable.

That durability is one of the best reasons to learn it.

It keeps tensor shape from becoming an afterthought.

## Summary

Broadcasting and transformations reveal the elegance of index relationships as
the foundation for tensor composition. What emerges is not a collection of
special-case operations, but a unified framework where shape compatibility,
memory efficiency, and computational intent are all expressed through index
dependencies. This chapter demonstrates how making transformations explicit
turns what could be opaque optimizations into transparent, analyzable
operations.

The principles that emerge are both simple and profound:

- A missing output index means invariance along that axis, enabling
  broadcasting as a natural consequence of index relationships;
- Axis reordering is a permutation of index names, making transposition a
  coordinate transformation rather than a data movement operation;
- `..batch` names a sequence of axes and lets equations survive rank changes,
  enabling polymorphic operations across different tensor shapes;
- `where` names derived indices and gives range analysis more information,
  turning conditional operations into structured transformations.

These principles create a foundation where transformations are not
afterthoughts, but integral parts of the computational model. The compiler can
reason about memory layouts, optimize data movement, and verify shape
compatibility because the intent is visible in the source. This visibility
transforms optimization from guesswork into systematic analysis.

The next chapter builds on this foundation by asking a transformative question:
once values are named, can the language ask how one value changes with respect
to another? The answer leads us into the world of automatic differentiation,
where relationships between values become the basis for computing derivatives.
