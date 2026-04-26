---
layout: book
title: "Chapter 1: Data and Indices"
---

# Chapter 1: Data and Indices

This chapter starts from the smallest possible Einlang program and builds up to
the central idea of the language: tensor structure should be visible in the
source. A scalar gives us a stable name; an index turns one equation into a
family of equations; multiple indices make rank and shape part of the program's
surface. This visibility is not merely syntactic convenience—it fundamentally
changes how we think about computation. By making structure explicit, we enable
the compiler to reason about shapes, dependencies, and transformations in ways
that implicit approaches cannot. The result is a language where the source code
serves as both specification and optimization guide, where every index reveals
computational intent rather than hiding it behind opaque operations.

## 1.1 Scalars: Naming the Smallest Number

> **Think More.** A name is the smallest abstraction a language can offer, but
> it is also a promise: later parts of the program may ask about this value
> again. What changes if we treat names not as conveniences, but as stable places
> where shape checks, dependency analysis, printing, and differentiation can
> attach? A discussion can start by comparing three views of a name: a label for
> the reader, a handle for the compiler, and a coordinate in the program's
> dependency graph. Which analyses become harder if the name can silently change
> meaning?

The first abstraction in a tensor language is not a tensor. It is a name. A
tensor program with no stable names quickly becomes a pile of temporary
expressions. A program with stable names can be inspected, differentiated, and
composed. This stability transforms programming from a sequence of operations
into a network of relationships, where each name becomes a point of leverage
for analysis and transformation. The choice of immutable bindings over mutable
variables is not arbitrary—it enables the compiler to track dependencies
across the entire program, opening doors to optimizations that would be
impossible with mutable state. When a name is stable, differentiation becomes
a local question between named values, and shape inference can propagate
constraints throughout the computation graph.

Formula:

```text
x = 5
```

Einlang:

```rust
let x = 5;
```

A scalar binding is the smallest useful Einlang program. `let` gives a value a
name. It does not assign into a mutable cell. This distinction matters deeply:
once `x` is bound, later expressions can rely on `x` meaning the same value
throughout its scope. Shape analysis, dependency analysis, and autodiff all
begin with this stable naming contract. The immutability creates a foundation
where the program's meaning is determined by its structure, not by the
sequence of state changes. This structural approach enables powerful analyses
that would be fragile or impossible in languages where names can be
reassigned.

```rust
let x = 5.0;
let y = x * x + 3.0 * x;
let dy_dx = @y / @x;
```

The derivative request is possible because both sides of the relationship were
named first. Einlang's autodiff syntax works on named bindings, not arbitrary
unbound fragments. This design choice elevates differentiation from a
library function to a first-class language construct, where the relationship
between `y` and `x` can be interrogated directly. The named approach enables
compositional differentiation, where complex expressions can be differentiated
by composing simpler derivative relationships.

### The Environment Model

A useful way to read an Einlang program is as a growing environment:

```text
{}                         before any binding
{ x -> 5 }                 after let x = 5
{ x -> 5, y -> x*x + 3*x } after let y = ...
```

This is a semantic model, not a claim about how every backend stores values. The
runtime may evaluate eagerly, delay a computation, or lower a tensor expression
to a backend kernel. The stable fact is that later expressions refer to earlier
bindings by name.

Redeclaration in the same scope is deliberately rejected:

```rust
let x = 1;
let x = x + 1;
```

The second line should be a compile-time error, because the same scope already
contains a binding named `x`. If a later value needs a name, give it a distinct
one:

```rust
let x = 1;
let x_next = x + 1;
```

This keeps the program analyzable: shape facts and derivative facts attach to
stable bindings, not to mutable boxes or ambiguous same-scope rebinding.

### Names as Places to Ask Questions

The name is also where later questions attach. A derivative request:

```rust
let dy_dx = @y / @x;
```

is meaningful because `x` and `y` are both bindings in the source environment.
The question is not "differentiate the preceding line" or "differentiate this
anonymous callback." It is "how does the named value `y` change with respect to
the named value `x`?"

That distinction becomes important in larger programs. We may name a hidden
state, a loss, a row of a matrix, or an intermediate normalization constant. Any
of those names can become the target of later inspection. This naming strategy
transforms debugging from runtime observation to static analysis, where the
relationships between values are visible in the source code itself.

## 1.2 Vectors: Bringing Subscripts Back

In ordinary programming, a vector is often introduced as a container. In this
book, it is more useful to introduce a vector as an indexed family of scalar
values. The container is one possible representation of the family. The
subscript is the idea. This conceptual shift from storage to structure enables
the compiler to reason about the vector's properties independently of how it's
stored, opening doors to optimizations that container-based thinking obscures.
When we think in terms of families rather than containers, operations become
compositional and analyzable in ways that traditional array programming cannot
match.
parallelization, or symbolic manipulation. Different programming models emerge
- imperative loops favor control over execution order, while declarative
equations favor composition and analysis. The choice reflects fundamental
trade-offs between expressiveness and analyzability, where indexed equations
enable powerful compiler transformations but require programmers to think in
terms of families rather than sequences.

In ordinary programming, a vector is often introduced as a container. In this
book, it is more useful to introduce a vector as an indexed family of scalar
values. The container is one possible representation of the family. The
subscript is the idea.

Formula:

```text
v_i = 2i
w_i = v_i + i
```

Einlang:

```rust
let v[i in 0..5] = i * 2;
let w[i] = v[i] + i;
```

The name `i` is an index placeholder scoped to the indexed definition. It is not
a host-language loop variable that remains in scope afterward. One expression
defines the whole vector.

When the range is explicit, as in `i in 0..5`, the output length is known from
the declaration. When the range is omitted, the compiler infers it from indexed
uses such as `v[i]`. In `let w[i] = v[i] + i;`, `i` ranges over the first axis of
`v`.

The compiler-visible fact is:

```text
one output index -> rank 1 tensor
```

No string API or hidden loop needs to be decoded.

### Indexed Binding as a Family of Scalars

The definition:

```rust
let v[i in 0..5] = i * 2;
```

can be read as five scalar equations:

```text
v[0] = 0 * 2
v[1] = 1 * 2
v[2] = 2 * 2
v[3] = 3 * 2
v[4] = 4 * 2
```

The indexed program is not less precise than the expanded version. It is more
precise about the abstraction: all five points belong to the same family and
are generated by the same rule. A loop is one possible implementation of this
family; it is not the source-level idea.

This is the first recurring pattern in the book. We write the mathematical
family directly and let the compiler derive the mechanical traversal.

### The Index Is Scoped

The index name `i` does not leak:

```rust
let v[i in 0..5] = i * 2;
let w[i] = v[i] + i;
```

The `i` in the first line and the `i` in the second line are separate binders
that happen to use the same spelling. This is like using the name `x` as a
function parameter in two different functions. The spelling is reused; the scope
is different.

This matters because it prevents accidental global meaning from attaching to an
index. There is no single universal `i` in the program. There are only local
index binders whose ranges are inferred or declared in context.

## 1.3 Matrices and Higher-Rank Tensors

> **Think More.** Rank is often treated as metadata, something discovered after
> an array already exists. Einlang moves rank into syntax. Does that make tensor
> programs more verbose, or does it make intent harder to hide? Where should a
> language draw the line between explicit structure and visual noise? A useful
> answer depends on what the language wants to make checkable: shape agreement,
> axis use, and errors near the source.

Once we understand a vector as a one-index family, a matrix is not a new kind of
object. It is a two-index family. A higher-rank tensor is the same idea repeated:
more coordinates are needed to select a point.

Formula:

```text
C_{i,j} = A_{i,j} + B_{i,j}
```

Einlang:

```rust
let C[i, j] = A[i, j] + B[i, j];
```

The number of indices on the left determines the result rank. A vector has one
axis, a matrix has two, and higher-rank tensors have more.

For this example, `i` is inferred from axis 0 of `A` and `B`; `j` is inferred
from axis 1 of `A` and `B`. Those ranges must agree. If the shapes are
incompatible, the compiler reports a shape mismatch.

The shape rule follows from the indexed reads. The syntax has already exposed
the relationship between output axes and input axes.

### Rank Is Visible

Rank is the number of axes in a rectangular tensor. In Einlang, the output rank
is visible in the binding form:

```rust
let a[i] = x[i];             // rank 1
let b[i, j] = x[i, j];       // rank 2
let c[i, j, k] = x[i, j, k]; // rank 3
```

That visibility seems small, but it changes the way errors can be reported. If
a matrix is read as `A[i]`, the program has supplied one index to a two-axis
value. If a vector is read as `v[i, j]`, the program has supplied too many
indices. The compiler does not have to interpret a string or inspect an opaque
callback to find the mismatch. The mismatch is at the subscript.

Rank visibility is also a teaching device. The notation reminds the reader that
a tensor is not just "an array" but a value with axes, and those axes can be
named.

### Shape as Agreement Between Uses

Consider:

```rust
let C[i, j] = A[i, j] + B[i, j];
```

The index `i` appears in two reads, `A[i, j]` and `B[i, j]`. That is not merely
two loops using the same counter. It is a claim that the first axis of `A` and
the first axis of `B` are the same index domain. Likewise for `j` and the second
axis.

Shape checking is therefore a consistency check over index meanings. The
compiler asks whether all uses of `i` agree, and whether all uses of `j` agree.
This is a richer view than "do these arrays have compatible sizes?" It says why
the sizes must be compatible: the source program gave them the same index name.

## 1.4 Slices and Subtensors

> **Think More.** A slice fixes some coordinates and leaves others free. This is
> more than a convenience for extracting data; it is a way of asking which axes
> still matter after part of the context has been chosen. How should a compiler
> represent that partial choice: as a view, a copy, a delayed expression, or a
> proof obligation about bounds? Each answer changes performance, diagnostics,
> and the programmer's mental model.

Einlang uses zero-based indexing:

```rust
let col[i] = A[i, 0];
let row[j] = A[4, j];
```

Fixing one index removes one axis from the result. `A[i, 0]` varies only with
`i`, so `col` is a vector. `A[4, j]` varies only with `j`, so `row` is also a
vector.

Constant indices can be checked against known shapes. If `A` has type
`[f32; 3, 4]`, then `A[4, j]` is invalid because the first axis has only
indices `0`, `1`, and `2`. Dynamic shapes may require runtime checks, but the
same rule holds: each subscript must be valid for its axis.

### Lowering Dimension by Dimension

Slicing is not a separate operation in the core notation. It is ordinary
indexing with some axes fixed and some axes left symbolic:

```rust
let cell = A[4, 0];    // both axes fixed, scalar
let row[j] = A[4, j];  // first axis fixed, second survives
let col[i] = A[i, 0];  // second axis fixed, first survives
```

The output shape follows from the surviving index names. This makes slicing fit
the same model as everything else in this part: values are families, and the
indices that remain variable determine the family produced by the binding.

### A Complete Worked Reading

Read the following fragment from left to right:

```rust
let image2[i, j] = image[i, j] * image[i, j];
let top_row[j] = image2[0, j];
let energy = sum[j](top_row[j]);
```

The first binding defines a two-axis family. Both `i` and `j` survive, so
`image2` has the same two index domains as `image`. The second binding fixes the
first axis at `0`, so only `j` survives. The third binding consumes `j` with a
sum, so no axis survives and `energy` is scalar.

This is a tiny program, but it already shows the compositional story: define a
family, restrict the family, reduce the remaining family to a scalar. No loops
or temporary mutation are part of the source explanation.

## 1.5 Algorithms with Indices

> **Think More.** An algorithm is not just a sequence of steps; it is a pattern
> of index relationships that can be composed and analyzed. When we write
> algorithms with explicit indices, we expose the data dependencies that loops
> often hide. What optimizations or error messages become possible when those
> relationships are visible in the source?

Beyond basic indexing, Einlang's index notation enables direct expression of
many common algorithms. Let's explore several examples that demonstrate how
indexing patterns capture algorithmic intent.

### Linear Search

Finding the first occurrence of a value in an array:

```rust
let data[i in 0..10] = i * i;  // [0, 1, 4, 9, 16, 25, 36, 49, 64, 81]
let target = 25;

// Find first index where data equals target
let matches[i] = data[i] == target;
let positions[j in 0..10] = j;
let valid_positions[k] = positions[k] where matches[k];
let first_match = valid_positions[0] if len(valid_positions) > 0 else -1;
```

This expresses the search as a data-parallel operation: compute all matches,
then select the valid positions. Different backends might implement this as a
sequential scan, a parallel mask operation, or GPU kernels.

### Finding Maximum with Index

Locating both the value and position of the maximum:

```rust
let values[i in 0..5] = [3, 7, 2, 9, 1][i];

// Find maximum value
let max_val = max[i](values[i]);

// Find index of maximum (first occurrence)
let is_max[i] = values[i] == max_val;
let indices[j in 0..5] = j;
let max_indices[k] = indices[k] where is_max[k];
let max_index = max_indices[0];
```

The index-aware approach reveals that finding maxima involves both value
comparison and position tracking. This enables optimizations like SIMD
comparisons or tree-based reductions.

### Counting Elements

Counting how many elements satisfy a condition:

```rust
let numbers[i in 0..8] = i % 3;  // [0, 1, 2, 0, 1, 2, 0, 1]

// Count elements equal to 1
let is_one[i] = numbers[i] == 1;
let count = sum[i](is_one[i] ? 1 : 0);
```

This pattern extends to more complex predicates and can be composed with other
operations. The implementation might use population count instructions, parallel
reduction, or sequential accumulation depending on the target architecture.

### Basic Selection Sort

A sorting algorithm expressed through index manipulation:

```rust
let arr[i in 0..4] = [3, 1, 4, 2][i];

// For each position, find the minimum remaining element
let sorted[j in 0..4] = min[k in j..4](arr[k]);

// This creates [1, 2, 3, 4] - the sorted array
```

While not a complete in-place sort (Einlang favors functional approaches), this
demonstrates how index ranges can express algorithmic patterns. More complex
sorts like quicksort or mergesort can be expressed using recurrence patterns
covered in later chapters.

### Implementation Models for Indexing

Different compilation strategies reveal different performance characteristics:

**Eager Evaluation Model**: Allocate full arrays and compute all elements
immediately. Simple to implement but may waste memory for sparse computations.

**Lazy Evaluation Model**: Defer computation until elements are actually
accessed. Useful for large arrays where only portions are used, but can lead to
unpredictable performance.

**Kernel Generation Model**: Compile index patterns into optimized loops or
SIMD operations. The compiler analyzes the index relationships to choose the
best implementation strategy.

**Symbolic Model**: Treat indexed expressions as symbolic templates that can
be manipulated algebraically before evaluation. Enables powerful optimizations
like constant folding across indices.

Each model has trade-offs in compilation time, runtime performance, and memory
usage. The choice depends on the target hardware and usage patterns.

## 1.6 Reading Programs as Structured Equations

One of the easiest ways to misunderstand Einlang is to read it as ordinary code
that merely happens to contain brackets. That reading misses the point. The
central move of the language is not "put array syntax into a compiler." It is
"treat the visible structure of a formula as part of the source language."
Once we adopt that reading, a chapter like this stops being a tour of syntax and
starts becoming a lesson in how to see computation.

Consider again a small fragment:

```rust
let A[i, j] = x[i] * y[j];
let row0[j] = A[0, j];
let total = sum[j](row0[j]);
```

An imperative reader may translate this immediately into hidden loops:

```text
for i:
  for j:
    A[i, j] = ...
```

That translation is not wrong as a possible implementation, but it is too early.
The source program itself says something more useful. The first line defines a
two-index family. The second line fixes one coordinate and retains another. The
third line consumes the remaining coordinate. The program is therefore telling a
story about structure before it tells a story about execution. That order
matters. If we preserve it, we gain a vocabulary for shape checking, slicing,
optimization, and later differentiation. If we skip directly to loops, we lose
the very information the language was designed to expose.

This is a broader point about notation. Good notation does not merely abbreviate
longer procedures. It changes what kinds of questions become easy to ask. Roman
numerals can name numbers, but they make arithmetic awkward. Algebraic notation
can name the same quantities while making transformations visible. Einlang aims
for that second kind of improvement. It is not content with being a compact way
to write loops; it wants the program's structure to sit on the page in a form
that both people and compilers can interrogate.

### Three Simultaneous Readings

A useful discipline is to read any early Einlang fragment three ways at once.

First, read it denotationally. Ask: what family of values does this definition
describe? For

```rust
let v[i in 0..5] = i * 2;
```

the answer is a family with five points, one for each value of `i`, generated by
the rule `i * 2`.

Second, read it geometrically. Ask: which axes are present, and which
coordinates survive? A scalar has no axes. A vector has one axis. A matrix has
two. A slice is what remains after some coordinates are fixed. That geometric
reading becomes indispensable later, because transformations, broadcasting, and
reductions all depend on visible axis relationships.

Third, read it operationally, but only after the first two. Ask: what would an
implementation have to preserve for this definition to hold? It may use loops,
vectorized kernels, symbolic rewriting, lazy storage, or backend calls. The
source does not commit to one operational story, but it constrains all valid
stories. Any lowering must respect the family described and the axes named.

These three readings keep the language honest. They prevent the human reader
from collapsing the source into "whatever the runtime probably does," and they
prevent the implementation from treating syntax as decorative sugar over opaque
operations. The program stays meaningful at the level where structure is
visible.

### Why Explicitness Helps New Readers and Experts

There is a recurring fear that explicit index notation must be harder for
beginners because it shows more. In some cases, explicit notation does create an
up-front learning cost. But it also repays that cost quickly because the meaning
of a line can be recovered locally. A beginner who sees

```rust
let C[i, j] = A[i, j] + B[i, j];
```

can ask simple questions and get stable answers. How many axes does `C` have?
Two. Why? Because two indices survive on the left. Why do the shapes of `A` and
`B` have to agree? Because the same `i` and `j` appear in both reads. Where is
the result computed? At every pair of legal coordinates. That clarity is not
the enemy of pedagogy. It is often the thing that makes pedagogy possible.

Experts benefit for a related reason. Once the reader already knows array
programming, the pain point is usually not "I need a shorter way to write
addition." The pain point is that large tensor programs hide their assumptions
in library conventions, temporary reshapes, and host-language glue. Explicit
indices bring those assumptions back into the open. An expert can look at a line
and see rank, shape relationships, and dataflow without needing to mentally
reconstruct them from API contracts.

### Equations, Not Containers

The most important shift in this chapter is conceptual: tensors are introduced
as indexed families, not as containers first. That does not deny that a runtime
will eventually store data somewhere. It says storage is not the source-level
idea that best explains the language. A container-first story encourages us to
think in terms of filling slots. An equation-first story encourages us to think
in terms of defining a rule over a coordinate space.

That distinction becomes especially powerful once we leave dense arrays. A
family can be stored densely, sparsely, lazily, symbolically, or not at all
until demanded. The family notion survives all of those choices. A "container"
story quietly assumes one representation too early. By contrast, an indexed
equation remains stable across many backends and implementation strategies.

### Small Fragments as Compression

A compact Einlang fragment often contains more information than a longer
host-language equivalent. Consider the following Python-like pseudocode:

```text
for i in range(n):
    for j in range(m):
        C[i, j] = A[i, j] + B[i, j]
```

This code makes iteration order explicit, but the reasons for that order are not
part of the source. The fact that `C` is rank 2, that `A` and `B` must align
pointwise, and that the operation is elementwise rather than reductive all have
to be inferred from the whole construction. In Einlang,

```rust
let C[i, j] = A[i, j] + B[i, j];
```

the same information is compressed into the visible axis structure. Less text is
doing more semantic work. This is not terseness for its own sake. It is
compression that preserves the right information while discarding implementation
noise.

### A Habit for the Rest of the Book

Everything that follows depends on the habits introduced here. When we meet
reductions, we will ask which indices disappear. When we meet broadcasting, we
will ask which indices never appeared in a term. When we meet recurrence, we
will ask which coordinates refer backward to earlier ones. When we meet
autodiff, we will ask which named bindings define the relation we are
differentiating. Each of those later readings is already latent in this chapter.

That is why it is worth slowing down at the beginning. If a reader learns only
one thing here, it should be this: an Einlang program is not merely a sequence
of operations. It is a structured description of relationships over named values
and visible coordinates. The more faithfully we read it that way, the more of
the language becomes unsurprising.

## 1.7 Errors, Discipline, and the Cost of Ambiguity

A language that exposes structure also exposes mistakes earlier. This is one of
its main advantages, but it can feel severe if we are accustomed to systems that
defer many problems until runtime. It is therefore worth saying clearly what
kind of discipline this chapter asks for and why the discipline is productive.

The first demand is naming discipline. Reusing a name in the same scope is
forbidden because it muddies later questions. If one binding named `x` means two
different things at two different moments, then shape facts, derivative facts,
and dependency facts can no longer attach cleanly to that name. The language is
not being fussy. It is protecting the idea that a name is a stable site of
meaning.

The second demand is index discipline. Index names are not decorative marks that
can be shuffled carelessly. They state equalities of meaning across uses. If two
occurrences of `i` refer to the same coordinate domain, then they impose a
compatibility condition. If a read uses too many or too few indices, that is not
merely a typo in punctuation. It is a mismatch between the shape of a value and
the shape of the question being asked about it.

The third demand is range discipline. Sometimes a range is declared explicitly;
sometimes it is inferred from context. In either case, the language benefits
from the fact that a coordinate domain exists as a source-level object. A vague
"this probably runs over something compatible" is not enough. We want either a
spelled-out domain or enough indexed evidence to recover one.

### Why Strong Errors Are a Feature

In many numerical systems, the user sees a shape mismatch only after several
library calls have composed into a failing runtime operation. Worse, the error
may appear far from the source of the actual misunderstanding. A compiler that
understands indexed structure can often report the mismatch at the place where
the wrong relationship was asserted. That is more than convenience. It shortens
the gap between thought and correction.

Suppose a reader writes:

```rust
let C[i, j] = A[i, j] + v[j];
```

but intended `v[i]`. In a structure-visible system, the question "which axis is
`v` supposed to align with?" becomes available immediately. The names on the
page already encode the answer. We no longer have to debug a diffuse outcome
like "some broadcast happened and the result shape looks strange." The source
itself is a map of the intended relationships.

### Explicitness and Judgment

Explicit notation does not remove the need for judgment. A reader still decides
how much to name, where to introduce intermediate bindings, and when to split a
long definition into clearer parts. The examples in this chapter stay small so
that those decisions are easy to see. The same habit scales to larger programs:
name the relationships that matter, keep index domains visible, and make shape
claims explicit at the point where they are introduced.

### Discussion: Visible Structure Versus Convenience

The broader lesson of the chapter is not only that Einlang can write scalars,
vectors, and matrices. It is that a source language can choose where meaning
lives. In a library-heavy style, some structure is carried by function names,
some by array shapes, and some by runtime convention. In Einlang, the binding
name, the visible indices, and the ranges all participate in the same source
statement.

That choice is not free. The writer has to be precise about names and index
domains. The payoff is that readers and compiler passes can ask sharper
questions: which axes exist, which values agree pointwise, which index has been
fixed, and which shape claim failed? Convenience is still valuable, but this
chapter argues that convenience should not hide the relationships that later
analysis depends on.

## Summary

This chapter introduced tensors as named families of values. A scalar is a
named value, a vector is a one-index family, and a matrix is a two-index family.
Fixing an index produces a lower-rank value, and repeated index names express
shape relationships that the compiler can check.

- `let` binds an immutable name, creating stable points for analysis and
  transformation;
- output indices define tensor rank, making shape explicit rather than
  inferred;
- explicit ranges define output domains, enabling compile-time verification;
- omitted ranges may be inferred from indexed reads, balancing clarity with
  conciseness;
- constant indices can be checked against known shapes, catching errors early.

The point of this explicitness is not ceremony. It gives later chapters a stable
surface to build on: reduction can consume visible indices, broadcasting can
explain missing ones, recurrence can refer to earlier coordinates, and autodiff
can ask questions about named bindings.

The next chapter builds on this foundation by introducing reduction: the first
operation that deliberately removes an index and turns a family of values into a
summary.
