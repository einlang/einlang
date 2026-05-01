---
layout: book
title: "Naming Tensor Dimensions"
description: "A book about named tensor dimensions and compiler-visible structure."
---

# Naming Tensor Dimensions

_Making tensor roles visible to programs that would otherwise see only shape._

This book is about a common tensor-code problem: the program keeps the sizes
but loses the reason those sizes mattered.

Consider two lines:

```python
y = x + bias
z = y.reshape(batch, time, feature)
```

They may be correct. They may also hide a bug. Did `bias` mean "one value per
feature," or did it accidentally line up with some other axis of the same
length? Did the reshape preserve the intended roles, or only preserve the
number of elements? A shape checker can answer some questions here, but not all
of them.

Einlang is used in this book as a small reference language for writing those
roles down. The syntax is intentionally limited: indexed `let` bindings,
ordinary names, `sum`, derivative requests such as `@y / @x`, and recurrences
over an index such as `t`. I am using the language as a small instrument, not
as a catalog of features. The question is what a compiler can check once
coordinates have names.

Code blocks marked `rust` are Einlang source and use ordinary semicolons. Blocks
marked `text` are sketches: coordinate readings, dependency edges, or tables.
They are not meant to be parsed as source.

The core forms are small enough to keep in one place:

```rust
let y[i] = x[i] + 1;
let C[i, j] = sum[k](A[i, k] * B[k, j]);
let p[b, class] = softmax[class](logits[b, class]);
let pred[b] = argmax[class](p[b, class]);
let image[channel, row, col] = load_image();
let channels_last = move_channel[channel](image);
let swapped = swap[time, feature](x);
let dy_dx = @y / @x;
let h[t in 1..T] = step(h[t - 1], x[t]);
```

The bracketed names in calls such as `softmax[class]` and `argmax[class]` are
not decoration. They are coordinate arguments. They let common tensor
operations stay compact while still saying which coordinate is normalized,
consumed, preserved, or returned as an address.

That pattern appears throughout the book. A coordinate function is not a late
library convenience; it is the way common tensor ideas become reusable without
falling back to anonymous axis numbers. The expanded indexed form teaches the
contract. The coordinate-function form carries the same contract across a
function boundary.

Coordinate packs are the same idea at rank-polymorphic scale. A function can
say that one named axis matters while the surrounding axes should be inferred:

```rust
fn move_channel[channel, ..spatial](x: [f32; channel, ..spatial])
    -> [f32; ..spatial, channel]

let image[channel, row, col] = load_image();
move_channel[channel](image)
```

Here the caller names only the axis whose role matters. The `..spatial` pack is
not another argument to remember; it is inferred from the actual layout. When a
whole group must be chosen explicitly, the group is one parenthesized
coordinate argument, such as `layer_norm[(height, width, channel)](x)`.

Two distinctions are worth keeping close. An axis name is a semantic role:
`batch`, `time`, `feature`, `class`, `head`, `row`, `col`. A dimension size is
a number such as `64`. Two roles may have the same size and still not be
interchangeable. Also, Einlang does not write axis roles as scalar types.
Scalar and tensor types look like `i32`, `f32`, `[f32; 3, 4]`, `[f32; ?, ?]`,
and `[f32; *]`; an index such as `i` is a coordinate with a range, not a value
of type `batch`.

The compiler path behind the examples is concrete: parsing, name resolution,
AST-to-IR lowering, Einstein grouping, constraint classification, rest-pattern
preprocessing, range analysis, shape analysis, type inference, autodiff
rewriting, Einstein lowering, recurrence ordering, execution-fact recording,
and validation. You do not need to memorize that list before reading, but it
explains why the book keeps asking which fact should still be visible when a
later pass needs it.

The book uses a few words in a narrow way. An **axis name** is a role name such
as `batch`, `time`, `feature`, `head`, `row`, or `col`; an **axis position** is
a slot in a shape or layout; a **dimension size** is a numeric extent such as
`64`. A **coordinate map** states how input coordinates become output
coordinates. An **omitted coordinate** is present in the result but not used by
one term. A **consumed index** is local to a reduction, while a **surviving
index** remains in the result. A **shape contract** is a checked claim about
element type, rank, and extents. A compiler **transformation** is allowed to
rewrite the program only after those source facts have been captured.

## How the Chapters Fit

The chapters are ordered by the kind of fact that would otherwise be easy to
lose:

```text
named axes
rules from indices
compile-time autodiff
dependency graphs and storage
shape and type contracts
framework and compiler boundaries
principles under stress
```

That list is the spine of the book. Each part adds one kind of coordinate fact,
then asks the next part to reuse it:

```text
roles        say what an axis means
maps         say where a role moves
absence      says which role a term ignores
reduction    says which local role disappears
autodiff     asks which roles sensitivities must keep
recurrence   adds direction and observation
boundaries   preserve the facts across library calls
stress tests check whether the principle survives real model structure
```

The dependency is simple. If axes are named, index rules can talk about which
coordinates survive, which are consumed, and which are omitted from a term. Once
those relationships are in the source, autodiff rewrites, recurrence graphs,
shape contracts, and compiler transformations have something firmer than
position numbers to preserve.

Coordinate functions are the pressure valve that keeps this systematic without
making every program ceremonial. The expanded indexed form states the reference
meaning. The function form carries that meaning compactly: `softmax[class]`
names the normalized coordinate, `move_channel[channel]` names the moved
coordinate while inferring the surrounding pack, and `scan[t]` names the
ordered coordinate while leaving storage to lowering.

The difficulty gradient follows the same arc. Chapters 1 through 3 start with
static shape transformations, where the main question is "where did this
coordinate go?" Chapters 4 through 6 add local rules for omission, reduction,
and normalization. Chapters 7 through 9 reuse those rules for autodiff, where
the same coordinates now carry sensitivity. Chapters 10 through 12 add time and
observation, so dependencies and storage become part of the reading. Chapters
13 and 14 zoom out to framework and compiler boundaries. Chapters 15 and 16
then test the principle itself: first as a notation law, then under low-rank
communication and dynamic routing.

The standard-library references in the chapters are examples, not API
documentation: `transform_ops.ein` for coordinate maps, `reduction_ops.ein` for
reduction, `activations.ein` for softmax, `linalg_ops.ein` for contractions,
`recurrent_ops.ein` for time dependencies, and `attention_ops.ein` for a larger
communication pattern.

## Contents

Chapter numbers keep their stable URLs. The section titles describe the route
through the argument.

- [Preface](preface.html)

### Section I: Named Axes

- [1. What Can the Compiler Not See?](chapter-01-compiler-blindness.html)
- [2. Axis Roles Are Not Axis Positions](chapter-02-axis-roles.html)
- [3. Coordinate Maps in the Standard Library](chapter-03-coordinate-maps.html)

### Section II: Rules from Indices

- [4. What Does Broadcasting Hide?](chapter-04-broadcasting.html)
- [5. The Index That Leaves](chapter-05-index-that-leaves.html)
- [6. Softmax Has Three Coordinate Roles](chapter-06-softmax-coordinate-roles.html)

### Section III: Compile-Time Autodiff

- [7. What Is a Gradient?](chapter-07-gradients.html)
- [8. Matrix Multiplication Teaches the Pullback](chapter-08-matmul-pullback.html)
- [9. Local Derivatives, Global Shape](chapter-09-local-derivatives-global-shape.html)

### Section IV: Dependency Graphs and Storage

- [10. The Wall of Time Steps](chapter-10-time-steps.html)
- [11. Storage Follows Observation](chapter-11-storage-follows-observation.html)
- [12. An RNN Is a Dependency Graph](chapter-12-rnn-dependency-graph.html)

### Section V: Framework and Compiler Boundaries

- [13. If Dimensions Had Names Everywhere](chapter-13-named-dimensions-everywhere.html)
- [14. Attention as Named Communication](chapter-14-attention-named-communication.html)

### Section VI: Principles Under Stress

- [15. What the Notation Refuses to Hide](chapter-15-notation-refuses-to-hide.html)
- [16. Dynamic Routing and Low-Rank Communication](chapter-16-dynamic-routing-low-rank-communication.html)

### Appendix

- [Coordinate Diagnostics](appendix-coordinate-diagnostics.html)
- [Coordinate Reading Laws](appendix-coordinate-laws.html)

## A Useful Way to Read

For each example, pick one output coordinate and ask where it came from. Then
ask which coordinates were kept, which were summed away, which were omitted
from a term, and which were only present because of a boundary contract. That
small habit is often enough to reveal the bug the shape alone would miss.

Each chapter now ends with a small exercise. Treat those prompts as traps on
purpose: try the shape-compatible wrong version first, then use the coordinate
reading to explain why it is wrong.

The fastest study loop is:

```text
trace one cell
name the coordinates it reads
write the shape-compatible wrong line
state the law that rejects it
hide the mechanics behind a coordinate function only after the law is clear
```

That loop is intentionally repetitive. The repetition is the point: the same
few laws should explain reshape, broadcasting, reduction, autodiff, recurrence,
attention, and routing.
