---
layout: book
title: "Chapter 3: Coordinate Maps in the Standard Library"
---

# Coordinate Maps in the Standard Library

The standard library is where a notation has to earn its keep. It is easy to
look good on a two-line example. It is harder to stay readable when the address
map has real work to do.

Two examples from `stdlib/ml/transform_ops.ein` show the range: transpose is
simple enough to read at a glance; `depth_to_space` is dense enough that names
really matter.

The implementation path is concrete. Coordinate names are parsed into scoped
symbols and carried through IR. Range analysis and shape analysis then reason
about the same names the reader sees. A map such as
`result[..batch, i, j] = x[..batch, j, i]` is not decoration before the "real"
compiler starts; it is the thing later passes carry until a backend chooses a
view, copy, or fused access.

At this point the question shifts from "what are the axes called?" to "where
did each coordinate travel?" A transpose, flatten, or layout operation may be
implemented as storage arithmetic, but the source first has to say the map that
arithmetic is supposed to preserve.

The figures use a simple convention. A chip names a coordinate role before it
names a size. Green chips are carried unchanged, blue chips move, and amber
chips are packed or unpacked. The point is not to decorate the formula; it is
to make the same source fact visible to the eye that the indexed expression
makes visible to the compiler.

For coordinate maps, the operation name is not enough. Transpose, flatten, and
depth-to-space become reviewable only when the address relation is visible,
especially when the next operation relies on adjacency.

Before reading the examples, try to write a transpose without using the word
`transpose`. Write only the source address for one result address. If that
feels more precise than the operation name, you have found the chapter's main
point.

## Operation Name or Address Relation

Shape libraries usually present layout changes as operations: reshape,
transpose, flatten, permute. That is a natural interface for execution. The
small trap is that the operation name can sound more informative than it is.
It says what family of move was requested; it may not say which output
coordinate reads which input coordinate.

The alternative is to state a coordinate relation. Instead of making the
operation primary, the source states the address equation that every
implementation must preserve. A backend can still choose a view, a copy, a
strided access pattern, or a fused kernel. The design move is to separate the
semantic map from the storage strategy.

That separation is why coordinate maps belong early in the language. If a
reshape-like construct hides its map, later phases have to reconstruct meaning
from a sequence of operations. If the map is written directly, optimization can
happen after the source has already said what must stay true.

Erase the operator names for a moment and keep only a few coordinate examples:

```text
out[0, 0] reads in[0, 0]
out[0, 1] reads in[1, 0]
out[2, 5] reads in[5, 2]
```

A reader can recognize transpose from those witnesses. The reverse is not
always true: hearing the word "reshape" or "permute" may not tell the reader
which address relation is intended. The source relation is the stable object;
the operation name is a convenient surface.

This is close in spirit to `einops.rearrange`, where a string such as
`"b c h w -> b h w c"` makes a layout relation more visible than a sequence of
axis numbers. Einlang makes a different placement choice. The map is not a
string argument interpreted by a library call; it is source syntax that passes
through name resolution, range analysis, shape analysis, and lowering. That
lets later compiler passes reason about the same coordinate names the reader
sees, instead of treating the rearrangement pattern as opaque text.

## Transpose Carries a Prefix

The transpose operator is:

```rust
let result[..batch, i, j] = x[..batch, j, i];
```

The reusable form should preserve the same contract. A coordinate function
could expose the moving roles directly:

```rust
fn swap[j, k](x: [f32; ..left, j, ..middle, k, ..right])
    -> [f32; ..left, k, ..middle, j, ..right]
{
    let y[..left, k, ..middle, j, ..right] =
        x[..left, j, ..middle, k, ..right];
    y
}

let result = swap[j, i](x);
```

The call is short because the map is conventional, not because the coordinate
story disappeared. The implementation may lower to a view, but the function
boundary still says which two coordinates trade places and which surrounding
packs are carried. The function does not assume rank. It only requires that
`j` occur before `k` in the argument layout for this definition; a companion
definition can cover the reverse order if a library wants both directions.

<figure class="axis-map-figure">
  <p class="axis-map-title">Named Axis Snapshot: Transpose</p>
  <div class="axis-map-flow">
    <div class="axis-snapshot">
      <p class="axis-snapshot-label">Input `x`</p>
      <ul class="axis-list">
        <li class="axis-chip axis-chip-fixed">..batch</li>
        <li class="axis-chip axis-chip-moved">j</li>
        <li class="axis-chip axis-chip-moved">i</li>
      </ul>
    </div>
    <div class="axis-map-arrow">-></div>
    <div class="axis-snapshot">
      <p class="axis-snapshot-label">Result `result`</p>
      <ul class="axis-list">
        <li class="axis-chip axis-chip-fixed">..batch</li>
        <li class="axis-chip axis-chip-moved">i</li>
        <li class="axis-chip axis-chip-moved">j</li>
      </ul>
    </div>
  </div>
  <div class="axis-map-rules">
    <span>preserve: ..batch</span>
    <span>swap: result[..., i, j] reads x[..., j, i]</span>
  </div>
  <figcaption>The prefix is carried unchanged; only the two trailing coordinates trade places.</figcaption>
</figure>

The line says three things at once. First, the result is a new binding. Second,
the batch-shaped prefix survives unchanged. Third, the last two coordinates
swap roles.

This is stronger than saying "transpose the last two axes." It also says what
does not move. If `x` has shape `[heads, batch, rows, cols]`, the prefix
`..batch` may contain `heads` and `batch`; both travel together. The final two
coordinates are the only part of the map being inverted.

Read one point:

```text
result[heads, batch, 2, 5] = x[heads, batch, 5, 2]
```

The prefix is not inspected or rearranged. It simply comes along. A backend may
implement the transpose as a view, a copy, or a fused access pattern, but the
address relation is the same.

This is the pattern to keep: a coordinate map does not first say how memory
moves. It says how one address in the result corresponds to an address in the
input. Memory layout is a later question.

## What `..batch` Really Buys

The rest prefix in the transpose example is not a decorative shorthand:

```rust
let result[..batch, i, j] = x[..batch, j, i];
```

It is a way to state a rank-polymorphic coordinate map. The operation should
transpose the final two coordinates while carrying every earlier coordinate
unchanged. If `x` is `[rows, cols]`, the prefix is empty. If `x` is `[b, rows,
cols]`, the prefix contains one coordinate. If `x` is `[heads, b, rows,
cols]`, the prefix contains two.

The compiler cannot leave that prefix as vague prose forever. Early in the IR
pipeline, rest-pattern preprocessing tries to expand the prefix when rank
information is available. Conceptually:

```text
..batch  over rank-4 input with two trailing explicit indices
    becomes batch.0, batch.1

result[batch.0, batch.1, i, j] =
    x[batch.0, batch.1, j, i]
```

That expansion matters because later passes want flat index lists. Range
analysis should reason about `batch.0` and `batch.1` like ordinary coordinates.
Shape analysis should compute the output shape from explicit dimensions, not
from an opaque prefix object. Einstein lowering should produce loops or
vectorized execution over concrete coordinate slots.

The pressure test is a generic coordinate function. A transpose implementation
should not need a separate definition for rank two, rank three, and rank four.
But a backend also cannot execute "some prefix" without knowing how many
coordinates it covers. Rest-pattern preprocessing is the bridge: it lets the
source say "preserve the whole leading context" while giving later compiler
passes the explicit coordinates they need.

This is the first appearance of a pattern the book will reuse. Human-facing
coordinate functions can compress a family of coordinate relations. The
compiler-facing IR eventually needs the family made explicit enough for range,
shape, and lowering to act on it.

That is also why coordinate packs belong in function signatures. A call such
as `swap[j, i](x)` names only the two roles being moved. The `..left`,
`..middle`, and `..right` packs are recovered from `x` and then stamped onto the
call as compiler facts. The caller does not count axes; the compiler still gets
the full explicit list before lowering.

## Depth to Space Is a Coordinate Unpacking

Now compare:

```rust
let result[b, c, i, j] = input[
    b,
    c * (r * r) + (i % r) * r + (j % r),
    i / r,
    j / r
];
```

<figure class="axis-map-figure">
  <p class="axis-map-title">Named Axis Snapshot: Depth to Space</p>
  <div class="axis-map-flow">
    <div class="axis-snapshot">
      <p class="axis-snapshot-label">Input `input`</p>
      <ul class="axis-list">
        <li class="axis-chip axis-chip-fixed">b</li>
        <li class="axis-chip axis-chip-packed">c * r * r + block</li>
        <li class="axis-chip axis-chip-moved">i / r</li>
        <li class="axis-chip axis-chip-moved">j / r</li>
      </ul>
    </div>
    <div class="axis-map-arrow">-></div>
    <div class="axis-snapshot">
      <p class="axis-snapshot-label">Result `result`</p>
      <ul class="axis-list">
        <li class="axis-chip axis-chip-fixed">b</li>
        <li class="axis-chip axis-chip-packed">c</li>
        <li class="axis-chip axis-chip-moved">i</li>
        <li class="axis-chip axis-chip-moved">j</li>
      </ul>
    </div>
  </div>
  <div class="axis-map-rules">
    <span>block = (i % r) * r + (j % r)</span>
    <span>old depth = c * (r * r) + block</span>
    <span>old spatial = (i / r, j / r)</span>
  </div>
  <figcaption>The new spatial coordinates split into quotients and remainders; the remainders are packed back into the old depth coordinate.</figcaption>
</figure>

The operation expands spatial resolution by moving information out of depth.
The formula explains how. The old spatial coordinates are the quotients:

```text
i / r
j / r
```

The offsets inside the new spatial block are the remainders:

```text
i % r
j % r
```

Those remainders are then packed into the old depth coordinate:

```text
c * (r * r) + (i % r) * r + (j % r)
```

This is exactly the sort of operation that can become opaque as a reshape
chain. Written as a coordinate map, each subexpression has a job. A reader can
test one point by hand. A compiler can preserve the relationship until a later
lowering pass decides how to implement it.

The point test is the useful discipline. Pick `r = 2`, `b = 0`, `c = 3`,
`i = 5`, and `j = 7`. The old spatial address is `(2, 3)` and the depth offset
inside the block is `3`, so the read is:

```text
input[0, 3 * 4 + 3, 2, 3]
```

That local calculation is what a reshape chain often hides. The formula has
enough structure that a reviewer can check a single output address without
simulating the whole tensor.

The formula also carries obligations. The ranges of `i` and `j` must make
`i / r`, `j / r`, `i % r`, and `j % r` meaningful for the intended output
shape. The old depth must have enough room for `r * r` offsets inside each
new channel. A coordinate map is therefore not only an address equation; it is
an address equation with domain facts. Those facts are exactly the kind of
thing a compiler can check before choosing a storage strategy.

## Flattening as the Smallest Coordinate Map

Before `depth_to_space`, there is a simpler map:

```rust
let flat[p] = x[p / width, p % width];
```

Here `p` is a one-dimensional coordinate. It is unpacked into a row and a
column. The quotient gives the row. The remainder gives the column. This is the
same arithmetic pattern used in larger shape transformations, just with fewer
roles.

The inverse map is:

```rust
let x[row, col] = flat[row * width + col];
```

<figure class="axis-map-figure">
  <p class="axis-map-title">Named Axis Snapshot: Flatten and Unflatten</p>
  <div class="axis-map-flow">
    <div class="axis-snapshot">
      <p class="axis-snapshot-label">Two-dimensional view</p>
      <ul class="axis-list">
        <li class="axis-chip axis-chip-moved">row</li>
        <li class="axis-chip axis-chip-moved">col</li>
      </ul>
    </div>
    <div class="axis-map-arrow">-></div>
    <div class="axis-snapshot">
      <p class="axis-snapshot-label">Flat view</p>
      <ul class="axis-list">
        <li class="axis-chip axis-chip-packed">p = row * width + col</li>
      </ul>
    </div>
  </div>
  <div class="axis-map-rules">
    <span>unpack: row = p / width</span>
    <span>unpack: col = p % width</span>
    <span>pack: p = row * width + col</span>
  </div>
  <figcaption>The same size can preserve or destroy intent depending on whether the packed coordinate keeps its row and column story visible.</figcaption>
</figure>

Now row and column are packed into one position. Again, the formula is not
about copying memory first; it explains which old coordinate
corresponds to which new coordinate.

This matters because flattening is often treated as harmless. It can be
harmless when the next operation truly wants a feature vector. It can also be
the moment where spatial roles are thrown away too early. The named form makes
the loss explicit:

```text
row, col -> p
```

After that map, a later expression that only sees `p` cannot distinguish
"nearby row" from "nearby column" unless some other structure restores the
roles. The coordinate map is honest about the forgetting.

## A Reader's Checklist

For any coordinate map, ask four questions:

- Which coordinates appear in the result?
- Which input coordinates are read?
- Is any coordinate packed into arithmetic?
- Is any coordinate unpacked by division or remainder?

These questions work for transpose, flatten, `depth_to_space`, and many
layout-sensitive model operations. The details change, but the reading habit
stays stable.

## Can the Map Run Backward?

If `r = 2`, what input coordinate supplies `result[b, c, 5, 7]`?

The old spatial position is `(2, 3)`. The within-block offset is `(1, 1)`. The
old depth coordinate is `c * 4 + 3`. That is the whole operation in one
coordinate reading.

The larger question is what kind of operation this is. Nothing has been
summed, averaged, or discarded. The value has moved through a coordinate map.
That distinction will matter later, because a coordinate map can preserve
information even when it changes the apparent layout completely.

One final way to test a coordinate map is to ask whether it is reversible. A
transpose is reversible: swapping `i` and `j` twice returns to the original
coordinates. A flatten can be reversible if the original width is known:

```text
p -> (p / width, p % width)
```

`depth_to_space` is also meant to have an inverse shape transformation. That
does not mean every implementation must literally run the inverse. It means
the coordinate story is strong enough that the inverse can be stated. When a
map cannot be reversed, the notation should make the loss visible. A reduction,
for example, is not a coordinate map of this kind; it consumes information.

This distinction prepares the next section. Some operations move coordinates
around. Some operations omit coordinates. Some consume coordinates. The source
should let the reader tell which kind of event occurred.

## Maps before Layout

The word "map" is doing important work. A coordinate map says how to translate
an output address into an input address. It does not say whether the runtime
should allocate new memory, create a view, change strides, or fuse the access
into another operation.

That separation is a recurring design principle:

```text
source relation first
storage strategy later
```

For transpose, a backend may be able to return a view. For `depth_to_space`, it
may choose a copy or a specialized kernel. For flattening, it may do nothing if
the layout is already contiguous. The source formula does not need to commit to
those decisions. It needs to state the relationship that all valid
implementations must preserve.

This is why coordinate maps are a better foundation than a sequence of
imperative layout steps. A sequence says how one implementation proceeds. A map
says what must be true at every coordinate. Optimization can then happen
without erasing meaning.

The standard-library examples give the idea its first pressure test. Visible
dimensions are not limited to algebra written on a clean page. They can read
real layout operators, including ones whose correctness depends on quotient and
remainder arithmetic. A coordinate map may become a view, a copy, or a fused
access pattern, but the source relation remains the same: each output
coordinate must name the input coordinate it observes.

## Reversible Does Not Mean Obvious

Use a compact input so the whole map can be inspected, but keep the real issue
in view: two maps can be legal and reversible while preserving different
neighborhoods. Suppose `input[c, i, j]` has one batch item omitted for
simplicity, two channels, two rows, and two columns:

```text
input[0, 0, 0] = a
input[0, 0, 1] = b
input[0, 1, 0] = c
input[0, 1, 1] = d
input[1, 0, 0] = e
input[1, 0, 1] = f
input[1, 1, 0] = g
input[1, 1, 1] = h
```

A flattening map with `width = 2` is:

```rust
let flat[c, p] = input[c, p / 2, p % 2];
```

The four positions in `p` read:

```text
flat[c, 0] = input[c, 0, 0]
flat[c, 1] = input[c, 0, 1]
flat[c, 2] = input[c, 1, 0]
flat[c, 3] = input[c, 1, 1]
```

No values are combined. No values are duplicated. The map only changes the
address language. That is why it is reversible if the width is known:

```rust
let restored[c, row, col] = flat[c, row * 2 + col];
```

For `restored[1, 1, 0]`, the packed coordinate is `1 * 2 + 0 = 2`, so the read
is `flat[1, 2]`, which was `input[1, 1, 0]`. The proof of the inverse is a
coordinate proof, not a storage proof. A backend might implement both maps
without copying, or it might allocate a new array, but the semantic relation is
the same.

Now compare a mistaken map:

```rust
let bad[c, p] = input[c, p % 2, p / 2];
```

This also produces shape `[2, 4]`. It also reads only legal addresses. But it
changes the traversal order:

```text
bad[c, 1] = input[c, 1, 0]
bad[c, 2] = input[c, 0, 1]
```

If the next layer treats adjacent `p` positions as row-major neighbors, this
mistake matters. The shape checker alone has no reason to reject it. The
coordinate relation, however, makes the swap visible. A reviewer can see that
division should recover the row and remainder should recover the column for a
row-major flatten.

The example also shows what the compiler can check and what it cannot infer
from shapes alone. It can infer that `p` must range from `0` to `height *
width`. It can check that `p / width` and `p % width` stay within bounds when
the ranges are known. It can preserve the expression through lowering. But it
cannot know that row-major order was intended unless the source states the
map.

Coordinate maps therefore have a real difficulty gradient. First check that
the address arithmetic is legal. Then check whether the inverse exists. Then
ask the harder question: does the map preserve the adjacency that the next
operation assumes? A convolution, a patch extractor, and a tokenization step
may all accept the same flattened shape while relying on different adjacency
stories. The notation does not guess intent; it gives intent a place to be
written.

Choose one of these maps and run it backward. Part of the inverse is arithmetic;
another part is an assumption about layout or adjacency that the next operation
quietly relies on. That boundary is where the interesting mistakes live.

## Try It

Rewrite one `einops.rearrange` pattern as an Einlang coordinate map. Then ask
what changes when the map is source syntax instead of a string argument. Which
parts can name resolution, range analysis, and shape analysis inspect before a
backend chooses a layout?

**Line to keep:** the address relation is the truth; the operation name is its
surface expression.
