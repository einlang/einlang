---
layout: book
title: "Coordinate Reading Laws"
---

# Coordinate Reading Laws

This appendix collects the small laws used throughout the book. They are not
new syntax. They are the reading habits that make the syntax useful.

Each law has the same shape: read one coordinate, name the fact, then try the
shape-compatible wrong version. If the wrong version is hard to explain, the
law is probably doing real work.

## 1. Role Is Not Extent

Two coordinates can have the same size and still mean different things.

```text
feature_count == time_count
```

does not make `feature` and `time` interchangeable. A checker that reports
"size 64" has found an extent. A useful tensor diagnostic should also be able
to name the role.

## 2. The Address Relation Comes Before Layout

A coordinate map states how output addresses relate to input addresses:

```text
y[b * group_count + group, feature * slice_count + slice] =
    x[b, feature, group, slice]
```

The implementation may lower this to a view, a copy, a permutation, or a fused
kernel. The source-level claim is the address relation.

## 3. Free Coordinates Form the Result Family

In an indexed binding, the coordinates that remain free on the left describe
the result family:

```text
let y[i, j] = sum[k](A[i, k] * B[k, j])
```

The result is addressed by `i` and `j`. The coordinate `k` is local work, not
part of the result address.

## 4. Omission Means Independence

If a term does not mention a result coordinate, the term does not depend on
that coordinate:

```text
let y[b, feature] = x[b, feature] + bias[feature]
```

The bias is reused across `b`. This is the coordinate reading of
broadcasting. The useful question is not "can this singleton expand?" but
"which coordinate is the value independent of?"

## 5. Reduction Consumes a Local Coordinate

A reducer introduces a coordinate for local work and removes it from the
result:

```text
let row_total[row] = sum[col](A[row, col])
```

The row survives. The column leaves. Selection-shaped reducers follow the same
rule: `argmax[class]` consumes `class`, but returns an address in the consumed
domain.

## 6. Coordinate Functions Hide Mechanics, Not Contracts

A coordinate function is useful only if the call still says which coordinate
choice matters:

```text
softmax[class](logits)
move_channel[channel](image)
scan[t](step, h0, x)
```

The body can hide scalar arithmetic, loop shape, temporary buffers, and kernel
details. It should not hide the normalized coordinate, moved coordinate, or
ordered coordinate. Packs such as `..spatial` let the caller name the role that
matters while surrounding coordinates are inferred from argument rank.

## 7. Gradients Are Addressed by the Denominator

The shape of a derivative answer is determined by the value being
differentiated with respect to:

```text
@loss / @W
```

has the coordinates of `W`. Other coordinates may appear while the derivative
is computed, but they must be reduced, preserved, or otherwise accounted for by
the chain rule. A gradient is not just a scalar formula; it is a scalar formula
placed into a coordinate address.

## 8. Time Is a Directed Coordinate

A recurrence should expose the dependency edge before it becomes a loop:

```text
let h[t in 1..T] = step(h[t - 1], x[t])
```

The edge points backward in time. A line that reads `h[t + 1]` may describe a
different computation, but it is not the same forward recurrence. Once the edge
is visible, a backend can still lower it to an ordinary loop.

## 9. Observation Determines Storage

Defining a family is not the same as materializing every member:

```text
let h[t in 1..T] = step(h[t - 1], x[t])
let final = h[T - 1]
```

The recurrence defines the family. The later observation tells which members
must be retained, printed, differentiated through, or recomputed. Storage is a
policy over dependency and observation facts.

## 10. Boundaries Need Receipts

When a value crosses a module, host-language, or library boundary, comments are
not enough. The boundary should preserve the facts later indexed code will use:
rank, element type, extents when known, and coordinate roles when roles affect
correctness.

```text
image[batch, row, col, channel]
logits[batch, class]
hidden[time, batch, unit]
```

These names are not decoration. They are receipts that later reductions,
maps, derivative requests, and coordinate functions can check.

## The Study Loop

For a new tensor expression, use the same loop every time:

```text
1. Pick one output cell.
2. Name every input coordinate it reads.
3. Mark which coordinates survive.
4. Mark which coordinates are local and disappear.
5. Mark which result coordinates are omitted by each term.
6. Write the shape-compatible wrong version.
7. State the law that rejects the wrong version.
8. Only then hide the mechanics behind a coordinate function.
```

If this feels repetitive, good. A small set of laws should do a lot of work.
That is the point of the notation.
