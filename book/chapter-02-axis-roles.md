---
layout: book
title: "Chapter 2: Axis Roles Are Not Axis Positions"
---

# Axis Roles Are Not Axis Positions

An axis position is a number. An axis role is the reason that number matters.

In this book, a role means the semantic responsibility an axis carries at a
particular point in a computation. A role can be "batch example," "query
position," "class," "previous hidden unit," or "feature coordinate being
summed." The same physical axis may carry different local roles in different
expressions, and two roles may have the same extent without becoming
interchangeable.

In a tensor of shape `[16, 64, 128]`, position `0` might be batch, position `1`
might be feature, and position `2` might be a packed spatial or head dimension.
But the shape alone does not say that. It gives the extents, not the story.

This distinction matters because positions are easy to preserve while roles are
easy to lose. A reshape can keep the total number of elements correct. A
transpose can keep the rank correct. A flatten can produce the expected final
size. None of those facts prove that batch stayed batch or that feature stayed
feature.

This is where anonymous axes do their quiet work. They do not usually cause a
crash. They let two different stories fit the same tuple of numbers. The work here is to separate the numeric slot from the model role. Once those
two things are apart, later chapters can ask better questions than "which axis
number was that?" Two axes can both have extent `32`; that does not make
batch, time, and feature interchangeable.

## Position or Name

A language designer has an immediate choice. The first option is to keep axis
positions as the only source-level identity. This matches existing array
libraries and keeps notation short, but it makes every later operation depend
on memory of what position `0`, `1`, or `2` meant at that point in the program.

The second option is to keep positions in the implementation but attach
comments or type annotations outside the expression. That improves
documentation, but the operation itself can still say `axis=1` while the
meaning lives somewhere else. The compiler receives the number; the reader
supplies the role.

Existing named-dimension tools explore this middle ground in useful ways.
PyTorch Named Tensors and xarray can attach names to dimensions and use those
names for alignment. That is already better than a naked shape tuple. The
question this book presses is slightly different: does the operation's source
semantics actually use the role, or is the role still mostly metadata attached
to an array? A name that aligns two arrays is helpful. A name that also appears
inside `sum[feature]`, `softmax_over[class]`, or
`hidden[t] <- hidden[t - 1]` gives the compiler a stronger fact: which role was
consumed, preserved, omitted, or shifted.

So the argument is not "names on dimensions are useless." It is that names
become much more powerful when operations are written in terms of them, rather
than merely carrying them beside positional operations.

The third option is to make the role part of the expression:

```text
image[b, channel, row, col]
```

This is the choice Einlang makes. It costs a few names, but it removes a
permanent ambiguity. Once a role is part of the expression, later code can ask
whether that role survived, moved, disappeared, or was used as a local
coordinate. Without the name, those questions arrive too late.

The danger is easiest to see with two equal-sized axes. Suppose `time` and
`feature` both have extent `128`. A transpose that swaps them may leave the
shape unchanged. A positional program can look undisturbed while the meaning
has moved. The role name is the local clue that the move was intended.

## A Small Naming Discipline

The smallest useful move is to write the role as a coordinate:

```text
image[b, channel, row, col] = ...
```

Now the program has four coordinates with four jobs. A later expression can
choose to preserve some, consume some, or map some into new coordinates:

```text
gray[b, row, col] =
    sum[channel](weights[channel] * image[b, channel, row, col])
```

The coordinate `channel` is local to the sum and disappears. The coordinates
`b`, `row`, and `col` survive. The result is more specific than "rank 3"; it is a
family over batch and spatial position.

Read one point:

```text
gray[2, 17, 9]
```

This point is the weighted sum of channels at one batch item and one pixel. The
coordinate names make that sentence possible without looking up an axis table.

The same line also shows the design boundary. The reducer does not need a
global declaration that "axis 1 is channel." It introduces `channel` exactly
where the role matters and removes it exactly where the result no longer has a
channel coordinate. The formula is doing the documentation and the check at
the same time.

## When Positions Drift

A common bug begins innocently:

```python
x = x.transpose(1, 2)
y = layer(x)
```

If `layer` expects `[batch, feature, time]`, the transpose may have changed
more than memory layout. It may have changed the meaning of position `1`.

Named coordinates make that kind of drift visible:

```rust
let x2[b, time, feature] = x[b, feature, time];
let y[b, time, out] = layer(x2[b, time, feature]);
```

The second line now has to confront the actual roles. If the layer expects
`feature` before `time`, the mismatch is a local fact in the formula rather
than a later shape surprise.

## Role Names as Local Documentation

Good tensor code often has comments that look like this:

```python
# x: [batch, time, feature]
```

The comment is useful because the code needs more than a shape. It needs a
contract for each position. The problem is that the comment is not part of the
expression language. It may be correct when written and stale after a refactor.
It may describe the input to a function but not the intermediate value three
lines later.

An indexed binding brings that contract closer to the operation:

```text
projected[b, time, out] =
    sum[feature](x[b, time, feature] * W[out, feature])
```

Now the role names are doing work. `feature` is the coordinate consumed by the
projection. `out` is the coordinate introduced by the weight matrix and
surviving in the result. `b` and `time` are carried along. The binding is a
small piece of documentation, but it is also the formula.

Read one concrete point:

```text
projected[3, 12, 7]
```

This means "for batch item 3 and time step 12, compute output feature 7 by
summing over input features." That sentence is not only a paraphrase. It is
the coordinate structure of the expression.

## Same Shape, Different Program

Two tensors can share a shape and still be different kinds of objects:

```text
scores[b, class]
tokens[b, time]
```

Both might have shape `[32, 128]`. But `class` and `time` are not
interchangeable. Reducing over `class` computes something like a per-example
normalization or loss. Reducing over `time` computes something like a sequence
summary. A generic shape checker sees two axes of length `128`; the source
roles say what the axes mean.

This is where named coordinates become more than decoration. They let code
review focus on meaning:

```rust
let loss[b] = -sum[class](target[b, class] * log_probs[b, class]);
```

If someone accidentally writes `sum[b]`, the formula is visibly wrong. If they
write `axis=0`, the reader has to remember what axis `0` meant at that exact
point in the pipeline.

The discipline is modest: name the coordinate where the operation depends on
the role. Do not name everything for ceremony. Name the dimensions whose
meaning would otherwise be carried by memory.

## What Survives the Pool?

Consider the quiet promise made by this expression:

```rust
let pooled[b, channel] = max[row, col](image[b, channel, row, col]);
```

The roles that survive are more than positions in the result. They are the
parts of the original object still available for later reasoning. The consumed
roles are not only removed axes; they are facts that mattered locally and
then disappeared. Positions can tell you where an axis sits. Roles tell you why
it is there.

## Naming Without Over-Naming

There is a real danger here: if every tiny expression gets covered in long
coordinate names, the notation becomes heavy. The goal is not ceremonial
naming. The goal is to name the roles that determine whether the program is
correct.

A temporary scalar does not need a grand title. A local helper inside a small
formula may not need more than `i` or `j`. But a dimension that distinguishes
batch from time, feature from class, or head from within-head feature deserves
a name because confusing it changes the program.

A useful rule is:

```text
Name the coordinate when the role would otherwise live in a comment.
```

If the code needs a comment saying `[batch, time, feature]`, the roles are
important enough to appear in the notation. If a transformation depends on
which axis is time and which axis is feature, use names. If a reduction must
remove class but preserve batch, use names. If a shape is incidental and the
meaning is obvious from a local expression, keep the notation light.

This balance matters for a production language. Explicitness is valuable only
when it buys clarity. The goal is not maximal verbosity. The goal is to move
semantic roles out of the reader's memory and into the lines where those roles
are used.

A visible dimension is therefore more than a longer axis label. It is a role
that participates in a contract. Once that contract is written down, every
later operation can be read by asking which roles survive, which roles change,
and which roles disappear.

## Same Extent, Different Role

The implemented compiler does not need to believe that `batch` is a special
primitive type in order to benefit from the role. It only needs scoped names
and consistent uses. If `b` is introduced on the left side of:

```rust
let centered[b, t, f] = x[b, t, f] - mean[b, t];
```

then every occurrence of `b` inside the body refers to that coordinate. If a
term uses `mean[t, b]` instead, the address relation has changed. Depending on
the extents, shape analysis may catch the mismatch. Even when extents happen
to agree, the source now gives reviewers a local clue that the role order is
suspicious.

This is the modest power of axis roles. They do not prove the whole model
correct, but they make a class of wrong-but-shape-compatible programs visible.
They let each line state the role alignment it claims. Once those claims are
in the program, tests can focus on numerical behavior instead of trying to
infer which axis the author meant.

That shift changes code review. Instead of asking a reviewer to remember that
position `1` means time in this file but feature in another file, the line
itself carries the distinction. The reviewer can spend attention on the actual
claim: should this operation preserve time, consume it, or move it into a new
coordinate? That is a better use of human memory.

## Role Audit Under Pressure

Consider a batch of token embeddings:

```text
x[b, t, f] = ...
```

Here `b` is the example, `t` is the token position, and `f` is the feature
coordinate. A positional library may store the same value with shape
`[32, 128, 768]`. That shape is useful, but it does not say which axis can be
mixed without changing the meaning of the model.

A layer-normalization-style operation over features should keep `b` and `t`
fixed while reducing over `f`:

```rust
let mean[b, t] = sum[f](x[b, t, f]) / feature_count;
let centered[b, t, f] = x[b, t, f] - mean[b, t];
```

The role audit is immediate. The mean is addressed by `[b, t]`, so it computes
one value per example and token. The feature coordinate is local to the
reduction and does not survive into `mean`. The centered tensor restores `f`
because each feature value is shifted by the mean for its own token.

Now compare a batch normalization over examples:

```rust
let mean[t, f] = sum[b](x[b, t, f]) / batch_count;
let centered[b, t, f] = x[b, t, f] - mean[t, f];
```

This program has the same input and output shape as the previous one. Both
produce `centered[b, t, f]`. A shape-only review can easily miss the semantic
difference because the final tensor still has rank three and the same extents.
The coordinate names expose the difference: one program consumes `f`, the
other consumes `b`.

That is why roles cannot be reduced to positions. In a model where `batch` and
`time` both have length `32`, a mistaken normalization may even use an axis of
the right size. The error is not "dimension 0 has length 32 but expected 128."
The error is "the program treated examples as the distribution instead of
features." That message requires role information.

The same audit works for pooling. A temporal average over a sequence is:

```rust
let pooled[b, f] = sum[t](x[b, t, f]) / time_count;
```

The result keeps examples and features, but consumes time. A feature average
would be:

```rust
let pooled[b, t] = sum[f](x[b, t, f]) / feature_count;
```

Both are legal reductions. Neither is intrinsically more correct without the
model contract. The point is that the source should say which contract it is
using.

This is the concrete habit behind role names: read every operation as a small
audit.

```text
Which coordinates survive?
Which coordinates are consumed?
Which coordinate is being reused because it is absent from a term?
```

If the answer matters to correctness, the name is carrying real work. It is
not there for style. It gives the compiler and the reader a shared object to
reason about before the program collapses into anonymous positions.

A tensor shaped `[32, 32, 128]` is a useful last example: give the two `32`
axes different roles, then imagine an operation that remains legal after they
are swapped. The diagnostic you would want is not "size 32". It is the name of
the role that crossed the line.

## Try It

Write two formulas over a tensor shaped `[32, 32, 128]`: one where the first
`32` is batch and one where it is time. Then reduce over the wrong `32` axis on
purpose. The trap is that every extent still agrees. The lesson is to name the
role whose loss would make the result meaningless.

**Line to keep:** a role is not where an axis sits, but why it is there.
