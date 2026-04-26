---
layout: book
title: "Chapter 1: What Can the Compiler Not See?"
---

# What Can the Compiler Not See?

Start with a familiar fragment:

```python
x = torch.randn(16, 64, 128)
y = x.reshape(16, 64, 4, 32).transpose(1, 2).reshape(16 * 4, 64 * 32)
```

Nothing here is strange. That is exactly why it is a good first example. A
dimension of size `128` is split into `4` and `32`. One axis is moved. Two
pairs of axes are flattened. A working engineer can read the code and
reconstruct the intention.

Now ask a colder question. Which dimension is the batch? Which one is the
feature axis? Which part of the split dimension is a group, and which part is a
slice inside that group?

The code has the arithmetic, but the roles live mostly in the reader's memory.
The runtime sees shapes. The compiler sees a sequence of shape-compatible
operations. The program itself does not say why axis `1` mattered before the
transpose, or what semantic role the `4` and the `32` are supposed to play.

That is the first blind spot. Tensor code often preserves enough information
to run, but not enough information to explain itself.

This is not only a toy failure mode. Imagine an image model that packs
`batch * head` into one coordinate before a projection layer. During a refactor,
the packing order changes, but the unpacking code downstream still assumes the
old story. The tensors keep the same rank and extent. Training continues.
Only later, when an exported model is compared against a reference service, do
the probabilities drift because examples have been grouped under the wrong
attention head. The bug did not need a crash to be serious. It only needed two
coordinate stories that shared the same shape.

That first act of disappearance is worth isolating. A shape edit can be
legal while the coordinate roles that made it meaningful have already slipped
out of the source. In this book, a tensor line is not only an operation over
sizes. It is, or should be, a claim about where named coordinates went.

So the reshape chain is not suspicious because it is complicated. It is
suspicious because it proves element counts while leaving `b`, `feat`,
`group`, and `slice` to be reconstructed by the reader.

## Shape Trace or Coordinate Claim

One design path is to accept shape traces as the record that matters. A program
can record that a tensor was reshaped, transposed, and reshaped again. That is
enough to execute the operation, but it leaves the compiler with a history of
shape edits rather than a statement of intent.

Another path is to require every reshape-like operation to carry a coordinate
claim: which named input coordinates become which output coordinates. The
history of operations may still exist, but the source also contains the
relationship those operations are meant to preserve.

Einlang follows the second path. It treats a shape transformation as a claim
about coordinates before it treats it as a storage maneuver. The rule appears
here for the first time: do not hide a fact that later reasoning must recover.

Give a compiler only the three shapes in the opening example and ask it
whether the transformation was correct. It can check element counts. It can
check that each reshape is legal. It can even replay the axis permutation. But
it cannot know whether `4` meant groups, heads, patches, or something else.
The missing fact is not a number. It is a role.

## The Same Move With Named Coordinates

Write the same intention as a coordinate relation:

```text
y[b * group, feat * slice] = x[b, feat, group, slice]
```

This line is not a new reshape API. It is a different kind of statement: the
output coordinates are built from named input coordinates.

The mental move is slower than the line makes it look. First imagine one input
cell, `x[b, feat, group, slice]`. The refactor wants `group` to travel with
`b`, and `slice` to travel with `feat`. Only after that sentence is clear does
the address arithmetic appear:

```text
first output coordinate   b * group
second output coordinate  feat * slice
```

The equation is the final form of that sentence. It is not meant to impress
the reader with clever packing arithmetic; it is meant to keep the pairing
decision visible.

The ranges would make the relation concrete:

```text
b     in 0..16
feat  in 0..64
group in 0..4
slice in 0..32
```

The important part happens before storage layout enters the discussion. The
program says that `b` and `group` combine into one output coordinate, while
`feat` and `slice` combine into the other. If the implementation later lowers
this to a view, a copy, or a fused kernel, that is a backend decision. The
source has already stated the coordinate map.

Read one concrete point:

```text
y[3 * 2, 17 * 9] = x[3, 17, 2, 9]
```

The arithmetic is not the lesson. The lesson is that the names remain attached
to the operation. You can point to `group` and ask where it went. You can point
to `feat` and ask whether it was packed with the right partner. The code is no
longer only a trail of axis numbers.

The ambiguity becomes sharper when two layouts have the same extent. If the
intended layout was:

```text
y[b * group, feat * slice] = x[b, feat, group, slice]
```

then the competing layout:

```text
y[group * b, feat * slice] = x[b, feat, group, slice]
```

may still have the same result shape. The difference is visible only because
the equation exposes which coordinate is slow and which coordinate is fast in
the packed address. Shape compatibility does not settle that question.

## What the Binding Adds

An indexed `let` adds a family of values to the program environment. In:

```text
y[b * group, feat * slice] = x[b, feat, group, slice]
```

the binding is more than `y`. It is the whole family of `y` coordinates
described by the left-hand side. The right-hand side explains how each member
of that family is read from `x`.

This is the basic reading discipline:

- a binding gives a value a stable name;
- free coordinates describe the shape of a family;
- coordinate expressions describe how output positions relate to input
  positions;
- later compiler passes may choose an evaluation strategy without erasing the
  source-level relation.

If this sounds too formal, use the simpler test: every important axis should
have a role that can be checked at the line where it moves. If `group` is
packed with the wrong partner, the relation should look wrong locally.

## A Bug That Still Has the Right Shape

The most dangerous version of the reshape bug is not the one that crashes. The
dangerous version keeps running.

Suppose a model expects a packed coordinate to mean:

```text
packed = b * group
```

but a later edit silently changes the intended packing to:

```text
packed = group * b
```

The product may still have the same size. The resulting tensor may still flow
through the next layer. If the downstream operation only checks rank and
extent, nothing necessarily fails at the boundary. The values are simply being
read under the wrong story.

Named coordinates do not solve every such bug automatically, but they change
where the bug has to live. Instead of hiding in a chain of reshapes, the
packing relation is written as a relation among names. A reviewer can ask:

```text
Should batch be the slow coordinate or the fast coordinate here?
Should group be packed with batch at all?
Should feature be paired with slice, or should slice be spatial?
```

Those are semantic questions. A shape tuple cannot answer them. A coordinate
equation at least gives them a place to attach.

That is why the chapter begins with compiler blindness rather than syntax. The
syntax matters only when it turns an unwritten assumption into something the
reader can inspect. The moment a reader can say "wait, why is `group` packed
with `b`?" the notation has already done useful work.

## Where This Leads

The rest of the first section follows this distinction from a single notation
into real operators. Chapter 2 separates axis roles from axis positions.
Chapter 3 reads standard-library coordinate maps such as transpose and
`depth_to_space`.

For now, the only point is the first one: a shape can be correct while the
program still withholds the reason it is correct. Named coordinates are the
way visible-index notation asks the program to stop withholding that reason.

## When Shape Is Not Meaning

Before storage enters the discussion, this line already raises a larger
question:

```text
y[b * group, feat * slice] = x[b, feat, group, slice]
```

Which named coordinates are packed into the first output coordinate? Which are
packed into the second? The answer is not notation trivia; it is the semantic
content that a shape-only program leaves behind. Once that content is visible,
the reader can begin to ask whether the transformation says what the model
intended.

The lesson is not that shape operations are bad. They are useful and often
efficient. The point is narrower: a shape operation is not the same as a
semantic operation. When dimensions have names, a tensor program can say more
of what it means.

## The First Habit

Broadcasting, reduction, gradients, recurrence, and attention all become
easier to read once axes stop being anonymous slots. The first habit is
therefore not a feature of a language, but a way of reading:

```text
Do not ask only whether the shape is valid.
Ask what relationship among named coordinates the line states.
```

The habit is deliberately modest. It does not require a full language or a new
backend. It only requires enough notation to make a dimension role visible.
Once the role is visible, more specific questions become possible: was the
role preserved, consumed, broadcast, differentiated, or used as a time
dependency?

This also explains why the chapter starts with a reshape chain rather than a
mathematical formula. Reshape-heavy code is where many engineers first feel the
gap between "the code works" and "the code says what I meant." The gap is not
ignorance. It is a limitation of the source representation. Visible dimensions
are an attempt to narrow that gap.

## A Shape-Correct Mistake

Start with a tensor whose intended roles are easy to name:

```text
x[b, feat, group, slice]
```

Let `b` range over two examples, `feat` over three features, `group` over two
groups, and `slice` over four positions inside each group. The positional
shape is therefore `[2, 3, 2, 4]`. If the program flattens the first and third
coordinates together and the second and fourth coordinates together, a visible
coordinate relation can say:

```text
y[b * group_count + group, feat * slice_count + slice] =
    x[b, feat, group, slice]
```

Now read one cell. Suppose `group_count = 2` and `slice_count = 4`. The output
cell `y[3, 9]` unpacks as:

```text
3 = 1 * 2 + 1      so b = 1, group = 1
9 = 2 * 4 + 1      so feat = 2, slice = 1
```

The source relation says that `y[3, 9]` observes `x[1, 2, 1, 1]`. A reviewer
can check the intended role of every term: `b` is packed with `group`, and
`feat` is packed with `slice`.

Now write the shape-only version:

```python
y = x.reshape(2, 3, 2, 4).transpose(0, 2, 1, 3).reshape(2 * 2, 3 * 4)
```

This is a reasonable implementation path, but the meaning is spread across
three operations. The compiler can check that the element count is preserved.
It can check that the transpose order is a permutation. It can produce the
final shape `[4, 12]`. What it cannot see, from the final shape alone, is that
the first output coordinate is "batch packed with group" rather than "feature
packed with group" or "batch packed with slice."

The real pressure appears when two roles have compatible extents. A wrong
version can still look plausible:

```text
bad[b * slice_count + slice, feat * group_count + group] =
    x[b, feat, group, slice]
```

If the role extents differ, some later size check may catch the error. If the
roles share extents, or if later code only expects a flat feature vector of the
same total size, the mistake can survive. The map is wrong not because the
arithmetic is illegal. It is wrong because it preserves the wrong semantic
neighborhoods.

The visible form gives a local place for the mistake to appear. A reader can
ask, "Should `slice` travel with `b`?" That is the semantic question hidden
inside many production reshape bugs. It is much harder to ask from
`reshape(...).transpose(...).reshape(...)` without
reconstructing the whole coordinate map by hand.

The first lesson is mechanical. Shape tells us how many addresses exist. A
coordinate relation tells us how those addresses were constructed. The
compiler cannot preserve a role that never enters the source. Once the program
states the relation, later phases have something precise to check and lower.

## What the Compiler Receives

The difference can be stated as a small contract. In the shape-only version,
the compiler receives operations over extents:

```text
[2, 3, 2, 4] -> [2, 2, 3, 4] -> [4, 12]
```

In the coordinate version, it receives a relation between addresses:

```text
output row    b * group_count + group
output col    feat * slice_count + slice
input cell    x[b, feat, group, slice]
```

The second form says which old roles were combined into which new roles. That
fact can feed range checks, shape checks, lowering decisions, and later
explanations. If an error message can point to `group` being packed with the
wrong coordinate, the source has already done useful work.

The rest of the book keeps applying this test. A dimension name is worth
writing when it turns a hidden convention into a source fact that a reader or
compiler can inspect.

A good way to leave the opening example is to pick one output coordinate and
name every input coordinate that contributed to it. The awkward part is not
recovering the answer; it is noticing how much of the answer came from memory
rather than from the program text.

## Try It

Take a tensor with shape `[2, 3, 2, 4]` and invent two different reshape chains
that both end at `[4, 12]`. For each chain, write one line that maps
`y[row, col]` back to `x[b, feature, group, slice]`. The exercise is done when
you can point to the exact coordinate that changed meaning while the final
shape stayed legal.

**Line to keep:** shape counts addresses; coordinates explain addresses.
