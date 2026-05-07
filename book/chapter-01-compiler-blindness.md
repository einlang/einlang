---
layout: book
title: "What Can the Compiler Not See?"
---

# What Can the Compiler Not See?

The CI run is green. Every test passes. The shapes at every layer are exactly
what they were last week. You merge the PR, deploy, and go to lunch. Three hours
later the metrics begin to drift — slowly, silently, with no error message and
no crash. By evening every prediction is wrong by a small, systematic amount
that none of your monitoring caught, because none of your monitoring checks
whether axis 2 still means `time`.

The PR that caused it was a one-line change: a data loader swapped two
dimensions of the same size. The shapes never changed. The compiler saw nothing.
Every downstream reshape accepted the numbers eagerly. The model continued to
train and converge — on the wrong pattern.

> "By relieving the brain of all unnecessary work, a good notation sets it free
> to concentrate on more advanced problems."
>
> — Alfred North Whitehead, *An Introduction to Mathematics* (1911)

Whitehead's point cuts both ways. A notation that omits a necessary fact does
the opposite: it forces the brain to carry what the notation should have
carried. A compiler can check that your program is well-formed. It cannot check
that your program means what you think it means — unless the meaning lives in
the source, where the compiler can see it.

Part I asks one question in six settings: what facts about tensor axes can the
notation preserve, and what facts does it let slip? Six chapters. Six places
where a hidden coordinate role can produce a shape-correct bug. By the end of
Part I, you will read every tensor line as a small audit: which names survive,
which are consumed, which are silently omitted. You will never look at
`dim=-1` the same way again.

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

Einops improves exactly this kind of line:

```python
y = rearrange(x, "b f (g s) -> (b g) (f s)", g=4, s=32)
```

That is much more readable than a chain of reshape and transpose calls. The
pattern tells us that `g` travels with `b` and `s` travels with `f`. The
question this book asks is what survives when that relation becomes a helper.
As a boundary sketch:

```text
fn pack_grouped[group, slice](x: [f32; b, f, group, slice])
    -> [f32; b * group, f * slice]
```

The einops pattern is a clear local operation. The coordinate-aware signature
is the same idea at a boundary: the caller and later compiler passes can still
see which split roles were packed. A real implementation still has to say how
linearized coordinates are formed. The point here is smaller and sharper: once
the relation is promoted to a library helper, the split roles should not fall
back into comments and convention. A reshape that accepts an integer where it
should have demanded a name is a bug waiting for a compatible shape.

That is the first blind spot. Tensor code often preserves enough information
to run, but not enough information to explain itself.

The way to read the example is to choose one output address and reconstruct
the input roles that produced it. Then make the reconstruction deliberately
wrong while keeping the shape valid. If two different role stories survive the
same extent checks, the missing information is not an implementation detail.
It is the thing the source failed to say.

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

We are not inventing names for ceremony. We are naming the facts that the
reshape arithmetic erases. The compiler sees integers and multiplication.
The reader needs to see batch, feature, and the pairing decision. The gap
between those two views is exactly the blind spot this chapter is about:
the notation determines whether a fact passes into the compiler or stays
locked in the author's memory.

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
here for the first time:

> **The Hiding Law (first draft):** Do not hide a fact that later reasoning must recover.

This sentence will carry through every chapter.

Give a compiler only the three shapes in the opening example and ask it
whether the transformation was correct. It can check element counts. It can
check that each reshape is legal. It can even replay the axis permutation. But
it cannot know whether `4` meant groups, heads, patches, or something else.
The missing fact is not a number. It is a role.

## The Same Move With Named Coordinates

Before the correct relation, try a tempting but weak naming attempt:

```text
tmp[b, feat, group, slice] = x[b, feat, group, slice]
y[row, col] = tmp[?, ?, ?, ?]
```

This names the intermediate axes, but it still makes the final packing a
mystery. The names do not participate in the address relation that matters.
They are labels on the side of the operation, not the operation itself.

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

This is the thesis in miniature. The arithmetic is not the hard part -- any
reshape chain can compute the same linearized index. The hard part is
keeping the semantic choice visible after the index is computed. When the
notation buries which coordinate traveled with which, every future reader
must recover that fact from the surrounding code -- and the surrounding
code may have changed in the six months since this line was written.

Once that relation is stable, it can be named without losing the contract:

```text
let y[b * group, feat * slice] =
    pack_grouped_features[group, slice](x[b, feat, group, slice])
```

The bracketed coordinates are the important part of the hypothetical helper.
They say which split roles are being packed, so the call is not merely
`reshape(...)` with a friendlier name. The long indexed line is the reference
meaning; the coordinate function is the reusable boundary.

This is the first appearance of a pattern that will return throughout the
book. Write the coordinate relation once, then let a named operation carry that
relation without making every call site repeat the arithmetic. The abstraction
is allowed to hide loops, indexing details, storage choices, and scalar
plumbing. It is not allowed to hide the coordinate choice that makes the
operation meaningful.

That is why the coordinate argument belongs on the call:

```text
pack_grouped_features[group, slice](x)
```

The caller is not supplying ordinary numeric parameters. It is saying which
roles are being rearranged. A shape-only helper would ask the reader to infer
that fact from positions and extents. A coordinate function makes it part of
the contract.

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

Notice what just happened. Two layout equations, same output shape,
completely different coordinate neighborhoods. A shape checker approves
both. The only tool that distinguishes them is the names on the
coordinates. This is not a hypothetical edge case -- it is the normal case
whenever two axes share an extent. And it is why naming is not decoration.
It is the difference between a check the compiler can run and a check only
the author's memory can perform.

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

### Where This Leads

This chapter showed one operation — reshape — and one question: when dimensions
move, does the source say where they went? The rest of Part I asks the same
question about different operations. Broadcast. Reduction. Softmax. Each hides a
coordinate fact in its own way. Each produces a program that still runs.

By the end of Part I, reading a tensor line will mean reading which names survive,
which are consumed, which are silently omitted. That audit is the book's
fundamental reading discipline. But Part I only teaches you to *notice* hiding.
It does not yet ask what happens *because* of it — what breaks downstream when a
hidden role propagates through a gradient, a recurrence, or a module boundary.
That question — the first deepening — belongs to Part II.

A reshape that swallowed a coordinate name is the smallest hiding. The cost is
invisible to the compiler and the shape checker. But a hidden role is a debt:
the gradient will need it, the recurrence will depend on it, the module boundary
will mismatch without it. Part I gives you the eyes. Part II gives you the bill.

Chapter 2 separates axis roles from axis positions. Chapter 3 reads
standard-library coordinate maps — transpose, `depth_to_space`, the operations
that rearrange while the reader tries to keep the story straight.

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
semantic operation. When dimensions have names, a tensor program can state more
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

```
   Coordinate Packing Map

   Input: x[b, feat, group, slice]     Output: y[row, col]
          group_count=2 slice_count=4   row=b*group  col=feat*slice

   +---+------+---+---+                +---------+----------+
   | b | feat | g | s |                | row     | col      |
   +---+------+---+---+                +---------+----------+
   | 1 |  2   | 1 | 1 |  ------>       | 3       | 9        |
   +---+------+---+---+                +---------+----------+
     |    |     |   |                     ^          ^
     +--+-'     +---+---------------------+          |
        |           +--------------------------------+
   b*group=3                              feat*slice=9

   One coordinate relation replaces reshape+T+reshape chain.
```

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

The pressure increases when two roles have compatible extents. A wrong version
can still look plausible:

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

This is the Hiding Law in its simplest form. The coordinate relation is
the fact that later reasoning must recover. If the source hides it --
buries it in a chain of reshapes -- every downstream reader, checker, and
optimizer must reconstruct it from positions and extents. If the source
states it, the compiler can preserve it, the reviewer can audit it, and
the error message can name the specific role that crossed the wrong
boundary.

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

## Summary

A shape is a count of addresses. A coordinate relation is a statement about how
those addresses were constructed from named roles. The compiler can check the
count from shapes alone. It cannot recover the roles unless the source states
them.

The reshape chain `x.reshape(...).transpose(...).reshape(...)` is not wrong. It
is incomplete: it preserves element counts while dropping the coordinate story.
When two roles share the same extent—batch and time both at 32, group and head
both at 4—a shape-only program cannot tell them apart. A coordinate relation like
`y[b * group, feat * slice] = x[b, feat, group, slice]` makes the packing
decision local and reviewable.

This chapter introduced three habits that will return throughout the book:

1. Read one output address and reconstruct every input coordinate that produced it.
2. Make that reconstruction deliberately wrong while keeping the shape valid—if
   two different role stories survive the same extent checks, the source failed
   to say something important.
3. Ask not only whether the shape is valid, but what relationship among named
   coordinates the line states.

The next chapter gives those roles a vocabulary. Where this chapter asked "did
the source withhold the role?", Chapter 2 asks "what is a role, and how does it
differ from a position?"

## Try It

This is the exercise that teaches the habit. Don't skip it.

Take a tensor `x` with shape `[B, T, F]` where `B` is batch, `T` is time, and
`F` is feature. Pick one output cell in the transposed tensor and write down
every input coordinate that contributed to it, using the Chapter 1 habit: read
one output address and reconstruct the input roles that produced it. Now write
the shape-only PyTorch version: `x.permute(1, 0, 2)`. This runs. But suppose a
refactor changes the input to `[T, B, F]` without changing the semantic
roles—time is still time, batch is still batch. The `permute(1, 0, 2)` line
still runs. What does it now compute? Write the coordinate relation for both the
original and the silently-changed version, and circle the coordinate that changed
meaning without changing value. You have just performed your first coordinate
audit. The shape checker was satisfied both times. The coordinate relation was
the only thing that noticed.

A colleague checks in this line:

```python
x.reshape(B, H, -1, F).transpose(1, 2).reshape(B, -1, H * F)
```

They claim it packs `H` (head) with `F` (feature) and leaves the middle spatial
dimension separate. Before you trust it, work backward: pick output cell
`y[b, spatial, hf]` and reconstruct which input cell it reads. Then write the
coordinate relation that makes their intent explicit:

```text
y[b, spatial, h * f_count + f] = x[b, h, spatial, f]
```

The formula states the packing decision. Now write the shape-compatible wrong
version that accidentally unpacks `H` in a different order but still produces
output shape `[B, spatial, H*F]`:

```text
wrong[b, spatial, f * h_count + h] = x[b, h, spatial, f]
```

Both assign `y[0, 5, 7]` to some input cell. They compute different cells. The
shapes are identical. Now the hardest sub-question: your codebase has a
downstream operation that does `sum` over the last axis (now `H*F`). In the
correct version, that sum pools across head-and-feature. In the wrong version, it
pools across... what, exactly? Which diagnostic question catches this before
training starts? The diagnostic question is not "does the shape match?" It is
"when I unpack index 7 on the last coordinate, do I recover the head I intended
or the feature I intended?" This is a question only the coordinate relation can
answer.

Design a coordinate function `pack_grouped[g, s]` that packs `g` with batch and
`s` with feature, working over any rank. Use rest packs so the function does not
assume a fixed number of batch or feature coordinates:

```text
fn pack_grouped[g, s](x: [f32; ..batch, g, ..feature, s])
    -> [f32; ..batch * g, ..feature * s]
```

Now design the sibling function `pack_grouped_transpose[g, s]` that packs `g`
with feature and `s` with batch:

```text
fn pack_grouped_transpose[g, s](x: [f32; ..batch, s, ..feature, g])
    -> [f32; ..batch * s, ..feature * g]
```

If `group_count == feature_count` in the input, calling the wrong function
produces a result with the same output shape. Both functions produce
`[batch_prefix * group_count, feature_prefix * group_count]`. The shapes agree.
The compiler sees legal extents everywhere. But one function says "group travels
with batch" and the other says "group travels with feature." Two coordinate
functions, identical shapes, completely different semantics—and only the bracket
tells you which is which.

Try this: set `batch_count = 3`, `feature_count = 4`, `group_count = 4`. Call
both functions and pick one output cell to trace through both paths. Show that
the two functions read different input cells for the same output coordinate
`[5, 9]`. If a reviewer sees `pack_grouped[group, slice](x)` and
`pack_grouped_transpose[group, slice](x)` side by side in a diff, can they tell
from the function name alone which one is correct? They cannot. They must read
the brackets. The bracket carries the semantic weight that the function name
cannot. Two characters—`[g, s]` versus the transposed variant—encode two
completely different coordinate stories. No shape check can tell them apart.
Only the named coordinates can.

**Line to keep:** roles do not live in numbers; they live in meaning.

### Where This Leads

You have now seen what happens when a coordinate role is hidden inside an integer:
a reshape produces compatible shapes, the compiler approves, and the bug goes
undetected. The lesson is not that reshape is dangerous. It is that the notation
determines what the compiler can check.

<div class="pause" markdown="1">
**Pause.** Before you continue, open a file in your own codebase that contains
a reshape, a transpose, or a `view`. Pick one. Ask: which coordinate roles
does this operation assume? Are those roles stated in the source, or do they
live in a comment three functions up? If a colleague swapped two axes of equal
size in the data pipeline, would this line catch it — or would it silently
produce the right shape with the wrong meaning? Do not answer here. Just look.
The rest of this book is about what you noticed.
</div>

From this, a rule begins to take shape. It is not yet the full Hiding Law, but
its first draft: **the notation must record every fact that correctness depends
on.** A correctness fact that lives only in a comment, a convention, or a memory
is a correctness fact the compiler cannot enforce. The notation is the contract
between the programmer and the machine. What the notation omits, the machine
cannot verify.

This chapter examined one kind of hidden role: a coordinate name swallowed by a
reshape. The next chapter asks a sharper question. If roles survive their first
encounter with the notation, how do we tell the compiler which roles are
*consumed* by an operation, which *survive*, and which are *omitted* from a
term? The answer requires us to separate what an axis is (its role) from where
it happens to live (its position). The distinction is the foundation of
everything that follows.
