---
layout: book
title: "If Dimensions Had Names Everywhere"
---

# If Dimensions Had Names Everywhere

> "The limits of my language mean the limits of my world."
>
> — Ludwig Wittgenstein, *Tractatus Logico-Philosophicus* (1922)

Part IV is the stress test. The coordinate vocabulary that survived single
operations must now cross module boundaries, multi-head parallelism, attention
protocols, and dynamic expert routing. If the language cannot name the
coordinate role at scale, the role becomes invisible — and the limit of the
language becomes the limit of what the program can check.

You have finished wiring the encoder to the decoder. The model definition looks
right. You run the training script. It crashes:

```
RuntimeError: mat1 and mat2 shapes cannot be multiplied
(32 × 512) and (768 × 512)
```

You trace the shapes. The encoder outputs `(batch, time, feature)`. The
decoder expects `(batch, time, hidden)`. Both are 3D tensors. Both have the
same batch and time extents. The last dimension is 512 in one place and 768 in
the other. Two numbers that don't connect, buried in a stack trace 30 lines
deep.

Now imagine the same error with named coordinates. The compiler says:

```
Coordinate mismatch: encoder outputs 'feature' but decoder expects 'hidden'.
Both have shape (batch=32, time=8), but the last coordinate differs.
Did you forget a projection layer?
```

The error message names the problem. Two different things have the same shape,
and the compiler knows they are different because they are called different
things. The diagnostic changes from "two numbers don't line up" to "these two
concepts don't match."

Earlier chapters examined individual moves: a reshape, a reduction, a gradient.
The question was whether one expression could be read clearly. This chapter
changes scale. The question is whether a framework boundary can preserve the
same facts when values travel through APIs, modules, and files.

## Selective Visibility

A named-dimension framework could make every axis carry a name at all times.
That sounds clean until the fifth temporary in a small helper wears a full
ceremonial title. Many intermediate values are local enough that long names add
noise.

The opposite design keeps names entirely out of the framework and relies on
convention. That is familiar, but it puts the most important promises in
comments, documentation, and memory. The code can remain shape-correct while
role meaning leaks away.

The useful design is selective visibility. Names should appear where a role
affects correctness: at boundaries, reductions, broadcasts, reshapes,
derivative requests, recurrences, and model-level APIs. This is not names
everywhere. It is names at the places where hiding the role would make the next
line harder to trust.

A simple decision procedure:

```text
Does confusing this axis change correctness?
    yes → name it.
Will this axis be reduced, normalized, shifted, or packed?
    yes → name it.
Will a term omit this axis and thereby broadcast?
    yes → name it.
Will a derivative request need to explain where this axis went?
    yes → name it.
Otherwise, keep it anonymous if the local expression remains clear.
```

The same procedure as a flowchart:

```text
Does a wrong role keep the same shape?
    no  → a plain shape may be enough locally.
    yes → will the role be reduced, broadcast, packed, normalized, or
           differentiated?
           no  → keep the name optional.
           yes → make the coordinate visible at the boundary.
```

Spend names where they buy review, checking, or diagnostics. Do not spend them
on ceremony.

The decision procedure is not formulaic. It is a discipline for asking one
question at every boundary: will my future self need to know which coordinate
this was? If the answer is yes, the name is not decoration. It is a receipt the
source is writing to the person who debugs this code at midnight, with half the
context gone and the shape comments stale.

## The Boundary Receipt

When a framework receives an image, a token sequence, or a batch of logits, a
comment is not a receipt. The receiving code needs a contract that later
indexed expressions can actually lean on.

Imagine a model boundary that loads an MNIST digit:

```rust
use std::io::load_npy;
let image = load_npy("digit.npy") as [f32; 28, 28];
let centered[row, col] = image[row, col] - 0.5;
```

The cast is an implementation-level check: element type `f32`, rank two, extents
`28, 28`. The coordinates `row` and `col` are roles introduced by the indexed
declaration. Together they let the next formula say what it means.

Now suppose a flattened file arrives:

```rust
let image = load_npy("digit_flat.npy") as [f32; 28, 28];
```

If the file contains `[f32; 784]`, the boundary cast fails before the indexed
formula runs. That is not philosophy; it is a division of responsibility. The
host language loads dynamic data. Einlang receives a shaped value. The
coordinate formula relies on that check.

The boundary layers:

```text
host value          positional array from a file or library
boundary receipt    [f32; 28, 28]       type, rank, extents
coordinate roles    row, col            introduced by indexed binding
formula             centered[row, col]  the computation
```

Each layer answers a different question. The type tells what kind of elements
and how many dimensions. The extents tell how many positions. The coordinate
names tell how the positions should be read.

## Coordination Across Modules

This is the third deepening of the book. Until now, every coordinate audit was
performed on a single operation, written by a single author, in a single file.
Now coordinates must cross boundaries where the writer of the encoder and the
writer of the decoder may be different people, working in different repositories,
who have never spoken. The name is the contract. If it disappears at the
boundary, the contract becomes a convention — and conventions survive only in
memory. An abstraction is only as strong as the primitive it carries across the
boundary. When the primitive is a position, the abstraction is a comment. When
the primitive is a name, the abstraction is a check.

Once names exist at boundaries, they can travel. An encoder produces
`output[time, batch, feature]`. A decoder consumes `input[time, batch, hidden]`.
At the join, the compiler sees `feature` and `hidden` and asks: are these the
same coordinate? If not, is there a projection that maps one to the other?

Without names, the join is purely positional: axis 2 must have extent 512 on
one side and 768 on the other. It doesn't. Crash. With names, the compiler can
say what failed: "you called `feature` what the decoder calls `hidden`."

This is not a new kind of type. It is the existing shape check augmented with
the fact that two coordinates claimed different roles. The type system knows
`f32`. The shape system knows `(batch, time, 512)` and `(batch, time, 768)`. The
names say the last coordinate is `feature` on one side and `hidden` on the
other. The mismatch is not between numbers; it is between concepts.

This is not a type-safety argument dressed up as philosophy. It is the Hiding
Law crossing a file boundary. When `encoder.rs` exports `output[time, batch,
feature]` and `decoder.rs` imports `input[time, batch, hidden]`, the join is a
fact that spans two modules. The compiler can check the role only if the role
survived the boundary. The Hiding Law does not care about file organization. It
cares about whether the fact is available where the check must happen.

## What Becomes Easier to Ask

With visible coordinates, common review questions become local:

```text
Did batch survive?
Which coordinate was normalized?
Which coordinate was reduced?
Which coordinate names time?
Which coordinate was split into heads?
```

These questions already exist in every tensor program. The difference is where
the answers live. In shape-oriented code, the answers live in comments,
conventions, diagrams, and the reader's memory. In a visible-index style, the
answers live in the program text.

That does not remove the need for tests. It changes what tests confirm. Instead
of using tests to discover which axis a formula meant, tests can check whether
the stated formula behaves numerically as expected.

Sixteen chapters, one pattern: when the answer lives in the source, every
question about it becomes a compiler query instead of a memory query. When the
answer lives in a comment, it rots. When it lives in a convention, it drifts.
When it lives only in the reader's head, it disappears the moment the reader
turns the page. The coordinate name is a bet that the question will be asked
again.

## Functions That Preserve Roles

A function can hide useful things. It can also hide the one thing the reader
needed.

This function is fine:

```rust
fn relu_scalar(x: f32) -> f32 {
    if x > 0.0 { x } else { 0.0 }
}

let y[b, f] = relu_scalar(x[b, f]);
```

It hides scalar control flow, not coordinate structure. The caller still says
that `y[b, f]` reads `x[b, f]`. The abstraction removes noise without erasing
the part of the tensor program we care about.

This boundary is more suspicious:

```text
y = normalize(x);
```

The call may be perfectly implemented. The problem is that the call site no
longer says whether normalization consumes `feature`, `time`, `batch`, or
`class`. If that choice is part of correctness, the function name has hidden
the fact the reader needed.

A better boundary names the role:

```rust
let y[b, class] = softmax[class](logits[b, class]);
```

The function is a named coordinate operation. It hides the stable
implementation, but the call site still says which coordinate supplies the
distribution.

A rank-polymorphic helper makes the same point:

```rust
fn move_channel[channel, ..spatial](x: [f32; channel, ..spatial])
    -> [f32; ..spatial, channel]

let image[channel, row, col] = load_image();
move_channel[channel](image)
```

Only `channel` is named at the call site because only that role is the choice.
The spatial pack is inferred. If the caller must choose a whole group, the
group is one coordinate argument: `layer_norm[(height, width, channel)](x)`.

The rule is simple: hide scalar mechanics, hide stable implementation choices,
do not hide the coordinate role that decides correctness.

## The Classifier Boundary

Put the pieces together around a classifier:

```rust
let logits[b, class] = model(image[b, row, col]);
let probs[b, class] = softmax[class](logits[b, class]);
```

The type and shape system knows `logits` has rank two. The names say the second
coordinate is a class distribution. Now consider a shape-compatible but
meaning-wrong variant:

```rust
let bad[b, class] = softmax[b](logits[b, class]);
```

Both formulas produce a tensor addressed by `[b, class]`. The difference is not
the resulting shape; it is the coordinate treated as the distribution. A
named-coordinate framework can produce:

```text
expected normalization over class, found normalization over batch
```

The diagnostic names the broken relationship. It does not report incompatible
extents—it reports a role violation.

## Gradual Adoption

A large codebase does not need a flag day. Start at the boundaries where data
enters the model: images get `b`, `row`, `col`; token batches get `b`, `t`;
logits get `class`. Next, name the places where axes are consumed: normalization,
reductions, attention softmax, pooling, and loss aggregation. Finally, name the
shape-changing paths: reshape, flatten, pack, unpack, and dispatch.

Each step should leave existing numerical tests in place while adding one new
coordinate-level check. The path is not a flag day; it is a sequence of small
receipts at the places where shape-correct bugs have already cost time.

## The Shift

Familiar operations would change character:

```text
reshape       coordinate packing and unpacking
transpose     coordinate reordering
broadcast     non-dependence on a coordinate
matmul        one consumed coordinate and two surviving coordinates
gradient      transformed sensitivity structure
recurrence    dependency over a visible time axis
```

The framework would not guess that axis `0` is batch because convention says so.
It would see `b`. It would not infer from a singleton dimension that something
is being repeated. It would see the missing coordinate. It would not treat every
loop as opaque before analysis. It would see a recurrence boundary and an
offset.

At framework scale, this is the change: an axis role stops being a convention
remembered by the caller and becomes something the operation can actually refer
to.

Part I taught us to name coordinates. Part II taught us to trace their
gradients. Part III taught us to give time a direction. Now Part IV asks:
can these names survive a real program, with module boundaries, multi-head
parallelism, and dynamic expert routing? Chapter 13 has tested the first of
these — the module boundary — and found that the answer is yes, but only if
we spend names at the join points where roles must be checked. The next
chapters test the harder boundaries: attention, hiding, and routes chosen by
the data itself.

## Try It

Start with a model boundary written with a shape cast. Consider this pair of
lines:

```rust
let image = load_npy("digit.npy") as [f32; 28, 28];
let centered[row, col] = image[row, col] - 0.5;
```

If the file actually contains `[f32; 784]`, the first line catches the error at
the cast because the shape disagrees. If the file contains `[f32; 28, 28]` but
the pixels are stored in column-major order, the shape is right but the `row`
and `col` roles are swapped — and the cast does not catch this because it only
checks extent, not role. To make this safe for a batch of images, you add a
batch coordinate name at the boundary; the cast then expects `[f32; batch, 28,
28]` and rejects any input whose rank or extent does not match.

Now consider a migration plan for a 47-call-site codebase. A colleague proposes
adding one coordinate name at a time. At call site 12, they annotate `x[batch,
time, hidden]`. At call site 23, they annotate `W[hidden, out]`. The model still
runs because the host language strips names at the boundary. But by call site
30, they discover that `hidden` at call site 12 refers to 512 features and
`hidden` at call site 23 refers to 768 units. The numbers were always different,
but without names the difference was a runtime crash. With names, it can be a
compile-time mismatch. The compiler error message should name both coordinates,
state their extents (`hidden_512` versus `hidden_768`), and suggest the likely
fix: insert a projection layer that maps one to the other.

A better migration plan does not name every call site. Instead, name the three
boundaries where data enters the model (Step 1), then the four places where axes
are consumed (Step 2), then the two shape-changing paths (Step 3). Each step
names one coordinate and earns its cost by preventing a specific category of
positional-axis bug. For each step, write a test that would remain green — the
tests check numerical values, not coordinate names. The names are for humans and
the compiler; the numerical tests are for correctness. The bug most likely to be
caught by Step 2 is a reduction over the wrong axis where the shapes happen to
be compatible. In positional code, `sum(dim=1)` and `sum(dim=2)` look equally
plausible when extents coincide. Named coordinates turn the ambiguity into a
compiler check.

Next, write a function that moves a coordinate through a rest pack, formulated
as a migration step. You start with positional code that computes a transpose by
number:

```python
# positional: move axis 1 (channel) to the end
x = x.transpose(0, 2, 3, 1)  # NCHW -> NHWC
```

At call site 17, this transpose is written once. At call site 42, it is written
again with a different axis numbering because the input arrives in a different
layout. The migration proceeds in three steps.

Step 1 names the coordinate at the model boundary:

```rust
let image[batch, channel, height, width] = load_batch();
```

Step 2 replaces the positional transpose with a named one:

```rust
let nhwc[batch, height, width, channel] = image[batch, channel, height, width];
```

The indexed binding does the reordering in one line. No axis positions to
remember. The right-hand side reads `image` in the old order. The left-hand side
writes `nhwc` in the new order.

Step 3 writes the function that makes this reusable across all 47 call sites,
taking a coordinate name as a parameter and reordering accordingly:

```rust
fn move_channel_last[channel, ..spatial](x: [f32; channel, ..spatial])
    -> [f32; ..spatial, channel]
{
    let result[..spatial, channel] = x[channel, ..spatial];
    result
}
```

At call site 42, where the input has a different layout, this function still
works — but only if the caller correctly identified which coordinate is
`channel`. If the caller passes the wrong coordinate, the function moves the
wrong axis. The coordinate name at the call site is the contract. The function
cannot verify it. The caller must supply it.

The deeper point is that named coordinates do not eliminate all bugs. They move
the bug from "which axis number was that?" to "which coordinate is that?" The
difference is that the name is a declaration the reader can check. An axis
number is a positional convention the reader must remember. The three-step
migration plan is not a flag day. It is a sequence of small receipts, each one
purchased by a past debugging session. Each step adds one fact to the source,
and each fact earns its cost by preventing a specific category of positional-axis
bug.

**Line to keep:** names should appear where roles affect correctness, not
everywhere at once.

### Where This Leads

Part IV is the stress test. Parts I, II, and III worked one operation at a time
— a reshape, a gradient, a recurrence. Each chapter asked: can named coordinates
make this operation readable? Part IV asks the harder question: can the same
small vocabulary survive the full complexity of a production model?

Chapter 13 takes coordinates across module boundaries, asking whether names can
survive APIs, files, and framework joins. Chapter 14 applies them to multi-head
attention — the operation that dominates modern architectures. Chapter 15 draws
the line between what notation should show and what it should hide. Chapter 16,
the final stress test, asks whether named coordinates can survive dynamic
routing, where the communication graph is chosen by the data itself.

If the principle holds through all four chapters — if the coordinate role
remains visible through APIs, attention patterns, the hiding-law boundary, and
runtime routing decisions — then it is not a notation for textbooks. It is a
notation for the code you actually write.

Notation has to be willing to show you something. Chapter 14 asks: when you
write attention, what does the notation refuse to hide?