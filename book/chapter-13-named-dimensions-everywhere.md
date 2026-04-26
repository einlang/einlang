---
layout: book
title: "Chapter 13: If Dimensions Had Names Everywhere"
---

# If Dimensions Had Names Everywhere

The earlier sections stayed close to individual moves. A reshape chain became a
coordinate map. Broadcasting became non-dependence on a coordinate. Reduction
became the disappearance of a local coordinate. Gradients became structured
sensitivity questions. Time steps became recurrence over a visible axis.

Now zoom out:

```text
What if a deep learning framework were designed around named dimensions
from the beginning?
```

Not as labels taped onto an otherwise implicit system. Not as metadata that
survives only while convenient. Imagine writing library code where named
coordinates are ordinary material, not comments that sit beside anonymous
shape tuples.

The boundary is the important part. If a framework receives an image, a token
sequence, or a batch of logits, a comment is not a receipt. The receiving code
needs a contract that later indexed expressions can actually lean on.

The boundary between role names and implemented types matters here. Comments
may describe an image as `[batch, row, col]`; checked code still needs rank,
extents, element type, and a place for those role names to matter.

## Names Everywhere, or Names Where They Matter

A named-dimension framework could make every axis carry a name at all times.
That sounds clean until the fifth temporary in a small helper starts wearing a
full ceremonial title. Many intermediate values are local enough that long
names add little.

The opposite design keeps names out of the framework and relies on convention.
That is familiar, but it puts the most important promises in comments,
documentation, and memory. The code can remain shape-correct while role meaning
leaks away.

The more useful design is selective visibility. Names should be available where
a role affects correctness: at boundaries, reductions, broadcasts, reshapes,
derivative requests, recurrences, and model-level APIs. This is not names
everywhere. It is names at the places where hiding the role would make the next
line harder to trust.

Take a notebook model and remove every shape comment. If the code still makes
clear which axis is batch, which is time, and which is feature, the framework
is carrying the load. If the explanation collapses, the important part was
living in prose outside the program. Selective visibility moves the fragile
parts of that prose into the source.

One decision procedure is deliberately plain:

```text
Does confusing this axis change correctness?
    yes -> name it.
Will this axis be reduced, normalized, shifted, or packed?
    yes -> name it.
Will a term omit this axis and thereby broadcast?
    yes -> name it.
Will a derivative request need to explain where this axis went?
    yes -> name it.
Otherwise, keep it anonymous if the local expression remains clear.
```

The goal is not to win a ceremony contest. It is to spend names where they buy
review, checking, or diagnostics.

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

The framework would not have to guess that axis `0` is batch because convention
says so. It would see `b`. It would not have to infer from a singleton
dimension that something is being repeated. It would see the missing
coordinate. It would not have to treat every loop as opaque before analysis. It
would see a recurrence boundary and an offset.

At framework scale, that is the change: an axis role stops being a convention
remembered by the caller and becomes something the operation can actually
refer to.

## What Becomes Easier to Ask

With visible coordinates, common review questions become local:

```text
Did batch survive?
Which coordinate was normalized?
Which coordinate was reduced?
Which coordinate names time?
Which coordinate was split into heads?
```

These questions already exist in current tensor programs. The difference is
where the answers live. In shape-oriented code, the answers often live in
comments, conventions, diagrams, and local memory. In a visible-index
style, the answers can live in the program text.

That does not remove the need for tests. It changes what a test is confirming.
Instead of using tests to discover which axis a formula meant, tests can check
whether the stated formula behaves numerically as expected.

## What the Type Checker Really Knows

It is tempting to say that an axis name becomes a type. That is a useful
intuition only if it does not obscure the implemented syntax. Einlang's type
checker knows scalar types such as `i32` and `f32`, rectangular tensor types
such as `[f32; 2, 3]`, wildcard extents such as `[f32; ?, ?]`, and dynamic-rank
contracts such as `[f32; *]`.

The coordinate binder is separate:

```rust
let centered[i, j] = image[i, j] - 0.5;
let row_sum[i] = sum[j](image[i, j]);
let explicit = sum[i in 0..10](data[i]);
```

The names `i` and `j` are scoped integer coordinates. Their ranges are inferred
from indexed operands or written explicitly with `in`. The compiler checks
that those ranges agree with rectangular shapes. It does not use syntax like
`i: batch` to say "this coordinate has type batch."

So the implemented check has three parts:

```text
element type      f32, i32, bool, ...
rectangular shape rank and extents
coordinate names  scoped index roles in formulas
```

Together, those three facts are enough to catch many shape and role errors at
compile time without pretending that Einlang has a separate "axis type"
syntax.

Read the check like a receipt, not a theorem:

```text
required fact       received fact        status
rank 2              image has rank 2     ok
element f32         image elements f32   ok
row coordinate      first index role     ok
col coordinate      second index role    ok
```

The receipt is useful because it is attached to a boundary the compiler can
reject. A comment can say that an image is `[row, col]`; a contract gives
later indexed code something firmer to rely on.

## What This Would Not Solve

Named dimensions would not make numerical programming easy. They would not
remove the need for excellent kernels, memory planning, distributed execution,
debugging tools, profilers, or interop with existing ecosystems.

They would also introduce their own costs. Names can become noisy. A language
that forces too much ceremony will lose the reader before it helps the
compiler. A useful notation must reveal structure without making every line
feel like paperwork.

That is why the argument has stayed focused. It has not tried to design an
entire deep learning framework. It has only shown a test for framework
features: if coordinate structure affects correctness, the program needs a
place to state it.

## Are Functions the Right Abstraction?

A tensor language cannot avoid functions. Standard-library operations,
activation functions, helper computations, and module boundaries all need a
way to name reusable behavior. The interesting question is smaller: when is a
function a useful name, and when is it a curtain?

This function is a good abstraction:

```rust
fn relu_scalar(x: f32) -> f32 {
    if x > 0.0 { x } else { 0.0 }
}

let y[b, f] = relu_scalar(x[b, f]);
```

The function hides scalar control flow, not coordinate structure. The caller
still says that `y[b, f]` reads `x[b, f]`. The abstraction removes noise
without erasing the part of the tensor program we care about.

This function boundary is more suspicious:

```text
normalize(x) = ...
y = normalize(x);
```

The call may be perfectly implemented. The problem is that the call site no
longer says whether normalization consumes `feature`, `time`, `batch`, or
`class`. If that choice is part of correctness, the function name has hidden
the one fact the reader needed.

A better boundary names the role in the interface or leaves the indexed
relation at the call site:

```rust
let y[b, class] = softmax_over[class](logits[b, class]);
```

or, when the library function exists:

```text
softmax_over_class(logits)
```

with a rule that states which coordinate is consumed. The function is then
a named coordinate operation, not just a shortcut around a block of code.

The implementation gives this design some room. Parameters without type
annotations are monomorphized at call sites, and specialized function bodies
can be processed by rest-pattern, shape, and type passes. A generic helper is
not automatically a permanent black box. Once the compiler sees a call with
concrete argument types and shapes, it can specialize the function body and
analyze the resulting IR.

But monomorphization does not answer the design question by itself. A function
can still be the wrong abstraction if its name hides the only coordinate fact
that matters. The useful split is:

```text
hide scalar mechanics
hide stable implementation choices
do not hide the coordinate role that decides correctness
```

Functions belong in Einlang, but not as a replacement for indexed relations.
They are good when they preserve or state the coordinate choice. They are bad
when they make the caller say only "do the thing" while the meaning of the
thing lives elsewhere.

## Where the Question Goes Next

The rest of the chapter keeps the question at framework scale. What would
better diagnostics look like? Where would names become annoying? How would a
named-coordinate program cross into positional libraries? The answer cannot be
"name everything." It has to be a placement rule.

## What a Framework Would Gain

A framework built around named dimensions would gain a different default
debugging posture. Instead of asking "why did these shapes fail to align?", the
first question could be "which coordinate relationship did this line claim?"

For example, a framework could report that a formula tries to normalize over
`batch` when surrounding code expects normalization over `class`. It could
distinguish an omitted `time` coordinate from an omitted `feature` coordinate. It
could show that a recurrence reads `state[t + 1]` in a forward time range. It
could preserve the difference between a head coordinate and a within-head
feature coordinate through lowering.

Those diagnostics are not guaranteed by names alone. They require a compiler
and a type/shape system that take the names seriously. But without the names,
the diagnostics have little source material to work with. The framework
is left to infer intent from positions and sizes.

At framework scale, this is the main payoff: a rejection can name a broken
relationship rather than only incompatible extents. The error can be about
which role was consumed, not only which axis length failed to align.

## If the Error Named the Role

Imagine a classifier boundary written with roles:

```rust
let logits[b, class] = model(x[b, feature]);
let probs[b, class] = softmax_over[class](logits[b, class]);
```

The framework now has enough source material to distinguish a good
normalization from a shape-compatible but meaning-wrong one:

```rust
let bad[b, class] = softmax_over[b](logits[b, class]);
```

Both formulas can produce a tensor addressed by `[b, class]`. The difference
is not the resulting shape; it is the coordinate that was treated as the
distribution. A useful framework error could therefore speak in roles:

```text
expected normalization over class, found normalization over batch
```

That message is the larger design choice in miniature. The system is not only
checking whether arrays can be aligned. It is checking whether the line states
the coordinate relationship the surrounding model promised. The names make the
mistake local enough to report.

## What It Would Cost

The cost is also real. Names create surface area. APIs have to decide when
names must match, when they can be renamed, and when a library function is
allowed to introduce or consume a coordinate. Error messages have to explain
coordinate relationships without overwhelming the user. Interop with existing
shape-based libraries has to translate between named roles and positional
axes.

Interop is where the design becomes most concrete. A call into an existing
kernel may still need a positional layout:

```text
[b, time, feature] -> axis order [0, 1, 2]
```

The question is whether that translation is a silent convention or an explicit
boundary. If the boundary records that `time` became position `1`, then the
named world can resume after the call. If the boundary drops the role, the
program has re-entered the old regime where later code must remember what the
axis meant. A practical framework would need good tools for this crossing:
renaming, projecting names into positional APIs, and restoring names from
well-specified library promises.

The argument is not for names everywhere at all times. It is for making names
available at the places where roles affect correctness. The framework question
is not "can we eliminate anonymous shapes?" It is "can we stop forcing
important dimension meaning to live only in convention?"

A migration path for an existing codebase should be gradual. Start at the
dangerous axes: logits versus batch, time versus feature, head versus
within-head feature, and any dimension that is normalized or reduced. Add
boundary receipts first, so data entering the system has rank and role
expectations. Then replace the riskiest reshapes, broadcasts, and reductions
with named-coordinate forms. Only after those hot spots are stable should a
team consider naming calmer internal temporaries. The useful path is not a
flag day; it is a sequence of small receipts at the places where shape-correct
bugs have already cost time.

## From Local Reading to Frameworks

The local examples now point toward a larger question. Once a single
expression can expose its coordinate roles, model architecture can be read the
same way: as a network of rules among named dimensions. That does not
prove that an entire ecosystem should be rebuilt around names, but it sharpens
the question. Once dimensions have names in small examples, it becomes natural
to ask what current frameworks lose when those names vanish at the boundary of
every operation.

The imagined framework would also change documentation. Instead of documenting
every layer with a prose shape convention, examples could show the coordinate
rule directly. A layer would accept more than `[B, T, D]`; it would accept
values whose roles are batch, time, and feature. A reshape would not only
produce another tuple; it would state which roles were packed, split, or
discarded.

That is a different kind of API promise. It is not only about preventing
mistakes. It is about keeping the model discussable. When code, explanation,
and compiler facts use the same coordinate names, fewer claims have to live in
a parallel layer of comments. A framework that preserves those distinctions
would not make models automatically correct, but it would make their coordinate
roles harder to lose.

## Boundary Receipts Are Not Comments

Imagine a model boundary that receives images from a host language. The hard
part is not loading the array; it is deciding where dynamic data becomes a
checked value that indexed code is allowed to trust:

```rust
use std::io::load_npy;
let image = load_npy("digit.npy") as [f32; 28, 28];
let centered[row, col] = image[row, col] - 0.5;
```

The cast is an implementation-level check: element type `f32`, rank two, and
extents `28, 28`. The coordinates `row` and `col` are roles introduced by the
indexed declaration. Together they let the next formula say what it means.

Now suppose a flattened file arrives instead:

```rust
use std::io::load_npy;
let image = load_npy("digit.npy") as [f32; 28, 28];
```

If the file contains `[f32; 784]`, the boundary cast should fail before the
indexed formula runs. That is not a philosophical point; it is a practical
division of responsibility. The host language may load dynamic data. Einlang
receives a shaped value. The coordinate formula relies on that check.

A framework with named dimensions everywhere would make this boundary even
more explicit:

```text
host value          positional array from a file
boundary receipt    [f32; 28, 28]
coordinate roles    row, col
formula             centered[row, col]
```

Each layer answers a different question. The type tells what kind of elements
and how many dimensions. The extents tell how many positions. The coordinate
names tell how the positions should be read.

Now move to a classifier:

```rust
let logits[b, class] = model(image[b, row, col]);
let probs[b, class] = softmax_over[class](logits[b, class]);
```

The type and shape system may know that `logits` has rank two. The names say
that the second coordinate is a class distribution. If the model accidentally
normalizes over `b`, the output shape can remain `[b, class]`, but the
distribution is wrong. A named-coordinate framework can report the role error
instead of leaving the reader to infer it from a suspicious accuracy curve.

This example also shows why the book does not ask for names everywhere at all
times. Inside `model`, a temporary scalar accumulator may not need a long role
name. But at boundaries, reductions, reshapes, and normalizations, the role is
part of correctness. If it lives only in prose, the compiler cannot help.

The implemented Einlang compromise is modest and concrete. It has ordinary
scalar types and rectangular tensor shapes. It introduces coordinate names
where indexed formulas need them. It does not pretend that `row` or `class` is
a new primitive scalar type. The useful part comes from combining the ordinary
type check with scoped coordinate roles at the places where the formula makes
a claim. The difficulty gradient is the boundary: before the cast, the host
value is only data; after the cast and indexed binding, it is a value with
rank, extent, element type, and coordinate roles.

## Shape Inference Builds the Check

Shape analysis is where many earlier facts become one checked shape. A literal can
give a shape:

```rust
let x = [[1.0, 2.0], [3.0, 4.0]];
```

The nested array tells the compiler that `x` is rectangular with shape `(2,
2)`. A cast can give a boundary shape:

```rust
let weights = load_npy("W.npy") as [f32; 784, 10];
```

An indexed declaration can give a result shape from coordinate domains:

```rust
let y[i, j] = x[i, j] + 1.0;
```

If `i` and `j` are inferred from `x`, then `y` receives the same extents. If
they are written explicitly, those ranges become the result extents. Shape
analysis is the pass that merges these sources of information: literal structure,
type annotations, inferred index ranges, explicit ranges, rest-prefix
expansion, and grouped Einstein clauses.

This is also why shape and type are adjacent but not identical. `[f32; 2, 2]`
contains an element type and a rectangular shape. The expression:

```rust
let z[i, j] = x[i, j] + 1.0;
```

needs both. Shape analysis can say `z` is two-dimensional with the same extents
as `x`; type inference can say the element type is `f32`, assuming `x` is
`f32` and the literal is compatible. A wrong rank is a shape issue. Adding a
string to a float is a type issue. Many tensor bugs involve both, but the
compiler keeps the facts separate so diagnostics can point to the right
check.

When the book says that an axis acts like a promise, this is the concrete
meaning. There is no magic dimension type hiding in the implementation. The
promise is assembled by passes that agree: name resolution gives identity,
rest-pattern preprocessing turns prefixes into slots when possible, range
analysis gives domains, shape analysis gives extents, and type inference gives
scalar meaning.

When an image tensor arrives from a host language, `image[b, row, col]` should
not be the first place where its structure is discovered. The boundary should
already have said what can be checked as shape and what must be preserved as
role.

## Try It

Pick one axis in a real model and decide whether it should be named. Use three
questions: can the wrong role have the same size, can the axis be reduced or
broadcast, and can a derivative later ask about it? If all answers are no,
leave it anonymous. If any answer is yes, give it a role name.

**Line to keep:** name dimensions where future reasoning would otherwise have
to reverse-engineer them.
