---
layout: book
title: "Chapter 15: What the Notation Refuses to Hide"
---

# What the Notation Refuses to Hide

The earlier chapters used the same rule in many disguises: name the coordinate
role before a later pass has to recover it. This chapter makes that rule
explicit and asks where the notation should stop.

Every notation hides something. That is not a flaw; it is the price of being
usable. The question is whether it hides the right thing.

Einlang's visible-index style refuses to hide coordinate roles. It asks the
programmer to say which dimensions survive, which are consumed, which are
omitted from a term, and which are mapped into new positions. That choice is
useful, and sometimes annoying. Both facts matter.

The governing standard is not maximal explicitness. Abstraction is welcome. The
line is crossed only when an abstraction erases a coordinate choice that
correctness, differentiation, storage, or lowering will later need back.

The recurring question is not how to make every detail explicit. It is which
coordinate facts must survive long enough for review, checking, differentiation,
and lowering to rely on them.

The deletion test is the sharpest way to find the boundary. Remove comments,
helper names, and diagrams, then ask which coordinate facts remain in the
program itself. Facts that disappear but decide correctness should not be left
to memory. They belong in the notation or in a coordinate-function boundary.

## Visibility as a Design Law

The law that has been repeated throughout the book is not "make everything
explicit." That would be a bad language. The law is narrower:

```text
Do not hide a fact that later reasoning must recover.
```

Axis roles, omitted coordinates, consumed indices, derivative addresses, and
time dependencies all pass this test. They are easy to know when the formula is
written and expensive to rediscover after the formula has been lowered into
axis numbers, layout operations, or execution history.

Other facts are good candidates for hiding. Register allocation, temporary
buffer reuse, device placement heuristics, kernel fusion order, tiling shape,
and the exact loop schedule often belong below the source. They matter for
performance, but they should not be facts a reader must recover to know
whether `class` or `batch` was normalized. A compiler earns trust by hiding
implementation choices while preserving the coordinate contract they implement.

A reasonable hidden list looks like this:

```text
hide: register allocation, temporary buffers, device placement
hide: fusion order, tiling, vector width, kernel selection
show: consumed coordinates, omitted coordinates, address maps
show: derivative addresses, recurrence edges, dynamic routes
```

The boundary is not about simplicity versus complexity. It is about which
facts a future explanation will need.

This is how the language can stay small without becoming arbitrary. It does
not include a feature because the feature is expressive in general. It includes
a feature when the feature exposes a tensor fact that later reading or lowering
will need.

Try the deletion test. Remove every axis comment, every diagram arrow, and
every prose reminder that says "this dimension is time" or "this one is
class." What facts would the program still state? The notation is valuable for
the facts that survive that deletion.

## The Productive Cost

Named coordinates take space:

```text
output[..batch, i, j] =
    sum[k](a[..batch, i, k] * b[..batch, k, j])
```

This is longer than a call to `matmul`, but it also tells you more. The batch
prefix is carried. The coordinate `k` is consumed. The coordinates `i` and `j`
survive.

The extra words earn their keep when the relationship matters: when the code is
being taught, transformed, differentiated, optimized, or debugged.

The extra words are less useful when the operation is already obvious and
stable. A good system can still provide library functions. We do not need to
write everything at maximum explicitness forever; we need a source form that
can reveal the structure when the structure matters.

The notation should be available as a lens, not imposed as decoration. A
coordinate function is fine when the contract is conventional and carried at
the call boundary. The fully indexed form earns its place when the wrong axis
would still have the right shape.

The same bargain applies to user-defined functions. A function is not
automatically too opaque for Einlang, and it is not automatically the right
abstraction either. It is useful when it hides mechanics that do not affect the
coordinate contract:

```rust
fn clamp_scalar(x: f32, lo: f32, hi: f32) -> f32 {
    if x < lo { lo }
    else if x > hi { hi }
    else { x }
}

let clipped[i] = clamp_scalar(x[i], 0.0, 1.0);
```

The call does not hide which coordinate survives. It only hides scalar
branching. A function becomes expensive when it hides the axis decision:

```rust
let y = normalize(x);
```

If the program's correctness depends on whether normalization consumes
`feature`, `time`, or `batch`, the abstraction has hidden the important fact.
The right response is not "never use functions." It is to make the function
boundary carry the coordinate contract, either in the name, the type, or the
indexed call site.

Coordinate-aware functions are the important compromise. They are how the
language avoids forcing every common operation to be expanded into raw sums and
indices. A call such as:

```rust
let p[b, class] = softmax[class](logits[b, class]);
let pred[b] = argmax[class](p[b, class]);
let image_hw_c = move_channel[channel](image);
```

is still compact, but it has not fallen back to positional convention. The
bracketed coordinate is part of the call's meaning: `softmax` normalizes across
`class`, `argmax` returns addresses in the `class` domain, and `move_channel`
moves the chosen channel role while inferring the spatial pack. The function
can hide a stable implementation while exposing the coordinate fact that
review, autodiff, and lowering must preserve.

That makes coordinate functions more than standard-library convenience. They
are the abstraction boundary for tensor programs. Ordinary scalar functions can
hide scalar mechanics. Coordinate functions can hide tensor mechanics only
after naming the coordinate they consume, preserve, create, or address.

## What Remains Outside

Visible dimensions do not replace the rest of a software system. Data loading,
plotting, distributed orchestration, file formats, and user interfaces still
belong elsewhere. Even inside numerical work, kernels and runtimes still matter.

The language's center is smaller:

```text
state tensor relationships in a form both reader and compiler can inspect
```

That center is enough here. It is not a universal language claim. It is a
claim about where tensor meaning should live.

## The Placement Rule

The useful split is not "Einlang versus everything else." It is a placement
rule:

```text
ecosystem work        host language
tensor relationships  Einlang
lowering and storage  compiler/runtime
```

Data loading, plotting, tokenizers, file formats, and orchestration belong
comfortably in a host language. Tensor equations, reductions, derivative
requests, and recurrences are the part Einlang tries to make unusually visible.
Scheduling, checkpointing, materialization, and kernel selection belong to the
compiler and runtime once the source has stated the relationship.

This split is not a retreat. It is how the small core stays sharp. A language
that tries to own every task must give many constructs weaker meanings. A
focused notation can give a few constructs stronger meanings. The goal is not
to replace the surrounding ecosystem; it is to preserve tensor structure at the
point where the ecosystem would otherwise reduce it to shape convention.

Elementwise choice belongs in the core because it can preserve coordinate
structure:

```rust
let clipped[i] =
    if x[i] < lo { lo }
    else if x[i] > hi { hi }
    else { x[i] };
```

Walking a filesystem does not belong in the core. Its errors are paths,
permissions, encodings, and formats. Clipping has coordinate semantics. Put
each job where its failures can be explained best.

## The Last Question

When you see tensor code now, ask:

```text
Which dimension has a name only in my head?
```

That question is the seed. Sometimes the answer will not matter. Sometimes the
shape convention is enough. But when the answer does matter, a visible
coordinate can turn a hidden assumption into a local fact.

The question travels well. It can be asked in a notebook, in a code review, in
a compiler, or in the margin of a paper. The notation is one answer to it, not
the only possible answer. What matters is refusing to leave dimension meaning
permanently in someone's head.

Imagine a developer five years from now reading a compiler error:

```text
normalization consumes batch, but classifier contract expects class.
value `logits[b, class]` grounds both coordinates; did you mean softmax[class]?
```

That error is possible only if the source preserved the role. Without the
coordinate contract, the future developer gets a shape, a stack trace, and a
guess.

## The Bargain in One Place

The bargain is not that explicit indices are always shorter. They are not. It
is not that every tensor operation benefits from hand expansion. It often does
not. The bargain is that when a dimension role matters, the source has a way
to say so directly.

This gives the reader a practical test, and it is deliberately small:

```text
Would a wrong axis still have the right shape?
Would a future reader need a comment to know what this dimension means?
Would a derivative or recurrence depend on this coordinate role?
Would a compiler optimization need to preserve this relationship?
```

If the answer is yes, visible coordinates are likely worth their cost. If the
answer is no, a library call may be the clearer expression. A mature system
needs both: high-level operations for common cases and explicit coordinate
forms for the places where meaning would otherwise disappear.

## What the Reader Should Keep

The portable habit is small enough to carry outside Einlang. When reading tensor
code in any framework, ask:

```text
Which coordinates survive?
Which coordinates are consumed?
Which coordinates are omitted from a term?
Which coordinates are packed or unpacked?
Which coordinate names time?
```

These questions do not require a new language. They are questions about the
program you already have. Einlang matters here because it explores what happens
when those questions are not only in the reader's head but in the source
notation itself.

Einlang makes that pressure explicit. You can disagree with the syntax and
still keep the demand it makes: when an axis carries meaning, do not make the
next reader recover that meaning from position alone.

## One Last Line Reading

Take one ordinary layer:

```rust
let y[b, out] = sum[in](x[b, in] * W[out, in]) + bias[out];
```

Most of the book is present in that line. Batch `b` survives. Output feature
`out` survives. Input feature `in` is local to the sum and leaves. The bias
omits `b`, so it is reused for every example. If a scalar loss later asks for
`@loss / @bias`, the omitted batch coordinate must be collected:

```rust
let dbias[out] = sum[b](dy[b, out]);
```

This line is the compact form of the bargain. The source did not become a full
execution plan, and it did not materialize a Jacobian. It simply kept the
coordinate roles visible long enough for the reader and the compiler to ask the
right next question.

## What the Compiler Chain Keeps

The opening reshape chain had correct shapes and hidden roles. The stronger
standard is this: tensor code should be allowed to state the coordinate
relationships that make it correct.

Einlang's compiler chain makes that standard traceable:
the source is parsed, names are resolved, indexed formulas lower to IR, ranges
and shapes are analyzed, types are inferred, autodiff requests are rewritten,
Einstein and recurrence forms are lowered, execution facts are recorded, and
validation checks the result. The compiler sees only what the source gives it.
Named dimensions give it more to work with than positional shapes alone.

That standard does not solve numerical programming. It does not replace
libraries, kernels, tests, or profiling. It only asks the program to state the
coordinate facts that later reasoning depends on.

The claim is intentionally small. It does not promise to organize an entire
software system. It does not promise that every shape bug disappears. It does
not promise that performance follows automatically from clarity. It only says
that dimensions often carry meaning, and that the source should be able to say
so when later analysis needs it.

One practice matters most: whenever tensor code feels fragile, write the
coordinates down. Name the dimensions. Mark the one omitted from a term. Mark the
one that is consumed. Mark the one that points backward in time. This pass may
confirm that the original code was fine. It may also show that the program was
depending on a convention nobody had written.

That check is the measure of success: not a new syntax for its own sake, but a
better way to read the tensor programs we already write.

One restraint remains. A notation that states everything states too
much, because the reader drowns in detail. The practical question is which
facts deserve syntax. Dimension roles often do, because they sit at the
boundary between review, checking, and lowering.

That boundary is where many tensor bugs live. A wrong axis, a mistaken
broadcast, a reduction over the wrong coordinate, a recurrence that reads the
future: these are not exotic failures. They are ordinary failures of hidden
structure. Visible dimensions do not make the programmer infallible. They make
the structure available for review and compiler checks.

That is the standard the attention examples apply next.

## A Tiny Attention Kernel

Attention is a useful pressure test because the shape can be friendly
while the communication pattern is wrong. The visible-index version keeps the
roles on the line:

```rust
let scores[b, h, i, j] = sum[d](Q[b, h, i, d] * K[b, h, j, d]);
let context[b, h, i, d] = sum[j](weights[b, h, i, j] * V[b, h, j, d]);
let out[b, i, o] = sum[h, d](context[b, h, i, d] * W_out[h, d, o]);
```

Read the three lines as one contract. `b` preserves examples. `h` separates
heads until the output projection consumes it. `i` is the query position that
survives. `j` is the key/value position that is scanned. `d` is a local feature
coordinate inside the score and context calculations, and `o` is the produced
output feature.

The common broken version is small:

```text
context[b, h, i, d] = sum[j](weights[b, h, i, j] * V[b, h, i, d])
```

That expression can have the same result shape. It is still wrong for ordinary
attention because the value no longer comes from the position being selected
by `j`. The weights scan keys, but the gathered value is fixed at the query.
A compiler that sees only ranks may have nothing useful to say. A compiler
that sees the named coordinates can report the role mismatch where it happens:
the term under `sum[j]` does not read the value at `j`.

Imagine a compiler error five years after this style has become ordinary:

```text
attention.ein:89: reverse-mode shape conflict
  head was reused implicitly at line 47, but sensitivity is later collected
  across head at line 89.
  Did you mean to reduce feature instead of head?
```

The exact wording is fiction. The possibility is not. Such an error needs
source-level roles: which coordinate was reused, which coordinate was reduced,
and where the derivative route forced the two facts to meet. If all the
compiler has is a pair of shape tuples, the best message is much poorer.

## One Layer, Many Contracts

Return to one ordinary layer. It looks familiar enough to be dismissed, but it
contains nearly every contract the book has been building toward:

```rust
let z[b, out] = sum[in](x[b, in] * W[out, in]) + bias[out];
let y[b, out] = relu(z[b, out]);
let loss = sum[b, out](y[b, out]);
```

Read it as something the compiler can keep track of.

The first line says that `b` survives. Examples do not mix. The coordinate
`out` survives because the layer produces output features. The coordinate
`in` is local to the sum. The bias omits `b`, so it is shared across examples.

The second line says that `relu` is local at `[b, out]`. It does not mix batch
items or output features. The third line consumes both `b` and `out` to
produce a scalar loss.

Now ask for gradients:

```rust
let dloss_dW = @loss / @W;
let dloss_dbias = @loss / @bias;
let dloss_dx = @loss / @x;
```

The denominator of each request tells the address of the answer:

```text
dloss_dW[out, in]
dloss_dbias[out]
dloss_dx[b, in]
```

The omitted and consumed coordinates explain the reductions:

```text
dloss_dW[out, in]     collects b
dloss_dbias[out]      collects b
dloss_dx[b, in]       collects out
```

This is not a complete derivation of every scalar factor; `relu` contributes
its local gate, and the matrix multiply contributes the appropriate bridge
term. But the global shape of the backward pass is already visible from the
forward coordinates.

The same source also gives lowering something useful. A compiler can see that
`sum[in]` is a reduction, that `b` and `out` are output coordinates, that
`bias[out]` is broadcast along `b`, and that `relu` is elementwise over the
same family. Later passes may choose loops, vectorized NumPy evaluation,
specialized matrix multiplication, or a backend kernel. Those are execution
decisions. The indexed line is the target those decisions must preserve.

A shape-only program can implement the same layer. It may even be shorter:

```python
y = relu(x @ W.T + bias)
loss = y.sum()
```

That code is valuable and familiar. The visible-index version is not claiming
that every production program must expand every layer forever. Its claim is
that when the coordinate choice matters, the source should have a form that
can show it. If a bug depends on whether `bias` was shared across batch, or
whether `W` maps input features into output features rather than the reverse,
the indexed line gives the bug a place to live.

This audit is the whole book in miniature. Name the coordinates. Mark the
ones that survive. Mark the ones that are local. Notice the ones omitted
from a term. Then let the compiler pipeline preserve, transform, and lower
that structure.

That is the standard this chapter hands to the next stress test: when
correctness depends on a dimension role, the program should be able to state
it. Abstraction is welcome; it just should not
erase the coordinate choice the program relies on.

Take one real tensor layer and mark four things: coordinates that survive,
coordinates that are local, coordinates omitted from a term, and coordinates
mapped into new positions. The fact that becomes hardest to recover after
lowering is the one the source most needed to preserve.

## Try It

Pretend you are designing a new DSL. Write two lists: facts the DSL should hide
and facts it must never hide. Include at least one example from normalization,
one from a gradient, and one from a recurrence or route. Then defend the point
where you stopped naming.

**Line to keep:** do not hide facts that future reasoning must recover.
