---
layout: book
title: "What the Notation Refuses to Hide"
---

# What the Notation Refuses to Hide

> "Perfection is achieved not when there is nothing more to add, but when there
> is nothing left to take away."
>
> — Antoine de Saint-Exupéry, *Terre des Hommes* (1939)

Six months into the project, a senior engineer proposes a cleanup: "The
coordinate brackets are noise. We know `class` is always the last axis.
Abstract them behind a helper and the code gets shorter." The team agrees. The
helper is written. The brackets disappear. The code is cleaner.

A month later someone adds a multi-label loss. It needs `class` to be the FIRST
axis, not the last. The compiler says nothing — the helper hid the coordinate
role, so the compiler no longer knows which axis is `class`. The shapes are
compatible either way. The loss descends. The model converges. The predictions
are wrong for a reason invisible to every tool in the pipeline.

The cleanup didn't hide an implementation detail. It hid the one fact that
later reasoning had to recover. Three weeks later, someone else debugs the
multi-label loss at 3 AM. The loss descends. The shapes are fine. The bug is
invisible because the coordinate role was visible once — and then it wasn't.
The cleanup traded a local fact for a silent failure. That is the line this
chapter draws.

Every notation hides something. That is not a flaw; it is the price of being
usable. The question is whether it hides the right thing. Einlang's
visible-index style refuses to hide coordinate roles. That choice is useful,
and sometimes annoying. Both facts matter.

## The Hiding Law

The law that has guided the entire book is one sentence:

```text
Do not hide a fact that later reasoning must recover.
```

Axis roles, consumed coordinates, omitted coordinates, derivative addresses,
and time dependencies all pass this test. They are easy to state when the
formula is written and expensive to rediscover after the formula has been
lowered into axis numbers, layout operations, or execution traces.

Other facts are good candidates for hiding. Register allocation, temporary
buffer reuse, device placement heuristics, kernel fusion order, tiling shape,
and the exact loop schedule belong below the source. They matter for
performance, but they should not be facts a reader must recover to know whether
`class` or `batch` was normalized.

A concrete boundary:

```text
hide: register allocation, temporary buffers, device placement
hide: fusion order, tiling, vector width, kernel selection
show: consumed coordinates, omitted coordinates, address maps
show: derivative addresses, recurrence edges, dynamic routes
```

The boundary is not about simplicity versus complexity. It is about which facts
a future explanation will need.

The fourteen chapters were not a list. They were an arc.

You cannot understand why a hidden coordinate breaks a gradient until you first
learn to notice the coordinate was hidden at all. You cannot understand why time
compounds the problem until you have watched a single backward pass silently sum
over the wrong axis. Each chapter assumed the scar tissue from the previous one.

The book opens with a reshape that swallowed an integer where it should have
demanded a name. It closes with a multi-head attention block whose communication
protocol — who asks, who answers, which head is speaking — is either stated in
the source or stored in the reader's head. The distance between those two
sentences is the distance this book was built to cross. Fourteen chapters. One
rule.

A notation is a prosthetic memory. It carries facts so your brain does not have
to. A good one carries the facts that later reasoning will need. A bad one
carries the facts that are easy to encode and leaves the hard ones to you. The
difference is not the notation's power. It is the notation's honesty about what
it refuses to record.

## The Compiler Reads Coordinates Too

There is a quieter proof of the Hiding Law, and it lives inside the compiler
that enforces it.

The Einlang compiler is not a single pass. It is a chain of readers, and each
reader depends on coordinate names to do its job. Walk them in order.

**Shape inference** reads axis names to decide whether an expression is legal
before it runs. `sum[k](A[i, k] * B[k, j])` succeeds if `k` appears in both `A`
and `B` and the non-reduced axes align. Without names, the check is integer
matching: shape `[32, 64]` times shape `[64, 128]` is only legal if the second
dimension of the first equals the first of the second. That is the positional
contract. Under names, the contract is: `i` survives from `A`, `j` survives from
`B`, `k` appears in both and is consumed. The rule is the same rule a human uses
to read the expression. The compiler becomes a reader of the same notation the
programmer reads.

**Gradient lowering** reads axis names to build the backward pass. When the
compiler sees `@loss / @W[c, i]`, it must decide which coordinates to sum over
and which to preserve. The rule is: preserve the coordinates of `W`, sum over
everything else. That rule works because the forward pass already recorded which
coordinates survive, which are consumed, and which are omitted. The gradient
passer reads that record backward. It is the same rule the programmer learns in
Chapter 9, and the compiler applies it mechanically, because the names make it
mechanically applicable.

**Storage planning** reads axis names to decide which tensors can share memory.
A recurrence `fib[n in 2..N] = fib[n-1] + fib[n-2]` creates a dependency along
`n`: each step reads the previous step. The compiler sees the index offset
`n-1`, recognizes the backward edge, and allocates a single buffer that updates
in place — no materialization of the full sequence is needed if only the final
value is observed. The same logic applies to any recurrent coordinate: `t` in a
time series, `iter` in an optimization loop, `layer` in a deep network unrolled
as a recurrence. The compiler inspects the dependency graph and decides: this
coordinate is recurrent, allocate once and update in place. That coordinate is
consumed by a reduction, free after the sum completes. Another is broadcast,
share across the batch. The decisions follow from the names, and the names make
the decisions auditable. A compiler that allocates without names must infer
these facts from access patterns. A compiler that allocates with names reads
them from the source.

**Kernel fusion** reads axis names to decide which operations can be merged.
Two elementwise operations on `x[b, t, d]` can fuse into one pass over `[b, t,
d]`. A reduction over `d` cannot fuse with an operation that preserves `d`,
because the coordinate disappears. The names draw the boundary: operations that
share surviving coordinates can fuse; operations across a reduction boundary
cannot. The compiler does not need to compute this from dataflow graphs. It
reads it from the same coordinate audit the programmer learns in Chapter 2.

This is the design principle the compiler is built on. Every pass consumes a
coordinate fact that the source makes explicit. No pass must reconstruct a
coordinate fact from shapes, access patterns, or tracing. The source says it;
the compiler reads it; the pass obeys it.

The practical consequence is flexibility. Because the compiler reads coordinate
names and not positional shapes, the execution strategy can change while the
coordinate contract remains stable. The same program `let y[b, c] = sum[i](x[b,
i] * W[c, i]) + bias[c]` can run on a NumPy eager backend, an IREE compiled
backend, or a future scheduler that fuses the sum and the broadcast into a
single kernel. The backends differ. The coordinate audit is constant. That is
the design separation that names enable: the notation records *what* the
coordinates do; the compiler chooses *how* to execute them.

The Hiding Law is not just a rule the compiler enforces on programmers. The
compiler's own passes obey it. Shape inference reads axis names to check legality.
Gradient lowering reads them to build the backward pass. Storage planning reads
them to decide what lives in memory. Each pass is a reader. When a pass hides a
fact from the next pass, the compiler breaks for the same reason a program breaks
when it hides a coordinate role from the next reader.

Because the names are load-bearing, change stays additive. Insert a head
dimension. The new name `h` appears in the projection that creates it and the
concatenation that consumes it. Softmax still normalizes over `j`. Batch is
untouched. Nothing reindexes. The old names did not move because they were never
tied to a position.

## The Deletion Test

Try this on any tensor program. Remove every axis comment, every diagram arrow,
and every prose reminder that says "this dimension is time" or "this one is
class." What facts would the program still state?

The notation is valuable for the facts that survive that deletion.

Take a line from Chapter 2:

```rust
let y[b, out] = sum[in](x[b, in] * W[out, in]) + bias[out];
```

Delete comments. Delete axis names from documentation. Delete the mental note
that `b` is batch. What remains? The source itself says that `b` survives, that
`out` survives, that `in` is consumed, and that the bias omits `b`. Those facts
are in the text, not in the margin.

Now try the same deletion on a positional equivalent. `y = x @ W.T + bias`.
Delete the comment that says `dim=0` is batch and `dim=1` is feature. What
remains? Shape compatibility. Nothing about which axis means what.

The Deletion Test is not a purity ritual. It is a stress test for the thesis
of this book. Notation determines what you can notice. If deleting everything
outside the source leaves you with nothing about which axis means what, then
the notation has determined that you cannot notice a class of bugs. The bugs
still exist. They just became invisible to every automated check between the
keystroke and the production outage.

## The Productive Cost

Named coordinates take more characters:

```rust
output[b, i, j] = sum[k](a[b, i, k] * b[b, k, j])
```

This is longer than `matmul(a, b)`. It also states more: the batch prefix
carries, `k` is consumed, `i` and `j` survive. The extra words earn their keep
when the relationship matters—when the code is being taught, differentiated,
optimized, or debugged.

The extra words are less useful when the operation is already obvious and
stable. A good system provides both: library functions for common cases,
explicit forms for the places where meaning would otherwise disappear. The
notation should be a lens, not a decoration.

## Functions That Preserve the Contract

A function call can hide useful things or dangerous things. This function is
fine:

```rust
fn relu_scalar(x: f32) -> f32 { if x > 0.0 { x } else { 0.0 } }
let y[b, f] = relu_scalar(x[b, f]);
```

It hides scalar control flow, not coordinate structure. The caller still says
that `y[b, f]` reads `x[b, f]`.

This function is dangerous:

```rust
let y = normalize(x);
```

If correctness depends on whether normalization consumes `feature`, `time`, or
`batch`, the abstraction has hidden the important fact. The fix is not "never
use functions." It is to make the function boundary carry the coordinate
contract:

```rust
let p[b, class] = softmax[class](logits[b, class]);
let pred = argmax[class](p);
```

The bracketed coordinate is part of the call's meaning. The function hides a
stable implementation while exposing the coordinate fact that review, autodiff,
and lowering must preserve. That is the abstraction boundary for tensor
programs: hide scalar mechanics, hide stable implementation choices, do not
hide the coordinate role that decides correctness.

## The Placement Rule

The useful split is not "Einlang versus everything else." It is a placement
rule:

```text
ecosystem work        host language (data loading, tokenizers, plotting)
tensor relationships  Einlang (equations, reductions, derivatives)
lowering and storage  compiler/runtime (scheduling, checkpointing, kernels)
```

This split is not a retreat. It is how the small core stays sharp. A language
that tries to own every task must give many constructs weaker meanings. A
focused notation can give a few constructs stronger meanings.

Data loading, plotting, tokenizers, file formats, and orchestration belong
comfortably in a host language. Tensor equations, reductions, derivative
requests, and recurrences are the part Einlang tries to make unusually visible.
Scheduling, checkpointing, materialization, and kernel selection belong to the
compiler and runtime once the source has stated the relationship.

Why draw the line here and not somewhere else? Because the line answers one
question: whose job is it to remember this fact? Facts about what to compute
and which coordinates carry meaning are the programmer's job — they are the
program. Facts about how to schedule and where to place are the compiler's job —
they are the optimization. When the two mix in the same syntax, the reader
cannot tell whether a transpose is architecture or accident. The Placement Rule
is the Hiding Law applied to the language design itself: do not hide the facts
the programmer must reason about behind the facts the compiler must optimize.

## The Portable Habit

The portable habit is small enough to carry outside Einlang. When reading
tensor code in any framework, ask:

```text
Which coordinates survive?
Which coordinates are consumed?
Which coordinates are omitted from a term?
Which coordinates are packed or unpacked?
Which coordinate names time?
```

These questions do not require a new language. They are questions about the
program you already have. Einlang explores what happens when those questions
are not only in the reader's head but in the source notation itself. You can
disagree with the syntax and still keep the demand: when an axis carries
meaning, do not make the next reader recover that meaning from position alone.

## One Layer, All Contracts

Return to one ordinary layer. It looks familiar enough to be dismissed, but it
contains nearly every contract the book has been building toward:

```rust
let z[b, out] = sum[in](x[b, in] * W[out, in]) + bias[out];
let y[b, out] = relu(z[b, out]);
let loss = sum[b, out](y[b, out]);
```

Read it as something the compiler can track.

`b` survives. Examples do not mix. `out` survives because the layer produces
output features. `in` is local to the sum. The bias omits `b`, so it is shared
across examples. `relu` is elementwise—it does not mix batch items or features.
The loss consumes both `b` and `out` to produce a scalar.

Now ask for gradients:

```rust
let dloss_dW = @loss / @W;
let dloss_dbias = @loss / @bias;
let dloss_dx = @loss / @x;
```

The denominator of each request tells the address of the answer:

```text
dloss_dW[out, in]       collects b
dloss_dbias[out]        collects b
dloss_dx[b, in]         collects out
```

The global shape of the backward pass is visible from the forward coordinates
alone. No Jacobian was materialized. No execution plan was forced. The source
simply kept the coordinate roles visible long enough for the reader and the
compiler to ask the right next question.

A positional equivalent may be shorter:

```python
y = relu(x @ W.T + bias)
loss = y.sum()
```

That code is familiar and useful. The visible-index version is not claiming
every production program must expand every layer forever. Its claim is that
when the coordinate choice matters—when a bug depends on whether `bias` was
shared across batch, or whether `W` maps input features into output features
rather than the reverse—the source should have a form that can show it.

The Hiding Law does not say "never use library functions." It says "do not
hide a fact that later reasoning must recover." A library function that hides
scalar mechanics is fine. A library function that hides the coordinate being
normalized is not. The test is not the length of the expression. It is the
distance between the source and the fact that a future debugging session will
need. When that distance is zero, the notation has done its job.

## The Attention Test

Attention is the hardest stress test because shape can be friendly while the
communication pattern is broken. The visible-index version keeps roles on the
line:

```rust
let scores[b, h, i, j] = sum[d](Q[b, h, i, d] * K[b, h, j, d]);
let context[b, h, i, d] = sum[j](weights[b, h, i, j] * V[b, h, j, d]);
let out[b, i, o] = sum[h, d](context[b, h, i, d] * W_out[h, d, o]);
```

Three lines, one contract. `b` preserves examples. `h` separates heads until
the output projection consumes it. `i` is the query position that survives. `j`
is the key/value position that is scanned. `d` is a local feature coordinate.
`o` is the produced output feature.

The broken version is one character:

```text
context[b, h, i, d] = sum[j](weights[b, h, i, j] * V[b, h, i, d])
```

Same result shape. Wrong: the value no longer comes from the position selected
by `j`. The weights scan keys, but the gathered value is fixed at the query. A
compiler that sees only ranks may have nothing useful to say. A compiler that
sees named coordinates can report: the term under `sum[j]` does not read the
value at `j`.

Imagine the error message five years after this style has become ordinary:

```text
attention.ein:89: reverse-mode coordinate conflict
  value is read at coordinate i, but weights are distributed over coordinate j.
  Did you mean V[b, h, j, d]?
```

The exact wording is fiction. The possibility is not. Such an error needs
source-level roles: which coordinate was reduced, which was read, and where the
two facts failed to meet. If all the compiler has is a pair of shape tuples,
the best message is much poorer.

We are now one chapter from the book's end, and the question has not changed
since Chapter 1: when the notation has no place for a fact, that fact becomes
invisible — not just to the compiler, but to the act of reading and reasoning
itself. The Hiding Law is the rule for deciding which facts to rescue. The
Deletion Test is the procedure for checking whether they were rescued. The
Placement Rule is the map for knowing where to put them. Every chapter has
been a different setting for the same three tools.

## The Last Question

When you see tensor code now, ask:

```text
Which dimension has a name only in my head?
```

That question is the seed. Sometimes the answer will not matter. Sometimes the
shape convention is enough. But when the answer does matter, a visible
coordinate can turn a hidden assumption into a local fact.

The bargain is not that explicit indices are always shorter. They are not. It
is not that every tensor operation benefits from hand expansion. It often does
not. The bargain is that when a dimension role matters, the source has a way
to say so directly.

A practical checklist:

```text
Would a wrong axis still have the right shape?
Would a future reader need a comment to know what this dimension means?
Would a derivative or recurrence depend on this coordinate role?
Would a compiler optimization need to preserve this relationship?
```

If the answer is yes, visible coordinates earn their cost. If no, a library
call may be the clearer expression.

## Try It

Run the Deletion Test on this line from Chapter 9:

```rust
let z[batch, out] = sum[in](x[batch, in] * W[out, in]) + bias[out];
```

Delete every comment. Delete every mental note. What facts survive in the
source? The coordinate `bias` omits `batch` — you know this because `batch`
appears in the output coordinates `[batch, out]` and in the first term's
coordinates but is absent from `bias[out]`. The coordinate `in` is consumed by
the sum — you know this because `in` names the reduction bracket. If `bias`
accidentally had the coordinate `bias[batch, out]` instead of `bias[out]`, the
deletion-test reading would change: `batch` would no longer be omitted from
`bias`, meaning each batch element would get its own separate bias rather than
sharing a single vector. Now run the same deletion on the positional equivalent:

```python
z = x @ W.T + bias
```

Delete the comment that says `dim=0` is batch and `dim=1` is feature. What facts
survive? Shape compatibility. Nothing about which axis means what. Nothing about
which term omits which axis. The Deletion Test is not a formalism. It is a
practical question: how many facts about your program survive the deletion of
everything outside the source text? The named version keeps four facts (survival
of `batch`, survival of `out`, consumption of `in`, omission of `batch` from
`bias`). The positional version keeps zero.

Now consider a colleague who proposes a new DSL convention: every tensor carries
a string tag for each dimension. The tags are not used for type checking — they
are for documentation. You can write `x[batch, feature]` with named coordinates
and the reader sees the roles, but the compiler ignores them — the names are
documentation, not contracts. Does this pass the Deletion Test? The names are in
the source, not in comments. If you delete all comments, the names survive. But
do the names actually constrain anything? If you write `x[batch, feature]` and
then compute `softmax[batch](x)`, the compiler accepts it because the coordinate
names are tags, not contracts. Nothing checks that `batch` in `softmax[batch]`
matches a declared coordinate on `x`.

The minimum change to make the tags into contracts is that the compiler must
check that a coordinate named in a reduction bracket matches a coordinate
declared on the operand. `softmax[batch](x[batch, feature])` compiles.
`softmax[class](x[batch, feature])` produces an error. The check is: does the
normalization coordinate appear in the operand's coordinate set? If not, the
compiler reports a mismatch. A coordinate contract system must enforce several
rules:

```text
- Reduction must consume a coordinate present in the operand
- Omission (broadcast) must omit a coordinate present in the output
- Reshape must account for all coordinates in the source and all in the target
- Gradient must preserve the denominator's coordinate set
- Recurrence must read a smaller time index than the value being defined
```

Each rule is a local check on coordinate names. The compiler does not need to
understand the semantics of `batch` versus `class`. It only needs to verify that
the named coordinates appear where the operation says they should. The check
that would catch the most bugs in a 50,000-line codebase is reduction consuming
the wrong coordinate — `softmax[batch]` instead of `softmax[class]`. This is a
one-character bug that produces a shape-identical result and descends the loss.
The coordinate check catches it before training.

Next, write a Deletion Test report for one layer from your own codebase. The
report has four sections.

Section 1 asks what the source says. Pick one tensor layer — a convolution, an
attention block, an RNN cell, a normalization — and write it in named
coordinates:

```rust
let y[b, out, h, w] =
    sum[in, kh, kw](x[b, in, h+kh, w+kw] * W[out, in, kh, kw]) + bias[out];
```

List every fact this formula states without comments: coordinates that survive
(`b, out, h, w`), coordinates that are consumed (`in, kh, kw`), coordinates
omitted from a term (`b` omitted from `bias`), and coordinates mapped (`h+kh,
w+kw` in the input).

Section 2 asks what the positional version says. Write the same layer in a
positional framework:

```python
y = F.conv2d(x, W, bias, stride=1, padding=0)
```

List every fact this formula states without comments. The shape of `x` is
`[batch, in, h, w]`. The shape of `W` is `[out, in, kh, kw]`. The shape of
`bias` is `[out]`. The shapes are in the tensors. But what the tensors do not
say: whether `bias[out]` is shared across batch (yes, because it lacks a batch
dimension), whether `in` is consumed (yes, by the convolution), whether the
stride is 1 (yes, in the argument), and whether the padding is zero (yes, in
the argument).

Section 3 catalogs the facts that survive deletion. Go through each fact from
Section 1 and ask whether it survives in Section 2 after deleting all comments:

```text
Fact                          Named source   Positional source (no comments)
─────────────────────────────────────────────────────────────────────────────
b survives                    yes            rank of output = 4, but which dim?
out survives                  yes            rank of output = 4, but which dim?
h, w survive                  yes            spatial dims positions 2,3
in consumed                   yes            gone in output shape
kh, kw consumed               yes            gone in output shape
b omitted from bias           yes            bias rank = 1, but which role?
bias shared across batch      yes            implicit from rank difference
```

The positional source keeps shape facts (rank, extent) and loses role facts
(which coordinate is batch, which coordinate is class, which coordinate was
omitted from which term).

Section 4 identifies the fact that costs the most to recover. Among the facts
lost in the positional version, which one cost you or a colleague the most
debugging time? Write the story in three sentences: what the code looked like,
what was wrong, and what fact would have prevented the bug if it had survived.
For example: "The code computed `softmax(dim=1)` instead of `softmax(dim=2)`.
Both produced tensors of shape `[B, T, C]`. The loss descended. For three days
the model learned with normalized time steps instead of normalized class
probabilities. If the source had said `softmax[class]` and the compiler had
checked that `class` was present, the bug would have been a compile error
instead of a silent semantics change."

The named formula is longer, but each extra character states a fact the
positional version buries. The coordinate names distinguish `b` from `out` from
`in` — roles that a shape tuple merges into integers. The Deletion Test
separates implementation facts (kernel size, stride) from semantic facts (which
axis is reduced) and shows that only the latter need source-level names. The
report format gives you a procedure to apply to any layer in any codebase,
regardless of framework.

**Line to keep:** do not hide facts that future reasoning must recover.

### Where This Leads

This chapter made the hiding law explicit. Coordinate roles, consumed axes,
derivative addresses, recurrence edges—these pass the test. Register allocation,
tiling, fusion order—these earn the right to stay hidden.

But the test has a harder edge case: dynamic routing. When the path a value
takes depends on the data itself—not just the coordinate structure—what happens
to the coordinate contract? Can the compiler still check that roles are
preserved when the route is only known at runtime?

Chapter 16 is the book's last stress test: dynamic routing with low-rank
communication. It asks whether named coordinates can survive a world where
values choose their own paths.