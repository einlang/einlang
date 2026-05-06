---
layout: book
title: "What the Notation Refuses to Hide"
---

# What the Notation Refuses to Hide

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
later reasoning had to recover. That is the line this chapter draws.

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

What follows is the book's metacircular moment — the point where we stop
applying the rule and examine the rule itself, by walking backward through every
chapter to see what each one taught us about hiding.

The fourteen chapters did not simply repeat the same lesson in different
settings. They *deepened* it. Each Part revealed something about hiding that the
previous Part could not see.

**Part I (Chapters 1–6): Hiding is possible.** The surface lesson. A reshape
chain can preserve element counts while erasing the coordinate story. A
broadcast can mask the fact that a value is shared across a role. A `dim=-1` can
hide which of three coordinate relationships — survive, consume, omit — is in
play. Part I taught us to *notice* hiding. The question was: what facts does the
notation let slip? The answer, at this stage, was: many. And the loss still
descends.

Chapter 1 showed that a reshape chain hides the packing relation. Chapter 2
showed that a position hides the role. Chapter 3 showed that a coordinate map
written as a sequence of positional operations hides the address equation.
Chapter 4 showed that broadcasting hides the claim of independence. Chapters 5
and 6 showed that a reduction and a normalization each hide which coordinate was
consumed — and that three distinct coordinate relationships can hide behind a
single integer argument.

By the end of Part I, the coordinate audit — survive, consume, omit — was a
reading discipline. But the discipline was static. We could audit a single
expression. We had not yet asked what happens when one expression feeds another,
or when the autodiff engine traces backward through them, or when values depend
on earlier versions of themselves.

**Part II (Chapters 7–9): Hiding has consequences.** The first deepening. Part I
was about what the forward pass states. Part II revealed that the forward pass
also *implies* — it implies a gradient structure. And whether that structure is
visible depends on whether the coordinate names survive the backward pass. The
autodiff engine is a reader too. If the forward notation hid a sharing decision,
the backward pass cannot reveal it — it can only reproduce the ambiguity as a
gradient that silently sums over the wrong coordinate.

Chapter 7 showed that a gradient denominator is an address: `@loss/@W` means
"collect sensitivity at W's coordinates." Chapter 8 showed that the transpose
practitioners memorize is coordinate alignment, not an axiom of calculus — the
sum coordinate in a pullback is set subtraction on coordinate sets. Chapter 9
showed that local scalar derivatives and global gradient shapes are separated by
one fact: which coordinates the value was shared across. The scalar rule is
always the same. The shape rule is always an invoice. The invoice is legible
only if the coordinates are named.

By the end of Part II, we understood that hiding is not just a readability
problem. It is a *correctness* problem. A hidden coordinate in the forward pass
becomes a silent gradient bug in the backward pass. But all our programs were
still static DAGs. We had not yet introduced a coordinate whose values depend on
earlier values of that same coordinate.

**Part III (Chapters 10–12): Hiding compounds over time.** The second
deepening. Time is the first coordinate with inherent direction — `t` reads
`t-1`, never `t+1`. A loop hides that direction in mutable state. The same
variable `h` that reads `h[t-1]` also writes `h[t]`. The dependency edge — who
reads whom — is invisible in the loop but explicit in the recurrence. And when
time direction is hidden, three things break at once: the reviewer cannot verify
causality, the compiler cannot plan storage, and the autodiff engine cannot
schedule the backward pass.

Chapter 10 showed that a loop is an execution order; time is a dependency
relationship. Chapter 11 showed that storage is a negotiation between what is
defined, what is observed, and what can be recomputed — and the compiler cannot
optimize storage it cannot observe. Chapter 12 showed that an RNN is not a loop
calling a cell; it is a communication graph made of named coordinates, with
edges labeled by time direction, hidden-unit role, and batch isolation.

By the end of Part III, we understood that hiding compounds. In a recurrence,
hiding the time direction does not just hide one fact — it hides the distinction
between a program that can be optimized (causal, storable, reversible) and a
program that can only be executed. But all our recurrences, all our gradients,
all our audits had been performed on single operations in isolation. We had not
yet asked what happens when named coordinates cross a module boundary.

**Part IV (Chapters 13–14): Hiding scales.** The third deepening. Parts I–III
worked one operation at a time. A reshape. A softmax. A matmul pullback. A
recurrence. Part IV asked: can the same coordinate vocabulary survive *system*
scale? When an encoder emits `output[time, batch, feature]` and a decoder
expects `input[time, batch, unit]`, the coordinates `feature` and `unit` must
meet. If the notation cannot name that transition, the compiler cannot check it.
A mismatch of `768` and `512` crashes at runtime. A mismatch of `feature` and
`unit` could be caught at compile time — if the names survived the boundary.

Chapter 13 showed that named dimensions at module edges turn silent semantic
mismatches into compiler errors. Chapter 14 showed that attention is not matrix
multiplication — it is a communication protocol, and the notation either names
the protocol or buries it in a transpose. Seven conventions must be held in the
programmer's head to read a positional multi-head attention block correctly.
Named coordinates make each convention a fact the compiler can check.

**Chapter 15: The law behind the deepenings.** Part I showed that hiding is
possible. Part II showed that hiding has consequences. Part III showed that
hiding compounds. Part IV showed that hiding scales. Each Part was necessary.
You cannot understand the consequences of hiding (Part II) until you can notice
hiding (Part I). You cannot understand how hiding compounds (Part III) until you
understand its consequences for a single backward pass (Part II). You cannot
understand how hiding scales (Part IV) until you understand what happens when it
compounds (Part III) and what its consequences are (Part II) and what it looks
like in the first place (Part I).

Every chapter was a corollary of one sentence: *Do not hide a fact that later
reasoning must recover.* But the fourteen chapters were not fourteen repetitions
of the same insight. They were fourteen steps up a single staircase. At the
bottom: a reshape that accepted an integer where it should have demanded a name.
At the top: a multi-head attention block whose communication protocol is either
a fact in the source or a memory in the reader's head. The distance between
those two is the distance this book was built to cross.

This is not a rule about software. It is a rule about cognition. Every notation
is a prosthetic memory — it carries facts so your brain does not have to. A good
notation carries the facts that later reasoning will need. A bad one carries the
facts that are easy to encode, and leaves the hard ones to your memory. The
difference between them is not the notation's power. It is the notation's
honesty about what it refuses to record.

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

And here is the turn. The Hiding Law is not a rule the compiler enforces on
programmers. It is a rule the compiler's own architecture obeys. The same
principle that tells the programmer "do not hide the coordinate role" tells the
compiler writer "do not hide the coordinate role from the next pass." The
compiler that enforces the Hiding Law is itself built on it.

This is not a coincidence. It is the Hiding Law at full depth: a single
principle that organizes both the notation the programmer writes and the
compiler that reads it. Every pass is a reader. Every reader needs the names.
The compiler is the final proof that the names are not decoration. They are the
one thing every pass agrees on — and the one thing no pass can afford to lose.

And because the names are load-bearing, change stays additive. Insert a head
dimension — the new name `h` appears in the projection that creates it and the
concatenation that consumes it. The softmax still normalizes over `j`. The batch
coordinate `b` is untouched. Nothing reindexes. Nothing shifts. The old names
did not move because they were never tied to a position in the first place.
Additive change is not a separate principle from the Hiding Law. It is the same
principle, viewed from the future: do not hide what reasoning must recover, and
the old facts will not need to be rewritten when the new facts arrive.

You have now seen the same pattern in every chapter of this book: a mathematical
operation stated in coordinates, written first in the positional style the
reader already knows, then in the named style that keeps the coordinate roles on
the page. The sum that consumes a coordinate. The derivative that collects at a
name. The recurrence that chains a value through successive steps. The rest
pattern that says "the rest" without counting. Each chapter taught one
mechanism. The mechanism was the same: name the coordinate, audit the roles, let
the notation carry what the reader would otherwise have to remember.

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