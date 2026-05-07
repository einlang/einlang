---
layout: book
title: "Introduction: The Shape-Meanings Gap"
---

# Introduction: The Shape-Meanings Gap

> "What we observe is not nature itself, but nature exposed to our method of
> questioning."
>
> — Werner Heisenberg, *Physics and Philosophy* (1958)

I first ran into this problem while reading someone else's tensor program. The
code ran. The shapes lined up. The loss went down. And still, the most
important facts about the program lived entirely outside it — in a comment, in a
variable name, in a convention about which axis position meant `time`, or
simply in the author's head.

Every line worked. But ask what would happen if someone upstream swapped two
dimensions of the same size. The program would continue to run. The loss would
continue to descend. The shapes would remain compatible. The bug would be
invisible to the compiler, the test suite, and the training dashboard. Only the
meaning would change.

I have spent years reading tensor code written by excellent engineers. The
single most common failure mode is not a shape error. It is a role error with a
compatible shape. The code ran. It was wrong. Notation that only records shape
gave the reader no place to notice.

Maybe you have been on the other side of this. You deploy the model on a
Tuesday. The metrics look exactly like they did in training. You go home. You
sleep. Wednesday morning your colleague messages: "the predictions are all
slightly wrong, but none of the shapes changed." You stare at the screen. She's
right. The tensor shapes are correct at every layer. The loss went down. The
eval metrics are normal. The bug is invisible to every tool you have.

You spend the morning tracing shapes. `(32, 128, 768)` becomes `(32, 128, 768)`
becomes `(32, 768)`. Good. The afternoon you trace meaning. Batch is 32. Time
is 128. Feature is 768. Also good. The evening you trace the unreported change.

Three weeks ago someone refactored the data pipeline to swap axis conventions.
Time moved from position 1 to position 2. Batch stayed at position 0. The
shapes didn't change because the extents happened to coincide — batch size was
32, time steps were 128, and another model had a 128-class vocabulary. The
reshape chain downstream accepted the numbers eagerly. The loss descended
because the model can learn any pattern you give it, including the pattern
produced by silently swapping time with class.

You found it at 11 PM. Not with a shape checker. Not with a type system. Not
with a debugger. You found it by printing a tensor, staring at the numbers, and
realizing that the quantity decreasing over "time" was actually the class
distribution of the previous example.

This is the shape-meanings gap. The shape says *how many*. The role says *which
one*. Every tensor framework knows the shape. None of them know the role. The
role lives in variable names, in comments, in the whiteboard diagram that
nobody photographed, in the convention that "batch is always dim 0," in the
memory of the person who wrote the code and left the team four months ago.

Notation is not a neutral container for thought. It is the machinery thought
runs on. Change the notation, and you change what you can think. The way we write
things down determines what we can notice, what we can check, and what we can
safely forget. When a notation omits a fact, that fact becomes harder to reason
about — not because the fact is hard, but because the notation offers no place to
put it.

This is the thesis, and I am going to say it once plainly: notation is a lens. What
it omits becomes invisible — not just to the compiler, but to anyone reading the
code. The rest of this book asks one question repeatedly. What happens when you
refuse to let the notation hide the coordinate role?

We close the gap by putting the role in the source where the shape already lives.
If the shape gets a position in the tensor, the role gets a name.

There is an older way of thinking about this.

Structure and Interpretation of Computer Programs opens with a claim that has
organized every language-design argument since. A powerful language, it says, rests
on primitive expressions, means of combination, and means of abstraction. Take away any one and you have something less than a
language. I read that as an undergraduate and did not understand how deep it went
until I tried to design one.

Tensor programs have all three, but the primitive is wrong. The atom of a
positional tensor program is an integer — the axis position. The means of
combination are reshape, transpose, `einsum`, and `dim=` arguments. The machinery
works. But an axis position is a number the compiler can count. It cannot carry
the reason the number matters. That reason lives in variable names, in comments,
in the memory of whoever wrote the code. It lives anywhere except in the notation
itself. And because the primitive is silent, the combinations inherit the silence
and the abstractions paper over it.

This book is about what happens when you fix the primitive. The axis name replaces
the axis position. The coordinate audit — survive, consume, omit — becomes the
first thing you learn, because it is the first thing the notation can check.
Reduction, broadcast, recurrence leave a trace in the audit. Coordinate-aware
functions carry the names across boundaries.

## The Gap in One Example

Take an ordinary line of code. You have written it a hundred times:

```python
x = x.reshape(batch, time, feature)
y = torch.softmax(scores, dim=-1)
loss = loss_fn(y, target)
```

Three lines. Every line runs. Every shape is compatible. Now ask the questions
that a shape checker cannot answer:

Which dimension did the reshape preserve the *role* of, as opposed to the *size*
of? Which axis was normalized by the softmax — and would the program still run
if you normalized a different axis by mistake? What would happen if someone
upstream swapped `batch` and `time`, and the two happened to have the same size?

None of these are exotic. They are the ordinary facts that decide whether a
training loop is converging for the right reason, or whether a production model
is silently computing the wrong statistic. The shapes are right. The roles are
invisible.

Now write the same program with the role in the source:

```rust
let image[batch, channel, row, col] = load_batch();
let logits[batch, class] = model(image);
let probs[batch, class] = softmax[class](logits[batch, class]);
let loss = cross_entropy[class](probs, target);
```

Each bracketed name is a coordinate argument. `softmax[class]` says the class
coordinate is the one being normalized. `cross_entropy[class]` says the loss is
computed over the class distribution. The batch coordinate survives every line
without being mentioned in a bracket — it is a survivor, not a decision. If
someone upstream swaps `batch` and `class`, the compiler sees the mismatch
before the first forward pass.

This is not a type system in the traditional sense. The types are still `f32`
and `[f32; batch, class]`. The shape system still knows extents. The difference
is that the coordinate roles are part of the source, which means they are part
of what the compiler can check, the autodiff engine can preserve, and the next
reader can see without reconstructing them from comments, conventions, and
variable names.

## The Instrument

Einlang is the small language used for the examples. It is a microscope, not a
production framework. It has just enough syntax to write tensor formulas with
named coordinates, reductions, derivatives, and recurrences — and just enough
compiler machinery to show what those formulas preserve.

It is not a replacement for PyTorch, JAX, NumPy, or your existing compiler
stack. It is a deliberately narrow object. The question is not "should you
rewrite everything in Einlang?" but "what facts would still be visible if you
could?"

The full language is small enough to read in one page:

```rust
let y[i] = x[i] + 1;                              // elementwise
let C[i, j] = sum[k](A[i, k] * B[k, j]);          // matmul
let p[b, class] = softmax[class](logits[b, class]); // normalize
let pred = argmax[class](p);                       // argmax
let image[channel, row, col] = load_image();       // named channels
let channels_last = move_channel[channel](image);  // rearrange
let dy_dx = @y / @x;                               // derivative
let h[t in 1..T] = step(h[t-1], x[t]);            // recurrence
```

The bracketed names in `softmax[class]`, `argmax[class]`, and
`move_channel[channel]` are coordinate arguments — the one role the caller must
choose. The bracketed name is not decoration. It is the contract.

Throughout this book, code blocks marked `rust` are Einlang source. Blocks
marked `text` are coordinate readings, dependency edges, or diagnostic tables —
they are not meant to be parsed. Einlang has scalar and tensor types such as
`i32`, `f32`, `[f32; 3, 4]`, `[f32; ?, ?]`, and `[f32; *]`. You do not need to
memorize the compiler path behind the examples. You only need to trust that when
the book says "the compiler can check this," the machinery exists to check it.

Einlang inherits from Einstein notation (index names as self-documentation),
named tensors in PyTorch and xarray (labels for alignment), Einops (readable
rearrangement), shape-annotation systems (value at function boundaries), and
compiler stacks from Tullio to Halide (scheduling below the source). The
hypothesis: if coordinate roles live in the source rather than documentation,
the compiler can check them.

## Who This Is For

This book is for people who build things below the level of an API call — a
compiler pass, a numerical library, an autodiff engine, a tensor DSL, or a
notation shared by a team. If you mainly want another framework function to
call, this book will probably spend too much time under the floorboards. That
is by design.

If you have ever stared at `RuntimeError: mat1 and mat2 shapes cannot be
multiplied` at 3 AM and wished the traceback told you which dimension was
supposed to be `head` and which one was supposed to be `feature`, this book is
written for you.

## A First Encounter

Enough preface. You learn a notation the way you learn a musical instrument — by
making sounds, not by reading about acoustics. This section is a twenty-minute
encounter with Einlang. We will write a small classifier, trace one prediction
cell through every line, and watch the coordinate roles survive, disappear, and
announce themselves in brackets. No theory yet. Just the program and the habit.

### One Layer, with Names

Start with the simplest linear layer you can write. Input features go in.
Output features come out. The weight matrix sits between them:

```rust
let y[out] = sum[in](x[in] * W[out, in]);
```

Read this aloud. "Let `y` at coordinate `out` be the sum over `in` of `x` at
`in` times `W` at `out, in`." The output coordinate is `out`. The reduction
coordinate is `in` — it appears in `x` and `W`, but not in `y`. Every `y[out]`
is one number, built from all the `in` positions. 

Try reading it aloud yourself, once. The rhythm of "sum over in" matching
"in disappears from the result" is not an accident. The notation is designed so
that saying it and understanding it are the same act.

Pick a concrete size. `in` ranges over 4 features. `out` ranges over 3 classes.
Give `x` some numbers and `W` some numbers:

```text
x[0] = 2.0    x[1] = 1.0    x[2] = 0.0    x[3] = 3.0

W[0, 0] = 0.5   W[0, 1] = 0.0   W[0, 2] = 0.2   W[0, 3] = 0.1
W[1, 0] = 0.0   W[1, 1] = 0.3   W[1, 2] = 0.0   W[1, 3] = 0.4
W[2, 0] = 0.1   W[2, 1] = 0.0   W[2, 2] = 0.5   W[2, 3] = 0.0
```

Trace `y[0]`:

```text
y[0] = sum[in](x[in] * W[0, in])
     = x[0]*W[0,0] + x[1]*W[0,1] + x[2]*W[0,2] + x[3]*W[0,3]
     = 2.0*0.5    + 1.0*0.0    + 0.0*0.2    + 3.0*0.1
     = 1.0        + 0.0        + 0.0        + 0.3
     = 1.3
```

Trace `y[1]`:

```text
y[1] = x[0]*W[1,0] + x[1]*W[1,1] + x[2]*W[1,2] + x[3]*W[1,3]
     = 2.0*0.0      + 1.0*0.3      + 0.0*0.0      + 3.0*0.4
     = 0.0          + 0.3          + 0.0          + 1.2
     = 1.5
```

Trace `y[2]`:

```text
y[2] = x[0]*W[2,0] + x[1]*W[2,1] + x[2]*W[2,2] + x[3]*W[2,3]
     = 2.0*0.1      + 1.0*0.0      + 0.0*0.5      + 3.0*0.0
     = 0.2          + 0.0          + 0.0          + 0.0
     = 0.2
```

The result: `y = [1.3, 1.5, 0.2]`. Class 1 has the highest score. The model
predicts class 1.

Already the coordinate names have done work. `in` is consumed by `sum[in]` —
you can see this because `in` appears inside the sum bracket but not on the
left-hand side. `out` survives — it appears on both sides. If someone wrote
`sum[out]` instead, the compiler would object: the result `y[out]` needs `out`
to survive, but the sum is trying to consume it. The names catch the mistake
before a single number is computed.

You have not yet left the first formula, and already the coordinate audit —
survive, consume, omit — is answering the question that shape checkers cannot
ask.

### Adding a Bias

A bias term is shared across all output classes:

```rust
let y[out] = sum[in](x[in] * W[out, in]) + bias[out];
```

Add concrete bias values:

```text
bias[0] = 0.1    bias[1] = -0.2    bias[2] = 0.3
```

Now retrace `y[1]`:

```text
y[1] = sum[in](x[in] * W[1, in]) + bias[1]
     = 1.5 + (-0.2)
     = 1.3
```

The bias term `bias[out]` mentions `out` but omits `in`. That omission is a
statement: `bias[out]` does not depend on `in`. The same bias value serves
every input position. In a positional framework, this is called broadcasting.
In named coordinates, it is simply the absence of a name.

### Adding a Batch

Real programs process many examples at once. Add a batch coordinate:

```rust
let y[batch, out] = sum[in](x[batch, in] * W[out, in]) + bias[out];
```

Now `x` varies across `batch` and `in`. `W` varies across `out` and `in`. The
bias varies only across `out`. The batch coordinate appears in `x` and in `y`,
but not in `W` and not in `bias`. That means every batch example shares the
same weight matrix and the same bias vector. The names state this directly.

Pick a second batch example:

```text
Batch 0: x[0, 0]=2.0  x[0, 1]=1.0  x[0, 2]=0.0  x[0, 3]=3.0  (the one we traced)
Batch 1: x[1, 0]=0.0  x[1, 1]=2.0  x[1, 2]=1.0  x[1, 3]=0.0
```

Trace `y[1, 1]` (batch 1, class 1):

```text
y[1, 1] = sum[in](x[1, in] * W[1, in]) + bias[1]
        = x[1,0]*W[1,0] + x[1,1]*W[1,1] + x[1,2]*W[1,2] + x[1,3]*W[1,3] + bias[1]
        = 0.0*0.0       + 2.0*0.3       + 1.0*0.0       + 0.0*0.4       + (-0.2)
        = 0.0           + 0.6           + 0.0           + 0.0           + (-0.2)
        = 0.4
```

The two batch examples computed `y[0, 1] = 1.5` and `y[1, 1] = 0.4`. Same `W`,
same `bias`, different `x`. The batch coordinate isolates examples. No batch
item reads another's data. The names make this visible: `batch` appears next to
`x` and `y`, but not next to `W` or `bias`.

### Softmax: Naming the Normalized Coordinate

A linear layer produces logits. To get class probabilities, apply softmax:

```rust
let probs[batch, class] = softmax[class](y[batch, class]);
```

The bracket names the coordinate being normalized. For each batch example, the
class probabilities sum to 1. The batch coordinate is not normalized — each
example gets its own distribution.

Trace `probs[0, :]` from `y[0] = [1.3, 1.5, 0.2]`:

```text
First, stabilize with the max (class 1 has 1.5):
y_stable = [1.3 - 1.5, 1.5 - 1.5, 0.2 - 1.5]
         = [-0.2, 0.0, -1.3]

Exponentiate:
exp_y = [exp(-0.2), exp(0.0), exp(-1.3)]
      = [0.819, 1.0, 0.273]

Sum (the denominator):
z = 0.819 + 1.0 + 0.273 = 2.092

Normalize:
probs[0, 0] = 0.819 / 2.092 = 0.391
probs[0, 1] = 1.0   / 2.092 = 0.478
probs[0, 2] = 0.273 / 2.092 = 0.130
```

Sum of probs for batch 0: 0.391 + 0.478 + 0.130 = 0.999 (rounding). Class 1
still has the highest probability.

Now the deliberate mistake. Write `softmax[batch]` instead:

```rust
let bad[batch, class] = softmax[batch](y[batch, class]);
```

The shapes are identical. Both produce `[batch, class]`. But now the
normalization runs over the wrong coordinate — the three class scores compete
across batch examples rather than within each example. If `batch_count ==
class_count`, the shapes are square and no runtime error occurs. The loss still
descends. The model learns a different thing.

The bracket `[class]` is the contract. It says which coordinate is the
distribution. A compiler that sees `softmax[batch]` where the data carries
`[batch, class]` can report: "normalizing over batch, but class is the
coordinate with multiple entries per example. Did you mean `softmax[class]`?"

### Argmax: Naming the Selection Coordinate

After softmax, take the most probable class:

```rust
let pred = argmax[class](probs[batch, class]);
```

`argmax[class]` consumes the `class` coordinate and returns a scalar per batch
item — the index of the maximum class. For batch 0, class 1 had the highest
probability (0.478), so `pred[0] = 1`.

The bracket `[class]` tells the compiler two things: which coordinate to scan
for the maximum, and which coordinate disappears from the result. If you write
`argmax[batch]` instead, the compiler can report that `batch` still appears in
the result shape — `argmax` reduces over its bracketed coordinate, but `batch`
survives.

### The First Derivative

Add a loss and ask for a gradient:

```rust
let loss = cross_entropy[class](probs[batch, class], target[batch]);
let dloss_dW = @loss / @W;
```

The denominator `@W` says the gradient lives at `W`'s coordinates: `[out, in]`.
The compiler knows this before computing a single number. It reads `W`'s
coordinate set from its own bookkeeping, reads the forward expression to find
which output coordinates `W` influences, and computes the path coordinates —
those in the output but not in `W`. The gradient formula follows mechanically.

For the linear layer `y[batch, out] = sum[in](x[batch, in] * W[out, in]) +
bias[out]`, the gradient of `W` is:

```rust
let dW[out, in] = sum[batch](dloss_dy[batch, out] * x[batch, in]);
```

The sum over `batch` appears because `W[out, in]` was used once per batch
example — one weight cell contributed to `y[0, out]`, `y[1, out]`, `y[2,
out]`, and so on. When sensitivity arrives at those output cells, all routes
lead back to the same `W[out, in]`. The gradient must sum them. The coordinate
`batch` is not part of `W`'s address, so it becomes a reduction in the
gradient.

Trace one cell: `dW[1, 2]`. This weight multiplies `x[*, 2]` to contribute to
`y[*, 1]`. Both batch examples use it:

```text
dW[1, 2] = dloss_dy[0, 1] * x[0, 2] + dloss_dy[1, 1] * x[1, 2]
```

If `dloss_dy = [[0.1, -0.3, 0.2], [0.0, 0.4, -0.1]]`:

```text
dW[1, 2] = (-0.3) * 0.0 + 0.4 * 1.0 = 0.0 + 0.4 = 0.4
```

The gradient shape `[out, in]` matches `W`'s shape. The coordinate accounting
derived the `sum[batch]` without memorization. It is set subtraction:
`{batch, out}` minus `{out, in}` equals `{batch}`.

### The Four Questions, Applied

Every encounter in this book uses the same four questions. Apply them to the
linear layer one more time:

```text
For the expression: y[batch, out] = sum[in](x[batch, in] * W[out, in]) + bias[out]

1. Which coordinates survive?
   → batch, out. They appear on the left-hand side.

2. Which coordinates are consumed?
   → in. It appears inside sum[in] and is absent from y.

3. Which coordinates are omitted from a term?
   → batch is omitted from bias[out]. The same bias serves every example.
   → batch and in are omitted from W[out, in]. W is shared across all examples
     and reused for every input position.

4. Which coordinate is the address of the gradient?
   → @loss/@W lives at [out, in] — W's coordinates.
   → @loss/@x lives at [batch, in] — x's coordinates.
   → @loss/@bias lives at [out] — bias's coordinates.
```

These four questions do not change when the example grows. You will ask them
about broadcasting. About softmax. About matmul pullbacks. About RNNs. About
multi-head attention. The questions stay the same because the coordinate
structure stays the same — only the number of coordinates changes.

### A Tiny Recurrence

Before leaving this encounter, meet time as a coordinate:

```rust
let fib[0] = 0;
let fib[1] = 1;
let fib[n in 2..10] = fib[n - 1] + fib[n - 2];
```

The coordinate `n` is time. Each `fib[n]` reads `fib[n-1]` and `fib[n-2]` —
two backward edges. The index expression states the direction: `n - 1` and `n -
2` are smaller than `n`. A compiler pass can verify this mechanically. If
someone writes `fib[n + 1]`, the index points forward, and the compiler reports
that a forward sweep cannot compute this recurrence.

Trace the first few:

```text
fib[2] = fib[1] + fib[0] = 1 + 0 = 1
fib[3] = fib[2] + fib[1] = 1 + 1 = 2
fib[4] = fib[3] + fib[2] = 2 + 1 = 3
fib[5] = fib[4] + fib[3] = 3 + 2 = 5
fib[6] = fib[5] + fib[4] = 5 + 3 = 8
fib[7] = fib[6] + fib[5] = 8 + 5 = 13
fib[8] = fib[7] + fib[6] = 13 + 8 = 21
fib[9] = fib[8] + fib[7] = 21 + 13 = 34
```

A loop is an execution order. A recurrence is a dependency relationship. The
difference is visible in the source: `fib[n-1]` states the edge, while `fib =
fib + ...` inside a loop buries it in mutable state.

### What the Encounter Showed

In twenty minutes, you have traced one prediction cell through:

```text
- A linear layer with named input and output coordinates
- A bias that omits the batch coordinate (broadcasting, made explicit)
- A batch coordinate that isolates examples
- A softmax that names the normalized coordinate
- An argmax that names the selection coordinate
- A gradient that inherits its address from the denominator
- A recurrence that states its dependency direction in the index
```

Every fact was visible in the source. No comment was needed. No convention was
assumed. The coordinate names did the work that shape tuples do not.

## The Habit

Every chapter in this book follows the same pattern. Start with a program you
could write today — in PyTorch, in JAX, in NumPy. Find the fact the notation hid.
Rewrite the program so the fact stays visible. End with an exercise that asks
you to write the shape-compatible wrong version first — because a bug that
still runs is the bug you will ship.

The chapters build, but the habit does not change:

For every example, pick one output cell and trace backward:

```text
Which coordinates did this cell read?
Which coordinates were summed away?
Which coordinates were present in the formula but absent from one term?
Which coordinate is the address of this gradient?
```

That habit — four questions, repeated — is the book's payload. Einlang is the
notation that makes the answers too visible to skip.

The fastest study loop:

```text
trace one cell
name the coordinates it reads
write the shape-compatible wrong version
state the law that rejects it
hide the mechanics behind a coordinate function only after the law is clear
```

## The Journey

The remaining chapters walk a single idea through four parts of increasing
pressure. Each part asks the same question in a new setting. Each setting would
be easier to skip. Don't. The idea compounds.

**Part I: Coordinates** (Chapters 1–6). We separate axis roles from axis
positions, learn to read coordinate maps as address equations, make broadcasting
explicit, and watch softmax reveal the difference between normalization
coordinates and surviving coordinates. By the end of Part I, you will read every
tensor line as a small audit of which names survive, which are consumed, and
which are silently omitted. You will never look at `dim=-1` the same way again.

**Part II: Derivatives** (Chapters 7–9). We turn the coordinate audit onto
automatic differentiation. A forward expression already knows which input cells
influence which output cells — the backward pass is just collecting sensitivity
along those routes. You will learn to read a gradient denominator as an address:
`@loss / @W` means "collect sensitivity at `W`'s coordinates." No Jacobian
materialization required. The transpose you memorized will become a consequence
you can derive.

**Part III: Time and Recurrence** (Chapters 10–12). Coordinates that depend on
earlier versions of themselves introduce the first coordinate with inherent
direction. A recurrence edge is a dependency the compiler must preserve through
optimization. You will learn to see time as a coordinate role — and to see a
loop as the story a recurrence tells when it runs, not the story it means.

**Part IV: Full Applications** (Chapters 13–16). Named dimensions meet
attention, dynamic routing, module boundaries, and the question of what notation
refuses to hide. These chapters are the book's stress test: can the same small
vocabulary that read a dot product also read a multi-head attention block? A
low-rank communication pattern? A dynamic expert route that changes per token?
Chapter 15 will show that every rule in the preceding fourteen chapters follows
from a single sentence. Chapter 16 will close the arc that began on a Tuesday.

The book ends with two appendices: a field manual of twelve failure patterns and
sixteen coordinate reading laws — the rules extracted and numbered, ready for
use without the surrounding story.

## The Bargain

This book will ask you to write more characters in a few critical places. The
extra text is the cost of a local fact. In exchange, you get to stop maintaining
a separate mental map of which dimension means what — and so does the compiler,
and so does the autodiff engine, and so does the next person who reads your
code.

One rule returns more than any other:

```text
Do not hide a fact that later reasoning must recover.
```

That rule is not a demand for verbosity. Good notation hides plenty — register
allocation, kernel selection, fusion order, temporary buffers. It should not
hide which dimension is time, which one is class, which one was consumed by a
reduction, or which one is the address of a gradient. Those are not
implementation details. They are the facts that decide whether the program means
what you think it means.

**Line to keep:** the shape tells you whether the program ran. The coordinates
tell you whether it ran for the right reason. Fifteen characters in three places
is a small price for a fact the compiler can check forever.
