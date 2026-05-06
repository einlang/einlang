---
layout: book
title: "Time Steps Are Not Loops"
---

# Time Steps Are Not Loops

> "What then is time? If no one asks me, I know what it is. If I wish to explain
> it to him who asks, I do not know."
>
> — Augustine of Hippo, *Confessions* (c. 400)

Everyone can write a loop. The hard part is saying what depends on what: which
values must come before which others, which can run in parallel, which must be
stored and which can be forgotten. A loop tells you what runs. It does not tell
you what a time step means. For that, you need to name the index.

Part II showed that hiding has consequences — the forward pass's omissions
become the backward pass's sums. Part III introduces a new dimension to the
problem: direction. When `h[t]` reads `h[t-1]`, the dependency has an arrow.
Hiding that arrow in mutable state does not just hide one fact. It hides four:
causality, storability, reversibility, and optimizability. Time is the first
coordinate in this book whose name carries direction — and direction is the
difference between a program the compiler can transform and a program it can
only execute.

## The Bug That Read Tomorrow's Token

You wrote a language model. The training loop runs for three days. When it
finishes, you check the metrics. Perplexity on the training set: 42. Perplexity
on the validation set: 38.

Validation is better than training. That never happens. Your first thought:
the validation set is easier. You compute unigram entropy on both—identical.
Your second thought: the checkpoint was taken at a lucky moment. You restart
from scratch—same result. Your third thought is the one you should have had
first.

You check the model code:

```python
h = init_state
for t in range(T):
    h = rnn_cell(h, x[t])
```

Looks fine. But `rnn_cell` is 200 lines long and was written by an intern
eight months ago. Buried in its body:

```python
# Line 143: attention-like gating over "context"
context = sum(w * x[t+1])  # accidentally reads the next token
```

The loop runs. `x[t+1]` exists because the sequence buffer has `T+1` entries.
The shapes are fine. The loss descends. But at every time step, the model
peeks at the answer. On the training set, it learns to rely on that future
peek—but training applies teacher forcing anyway, so the effect is subtle. On
the validation set, the future peek is still available (the sequence buffer
works the same way), so validation perplexity is good—even slightly better,
because the shorter validation sequences make the future token an even stronger
signal.

The model never learned to predict. It learned to copy.

A loop cannot catch this. `for t in range(T)` and `for t in range(T-1)` look
identical in a code review. The index `t+1` is inside a 200-line function body.
The loop only says what to execute, not what depends on what.

A recurrence catches it at the definition site. The index expression itself
reveals the direction:

```rust
// Correct: reads the past
let h[t in 1..T] = step(h[t - 1], x[t]);

// Bug: reads the future  (visible at a glance)
let h[t in 0..T-1] = step(h[t + 1], x[t]);
```

You do not need to read `step`'s body. The index `t + 1` in the recurrence
definition points forward. A compiler pass can verify: every read of a
recurrence family uses a smaller index. The loop buries the direction in
mutable state updates. The recurrence names it in the defining equation.

## Loop vs. Recurrence

The mathematical form of a recurrence is the same across every domain:

$$h_t = f(h_{t-1}, x_t), \quad h_0 = h_{\text{init}}$$

$t$ is the index. $h_{t-1}$ is the dependency edge — today reads yesterday.
$x_t$ is the external input at this step. The formula says nothing about loops,
mutable variables, or execution order. It only says what depends on what.

This is the second deepening of the book. The gradient (Part II) traced
sensitivity backward through a static graph — one operation feeding another.
Time traces state forward through a dynamic graph — a value depending on earlier
versions of itself. The two directions interact: the backward pass of a
recurrence must run in reverse time, and the compiler must know the dependency
direction to schedule it. Hiding the time direction in mutable state hides not
just readability but computability — the compiler cannot optimize what it cannot
see depends on what.

Write the same RNN twice:

```python
# Version A: loop (execution story)
h = h0
for t in range(1, T):
    h = step(h, x[t])
```

```rust
// Version B: recurrence (dependency story)
let h[0] = h0;
let h[t in 1..T] = step(h[t - 1], x[t]);
```

Version A commits early to one mutable slot and one serial schedule. The
dependency `h[t]` reads `h[t-1]` must be recovered from `step`'s body. Version
B states the dependency edge at the definition site. `h[t - 1]` is not an
implementation detail—it is the mark that today depends on yesterday.

The contrast is clearest when a dependency is wrong:

```rust
// This reads the future
let bad[t in 0..T] = bad[t + 1] + 1;
```

A loop version of this bug runs without complaint—`range(T)` still iterates
forward, the read of `bad[t+1]` is hidden in an expression, the program
produces numbers. The recurrence version displays the problem in the index
itself: `t + 1` points forward, and a forward simulation cannot have a future
dependency without a different contract (boundary-value problem, fixed-point
iteration, backward pass).

The quick check: circle every read of the same binding. If `h[t]` reads
`h[t - 1]`, the edge points backward. If it reads `h[t + 1]`, the edge points
forward. A compiler pass can perform this check mechanically because the
recurrence states the relation explicitly.

## The Three Questions a Recurrence Separates

A loop fuses three ideas into one mutable variable. A recurrence pulls them
apart:

```text
family       h[t]          the collection of values indexed by time
dependency   h[t] reads    which earlier value each step uses
             h[t - 1]
storage      chosen later   what to keep, what to discard
```

Take a scalar recurrence:

```rust
let h[0] = 1.0;
let h[t in 1..5] = 0.5 * h[t - 1] + x[t];
```

With inputs `x[1] = 10, x[2] = 20, x[3] = 30, x[4] = 40`:

```text
h[1] = 0.5 * h[0] + 10
h[2] = 0.5 * h[1] + 20
h[3] = 0.5 * h[2] + 30
h[4] = 0.5 * h[3] + 40
```

The important fact is not the arithmetic—it is the dependency direction. Every
`h[t]` reads `h[t - 1]`, never `h[t + 1]`. The index expression states that
direction in the value definition itself.

```
   Time Dependency Graph

   Correct: h[t] depends on h[t-1]
   h[0] ---> h[1] ---> h[2] ---> h[3] ---> h[4]
    0.5*+x1    0.5*+x2    0.5*+x3    0.5*+x4

   Each arrow points forward in evaluation:
   h[t] reads h[t-1], so to compute h[t] you need h[t-1] first.
   A forward sweep can follow the arrows.

   Wrong: h[t] depends on h[t+1]  (crossed out)
   h[0] -X-> h[1] -X-> h[2] -X-> h[3] -X-> h[4]
     reads     reads     reads     reads
    h[1]      h[2]      h[3]      h[4]

   Forward sweep cannot compute this --
   h[0] needs h[1] which hasn't been computed yet.
   Requires a different contract (boundary-value, fixed-point).
```

Now consider storage. If the program only asks for the final state:

```rust
let final = h[T - 1];
```

The recurrence needs only a one-step rolling window. If it asks for all states:

```rust
let all_states[t] = h[t];
```

The full sequence must remain available. The recurrence rule did not change,
but the storage demand did. The separation lets storage be a consequence of
observation, not a property of the recurrence itself.

This is the Hiding Law applied to time. A loop notation merges three facts
into one mutable variable: what values exist, which earlier value each step
reads, and which values must occupy memory. When those three facts are merged,
changing one — observing a different time step — silently changes the others.
When they are separate, the observation can change without rewriting the
recurrence. The notation determines whether storage is a decision you make
or a side effect you discover.

## Batch Isolation During Recurrence

Add a batch dimension:

```rust
let h[0, b] = init[b];
let h[t in 1..T, b] = 0.5 * h[t - 1, b] + x[t, b];
```

For `h[3, 7]`, the previous state is `h[2, 7]`. Batch stays fixed. The
recurrence does not mix `b = 7` with `b = 2`—it only walks backward in time.
If the formula accidentally read `h[t - 1, b_prev]` under a `sum[b_prev]`, the
model would communicate across batch examples. The shape would be valid. The
loss would decrease. The meaning would silently change.

This is the same coordinate isolation from earlier chapters, applied to a
dynamic axis. Time moves. Batch stays fixed. Named coordinates make the
boundary explicit.

## A Real RNN

The core of `stdlib/ml/recurrent_ops.ein`, with bias and activation dispatch
trimmed:

```rust
let hidden[0, b in 0..batch_size, h in 0..hidden_size] =
    if typeof(initial_h) == "rectangular" { initial_h[b, h] } else { 0.0 };

let hidden[t in 1..seq_length, b in 0..batch_size, h in 0..hidden_size] = {
    let z_cell =
        sum[i in 0..input_size](W[h, i] * X[t, b, i]) +
        sum[h_prev in 0..hidden_size](
            R[h, h_prev] * hidden[t - 1, b, h_prev]
        );
    tanh(z_cell)
};
```

Read one cell: `hidden[7, b, h]`. It is built from today's input `X[7, b, i]`
and yesterday's hidden state `hidden[6, b, h_prev]`. Three roles are visible
at once:

```text
t        the time coordinate being defined
i        the input feature consumed by the input projection
h_prev   the previous hidden coordinate consumed by the recurrent projection
```

Time changes. Batch stays fixed. Hidden coordinates are mixed through the
recurrent weight matrix. All of this is visible in the source without tracing
through a Python loop body.

## Time Has Direction

Time is the first coordinate in this book with an inherent direction. Batch can
be reordered. Feature can be reduced. Time in a forward recurrence carries
causality: `t` may depend on `t - 1`, but not on `t + 1`.

Previous chapters used named coordinates to check shape and role. Time adds one
more check: does a valid evaluation order exist? A recurrence rule is a set of
equations, but not every set of equations is a legal forward computation. The
index `t - 1` respects causality under the range `1..T`. The index `t + 1`
does not.

This is the wall between execution and dependency. A loop presents time as execution
steps—all the steps are already scheduled, so direction is irrelevant. A
recurrence presents time as a dependency axis where direction matters. The
separation divides two ways of reading the same computation.

Time is the first coordinate in this book with an inherent direction. Batch
can be reordered. Feature can be reduced. None of the coordinates in Part II
cared about evaluation order. Time cares. The question is not whether the
loop runs — it does. The question is whether the notation records the
direction of dependency, so that a future reader, a compiler pass, or a
storage planner can verify causality at the definition site rather than
reconstruct it from the loop body.

## Causality as a Coordinate Property

Causality is not a property of the loop body. It is a property of the index
expression. The recurrence statement `let h[t in 1..T] = step(h[t - 1], x[t])`
makes three causal claims:

1. **Monotonicity.** Every read of the recurrence family uses an index strictly
   less than the value being defined. `h[t - 1]` is less than `h[t]`. A
   compiler can check this mechanically: for every occurrence of `h[expr]` on
   the right-hand side, verify `expr < t` for all `t` in the range.

2. **Bounded memory.** The difference `t - expr` is the number of past steps
   consulted. `h[t - 1]` consults exactly one step. `h[t - k]` consults `k`
   steps. The maximum difference is the backward window. This number governs
   storage, not correctness—a recurrence with `h[t - 100]` is still causal,
   just with a larger memory requirement.

3. **Acyclicity.** The recurrence graph has no cycles. Every edge points from
   a smaller time index to a larger one. Forward simulation can follow the
   edges in index order. Backpropagation follows them in reverse.

Now consider three patterns and whether they satisfy these claims:

```rust
// Forward recurrence — satisfies all three
let h[t in 1..T] = step(h[t - 1], x[t]);

// Backward recurrence — violates claim 1 (reads larger index)
// This is a boundary-value problem, not a forward simulation
let h[t in 0..T-1] = step(h[t + 1], x[t]);

// Bidirectional — violates claim 1, requires different contract
let h[t in 0..T] = step(h[t - 1], h[t + 1], x[t]);
```

The second and third patterns are not "wrong." They are different contracts.
A backward recurrence solves a different problem (given the end state, work
backward). A bidirectional recurrence needs a different evaluation strategy
(fixed-point iteration or two-pass). The point is that the recurrence syntax
makes the contract visible. A loop makes all three look like `for t in range(T)`.

The check is syntactic:

```text
For every occurrence of h[g(t, ...)] on the right-hand side of
  let h[t in a..b] = ...
verify: g(t, ...) < t for all t in a..b.

If the check passes: the recurrence is causal. Forward simulation works.
If the check fails: the recurrence needs a different evaluation contract.
```

This check is impossible with a loop because the loop does not name the
dependency indices. `h = step(h, x[t])` says nothing about `h[t-1]` versus
`h[t+1]`. The information is in the body of `step`, and no compiler can
recover it from Python control flow.

The syntactic check is not an optimization. It is the difference between
catching the future-peek bug at the definition site and catching it three
days later when validation perplexity is better than training. The loop is
not wrong — it runs. But it withholds the one fact — index direction — that
would make the bug visible to a compiler. The notation determines not just
what the programmer can notice, but what the compiler can refuse.

## Try It

A loop is muscle memory. A recurrence is a different muscle. Convert this loop
to a recurrence:

```python
h = h0
for t in range(1, T):
    h = 0.5 * h + x[t]
```

Write the base case `h[0] = h0`. Write the recurrence `h[t in 1..T] = 0.5 *
h[t - 1] + x[t]`. Verify the dependency edge points to `t - 1`, not `t + 1`.
Add a batch coordinate and confirm no batch member reads another. The three
facts — base case, backward edge, batch isolation — are visible in three lines.

Now try a bidirectional recurrence. Write the forward pass `h_f[t]` that reads
`t-1` and the backward pass `h_b[t]` that reads `t+1`:

```text
let h_f[0, b] = init_f[b];
let h_f[t in 1..T, b] = step_f(h_f[t - 1, b], x[t, b]);

let h_b[T-1, b] = init_b[b];
let h_b[t in 0..T-1, b] = step_b(h_b[t + 1, b], x[t, b]);
```

For `h_f[7, 3]`: reads `h_f[6, 3]` — causal, depends on the past. For
`h_b[3, 3]`: reads `h_b[4, 3]` — anti-causal, depends on the future. The
backward recurrence cannot be computed in a forward sweep: at time `t`, the
value at `t+1` is not yet known. Flip the sequence — map `x_rev[t] = x[T-1-t]`
— and the backward recurrence on `x` becomes a forward recurrence on `x_rev`.
The index transforms from `t+1` to `t-1`. The coordinate transformation is
visible in the index expression.

Now consider a skip connection: `h[t]` reads both `h[t-1]` and `h[t-2]`. Write
the recurrence with two base cases:

```text
let h[0] = a;
let h[1] = b;
let h[t in 2..T] = step(h[t - 1], h[t - 2], x[t]);
```

The dependency window is 2 steps instead of 1. For a recurrence that reads
`h[t-k]`, you need `k` initial values. A compiler can compute this mechanically:
scan all index offsets in the definition of `h[t]`, find the maximum `k` such
that `t-k` appears, and require `k` base cases. The causality check is equally
mechanical: for every occurrence of `h[expr]` in the definition of `h[t]`,
verify `expr < t` for all `t` in the domain. Linear offsets like `t-k` with
`k > 0` pass. `t+k` fails. The recurrence notation makes this a local check
over index expressions. A loop body buries the check inside mutable state that
no compiler can analyze without whole-program reasoning.

**Line to keep:** a loop is an execution order; time is a dependency relationship.

Time is the first coordinate in this book with an arrow, but not the only one.
Fibonacci's `n` also flows forward. An optimizer's `iter` also flows forward.
The arrow is the point — not that the coordinate happens to be time, but that
the notation can record which way the dependency points. A recurrence that
reads tomorrow is a bug. A loop that reads tomorrow is a runtime value. The
difference is not whether the program compiles. It is whether the notation lets
you say "I meant the past."

### Where This Leads

Part III changes the question. Parts I and II examined static programs — a
reshape, a reduction, a gradient. The coordinates were fixed. The communication
graph was determined by the formula and did not change. Part III introduces
time, the first coordinate in this book with an inherent direction.

When a coordinate carries causality — when later values depend on earlier values
— the notation must answer a new question: does a valid evaluation order exist?
The answer cannot be recovered from a loop body because the loop buries the
dependency direction inside mutable state. A recurrence notation names the
dependency index, and the check becomes mechanical: does every read use a
smaller index than the value being defined?

We have separated execution from dependency. A loop is one way to run a
recurrence; a recurrence is a statement about which time index reads which other
time index. But separating execution from dependency raises the next question:
if `h[t]` is a family of values, which ones must be stored? The three chapters
of Part III separate three ideas that a loop merges into one: the definition of
a value family, the direction of dependency, and the choice of what to keep in
memory.

Chapter 11 shows how the same recurrence makes a different storage demand
depending on which values are observed. The rule is simple: storage follows
observation, and the dependency graph tells you what can be thrown away.