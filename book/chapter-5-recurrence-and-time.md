---
layout: book
title: "Chapter 5: Recurrence and Time"
---

# Chapter 5: Recurrence and Time

So far, each binding has described a value in terms of earlier names. Recurrence
adds time-like self-reference: a tensor may be defined by base cases and by a
rule for later indices. The result is a compact way to write sequences,
stateful models, and dynamic programs while keeping dependency analysis visible.
This explicit temporal structure transforms time from an operational detail
into a first-class index that can be analyzed, optimized, and composed,
enabling algorithms that reason about temporal patterns as systematically as
they reason about spatial patterns.

## 5.1 Linear Recurrence: Fibonacci

Recurrence makes time an index like any other, enabling the same analytical
tools that work on spatial dimensions to work on temporal ones. This
unification of space and time creates powerful opportunities for optimization
and analysis.

Formula:

```text
F_0 = 0
F_1 = 1
F_n = F_{n-1} + F_{n-2}
```

Einlang:

```rust
let fib[0] = 0;
let fib[1] = 1;
let fib[n in 2..20] = fib[n - 1] + fib[n - 2];
```

The clauses form one recurrent definition. Base clauses define boundary points;
the recurrent clause defines later points by reading earlier ones. This
structure makes the temporal dependencies visible, enabling the compiler to
reason about storage requirements, parallelization opportunities, and
optimization strategies.

The dependency offsets are `-1` and `-2`. That tells the compiler the maximum
lookback is two. If only the final value is needed, a rolling window of recent
values is enough. If later code observes the whole sequence, the runtime must
materialize the whole sequence.

### Recurrence Is Not a Loop

The Fibonacci program describes a dependency relation:

```text
fib[n] depends on fib[n - 1] and fib[n - 2]
```

A loop is one possible way to evaluate that relation. The source does not say
"iterate a variable and mutate an array." It says which points exist and which
earlier points each point reads. This is why the compiler can choose an
evaluation strategy and a storage strategy after reading the recurrence.

For this recurrence, a left-to-right schedule is valid:

```text
fib[0], fib[1], fib[2], fib[3], ...
```

But the schedule is a consequence of the dependency graph, not the source
abstraction itself.

## 5.2 Mutual Recurrence and Dependency Graphs

> **Think More.** Mutual recurrence shifts attention from textual order to
> dependency structure. If two definitions refer to earlier versions of one
> another, the compiler must decide whether the graph is well-founded. What
> should the language report when the graph has a real cycle rather than a valid
> time delay? This opens a conversation about diagnostics: should the compiler
> merely reject the program, or explain the dependency path that made evaluation
> impossible?

Formula:

```text
a_n = a_{n-1} + b_{n-2}
b_n = a_{n-1} - b_{n-1}
```

Einlang-style source:

```rust
let a[0] = 1.0;
let a[1] = 1.0;
let b[0] = 0.0;
let b[1] = 1.0;

let a[n in 2..T] = a[n - 1] + b[n - 2];
let b[n in 2..T] = a[n - 1] - b[n - 1];
```

The compiler builds a dependency graph from self-reads and cross-reads. A legal
evaluation order must compute every dependency before the point that reads it.

The storage idea generalizes: the required history is determined by the largest
backward offset that can still be read. Conservative full materialization is the
safe fallback when later reads cannot be bounded statically.

### Graphs Across Bindings

Self-recursion is only the simplest case. Mutual recurrence means one binding
can depend on another binding at a related point. The compiler's graph has nodes
like `a[n]` and `b[n]`, with edges such as:

```text
a[n] -> a[n - 1]
a[n] -> b[n - 2]
b[n] -> a[n - 1]
b[n] -> b[n - 1]
```

A cycle is only a problem if it prevents an order where dependencies are already
available. Backward edges are usually fine. Future reads are not:

```rust
let bad[t in 0..T] = bad[t + 1] + 1;
```

That line asks for a value from the future.

## 5.3 RNNs and Dynamic Programming

> **Think More.** RNNs and dynamic programs often look different in libraries,
> but both are recurrence patterns over state. If the language can see the
> dependency neighborhood, it can ask a deeper question: is the full history part
> of the result, or only a scaffold needed to compute the next state? From there,
> the discussion can move toward checkpointing, streaming, and what users should
> be allowed to promise about future reads.

Formula:

```text
h_t = tanh(W_hh h_{t-1} + W_xh x_t)
```

Einlang:

```rust
use std::math::tanh;

let h[0, i] = h0[i];
let h[t in 1..T, i] = tanh(
    sum[j](W_hh[i, j] * h[t - 1, j]) +
    sum[k](W_xh[i, k] * x[t, k]) +
    bias[i]
);
```

An RNN is a recurrence over time with tensor values at each time step. The
syntax does not change: `t` is an index, and `h[t - 1, j]` is a backward read.

Dynamic programming uses the same idea over larger index spaces:

```rust
let D[0, j in 0..N] = init_col[j];
let D[i in 0..M, 0] = init_row[i];
let D[i in 1..M, j in 1..N] =
    min(D[i - 1, j] + del,
        D[i, j - 1] + ins,
        D[i - 1, j - 1] + sub[i, j]);
```

The program states the dependency relation. The schedule and buffer layout are
consequences of that relation.

### Dynamic Programming as Geometry

In a two-dimensional recurrence, dependencies have directions:

```text
D[i - 1, j]     above
D[i, j - 1]     left
D[i - 1, j - 1] diagonal
```

Those directions tell the compiler which parts of the grid must be available
when computing a point. A row-major schedule works for this example because the
above row and the previous cell in the current row are already available. A read
such as `D[i, j + 1]` would point rightward, toward a value not yet computed in
that schedule.

The source program does not need to name row-major order. The dependency vectors
make the legal orders discoverable.

### A Worked Reading: Streaming State

Many time-series programs have the same shape:

```rust
let state[0, i] = init[i];
let state[t in 1..T, i] =
    decay * state[t - 1, i] + input[t, i];
let final[i] = state[T - 1, i];
```

The recurrence defines an entire history, but the final program observes only
the last point. The body reads `state[t - 1, i]`, so the lookback is one. If no
other expression reads the full history, the runtime can keep the current state
and the previous state rather than storing every time step.

This is not an optimization bolted onto a loop. It is a consequence of the
source equation. The equation names the dependency; the storage analysis reads
that dependency.

## 5.4 Temporal Algorithms and Patterns

> **Think More.** Treating time as an index lets a recurrence describe
> dependencies before storage is chosen. Which recurrences can be streamed,
> checkpointed, or parallelized, and what promises must the source make?

Explicit temporal structure enables advanced time-series algorithms.

### Kalman Filtering

State estimation with prediction and measurement updates:

```rust
let x[0] = initial_state;
let P[0] = initial_covariance;

let x_pred[t in 1..T] = F * x[t - 1] + B * u[t];
let P_pred[t in 1..T] = F * P[t - 1] * F_T + Q;

let y_pred[t in 1..T] = H * x_pred[t];
let residual[t in 1..T] = z[t] - y_pred[t];
let S[t in 1..T] = H * P_pred[t] * H_T + R;

let K[t in 1..T] = P_pred[t] * H_T * inv(S[t]);
let x[t in 1..T] = x_pred[t] + K[t] * residual[t];
let P[t in 1..T] = (I - K[t] * H) * P_pred[t];
```

Complex state estimation becomes a structured recurrence.

### Exponential Moving Averages

Time-weighted aggregations:

```rust
let alpha = 0.1;  // Smoothing factor
let data[t in 0..T] = time_series[t];

let ema[0] = data[0];
let ema[t in 1..T] = alpha * data[t] + (1 - alpha) * ema[t - 1];
```

Temporal smoothing with explicit dependency tracking.

### Wave Propagation

Physical simulations with spatial-temporal dependencies:

```rust
let u[0, x in 0..N] = initial_wave[x];
let u[1, x in 0..N] = initial_velocity[x] * dt + u[0, x];

let u[t in 2..T, x in 1..N-1] =
    2 * u[t - 1, x] - u[t - 2, x] +
    c * c * dt * dt / dx / dx * (u[t - 1, x + 1] - 2 * u[t - 1, x] + u[t - 1, x - 1]);
```

Spatial-temporal recurrence for physical simulation.

### Implementation Models for Recurrence

**Sequential Processing**: Compute each time step in order. Simple but
limited parallelism.

**Parallel Scan**: Use parallel prefix sum algorithms for associative
operations. Highly parallel but requires associativity.

**Rolling Window**: Maintain only recent states based on dependency analysis.
Memory efficient for long sequences.

**Graph-Based**: Build dependency graph and schedule computation. Flexible
but complex.

**Streaming**: Process data as it arrives, maintaining minimal state.
Efficient for real-time applications.

Each model affects how temporal algorithms are implemented and optimized.
- later observable reads can force more storage;
- multidimensional recurrences generalize the same idea to grids.

## 5.5 Scheduling, Causality, and Legal Evaluation Orders

Recurrence becomes much easier to reason about once we stop thinking of it as a
special loop syntax and start thinking of it as a description of causality. A
recurrent definition says which points may depend on which earlier points. From
that description, an implementation must recover a legal order of evaluation.
The key word is legal. Not every textual order, loop order, or parallel order
is valid. A valid schedule is one that never asks for a value before that value
has been made available.

This is already visible in a tiny example:

```rust
let fib[0] = 0;
let fib[1] = 1;
let fib[n in 2..20] = fib[n - 1] + fib[n - 2];
```

A left-to-right evaluation is legal because every requested predecessor lies to
the left. A future-looking rule such as `fib[n] = fib[n + 1] + 1` is not legal
under the same interpretation because it demands information from a point that
has not yet been established. The language is therefore carrying an implicit
notion of causal order even when it never says "for" or "while."

### Recurrence as a Partial Order

For simple one-dimensional examples it is tempting to equate the legal schedule
with ordinary numerical order. That temptation becomes limiting once recurrence
spreads across more dimensions or across multiple bindings. What really exists
is a partial order induced by dependencies. If `state[t]` depends on
`state[t - 1]`, then the earlier time must precede the later one. But if two
points do not depend on one another, the schedule may have freedom.

That freedom matters in systems work. It is what allows some recurrences to be
parallelized in chunks, tiled over space, or transformed into scan-like
algorithms. The source does not need to enumerate all legal schedules. It needs
only to state the dependency relations accurately enough that a legal schedule
can be discovered.

### Mutual Recurrence Clarifies the Point

Mutual recurrence makes the partial-order interpretation especially vivid:

```rust
let a[n in 2..T] = a[n - 1] + b[n - 2];
let b[n in 2..T] = a[n - 1] - b[n - 1];
```

There is no single binding whose values tell the whole story. Instead the
implementation has to respect a graph of edges among `a[n]`, `a[n - 1]`,
`b[n - 2]`, and `b[n - 1]`. If a reader can picture those edges, they can
already understand why some schedules are valid and others are not. This is one
of the chapter's central claims: recurrence is easier to discuss when it is
framed as dependency geometry rather than a mutation pattern.

### Causality Is a Semantic Resource

Causality is not merely a constraint. It is also a semantic resource. Once the
compiler knows that information flows in particular directions, it can begin to
prove useful things. It can establish that some results need only a rolling
window of history. It can detect illegal future reads. It can sometimes
parallelize along independent fronts of a dependency graph. It can also explain
errors in language that corresponds to the programmer's source rather than to a
lowered loop nest.

That explanatory power is important. Recurrence-heavy code in general-purpose
languages often fails in ways that are operational rather than structural:
uninitialized state, wrong update order, accidental in-place overwrite, or
subtle aliasing. Einlang's source form changes the failure surface. Errors are
more likely to be phrased as dependency mistakes: a point asks for the future, a
base case is missing, a needed predecessor is undefined. Those are better
questions because they live closer to the mathematical intent.

## 5.6 Storage, Forgetting, and the Meaning of History

The storage analysis that follows recurrence is one of the most appealing places
where source-visible structure becomes practical leverage. A recurrent program
often mentions an entire history of values, but not every part of that history
must survive in memory. The implementation can sometimes forget safely. The art
lies in knowing what may be forgotten and what must remain observable.

The simplest illustration is the linear recurrence with short lookback:

```rust
let ema[0] = data[0];
let ema[t in 1..T] = alpha * data[t] + (1 - alpha) * ema[t - 1];
```

If later code only asks for the final `ema[T - 1]`, then a full array of all
intermediate values is not semantically required. The current state and the
previous state may be enough. This is not a clever trick discovered after
optimization passes. It is a direct consequence of the dependency offsets and of
which later observations the source permits.

### History as Denotation, Storage as Strategy

It is important not to confuse these two layers. Denotationally, the recurrence
defines a history. Operationally, the implementation may compress that history
aggressively. The source stays clean because it does not force the user to write
their idea in the same impoverished state-machine style that an optimized
runtime may use.

This is one of the subtle advantages of recurrence over manual mutation. In
imperative code, the source often already looks like the storage optimization:
"keep a current variable and update it." That can be efficient, but it hides the
larger dependency relation. In Einlang, the source can remain denotationally
rich while the runtime adopts an efficient storage strategy behind the scenes.

### When Full History Matters

Of course the language cannot always forget. If later code prints the whole
sequence, differentiates through every time step, or indexes back into earlier
states, the history becomes semantically observable. That changes the storage
story. The runtime may need to preserve all points, or at least preserve enough
information to reconstruct them under the language's supported behavior.

This is exactly the kind of trade that becomes easier to articulate in a
structured recurrence language. We no longer ask vaguely whether "the loop is
stateful." We ask concretely: what are the backward offsets, and what later
reads force the past to remain available? Those are questions a compiler can
answer from source structure.

### Checkpointing and Recomputation

Between "store everything" and "store only the latest window" lies a broad
middle territory. Some systems may store sparse checkpoints and recompute
segments on demand. Others may keep compressed summaries. Still others may use
different strategies depending on whether the program is running forward for
inference or needs backward information for differentiation.

Einlang's recurrence notation does not encode all of those policies directly,
but it gives an implementation a reliable place to start. Because dependencies
are explicit, recomputation is a principled option rather than an act of
desperation. The runtime knows what it would have to replay and what facts must
remain stable.

## 5.7 Time as a Modeling Choice, Not an Afterthought

The wider significance of recurrence is that it makes time itself a modeling
dimension in the language. This is larger than Fibonacci, larger than RNNs, and
larger than dynamic programming tables. Once time is visible as an index, a
program can treat evolution, memory, and adaptation as ordinary structured
relations.

That visibility changes how one writes many familiar systems. A filter is no
longer "update an accumulator variable." It is a time-indexed family where each
point depends on the previous state and the new observation. A controller is not
merely "keep mutating internal state." It is a rule over successive versions.
A dynamic program is not just nested loops with a table; it is a surface over a
grid of dependencies. The recurrence vocabulary unifies these cases without
flattening their differences.

### Recurrence and Explanation

This matters for explanation as much as for execution. If a program is intended
to be read by humans, then the source should make temporal structure legible.
One reason recurrent neural networks, sequence models, and dynamic programs can
feel opaque in host-language code is that the mathematics lives in one place and
the update mechanics live in another. Recurrence syntax pulls them back
together.

A reader seeing

```rust
let h[t in 1..T, i] = tanh(
    sum[j](W_hh[i, j] * h[t - 1, j]) +
    sum[k](W_xh[i, k] * x[t, k]) +
    bias[i]
);
```

does not need separate prose to discover where time enters, what part of the
state is recurrent, and what part is new input. The program already says it.
That does not make every modeling decision obvious, but it greatly reduces the
translation work between the mathematical description and the code artifact.

### A Language-Level View of Temporal Computation

Seen broadly, recurrence is the point where the book's earlier themes acquire a
temporal dimension. Names remain stable. Indices remain visible. Reductions may
still occur inside the body. Derivative requests may later ask how a final value
depends on parameters threaded across time. The result is not a separate
mini-language for loops. It is the same structural language, extended so that
time-like self-reference becomes explicit and analyzable.

That is why recurrence deserves to be treated as a first-class construct instead
of an implementation pattern. It is a compact source form for causal structure,
storage opportunity, and temporal explanation all at once.

## 5.8 Why Time Changes the Meaning of Programs

There is a final reason recurrence matters so much: once time is explicit, the
meaning of a program stops being exhausted by its spatial shape alone. A tensor
at one instant is only part of the story. The language can now describe how
information persists, decays, accumulates, or branches across moments. This is
the beginning of a different view of computation, one in which state is not an
accident of mutable memory but a visible relation among versions.

That shift has consequences far beyond classic sequence models. It affects how
we think about simulation, online estimation, streaming analytics, and any
system whose outputs depend on remembered context. Once time is part of the
source, these tasks no longer need to hide inside imperative update folklore.
They can be expressed in the same structural vocabulary as the rest of the
language.

### Temporal Structure as Shared Vocabulary

This gives recurrence a social value as well as a technical one. When a modeler,
compiler writer, and reviewer look at the same recurrent program, they can all
point to the same visible dependency structure. One person may care about
correctness, another about storage, another about interpretability, but the
source gives them a common object to discuss. That shared vocabulary is one of
the quiet marks of a mature language feature. It supports not just execution,
but explanation and collaboration.

## 5.9 Recurrence Beyond the Canonical Examples

It is tempting to let recurrence stand for a small gallery of familiar examples:
Fibonacci, RNNs, dynamic programming tables, perhaps an exponential moving
average. Those examples are useful, but they can accidentally narrow the
reader's sense of what the construct is for. Recurrence is better understood as
a general language for dependence across ordered versions, whether those versions
represent moments in time, iterations of refinement, layers of accumulated
evidence, or stages in a simulation.

A numerical solver, for instance, may update an estimate from an earlier
estimate plus a correction term. A filtering system may fold new observations
into a running state. A simulation may advance a field using values from the
previous one or two steps. An online analytics pipeline may maintain rolling
statistics whose current values depend on earlier summaries. All of these fit
comfortably into the same structural picture: base cases establish initial
availability, and recurrent clauses describe how later versions read earlier
ones.

### Recurrence and Revision

One useful mental model is that recurrence expresses revision without mutation.
Instead of a variable that repeatedly changes identity, we have a family of
related versions, each one defined from others. That sounds like a small
terminological difference, but it changes the semantics dramatically. Revision
through recurrence keeps the history conceptually available even when the
runtime later decides not to store every point. The source therefore remains a
description of the whole evolving object, not merely a recipe for a hidden
in-place update.

This matters for explanation because many algorithms are most naturally
understood as sequences of revised states. The imperative update style often
optimizes away that conceptual sequence in the source itself. Recurrence lets the
program keep the sequence visible while still permitting an efficient
implementation.

### Structured Time and Human Reasoning

Time also changes the kind of reasoning humans can perform about code. A spatial
tensor definition asks us to imagine a set of coordinates. A recurrent
definition asks us to imagine those coordinates plus a direction of dependence.
That directional structure is cognitively powerful. We can ask what information
must be remembered, what information may be forgotten, which points are
independent once their predecessors are fixed, and where causality could be
broken by an illegal forward reference.

These are good questions because they live at the level of the source model
rather than at the level of machine control flow. A programmer can discuss them
with a reviewer, student, or compiler writer without first translating the
program into imperative operational jargon. That is part of what makes
recurrence a language feature rather than merely a runtime implementation
strategy.

### A Broader View of Time

Finally, recurrence invites us to treat time as something broader than clock
ticks. In many systems, "time" really means any ordered axis along which later
states depend on earlier ones: decoding position, refinement iteration, solver
step, simulation frame, curriculum stage, or streaming event index. The power of
Einlang's formulation is that it does not force these cases into radically
different surface idioms. The same dependency language can describe them all.

That breadth is why recurrence belongs at the center of a tensor language rather
than at its edges. A world of changing, accumulating, and stateful computation
cannot be expressed well if time-like dependence remains invisible. This chapter
argues that the right response is not to reintroduce general mutable machinery by
default, but to make temporal dependence itself explicit enough to analyze.

## 5.10 Discussion: Why Recurrence Belongs in the Core

Recurrence should not be read as a niche feature for sequence models. It is the
source form for any computation where later points depend on earlier points:
solvers, filters, simulators, decoders, dynamic programs, and iterative
optimization. If a tensor language can express shapes, reductions, and
derivatives but cannot express time-like dependence clearly, then an important
class of programs is pushed back into host-language mutation.

Making recurrence explicit gives both the reader and the compiler a better
account of what the program remembers. The source can show which earlier points
are needed, which points can be computed independently, and when a full history
is semantically required. That is the practical reason recurrence belongs next
to indexing, reduction, and autodiff rather than outside the core language.

## 5.11 A Practical Test for Recurrence

A useful final test is simple: can you point to any recurrent clause and say
which earlier points it needs, and why those points should already exist? If you
can, the temporal structure is probably clear. If you cannot, the ambiguity is
often real rather than stylistic.

Use the same test for storage decisions. If later code reads only a suffix of a
recurrent value, a rolling window may be possible. If later code can observe
arbitrary points, full materialization is usually required. The source-level
recurrence gives both the reader and the compiler the information needed to make
that distinction.

## Summary

Recurrence makes time-like dependence explicit in the same indexed style used
for tensors.

- Base clauses define boundary points, establishing the initial conditions
  that anchor temporal computation;
- Recurrent clauses define points from earlier points, making temporal
  dependencies visible and analyzable;
- Dependency offsets imply legal schedules, enabling the compiler to
  parallelize across time when dependencies allow;
- Lookback analysis implies possible rolling storage windows, optimizing
  memory usage for long-running computations.

Algorithms that depend on temporal patterns, including Kalman filtering,
recurrent neural networks, and dynamic programming, can then be read as indexed
dependencies rather than host-language mutation. The next chapter turns to the
boundary between Einlang and Python.
