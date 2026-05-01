---
layout: book
title: "Chapter 10: The Wall of Time Steps"
---

# The Wall of Time Steps

The common illusion is that a loop is the same thing as time. It is not. A loop
is one way to execute a dependency that should be visible before scheduling.

Sequence models are usually introduced with loops:

```python
h = init_state
for t in range(T):
    h = rnn_cell(h, x[t])
```

This is clear executable code, but it also builds a wall. The loop says "do this
next," then "do this next," then "do this next." The time dependency is there,
but the reader has to look through sequential control and mutation of `h` to
find it.

Ask what the program states. It states that `h` is updated repeatedly. It may
not state, in source-level form, that the value at time `t` depends on the
value at time `t - 1`. That relationship has to be recovered from the loop
body.

Loops are excellent execution devices. They are less direct as explanations of
time-shaped structure.

A loop tells an execution story. A recurrence tells a dependency story. The
two can lower to the same machine behavior, but they give the compiler and the
reader different source facts to work with. The important separation is between
`h[t]`, `h[t - 1]`, and the storage slot that may eventually hold them.

## Control Step or Dependency Axis

The natural design is to treat time as control flow. A loop counter advances,
a mutable state changes, and the program eventually produces a final value.
This matches the machine well. It also makes time disappear into execution
order.

The alternative is to treat time as an index of a value family. Then the
program can state that `h[t]` depends on `h[t - 1]` before deciding how to run
the computation. The compiler sees a dependency graph rather than only a
sequence of updates.

This is not a rejection of loops. It is a placement decision. Use loops when
choosing an execution order. Use a recurrence when the time dependency itself
is part of what the program is saying.

The contrast is clearest when the same recurrence is written twice:

```python
h = h0
for t in range(1, T):
    h = step(h, x[t])
```

and:

```rust
let h[0] = h0;
let h[t in 1..T] = step(h[t - 1], x[t]);
```

The first program commits early to one mutable slot and one serial schedule.
The second states the dependency edge. A compiler may still lower it to the
same loop, but it can first ask whether batch coordinates are independent,
whether only the final state is observed, or whether the full family must be
kept.

Two programs can have the same loop shape while answering different semantic
questions: does step `t` depend on `t - 1`, `t - 2`, or all earlier steps? A
loop counter alone does not answer. The answer lives in the body. A recurrence
puts the answer at the address where the value is defined.

## Time as an Index

Write the same idea as recurrence:

```rust
let h[0] = init_state;
let h[t in 1..T] = rnn_cell(h[t - 1], x[t]);
```

Now `h` is not a mutable cell. It is a family of values indexed by time. The
base binding gives one member of the family. The recurrence rule gives the rest
under a range constraint.

For each `t`, the dependencies are visible:

```text
h[t - 1]
x[t]
```

The offset `t - 1` is not an implementation detail. It is the little mark that
today depends on yesterday. The range `1..T` states where the recurrence rule
applies. The base case `h[0]` states where the dependency chain begins.

Initial state is therefore not an awkward extra outside the coordinate system.
It is the boundary member of the family. Some languages spell it as `h0`, some
as `h[-1]`, and some as an optional input to an RNN library call. In the
indexed form, the safest choice is to make the boundary explicit:

```rust
let h[0] = h0;
let h[t in 1..T] = step(h[t - 1], x[t]);
```

If a model really wants a state before the first input, it can choose a shifted
domain, but the choice should be visible. Hidden initial conditions are another
place where loop notation can smuggle a semantic boundary into setup code.

A loop says "next, next, next." A recurrence says "this point depends on that
earlier point." The second form gives the compiler and the reader a dependency
relation before it gives them a schedule.

A future helper can still be compact if it carries the time coordinate:

```text
let h[t] = scan[t](step, h0, x[t])
```

The point of the bracket is not syntax for its own sake. It prevents `scan`
from becoming an opaque loop-shaped box. The call says which coordinate orders
the dependency, leaving storage and scheduling for later lowering.

For `T = 4`, the dependency chain is almost too small to hide:

```text
h[1] reads h[0]
h[2] reads h[1]
h[3] reads h[2]
```

That chain is the program's time structure. A loop can execute it, but the
recurrence states it.

## Storage Is a Consequence

Suppose the program eventually asks only for:

```rust
let final = h[T - 1];
```

The recurrence itself needs only `h[t - 1]` to compute `h[t]`. If no later code
observes the full history, an implementation may keep a rolling one-step
window. The source dependency makes that possibility visible.

Now change the observation:

```rust
let all_states[t] = h[t];
```

The recurrence rule did not change, but the storage story did. The program now
observes every time step, so the full sequence must remain available. This is a
clean separation:

```text
dependency relation  what each point needs
observation          which points later code asks to keep
schedule/storage     how an implementation chooses to compute and retain them
```

Traditional loop code often blends those questions. Recurrence pulls
them apart.

## The Error You Want Named

A recurrence can also state mistakes plainly:

```rust
let bad[t in 0..T] = bad[t + 1] + 1;
```

This reads the future. Unless another boundary condition resolves the
dependency, the definition is not a forward recurrence under the stated range.
The problem is present in the index expression `t + 1`; it does not need to
wait for a runtime surprise.

The quick check is simple. Circle every read of the same binding. If the rule
for `h[t]` reads `h[t - 1]`, it points backward. If it reads `h[t + 1]`, it
points forward. A recurrence that points forward needs a different explanation
than ordinary forward evaluation.

This is a dependency-graph question, not only a syntax question. A legal
forward recurrence should admit a topological order over its time points: all
inputs to `h[t]` are available before `h[t]` is computed. The expression
`h[t - 1]` has that property under the range `1..T`. The expression
`h[t + 1]` does not, unless the program is really describing a backward pass, a
boundary-value problem, or a fixed-point equation. Those are valid ideas, but
they deserve different source contracts.

## An RNN Written as Recurrence

The RNN operator in `stdlib/ml/recurrent_ops.ein` is the same dependency idea
after it has met a real model. Here is the core, with bias handling and
activation dispatch trimmed so the dependency is easier to see:

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

The state has three visible coordinates: time `t`, batch `b`, and hidden unit
`h`. The input contribution consumes `i`. The recurrent contribution consumes
`h_prev`. The read of `hidden[t - 1, b, h_prev]` exposes a one-step temporal
dependency while preserving the batch coordinate.

Read one cell:

```text
hidden[7, b, h]
```

It is built from today's input `X[7, b, i]` and yesterday's hidden state
`hidden[6, b, h_prev]`. Time changes. Batch stays fixed. Hidden coordinates
are mixed through the recurrent weights.

That is a lot of information for a compiler, and it is not hidden behind a
Python loop.

## What Does Time Point To?

The RNN excerpt makes three roles visible at once:

```text
t       the time coordinate being defined
i       the input feature coordinate consumed by the input projection
h_prev  the previous hidden coordinate consumed by the recurrent projection
```

The point is not that recurrence avoids loops forever. The point is that the
loop becomes a lowering choice. The source first states the time-shaped
structure. Once those roles are visible, an implementation can choose an
execution order without pretending that the order is the meaning.

## Time as Direction

Time is the first coordinate here with an inherent direction. Batch can usually
be reordered. Feature can often be transformed or reduced. Time in a forward
recurrence carries causality: `t` may depend on `t - 1`, but not on an
unexplained `t + 1`.

That direction changes the meaning of visibility. In earlier chapters, naming
a coordinate helped the compiler check shape and role. Here, naming time helps
the compiler check whether an evaluation order exists. A recurrence rule is a
set of equations, but not every set of equations is a legal forward
computation.

This is why the chapter speaks of a wall. A loop presents time as execution
steps. A recurrence presents time as a dependency axis. Once time is visible as
an axis, later chapters can ask how much history must be stored and how an RNN
mixes previous hidden coordinates into the next state.

There is a second reason time deserves its own part. Time often looks like a
scalar loop counter in code, but semantically it behaves like an axis of the
data. A sequence model has values at many time coordinates. A dynamic program
fills a table along one or more ordered coordinates. A simulation advances a
state across discrete steps.

When those programs are written only as loops, the axis is implicit in control
flow. When they are written as recurrence, the axis becomes part of the value
being defined:

```text
h[t]
```

That tiny shift lets the same questions from earlier chapters return: which
coordinates survive, which are local, and which dependencies are visible? Time
adds direction, but it does not require a new way of reading.

## Time Direction and Batch Isolation

Take a recurrence with one scalar state. The arithmetic is restrained so the
harder question is visible: which way does the dependency edge point?

```rust
let h[0] = 1.0;
let h[t in 1..5] = 0.5 * h[t - 1] + x[t];
```

If the input values are:

```text
x[1] = 10.0
x[2] = 20.0
x[3] = 30.0
x[4] = 40.0
```

then the recurrence expands to:

```text
h[1] = 0.5 * h[0] + x[1]
h[2] = 0.5 * h[1] + x[2]
h[3] = 0.5 * h[2] + x[3]
h[4] = 0.5 * h[3] + x[4]
```

The important fact is not the arithmetic. It is the dependency direction.
Every `h[t]` reads `h[t - 1]`, never `h[t + 1]`. The index expression states
that direction in the value definition itself.

Now imagine the mistaken recurrence:

```rust
let h[t in 1..5] = 0.5 * h[t + 1] + x[t];
```

The shape of the family still looks like a time-indexed sequence. A loop might
even be written that tries to fill it backward. But if the surrounding contract
is forward simulation, the source has made a different claim: the present
depends on the future. A compiler pass that sees recurrence offsets can name
that problem much more directly than a runtime failure inside a loop.

The same trace becomes more interesting with batch:

```rust
let h[0, b] = init[b];
let h[t in 1..T, b] = 0.5 * h[t - 1, b] + x[t, b];
```

For `h[3, 7]`, the previous state is `h[2, 7]`. Batch stays fixed. The
recurrence does not mix `b = 7` with `b = 2`; it only walks backward in time.
That is a semantic boundary. If the formula accidentally read
`h[t - 1, b_prev]` under a `sum[b_prev]`, the model would be communicating
across batch examples. The shape might still be valid, but the meaning would
change.

This is why time as an axis belongs in the same book as reshape and
broadcasting. The question is still about visible coordinates. Which
coordinate is preserved? Which one is shifted? Which one is local? The new
ingredient is order. A spatial coordinate usually has no natural direction;
time does. Naming it lets the compiler and reader inspect not only that a
dependency exists, but also which way it points.

Lowering can then become an implementation choice. A backend may execute the
recurrence with a loop, vectorize independent batch coordinates, or use
special facts about the recurrence window. Those choices come after the source
has stated the dependency graph. The recurrence notation does not replace
execution; it gives execution a visible contract to follow.

## Why the Loop Is Not the Dependency

The same recurrence can be written as a loop:

```python
h = init
for t in range(1, T):
    h = step(h, x[t])
```

For execution, this is natural. For analysis, it fuses several ideas into one
mutable variable: the family of values, the current storage slot, and the time
dependency. The recurrence spelling separates them:

```text
family       h[t]
dependency   h[t] reads h[t - 1]
storage      chosen later
```

That separation is what later chapters use. Storage analysis can ask which
members of `h[t]` are observed. An RNN analysis can ask which non-time
coordinates are preserved while time moves. Autodiff can ask which
intermediate states might be needed backward. Those questions are harder to
ask when the only source object is a variable being overwritten.

The point is not that loops are bad. It is that a loop is an execution story,
while a recurrence is a dependency story. Einlang writes the dependency story
first and lets lowering choose the execution story afterward.

Once that dependency story is visible, a later implementation can still emit
an ordinary loop. The source has not rejected loops; it has delayed them until
after the time axis has been named.

Write one edge, `h[t, b, f]` reading its predecessor. Time moves; the other
coordinates explain what stays isolated. If `h` were only a mutable loop
variable, that distinction would have to be recovered after the fact.

## Try It

In `h[t] = f(x[t], h[t - 1])`, change one term so a time step reads future
information, as in a bidirectional model. Write the forward and backward
families with separate coordinates and ask how the storage demand changes when
future reads become part of the dependency graph.

**Line to keep:** a loop is an execution order; time is a dependency
relationship.
