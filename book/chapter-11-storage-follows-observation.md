---
layout: book
title: "Chapter 11: Storage Follows Observation"
---

# Storage Follows Observation

A recurrence tells us which values depend on which earlier values. It does not
automatically tell us how much history must be stored. Storage depends on two
things together: what each point reads, and which points later code observes.

That separation is easy to lose if we imagine a full array allocation the
moment we see `h[t]`. But the source family `h[t]` is not automatically a demand
to store every `h`. Storage depends on the dependency window and on which
members of the family are later observed.

The mistake is to let one notation carry three different burdens at once:
meaning, dependency, and storage. `h[t]` names a member of a family. The read
of `h[t - 1]` states a dependency window. A later use such as `h[T - 1]` or
`trace[t] = h[t]` states an observation. Only after those facts are separated
does the buffer question have a well-formed answer.

## Array Semantics or Family Semantics

If an indexed recurrence is treated immediately as an array allocation, the
language has already chosen a storage policy. Every time point appears to need
materialization because every time point has a name. That is simple, but it
throws away an optimization opportunity before analysis begins.

The other choice is to treat the recurrence first as a family of values. The
family has a dependency relation; storage is chosen later, after observations
are known. This lets the same source definition support a rolling state, a full
history, or a checkpointed evaluation depending on how the value is used.

Einlang's recurrence notation takes the second route. Define the values first;
commit to memory only after the program shows what it will observe.

This is a semantic claim first and an implementation hook second. The current
compiler performs recurrence ordering and records lowered execution facts that
backends can use, including vectorized and recurrence-aware execution paths.
The book's checkpointing examples describe the policy space those facts make
available; they are not a promise that every checkpoint schedule is already a
separate user-visible pass.

Consider:

```rust
let h[0] = init;
let h[t in 1..T] = step(h[t - 1], x[t]);
let final = h[T - 1];
```

The recurrence reads only one step back. If the only observed value is
`final`, an implementation can evaluate the sequence with a rolling slot:

```text
previous -> current -> previous -> current
```

The full array `h[0..T]` does not have to exist as stored data, even though it
exists as a family of values in the source model. This is a distinction between
meaning and storage: the recurrence defines all of the values, but the runtime
may not need to retain all of them.

Ask for the same recurrence twice. First ask only for the last state. Then ask
for a plot of every state. The recurrence did not change, but the storage
obligation did. Observation is not an afterthought; it is half of the storage
story.

## A Small Execution Trace

Take four time steps:

```text
h[0] = init
h[1] = step(h[0], x[1])
h[2] = step(h[1], x[2])
h[3] = step(h[2], x[3])
```

If the program only asks for `h[3]`, the runtime can overwrite:

```text
slot = h[0]
slot = step(slot, x[1])
slot = step(slot, x[2])
slot = step(slot, x[3])
```

This execution does not deny that `h[1]` and `h[2]` are meaningful source
values. It only says they do not need long-term storage for this observation.
That distinction is hard to make when the source program is already written as
mutation. The recurrence gives names to the whole family first; the storage
plan comes second.

Put differently, there are two valid views of the same definition:

```text
source family: h[0], h[1], h[2], h[3]
runtime slots: previous, current
```

The first is the semantic object. The second is one legal storage plan under a
particular observation.

This is a small victory for the old separation between interface and
implementation. The source family says what values exist as a mathematical
object. The storage plan says which of those values must occupy memory at the
same time. That is similar in spirit to separating a lazy value from its
eventual evaluation, or a type signature from a compiled layout. The important
thing is that the source can talk about `h[t]` without immediately demanding a
full array.

## When History Is Observed

Now change only the observation:

```rust
let all[t] = h[t];
```

The recurrence is the same. The dependency offset is still `t - 1`. But the
program now asks for every member of the family, so the implementation must
make the whole history available.

This distinction is important:

```text
definition  what values mean
observation which values are demanded
storage     what the runtime must retain
```

In a loop, these concerns are often entangled with mutation. In a recurrence,
they can be discussed separately. That separation is useful even when the final
runtime still lowers the recurrence to a loop.

## Print Is Observation Too

Observation is not only a model output. Debugging can observe a value as well:

```rust
let h[0] = init;
let h[t in 1..T] = step(h[t - 1], x[t]);
print(h);
```

Before the `print`, an implementation may have room to keep only a rolling
state if no later expression needs the full history. After `print(h)`, the
program has asked to see the family. That request changes what must be
evaluated and materialized. A debugging statement is therefore part of
the storage story.

This is why printing is more subtle in a language of formulas. Printing is not
only display. It is an observation boundary. It turns an expression, a
recurrence family, or a derivative request into a concrete
witness.

For a smaller example:

```rust
let z[i] = x[i] * x[i];
print(z);
print(shape(z));
```

The first line defines a family of values. The print asks the runtime to show
that family. The shape print asks for a different witness: not the values, but
the coordinate extent. Both observations reduce implementation freedom because
the program has made a demand.

Coordinate functions fit into the same discipline. A call such as
`move_channel[channel](image)` or `scan[t](step, h0, x)` states a coordinate
contract; it does not by itself demand a particular buffer layout. The
implementation may choose a view, a copy, a fused loop, a rolling state, or a
checkpointed schedule if the later observations permit it. What the function
boundary preserves is the semantic fact: which coordinate moved, which
coordinate ordered the recurrence, and which surrounding coordinates survived.

## A Wider Window

Some recurrences need more history:

```rust
let y[0] = a;
let y[1] = b;
let y[t in 2..T] = y[t - 1] + y[t - 2];
```

If only the final value is observed, a two-value rolling window is enough. The
source makes that visible through the offsets `t - 1` and `t - 2`.

If a later expression reads every `y[t]`, the storage plan changes. The
meaning of the recurrence does not.

## The Caveat

The largest backward offset is not a complete storage plan. It is only the
local dependency window. Other forces may require more:

```text
later code may observe intermediate states
reverse-mode differentiation may need saved activations
debugging may request a trace
checkpointing may trade recomputation for memory
```

Visible recurrence does not solve those trade-offs by itself. It makes the
first fact explicit: what each point needs in order to be computed. Once that
fact is visible, storage choices become compiler/runtime policy rather than
guesswork hidden inside a loop.

## Checkpointing as a Policy

Reverse-mode differentiation is the clearest case where the caveat matters. A
training run may need intermediate states during the backward pass. Keeping
every state is simple but memory-heavy. Recomputing every state is memory-light
but expensive. Checkpointing chooses points in between: save some states,
recompute segments as needed.

The recurrence notation does not choose the checkpoint policy. It gives the
policy the dependency facts:

```text
h[t] needs h[t - 1]
y[t] needs y[t - 1] and y[t - 2]
```

With those facts, the compiler or runtime can reason about legal
recomputation. Without them, it has to recover the temporal structure from a
loop body and mutation pattern.

This is why the chapter says storage follows observation, not "storage is
solved by recurrence." Observation includes ordinary outputs, derivative
requests, debug traces, and profiling hooks. Each observation changes what
must be retained or recomputed.

## Which Observations Force Memory?

The largest backward offset in a recurrence gives the local history window for
forward computation. But storage is not determined by that window alone. Later
uses, derivative requests, and debugging observations can demand more storage
than the recurrence itself. The source tells us the dependency; observation
tells us how much of the family must become concrete.

## Definition Before Storage

The storage discussion keeps the recurrence story honest. It would be too easy
to say "visible dependency offsets imply storage optimization" and stop there.
The truth is subtler: visible offsets give the compiler a fact it can use, but
storage remains a policy decision under observation.

That distinction is important for production systems. A compiler should be
able to use the one-step dependency of an RNN when only the final state is
needed. It should also be able to keep all states when the output sequence is
requested. It should be able to choose checkpointing when differentiation makes
the memory/recompute trade-off worthwhile.

The source notation should not pretend those policies are all the same. It
should expose the dependency relation clearly enough that different policies
can be justified. Visible structure does not remove implementation choices; it
makes their inputs explicit.

A useful warning follows from this: do not confuse a recurrence definition with
an array allocation request. The notation:

```text
h[t] = ...
```

defines a family of values. It does not necessarily demand that every `h[t]`
be stored simultaneously. Whether the family is materialized depends on later
uses and runtime policy.

That distinction is one of the quiet benefits of a declarative source form. It
lets the program describe the mathematical object first and lets the
implementation decide how much of that object must become memory at once. The
better the dependency information, the more room the implementation has to make
that decision responsibly.

The practical advice is therefore conservative: never infer storage only from
the surface existence of an indexed family. First read the dependency window.
Then read the observations. Only then talk about materialization.

## Same Recurrence, Different Observations

Consider this recurrence. The dependency is simple; the storage question is
not, because observation changes the contract:

```rust
let h[0] = init;
let h[t in 1..8] = step(h[t - 1], x[t]);
```

The dependency window is one step. That fact alone permits a rolling execution
for a final-state query:

```rust
let final = h[7];
```

An implementation can keep a single live state:

```text
slot = h[0]
slot = step(slot, x[1])
slot = step(slot, x[2])
...
slot = step(slot, x[7])
```

The source family still contains `h[0]` through `h[7]`. The storage plan does
not. The difference is legal because the observation only demands the last
member of the family.

Now add one line:

```rust
let trace[t] = h[t];
```

The recurrence definition has not changed. The dependency window is still one
step. But the observation now demands every time point, so the runtime must
make all eight values available. A rolling slot is no longer enough unless the
runtime also emits or stores each value as it is produced.

A third observation changes the policy again:

```rust
let every_other[u] = h[2 * u];
```

Now not every time point is externally needed. A storage planner could retain
only even states for the output while still using odd states transiently during
computation. The source recurrence and the observation together determine the
obligation.

This is the audit:

```text
definition    h[t] depends on h[t - 1]
observation   final, trace, or every_other
policy        rolling slot, full history, partial materialization
```

Checkpointing is a more complex version of the same separation. Suppose
a reverse pass needs intermediate states, but memory cannot hold all of them.
The dependency graph says which states can be recomputed from which earlier
states. A checkpoint schedule might store `h[0]`, `h[4]`, and `h[7]`, then
recompute segments when the backward computation needs them. The recurrence
notation does not choose that schedule, but it gives the schedule a graph that
is explicit rather than recovered from mutation.

Debugging belongs in the same analysis. A line such as:

```rust
print(h);
```

is not harmless from the storage point of view. It observes the family. If a
compiler had planned to keep only the final state, the print changes the
contract. The user has asked for a witness of all `h[t]` values, so the
runtime must either materialize them or produce them in an observable stream.

This concrete audit prevents two opposite mistakes. The first mistake is to
assume every indexed family is an array allocation. That gives up optimization
too early. The second is to assume a small dependency window always means
small memory. That ignores observations, debugging, and autodiff. The visible
source form lets the program state the family first and lets implementation
policy respond to what is actually demanded.

## The Minimal Compiler Rule

A practical compiler can begin with a conservative rule:

```text
dependency window gives what is needed next
observation set gives what must remain available
```

For `h[t]` reading `h[t - 1]`, the next-step dependency is small. For
`print(h)` or `let trace[t] = h[t]`, the observation set is large. For
`let final = h[T - 1]`, the observation set is small. This rule does not solve
all memory planning, but it prevents the most common confusion: source
families are semantic objects, not automatic allocation demands.

Once that distinction is stable, more advanced policies can be discussed
without changing the source meaning. Full materialization, rolling buffers,
streaming output, and checkpointing are different answers to the same visible
dependency and observation facts.

This is one reason coordinate-aware helpers are worth more than convenience.
They allow the source to become more compact without collapsing the difference
between meaning and storage. A shape-only helper often forces a reader to ask,
"Was this a layout trick or a semantic transformation?" A coordinate function
answers at the boundary: the semantic transformation is the named coordinate
contract; the layout trick is still up to lowering.

A minimal pass can be sketched like this:

```text
for each recurrence family h:
    window = largest backward offset read by h's rules
    observations = all later reads of h outside the recurrence rule

    if observations == {last time point} and window is finite:
        choose rolling buffer of size window
    else if observations include all h[t]:
        materialize full history
    else:
        materialize observed points and keep transient window for evaluation
```

Real systems need aliasing, autodiff, debugging, and checkpoint policy layered
on top. The sketch is still useful because it shows the compiler question:
first compute the dependency window, then compute the observation set, then
choose storage.

`let final = h[T - 1]` and `let trace[t] = h[t]` may share the same recurrence.
They do not make the same observation. That one difference is enough to change
which storage choices remain possible.

## Try It

Write the same RNN recurrence twice. In the first program, observe only
`h[T - 1]`. In the second, bind `trace[t] = h[t]`. For each program, sketch the
storage plan a compiler could choose: rolling state, full materialization, or a
hybrid. The recurrence definition is the same; the observation changed.

**Line to keep:** storage allocation is a negotiation between what is defined
and what is observed.
