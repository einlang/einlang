---
layout: book
title: "Storage Follows Observation"
---

# Storage Follows Observation

> "Premature optimization is the root of all evil."
>
> — Donald E. Knuth, "Structured Programming with go to Statements" (1974)

You are debugging an autoregressive model at midnight. The sequence is 10,000
tokens long. Training has been running for an hour, then OOM. The memory profile shows: 10K
time steps × batch 32 × hidden 768, all in float32, all materialized.

Your colleague says "use `torch.no_grad()` for inference." That helps—but it's
the wrong diagnosis. The real question is not whether gradients flow. It's
whether the program ever said it needed 10,000 time steps of stored history.

Chapter 10 established that a recurrence says `h[t]` reads `h[t-1]`; a loop is
one way to run it. Now the harder question: if `h[t]` is a family of values,
which ones must actually occupy memory?

## Definition Is Not Storage

A recurrence defines what values exist. It does not say which ones must be
stored simultaneously. That decision belongs to observation — which members of
the family does later code actually ask for?

This separation — definition from storage, storage from observation — is the
thesis of the book applied to memory. A loop notation merges all three into one
mutable variable. It makes every value look equally real, equally stored. A
recurrence notation separates them: what exists semantically, what must occupy
memory, and what later code actually asks for. The notation determines what you
can notice about storage, and a notation that merges definition with allocation
makes it impossible to ask "which of these values does the program actually
observe?" without tracing every use.

Consider a one-step recurrence:

```rust
let h[0] = init;
let h[t in 1..T] = step(h[t - 1], x[t]);
```

The dependency window is one step: each `h[t]` needs only `h[t-1]`. If the only
observed value is the final state:

```rust
let final = h[T - 1];
```

An implementation can use a single rolling slot:

```text
slot = h[0]
slot = step(slot, x[1])
slot = step(slot, x[2])
slot = step(slot, x[3])
```

The full array `h[0..T]` never exists as stored data. The source family still
contains every `h[t]`—the semantics have not changed—but storage did not follow
definition. It followed observation.

Now change only the observation:

```rust
let trace[t] = h[t];
```

The recurrence is identical. The dependency window is still one step. But the
program now asks for every member of the family. A rolling slot is no longer
enough. The runtime must make all values available.

Same recurrence, different observations, different storage.

## Three Observations, Three Policies

Take a concrete sequence of 8 time steps:

```rust
let h[0] = init;
let h[t in 1..8] = step(h[t - 1], x[t]);
```

**Observation A: final state only.**

```rust
let final = h[7];
```

Policy: rolling slot of size 1. The eight source values exist semantically; the
runtime keeps one at a time.

**Observation B: full trace.**

```rust
let trace[t] = h[t];
```

Policy: full materialization. All eight values must be available simultaneously
because `trace[t]` reads arbitrary `t`.

**Observation C: every other step.**

```rust
let every_other[u] = h[2 * u];
```

Policy: partial materialization. Even-indexed states are retained for output;
odd-indexed states are computed transiently and discarded after their successor
is produced.

The recurrence definition never changed. Only the observation did. That one
difference determines which storage policies remain possible.

Trace one run. `h[0] = 1.0`. The first step reads `h[0]` and `x[1] = 2.0`,
producing `h[1] = 2.5`. The second reads `h[1]` and `x[2] = 0.0`, producing
`h[2] = 1.25`. By step seven, `h[7] = 2.3515625`. Eight values exist in the
mathematical definition. How many occupy memory depends on what later code
asks for:

**Policy A** (only `h[7]` observed): one rolling slot.
`slot` starts at 1.0, becomes 2.5, then 1.25, ..., finally 2.3515625. The
seven intermediate values never coexist in memory.

**Policy B** (full `trace[t] = h[t]`): all eight values materialized.
`[1.0, 2.5, 1.25, 1.625, 0.8125, 1.40625, 0.703125, 2.3515625]`.

**Policy C** (`every_other[u] = h[2*u]`): four values retained.
`h[0]=1.0, h[2]=1.25, h[4]=0.8125, h[6]=0.703125`. The odd-indexed values
`h[1], h[3], h[5], h[7]` are computed transiently when their successors need
them, then discarded.

```
   Storage Follows Observation

   Recurrence: h[t] = 0.5 * h[t-1] + x[t],  h[0] = 1.0

   Values defined (all exist semantically):
   t=0    t=1    t=2    t=3    t=4    t=5    t=6    t=7
   [1.0] [2.5] [1.25] [1.625] [0.81] [1.41] [0.70] [2.35]
     │      │      │       │       │       │       │       │
     └──────┼──────┼───────┼───────┼───────┼───────┼───────┘
            └──────┼───────┼───────┼───────┼───────┘
                   └───────┼───────┼───────┼───────┘
                           └───────┼───────┼───────┘
                                   └───────┼───────┘
                                           └───────┘
   Policy A (observe h[7]):    [·]   [·]   [·]   [·]   [·]   [·]   [2.35]
   One rolling slot.           stored: █ (1 slot rolling)

   Policy B (observe all):     [1.0] [2.5] [1.25] [1.625] [0.81] [1.41] [0.70] [2.35]
   Full materialization.       stored: ████████████████████████████████████████████

   Policy C (observe even):    [1.0]  ·   [1.25]   ·    [0.81]   ·    [0.70]   ·
   Partial materialization.    stored: █     -     █       -      █       -      █     -
                               transient: h[1], h[3], h[5], h[7] computed and discarded
```

Same recurrence. Same values. Three different storage footprints. The
difference is the observation set, and the observation set is visible in the
source.

This is the central thesis of Part III. Definition, storage, and observation
are three separate things. A loop fuses them into one mutable variable: the
variable IS the definition, IS the storage, and IS observed every time it is
read. A recurrence notation splits them apart. The recurrence defines what
exists. The observation says what is needed. Storage is the negotiation
between the two. When the notation withholds this separation, the compiler
cannot negotiate — it can only allocate.

## Print Is Observation Too

Debugging statements are not harmless from a storage perspective:

```rust
let h[0] = init;
let h[t in 1..T] = step(h[t - 1], x[t]);
print(h);
```

Before the `print`, an implementation may have planned a rolling state—no other
code observes intermediate values. After `print(h)`, the program has asked to
witness the entire family. That request changes what must be materialized. A
debugging print is an observation boundary.

This applies beyond time. A shape inspection is also an observation:

```rust
let z[i] = x[i] * x[i];
print(shape(z));
```

This does not demand the values of `z`, but it does demand the extent of `i`.
Both are observations that reduce implementation freedom.

A debugging print is an observation boundary. This fact is invisible in a
loop — `print(h)` looks like a harmless diagnostic. In a recurrence notation,
`print(h)` reveals itself as a request for the full family, and that request
changes what the compiler may discard. The Hiding Law applies to observation
as much as to computation: when the notation hides which values are observed,
the compiler cannot distinguish a rolling window from a full materialization.
The program pays the price in memory, and the programmer pays the price in
surprise.

## A Wider Window

Some recurrences need more than one step of history:

```rust
let y[0] = a;
let y[1] = b;
let y[t in 2..T] = y[t - 1] + y[t - 2];
```

The dependency window is two steps. If only `y[T - 1]` is observed, a two-slot
rolling buffer suffices. The source makes the window visible through the offsets
`t - 1` and `t - 2`. The compiler does not need to recover this fact from a
loop body.

## The Caveat: Observation Is Not Only Output

The largest backward offset gives the local dependency window for forward
computation. But storage is not determined by that window alone. Other forces
may demand more:

- **Autodiff**: reverse-mode differentiation needs intermediate states for the
  backward pass, even if the forward pass only observes the final state.
- **Debugging**: a `print(h)` or a trace request changes the observation set.
- **Checkpointing**: a policy may store some states and recompute others, trading
  memory for recomputation.

Visible recurrence does not solve these trade-offs. It makes the first fact
explicit: what each point needs in order to be computed. Once that fact is
visible, storage choices become compiler/runtime policy rather than guesswork
hidden inside a loop.

## Checkpointing as Policy

Training with backpropagation through time is the clearest case. The forward
pass may observe only the final state, but the backward pass needs every
intermediate activation. Keeping all states is simple but memory-heavy.
Recomputing every state from scratch is memory-light but expensive.
Checkpointing chooses points in between.

The recurrence notation does not choose the checkpoint schedule. It gives the
schedule the graph it needs:

```text
h[t] needs h[t - 1]                       // one-step dependency
y[t] needs y[t - 1] and y[t - 2]          // two-step dependency
```

With these facts, the compiler can reason about legal recomputation. A schedule
might store `h[0]`, `h[4]`, and `h[7]`, then recompute segments when the
backward pass needs intermediate values. Without explicit dependency edges, the
compiler must recover this structure from a loop body and mutation pattern.

Both JAX and PyTorch give the programmer a manual checkpointing primitive.
`jax.checkpoint` (formerly `jax.remat`) wraps a function and tells the autodiff
engine: do not store intermediate activations from this function; recompute them
during the backward pass. `torch.utils.checkpoint.checkpoint` does the same
thing with a slightly different contract — it takes `function, *args` and
returns outputs, discarding intermediate activations. Both work. Both let you
trade memory for compute. But both require the programmer to decide which
functions to checkpoint, with which stride, and whether the recomputation
dependency window is satisfied. The stride constraint from this chapter — that
a checkpoint stride larger than the dependency window makes recomputation
exponentially expensive — is invisible in these APIs. The programmer receives
no warning when stride exceeds window + 1. The shapes are correct. The code
runs. The memory savings appear in the profiler. The recomputation overhead
appears weeks later as a training throughput regression that nobody can trace
to the checkpoint annotation. When the dependency window is visible in the
source, the compiler can at least warn. When it is hidden inside a function
body, the compiler can only execute.

## The Minimal Compiler Rule

A practical compiler can begin with a conservative rule:

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

Real systems layer aliasing, autodiff, debugging, and checkpointing on top. The
sketch is still useful because it shows the compiler question: first compute the
dependency window, then compute the observation set, then choose storage.

Do not confuse a recurrence definition with an array allocation request. Writing
`h[t] = ...` defines a family of values. It does not demand that every `h[t]` be
stored simultaneously. Whether the family is materialized depends on later uses
and runtime policy. The better the dependency information, the more room the
implementation has to make that decision responsibly.

A loop gives the compiler one instruction: allocate and overwrite. A
recurrence gives the compiler a dependency graph and an observation set, and
asks: what is the cheapest way to satisfy the observations? The first admits
only one storage policy. The second admits as many policies as there are
points on the trade-off between memory and recomputation. The notation does
not choose the policy. It makes the choice possible.

## Try It

Before you solve the storage problem, write the version that runs but is wrong.
This is the trap. It teaches more than the correct answer.

### The Trap: Materialize Everything

You have a 10-step recurrence. Only `h[9]` is used in the final loss:

```text
let h[0] = init;
let h[t in 1..9] = step(h[t - 1], x[t]);
loss = loss_fn(h[9], target);
```

Write the storage plan that keeps every `h[t]` in memory. All ten values. The
shapes are all correct. The program runs. The loss computes. What is hidden?

The hidden waste: you defined 10 values and observed 1, but stored 10. The
compiler can't help because you never stated which values are observed — you
just allocated an array. Storage didn't follow observation. Storage followed
the definition.

This is the shape-compatible wrong version of a storage plan. It runs. It
computes the right loss. It uses ten times the memory it needs, and for a
10-step recurrence nobody cares. For a 10,000-step recurrence, it's the
difference between a program that fits in RAM and one that doesn't.

Now write the coordinate reading that diagnoses it:

```text
defined: h[0], h[1], ..., h[9]          (10 values)
observed: h[9]                           (1 value, via loss_fn)
stored: all 10                           (materialized)
waste: 9 values stored, never observed   (90% of allocation)
```

The fix: state the observation. `let final = h[9]` tells the compiler that only
one value escapes. A compiler that respects observation can then choose a
rolling slot — one location, overwritten 10 times.

### The Second Trap: Wrong Stride

Now a recurrence with a 2-step dependency window:

```text
let h[0] = init;
let h[t in 1..6] = step(h[t - 1], h[t - 2], x[t]);
```

You decide to checkpoint with stride 3. Checkpoints at t=0 and t=3. During the
backward pass, you need `h[5]`. It's not stored. You recompute from `h[3]`:
`h[3] → h[4] → h[5]`. For `h[4]`, you need `h[3]` (stored) and `h[2]`
(transient). You recompute `h[2]` from... what? The nearest earlier checkpoint
is `h[0]`. To get `h[2]`, you need `h[1]` and `h[0]`. To get `h[1]`, you need
`h[0]` (stored) and `h[-1]` — which doesn't exist.

The shapes are all correct. The recurrence definition is correct. The stride
looks reasonable. But the backward pass hits an index out of bounds because the
checkpoint stride didn't account for the dependency window.

Write the coordinate reading:

```text
dependency window: h[t] reads h[t-1], h[t-2]  → window = 2
stride: 3
constraint: stride must be ≤ window + 1 for recomputability
           3 > 2 + 1 = 3 → stride 3 is at the boundary
actual problem: recomputing h[2] from h[0] requires h[1],
               which requires h[0] and h[-1] — out of bounds
valid strides for window=2: stride ≤ 3, and with stride=3,
               h[1] and h[2] must be recomputed from h[0],
               but h[1] needs h[-1] — FAIL
```

The minimum safe stride for a window of size w is 1 (store everything).
Stride s is safe when there exists a path from every `h[t]` to the nearest
stored checkpoint using only values that are either stored or can be
recomputed. For window=2, the practical maximum stride with one level of
recomputation is also 2 — because you need two stored values to recompute a
state that depends on two predecessors. With stride=3, you'd need to go back
two transient steps from h[3] to h[0], and the dependency on h[-1] breaks.

Now write the three valid storage plans for this recurrence:

```text
stride=1: store all 7 values → 7 stored, 0 transient
stride=2: store h[0], h[2], h[4], h[6] → 4 stored, max 1 transient
          recompute h[1] from h[0], h[3] from h[2], h[5] from h[4]  ✓
stride=3: store h[0], h[3], h[6] → 3 stored
          recompute h[1] from h[0] ✓
          recompute h[2] from h[1], h[0] ✓ (h[1] just recomputed)
          recompute h[4] from h[3], h[2] — h[2] is now gone
          need to recompute h[2] AGAIN from h[1], h[0] — h[1] is gone
          chain grows exponentially → recompute cost dominates
```

So stride=3 is technically possible but the recomputation cost explodes because
transient values from earlier chains are discarded before later chains can reuse
them. The practical limit is stride ≤ window + 1 with single recomputation, and
stride ≤ window for efficient recomputation.

### The Payoff

Now return to the recurrence from the start of this chapter, and apply the
observation rule to two different programs:

```text
let h[0] = init;
let h[t in 1..8] = step(h[t - 1], x[t]);
```

First program: observe only `h[7]`. Second program: bind `trace[t] = h[t]`.
For each, sketch the storage plan a compiler could choose: rolling state, full
materialization, or a hybrid. The answers are already in the chapter — the
exercise is to state them as coordinate readings.

Now add a third version with a print:

```text
let h[0] = init;
let h[t in 1..8] = step(h[t - 1], x[t]);
let final = h[7];
print(h);
```

Before the `print`, the compiler could use a rolling slot—only `final` is
observed. After `print(h)`, the observation set changed. Does the compiler
materialize the full history, or does it stream values during evaluation? The
answer depends on whether the compiler can evaluate-and-print in a single pass.
If it can, the rolling slot still works. If the print expects the full array,
materialization is forced.

A training loop for an RNN observes the full output sequence for the loss
function. The forward pass needs every `h[t]`. Does the observation distinction
from this chapter still matter, or is full materialization forced by the loss?
The loss forces observation of every `h[t]`, so the minimal storage plan IS full
materialization — but only during the forward pass. Once the forward pass
completes and the loss is computed, the backward pass also needs every `h[t]`.
You could store them all (simple, memory-heavy) or checkpoint some and recompute
others (complex, memory-light). The observation distinction tells you which
`h[t]` the forward pass itself needs, which is the minimum set you must either
store or be able to recompute. When does checkpointing become relevant? When the
sequence length is long enough that storing all `h[t]` exceeds memory. The
checkpointing policy stores every k-th state and recomputes the intermediate
states during the backward pass. This is a storage policy, not a change to the
recurrence.

Now implement gradient checkpointing using coordinate logic. Use the same 8-step
recurrence from earlier in this chapter. A forward recurrence `h[t]` saves every
`stride`-th state. For `stride = 3`:

```text
h[0] = 1.0        → stored (stride 0)
h[1] = 2.5        → transient
h[2] = 1.25       → transient
h[3] = 1.625      → stored (stride 1)
h[4] = 0.8125     → transient
h[5] = 1.40625    → transient
h[6] = 0.703125   → stored (stride 2)
h[7] = 2.3515625  → transient (or stored if needed as final output)
```

Eight values defined. Three stored. Five recomputed on demand.

During the backward pass, the autodiff engine needs `h[t]` for every `t` to
compute the gradient of `step` with respect to its parameters. It starts at
`t=7`:

```text
t=7: needs h[7] → not stored, recompute from h[6]:
     h[7] = 0.5 * 0.703125 + 2.0 = 2.3515625  ✓
t=6: needs h[6] → stored (stride 2), use directly
t=5: needs h[5] → not stored, recompute from h[6]:
     h[5] = 0.5 * h[4] + 1.0 → need h[4] first
     h[4] = 0.5 * h[3] + 0.0 → h[3] is stored (stride 1)
          = 0.5 * 1.625 + 0.0 = 0.8125
     h[5] = 0.5 * 0.8125 + 1.0 = 1.40625  ✓
t=4: needs h[4] → just recomputed above, reuse
t=3: needs h[3] → stored (stride 1), use directly
t=2: needs h[2] → not stored, recompute from h[3]:
     h[2] = 0.5 * h[1] + 0.0 → need h[1] first
     h[1] = 0.5 * h[0] + 2.0 → h[0] is stored (stride 0)
          = 0.5 * 1.0 + 2.0 = 2.5
     h[2] = 0.5 * 2.5 + 0.0 = 1.25  ✓
t=1: needs h[1] → just recomputed above, reuse
t=0: needs h[0] → stored (stride 0), use directly
```

The backward pass visits `t=7,6,5,4,3,2,1,0`. At each step, if the state is
stored, use it. If not, recompute from the nearest earlier checkpoint. For
stride 3 and a 1-step dependency window, the maximum recomputation chain length
is `stride - 1 = 2` steps (recomputing `h[5]` required recomputing `h[4]`
first).

Now trace the memory at each backward step:

```text
Backward step    Stored in memory         Recomputed transient     Total in RAM
─────────────────────────────────────────────────────────────────────────────────
t=7 compute      h[0], h[3], h[6]         h[7]                    4 values
t=7 grad done    h[0], h[3], h[6]         —                       3 values
t=6 grad done    h[0], h[3], h[6]         —                       3 values
t=5 compute      h[0], h[3], h[6]         h[4], h[5]              5 values (peak)
t=5 grad done    h[0], h[3], h[6]         h[4] (keep for t=4)     4 values
t=4 grad done    h[0], h[3], h[6]         —                       3 values
t=3 grad done    h[0], h[3], h[6]         —                       3 values
t=2 compute      h[0], h[3], h[6]         h[1], h[2]              5 values (peak)
t=2 grad done    h[0], h[3], h[6]         h[1] (keep for t=1)     4 values
t=1 grad done    h[0], h[3], h[6]         —                       3 values
t=0 grad done    h[0], h[3], h[6]         —                       3 values
```

Full materialization would keep all 8 values. Checkpointing with stride 3 keeps
3 stored plus up to 2 transient — 5 values maximum. For a 100-step sequence with
stride 10, the maximum memory is 10 stored + 9 transient = 19 values instead of
100. The trade is compute: each transient chain of length up to 9 must be
re-run.

Visualize the checkpointing schedule as a coordinate grid:

```
   Checkpointing with stride=3, 8 time steps

   t=0    t=1    t=2    t=3    t=4    t=5    t=6    t=7
   [█]    [·]    [·]    [█]    [·]    [·]    [█]    [·]
    S              S              S              S

   █ = stored (stride-aligned),  S = checkpoint
   · = recomputed on demand during backward pass

   Backward pass recomputation chains:
   t=7: h[6] → h[7]                                          (1 step)
   t=5: h[3] → h[4] → h[5]                                   (2 steps)
   t=2: h[0] → h[1] → h[2]                                   (2 steps)

   Maximum chain length = stride - 1 = 2
   Storage factor = 1 / stride ≈ 1/3 of full materialization
```

This is the separation this chapter fought for. The recurrence `h[t] =
step(h[t-1], x[t])` defines what values exist. The stride coordinate decides
which ones are kept. The recurrence formula itself does not change when the
stride changes — only the checkpointing policy changes. A loop buries this
choice in mutable state and manual save/restore logic. A recurrence with a
stride coordinate makes the policy visible in one parameter.

The minimal stride is determined by the dependency window, not by the
recurrence formula. For `h[t] = step(h[t-1], x[t])`, the backward dependency
window is 1 step — each state needs only its immediate predecessor for
recomputation. For `h[t] = step(h[t-1], h[t-2], x[t])`, the window is 2 steps,
and the maximum chain length becomes `stride - 2`. A compiler can compute this
from the recurrence definition: scan the index offsets in the definition of
`h[t]`, find the largest backward offset `k` such that `h[t-k]` appears, and
set the minimum stride constraint accordingly.

**Line to keep:** storage allocation is a negotiation between what is defined
and what is observed.

### Where This Leads

Recurrence, observation, and stride are three separate things. A loop merges them
into one mutable variable — the variable IS the definition, IS the storage, IS the
observation. Named coordinates keep them separate, so each can be reasoned about
independently. The compiler cannot optimize storage it cannot observe. The
programmer cannot control storage the notation does not let them name.

Storage follows observation. The rule is simple until you draw the dependency
graph for a real model. An RNN's recurrence reads one step back — a tiny window.
But training with backpropagation through time needs every intermediate state.
The observation set ballooned, and storage followed. What changed was not the
recurrence. It was which values someone asked to see. The notation that separates
definition from observation can answer that question at a glance. The notation
that merges them can only allocate — and hope.

Chapter 12 draws the full RNN dependency graph: time edges, weight edges, batch
isolation, and the backward edges that autodiff adds.