---
layout: book
title: "An RNN Is a Dependency Graph"
---

# An RNN Is a Dependency Graph

A colleague reviews your RNN code:

```python
h = h0
for t in range(T):
    h = tanh(W_h @ h + W_x @ x[t])
```

She pauses at `@ h`. "Are you sure `hidden[t]` reads `hidden[t-1]` from the
same batch element? Or is `h` accidentally mixing batch and hidden dimensions?"

You check. The shapes are `(batch, hidden)`. `W_h @ h` contracts the hidden
dimension. Batch stays independent. But this is knowledge in your head, not in
the code. The loop says `h` is overwritten; it doesn't say `hidden[t, b, h_cur]`
reads `hidden[t-1, b, h_prev]`. A shape checker sees `(batch, hidden) × (hidden,
hidden) → (batch, hidden)` and nods. Shape is a silhouette—it tells you
dimensions match, not what each dimension means.

## Read One Cell, Draw One Edge

The RNN standard-library core, with bias and activation dispatch trimmed:

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

Pick one cell: `hidden[7, 3, 10]`. The input term reads:

```text
X[7, 3, i]              for every input feature i
```

The recurrent term reads:

```text
hidden[6, 3, h_prev]    for every previous hidden unit h_prev
```

```
   RNN Dependency Graph

   Time -->  t=0      t=1      t=2      t=3
             |        |        |        |
   x[t] ----> h[0] -> h[1] -> h[2] -> h[3] ---> y[t]
              ^        ^        ^        ^
              |        |        |        |
          init_h    h[0]     h[1]     h[2]

   h and h_prev: same size, different roles
   h[t] is the coordinate being produced now
   h_prev is the coordinate being scanned from t-1
   Dependency direction distinguishes them.
```

Three relationships in two lines:

```text
time:       t=7 reads t=6          (backward one step)
batch:      b=3 stays b=3          (no cross-example communication)
hidden:     h_prev is scanned, h=10 is produced    (role change through R)
```

The two sums have different meanings. The first mixes input features at the
current time. The second mixes previous hidden units from the previous time.
Both are sums, but calling them both "axis 2" would lose that distinction.

This is not a naming preference. It is the central thesis of the book applied
to a single cell of a single layer. Three relationships in two lines — time
direction, batch isolation, hidden-unit role — are each visible as a
coordinate name. Change one name and the dependency edge changes. Remove the
names and all three distinctions collapse into dimension integers. The
reviewer, the compiler, and the storage planner all need the same facts. The
notation either gives them the facts or forces them to reconstruct the facts
from a loop body and a prayer.

## The Dependency Edge in One Line

The core edge:

```text
hidden[t, b, h] ← hidden[t - 1, b, h_prev]
```

Read that as a dependency figure, not just notation. Three facts on one edge:

- **Time**: `t` reads `t - 1`. Direction is backward. Causality is visible.
- **Batch**: `b` stays `b`. Examples do not communicate.
- **Hidden role**: `h_prev` (previous unit) is scanned into `h` (current unit)
  through `R[h, h_prev]`.

Change one label and the graph changes:

```text
hidden[t, b, h] ← hidden[t - 1, b, h_prev]     correct: same batch
hidden[t, b, h] ← hidden[t - 1, b_prev, h_prev]  wrong: batches communicate
hidden[t, b, h] ← hidden[t, b, h_prev]           wrong: reads current time
hidden[t, b, h] ← hidden[t + 1, b, h_prev]       wrong: reads future
```

A shape checker sees compatible dimensions for all four. The named graph
distinguishes them.

## Square Weights Hide Direction

The recurrent matrix `R[h, h_prev]` is square when `hidden_size` equals itself.
This makes a transposition invisible to shape checks:

```text
correct:   sum[h_prev](R[h, h_prev] * hidden[t - 1, b, h_prev])
bug:        sum[h_prev](R[h_prev, h] * hidden[t - 1, b, h_prev])
```

Both produce an array of shape `[batch, hidden]`. Both are numerically nonzero.
But the correct version reads row `h` of `R`, collecting influence from all
previous units into the current unit. The bug reads column `h`, transposing the
source-target relationship.

If `R` were a learned matrix, both versions would train. Both loss curves would
descend. The only difference is that one learns the forward transition and the
other learns its transpose. That difference is invisible to dimensional
analysis but visible in the coordinate names.

## Two Hidden Coordinates, One Hidden Size

The RNN uses two coordinate names for hidden units: `h` and `h_prev`. Both range
over `0..hidden_size`. Why not reuse `h`?

Because they play different roles. `h` is the coordinate being produced—it
selects a row of `W` and `R`. `h_prev` is the coordinate being scanned—it
selects a column of `R` and a position in the previous hidden state. If both
used the same name, the compiler could not distinguish "produce this unit" from
"scan previous units." The role distinction lives in the names.

The same principle applies to the base case:

```rust
let hidden[0, b, h] = initial_h[b, h];
```

No time dependency. No `h_prev` scan. The base case anchors the family at
`t = 0`. The recurrence starts at `t = 1` precisely because `hidden[t - 1]`
needs `hidden[0]` for the first step. If the base case had shape `[h, b]` or
omitted `b`, the error would be semantic before it was numerical.

Two coordinate names, one integer size. `h` and `h_prev` both range over
`0..hidden_size`, but they mean different things. The question is not whether
the integers match. It is whether the notation has a place for the fact that
they represent different roles — producer and consumer, current unit and
previous unit, row index and column index. When the notation has no place for
that fact, the reader must supply it from memory. The Hiding Law does not
care whether the fact is hidden by omission or hidden by overloading. It only
cares whether later reasoning can recover it.

## Output Is Observation

The standard RNN returns both the full sequence and the final state:

```rust
let Y[t, b, h] = hidden[t, b, h];
let Y_h[b, h] = hidden[seq_length - 1, b, h];
```

Two observations, same recurrence. `Y` asks for every time step. `Y_h` asks only
for the last one. As Chapter 11 argued, that observation matters for storage.
The recurrence defines the hidden family; the outputs decide which parts are
externally visible.

## What a Loop Hides

Return to the loop:

```python
h = h0
for t in range(T):
    h = tanh(W_h @ h + W_x @ x[t])
```

From this code, a compiler cannot answer:

```text
- Does time move backward or forward?
- Do batch examples communicate?
- Does the recurrent matrix map h_prev→h or h→h_prev?
- Is the base case a separate value or the first state of a mutable variable?
```

Each answer requires tracing into `tanh`, into the matrix multiply, into the
initialization. The recurrence spelling answers all four at the binding site.

The loop is not wrong. It is one way to run a recurrence. But it is an execution
story, not a dependency story. A loop tells you what runs. A recurrence tells you
what depends on what. They are not the same, and the difference is the difference
between a program you can optimize and a program you can only execute.

Execution and meaning are different things. A loop records what the machine
did. A recurrence records what the values depend on. Both can produce the same
numbers. But the dependency story is what the reviewer, the compiler, and the
storage planner all need to verify. When the notation records only the
execution, those three readers must each reconstruct the dependency graph
independently — and each may reconstruct a different graph. The notation
determines not just what you can notice, but what you and your tools can
agree on.

## Masking: Time with Gaps

Variable-length sequences add one more coordinate fact. A mask over time and
batch:

```rust
let hidden[t in 1..seq_length, b, h] =
    if mask[t, b] {
        candidate[t, b, h]
    } else {
        hidden[t - 1, b, h]
    };
```

The mask is not just a boolean array of compatible shape. It says "this time
point in this batch example is real." At a padded position, the hidden state is
carried forward unchanged. The dependency edge still reads `t - 1`, same `b`.
The mask selects between update and carry, but it does not change the
communication graph.

## The Recurrence Contract Checklist

Every recurrence must satisfy four checks. You can verify them by reading the
binding site, without tracing into the step function's body:

```text
1. Base case exists:   hidden[0, ...]   is defined (anchor)
2. Time goes backward:  hidden[t] reads hidden[t - n] for n > 0
3. Batch stays fixed:   hidden[t, b] reads hidden[t - n, b] — same b
4. Roles are distinct:  h (output unit) ≠ h_prev (scanned unit)
```

These are not runtime assertions. They are coordinate-level facts visible in
the source. A compiler can check them mechanically:

- Check 1: Does the family have a definition at the start of the range?
- Check 2: Does every read of the family use an index `< t`?
- Check 3: Does the same batch coordinate appear in both the left-hand side
  and every read of the recurrence family?
- Check 4: Does the output coordinate name differ from the scanned coordinate
  name in the recurrent term?

Violating any of the four produces a valid program with wrong semantics. The
loss goes down. The shapes are fine. Only the coordinate names reveal the
violation.

The four checks are not rules you memorize. They are facts the recurrence
notation makes visible. A loop body can satisfy all four or violate all four,
and the reviewer cannot tell without tracing the mutable variable through
every operation. The recurrence notation makes each check a local property of
the binding site. This is the Hiding Law in its operational form: the
notation should make the contract checkable at the point where the contract
is declared. When it does not, verification becomes archaeology — and
archaeology is what reviewers do when the notation has already failed them.

## Scaling to Bigger Cells

An LSTM adds a cell state and several gates. A GRU adds reset and update gates.
The named pieces multiply, but the questions do not change:

```text
Which state is from the previous time step?
Which coordinates are scanned?
Which output coordinate is being produced now?
```

Write the LSTM with named coordinates. The gates are local formulas; the
dependency edges are the same:

```rust
// LSTM: two state families, four gates, same coordinate rules
let h[0, b, u] = init_h[b, u];
let c[0, b, u] = init_c[b, u];

let h[t in 1..T, b, u] = {
    let i_gate = sigmoid(sum[u_prev](W_i[u, u_prev] * h[t-1, b, u_prev])
                       + sum[i](U_i[u, i] * x[t, b, i]));
    let f_gate = sigmoid(sum[u_prev](W_f[u, u_prev] * h[t-1, b, u_prev])
                       + sum[i](U_f[u, i] * x[t, b, i]));
    let o_gate = sigmoid(sum[u_prev](W_o[u, u_prev] * h[t-1, b, u_prev])
                       + sum[i](U_o[u, i] * x[t, b, i]));
    let c_cand = tanh(sum[u_prev](W_c[u, u_prev] * h[t-1, b, u_prev])
                    + sum[i](U_c[u, i] * x[t, b, i]));

    let c[t, b, u] = f_gate * c[t-1, b, u] + i_gate * c_cand;
    o_gate * tanh(c[t, b, u])
};
```

Read one cell: `h[7, 3, 10]`. The four gates each scan `h[t-1, b, u_prev]` via
`W_*[u, u_prev]` and `x[t, b, i]` via `U_*[u, i]`. The cell state `c[t, b, u]`
reads `c[t-1, b, u]` — same `u`, no scan, because the forget gate applies
elementwise. The output gate reads `c[t, b, u]` — same time step, same unit.

Every coordinate follows the same four rules from the simple RNN. Time reads
`t-1`. Batch reads `b`. Hidden unit `u` is produced; `u_prev` is scanned. The
formula is longer but the coordinate contracts are identical. The reader does
not need to learn new rules for the LSTM. They apply the same checklist.

Now a GRU, compressed to show the pattern:

```rust
let h[t in 1..T, b, u] = {
    let z = sigmoid(sum[u_prev](W_z[u, u_prev] * h[t-1, b, u_prev]) + ...);
    let r = sigmoid(sum[u_prev](W_r[u, u_prev] * h[t-1, b, u_prev]) + ...);
    let c = tanh(sum[u_prev](W_c[u, u_prev] * (r * h[t-1, b, u_prev])) + ...);
    (1 - z) * h[t-1, b, u] + z * c
};
```

Two gates instead of four. Same `u` vs `u_prev` distinction. Same `t-1` edge.
The recurrence contract checklist applies unchanged.

## Try It

A dependency graph is learned by drawing it, not by reading about it. Draw the
dependency graph for this simple RNN:

```rust
let h[0, b] = 0.0;
let h[t in 1..5, b] = 0.5 * h[t - 1, b] + x[t, b];
```

Then consider three wrong variants. Which of the four recurrence contract checks
does each violate?

```text
hidden[t, b] = 0.5 * hidden[t + 1, b] + x[t, b]       // reads future (check 2)
hidden[t, b] = 0.5 * hidden[t - 1, b_prev] + x[t, b]   // batch mixing (check 3)
hidden[t, b] = 0.5 * hidden[t, b_prev] + x[t, b]       // no time shift (check 2)
```

A shape checker sees compatible dimensions for all three. The contract checklist
catches them from the coordinate names alone.

Now take the LSTM formula from this chapter and trace one cell: `h[7, 3, 10]`.
Every gate scans `h[t-1, b, u_prev]` — each gate has its own weight matrix but
the same scanned coordinate `u_prev`. The cell state update `c[t, b, u] = f_gate
* c[t-1, b, u] + i_gate * c_cand` reads `c[t-1, b, u]` with the same `u` — no
scan, because the forget gate applies elementwise. If it read `c[t-1, b, u_prev]`
under `sum[u_prev]`, the cell state would mix across hidden units, erasing the
elementwise memory that makes LSTMs work. Verify the recurrence contract
checklist for the LSTM: base case exists, time goes backward, batch stays fixed,
roles are distinct. All four pass with the same checks as the simple RNN.

Finally, consider a design with two hidden coordinates — `short_term` (fast,
resets each step) and `long_term` (slow, gated update):

```rust
let short_term[0, b, u] = init_short[b, u];
let long_term[0, b, u] = init_long[b, u];

let short_term[t in 1..T, b, u] =
    tanh(sum[u_prev](W_s[u, u_prev] * short_term[t-1, b, u_prev])
       + sum[i](U_s[u, i] * x[t, b, i]));

let long_term[t in 1..T, b, u] = {
    let candidate =
        tanh(sum[u_prev](W_l[u, u_prev] * long_term[t-1, b, u_prev])
           + sum[i](U_l[u, i] * x[t, b, i]));
    let gate = sigmoid(sum[u_prev](W_g[u, u_prev] * short_term[t-1, b, u_prev])
                     + sum[i](U_g[u, i] * x[t, b, i]));
    gate * long_term[t-1, b, u] + (1.0 - gate) * candidate
};
```

Three edge types: fast (`short_term[t] ← short_term[t-1]`), slow (`long_term[t]
← long_term[t-1]`), and cross (`long_term[t]` reads `short_term[t-1]` through
the gate). All three are hidden inside one `hidden_size` integer in a positional
implementation. In a loop, `gate = sigmoid(W_g @ short + ...)` reads `short`.
The bug: `gate = sigmoid(W_g @ long + ...)` reads `long` instead. Both are
tensors of shape `[batch, hidden]`. The code runs. The loss descends. The error
lives on the cross edge — the gate was supposed to read `short_term[t-1]` but
silently reads `long_term[t-1]`. Two tensors of identical shape distinguished
only by their coordinate name. The name is the contract.

**Line to keep:** an RNN is not a loop calling a cell; it is a communication
graph made of named coordinates.

The graph is not a picture of the code. The graph is what the code *means*. The
loop is what the code *does*. The distance between them is the distance between
reading a map and walking the road. Someone who walks the road gets where they
are going. Someone who reads the map knows why.

Recurrence is the third kind of combination. Reduction consumes a coordinate.
Broadcast shares one. Recurrence chains one — each value reads an earlier value
of the same name. The audit still works. Survive, consume, omit still apply.
But now one of the survivors carries an edge, and that edge is the difference
between a program the compiler can schedule and a program it can only execute.

### Where This Leads

Part III is now complete. We have learned that hiding compounds. Part I taught
us to notice hiding. Part II taught us that hiding has consequences — the
gradient inherits the silence. Part III taught us that when time is involved,
hiding does not just add one more hidden fact. It multiplies. Hiding the time
direction hides causality, hides the storage plan, hides the backward-pass
schedule, hides the distinction between a program you can optimize and a program
you can only execute. One hidden index, four invisible properties.

But Parts I through III all worked on single operations in isolation. A
recurrence. A gradient. A softmax. We have been examining individual trees.
Real programs are forests: an encoder feeding a decoder, a loss function reading
both, a data loader upstream of everything. Each module names its coordinates
independently. The question Part IV asks is: can the names survive the
boundaries between them?

When an encoder emits `output[time, batch, feature]` and a decoder expects
`input[time, batch, unit]`, the coordinates `feature` and `unit` must meet. If
the notation cannot name that transition, the compiler cannot check it. A
mismatch of `768` and `512` crashes at runtime. A mismatch of `feature` and
`unit` could be caught at compile time — but only if the names survive the
module boundary.

Part IV is the stress test. The same coordinate vocabulary that survived
reshape, reduction, gradient, and recurrence must now survive module boundaries,
multi-head parallelism, attention protocols, and dynamic expert routing. If it
can, the principle is not a local convenience — it is a system property. If it
cannot, we have learned something important about the limits of naming.

Chapter 13 begins the test: a program where every tensor dimension has a name,
and the compiler reads those names across every function boundary.