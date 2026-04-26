---
layout: book
title: "Chapter 12: An RNN Is a Dependency Graph"
---

# An RNN Is a Dependency Graph

The RNN standard-library excerpt is small enough to read, but dense enough to
reward a slow reading:

The reward is a graph of allowed communication. Time, batch, input feature,
current hidden unit, and previous hidden unit are not decorative names. They
are the labels that say which values may talk to which other values. Collapse
them all into a cell call and the graph is still there, but it has to be
rediscovered.

An RNN cell is easy to hide behind an API. Read it instead as labeled
communication: time moves, batch stays isolated, and hidden roles are mapped by
the recurrent weight.

## Cell API or Visible Recurrence

One design is to hide the recurrence inside a cell API. The user calls
`rnn_cell` repeatedly, and the library owns the state transition. That is a
good abstraction for use. It is less helpful when we want to know what the
compiler can check.

Another design exposes the recurrence relation itself. The cell still exists,
but the source also names time, batch, input feature, current hidden unit, and
previous hidden unit. That makes the dependency graph inspectable without
opening every implementation detail of the cell.

The RNN standard-library example follows this second path. It shows enough of
the recurrence to expose the dependency graph while still leaving activation
dispatch, bias handling, and lowering as implementation concerns.

The compiler does not need a special "RNN type" to begin this analysis. It
sees indexed declarations, explicit ranges such as `t in 1..seq_length`,
reductions over `i` and `h_prev`, and the offset read `hidden[t - 1, b,
h_prev]`. Recurrence ordering and lowering then use those ordinary coordinate
facts to choose a legal evaluation order.

Close the implementation of `rnn_cell` and ask what the compiler can still
know. It may know that the cell is called in a loop, but not which coordinate
mixes input features, which coordinate mixes previous hidden units, or whether
batch members communicate. The recurrence spelling leaves those parts on the
page.

```rust
let hidden[t in 1..seq_length, b in 0..batch_size, h in 0..hidden_size] = {
    let z_cell =
        sum[i in 0..input_size](W[h, i] * X[t, b, i]) +
        sum[h_prev in 0..hidden_size](
            R[h, h_prev] * hidden[t - 1, b, h_prev]
        );
    tanh(z_cell)
};
```

The binding being defined is `hidden[t, b, h]`. That gives the output roles:
time, batch, hidden unit.

Inside the block, the input projection consumes `i`:

```text
sum[i](W[h, i] * X[t, b, i])
```

The recurrent projection consumes `h_prev`:

```text
sum[h_prev](R[h, h_prev] * hidden[t - 1, b, h_prev])
```

The batch coordinate `b` is not consumed. It identifies which sequence in the
batch is being updated. The hidden coordinate `h` is the new unit being
computed. The previous hidden coordinate `h_prev` is scanned to collect
influence from the earlier state.

## Two Sums, Two Kinds of Mixing

The two reductions in the RNN cell have different meanings:

```text
sum[i](W[h, i] * X[t, b, i])
sum[h_prev](R[h, h_prev] * hidden[t - 1, b, h_prev])
```

The first mixes input features at the current time step. The second mixes
hidden units from the previous time step. Both are sums, but their local
coordinates have different roles. Calling them both "axis 2" would lose that
distinction.

For `hidden[7, 3, 10]`, the input term reads:

```text
X[7, 3, i]
```

for every input feature `i`. The recurrent term reads:

```text
hidden[6, 3, h_prev]
```

for every previous hidden unit `h_prev`. Time moves backward only in the second
term. Batch stays fixed in both. The output hidden coordinate `h = 10` selects
which row of `W` and `R` is used.

That single cell carries the whole RNN contract:

```text
current input row        X[7, 3, i]
previous hidden row      hidden[6, 3, h_prev]
current hidden unit      h = 10
```

The model is doing more than iterating. It is mixing two different coordinate
families into one new hidden coordinate.

## What a Loop Hides

The loop version:

```python
for t in range(T):
    h = rnn_cell(h, x[t])
```

contains the same idea, but the names are inside the cell implementation and
the mutation of `h`. The recurrence version keeps the dependency graph visible
at the binding site:

```text
hidden[t, b, h] depends on hidden[t - 1, b, h_prev]
```

This is a statement the compiler can inspect. It can check direction in time.
It can see that batch is preserved. It can see that previous hidden units are
mixed into new hidden units.

## The Base Case Is Part of the Graph

A recurrence is not complete without its base case:

```rust
let hidden[0, b, h] =
    if typeof(initial_h) == "rectangular" { initial_h[b, h] } else { 0.0 };
```

This line defines the first time slice. The recurrent rule starts at
`t = 1`, so the read of `hidden[t - 1, b, h_prev]` is legal. Without the base
case, `hidden[1]` would depend on an undefined `hidden[0]`.

The base case is often implicit in loop code:

```python
h = init_state
for t in range(T):
    h = cell(h, x[t])
```

The initialization is there, but it is expressed as the first value of a
mutable variable rather than as one member of a time-indexed family. The
recurrence form puts the base case and the recurrence rule in the same
coordinate system.

## Output Is Another Observation

The standard RNN returns both the sequence and the final hidden state:

```rust
let Y[t, b, h] = hidden[t, b, h];
let Y_h[b, h] = hidden[seq_length - 1, b, h];
```

These two lines observe the same recurrence differently. `Y` asks for every
time step. `Y_h` asks only for the last one. As chapter 11 argued, that
observation matters for storage. The recurrence defines the hidden family; the
outputs decide which parts of that family are externally visible.

## Who Communicates Across Time?

For `hidden[7, 3, 10]`, what earlier hidden values are read?

The answer is all `hidden[6, 3, h_prev]` for legal `h_prev`. Time goes from
`7` to `6`. Batch stays `3`. The hidden coordinate is reduced through the
recurrent weights.

This is the important separation: recurrence mixes hidden units across time,
but it does not mix examples across the batch. If batch were reduced here, the
model would be making examples communicate with one another. The coordinates
make that boundary visible.

## The Recurrent Contract

The time-axis idea becomes model-shaped in an RNN. The earlier recurrence
examples could be read as simple mathematical sequences. The RNN shows that
the same notation can describe a familiar neural computation with batch, input
features, hidden units, and time all present at once.

This is the test for visible-index discipline. If the coordinate names only
work for miniature recurrences, they are not enough. But in the RNN, the same
questions still work:

```text
Which coordinate is time?
Which coordinate is batch?
Which coordinates are consumed by sums?
Which previous state is read?
Which outputs observe the hidden family?
```

The reward is not that the RNN becomes simple. It does not. The reward is that
the dependency structure remains inspectable even as the example becomes more
realistic. That is what visible dimensions buy.

A final detail is worth noticing: the RNN has two different hidden-related
coordinates, `h` and `h_prev`. They range over the same hidden size, but they
do not play the same role. The coordinate `h` names the hidden unit being
produced. The coordinate `h_prev` names the previous hidden unit being scanned.

Using two names prevents a subtle confusion. The recurrent matrix
`R[h, h_prev]` is not only indexed by "hidden, hidden." It maps from previous
hidden units to current hidden units. If those roles are swapped, the shape may
still be square, but the transformation is transposed. The names keep source
attention on direction: from previous state into current state.

The RNN has the same failure mode as the earlier reshape and softmax examples:
shape can be right while role is wrong. Coordinates give that role a place to
stand in the source.

The same reading scales to more complex recurrent cells. An LSTM adds a cell
state and several gates. A GRU adds reset and update gates. Those mechanisms
introduce more named pieces, but the questions do not change: which state is
from the previous time step, which coordinates are scanned, and which output
coordinate is being produced now?

For an LSTM, the hidden state and cell state become two value families, often
`h[t, b, unit]` and `c[t, b, unit]`. The input, forget, output, and candidate
gates add more local formulas, but they still preserve `b`, read `t - 1`, and
scan feature or previous-unit coordinates. A GRU changes the gate equations,
not the basic dependency audit. The visible graph grows, but its labels are
the same kind of labels.

Variable-length sequences add one more practical coordinate fact. A mask can
be read as a boolean value over time and batch:

```text
mask[t, b]
```

The recurrence can use it to choose between updating and carrying the previous
state:

```text
hidden[t, b, h] =
    if mask[t, b] { candidate[t, b, h] } else { hidden[t - 1, b, h] }
```

The mask is not just an array with a compatible shape. It says which time
points in which batch examples are real observations. Whether a system treats
that as a guard, a broadcasted boolean term, or a boundary condition, the
coordinate roles `t` and `b` should be explicit.

That is why the simple RNN is enough for this chapter. It is not the most
capable recurrent model, but it exposes the grammar of recurrence without
burying it under gates. Once the grammar is visible, larger cells are additions
to the same dependency story.

## Square Weights Hide Direction

Take one cell from the recurrence. The difficulty is that `h` and `h_prev`
often have the same extent, so transposing the recurrent relation can remain
shape-correct:

```text
hidden[t, b, h]
```

Suppose `t = 3`, `b = 2`, and `h = 5`. The input projection reads:

```text
sum[i](W[5, i] * X[3, 2, i])
```

The recurrent projection reads:

```text
sum[h_prev](R[5, h_prev] * hidden[2, 2, h_prev])
```

The two scans have different meanings. The `i` scan mixes input features at
the current time. The `h_prev` scan mixes previous hidden units from the
previous time. The batch coordinate `2` is preserved in both. The current
hidden coordinate `5` selects a row of both `W` and `R`.

A loop implementation might hide this inside a call:

```python
h = cell(x_t, h)
```

That is fine as an API boundary, but it hides the dependency audit. From the
call alone, a compiler cannot tell whether examples in the batch communicate,
whether time moves backward, or whether `R` maps previous hidden units to
current hidden units or the reverse. The indexed cell states those facts.

Now consider a transposed recurrent matrix bug:

```text
sum[h_prev](R[h_prev, h] * hidden[t - 1, b, h_prev])
```

If `R` is square, the shape is still valid. The formula still reduces over
`h_prev` and produces `hidden[t, b, h]`. But the role of the matrix has
changed. The old formula reads row `h` of `R`, collecting influence into the
current hidden unit. The transposed formula reads column `h`, using weights
whose source and target roles have been swapped. Shape alone may not catch
this. The names make the direction visible.

A second bug is batch mixing:

```text
sum[b_prev, h_prev](R[h, h_prev] * hidden[t - 1, b_prev, h_prev])
```

This program says that example `b` at the current time depends on all previous
examples. That may be a deliberate attention-like operation, but it is not the
ordinary RNN contract. The presence of `b_prev` under the sum is a local sign
that batch isolation has been broken.

The base case can be audited the same way:

```text
hidden[0, b, h] = initial_h[b, h]
```

It has no time dependency. It anchors the family. The recurrence rule starts
at `t = 1` precisely because `hidden[t - 1, b, h_prev]` needs `hidden[0, b,
h_prev]` for the first step. If the base case had shape `[h, b]` or omitted
`b`, the error would be semantic before it was numerical.

This is why the chapter calls an RNN a dependency graph. The graph nodes are
not only operations. They are indexed values. The edges carry coordinate
relations: same batch, previous time, previous hidden scanned into current
hidden. Once those edges are visible, recurrence ordering and lowering have
real structure to preserve.

## The Dependency Edge in One Line

The core edge can be summarized as:

```text
hidden[t, b, h] <- hidden[t - 1, b, h_prev]
```

Read that line as a dependency figure, not just as notation. The labels on the
edge are the whole point: time moves backward by one step, batch is preserved,
and the hidden coordinate changes from the previous hidden role to the current
hidden role through `R[h, h_prev]`. If those labels disappear, the graph may
look cleaner, but it stops explaining what communication is allowed.

That one line contains three different relationships. Time moves from `t` to
`t - 1`. Batch is equal on both sides. Hidden changes role: `h_prev` is scanned
and `h` is produced. The recurrent matrix `R[h, h_prev]` labels that hidden
role change.

If any part of the edge changes, the model changes. If time reads `t + 1`, the
direction changes. If batch reads `b_prev`, examples communicate. If `R` is
addressed as `R[h_prev, h]`, the hidden transition is transposed. A shape
checker may catch some of these errors, but the named graph explains all of
them in the same language.

Drawing the dependency graph from source coordinates changes what can be
checked. The graph is not an implementation artifact discovered after the
fact; it is the meaning of the recurrence.

This also makes tests easier to aim. A numerical test can compare final hidden
values, while a source inspection can ask whether batch was preserved, time
went backward, and hidden roles were mapped in the intended direction. Those
are different guarantees, and the visible graph gives each one a place.

The graph is therefore a reading tool as much as a compiler tool: it tells the
reader which communications are allowed before any numerical output is checked.

The edge `hidden[t, b, h] <- hidden[t - 1, b, h_prev]` is worth drawing once.
Change one coordinate on the right-hand side and the model changes: time can
reverse, examples can communicate, or the hidden transition can transpose. The
labels are what make those failures visibly different.

## Try It

Add a mask to the recurrence and decide whether a masked step reads
`candidate[t, b, h]` or carries `hidden[t - 1, b, h]`. Then ask whether any
token reads another token's batch coordinate. The exercise is to find the graph
edge, not the loop body.

**Line to keep:** an RNN is not a loop with tensors; it is a dependency graph
with a time coordinate.
