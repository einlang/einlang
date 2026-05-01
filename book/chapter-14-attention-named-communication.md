---
layout: book
title: "Chapter 14: Attention as Named Communication"
---

# Attention as Named Communication

The tempting story is that attention is just matrix multiplication with a
softmax in the middle. That story is useful for kernels, but too small for
debugging: it forgets who asks, who answers, and which value comes home.

Attention is often introduced through matrix products:

```text
softmax(Q K^T) V
```

That expression is compact, but it hides the roles of the sequence positions.
A standard-library version can make those roles explicit.

This is where the earlier ideas have to work under load. Query position, key
position, value position, feature, and head can share extents while doing
different jobs. Attention is useful precisely because those jobs are different.

Attention is a larger example of the same habit. Query positions ask, key
positions answer, and values must be gathered at the answering coordinate
rather than at the query by accident.

The most honest way to read attention is to pick one query position, list the
key positions it scans, and then follow the chosen key to the value read. The
common wrong version gathers values at the query coordinate. It can still look
like ordinary matrix code, but the communication protocol has collapsed: the
answer no longer determines what comes back.

## Special Operator or Named Communication

A language could treat attention as a special operator with a fixed signature.
That is practical, but it risks making attention another opaque primitive: the
implementation may be fast while the communication pattern is hidden behind a
name.

The opposite choice is to expand attention into low-level matrix operations
and reshapes. That exposes more machinery, but it can still lose the roles of
query position, key position, value feature, head, and batch.

The visible-index design takes a third path. It can use attention as a library
operation, but the coordinate story remains expressible: one coordinate asks,
one coordinate answers, and another coordinate carries the value being
returned.

Erase the words query, key, and value from an attention implementation and
leave only matrix shapes. The computation may still run, but the communication
story becomes external knowledge. The design choice here is to keep that story
in the coordinate names while still allowing the implementation to lower to
the familiar matrix products.

```rust
let scores[b, i, j] =
    sum[d](Q[b, i, d] * K[b, j, d]) * scale;
```

The coordinate `i` is the query position. The coordinate `j` is the key
position. The coordinate `d` is the feature coordinate consumed by the dot
product. The result keeps `b`, `i`, and `j`: for each batch item, every query
position receives a score for every key position.

Read one score:

```text
scores[b, 4, 9]
```

This is the compatibility between query position `4` and key position `9` in
batch item `b`. It is computed by scanning feature coordinate `d`. The score
does not yet contain a value vector. It is only a relation between two
positions.

This is already more explicit than `Q K^T`. The matrix product is correct, but
the names explain which side is asking and which side is being compared.

The difference matters when `i` and `j` have the same extent. Self-attention
often compares a sequence with itself, so both positions range over the same
length. A shape checker sees two equal axes. The source names say which one is
the question being updated and which one is the position being listened to.

## From Scores to Values

After normalization, the output is:

```rust
let output[b, i, d] =
    sum[j](attention_weights[b, i, j] * V[b, j, d]);
```

Now `j` is consumed. The query position `i` survives. The feature coordinate
`d` returns, this time as the value feature being gathered.

Read one point:

```text
output[b, 4, d]
```

This is the value vector for query position `4`. It is built by scanning all
key/value positions `j`, weighting each `V[b, j, d]` by how much query `4`
attends to position `j`.

The names give the sentence its grammar. Query asks. Key answers. Value is
carried home.

## The Softmax Coordinate

Between scores and values sits normalization:

```text
attention_weights[b, i, j] =
    softmax[j](scores[b, i, j])
```

For each fixed batch item `b` and query position `i`, the weights over all
key positions `j` should sum to one. The key coordinate is the coordinate being
normalized. The query coordinate is not reduced. It identifies which row of the
attention matrix is being normalized.

As a function boundary, the same contract would name the answering coordinate:

```rust
fn attention_softmax[key](scores: [f32; ..batch, query, key])
    -> [f32; ..batch, query, key]
```

The caller supplies `key`; the query and batch-like coordinates are inferred.
That is the whole point of the bracket: it distinguishes "normalize over keys"
from "normalize over queries" without counting positions.

This is a common source of mistakes. Normalizing over `i` would answer a
different question: for each key, how do query positions distribute? That may
be useful in some analysis, but it is not ordinary attention. Named coordinates
make the difference visible:

```text
sum[j](attention_weights[b, i, j])  expected row sum
sum[i](attention_weights[b, i, j])  different object
```

Again the question is local: which coordinate does the normalization consume?

Masking adds another coordinate-sensitive layer. In causal attention, a query
position may listen only to keys at or before itself:

```text
allowed[i, j] = j <= i
```

That mask is more than a boolean matrix with the right shape. Its meaning
depends on which coordinate is the query and which is the key. Swapping `i` and
`j` changes the direction of time:

```text
j <= i   past and present keys are visible
i <= j   future keys become visible instead
```

The usual implementation may add a large negative value before softmax, but
the coordinate relation is the thing that makes the mask causal. This is
another case where visible names help distinguish a layout trick from the
model contract it implements.

## Why This Is Larger Than Attention

The attention example combines earlier ideas:

- `sum[d]` is a consumed feature coordinate;
- `softmax` normalizes over key positions;
- `sum[j]` gathers values;
- `b` survives as batch throughout;
- `i` survives as the position being updated.

The operation is not simple, but the style of reading is the same as for dot
product or broadcasting. That is the promise of a small notation: the questions
do not change when the example grows.

## Multi-Head Structure

Multi-head attention adds a head coordinate:

```text
Q_h[b, h, i, d] = Q[b, i, h * head_dim + d]
```

The original feature coordinate is split into `h` and `d`: which head, and
which feature inside that head. Later, the heads are packed back together. The
important point is that head identity is more than a reshape artifact. It is a
semantic coordinate in the model.

A reader can now ask:

```text
Does softmax normalize over key position j, separately for each head h?
Does the output preserve query position i?
Does the final projection mix heads, or only repack them?
```

Those are model questions. They become easier to ask when the code gives names
to the roles involved.

A compact way to state the multi-head contract is:

```text
scores[b, h, i, j]  compare query i with key j inside head h
output[b, h, i, d]  gather value feature d for query i inside head h
```

The head coordinate survives through the attention pattern. Whether a later
projection mixes heads or only repacks them is a separate coordinate question,
not a mystery hidden inside reshape code.

Written fully, the score and gather keep `h` fixed while communication happens
within that head:

```text
scores[b, h, i, j] = sum[d](Q[b, h, i, d] * K[b, h, j, d])
out[b, h, i, d]    = sum[j](weights[b, h, i, j] * V[b, h, j, d])
```

The absence of `sum[h]` is the point. Heads do not communicate inside the
attention pattern. If a later output projection consumes `h`, that is a new
operation with its own coordinate contract.

## What Did the Query Scan?

For `output[b, 4, d]`, list the coordinates that were scanned to produce it.
The value feature `d` survives. The query position `4` survives. The key
position `j` is consumed while gathering values. Earlier, the feature coordinate
inside the score was consumed while comparing query and key vectors.

The broader picture is communication. A query coordinate does not simply index
a row of a matrix; it names a position that asks other positions for
information. The key coordinate names the positions being listened to. The
value coordinate names what is carried back. Attention becomes easier to read
when those three roles remain distinct.

## Communication With Coordinates

Attention is a large example because it gathers many threads at once.
It has matrix multiplication, softmax, reduction, broadcasting-like reuse, and
coordinate packing for heads. It is exactly the kind of code that often becomes
a sequence of reshapes, transposes, and batched matmuls.

The visible-coordinate reading does not deny that optimized implementations
will use those operations. It changes what the source is allowed to say before
optimization begins:

```text
i asks
j answers
d is compared or carried
h separates heads
b keeps examples apart
```

That is a compact model of attention's structure. The model is not a substitute
for performance engineering, but it gives performance engineering something
precise to preserve.

This is where the compiler-transformation story becomes concrete. Autodiff is
one compile-time transformation, but it is not the only compiler service
enabled by named coordinates. The implemented pipeline also supports Einstein
lowering, recurrence ordering, lowered execution facts, and backend
vectorization choices. Other transformations, such as explicit batching APIs
or checkpoint schedulers, belong to the same conceptual family only when they
can be justified by the same source facts: which coordinates survive, which are
independent, which are reduced, and which are observed.
Where the implementation does not yet expose a transformation as a user-level
pass, the book treats it as a design direction rather than as a completed
feature.

Attention also prepares the later stress tests. If a notation can keep
attention readable without inventing a special story for attention, then the
notation has some claim to generality. It is more than a prettier spelling for
matrix multiplication. It is a way to keep named communication patterns visible
inside larger tensor programs.

There is a caution here as well. Production attention implementations often
fuse operations, tile memory, use specialized kernels, cache keys and values,
or split work across devices. Visible coordinates do not replace those
techniques. They describe the relationship those techniques must preserve.

Tensor Comprehensions and TVM TE are useful analogies here. Their compute
definitions can describe reductions such as attention's score contraction:

```text
score[b, h, i, j] = sum[d](Q[b, h, i, d] * K[b, h, j, d])
```

Later scheduling or lowering decides how to tile, cache, vectorize, or bind
the work to hardware. Einlang is interested in the same relation as a
source-level contract: `i` is the query coordinate, `j` is the key coordinate,
`d` is the feature coordinate being contracted, and `h` separates heads. A
tensor comprehension, a TE schedule, a Triton kernel, or an XLA/MLIR lowering
should be free to optimize that computation. It should not have to rediscover
those roles from a pile of axis numbers.

That separation is the same one seen in earlier chapters: source relation
first, lowering strategy later. The source can say that `i` queries `j`; the
runtime can decide how to compute that relation efficiently. A good notation
does not fight optimization. It gives optimization a precise semantic target.

The practical test is simple: after reading an attention implementation, can
you identify the query coordinate, the key coordinate, the value feature
coordinate, the head coordinate, and the batch coordinate? If the answer
requires a diagram outside the code, the source is asking memory to do too much
work.

When the names are present, the diagram and the code can reinforce each other.
The notation does not replace the mental picture of attention; it gives that
picture coordinates.

## Shape-Compatible Broken Attention

Take one batch item, one head, two query positions, two key positions, and two
features. This row is only a witness for the hard problem: in
self-attention, query and key positions often have the same extent, so a broken
communication pattern can keep the same shape.

```rust
let scores[i, j] = sum[d](Q[i, d] * K[j, d]);
```

For query position `i = 0`, the two scores are:

```text
scores[0, 0] = Q[0, 0] * K[0, 0] + Q[0, 1] * K[0, 1]
scores[0, 1] = Q[0, 0] * K[1, 0] + Q[0, 1] * K[1, 1]
```

The feature coordinate `d` is consumed. The query coordinate `0` survives. The
key coordinate `j` survives for the score table because the model still needs
one compatibility score per key.

After softmax over `j`, the weights for query `0` form a distribution:

```text
weights[0, 0] + weights[0, 1] = 1
```

The output gathers values:

```rust
let out[i, d] = sum[j](weights[i, j] * V[j, d]);
```

For `out[0, 1]`:

```text
out[0, 1] =
    weights[0, 0] * V[0, 1] +
    weights[0, 1] * V[1, 1]
```

Now `j` is consumed. The query coordinate `i = 0` survives, and the value
feature `d = 1` survives. The sentence is precise: query position `0` gathers
feature `1` from all value positions, weighted by how strongly it attended to
their keys.

A deliberately wrong normalization makes the failure local:

```text
bad_weights[b, h, i, j] = softmax[i](scores[b, h, i, j])
```

Ordinary attention needs `softmax[j]` because each fixed query distributes
over keys. The bad line consumes the query coordinate instead. A
named-coordinate checker has a role-level objection: the operation tries to
normalize away the coordinate that the output is supposed to preserve as the
asking position.

This row catches two common mistakes. First, normalizing over `i`
would make each key distribute over queries:

```text
sum[i](weights[i, j]) = 1
```

That can be a meaningful diagnostic object, but it is not ordinary attention.
Ordinary attention fixes a query and distributes over keys:

```text
sum[j](weights[i, j]) = 1
```

Second, using `V[i, d]` instead of `V[j, d]` in the gather would ignore the key
position being attended to:

```text
sum[j](weights[i, j] * V[i, d])
```

The shape can still work because `i` and `j` often have the same sequence
length. But the communication pattern collapses: every term in the sum carries
the same value vector for the query, so the weights no longer choose among
positions. The coordinate names make the mistake visible. The value must be
read at the key/value coordinate `j`, not the query coordinate `i`.

Multi-head attention adds one more fixed coordinate to this audit:

```text
scores[h, i, j] = sum[d](Q[h, i, d] * K[h, j, d])
```

For each head `h`, query `i` compares against key `j` inside that head's
feature space. If the final projection later mixes heads, that is a separate
operation with its own coordinate relation. Attention itself keeps head
identity available long enough to ask which part of the model communicated.

That is why attention belongs near the end of the book rather than near the
beginning. It is not a new kind of notation. It is the earlier notation under
load: one coordinate asks, one coordinate answers, one coordinate carries the
value, one coordinate selects the head, and a reduction decides who was
listened to.

In `scores[h, i, j] = sum[d](Q[h, i, d] * K[h, j, d])`, `i` asks and `j`
answers. The value gather has to respect that. If it reads `V[h, i, d]` rather
than `V[h, j, d]`, the weights no longer choose among positions; the
communication pattern has collapsed.

## Try It

Modify the attention sketch so query and key share a `head` coordinate but use
different feature coordinates, such as `dq` for query/key comparison and `dv`
for values. Write the names before writing shapes. Then check whether a
mistaken reuse of `dq` as `dv` becomes visible at compile time.

**Line to keep:** attention is not matrix multiplication; it is a communication
protocol between named roles.
