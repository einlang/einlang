---
layout: book
title: "Dynamic Routing and Low-Rank Communication"
---

# Dynamic Routing and Low-Rank Communication

The mixture-of-experts model has been training for two days. The loss is
descending. The throughput is excellent — only two experts active per token. On
day three someone checks the expert utilization dashboard and discovers that
Expert 3 handles 94% of all tokens. The other seven experts are idle. The model
has learned to route every token to the same expert.

No shape error caught this. No gradient check caught this. The loss went down.
The router emitted valid logits. Every tensor had the right shape. The model
was a mixture of experts in name only — and nothing in the source said so.

This is the failure mode that dynamic communication patterns introduce. The
communication graph is not fixed by the formula. It is chosen by the data. When
it collapses, the shapes stay correct and the loss still descends. The bug is
not a wrong value. It is a wrong route.

Everything so far has been about communication patterns the compiler can see
before a single number flows. A matmul always contracts `k`. A softmax always
normalizes `class`. A recurrence always reads `t-1`. The communication graph is
determined by the structure of the formula, not by the values that flow through
it. That is the assumption under which the named-coordinate principle has been
tested for fifteen chapters.

But the models being deployed today — the ones with linear attention and mixture
of experts — change their communication topology based on the data itself. A
token chooses its expert. A query-key conversation is compressed through a
bottleneck whose effective size is chosen by a feature map. The communication
graph is not merely large; it is *dynamic*. The paths are not fixed at compile
time. They are part of what the model computes.

If the named-coordinate principle breaks here, it is a notation for textbooks,
not for production. If it survives — if the same small vocabulary that read a
dot product can also read a dynamic routing plan — then the principle has earned
the right to be called general.

This is the book's last stress test. Do not read it casually.

You have followed the coordinate role through every operation in the book.
Reshape. Reduction. Broadcast. Gradient. Recurrence. Attention. In every case,
the rule held: when a coordinate role decides correctness, name it.

Now one test remains. Two shortcuts to dense attention. Both practical. Both
deployed at scale. Both dependent on data. Both ask the same question: can you
still name the coordinate where communication changes shape?

## Dense Attention, Then Two Shortcuts

Start from the dense attention pattern you know:

```rust
let scores[b, h, i, j] = sum[d](Q[b, h, i, d] * K[b, h, j, d]);
let weights[b, h, i, j] = softmax[j](scores[b, h, i, j]);
let out[b, h, i, v] = sum[j](weights[b, h, i, j] * V[b, h, j, v]);
```

Every query position `i` consults every key position `j`. The score table
produces a tensor with both `i` and `j`. That is the communication contract, and
it is the source of the cost. Two strategies try to avoid it.

One compresses the communication path through a fixed bottleneck coordinate
`r`. This is **linear attention**. The key-value sequence is summarized before
any query reads it. The direct path `i → j` is replaced by `i → r → j`.

The other lets each token choose its own communication path at runtime. This is
**mixture of experts**. A gate assigns each token to one expert, and the
token's computation follows that dynamic route.

Both strategies change the shape of communication. Both depend on data. Both
hide their coordinate story in positional-axis notation. Both ask the same
question.

## The Score Table Is a Coordinate Contract

Before shortcuts make sense, feel the weight of the full table. The score
tensor `scores[b, h, i, j]` is not just a large array. It is a contract:
for every batch item `b`, every head `h`, every query position `i`, and every
key position `j`, the model computes a direct interaction strength.

Draw it. Four query positions, four key positions, one head:

```text
            j (key positions) →
            0       1       2       3
        ┌───────┬───────┬───────┬───────┐
    0   │ s[0,0]│ s[0,1]│ s[0,2]│ s[0,3]│
i       ├───────┼───────┼───────┼───────┤
↓   1   │ s[1,0]│ s[1,1]│ s[1,2]│ s[1,3]│
        ├───────┼───────┼───────┼───────┤
    2   │ s[2,0]│ s[2,1]│ s[2,2]│ s[2,3]│
        ├───────┼───────┼───────┼───────┤
    3   │ s[3,0]│ s[3,1]│ s[3,2]│ s[3,3]│
        └───────┴───────┴───────┴───────┘
```

Each cell `s[i,j]` is a sum over the feature coordinate `d`:

```text
s[i,j] = Q[i,0]*K[j,0] + Q[i,1]*K[j,1] + ... + Q[i,d-1]*K[j,d-1]
```

Every row is a query's assessment of all keys. Every column is a key's
assessment by all queries. The coordinates `i` and `j` name the two sides of
the communication. The contract is simple: every pair meets.

The cost is the contract. For a sequence of length N, the table has N^2 cells.
Each cell costs O(d). Total: O(N^2 d). This is why attention dominates
computation for long sequences. The problem is not the operations per cell. The
problem is the number of cells — the Cartesian product of `i` and `j`.

Both shortcuts try to answer: can we reduce the cost without losing the
information that each cell was supposed to carry? The answer is never a simple
yes. It is always a trade, and the terms of the trade live in a coordinate.

Every chapter until now has worked with a fixed communication graph. The
coordinates were named, the edges were explicit, the compiler could audit them.
Now we introduce a coordinate whose job is to change which other coordinates
communicate — and ask whether the names still hold. This is the book's last
stress test. If the same small vocabulary that read a dot product in Chapter 2
can also read a dynamic routing plan in Chapter 16, the principle has earned
the right to be called general.

## Linear Attention: The Bottleneck Coordinate

The compact formula is:

```text
softmax(Q K^T) V  becomes  phi(Q) (phi(K)^T V)
```

This hides the coordinate story. Unfold it.

The trick is associativity. Matrix multiplication is associative:
`(Q K^T) V = Q (K^T V)`. The two orders compute the same result but have
different intermediate shapes. The first order produces `[i, j]` — the expensive
N×N table. The second order produces `[d, v]` — a small d×v matrix.

But linear attention does not use the raw feature dimension. It projects through
a feature map `phi` into a bottleneck coordinate `r`:

```rust
let Q_phi[b, h, i, r] = phi(sum[d](Q[b, h, i, d] * W_q[d, r]));
let K_phi[b, h, j, r] = phi(sum[d](K[b, h, j, d] * W_k[d, r]));
```

Follow the coordinates. `d` is consumed into `r`. The feature dimension does
not survive. Position `i` and position `j` each attach to their own copy of
`r` — but they do not yet meet. The direct path `i → j` has not been built.

Now watch the key-value summary absorb `j`:

```rust
let KV[b, h, r, v] = sum[j](K_phi[b, h, j, r] * V[b, h, j, v]);
```

All key positions are consumed. The summary has coordinates `[b, h, r, v]`. For
each batch item and head, it is an `r × v` matrix — regardless of sequence
length. That is the cost saving. `j` cost N; `r` costs whatever the bottleneck
size is.

The output reads the summary:

```rust
let out[b, h, i, v] = sum[r](Q_phi[b, h, i, r] * KV[b, h, r, v]);
```

The coordinate `r` is consumed. The output has `[b, h, i, v]` — the same shape
as dense attention's output. But the path from `i` to `j` never existed as a
direct edge. It was routed through `r`:

```text
i  ──→  r  ──→  j
Q_phi    KV    K_phi
```

That is the bottleneck. A shape-only implementation sees the feature dimension
projected to a smaller size. A coordinate reading says more: `r` is the rank
of the communication plan. Every query reads every key, but only through the
shared vocabulary of `r` dimensions.

```
   Linear Attention Bottleneck

   Dense attention:
   scores[i,j] ---> all i-j pairs meet directly, cost O(N^2)
   +---+---+---+---+
   |   |j=0|j=1|j=2|
   +---+---+---+---+
   |i=0| * | * | * |
   |i=1| * | * | * |
   |i=2| * | * | * |
   +---+---+---+---+

   Linear attention: route through r
   +---+---+        +---+---+
   |Q_phi|--> r=0,1 -->|KV |
   +---+---+        +---+---+
   i -> r -> summary  then  summary -> out
   No [i,j] table. Cost O(N * r * d).

   r = bottleneck coordinate: replaces the direct i-j path.
   Small r -> queries share the same compressed summary.
```

The question is not whether linear attention works. It is whether the notation
names the bottleneck or buries it in a projection that happens to produce the
right shapes. The coordinate `r` is the fact that a later reader will need to
recover when debugging why an attention pattern collapsed, why expressiveness
plateaued, or why the model cannot distinguish between two queries that should
attend differently. The Hiding Law does not care that `r` is small. It cares
that `r` is the only place where the limit on communication has a name.

## A Concrete Numerical Trace

Four query positions, four key positions, bottleneck rank r=2. Follow one
output cell `out[i=1, v]` from coordinates through numbers to result.

Q_phi at r=2:

```text
i=0: [1.0, 0.5]    i=2: [1.0, 0.2]
i=1: [0.5, 1.0]    i=3: [0.3, 1.0]
```

K_phi at r=2:

```text
j=0: [1.0, 0.3]    j=2: [0.8, 0.5]
j=1: [0.3, 1.0]    j=3: [0.5, 0.8]
```

V at v=2 (values for each key position, first value dimension only, for
brevity):

```text
j=0: 0.7    j=1: 0.3    j=2: 0.5    j=3: 0.1
```

Step 1: Compute KV[b, h, r, v] = sum[j](K_phi[j, r] * V[j, v]).

```text
At r=0: 1.0*0.7 + 0.3*0.3 + 0.8*0.5 + 0.5*0.1
      = 0.70 + 0.09 + 0.40 + 0.05 = 1.24

At r=1: 0.3*0.7 + 1.0*0.3 + 0.5*0.5 + 0.8*0.1
      = 0.21 + 0.30 + 0.25 + 0.08 = 0.84
```

Step 2: Read out[i=1] = sum[r](Q_phi[i=1, r] * KV[r]).

```text
0.5 * 1.24 + 1.0 * 0.84 = 0.62 + 0.84 = 1.46
```

Now the full 4×4 score table (what dense attention would compute, raw inner
products before softmax). Each cell s[i,j] = sum[r](Q_phi[i, r] * K_phi[j, r]):

```text
         j=0    j=1    j=2    j=3
    ┌─────────────────────────────
i=0 │  1.15   0.80   1.05   0.90
i=1 │  0.80   1.15   0.90   1.05
i=2 │  1.06   0.50   0.90   0.66
i=3 │  0.60   1.09   0.74   0.95
```

This 4×4 matrix has rank at most 2 — because every entry is a sum over r of
Q_phi[i,r] * K_phi[j,r]. No matter how many query and key positions there are,
the score table cannot express more than r independent patterns of
communication. The bottleneck coordinate `r` is the ceiling on expressiveness.

Trace the error. The table shows that i=1 attends most strongly to j=1 (1.15)
and j=3 (1.05). That pattern makes sense given Q_phi[1] = [0.5, 1.0] and
K_phi[1] = [0.3, 1.0], K_phi[3] = [0.5, 0.8]. But what if the real attention
pattern needed i=1 to attend strongly to j=2 (which has value 0.90 — second
place) while ignoring j=3 entirely? With r=2, the model cannot express that
preference. The bottleneck limits the number of independent attention patterns
that can coexist within one head.

This is not a bug you debug. It is an architecture. But the architecture's
limit has a coordinate name. When r is too small, distinct attention patterns
are forced to share the same bottleneck address.

## Causal Linear Attention Is a Recurrence

Causal masking breaks the simple global summary. A query at time `t` may read
only keys at positions `j ≤ t`. You can no longer sum over all `j` once. The
key-value summary becomes time-indexed — a prefix sum over the time coordinate:

```rust
let state[b, h, 0, r, v] = K_phi[b, h, 0, r] * V[b, h, 0, v];

let state[b, h, t, r, v] =
    state[b, h, t-1, r, v] + K_phi[b, h, t, r] * V[b, h, t, v];

let out[b, h, t, v] = sum[r](Q_phi[b, h, t, r] * state[b, h, t, r, v]);
```

Trace the prefix sum for four timesteps. The state at each step is the
cumulative key-value summary visible up to that point:

```text
t=0:  state[0] = K_phi[0] * V[0]
      out[0]   = sum[r](Q_phi[0] * state[0])

t=1:  state[1] = state[0] + K_phi[1] * V[1]
      out[1]   = sum[r](Q_phi[1] * state[1])

t=2:  state[2] = state[1] + K_phi[2] * V[2]
      out[2]   = sum[r](Q_phi[2] * state[2])

t=3:  state[3] = state[2] + K_phi[3] * V[3]
      out[3]   = sum[r](Q_phi[3] * state[3])
```

Each query at time `t` reads only the state accumulated up to `t`. The key at
`t=3` is invisible to queries at `t=0,1,2`. The causal constraint is enforced
by the recurrence direction, not by a mask applied after the full score table
is computed. That is the efficiency: you never materialize `[i, j]`.

This is not just a matrix multiplication reordered for speed. It is a prefix
recurrence over time. The state carries coordinates `[b, h, r, v]` forward
through `t`. The recurrence is explicit — not hidden inside a scan helper with
an anonymous accumulator dimension. That contract matters for storage planning
(you need one state buffer per timestep if you checkpoint, or you reverse the
recurrence during backpropagation), for streaming inference (each new token
updates the state in O(1) per head), and for reverse-mode differentiation.

Now trace the gradient. In the backward pass, `state[t=2]` receives sensitivity
from two sources: from `out[t=2]` (the query at time 2 reads it) and from
`state[t=3]` (the recurrence carries it forward). The sensitivity at the state
coordinate is a sum over two incoming edges:

```text
dL/d(state[2]) = @out[2]/@state[2] + @state[3]/@state[2]
```

The first term comes from the query read. The second term comes from the
recurrence. Both paths meet at `state[2]`. If either is dropped, the gradient
is incomplete. A scan primitive that hides the accumulator coordinate makes
this bifurcation invisible. The coordinate reading makes it a named fact.

## When the Bottleneck Is Too Small

The bottleneck coordinate `r` is not free. It costs memory and computation, so
there is pressure to keep it small. But when `r` is too small, distinct queries
are forced through the same narrow channel and become indistinguishable.

Take two queries at positions `i=1` and `i=3` from the numerical trace above.
They want different things. Query 1 attends most to key 3 (1.05) followed by
key 1. Query 3 attends most to key 1 (1.09) followed by key 3 (0.95). With
r=2, both preferences can be represented — the rank-2 bottleneck can carry
independent attention patterns.

Reduce `r` to 1. Now `Q_phi[i, r=0]` is a single number per query position. The
score `s[i,j] = Q_phi[i,0] * K_phi[j,0]` is the outer product of two vectors.
Every row is a scalar multiple of every other row. Every column is a scalar
multiple of every other column. The entire N×N table is rank-1.

Two queries `i=1` and `i=3` that should attend to different keys now differ
only by a multiplicative factor. If `Q_phi[1,0] = 2.0` and `Q_phi[3,0] = 0.5`,
query 1 attends to all keys exactly four times as strongly as query 3. The
ranking of keys is identical for both queries. The bottleneck coordinate `r=0`
has become a collision domain: every query shares the same attention pattern up
to a scalar.

This is not a shape error. The shapes are right: `Q_phi[b, h, i, 1]`,
`K_phi[b, h, j, 1]`, `KV[b, h, 1, v]`, `out[b, h, i, v]`. Every reduction
consumes the right coordinate. The program runs. The loss descends. The model
simply cannot express the attention patterns the task requires. The bottleneck
coordinate is the fact that names the limit, and the fact that would explain
the accuracy ceiling to a debugging session six months later.

## Gradients Through the Bottleneck

The bottleneck coordinate `r` is also where sensitivities mix. In the forward
pass, all key positions contribute to `KV[b, h, r, v]`. In the backward pass,
a single `K_phi[b, h, j, r]` receives sensitivity through every value feature
`v` that used the same `r`, and every query position `i` that later read the
summary.

The dense attention path has not vanished. It has been routed through a
lower-dimensional address. The diagnostic questions are local:

```text
Where did i and j stop talking directly?
Which coordinate replaced the direct communication path?
Which gradients are forced to meet at that coordinate?
```

For linear attention, the answer to all three is `r`.

This is Chapters 7–9 applied to a new structure. The forward edges that a
gradient must follow are the same edges the coordinate audit traced in the
forward pass. `Q_phi[i,r]` influences `out[i,v]` through every `r` and `v`.
`K_phi[j,r]` influences `out[i,v]` through every `i`, `r`, `v`. The sensitivity
paths fan out through the bottleneck and reconverge at the source coordinates.
Naming `r` makes the fan-out visible. An unnamed feature dimension that happens
to be small makes it look like any other projection.

## Mixture of Experts: The Dynamic Route

Mixture of Experts changes the problem. The communication graph is not merely
compressed; it is chosen by the data.

A gate computes scores over experts:

```rust
let gate_score[b, t, e] = sum[d](x[b, t, d] * W_gate[d, e]);
let gate_prob[b, t, e] = softmax[e](gate_score[b, t, e]);
let route[b, t] = argmax[e](gate_prob[b, t, e]);
```

The coordinate `e` ranges over a static domain — there are `num_experts`
experts. But which expert a token visits is chosen per token. The result
`route[b, t]` is a content-dependent label. It says where this token will go.

This makes MoE different from every earlier example. Coordinates such as `i`,
`j`, and `k` ranged over static domains with fixed communication patterns. The
expert domain is static, but the assignment is dynamic. The compiler can still
reason about the role only if the route is named:

```text
route[b, t] = chosen expert for token [b, t]
```

Without that name, the routing decision becomes a scatter implementation detail.
With the name, later code can ask which tokens went where. The coordinate `e`
names the space of experts. The coordinate `route[b, t]` names the specific
choice made by the gate. Both are facts the source should carry.

This is the Hiding Law's hardest test case. In every earlier chapter, the
communication graph was static — a matmul contracts `k`, a softmax normalizes
`class`, a recurrence reads `t-1`. The coordinates were fixed and the compiler
could verify them before a single number flowed. Here, the communication graph
is chosen by the data. The compiler cannot know which expert each token will
visit. But it can verify that the routing coordinate `e` is declared, that
`route[b, t]` is used consistently in dispatch and gather, and that `keep[b, t]`
guards against overflow. The static check becomes a contract on the dynamic
coordinate. The Hiding Law survives the transition from static to dynamic because
the question does not change: can later reasoning recover this fact, or was it
buried in a scatter?

## One Token Through the Full Round Trip

Follow token `x[b=2, t=7, d]` through every step of the MoE pipeline. The
token lives in batch 2, at sequence position 7, with feature dimension `d`.
There are four experts, indexed `e` from 0 to 3.

Step 1 — Gate scores:

```text
gate_score[2, 7, 0] = 1.2   gate_score[2, 7, 1] = 0.3
gate_score[2, 7, 2] = 2.7   gate_score[2, 7, 3] = -0.5
```

Step 2 — Softmax over `e`:

```text
gate_prob[2, 7, 0] = 0.16   gate_prob[2, 7, 1] = 0.07
gate_prob[2, 7, 2] = 0.72   gate_prob[2, 7, 3] = 0.03
```

Step 3 — Argmax over `e`:

```text
route[2, 7] = 2       // expert 2 wins
```

Step 4 — Dispatch. The token is assigned a slot inside expert 2. The slot
depends on how many other tokens also chose expert 2 and arrived before this
one. Assume expert 2 has capacity 4, and one token already occupies slot 0:

```text
slot[2, 7] = 1        // second token in expert 2
keep[2, 7] = true     // slot < capacity, token survives
```

Step 5 — Expert computation. The token enters expert 2's feed-forward network
at slot 1:

```text
expert_input[2, 1, d] = x[2, 7, d]
expert_output[2, 1, o] = FFN_2(expert_input[2, 1, d])
```

Step 6 — Gather. The output returns to token coordinates:

```text
y[2, 7, o] = expert_output[route[2,7], slot[2,7], o]
            = expert_output[2, 1, o]
```

The token has traveled: `(b=2, t=7) → gate → e=2 → slot=1 → FFN_2 → (b=2, t=7)`.
The expert coordinate `e=2` was never a dense axis like attention's `j`. It was
a routing label — chosen at runtime, used for dispatch, consumed by the gather.

Every coordinate in this round trip has a role. `d` was consumed by the gate
projection. `e` was consumed by the argmax. `t` and `b` survived the entire
round trip. `slot` was created by the dispatch and consumed by the gather. Name
each one. A positional implementation that uses `dim=1` for the expert choice
and `dim=2` for the slot leaves every reader — and the compiler — to guess which
index means what.

## The Dispatch Table

After routing, tokens are rearranged from `[b, t, d]` to `[e, c, d]`: expert
index, capacity slot, feature. Draw it for eight tokens, four experts, capacity
3 per expert:

```text
             Expert 0       Expert 1       Expert 2       Expert 3
             (3 slots)      (3 slots)      (3 slots)      (3 slots)
slot 0:   [b=0,t=3]      [b=0,t=0]      [b=2,t=7]      [b=0,t=1]
slot 1:   [b=1,t=5]      [b=2,t=2]      [b=1,t=4]      [b=1,t=6]
slot 2:   ---------      [b=0,t=8]      ---------      ---------
                        [b=3,t=0] → OVERFLOW, keep=false
```

Seven tokens found a slot. The eighth token — `[b=3, t=0]` — chose expert 1,
but expert 1's capacity of 3 was already full. The token is dropped. Its output
will be a fallback value. Its gradient will not flow through the expert. All of
this is invisible to a shape check because the output tensor still has the
right shape at `[b, t, o]` — the dropped token's output slot is filled with the
fallback.

The table is the dispatch contract in one picture. The rows are capacity slots
`c`. The columns are experts `e`. The cells are token identities `[b, t]`. Empty
cells are wasted capacity. Overflow cells are invisible to the output shape but
visible to the loss. Every empty cell and every overflow cell is a fact the
source should be able to name.

```
   MoE Dispatch Table: tokens[b,t] -> experts[e], capacity slots[c]

   +------+----------+----------+----------+----------+
   | slot | Expert 0 | Expert 1 | Expert 2 | Expert 3 |
   +------+----------+----------+----------+----------+
   | c=0  | b=0,t=3  | b=0,t=0  | b=2,t=7  | b=0,t=1  |
   | c=1  | b=1,t=5  | b=2,t=2  | b=1,t=4  | b=1,t=6  |
   | c=2  |   ---    | b=0,t=8  |   ---    |   ---    |
   +------+----------+----------+----------+----------+
                           |
                     b=3,t=0 --> OVERFLOW, keep=false

   Each token chooses an expert via route[b,t] = argmax[e](gate_prob)
   slot[b,t] = position within that expert's capacity
   keep[b,t] = (slot[b,t] < capacity)

   route, slot, keep: three coordinate facts a scatter/reshape hides.
```

## Capacity Slots: The Hidden Coordinate

Efficient MoE implementations batch tokens by expert. That requires a static
shape — each expert must allocate a fixed number of capacity slots. The dynamic
route creates a new coordinate:

```text
slot[b, t] = position of token [b, t] inside expert route[b, t]
keep[b, t] = slot[b, t] < capacity
```

The dispatched tensor has coordinates `[e, c, d]` where `e` is the expert and
`c` is the capacity slot inside that expert. A token writes to:

```text
expert_input[route[b, t], slot[b, t], d] = x[b, t, d]
```

The capacity coordinate `c` is not a model dimension. It is an allocation of
space inside each expert. If too many tokens choose the same expert, some
tokens may not fit. The mask `keep[b, t]` is therefore a semantic witness:
it tells whether token `[b, t]` survived routing.

### The Capacity Coordinate Is Not Optional

A program that drops overflow tokens without naming `keep` has hidden a fact
that debugging will almost certainly need. Consider what `keep[b, t] = false`
means in practice:

The token `[b, t]` is assigned a fallback value:

```rust
let y[b, t, o] =
    if keep[b, t] {
        expert_output[route[b, t], slot[b, t], o]
    } else {
        fallback[b, t, o]
    };
```

The token did not enter any expert. Its contribution to the loss comes entirely
from the fallback — typically a copy of the input, a zero vector, or a
fixed embedding. The gradient of the loss with respect to the expert parameters
receives zero from this token. The gate parameters receive gradient only through
the load-balancing loss (if one exists), not through the task loss. The token
is, for the purpose of expert learning, absent.

Now ask: does any shape check catch a missing `keep` mask? No. The output tensor
`y[b, t, o]` has the same shape whether `keep` is considered or not. A
positional implementation that fills overflow slots with the fallback value and
never names which tokens were dropped has made a critical semantic fact
invisible. The loss is higher. Accuracy is lower. Some experts are overloaded.
The debugging session has no coordinate to read.

This is the book's most dramatic example of the law from Chapter 15. The hiding
is legal — the shapes are correct. The recovery is expensive — you must
instrument the dispatch, count tokens per expert, and reconstruct the overflow
pattern from log statistics that may or may not have been collected. The source
should have named `keep`. It is a coordinate mask, not an implementation detail.

## The Round Trip

Once tokens are dispatched, each expert runs its own transformation:

```rust
let expert_output[e, c, o] = expert_ffn[e](expert_input[e, c, d]);
```

The result is gathered back to token coordinates:

```rust
let y[b, t, o] =
    if keep[b, t] {
        expert_output[route[b, t], slot[b, t], o]
    } else {
        fallback[b, t, o]
    };
```

The round trip is a coordinate map with a dynamic address. Token `[b, t]`
becomes `(route[b, t], slot[b, t])`, is processed, and returns to `[b, t]`.
The expert coordinate `e` is not a dense communication axis like attention's
`j`. It is a routing label chosen per token.

This is why MoE needs names. Without `route`, `slot`, and `keep`, the program
looks like scatter, gather, reshape, and mask manipulation. With the names, it
becomes a dynamic communication graph — and a reader can still check that the
dispatch and gather use consistent coordinates.

Read the gather as a coordinate address lookup:

```text
y[b, t, o] = expert_output[route[b, t], slot[b, t], o]
```

The address of the result is `route[b, t]` for the expert index and
`slot[b, t]` for the position inside that expert. Those are not constant
offsets. They are computed values. The compiler cannot statically verify that
the gather address is in bounds, but it can verify that the coordinate roles are
consistent: the gather reads from `[e, c, o]` using an address computed from
`[b, t]`. That consistency constraint is the MoE analogue of the shape check
for a static gather. It does not prevent overflow, but it says where overflow
lives.

## Load Balance as a Coordinate Reduction

MoE implementations add an auxiliary loss so tokens do not all choose the same
expert. The loss is easy to get subtly wrong because it is a statistic over
token and expert coordinates. Three variants are common. Each has a coordinate
formula. Each has a specific reduction that would be wrong.

### Auxiliary Loss

The standard load-balancing loss encourages uniform routing:

```rust
let fraction[e] = mean[b, t](if route[b, t] == e { 1.0 } else { 0.0 });
let mean_prob[e] = mean[b, t](gate_prob[b, t, e]);
let balance_loss = sum[e](fraction[e] * mean_prob[e]) * num_experts;
```

`fraction[e]` counts what fraction of tokens hard-routed to expert `e`. It
reduces over `[b, t]`, producing a statistic per expert. `mean_prob[e]`
measures the gate's soft preference for expert `e`, also reducing over
`[b, t]`. The loss is the sum over `e` of the product — it is minimized when
the hard assignment fractions match the soft probability means, which happens
when routing is uniform.

The coordinate audit: both `fraction[e]` and `mean_prob[e]` reduce over
`[b, t]`. The final sum reduces over `e`. A shape checker sees two valid
reductions. A coordinate checker sees two different stories — one counting
routing decisions, one measuring gate preference — and verifies that the
reduction coordinate is the same for both.

The common mistake: reducing over `e` instead of `[b, t]` for one of the
statistics. The shape is compatible. The number is wrong.

### Z-Loss

The z-loss penalizes large logits entering the softmax gate, encouraging the
gate to produce less extreme distributions:

```rust
let log_Z = log(sum[e](exp(gate_score[b, t, e])));
let z_loss = mean[b, t](square(log_Z));
```

Here `log_Z` is a scalar per token — the log of the softmax denominator. It
does not depend on `e` (the sum consumes `e`). The z-loss is the mean squared
log-normalizer over all tokens. It pushes the gate toward balanced, moderate
scores.

The coordinate audit: `e` is consumed inside `log_Z`. The resulting scalar has
no `e` coordinate — it is indexed only by `[b, t]`. The outer reduction is over
`[b, t]`. The shape is simple, but the fact that `e` was consumed inside the
log-normalizer is what a positional notation would bury inside the
implementation. A coordinate reading says: this scalar was produced by
collapsing the expert dimension. The dimension that disappeared is the one being
regularized.

### Expert Dropout

Some implementations randomly drop experts during training to prevent
over-specialization:

```text
for each expert e, with probability p_drop:
    mask[e] = 0
otherwise:
    mask[e] = 1
```

The coordinate `e` is not consumed. It is masked — a form of conditional
omission. The mask is applied before the softmax, so the gate cannot route to a
dropped expert. The effect is that `gate_prob[b, t, e]` is zero for dropped
experts, and the remaining experts receive all tokens.

The coordinate audit: `e` survives the operation but some of its values are
conditionally zeroed. The mask is a coordinate-level fact. If the source does
not name which experts were dropped, the load distribution after dropout is
inexplicable. The shapes are right. The routing pattern has a hole the shape
does not show.

Across all three variants, the same principle holds: load balance is a
coordinate reduction over `[b, t]` producing a statistic per expert `e`, and
the reduction coordinate must be named. Square matrices hide which axis is the
expert and which is the token. A coordinate reading makes both visible.

## The Straight-Through Estimator as a Coordinate Contract

Routing introduces a hard question for autodiff. The forward pass uses a
discrete choice:

```rust
let route[b, t] = argmax[e](gate_prob[b, t, e]);
```

The route is an integer. The derivative of `argmax` is zero almost everywhere
(flat regions) and undefined at the threshold (step). If you differentiate
through the route naively, the gate parameters receive zero gradient. The model
cannot learn to route.

The straight-through estimator (STE) solves this by lying. In the forward pass,
the route is discrete. In the backward pass, the gradient pretends the argmax
was a softmax:

```text
Forward:  route[b, t] = argmax[e](gate_prob[b, t, e])    // discrete choice
True ∂:   ∂route/∂gate_prob = 0 (almost everywhere)       // useless
STE:      ∂route/∂gate_prob ≈ ∂softmax/∂gate_prob         // surrogate
```

The STE sends sensitivity through `gate_prob[b, t, e]` as if the gate had
produced a soft distribution instead of a hard choice. The coordinate `e` is
consumed in the forward pass (argmax picks one) but the gradient pretends `e`
survives as a distribution.

Read the STE as a coordinate contract with a warning label:

```rust
fn ste_top1[e](p: [f32; ..left, e, ..right]) -> [i32; ..left, ..right] {
    argmax[e](p[..left, e, ..right])
}

@fn ste_top1[e](p: [f32; ..left, e, ..right]) {
    soft_surrogate_tangent[e](p, @p)
}
```

In the forward pass, `e` is consumed to produce a discrete route. The output has
no `e` coordinate — the choice has been made. In the backward pass, the
implementation uses a surrogate rule that sends sensitivity through
`gate_prob[b, t, e]`. The gradient appears to address a coordinate that the
forward pass consumed.

This is not a smooth coordinate transformation. It is an estimator with a
coordinate contract. The source should not pretend the gradient is exact:

```text
WARNING: gradient through route[b,t] is a straight-through estimator.
The sensitivity at gate_prob[b,t,e] is a surrogate. The coordinate e
was consumed in the forward pass. The gradient pretends it survived.
```

The notation says where the discrete decision happens and where the gradient is
pretending to pass. This is much clearer than hiding the rule inside a detach
expression. The coordinate `gate_prob[b, t, e]` carries the surrogate gradient.
The coordinate `route[b, t]` carries the discrete choice. The two are different
facts. Both belong in the source.

This is the book's final example of the hiding law. Hiding that the gradient is
a straight-through estimator hides the fact that the optimization landscape is
discontinuous. A future reader debugging training instability needs to know that
the gradient path through `e` is a fiction. The source should not make them
rediscover it from a detached tensor and a missing gradient.

Part I taught us to name coordinates. Part II traced their gradients. Part III
gave time a direction. Part IV asked whether those names could survive a real
program — and the answer, tested across module boundaries, multi-head attention,
the Hiding Law boundary, and now dynamic routing, is that they can. But they
survive only when the source names the coordinate where communication changes
shape. `r` for the bottleneck. `route` for the expert choice. `keep` for
overflow. `ste_top1[e]` for the gradient fiction. Sixteen chapters, one
principle: do not hide a fact that later reasoning must recover.

## Two Patterns, One Question

Linear attention and mixture of experts look completely different. Linear
attention adds a bottleneck coordinate `r` to compress communication — every
query reads every key, but through a shared low-rank vocabulary. MoE adds a
route coordinate `e` to partition computation — each token chooses one expert
and ignores the rest.

But both answer the same question: **which coordinate changes the shape of
communication?**

For linear attention:
- `r` compresses the query-key path from `[i, j]` to `[i, r] @ [r, j]`
- The communication cost drops from O(N^2 d) to O(N r d)
- The expressiveness drops from full rank to rank r
- The coordinate `r` names both the savings and the limit

For mixture of experts:
- `route[b, t]` partitions tokens across experts
- The computation cost drops from O(E) per token to O(1) per token (plus gate)
- The communication cost is the dispatch and gather
- The coordinate `route[b, t]` names the routing decision; `keep[b, t]` names
  the overflow

Two architectures. Two different coordinate mechanisms. One reading discipline:

```text
1. Find the coordinate that controls communication.
2. Name it.
3. Check that its role survives all transformations.
4. Ask what happens when it is too small or misrouted.
```

Apply this to any dynamic architecture — sparse attention, conditional
computation, retrieval-augmented generation, any model that chooses its own
communication topology. The coordinate that controls the topology is the fact
the source should not hide. The rest is implementation.

## Beyond Tensor Code

The same question appears outside neural networks. It is not a tensor problem.
It is a representation problem.

### SQL: The Join Key That Disappeared

A SQL query optimizer has relations, keys, join columns, filters, and
projections. A query plan rewrites the logical operations into a physical
execution graph — hash joins, index scans, filter pushdown, column pruning.

Consider this query:

```sql
SELECT orders.id, customers.name
FROM orders
JOIN customers ON orders.customer_id = customers.id
WHERE orders.date > '2025-01-01';
```

The join key is `orders.customer_id = customers.id`. The optimizer may reorder
the join, choose a hash table on `customers.id`, push the date filter below the
join, or prune unused columns. All of these are valid transformations. But a
query plan that erases which column was the join key and which column was merely
carried along has hidden a fact. If the plan later needs to explain why an index
was chosen, or if a cardinality estimate is wrong because the join key is skewed
and the optimizer treated it as uniform, the fact must be recovered from
heuristics — or left unrecovered, producing a slow plan with no visible cause.

The optimizer is the compiler. The join key is the coordinate. The principle is
the same: a fact that later reasoning must recover should survive
transformation.

### Map Projections: The Axis That Was Latitude

A map tile system projects the Earth's surface onto flat squares. Latitude and
longitude become pixel coordinates. At zoom level 0, one tile. At zoom level
10, a million tiles. The projection changes with every zoom level: Mercator at
one level, Web Mercator at another, a local UTM zone at a third.

A tile server that stores pixels as `image[0:256, 0:256, 0:3]` has three
dimensions with the same name: `dimension 0`. But in one projection, dimension
0 is east-west. In another, it is the direction with the least distortion. In a
transverse projection, dimension 0 no longer aligns with any compass direction.
A map client that confuses latitude with longitude because both were packed into
integer ranges `[0, 255]` will render tiles correctly (the shapes match) but
rotate the map, mislabel north, or compute distances that are wrong by a factor
of two at the equator and infinite at the pole.

The storage coordinate may change many times — pixel row, tile X, geohash byte.
But the source facts still matter: which direction is north, which axis wraps,
what distance is preserved, and where distortion is introduced. A coordinate
system does not prevent all mistakes. It makes the cost of each projection
visible.

These examples are not tensor programs. But they rhyme with the same rule. The
implementation may choose a plan, a layout, an index, or a tile. The source
should preserve the facts that make those choices meaningful.

## The Habit

Fifteen chapters of stress tests. Reshape. Reduction. Broadcast. Gradient.
Recurrence. Attention. Dynamic routing. The principle held: when a coordinate
role decides correctness, the source should be able to state it.

This chapter added one more stress test — dynamic communication — and the same
small vocabulary read it. `r` for the bottleneck. `e` for the expert.
`route[b, t]` for the dynamic choice. `keep[b, t]` for overflow. `slot[b, t]`
for the position inside the dispatch. Five names. Five facts that a positional
implementation buries inside scatter-gather machinery.

The book's actual payload is not Einlang syntax. It is five questions you can
ask about any tensor program tomorrow morning:

```text
1. Which coordinates did this cell read?
2. Which coordinates were summed away?
3. Which coordinates were present in the formula but absent from one term?
4. Which coordinate is the address of this gradient?
5. Which coordinate changed the shape of communication?
```

The first four come from the index page. The fifth comes from this chapter. It
is the culmination: after you know which coordinates survive, which are
consumed, which are omitted, and which carry gradients, you ask the architecture
question. Where does the communication graph change shape? Is the change visible
in the source, or is it buried in a scatter pattern that happens to have the
right dimensions?

These five questions are portable. They work in Einlang. They work in PyTorch
with named tensors. They work in JAX with xmap. They work in NumPy with
comments. They work in a whiteboard conversation where you have not yet chosen a
framework. The habit is not the syntax. The habit is asking the questions.

## Try It

Take the linear attention bottleneck from the chapter:

```rust
let Q_phi[b, h, i, r] = elu_plus_one(sum[d](Q[b, h, i, d] * W_q[d, r]));
let K_phi[b, h, j, r] = elu_plus_one(sum[d](K[b, h, j, d] * W_k[d, r]));
let KV[b, h, r, v] = sum[j](K_phi[b, h, j, r] * V[b, h, j, v]);
let out[b, h, i, v] = sum[r](Q_phi[b, h, i, r] * KV[b, h, r, v]);
```

The coordinate `r` is the bottleneck. The coordinate that disappears between
`K_phi[b, h, j, r]` and `KV[b, h, r, v]` is `j` — that is the coordinate whose
quadratic cost is eliminated. If `r = 1`, the maximum rank of the equivalent
score table `scores[i, j]` is rank 1. Every row is a scalar multiple of every
other row. Query `i=0` and query `i=1` attend to keys in exactly the same
ranking, differing only by a multiplicative factor `Q_phi[i=0, r=0] /
Q_phi[i=1, r=0]`. The coordinate `r` names the limit. The coordinate condition
that makes this limit invisible to shape checks is that `r_count` is a
hyperparameter: nothing in the shapes enforces that `r_count` is large enough to
carry the needed attention patterns. The bottleneck is a semantic limit, not a
shape limit.

Now imagine a colleague debugs an MoE model where some tokens produce NaN
outputs after 30,000 training steps. The loss spikes, recovers, spikes again.
The shapes check out at every layer. You ask to see the dispatch code:

```python
# positional: dispatch tokens to experts
capacity = (tokens_per_batch * batch_size) // num_experts
dispatch_mask = scatter(route, capacity)  # unclear what happens on overflow
```

The bug: `capacity` is computed as an average, so on 60% of batches some expert
receives more tokens than capacity. The overflow tokens write past the end of
the expert buffer. The gather reads uninitialized memory. The NaN appears
intermittently because it depends on whether expert assignments are lopsided for
a particular batch. The named-coordinate version catches this at the source
level:

```rust
let keep[b, t] = slot[b, t] < capacity;
let y[b, t, o] =
    if keep[b, t] {
        expert_output[route[b, t], slot[b, t], o]
    } else {
        fallback[b, t, o]  // or x[b, t, o] for residual
    };
```

The coordinate `keep[b, t]` has the same shape as the output. No dimension was
added. No shape changed. But a reader — and a compiler — can ask: did every
token survive routing? If `sum[b, t](keep[b, t]) < B * T`, overflow tokens
exist. The source names the fact that overflow is possible. A shape checker sees
`y[b, t, o]` and reports success whether `keep` exists or not. A coordinate
checker would report that `expert_output[route[b, t], slot[b, t], o]` reads from
coordinate `slot` at a position computed from `[b, t]`. The compiler cannot
statically verify the bound `slot[b, t] < capacity`, but it can verify that
`keep[b, t]` guards the read. The guard is the contract.

There is an even subtler variant: `keep[b, t]` is present, but the fallback also
flows through `keep` — the fallback computation accidentally reads
`expert_output` for the overflow token (which is garbage) before `keep` branches
away from it. This is a lowering bug, not a coordinate bug. The source-level
coordinate contract is correct. The compiled code evaluates the fallback
expression before the branch. The source cannot prevent this — it can only make
the contract visible so the lowering pass knows which values must not be read.

Next, design a dynamic router that uses top-2 instead of top-1. Each token
visits two experts. The coordinate `route[b, t]` becomes `routes[b, t, top_k]`
where `top_k` indexes the first and second choices.

The top-2 gate:

```rust
let gate_prob[b, t, e] = softmax[e](sum[d](x[b, t, d] * W_gate[d, e]));
let routes[b, t, 0] = argmax[e](gate_prob[b, t, e]);       // first choice
let masked_prob[b, t, e] = gate_prob[b, t, e] * (1.0 - one_hot(routes[b, t, 0]));  // remove winner
let routes[b, t, 1] = argmax[e](masked_prob[b, t, e]);     // second choice
```

The dispatch has each token occupy one slot in two different experts:

```rust
let slot0[b, t] = slot_in_expert[routes[b, t, 0], b, t];
let slot1[b, t] = slot_in_expert[routes[b, t, 1], b, t];
let keep0[b, t] = slot0[b, t] < capacity;
let keep1[b, t] = slot1[b, t] < capacity;
```

The gather:

```rust
let y[b, t, o] =
    gate_prob[b, t, routes[b, t, 0]] * expert_output[routes[b, t, 0], slot0[b, t], o]
  + gate_prob[b, t, routes[b, t, 1]] * expert_output[routes[b, t, 1], slot1[b, t], o];
```

Each term is weighted by the gate probability of the chosen expert. The weights
sum to less than 1.0 because they are the softmax probabilities of the top two
experts, not the full distribution. Some implementations renormalize the two
weights to sum to 1.0. The coordinate that distinguishes these versions is
`gate_prob[b, t, routes[b, t, top_k]]` versus `renormalized_prob[b, t, top_k]`.
The first preserves the gate distribution. The second redistributes mass from
the dropped experts.

Now consider the naive implementation: call top-1 twice, masking out the first
winner before the second call. After zeroing the first winner, the remaining
probabilities no longer sum to 1.0. But `argmax` only compares values, so the
second choice is correct even without renormalization. What about renormaling by
dividing by `(1 - p_winner)`? `argmax` is invariant to uniform scaling, so
renormalization also does not change the argmax. So where is the bug?

The real bug emerges from a different confusion. The gate probabilities
`gate_prob[b, t, e]` are already softmax outputs. If naive code applies softmax
again — applying softmax to values already in [0,1] that sum to 1 — softmax-of-softmax
is not renormalization but a double-softmax that further concentrates the
distribution:

```rust
// BUG: double-softmax — gate_prob is already a softmax output
let second_prob[b, t, e] = softmax[e](gate_prob[b, t, e]);  // gate_prob already sums to 1
let routes[b, t, 1] = argmax[e](second_prob[b, t, e]);
```

The second expert is chosen from a doubly-squashed distribution. Every choice is
arguably "wrong" in a way that still descends the loss. The coordinate `e` in
`gate_prob[b, t, e]` carries the contract that `sum[e](gate_prob[b, t, e]) =
1.0`. Any operation that masks, renormalizes, reuses, or applies
softmax-to-softmax on this coordinate must preserve or intentionally break this
sum constraint. The coordinate name `e` is the only source-level fact that
carries the contract. If the source omits `e` and uses positional operations
`softmax(dim=-1)` and `scatter(dim=1)`, the contract lives entirely in the
implementer's head.

Each mistake in a dynamic router can be diagnosed by identifying the coordinate
that reveals it:

```text
Mistake                                       Coordinate that reveals it
─────────────────────────────────────────────────────────────────────────────
double-softmax on gate_prob                   e appears inside two softmax[e] calls
no renormalization (using logits is fine)     irrelevant — argmax ignores scale
top-2 without capacity check                  keep[b, t, top_k] never computed
overflow token reads garbage                   slot[b, t, top_k] >= capacity, no guard
gate gradient doesn't reach overflow tokens   keep[b, t] = false blocks gradient
```

The coordinate `top_k` is a single name that turns top-1 into top-2; the rest of
the dispatch logic is parameterized. The expression `routes[b, t, top_k]` says
"each token chooses up to `top_k` experts" — the source states the communication
topology, not just the shapes. The gate, dispatch, and gather formulas are
unchanged between top-1 and top-2; only the `top_k` coordinate changes. The
architecture is the coordinate signature. The coordinate audit table gives a
procedure for diagnosing any dynamic routing bug: find the coordinate that
carries the contract, trace its usage, and check each operation that consumes or
masks it.

**Line to keep:** name the coordinate where communication changes shape.

### Where This Leads

This is the end of the book, but the beginning of the question. You have seen
named coordinates survive reshape, reduction, broadcast, gradient, recurrence,
attention, and dynamic routing. The principle held: when a coordinate role
decides correctness, the source should be able to state it.

The one thing this book has not done is stop you from writing positional-axis
code tomorrow morning. Tensor libraries exist. Deadlines exist. The habit this
book asks you to carry is smaller and more portable than any syntax: when you
write a tensor program, ask which coordinate roles are hidden, which are
consumed, which survive, and where a mistake would keep the right shape. Write
those facts down in whatever notation you have.

Every dimension meaning you capture is a dimension meaning the next reader — or
the compiler, or the autodiff engine, or your future self at 3 AM — will not
have to rediscover from a stack trace and a shape comment. That is not a small
gift. That is the difference between code that runs and code that can be read,
reviewed, refactored, and trusted by people who were not in the room when it was
written.

We began this book with a Tuesday. The shapes were right. The loss descended.
The model was wrong — silently, invisibly, for hours. The bug was invisible to
every tool because the notation had no place for the fact that axis 2 was
`class`, not `time`. A notation that recorded only shapes gave the reader no
place to notice the error. A notation that records roles makes the error a
mismatch the compiler can name.

If that Tuesday has a name now, if you can look at a tensor line and feel the
missing coordinate role the way a musician feels a missing beat, then the book
has done its work. The notation was always just the instrument. The habit was
always the point.

You came to this book for a notation. You are leaving with a habit. The
notation will change — new languages, new frameworks, new backends. The habit
will not. Ask what is hidden. Name what matters. The rest is syntax.

Notation determines what you can notice. When the notation names the coordinate
role, the role becomes a fact the compiler can check, the autodiff engine can
preserve, and the next reader can see without reconstructing it from memory.
When the notation omits the role, the role becomes invisible — not just to the
compiler, but to the act of reading and reasoning itself. This was the thesis of
the Introduction. It is the thesis of every chapter between. It is the thesis
you now have the tools to test, in your own code, on your own deadlines, with
your own notation — whatever that notation happens to be.

The Habit is four questions. The Bargain is one rule. That is what an
abstraction is: a thing small enough to carry, sharp enough to cut. Fifteen
characters in three places is a small price for a fact the compiler can check
forever.

Turn the page. There is nothing more to learn here. Only something to try.