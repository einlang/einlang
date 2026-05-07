---
layout: book
title: "Appendix: Coordinate Reading Laws"
---

# Appendix: Coordinate Reading Laws

You have reached the end of the book. What remains is the smallest possible
summary: sixteen laws, each one sentence, each tested against a shape-compatible
wrong version. The laws are the permanent residue of the reading habit — the
rules that survive when the examples are forgotten.

Each law has the same structure: a one-sentence statement, a minimal example, a
counterexample that violates the law while keeping shapes valid, and a note on
why the law matters. The laws are organized into five categories that follow the
book's progression.

## Shape and Role Laws

### 1/16. Role Is Not Extent

**The law:** Two coordinates can have the same size and still mean different
things.

**Example:**
```text
feature_count == time_count == 128
```
`feature` and `time` both have extent 128. But a model that normalizes over
`time` instead of `feature` is computing a completely different statistic.

**Counterexample (same shape, wrong meaning):**
```python
x = torch.randn(128, 128)
y = torch.softmax(x, dim=-1)  # dim=-1 is... feature? time? both are 128
```

**Why it matters:** Same extent, different meaning — a square matrix hides role swaps from every shape checker. (Chapter 1)

---

### 2/16. The Address Relation Comes Before Layout

**The law:** A coordinate map states how output addresses relate to input
addresses. The implementation may lower this to a view, a copy, or a fused
kernel, but the source-level claim is the address relation.

**Example:**
```text
y[b * group_count + group, feat * slice_count + slice] =
    x[b, feat, group, slice]
```

**Counterexample (shape-valid but wrong address):**
```text
y[group * b + b, feat * slice_count + slice] =  // wrong packing order
    x[b, feat, group, slice]
```

**Why it matters:** A shape trace records what happened; a coordinate relation records what was meant to happen — and survives questioning. (Chapter 1)

---

### 3/16. Free Coordinates Form the Result Family

**The law:** In an indexed binding, the coordinates that remain free on the
left describe the result family.

**Example:**
```text
let y[i, j] = sum[k](A[i, k] * B[k, j])
```
The result is addressed by `i` and `j`. Every `(i,j)` pair gets one scalar.
The coordinate `k` is local work, not part of the result address.

**Counterexample (shape-valid but wrong survivors):**
```text
let y[i, k] = sum[j](A[i, k] * B[k, j])  // k survives? k was supposed to be consumed
```

**Why it matters:** The left-hand side is the contract — change the survivors, change the computation, identical extents and all. (Chapter 5)

---

### 4/16. Omission Means Independence

**The law:** If a term does not mention a result coordinate, the term does not
depend on that coordinate.

**Example:**
```text
let y[b, f] = x[b, f] + bias[f]
```
`bias[f]` omits `b`. The same bias value is reused for every batch item. The
coordinate reading of broadcasting: not "can this singleton expand?" but
"which coordinate is the value independent of?"

**Counterexample (shape-valid but wrong independence claim):**
```python
x = torch.randn(128, 128)         # [feature, batch] after transpose
bias = torch.zeros(128)           # intended per-feature
y = x + bias                      # broadcasts over batch, not feature
```
When `feature == batch`, the shapes are square and the independence claim is
wrong but unchecked.

**Why it matters:** Broadcasting is a semantic claim dressed as a shape rule — when the claim is wrong and shapes match, you get the bug that trained. (Chapter 4)

---

## Reduction and Broadcast Laws

### 5/16. Reduction Consumes a Local Coordinate

**The law:** A reducer introduces a coordinate for local work and removes it
from the result.

**Example:**
```text
let row_total[i] = sum[j](A[i, j])   // i survives, j is consumed
```

**Counterexample (shape-valid but wrong consumption):**
```text
let row_total[j] = sum[i](A[i, j])   // j on the left, but i was consumed
```
When `A` is square, both produce a vector of the same length. They compute
different things.

**Why it matters:** The survivor is on the left; the consumed is inside `sum[...]`. When the matrix is square, both produce the same shape. (Chapter 5)

---

### 6/16. The Broadcast-Reduce Inverse Law

**The law:** A value broadcast along a coordinate in the forward pass receives
a summed gradient over that same coordinate in the backward pass.

**Example:**
```text
// Forward: bias[f] is reused across every b
let y[b, f] = x[b, f] + bias[f];

// Backward: bias[f] collects sensitivity from every b
let dbias[f] = sum[b](dy[b, f]);
```

**Counterexample (shape-valid but wrong backward):**
```text
let dbias[f] = dy[0, f];  // only reads batch 0, others silently dropped
```
Produces the right shape `[f]`. Ignores sensitivity from every other batch
item.

**Why it matters:** Forward omission becomes backward summation — the coordinate-level statement of the chain rule. (Chapters 4-5, 8)

---

## Derivative Laws

### 7/16. Gradients Are Addressed by the Denominator

**The law:** The shape of a derivative answer is determined by the value being
differentiated with respect to. `@loss / @W` has the coordinates of `W`.

**Example:**
```text
let y[i] = sum[j](W[i, j] * x[j])
let dW[i, j] = dy[i] * x[j]    // dW has W's coordinates [i, j]
```

**Counterexample (shape-valid but wrong address):**
```text
let dW[j, i] = dy[i] * x[j]    // transposed — right values, wrong layout
```
Produces the right set of numbers but addressed under swapped coordinates.

**Why it matters:** The denominator's coordinates are the gradient's address — every other coordinate in the route must be justified. (Chapter 7)

---

### 8/16. The Pullback Path Coordinate Law

**The law:** For a binary operation `C[survivors] = op(A[coords_a], B[coords_b])`,
the gradient of A sums over `output_coordinates ∖ coords_a`, and the gradient
of B sums over `output_coordinates ∖ coords_b`.

**Example (matmul):**
```text
let C[i, j] = sum[k](A[i, k] * B[k, j])

// A's coords: {i,k}. C's coords: {i,j}. Path: {j}.
let dA[i, k] = sum[j](dC[i, j] * B[k, j])

// B's coords: {k,j}. C's coords: {i,j}. Path: {i}.
let dB[k, j] = sum[i](dC[i, j] * A[i, k])
```

**Counterexample (shape-valid but wrong path):**
```text
let dA[i, k] = sum[i](dC[i, j] * B[k, j])  // sums over i instead of j
```
Produces shape `[i, k]` but `sum[i]` collapses the wrong coordinate.

**Why it matters:** The path coordinate is set subtraction on names — A's gradient sums over what A doesn't own but the output does. (Chapter 8)

---

### 9/16. The Denominator Jacobian Law

**The law:** When a forward expression reads multiple positions of the same
input through a reduction, the Jacobian has non-zero off-diagonal entries. The
coordinate names reveal this at the source level.

**Example:**
```text
// Softmax: probs[b,j] reads x[b,k] for ALL k through the denominator
// The Jacobian @probs[b,j]/@logits[b,k] has shape [b,j,k] and is NOT diagonal

// Sigmoid: s[b,j] reads only x[b,j]
// The Jacobian @s[b,j]/@x[b,k] IS diagonal (zero when j≠k)
```

**Counterexample (wrong dependency assumption):**
Treating softmax as elementwise (like sigmoid) when deriving the gradient.
The missed off-diagonal terms are the ones the denominator scan created.

**Why it matters:** Same shape, completely different Jacobian — the coordinate names reveal the dependency graph before a single derivative is taken. (Chapter 6)

---

## Recurrence Laws

### 10/16. Time Is a Directed Coordinate

**The law:** A recurrence should expose the dependency edge before it becomes a
loop. A causal coordinate only references smaller index values.

**Example:**
```text
let h[0] = h0;
let h[t in 1..T] = step(h[t - 1], x[t]);   // backward edge
```

**Counterexample (shape-valid but wrong direction):**
```text
let h[t in 0..T-1] = step(h[t + 1], x[t]);   // forward edge — reads the future
```
The index `t + 1` points forward. A forward simulation cannot compute this
without a different contract (boundary-value problem, backward pass, or cheat).

**Why it matters:** A loop buries the arrow of time; a recurrence states it at the definition site where the compiler can check it. (Chapter 10)

---

### 11/16. Observation Determines Storage

**The law:** Defining a family is not the same as materializing every member.
Storage follows observation, not definition.

**Example:**
```text
let h[t in 1..T] = step(h[t - 1], x[t])   // defines the family
let final = h[T - 1]                        // observes only the last member
// → storage: one rolling window
```

**Counterexample (wrong storage assumption):**
Allocating a full `[T, ...]` array when only `h[T-1]` is observed, or using
a rolling window when gradient computation needs every intermediate.

**Why it matters:** Definition, observation, storage — three different things. Confuse them and you allocate what you'll never need or discard what the gradient must replay. (Chapter 11)

---

### 12/16. Batch Isolation During Recurrence

**The law:** A recurrence that walks over time must not mix batch items. Each
batch member's hidden state is a separate recurrence instance.

**Example:**
```text
let h[0, b] = init[b];
let h[t in 1..T, b] = step(h[t - 1, b], x[t, b]);
// h[t, 7] reads h[t-1, 7] — batch 7 stays isolated from batch 2
```

**Counterexample (shape-valid but batch leakage):**
```text
let h[t in 1..T, b] = step(
    sum[b_prev](h[t - 1, b_prev]),   // MIXES all batch items
    x[t, b]
);
```
The result still has shape `[T, batch, hidden]`. The loss still decreases.
But each batch item now receives a summary of ALL batch items' histories.

**Why it matters:** A recurrence that mixes batch items still produces the right shape — the bug lives in the coordinate the source didn't name as fixed. (Chapters 10, 12)

---

## Notation Laws

### 13/16. Coordinate Functions Hide Mechanics, Not Contracts

**The law:** A coordinate function is useful only if the call still says which
coordinate choice matters.

**Example:**
```text
softmax[class](logits)          // the bracket names the normalized coordinate
move_channel[channel](image)    // the bracket names the moved coordinate
scan[t](step, h0, x)            // the bracket names the ordered coordinate
```

**Counterexample (hiding the contract):**
```text
normalize(logits)               // WHAT is normalized? class? batch? feature?
```
The call compiles. It runs. But the call site no longer states which
coordinate decision the function embodies.

**Why it matters:** Hide the mechanics, not the contract — the bracket must name the coordinate role that decides correctness. (Chapter 15)

---

### 14/16. The Hiding Law

**The law:** Do not hide a fact that later reasoning must recover.

**Example (facts to show):**
```text
show: consumed coordinates, omitted coordinates, address maps
show: derivative addresses, recurrence edges, dynamic routes
```

**Example (facts to hide):**
```text
hide: register allocation, temporary buffers, device placement
hide: fusion order, tiling, vector width, kernel selection
```

**Counterexample:**
A `dim=-1` softmax that hides which coordinate was normalized. The reader
must recover this fact from upstream context, variable names, or comments.
If the upstream context changes (data pipeline refactor), the fact is wrong
but the code still runs.

**Why it matters:** The show/hide boundary is not about complexity — it is about which facts a future explanation will need. Register allocation won't be. The normalized coordinate will. (Chapter 15)

---

### 15/16. The Capacity Law

**The law:** A dynamic route creates a capacity coordinate. Dropped tokens
must be named. The capacity decision must be visible in the source.

**Example:**
```text
let route[b, t] = argmax[e](gate_prob[b, t, e]);
let slot[b, t]  = count_assignments[b, t, route[b, t]];
let keep[b, t]  = slot[b, t] < capacity;
```

**Counterexample (silent dropping):**
```text
// Token silently dropped — no keep mask, no overflow named
let dispatched[e, c, d] = scatter(x, route);  // if c >= capacity, silently overwritten
```

**Why it matters:** Dropped tokens are invisible to shape checks — the `keep` mask is the semantic witness that names the discarded. (Chapter 16)

---

### 16/16. The Coordinate Function Law

**The law:** A coordinate function must name the role it consumes or transforms
in its bracketed arguments. Other axes may be packed. The bracketed role is
the contract the caller must satisfy.

**Example:**
```text
fn softmax[class](x: [f32; ..left, class, ..right]) -> [f32; ..left, class, ..right]
```
The bracket names `class` as the normalized coordinate. `..left` and `..right`
are rest packs — the function works over any surrounding rank.

**Counterexample (asking for too much):**
```text
fn softmax_explicit[batch, class, time](x: [f32; batch, class, time])
```
The caller must supply all three. But only `class` is the decision. The extra
names are ceremony that dilute the contract.

**Why it matters:** The bracket names the decision; the rest pack absorbs the ceremony. A good coordinate function makes the choice visible without making the common case verbose. (Chapters 15-16)

---

## Law Application Table

Which law catches which bug type, by operation:

| Operation | Laws to Check |
|---|---|
| reshape / pack | 1/16 (role), 2/16 (address), 13/16 (hide contract) |
| transpose | 1/16 (role), 2/16 (address) |
| broadcast / add bias | 4/16 (independence), 6/16 (inverse) |
| reduce / sum / mean | 3/16 (free coords), 5/16 (consumption), 6/16 (inverse) |
| softmax | 1/16 (role), 4/16 (independence), 5/16 (consumption), 9/16 (Jacobian) |
| matmul / dot product | 3/16 (free coords), 5/16 (consumption), 7/16 (denominator), 8/16 (path) |
| gradient / backward | 6/16 (inverse), 7/16 (denominator), 8/16 (path) |
| recurrence / scan | 10/16 (direction), 11/16 (storage), 12/16 (batch isolation) |
| attention | 1/16 (role), 4/16 (independence), 5/16 (consumption), 9/16 (Jacobian) |
| dynamic routing / MoE | 15/16 (capacity), 16/16 (coordinate function) |

## Failure-to-Law Cross-Index

Each failure pattern from [Appendix A](appendix-coordinate-diagnostics.html) maps to the laws that diagnose it:

| Failure Pattern (Appendix A) | Primary Laws | Diagnostic Question |
|---|---|---|
| 1. Same shape, swapped roles | 1/16 (role ≠ extent), 2/16 (address before layout) | Which coordinate is slow, which is fast? |
| 2. Broadcast over wrong role | 4/16 (omission = independence), 6/16 (broadcast-reduce inverse) | Which coordinate does each term omit? |
| 3. Reduction consumes wrong coordinate | 5/16 (reduction consumes local), 3/16 (free coords = result) | Which coordinate disappeared from the result? |
| 4. One axis, several scopes | 9/16 (denominator Jacobian), 3/16 (free coords), 5/16 (consumption) | Which scope is the survivor, which are the locals? |
| 5. Pullback sums wrong route | 8/16 (path coordinate), 7/16 (denominator address) | Which output coordinates did this cell influence? |
| 6. Time hidden inside mutation | 10/16 (time is directed), 12/16 (batch isolation) | Do any edges point forward in time? |
| 7. Storage inferred too early | 11/16 (observation determines storage) | Which values are observed vs. defined? |
| 8. Attention reads wrong position | 1/16 (role), 8/16 (path coordinate) | Does the gather read index `i` or index `j`? |
| 9. Low-rank bottleneck unnamed | 16/16 (coordinate function), 14/16 (hiding law) | Which coordinate replaced the direct communication? |
| 10. Dynamic routing hides overflow | 15/16 (capacity law), 14/16 (hiding law) | Can you trace whether each token survived routing? |
| 11. Coordinate function hides wrong fact | 13/16 (hide mechanics not contracts), 14/16 (hiding law) | Which bracketed coordinate grounds the operation? |
| 12. Coordinate function asks too much | 16/16 (coordinate function law) | Which coordinate is the decision vs. ceremony? |

## The Study Loop

For a new tensor expression, use the same loop every time:

```text
1. Pick one output cell. Give it concrete coordinates (e.g., y[3, 7]).
2. Name every input coordinate it reads. Trace the route.
3. Mark which coordinates survive (on the left-hand side).
4. Mark which coordinates are local and disappear (inside sum/max/mean[...]).
5. Mark which result coordinates are omitted by each term (broadcast).
6. Write the shape-compatible wrong version.
7. State the law that rejects the wrong version.
8. Only then hide the mechanics behind a coordinate function.
```

If this feels repetitive, good. A small set of laws should do a lot of work.
That is the point of the notation.

## How to Derive New Laws

When you encounter a new operation not covered by the existing laws, apply
this six-step recipe:

**Step 1 — Write the operation with named coordinates:**
```text
let output[survivors] = new_op[consumed](input[all_coords]);
```

**Step 2 — Identify survivors, consumed, and omitted coordinates:**
```text
survivors = {coordinates on the left-hand side}
consumed  = {coordinates inside the new_op bracket}
omitted   = {survivors not mentioned by a given term}
```

**Step 3 — Ask which coordinate carries the semantic decision:**
What changes if you consume a different coordinate? What role does the
consumed coordinate play in the model's logic?

**Step 4 — State the contract that coordinate must satisfy:**
"`new_op[class]` must normalize over `class`, not batch, not time."

**Step 5 — Write the shape-compatible wrong version:**
```text
let wrong[survivors] = new_op[wrong_coord](input[all_coords]);
```
When do the shapes match but the meaning differs?

**Step 6 — State what the wrong version violates:**
Formulate it as a one-sentence law. Bold it. Give it a number. Add it to
the table above.

**Example — deriving a new law for cross-attention:**
```text
1. let out[b, i, d] = sum[j](weights[b, i, j] * V_enc[b, j, d])
   // i = decoder position, j = encoder position

2. survivors: {b, i, d}, consumed: {j}, omitted from V_enc: {i}

3. The key decision: V is indexed by j (encoder position), not i (decoder).
   A mistake here swaps which position is read.

4. Contract: "cross-attention gathers from encoder positions j using
   decoder-computed weights over j."

5. Wrong: let out[b, i, d] = sum[j](weights[b, i, j] * V_enc[b, i, d])
   // reads from decoder position i — self-attention, not cross-attention
   When query_len == key_len, shapes are identical.

6. Law: "In cross-attention, the gather coordinate (encoder position) and
   the query coordinate (decoder position) are distinct roles. They must
   be named separately even when their extents match."
```

This recipe produced every law in this appendix. It will produce the next
one you need.
