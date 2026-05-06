---
layout: book
title: "Appendix: Coordinate Diagnostics"
---

# Appendix: Coordinate Diagnostics

This appendix is a field manual. It catalogs the twelve failure patterns that
recur throughout this book — and throughout production tensor code. The pattern
is always the same: a shape can be plausible while the coordinate story is
wrong.

If you arrived here from a stack trace at 3 AM, start with the [Quick Reference
Card](#quick-reference-card). If you have fifteen minutes, work through the
[Diagnostic Workflow](#diagnostic-workflow). If you recognize your symptom in
the failure patterns below, jump to that entry.

## Common Failure Patterns

Each pattern includes the wrong code and the right code, the diagnostic question
that catches it, a cross-reference to the chapter that explains the pattern in
depth, and a "still runs?" note — does the wrong version produce the right
shape?

### 1. Same shape, swapped roles

**Wrong:** `x.reshape(B, H, -1, F).transpose(1, 2).reshape(B, -1, H*F)`
when the intent was pack `H` with `F`, not with the spatial dimension.

**Right:** `y[b, feat * head, spatial] = pack_head[head](x[b, feat, head, spatial])`

**Diagnostic question:** "Write the output address as a function of named input
coordinates. Which coordinate is slow, which is fast, which is packed with
which?"

**Still runs?** Yes, if `head_count == spatial_count`. The output shape is
identical. Only the coordinate relation reveals the swap.

**Chapter:** 1, "What Can the Compiler Not See?"

---

### 2. Broadcast over the wrong role

**Wrong:**
```python
x = torch.randn(128, 128)        # [feature, batch]
running_mean = torch.zeros(128)  # intended per-feature
x_norm = x - running_mean        # broadcasts over batch, not feature
```

**Right:**
```rust
let x_norm[b, f] = x[b, f] - running_mean[f];
```

**Diagnostic question:** "List the coordinates each term mentions. The missing
coordinate is the broadcast coordinate. Is that the coordinate you intended to
broadcast over?"

**Still runs?** Yes, when `feature_count == batch_count`. The shapes are
identical. Only the coordinate roles differ.

**Chapter:** 4, "What Does Broadcasting Hide?"

---

### 3. Reduction consumes the wrong coordinate

**Wrong:**
```python
row_sums = A.sum(dim=1)   # intended: sum over columns
# After silent transpose upstream, this is a column sum
```

**Right:**
```rust
let row_sums[i] = sum[j](A[i, j]);   // i survives, j consumed
let col_sums[j] = sum[i](A[i, j]);   // j survives, i consumed
```

**Diagnostic question:** "Circle the coordinates on the left-hand side. The
coordinate inside `sum[...]` should be absent from the result. If a different
coordinate is absent, something is wrong."

**Still runs?** Yes, when the matrix is square. `A.sum(dim=0)` and `A.sum(dim=1)`
both produce a vector of the same length.

**Chapter:** 5, "The Index That Leaves"

---

### 4. One logical axis has several local scopes

**Wrong:**
```python
probs = torch.softmax(logits, dim=-1)
# dim=-1 conflates stability scan, denominator scan, and output
```

**Right:**
```rust
let m[b]        = max[q](x[b, q]);           // stability scan
let e[b, k]     = exp(x[b, k] - m[b]);       // stabilized values
let z[b]        = sum[k](e[b, k]);            // denominator scan
let probs[b, j] = e[b, j] / z[b];             // output
```

**Diagnostic question:** "Give each binding site its own coordinate name — `q`,
`k`, `j`. All three range over the same domain. Which one survives into the
output? Which ones are consumed by `max` and `sum`?"

**Still runs?** Yes. `softmax(dim=-1)` collapses all three roles. The output
shape is correct regardless of which coordinate plays which role.

**Chapter:** 6, "Softmax Has Three Coordinate Roles"

---

### 5. Pullback sums the wrong route

**Wrong:**
```python
dA = G @ B        # missing transpose on B
# dA has shape [i, k] — right shape, wrong values
```

**Right:**
```rust
let dA[i, k] = sum[j](G[i, j] * B[k, j]);   // j is the path coordinate
let dB[k, j] = sum[i](G[i, j] * A[i, k]);   // i is the path coordinate
```

**Diagnostic question:** "Hold one denominator cell fixed — say `A[3, 7]`.
List every output cell that read it. Sum exactly those sensitivity routes.
Does your formula sum over the same coordinate?"

**Still runs?** Yes, when matrices are square. `G @ B` and `G @ B.T` have the
same shape. The numbers differ, but nothing crashes.

**Chapter:** 8, "Matrix Multiplication Teaches the Pullback"

---

### 6. Time is hidden inside mutation

**Wrong:**
```python
h = h0
for t in range(T):
    h = step(h, x[t+1])    # accidentally reads the future
```

**Right:**
```rust
let h[0] = h0;
let h[t in 1..T] = step(h[t - 1], x[t]);   // backward edge
```

**Diagnostic question:** "Rewrite the state as `h[t]` and mark every read of
`h[t - n]` or `h[t + n]`. Do any edges point forward in time?"

**Still runs?** Yes, if `x[t+1]` exists. The loop body is just an expression.
The dependency direction is not checked.

**Chapter:** 10, "Time Steps Are Not Loops"

---

### 7. Storage is inferred from notation too early

**Wrong:** assuming `h[t in 1..T]` requires `T` array slots, when only `h[T-1]`
is observed.

**Right:** The recurrence defines the family. Observation determines what must
be stored. `let final = h[T-1]` needs only a rolling window.

**Diagnostic question:** "Separate three things: (1) which values are defined,
(2) which values are observed, (3) what storage policy follows from (1) and
(2)."

**Still runs?** Yes. Storage is a performance question, not a correctness
question — until you run out of memory.

**Chapter:** 11, "Storage Follows Observation"

---

### 8. Attention uses the right tensors at the wrong position

**Wrong:**
```rust
let output[b, i, d] = sum[j](weights[b, i, j] * V[b, i, d]);
//                                               ^^^ should be j, not i
```
The gather reads from the query position instead of the key position.

**Right:**
```rust
let output[b, i, d] = sum[j](weights[b, i, j] * V[b, j, d]);
```

**Diagnostic question:** "State the communication sentence: `i` asks, `j`
answers, and `V[j, d]` is carried back. Does the gather read `V` at index `i`
or index `j`?"

**Still runs?** Yes. `V[b, i, d]` and `V[b, j, d]` have the same shape when
`query_len == key_len` (self-attention). The model still trains — it just
attends to the wrong thing.

**Chapter:** 14, "Attention as Named Communication"

---

### 9. Low-rank attention hides the bottleneck

**Wrong:** linear attention implemented as ordinary matrix multiplications
without naming the bottleneck coordinate `r`. The approximation looks like a
feature transform.

**Right:**
```rust
let KV[b, h, r, v] = sum[j](K_phi[b, h, j, r] * V[b, h, j, v]);
let out[b, h, i, v] = sum[r](Q_phi[b, h, i, r] * KV[b, h, r, v]);
```

**Diagnostic question:** "Name the bottleneck coordinate `r`. Where did `i` and
`j` stop talking directly? Which coordinate replaced the direct communication
path?"

**Still runs?** Yes. The shapes are valid for any `r`. Only the approximation
quality changes — and that quality is invisible to shape checks.

**Chapter:** 16, "Dynamic Routing and Low-Rank Communication"

---

### 10. Dynamic routing hides dropped or overloaded tokens

**Wrong:** MoE dispatch that silently drops overflow tokens without naming the
`keep` mask.

**Right:**
```rust
let route[b, t] = argmax[e](gate_prob[b, t, e]);
let slot[b, t]  = count_assignments[b, t, route[b, t]];
let keep[b, t]  = slot[b, t] < capacity;
```

**Diagnostic question:** "Name `route[b, t]`, `slot[b, t]`, and `keep[b, t]`.
For any token, can you trace whether it survived routing? If the answer is
'the information is scattered across a mask tensor,' the source is hiding too
much."

**Still runs?** Yes. Dropped tokens receive a fallback value. The loss goes
down. The model learns — just not what you think about the dropped tokens.

**Chapter:** 16

---

### 11. Coordinate function hides the wrong fact

**Wrong:**
```python
y = normalize(x, dim=-1)   # what is dim=-1? time? class? feature?
```

**Right:**
```rust
let y = normalize[class](x);   // the bracket says what's normalized
```

**Diagnostic question:** "Rewrite the call with bracketed coordinate arguments.
If you cannot ground the bracketed coordinate in the argument's rank, the
helper is hiding the decision that matters."

**Still runs?** Yes. `dim=-1` always refers to some axis. The question is
whether it refers to the RIGHT axis.

**Chapter:** 2, "Axis Roles Are Not Axis Positions"

---

### 12. Coordinate function asks for too much

**Wrong:**
```rust
move_channel[channel, batch, height, width](x)
```
The caller repeats coordinates that are already determined by `x`.

**Right:**
```rust
move_channel[channel](x)   // surrounding coords inferred via rest packs
```

**Diagnostic question:** "Which coordinate is the decision? That one goes in
the brackets. Everything else belongs in the function body's rest packs."

**Still runs?** Yes. The verbose version is correct, just ceremonial. But
ceremony makes the important bracket invisible.

**Chapter:** 15, "What the Notation Refuses to Hide"

---

## Diagnostic Workflow

When you face a tensor shape problem, walk this decision tree:

```text
Did the program crash with a shape error?
  ├── YES → The shapes don't match. But WHY?
  │     → Audit: write the expected coordinate relation.
  │       Which coordinate has the wrong extent?
  │       → Pattern #1 (swapped roles) or #3 (wrong reduction)
  │
  └── NO → The shapes match, but the answer is wrong.
        → The bug is semantic. It has the right shape.
        → Ask: would the wrong role still have the right shape?
        → If YES:
            ├── Is it a broadcast bug?   → Pattern #2
            ├── Is it a reduction bug?   → Pattern #3
            ├── Is it a softmax bug?     → Pattern #4
            ├── Is it a gradient bug?    → Pattern #5
            ├── Is it a recurrence bug?  → Pattern #6
            ├── Is it an attention bug?  → Patterns #8-9
            └── Is it a routing bug?     → Pattern #10
```

**For gradient bugs specifically**, add the gradient audit:

```text
1. Write the denominator's coordinates (the gradient's address).
2. For each term in the forward expression that used the denominator:
   a. Which coordinates did the term omit? (broadcast → sum in backward)
   b. Which coordinates did the term consume? (reduce → broadcast in backward)
3. The path coordinate = output coordinates ∖ denominator coordinates.
   Insert sum over the path coordinate.
```

## Coordinate Smells

These patterns are not bugs yet, but they signal danger. Each is a place where
a future refactor is likely to silently break something.

| Smell | Why It's Dangerous | What to Write Instead |
|---|---|---|
| `reshape(...).transpose(...).reshape(...)` | Roles are spread across three ops. No single line states the coordinate relation. | `y[b*g, f*s] = x[b, f, g, s]` |
| `dim=-1` on a tensor whose rank changed | `-1` points to whatever axis happens to be last. A reshape two lines up silently changes what `-1` means. | `softmax[class](x)` |
| Square matrices used with positional APIs | `A.sum(dim=0)` and `A.sum(dim=1)` both produce vectors of the same length. No shape check can distinguish row sums from column sums. | `sum[j](A[i, j])` vs `sum[i](A[i, j])` |
| `.sum(dim=...)` over an axis that was transposed upstream | The axis index stays the same, but the role at that position changed. | `sum[class](x)` — the name follows the role, not the position |
| `x @ W.T + b` without comment | What is the batch dimension? What is the feature? The line runs but states nothing. | `y[b, out] = sum[in](x[b, in] * W[out, in]) + bias[out]` |
| `scatter`/`gather` without named route | Dynamic dispatch creates new coordinates (`route`, `slot`, `keep`). If they're unnamed, overflow and imbalance are invisible. | Name `route[b,t]`, `slot[b,t]`, `keep[b,t]` |
| Batch dimension absorbed into `dim=0` convention | "Batch is always dim 0" — until a data loader produces `[T, B, F]` instead of `[B, T, F]`. | `x[b, t, f]` |

## Quick Reference Card

Print this. Tape it to your monitor. Use it at 3 AM.

```text
╔══════════════════════════════════════════════════════════════╗
║              THE COORDINATE AUDIT (five questions)           ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  For any tensor expression, pick one output cell and ask:    ║
║                                                              ║
║  1. Which coordinates did this cell read?                    ║
║     → Those are the SURVIVORS. They must appear on the       ║
║       left-hand side AND in every term that inspects them.   ║
║                                                              ║
║  2. Which coordinates were summed away?                      ║
║     → Those are CONSUMED. They appear in sum/max/mean[...]   ║
║       and are absent from the result.                        ║
║                                                              ║
║  3. Which result coordinates are absent from one term?       ║
║     → That term is INDEPENDENT of those coordinates.         ║
║       The value is broadcast/reused along them.              ║
║                                                              ║
║  4. Which coordinate is the address of this gradient?        ║
║     → @loss/@X has X's coordinates. The gradient address     ║
║       IS the denominator's coordinates.                      ║
║                                                              ║
║  5. Would the wrong role still have the right shape?         ║
║     → If YES, write the coordinates down. The bug lives      ║
║       in the role the source didn't name.                    ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
```

## Worked Examples

### Example 1: The Silent Softmax Swap

**Initial symptom:** A deployed classifier's calibration drifts over three
weeks. Cross-entropy loss decreases normally throughout. No crash, no NaN.

**Diagnostic applied:** Pattern #4 (one logical axis, several scopes). The
data pipeline was refactored to produce `[class, batch]` instead of `[batch,
class]`. The line `softmax(logits, dim=-1)` still runs. Previously it
normalized each example's class scores. Now it normalizes each class's batch
scores — the examples compete with each other inside every class.

**Root cause:** `dim=-1` silently changed meaning because the last axis changed
from `class` to `batch`. When `num_classes == batch_size`, the shapes are
square and no error is raised.

**Coordinate fix:**
```rust
let probs[batch, class] = softmax[class](logits[batch, class]);
```
`class` in the bracket is invariant under transposition. The name follows the
role, not the position.

---

### Example 2: The Gradient That Averaged Across Batch

**Initial symptom:** Training runs. Loss decreases. But the model's per-example
predictions are suspiciously uniform — as if every example receives an
average gradient.

**Diagnostic applied:** Pattern #5 (pullback sums wrong route). A batched
matmul's gradient accidentally included `batch` in the reduction:

```rust
let dA[batch, i, k] = sum[batch, j](G[batch, i, j] * B[batch, k, j]);
//                                          ^^^^^ should not be here
```

The gradient preserves `[batch, i, k]` shape but the `sum[batch]` means every
example receives the SAME sensitivity, averaged across the batch.

**Root cause:** Copy-paste from an unbatched pullback, with `batch` added to
the sum "for completeness."

**Coordinate fix:** Remove `batch` from the reduction:
```rust
let dA[batch, i, k] = sum[j](G[batch, i, j] * B[batch, k, j]);
```

---

### Example 3: The RNN That Read the Future

**Initial symptom:** Validation loss is suspiciously good. Better than
training loss. The model seems to "predict" perfectly.

**Diagnostic applied:** Pattern #6 (time hidden in mutation). The RNN loop
indexed `x[t+1]` instead of `x[t]` inside the step function. The recurrence
was cheating by reading the next token.

**Root cause:** The loop-based code did not distinguish between "compute from
past" and "read from future." The shape `[batch, time, feature]` was valid
either way.

**Coordinate fix:**
```rust
let h[0, b] = init[b];
let h[t in 1..T, b] = step(h[t - 1, b], x[t, b]);
```
The index `t - 1` is a visible backward edge. A compiler pass can check that
every read of the recurrence family uses a smaller index.

---

**Line to keep:** the shape tells you whether the program ran. The coordinates
tell you whether it ran for the right reason.
