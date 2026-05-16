---
layout: book
title: "Chapter 4 · The Broadcast Self-Audit"
---

# Chapter 4 · The Broadcast Self-Audit

> "Silence is not absence. Silence is a claim. And claims can be checked."

*Combinations · The inversion rule: what broadcasts forward collects backward*

---

Every broadcast you write is a claim you didn't know you were making.

The claim is: *this value does not depend on the coordinate I am omitting.* When you write `out[i, j] = A[i, j] + bias[j]`, the omission of `i` from `bias[j]` claims the bias is the same for every `i`. When you write `scaled[batch, class] = logits[batch, class] / temperature[class]`, the omission of `batch` from `temperature[class]` claims the temperature is the same for every batch element.

Most of the time, the claim is true. The bias genuinely doesn't depend on the batch element. The temperature genuinely doesn't depend on the class. But when the claim is false, the code still runs. The shapes still match. The loss still descends — just to a higher plateau. And you spend an afternoon wondering why your adaptive class weights aren't adapting.

The broadcast self-audit is three questions you ask before the broadcast becomes a bug. Thirty seconds. It catches the afternoon.

Now read two lines:

```rust
// Forward
let out[i, j] = A[i, j] + bias[j];

// Backward: gradient of bias
let d_bias[j] = sum[i](d_out[i, j]);
```

Forward: `bias[j]` omits `i` in its index pattern. The coordinate `i` is absent, so `bias` is copied along `i`. Broadcast.

Backward: `d_bias[j]` is the gradient with respect to `bias`. `d_out[i, j]` carries gradient signals from every `(i, j)` position — `bias` contributed to all of them equally. To update `bias`, collect all those signals: sum over `i`. Reduction.

The coordinate that was broadcast forward (`i`) is the coordinate reduced backward. This is the Inversion Rule. Every forward operation has a backward dual. Broadcast becomes reduction. Reduction becomes broadcast.

```rust
// Forward: reduction consumes j
let row_sum[i] = sum[j](matrix[i, j]);

// Backward: broadcast j back
let d_matrix[i, j] = d_row_sum[i];
```

Forward: `sum[j]` consumes `j`. Every `j` position collapses into a single sum. Backward: `d_row_sum[i]` broadcasts along `j` — every `j` position receives the same gradient signal. The consumed coordinate is reborn as a broadcast.

Two lines. Two directions. One rule. The Inversion Rule is not mathematics bolted onto the coordinate system. It is the coordinate system, read in reverse.

---

**Derive it yourself.** Given `let scaled[batch, class] = logits[batch, class] / temperature[class]`, write the backward gradient for `temperature`. Which coordinate does the sum go over? Why? Write it down before reading further.

---

## The Self-Audit: Three Questions

Now apply this to your own code. Every broadcast you write is a claim. The claim is: *this value does not depend on the coordinate I am omitting.* If the claim is false, the forward pass is wrong. If the claim is true but the backward pass doesn't reduce over the omitted coordinate, the gradient is wrong.

Three questions for every broadcast:

**Question 1: What coordinate am I broadcasting over?** Is the name visible in the code, or is it inferred from position?

In `out[i, j] = A[i, j] + bias[j]`, the omitted coordinate is `i`. The code says so: `A` has `(i, j)`, `bias` has `(j)`. The difference is `{i}`. The broadcast is visible in the index patterns.

In `out = A + bias`, the broadcast is invisible—the shapes determine what happens. The code doesn't say which coordinate is being broadcast over.

**Question 2: Is independence genuinely justified?** Does the broadcast value genuinely not depend on that coordinate?

A bias term in a linear layer should not depend on the batch index. Each sample gets the same bias. The broadcast over `batch` is semantically justified.

A temperature scaling factor in a softmax should not depend on the class index. The broadcast over `class` is semantically justified.

But what about a mask that you broadcast over the sequence length? If the mask depends on the sequence position—if later positions are masked differently than earlier ones—then broadcasting a single mask value over all positions is semantically wrong. The shapes would work. The code would run. But the mask would not encode the position-dependent pattern you intended.

The broadcast self-audit asks: *is this broadcast a computational convenience, or a semantic claim?* If it is a semantic claim, is the claim true?

**Question 3: What will the gradient do?** Does the backward reduction produce the right shape for the parameter update?

In `d_bias[j] = sum[i](d_out[i, j])`, the sum over `i` produces a gradient of shape `(j)`—exactly `bias`'s shape. The parameter update `bias -= lr * d_bias` is well-shaped.

But what if you wrote the broadcast differently? What if `bias` had shape `(1, j)` and broadcasting expanded it to `(i, j)`? The gradient would still be `sum[i](d_out[i, j])`, producing `(j,)`. If your optimizer expects `(1, j)`, you need a reshape. The reshape is a positional hack to make the shapes align. The named version produces the correct shape by construction—the gradient has the same coordinates as the parameter.

---

## The Auditor's Toolkit

A systematic procedure. Given any expression that contains a broadcast, you can audit it with these steps:

1. **List the coordinate sets.** Write down the coordinates of every tensor in the expression.

2. **Compute the broadcast sets.** For each term, subtract its coordinate set from the output coordinate set. The difference is what that term broadcasts over.

3. **Check justification.** For each broadcast, ask: is it semantically correct for this term to be independent of these coordinates?

4. **Predict the gradient.** For each broadcast, write the backward reduction: sum over the broadcast set. Verify that the result has the same coordinates as the parameter.

Let's apply this to a realistic example. Here is a layer normalization with a learnable scale and shift:

```rust
fn layer_norm[feature](x: [f32; ..batch, feature],
                        gamma: [f32; feature],
                        beta: [f32; feature])
    -> [f32; ..batch, feature]
{
    let mean[..batch] = mean[feature](x[..batch, feature]);
    let centered[..batch, feature] = x[..batch, feature] - mean[..batch];
    let var[..batch] = mean[feature](centered[..batch, feature] ** 2.0);
    (centered[..batch, feature] / (var[..batch] ** 0.5 + 1e-5)) * gamma[feature] + beta[feature]
}
```

Step through the auditor's toolkit.

**Step 1: Coordinate sets.**

- `x`: `{..batch, feature}`
- `mean`: `{..batch}` — `feature` was consumed by `mean[feature]`
- `centered`: `{..batch, feature}`
- `var`: `{..batch}` — `feature` was consumed by `mean[feature]`
- `gamma`: `{feature}`
- `beta`: `{feature}`

**Step 2: Broadcast sets.**

The final expression is `centered / (var ** 0.5 + eps) * gamma + beta`. The output coordinates are `{..batch, feature}`.

- `centered`: has `{..batch, feature}`. Broadcast set = `{}`. No broadcast.
- `var`: has `{..batch}`. Broadcast set = `{feature}`. `var` broadcasts over `feature`.
- `gamma`: has `{feature}`. Broadcast set = `{..batch}`. `gamma` broadcasts over `..batch`.
- `beta`: has `{feature}`. Broadcast set = `{..batch}`. `beta` broadcasts over `..batch`.

**Step 3: Justification.**

- `var` broadcasts over `feature`: justified. The variance is computed per-batch-element, then applied to all features. This is the definition of layer normalization.
- `gamma` broadcasts over `..batch`: justified. `gamma` is a per-feature parameter. Every batch element gets the same scale.
- `beta` broadcasts over `..batch`: justified. Same reasoning as `gamma`.

**Step 4: Gradient prediction.**

- `d_var[..batch] = sum[feature](d_out[..batch, feature] * ...)`. The gradient sums over `feature`—the broadcast set. Result: `{..batch}`, matching `var`.
- `d_gamma[feature] = sum[..batch](d_out[..batch, feature] * ...)`. The gradient sums over `..batch`—the broadcast set. Result: `{feature}`, matching `gamma`.
- `d_beta[feature] = sum[..batch](d_out[..batch, feature])`. Same. Result: `{feature}`, matching `beta`.

Every gradient has the same coordinates as its parameter. The broadcast sets from Step 2 become the reduction sets in Step 4. The Inversion Rule, applied mechanically.

---

## When the Audit Fails

A programmer writes a temperature-scaled softmax. The intent is per-class temperatures:

```rust
let temperature[class] = get_per_class_temperature();
let scaled[batch, class] = logits[batch, class] / temperature[class];
```

`temperature` broadcasts over `batch` but not `class`. The auditor asks: is `temperature` independent of `batch`? Yes. Independent of `class`? No—and the index pattern correctly omits `batch` but includes `class`.

Now suppose the programmer accidentally wrote:

```rust
let temperature = get_per_class_temperature();  // returns scalar by mistake
let scaled[batch, class] = logits[batch, class] / temperature;
```

`temperature` is a scalar—broadcasts over everything. The shapes work. The loss descends but plateaus higher. The auditor's Question 2 catches it: the broadcast claims `temperature` is independent of `class`. The claim is false.

A second example. Adaptive class weights for a weighted loss:

```rust
let class_weights = compute_adaptive_weights(losses);  // BUG: returns scalar by accident
let weighted[batch] = mean[class](losses[batch, class] * class_weights);
```

`compute_adaptive_weights` was supposed to return `[f32; class]` but returns a scalar. The scalar broadcasts over `class`—every class gets the same weight. The adaptive weighting is silently disabled. The auditor asks: is `class_weights` independent of `class`? It shouldn't be.

In a positional framework, both bugs survive because `(batch, class) / scalar` and `(batch, class) * scalar` are perfectly valid. The broadcast is silent. The audit makes it speak.

---

## The Inversion Rule in One Diagram

```
Forward                         Backward
-------                         --------

Reduction consumes {j}           →    Broadcast {j} back
    sum[j](A[i, j])                      d_sum[i] → d_A[i, j]

Broadcast omits {i}              →    Reduction collects over {i}
    A[i, j] + bias[j]                   d_bias[j] = sum[i](d_out[i, j])

Permute rearranges {i, j}        →    Permute rearranges back
    y[j, i] = A[i, j]                   d_A[i, j] = d_y[j, i]

Elementwise preserves             →    Elementwise preserves
    y[i, j] = f(x[i, j])                d_x[i, j] = f'(x[i, j]) * d_y[i, j]
```

Every forward operation has a backward dual. The dual is not a separate rule. It is the forward rule, read backward, with the coordinate names as the thread connecting the two directions.

Reduction → Broadcast. Broadcast → Reduction. Permute → Permute. Elementwise → Elementwise.

Think of the forward pass as shopping: you walk through the aisles, items enter your cart, some are consumed (reduction), some are copied (broadcast). The backward pass is restocking: the manager reads the record backward, replenishing what was consumed and collecting what was copied. The shopping cart will return in Chapter 8 when we derive gradients systematically.

The coordinate names are on both sides of the receipt. The Inversion Rule is the guarantee that the two sides match.

---

## The Audit as a Habit

The broadcast self-audit is not a tool. It is a habit. You don't run it. You ask it.

Before you merge a pull request that contains a broadcast, ask the three questions. Before you write a custom backward pass, trace the Inversion Rule for every broadcast in the forward pass. Before you debug a gradient that's the wrong shape, check whether the broadcast set and the reduction set match.

The questions cost seconds. The bugs they catch cost hours. The ratio—as the epilogue will remind you—is favorable.

But there is a deeper reason to practice the audit. Every time you ask "what coordinate am I broadcasting over?" you are doing something that positional notation makes difficult and named notation makes easy: you are connecting the operation to its intent. The broadcast is not just a shape compatibility check. It is a semantic claim. The audit makes the claim explicit.

---

## The Consumption Self-Audit

Broadcast and consumption are duals. The broadcast self-audit asks: *what coordinate am I silent on?* The consumption self-audit asks: *what coordinate am I erasing?* Both deserve their own diagnostic tool.

Just as a broadcast is a claim of independence, a reduction is a claim of dispensability. `sum[class](x[batch, class])` claims: *the coordinate `class` can be collapsed without losing information that other coordinates depend on.* If `class` carries structure that downstream operations rely on, the reduction is semantically wrong—even if the shapes match.

Three questions for every reduction:

**Question 1: What coordinate am I consuming?** Is the name visible in the code?

In `let row_sums[i] = sum[j](matrix[i, j])`, the consumed coordinate is `j`. The reduction bracket says so. In `x.mean(dim=1)`, the consumed coordinate is "whatever is at position 1." The name is absent.

**Question 2: Does this coordinate appear in every operand of the reduction body?**

In `sum[k](A[i, k] * B[k, j])`, `k` appears in both `A` and `B`. The reduction is well-formed. In `sum[class](x[batch, channel] + bias[channel])`, `class` appears nowhere—the compiler reports "reduction coordinate `class` not found." The check is mechanical.

**Question 3: What will the backward pass do?** The consumed coordinate becomes a broadcast in the gradient.

Forward: `let row_sums[i] = sum[j](matrix[i, j])`. Consumed: `j`. Backward: `d_matrix[i, j] = d_row_sums[i]`. The forward reduction over `j` becomes a backward broadcast over `j`. The Inversion Rule, applied to consumption.

Now put the two audits side by side:

```
BROADCAST SELF-AUDIT                    CONSUMPTION SELF-AUDIT
─────────────────────                   ───────────────────────
Q1: What coordinate am I silent on?     Q1: What coordinate am I consuming?
Q2: Is independence genuinely true?     Q2: Is it in every operand?
Q3: What will the gradient collect?     Q3: What will the gradient broadcast back?

Forward: omit coordinate → broadcast   Forward: consume coordinate → reduce
Backward: sum over omitted coordinate   Backward: broadcast the consumed coordinate
```

The two audits are the same audit, read in opposite directions. A broadcast claims independence. A reduction claims dispensability. Both claims are recorded in the brackets. Both claims are checkable. Both claims have backward consequences that the Inversion Rule predicts.

The broadcast audit catches the bug where a value is copied over a coordinate it should depend on. The consumption audit catches the bug where a coordinate is erased that downstream operations need. Together, they cover the two ways a coordinate's identity can be lost: by being ignored, or by being destroyed.

---

## The Double Audit: When Broadcasts Compose

Most real code has more than one broadcast. A linear layer with bias has one. A layer normalization has four. When broadcasts compose, their backward reductions compose too. The auditor's toolkit handles them mechanically—one broadcast at a time. But the interactions are worth tracing once, so you develop an instinct for finding the hidden ones.

Here is a complete attention projection block:

```rust
let context[..b, head, seq_q, d] =
    sum[seq_k](weights[..b, head, seq_q, seq_k] * V[..b, head, seq_k, d]);
let output[..b, seq_q, head, d_out] =
    sum[d](context[..b, head, seq_q, d] * W_o[head, d, d_out]);
let final[..b, seq_q, d_out] = output[..b, seq_q, d_out] + b_o[d_out];
```

Three lines. Let the auditor walk through each.

**Line 1: `sum[seq_k](weights * V)`**

```
Output coordinates: {..b, head, seq_q, d}
weights: {..b, head, seq_q, seq_k}  → broadcast: {} (no omission, seq_k is reduced)
V:       {..b, head, seq_k, d}      → broadcast: {seq_q} (V omits seq_q)
```

`V` broadcasts over `seq_q` inside the reduction. This is correct: `V` provides values at each `seq_k` position regardless of which `seq_q` is querying. The backward reduction: `dV[..b, head, seq_k, d] = sum[seq_q](d_context[..b, head, seq_q, d] * weights[..b, head, seq_q, seq_k])`. The broadcast set `{seq_q}` becomes the reduction set.

**Line 2: `sum[d](context * W_o)`**

```
Output coordinates: {..b, seq_q, head, d_out}
context: {..b, head, seq_q, d}  → broadcast: {}
W_o:     {head, d, d_out}       → broadcast: {..b, seq_q}
```

`W_o` broadcasts over `..b` and `seq_q`. Correct: the weight is the same for all batch elements and query positions. Backward: `dW_o[head, d, d_out] = sum[..b, seq_q](d_output[..b, seq_q, head, d_out] * context[..b, head, seq_q, d])`.

**Line 3: `output + b_o`**

```
Output coordinates: {..b, seq_q, d_out}
output: {..b, seq_q, d_out}  → broadcast: {}
b_o:    {d_out}              → broadcast: {..b, seq_q}
```

Backward: `db_o[d_out] = sum[..b, seq_q](d_final[..b, seq_q, d_out])`.

**Putting it together.** Three expressions. Five broadcasts—two of them hidden inside reductions. Every broadcast has a backward reduction over the same coordinate set. The Inversion Rule holds for all of them.

Now ask yourself: in the PyTorch version of this block, how many would you notice? Bias over batch is obvious. Weight over batch and sequence is visible but easy to miss. `V` broadcasting over `seq_q` inside a reduction over `seq_k`—that's nearly invisible. Two of five broadcasts escape notice entirely. The audit reveals them.

---

Take a breath. This was the densest section of the chapter. Three lines of code, five broadcasts, five backward reductions. If it felt like a lot—good. It is a lot. But here is what matters: you never have to do this audit for these three lines again. The next time you see `sum[seq_k](weights * V)` in a transformer, you already know: `V` broadcasts over `seq_q`, and the backward pass sums over `seq_q`. The audit isn't a procedure you run every time. It's an instinct you build by running it once and then remembering the answer.

The procedure worked. It gave you the answer. Now the answer is yours.

---

## The Audit Without Einlang

You do not need Einlang to perform a broadcast self-audit. You need to know which coordinate is being broadcast over. The question is the same in any framework. The difference is how hard the framework makes it to answer.

In PyTorch, broadcasting is shape-driven. `(32, 64) + (64,)` broadcasts along axis 0. Which coordinate is axis 0? The code doesn't say. You infer it from context: axis 0 is probably `batch`, axis 1 is probably `feature`. But "probably" is not a check.

In Einlang, broadcasting is name-driven. `out[i, j] = A[i, j] + bias[j]` omits `i`. The omitted coordinate is the broadcast coordinate. The code says it.

The audit questions are the same. But in PyTorch, answering Question 1 ("what coordinate am I broadcasting over?") requires shape reconstruction. In Einlang, answering Question 1 requires reading a bracket. The audit is the same. The effort is not.

In PyTorch, `output = projected + b_o` requires knowing that `projected` has shape `(batch, seq_q, d_out)` and `b_o` has `(d_out,)`. Transpose `projected` upstream, and the broadcast alignment silently changes. The Einlang version `projected[..batch, seq_q, d_out] + b_o[d_out]` is layout-independent—the name `d_out` identifies the shared axis regardless of position. The audit is the same for 2D or 6D tensors, because the number of names, not the number of axes, determines the work.

---

### What the Audit Reveals

The last broadcast you wrote—intentionally or not—is in your code right now, in some `A + b` or `scale * x` or `mean[dim]` with `keepdim=True`. Apply the three questions. Every broadcast is a claim of independence. The broadcast set—the coordinates the operand omits—is the claim written in set-subtraction notation.

In a typical codebase, at least one broadcast fails the semantic question: does the broadcasting operand genuinely not depend on those coordinates? "Probably" or "I think so" is not a yes. That broadcast is a claim of independence that is not confidently true. It is a bug waiting for the right input shape.

The audit reveals it. Not because the audit is sophisticated—it is three questions and a set subtraction. Because the audit asks a question the code itself does not. Positional notation records that a broadcast happened. Named notation records which coordinate it happened over. The difference is whether the claim was recorded.

The audit catches individual broadcasts. But normalization functions—LayerNorm, RMSNorm, GroupNorm, InstanceNorm—share a deeper structure: a reduce-broadcast-elementwise skeleton that is identical across all of them, differing only in which coordinates play which roles. Chapter 5 extracts that skeleton. When four functions written by four people turn out to be the same function with different coordinate arguments, the pattern is not a coincidence. It is a design law.
