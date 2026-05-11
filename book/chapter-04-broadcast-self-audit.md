---
layout: book
title: "Chapter 4 · The Broadcast Self-Audit"
---

# Chapter 4 · The Broadcast Self-Audit

> "Silence is not absence. Silence is a claim. And claims can be checked."
>
> — The author, after a 4 AM broadcast bug

*Combinations · The inversion rule: what broadcasts forward collects backward*

---

You have now seen two coordinate operations: reduction (which consumes a coordinate) and broadcasting (which copies along one). Chapter 2 introduced them as separate primitives. Chapter 3 showed how coordinate-aware functions thread them through signatures.

But there is a deeper relationship between them—one that this chapter is dedicated to making explicit. It is the relationship that governs every gradient, every parameter update, every backward pass you will ever write. And it can be stated in one sentence:

**What broadcasts in the forward pass is reduced in the backward pass. What is reduced in the forward pass is broadcast in the backward pass.**

This is the Inversion Rule.

---

## Two Lines, Two Directions

Read these two lines:

```rust
// Forward
let out[i, j] = A[i, j] + bias[j];

// Backward: gradient of bias
let d_bias[j] = sum[i](d_out[i, j]);
```

Forward: `bias[j]` omits `i` in its index pattern. The coordinate `i` is absent. So `bias` is copied along `i`—every position along `i` receives the same `bias` value. This is a broadcast.

Backward: `d_bias[j]` is the gradient with respect to `bias`. How to get it: `d_out[i, j]` carries gradient signals from every `(i, j)` position. `bias` contributed to all of them equally. To update `bias`, collect all those signals. The collection is a sum—over `i`. The coordinate that was broadcast forward is reduced backward.

Now the inverse:

```rust
// Forward
let row_sum[i] = sum[j](matrix[i, j]);

// Backward: gradient of matrix
let d_matrix[i, j] = d_row_sum[i];  // broadcast j back
```

Forward: `sum[j]` consumes `j`. The output `row_sum[i]` has no `j`. Every `j` position was collapsed into a single sum.

Backward: `d_row_sum[i]` is the gradient with respect to `row_sum`. To send it back to `matrix[i, j]`, broadcast `d_row_sum` along `j`—every `j` position receives the same gradient signal. The coordinate that was reduced forward is broadcast backward.

Two lines. Two directions. One rule. The Inversion Rule is not a separate piece of mathematics bolted onto the coordinate system. It is the coordinate system, read in reverse.

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

Now let's see what happens when a broadcast is shape-correct but semantically wrong.

A programmer writes a temperature-scaled softmax:

```rust
let logits[batch, class] = model(x[batch, feature]);
let scaled[batch, class] = logits[batch, class] / temperature;
let probs[batch, class] = softmax[class](scaled[batch, class]);
```

`temperature` is a scalar. It broadcasts over both `batch` and `class`. The auditor asks Question 2: is independence justified? Yes—temperature is a global scaling factor that applies uniformly to all logits.

But the programmer intended `temperature` to be per-class—different classes get different temperatures. They wrote:

```rust
let temperature[class] = get_per_class_temperature();
let scaled[batch, class] = logits[batch, class] / temperature[class];
```

Now `temperature` broadcasts over `batch` but *not* over `class`. The auditor asks: is `temperature` independent of `batch`? Yes—all batch elements share the same per-class temperatures. Is `temperature` independent of `class`? No—and the index pattern `temperature[class]` does *not* broadcast over `class`. The code is correct.

Now suppose the programmer accidentally wrote:

```rust
let temperature = get_per_class_temperature();  // returns scalar by mistake
let scaled[batch, class] = logits[batch, class] / temperature;
```

`temperature` is a scalar broadcasting over everything. The shapes work. The code runs. But the per-class variation is gone—every class gets the same temperature. The loss descends but plateaus higher. The bug is a broadcast that is wider than intended.

The auditor's Question 2 catches this: "is independence genuinely justified?" The programmer intended `temperature` to depend on `class`. But the scalar `temperature` is independent of `class`—it broadcasts over it. The broadcast is a claim that `temperature` doesn't depend on `class`. The claim is false. The audit fails.

In a positional framework, `temperature` is just a number. The positional notation doesn't distinguish between "this is a scalar because it's genuinely global" and "this is a scalar because I forgot to make it per-class." The broadcast self-audit forces the distinction.

Now consider a more subtle failure—one that the audit catches but positional debugging would miss for hours. A programmer is implementing a weighted loss:

```rust
let losses[batch, class] = cross_entropy_per_class(logits[batch, class], labels[batch, class]);
let class_weights[class] = get_class_weights();
let weighted[batch] = mean[class](losses[batch, class] * class_weights[class]);
```

`class_weights[class]` broadcasts over `batch`. The audit asks: is `class_weights` independent of `batch`? Yes—class weights are per-class, shared across all batch elements. The broadcast is justified.

Now the programmer refactors, making class weights adaptive based on batch statistics:

```rust
let class_weights = compute_adaptive_weights(losses);  // BUG: returns scalar by accident
let weighted[batch] = mean[class](losses[batch, class] * class_weights);
```

`compute_adaptive_weights` was supposed to return `[f32; class]` but returns a scalar. The audit catches this: `class_weights` broadcasts over both `batch` and `class`. But `class_weights` should depend on `class`—the broadcast over `class` is semantically wrong. The loss will compile, run, and descend. But every class gets the same weight. The adaptive weighting is silently disabled.

In a positional framework, `(batch, class) * scalar` and `(batch, class) * (class,)` are both valid—the shape of `class_weights` determines the behavior, and a shape mismatch produces a different broadcast instead of an error.

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

The shopping cart model from Chapter 7 is this diagram, narrated as a story. The forward pass is shopping: you walk through the aisles, items enter your cart, some are consumed (reduction), some are copied (broadcast). The backward pass is restocking: the manager reads the record backward, replenishing what was consumed and collecting what was copied.

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

Most real code has more than one broadcast. A linear layer with bias has one (bias over batch). A layer normalization has four (mean over feature, variance over feature, gamma over batch, beta over batch). When broadcasts compose, their backward reductions compose too. The auditor's toolkit handles them mechanically—one broadcast at a time—but the interactions are worth tracing.

Here is a multi-head attention projection followed by a residual connection:

```rust
fn attention_block[head, seq_q, seq_k, d](
    Q: [f32; ..batch, head, seq_q, d],
    K: [f32; ..batch, head, seq_k, d],
    V: [f32; ..batch, head, seq_k, d],
    W_o: [f32; head, d, d_out],
    b_o: [f32; d_out]
) -> [f32; ..batch, seq_q, d_out]
{
    let scores[..batch, head, seq_q, seq_k] =
        sum[d](Q[..batch, head, seq_q, d] * K[..batch, head, seq_k, d]) / (d ** 0.5);
    let weights[..batch, head, seq_q, seq_k] = softmax[seq_k](scores[..batch, head, seq_q, seq_k]);
    let context[..batch, head, seq_q, d] =
        sum[seq_k](weights[..batch, head, seq_q, seq_k] * V[..batch, head, seq_k, d]);
    let projected[..batch, seq_q, head, d_out] =
        sum[d](context[..batch, head, seq_q, d] * W_o[head, d, d_out]);
    projected[..batch, seq_q, d_out] + b_o[d_out]
}
```

Walk through the auditor's toolkit for the final expression `projected + b_o`. Output coordinates: `{..batch, seq_q, d_out}`. `projected` has `{..batch, seq_q, d_out}`—no broadcast. `b_o` has `{d_out}`—broadcasts over `{..batch, seq_q}`.

Now the gradient: `d_b_o[d_out] = sum[..batch, seq_q](d_out[..batch, seq_q, d_out])`. The broadcast set from Step 2 becomes the reduction set in the backward pass.

But there are hidden broadcasts too. Look at the `context` computation: `sum[seq_k](weights * V)`. `V` has `{..batch, head, seq_k, d}`. The output `context` has `{..batch, head, seq_q, d}`. Inside the sum, `weights[..batch, head, seq_q, seq_k] * V[..batch, head, seq_k, d]`—`V` omits `seq_q`, so `V` broadcasts over `seq_q`. Every `(seq_q, seq_k)` position within a given `(batch, head)` receives the same `V` value from `seq_k`. That broadcast is inside a reduction over `seq_k`. The backward pass: `dV[..batch, head, seq_k, d] = sum[seq_q](d_context[..batch, head, seq_q, d] * weights[..batch, head, seq_q, seq_k])`. The forward broadcast over `seq_q` becomes the backward sum over `seq_q`.

A double audit traces every broadcast in the expression and verifies that each backward reduction matches. The procedure is the same for one broadcast or ten. The coordinate sets tell you what to check.

---

### Auditing an Attention Block

Here is a smaller but complete attention block—the projection, not the full attention:

```rust
let context[..b, head, seq_q, d] =
    sum[seq_k](weights[..b, head, seq_q, seq_k] * V[..b, head, seq_k, d]);
let output[..b, seq_q, head, d_out] =
    sum[d](context[..b, head, seq_q, d] * W_o[head, d, d_out]);
let final[..b, seq_q, d_out] = output[..b, seq_q, d_out] + b_o[d_out];
```

The auditor's toolkit applied to each expression: output coordinate set, each operand's coordinate set, the broadcast set (output minus operand), the semantic justification, and the backward reduction. Here is the audit, line by line.

**Line 1: `sum[seq_k](weights * V)`**

```
Output coordinates: {..b, head, seq_q, d}
weights: {..b, head, seq_q, seq_k}  → broadcast: {} (no omission, but seq_k is reduced)
V:       {..b, head, seq_k, d}      → broadcast: {seq_q} (V omits seq_q)
```

`V` broadcasts over `seq_q` inside the reduction. This is correct: `V` provides values at each `seq_k` position, and those values are the same regardless of which `seq_q` is querying. The backward reduction: `dV[..b, head, seq_k, d] = sum[seq_q](d_context[..b, head, seq_q, d] * weights[..b, head, seq_q, seq_k])`. The broadcast set `{seq_q}` becomes the reduction set.

**Line 2: `sum[d](context * W_o)`**

```
Output coordinates: {..b, seq_q, head, d_out}
context: {..b, head, seq_q, d}  → broadcast: {} (no omission, d is reduced)
W_o:     {head, d, d_out}       → broadcast: {..b, seq_q} (W_o omits batch and seq_q)
```

`W_o` broadcasts over `..b` and `seq_q`. This is correct: the output projection weight is the same for all batch elements and all query positions. The backward reduction: `dW_o[head, d, d_out] = sum[..b, seq_q](d_output[..b, seq_q, head, d_out] * context[..b, head, seq_q, d])`.

**Line 3: `output + b_o`**

```
Output coordinates: {..b, seq_q, d_out}
output: {..b, seq_q, d_out}  → broadcast: {}
b_o:    {d_out}              → broadcast: {..b, seq_q}
```

`b_o` broadcasts over `..b` and `seq_q`. Correct: output bias is the same for all batch elements and query positions. Backward: `db_o[d_out] = sum[..b, seq_q](d_final[..b, seq_q, d_out])`.

**Putting it together.** Three expressions. Five broadcasts (one hidden inside the first reduction, two explicit in the projections, one in the bias, one more inside the backward). Every broadcast has a semantic justification—"this value does not depend on that coordinate." Every broadcast has a corresponding backward reduction over the same coordinate set. The Inversion Rule holds for all of them.

Now ask yourself: in the PyTorch version of this block, how many of these five broadcasts would you have noticed? The `b_o` broadcast is obvious—`projected + b_o` with different shapes. The `W_o` broadcast is less obvious—`torch.matmul` or `einsum` hides it. The `V` broadcast inside the `sum[seq_k]` reduction is nearly invisible—it's inside a `matmul` or `einsum` where the broadcasting is implicit. Two of the five broadcasts would have escaped notice entirely. The audit reveals them.

---

## The Audit Without Einlang

You do not need Einlang to perform a broadcast self-audit. You need to know which coordinate is being broadcast over. The question is the same in any framework. The difference is how hard the framework makes it to answer.

In PyTorch, broadcasting is shape-driven. `(32, 64) + (64,)` broadcasts along axis 0. Which coordinate is axis 0? The code doesn't say. You infer it from context: axis 0 is probably `batch`, axis 1 is probably `feature`. But "probably" is not a check.

In Einlang, broadcasting is name-driven. `out[i, j] = A[i, j] + bias[j]` omits `i`. The omitted coordinate is the broadcast coordinate. The code says it.

The audit questions are the same. But in PyTorch, answering Question 1 ("what coordinate am I broadcasting over?") requires shape reconstruction. In Einlang, answering Question 1 requires reading a bracket. The audit is the same. The effort is not.

In PyTorch, `output = projected + b_o` requires knowing that `projected` has shape `(batch, seq_q, d_out)` and `b_o` has `(d_out,)`. Transpose `projected` upstream, and the broadcast alignment silently changes. The Einlang version `projected[..batch, seq_q, d_out] + b_o[d_out]` is layout-independent—the name `d_out` identifies the shared axis regardless of position. The audit is the same for 2D or 6D tensors, because the number of names, not the number of axes, determines the work.

---

*The last broadcast you wrote—intentionally or not—is in your code right now, in some `A + b` or `scale * x` or `mean[dim]` with `keepdim=True`. Apply the three questions. Is the answer to Question 2 a confident yes? If not, that broadcast is a claim that deserves a name.*

---

### Stop and Think: Audit Your Own Broadcasts

Your most recent project almost certainly contains implicit broadcasts: `+`, `*`, `/`, `-` between tensors of different ranks. Each broadcast is a claim of independence. The audit questions make the claims visible:

1. **What are the coordinate sets?** What are the output coordinates? What are each operand's coordinates? If you know the shape conventions of your project, you already know the answer. If not, the `.shape` attribute tells you.

2. **What is the broadcast set?** For each operand, subtract its coordinate set from the output set. The difference is what that operand broadcasts over. This is coordinate set subtraction—the same operation from Chapter 2.

3. **Can you name the broadcast coordinate?** Not "axis 0" or "the first dimension"—the coordinate name. Is it `batch`? `channel`? `height`? `width`? If you cannot name it, the broadcast's identity is untethered from any declaration. The claim exists, but no one recorded it.

4. **Is it justified?** Does the broadcasting operand genuinely not depend on those coordinates? If the answer is "probably" or "I think so" rather than "yes, by construction"—the broadcast deserves a second look.

In a typical codebase, at least one broadcast fails question 4. That broadcast is a claim of independence that is not confidently true. It is a bug waiting for the right input shape.
