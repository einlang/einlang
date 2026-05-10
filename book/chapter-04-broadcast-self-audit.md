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

This is the Inversion Rule. This chapter is about what it means, how to check it, and why it catches the class of bugs that are shape-correct but semantically wrong.

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

Backward: `d_bias[j]` is the gradient with respect to `bias`. How do we get it? `d_out[i, j]` carries gradient signals from every `(i, j)` position. `bias` contributed to all of them equally. To update `bias`, we must collect all those signals. The collection is a sum—over `i`. The coordinate that was broadcast forward is reduced backward.

Now the inverse:

```rust
// Forward
let row_sum[i] = sum[j](matrix[i, j]);

// Backward: gradient of matrix
let d_matrix[i, j] = d_row_sum[i];  // broadcast j back
```

Forward: `sum[j]` consumes `j`. The output `row_sum[i]` has no `j`. Every `j` position was collapsed into a single sum.

Backward: `d_row_sum[i]` is the gradient with respect to `row_sum`. To send it back to `matrix[i, j]`, we must broadcast `d_row_sum` along `j`—every `j` position receives the same gradient signal. The coordinate that was reduced forward is broadcast backward.

Two lines. Two directions. One rule. The Inversion Rule is not a separate piece of mathematics bolted onto the coordinate system. It is the coordinate system, read in reverse.

---

## The Self-Audit: Three Questions

Now apply this to your own code. Every broadcast you write is a claim. The claim is: *this value does not depend on the coordinate I am omitting.* If the claim is false, the forward pass is wrong. If the claim is true but the backward pass doesn't reduce over the omitted coordinate, the gradient is wrong.

Three questions for every broadcast:

**Question 1: What coordinate am I broadcasting over?** Is the name visible in the code, or is it inferred from position?

In `out[i, j] = A[i, j] + bias[j]`, the omitted coordinate is `i`. The code says so: `A` has `(i, j)`, `bias` has `(j)`. The difference is `{i}`. The broadcast is visible in the index patterns.

In `out = A + bias`, the broadcast is invisible. The shapes determine what happens. If `A` is `(32, 64)` and `bias` is `(64,)`, NumPy broadcasts along axis 0. The code doesn't say which coordinate is being broadcast over. You infer it from shapes.

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

Let's build a systematic procedure. Given any expression that contains a broadcast, you can audit it with these steps:

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

In a positional framework, `temperature` is just a number. `logits / temperature` is valid regardless of whether `temperature` should be per-class or global. The positional notation doesn't distinguish between "this is a scalar because it's genuinely global" and "this is a scalar because I forgot to make it per-class." The broadcast self-audit forces the distinction.

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

## The Audit Without Einlang

You do not need einlang to perform a broadcast self-audit. You need to know which coordinate is being broadcast over. The question is the same in any framework. The difference is how hard the framework makes it to answer.

In PyTorch, broadcasting is shape-driven. `(32, 64) + (64,)` broadcasts along axis 0. Which coordinate is axis 0? The code doesn't say. You infer it from context: axis 0 is probably `batch`, axis 1 is probably `feature`. But "probably" is not a check.

In einlang, broadcasting is name-driven. `out[i, j] = A[i, j] + bias[j]` omits `i`. The omitted coordinate is the broadcast coordinate. The code says it.

The audit questions are the same. But in PyTorch, answering Question 1 ("what coordinate am I broadcasting over?") requires shape reconstruction. In einlang, answering Question 1 requires reading a bracket. The audit is the same. The effort is not.

Here is the PyTorch version of the attention block final line:

```python
output = projected + b_o  # b_o: (d_out,), projected: (batch, seq_q, d_out)
```

What coordinate is `b_o` broadcasting over? You must know the shapes. `projected` has shape `(batch, seq_q, d_out)`. `b_o` has shape `(d_out,)`. The broadcast aligns to the right: `b_o` is expanded over `(batch, seq_q)`. Those are two coordinates. Are they both semantically justified? `b_o` is an output bias—it should not depend on `batch` or `seq_q`. Yes, justified.

Now change one thing upstream. `projected` is transposed to `(d_out, batch, seq_q)` by a refactoring. `output = projected + b_o` still runs. `b_o` now broadcasts over `(batch, seq_q)`—same set of coordinates, but the positional alignment is different. The shapes `(d_out, batch, seq_q)` and `(d_out,)` align on axis 0, broadcasting over axes 1 and 2. The result is correct. But only because the refactoring preserved `d_out` as the first axis. If it had moved `d_out` to the middle, the broadcast would silently change.

The einlang version `projected[..batch, seq_q, d_out] + b_o[d_out]` does not care where `d_out` sits in the positional layout. The name `d_out` identifies the shared axis regardless of position. The broadcast is over `{..batch, seq_q}`—the coordinates in the output but not in `b_o`. The set subtraction is position-independent. The audit result is the same regardless of layout.

This is the broadcast self-audit's deepest value: it makes the audit layout-independent. The questions are about coordinate identities, not positions. The answers are stable under refactoring. The audit takes the same effort for a 2D tensor as for a 6D tensor—because the number of coordinate names, not the number of axes, determines the work.

In the next chapter, we see what happens when coordinate-aware functions compose into reusable skeletons—patterns that are identical across normalization, softmax, and attention, differing only in which coordinates play which roles. And we will discover that the skeleton itself is a broadcast self-audit, applied at the level of function signatures.

---

*Stop. Find the last broadcast you wrote—intentionally or not. It's in your code right now, in some `A + b` or `scale * x` or `mean[dim]` with `keepdim=True`. Apply the three questions. Is the answer to Question 2 a confident yes? If not, you have found a claim that deserves a name.*
