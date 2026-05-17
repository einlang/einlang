---
layout: book
title: "Chapter 8 · Names Through Differentiation"
---

# Chapter 8 · Names Through Differentiation

> "In the forward pass, you eliminate information. In the backward pass, you guess."

*Combinations · Automatic differentiation and the pullback*

---

The last chapter ended with index arithmetic — `oh + kh` in the forward pass became `ih - kh` in the backward pass. That inversion was not a coincidence. It was the Inversion Rule, applied to coordinates that carry arithmetic.

You've been computing gradients all week. `loss.backward()` handles them. You don't think about them. Then you write a custom backward pass — a `torch.autograd.Function` or a manual gradient check — and suddenly you're staring at a sum over the wrong axis, wondering which coordinate you missed.

The autograd engine computes gradients by tracing the forward pass and inverting each operation. When you write the backward pass by hand, you are the engine. And the engine's hardest question is: *over which coordinates do I sum?*

The answer is always the same. It's the Inversion Rule from Chapter 2, applied to every operation in the forward graph: forward broadcast becomes backward reduction, forward reduction becomes backward broadcast. This chapter shows that the five-step pullback — the procedure for deriving any gradient by hand — is coordinate set subtraction, applied in reverse.

---

## The Shopping Cart and the Restocking Run

An intuition model:

You are writing the backward pass for a custom layer. The forward code is in front of you:

```rust
let C[i, j] = sum[k](A[i, k] * B[k, j]);
```

You have `dC[i, j]`—the gradient of the loss with respect to `C`. You need `dA[i, k]`. You don't want to look up the formula. You don't want to memorize the transpose rule. You just want to derive it from what the forward code says.

Start from first principles. `A[i, k]` appears exactly where? Inside the `sum[k]`: it is multiplied by `B[k, j]` and then summed over `k`. For a specific element `A[i0, k0]`, which output cells does it contribute to? Every output cell where `i = i0` AND the sum includes `k = k0`. That means `C[i0, j]` for *all* `j`.

So `A[i0, k0]` sends a contribution `A[i0, k0] * B[k0, j]` to `C[i0, j]`. The gradient signal `dC[i0, j]` must flow back along that same path. The local derivative of `A[i, k] * B[k, j]` with respect to `A[i, k]` is `B[k, j]`. So the contribution from output cell `C[i0, j]` back to `A[i0, k0]` is `dC[i0, j] * B[k0, j]`.

Now sum over all the output cells that received a contribution. Those are all `j` positions. The coordinate `j` is in `C` but NOT in `A`. Sum over it:

```rust
let dA[i, k] = sum[j](dC[i, j] * B[k, j]);
```

You just derived the matmul pullback. You didn't memorize it. You didn't look it up. You did coordinate accounting: which coordinates does the output have that the operand doesn't? Sum over those.

What just happened: tracing which coordinates were "paths" from `A` to `C`. The coordinate `j` was a path—every output position along `j` received input from `A[i0, k0]`. To send the gradient back, you had to sum over that path. The coordinate `k` was consumed by the forward sum—it existed in `A` and was eliminated. The backward pass doesn't need to sum over `k` because `A[i, k]` only contributed to `C[i, j]` through the specific `k` value in the sum. Each `k` position's gradient is independent.

Here is the pattern. In the forward pass, some coordinates are *consumed*—a reduction (`sum[k]`) eliminates them. Some coordinates are *silent*—a broadcast copies a value along them without the value depending on them. In the backward pass, every consumption becomes a broadcast (the gradient must be spread back over what was consumed) and every silence becomes a reduction (all the copies must be collected).

The forward pass is shopping; the backward pass is restocking the same receipt in reverse.

---

## The Gradient as Coordinate Subtraction

The general rule:

`@loss / @W` has the same shape as `W`. The coordinates on the gradient result are exactly the coordinates on the denominator.

Apply this to matrix multiplication. Here is `C[i,j] = sum[k](A[i,k] * B[k,j])` and its gradient, forward and backward, side by side:

![Forward and backward: the gradient sums over coordinates in C but not in A](figures/gradient_pullback.svg)

On the left, `sum[k]` consumes `k` in the forward pass. On the right, the gradient `dA[i,k]` sums over `j` — the coordinate in `C` that is not in `A`. The figure reads the same in both directions: forward consumption becomes backward broadcast, forward broadcast becomes backward reduction.

The pullback rule in one sentence: **the gradient sums over the set-difference of coordinates between the output and the operand.** Output has `{i, j}`. Operand `A` has `{i, k}`. Difference: `{j}`. Sum over `j`. The missing coordinate `k` is provided by the other operand `B[k, j]`.

Coordinate accounting. No transpose rules. No memorization.

---

## The Five-Step Pullback Procedure

Before reading the procedure, try it yourself. Forward: `C[i, j] = sum[k](A[i, k] * B[k, j])`. You have `dC[i, j]` — the gradient of the loss with respect to `C`. You want `dA[i, k]`. Which output cells does `A[i0, k0]` contribute to? What's the local derivative? Which coordinate must you sum over?

Given a forward expression and a target operand, derive the gradient:

1. **Hold one cell** of the target operand. Choose a specific element—say `A[i0, k0]`.
2. **List every output cell that reads it.** For `C[i, j] = sum[k](A[i, k] * B[k, j])`, the held cell is read by every output where `i = i0` and the sum includes `k = k0`. That means `C[i0, j]` for *all* `j`.
3. **Attach the incoming gradient.** Each output cell `C[i0, j]` carries a gradient signal `dC[i0, j]`. The contribution from the path through `A[i0, k0]` is `dC[i0, j] * B[k0, j]`.
4. **Multiply by the local derivative.** For elementwise multiplication inside the sum, the local derivative of `A[i, k] * B[k, j]` with respect to `A[i, k]` is `B[k, j]`.
5. **Sum the routes.** The path coordinate—the coordinate in `C` but not in `A`—is `j`. Sum over it.

> **Path coordinate.** A coordinate is a *path coordinate* for an operand if it appears in the output of a forward computation but does not appear in that operand's index pattern. The gradient with respect to that operand sums over all of its path coordinates. Formally: `paths(operand, output) = {coordinates in output} \ {coordinates in operand}`.

The result: `dA[i, k] = sum[j](dC[i, j] * B[k, j])`. No calculus memorization. No transpose rules. Just coordinate accounting.

---

### The Pullback in One Example

Observe the five steps on a broadcast-add. Forward:

```rust
let out[i, j] = A[i, j] + bias[j];
```

Given `d_out[i, j]`, find `d_bias[j]`. What coordinates does the output have? `{i, j}`. What coordinates does `bias` have? `{j}`. The path coordinate (in output but not in bias) is `{i}`. Sum over `{i}`.

1. Hold one cell: `bias[j0]`.
2. Every output cell reads it: `out[i, j0]` for *all* `i`. The held `j0` value is copied to every `i` position.
3. Attach the incoming gradient: each output cell carries `d_out[i, j0]`. The contribution from the path through `bias[j0]` is `d_out[i, j0] * 1` (the local derivative of `x + bias` wrt `bias` is 1).
4. Local derivative: 1.
5. Sum over the path coordinates: output has `{i, j}`, bias has `{j}`. Difference: `{i}`. Sum over `i`.

Result: `d_bias[j] = sum[i](d_out[i, j])`. The broadcast coordinate `i` becomes the reduction coordinate. The Inversion Rule, mechanically applied.

Verify with coordinate set subtraction alone. Forward: `out[i, j] = A[i, j] + bias[j]`. `out` has `{i, j}`, `bias` has `{j}`. Set difference: `{i}`. Sum over `{i}`. `d_bias[j] = sum[i](d_out[i, j])`.

The pattern: the gradient sums over whatever is in the output but not in the operand. Five steps. No calculus memorization. The coordinate sets tell you what to sum over. The forward expression tells you what to multiply by.

---

## Convolution Gradients

A convolution is a sum of products with index arithmetic:

```rust
let conv[b, oc, oh, ow] = sum[ic, kh, kw](
    input[b, ic, oh + kh, ow + kw] * weight[oc, ic, kh, kw]
);
```

The gradient with respect to `weight` sums over everything that `weight` does not own:

```rust
let dW[oc, ic, kh, kw] = sum[b, oh, ow](
    dConv[b, oc, oh, ow] * input[b, ic, oh + kh, ow + kw]
);
```

The coordinates `b`, `oh`, `ow` are summed away because they appear in the output but not in `weight`. The coordinates `oc`, `ic`, `kh`, `kw` survive because they *are* `weight`'s coordinates.

Again: set subtraction. The formula is mechanically derivable from the coordinate sets. The same five steps produce the weight gradient with the correct index arithmetic.

**Derive it yourself: The weight gradient.** The linear layer is `z[b, out] = sum[in](x[b, in] * W[out, in]) + bias[out]`. You derived `d_bias[j]` above by coordinate set subtraction. Now derive `dW[out, in]`.

Apply the five steps. The output `z[b, out]` has coordinates `{b, out}`. The operand `W[out, in]` has coordinates `{out, in}`. Path coordinates in the output but not in `W`: `{b}`. But `W` also has `in`, which the output does not have—because `in` was consumed by the forward `sum[in]`. Consumed coordinates survive in the weight gradient. The path coordinates to sum over are `{b}`—the coordinate `W` was silent on in the forward pass.

Before reading further, write the answer. Then verify: does your `dW` have coordinates `{out, in}`? Does the sum go over `{b}`?

```rust
dW[out, in] = sum[b](dZ[b, out] * x[b, in])
```

The sum is over `b`—the coordinate `W` broadcasts over. `in` survives because each `in` position gets an independent gradient contribution through the forward sum. `out` survives because it is shared with the output. Two coordinates survive. One is summed away. The five steps, mechanically applied. You didn't memorize the formula. You read it off the coordinate sets.

---

## The Broadcast Self-Audit

A forward broadcast makes a promise. A backward gradient collects dependence. If the two disagree, the gradient is silently wrong.

Consider a linear layer with a bias:

```rust
let z[b, out] = sum[in](x[b, in] * W[out, in]) + bias[out];
```

The bias term omits `b`—it promises that the bias does not depend on the batch element. Now compute the gradient:

```rust
let d_bias[out] = sum[b](d_loss[b, out]);
```

The gradient sums over `b`. Why? Because in the forward pass, `bias[out]` was replicated across every batch element. Every batch element carries a piece of the gradient signal. To update `bias`, collect all those pieces. The omitted coordinate becomes the reduced coordinate in the backward pass.

Three questions for any broadcast:

1. **What coordinate am I broadcasting over?** Is the name visible in the code, or is it inferred from position?
2. **Is independence truly justified?** Does the broadcast value genuinely not depend on that coordinate?
3. **What will the gradient do?** Does the backward reduction produce the right shape for the parameter update?

These three questions are the broadcast self-audit. They catch the class of bugs where a broadcast is shape-correct but semantically wrong.

---

## Custom Differentiation with `@fn`

Some functions have derivatives that are better expressed directly:

```rust
fn relu(x) { if x > 0.0 { x } else { 0.0 } }

@fn relu(x) {
    if x > 0.0 { @x } else { 0.0 }
}
```

The `@fn` declaration shares the function's name and parameter list. Inside the body, `@x` refers to the tangent flowing through parameter `x`. Custom rules can be coordinate-aware:

```rust
@fn softmax[j](x: [f32; ..left, j, ..right]) {
    softmax_tangent[j](x, @x)
}
```

The coordinate parameter `j` appears in both the primal function and its derivative rule. The tangent computation follows the same coordinate contract as the primal.

---

## Where Clauses in the Backward Pass

A where clause acts as a gate in both directions:

```rust
let pos_sum = sum[i](data[i]) where data[i] > 0;
```

In the forward pass, only positive elements are summed. In the backward pass, the gradient signal is distributed only to the positive elements. Elements that were filtered out receive zero gradient. You don't write a separate backward filter. The where clause defines the domain of the operation, and the domain applies in both directions.

The where clause is the gate. Items that pass the condition in the forward pass receive gradient in the backward pass. Items that don't, don't. The condition is written once.

The most common where clause in practice: masked softmax. You have a sequence with padding. The logits for padding positions should be ignored—both in the forward pass (they should not contribute to the sum) and in the backward pass (they should receive zero gradient):

```rust
let probs[b, j] = softmax_over_valid[class](logits[b, class])
    where valid[b, class];
```

The `valid[b, class]` tensor is a boolean mask—`true` for real tokens, `false` for padding. The where clause gates every operation inside `softmax_over_valid`: the max reduction, the subtraction, the exponentiation, the sum reduction, and the division. Five operations. One gate.

In PyTorch, you'd write a custom masked softmax, or add `float('-inf')` to masked positions before the softmax, or use `torch.where` after the softmax to zero out padding. Each approach has its own backward behavior. The `-inf` trick works for softmax but not for mean. The `torch.where` approach leaves non-zero gradients on padding positions (they were computed, then zeroed—waste). The custom function is correct but requires writing a custom backward pass.

The where clause avoids all three. The gate is part of the operation's domain. The compiler reads it once, applies it to every operation inside the reduction, and generates both the forward gating and the backward gating. No custom backward code. No `-inf` hack. No wasted computation. The gate is written where it belongs—in the domain specification of the operation it gates.

Now a harder one. What if the where clause references a coordinate that is consumed by the operation?

```rust
let top_sum[b] = sum[class](logits[b, class]) where logits[b, class] > threshold[b];
```

The where clause references `class`—the coordinate being consumed by `sum`. The condition `logits[b, class] > threshold[b]` is a tensor of shape `(batch, class)`. Each `(b, class)` position has a boolean gate.

In the backward pass, the gradient signal `d_top_sum[b]` is distributed back to `logits[b, class]`—but only to positions where the gate was open. For each `b`, elements of `class` that were below `threshold[b]` receive zero gradient. The where clause is evaluated during the forward pass and its boolean mask is stored. The backward pass reads the mask and gates the gradient accordingly.

No separate backward code. No custom `@fn` rule. The where clause is part of the reduction's domain specification, and the compiler generates both the forward gating and the backward gating from it. The coordinate names in the condition—`class` in this case—tell the compiler which dimension the gate applies to.

---

## LayerNorm: A Complete Gradient Walkthrough

Tracing the gradient of LayerNorm end to end:

Forward pass:

```rust
fn layer_norm[feature](x: [f32; ..b, feature],
                      gamma: [f32; feature],
                      beta: [f32; feature])
    -> [f32; ..b, feature]
{
    let mean_val[..b] = mean[feature](x[..b, feature]);
    let centered[..b, feature] = x[..b, feature] - mean_val[..b];
    let var[..b] = mean[feature](centered[..b, feature] ** 2.0);
    (centered[..b, feature] / (var[..b] ** 0.5 + 1e-5)) * gamma[feature] + beta[feature]
}
```

`d_out[..b, feature]` is the gradient of the loss with respect to the LayerNorm output. Compute `dx[..b, feature]`, `d_gamma[feature]`, and `d_beta[feature]`.

**Step 1: Gradient of beta.** The output expression ends with `+ beta[feature]`. This is an elementwise addition. `beta` omits `..b` in its index pattern—it broadcasts over all batch dimensions. By the Inversion Rule, the gradient sums over the broadcast set:

```rust
let d_beta[feature] = sum[..b](d_out[..b, feature]);
```

`d_out` has `{..b, feature}`. `beta` has `{feature}`. Difference: `{..b}`. Sum over it. Result: `{feature}`, matching `beta`'s shape.

**Step 2: Gradient of gamma.** Same pattern. `gamma[feature]` broadcasts over `..b`. The gradient sums over `..b`:

```rust
let d_gamma[feature] = sum[..b](
    d_out[..b, feature] * centered[..b, feature] / (var[..b] ** 0.5 + 1e-5)
);
```

The local derivative of `gamma * x_norm` with respect to `gamma` is `x_norm`. Multiply by the incoming gradient. Sum over the broadcast set. Done.

**Step 3: Gradient of x.** This is the complex one—`x` contributes to the output through three paths: directly through `centered` (which contains `x`), and indirectly through `mean_val` and `var` (which were computed from `x`). But the coordinate accounting still works: `x` has `{..b, feature}`. The output has `{..b, feature}`. No coordinate difference. The gradient must have the same shape. No sum—the contributions from all three paths accumulate at each `(..b, feature)` position.

Coordinate set subtraction tells you *which* coordinates get summed. The local computation tells you *what* to sum. Together they produce the gradient without shape memorization.

---

## The Recurrence Gradient: Time as a Path Coordinate

Every gradient so far summed over spatial coordinates — `j` in matmul, `b` in bias, `oh, ow` in convolution. Recurrence introduces a new question: what happens when the forward pass steps through time?

Consider the simplest recurrence — a scalar accumulator:

```rust
let h[0] = x;
let h[t in 1..T] = h[t-1] + 1.0;
```

Before reading the answer, derive the gradient yourself. Forward: `h[0] = x`, `h[1] = h[0] + 1`, `h[2] = h[1] + 1`, `h[3] = h[2] + 1`. Given `dh[t]` (the gradient of the loss with respect to each `h[t]`), what is `dh[0]`? What is `dx`?

Trace the dependence. `h[0]` contributes to `h[1]`. `h[1]` contributes to `h[2]`. `h[2]` contributes to `h[3]`. The gradient signal must flow backward along every link in this chain.

Take a minute. Draw the dependence graph. Then read on.

---

`h[3]` depends on `h[2]`. By the chain rule, `dh[2]` receives `dh[3] * 1` (the local derivative of `h[t-1] + 1` with respect to `h[t-1]` is 1). But `h[2]` may also receive gradient directly if the loss reads `h[2]`. So `dh[2] = dh[2]_direct + dh[3] * 1`.

`h[2]` contributes to `h[3]`. `h[1]` contributes to `h[2]`. All the way back to `h[0]`. The gradient at step `t` receives contributions from two sources: the direct loss signal at step `t`, and the backpropagated signal from step `t+1`. The second source chains backward through every subsequent time step.

Unrolled for `T=4`:

```
dh[3] = dh[3]_direct
dh[2] = dh[2]_direct + dh[3]
dh[1] = dh[1]_direct + dh[2]
dh[0] = dh[0]_direct + dh[1]
dx    = dh[0]
```

Now a recurrence with a parameter — the kind that appears in every optimizer and RNN:

```rust
let h[0] = x;
let h[t in 1..T] = h[t-1] * W + b;
```

Given `dh[t]` for all `t`, find `dW`. Apply the five-step pullback.

**Step 1: Hold one cell** of `W`. `W` is a scalar (no coordinates). The held cell is `W`.

**Step 2: List every output cell that reads it.** The output has `{t}`. The parameter `W` appears at every time step. `W` is read by every `h[t]` for `t in 1..T` — the recurrence body multiplies `h[t-1]` by `W` at each step.

**Step 3: Attach the incoming gradient.** At step `t`, the contribution through `W` to `h[t]` is `dh[t] * h[t-1]` (local derivative of `h[t-1] * W` with respect to `W` is `h[t-1]`).

**Step 4: Multiply by the local derivative.** `h[t-1]` at each step.

**Step 5: Sum over the path coordinates.** The output has `{t}`. The operand `W` has `{}` (scalar). Path coordinate: `{t}`. Sum over `t`.

```rust
let dW = sum[t in 1..T](dh[t] * h[t-1]);
```

The coordinate set subtraction works exactly as before. `t` is the path coordinate — it appears in the output `h[t]` but not in the parameter `W`. Sum over `t`. The recurrence unrolling is the mechanism that makes the sum possible; the coordinate accounting tells you *what* to sum over.

The same pattern applies to `db`:

```rust
let db = sum[t in 1..T](dh[t] * 1.0);
```

And to `dx` (the gradient of the initial state). `x` has no coordinates where `h` has `{t}`. The path coordinate is `{t}`. But `x` only appears at `h[0]` — not at every `t`. The sum over `t` collapses to a single term: `dh[0]`, which incorporates the backpropagated signal from all future steps through the chain rule.

Now the general form. Forward:

```rust
let h[t in 0..T, i] = initial[i];
let h[t in 1..T, i] = f(h[t-1, i], W, i);
```

The parameter `W` omits `t`. The gradient sums over `t`:

```rust
let dW = sum[t in 1..T, i](dh[t, i] * ∂f/∂W);
```

The recurrence dimension `t` becomes a path coordinate — summed over in the gradient, exactly like `j` in matmul or `b` in bias. The unrolling is the mechanism. The coordinate set subtraction picks the summation axes. No new rules. No special BPTT formula to memorize. `t` is just another coordinate that the output has and the parameter doesn't.

**The recurrence backward pass is coordinate set subtraction where one of the coordinates happens to carry a dependence chain.** The chain rule unrolls the dependence. The coordinate sets determine the summation. Two mechanisms. One derivation.

This is why the compiler's recurrence analysis from Chapter 6 matters for differentiation. When the compiler detects `history_lookback_steps = 1` and `downstream_tail_steps = 1`, it allocates a rolling buffer of size 2. The same analysis tells the backward pass how many steps to unroll and which coordinates must be summed. The structural fact (`t-1` in the forward pass) determines both the memory strategy AND the gradient summation. One minus sign. Two compiler passes. Same coordinate name.

---

### Be the Compiler: Derive the RNN Gradient

An RNN cell with a hidden state and an output projection:

```rust
let h[t in 0..T, hidden] = initial[hidden];
let h[t in 1..T, hidden] = tanh(
    h[t-1, hidden] * W_h[hidden, hidden] + x[t, in] * W_x[in, hidden] + b[hidden]
);
let y[t, out] = h[t, hidden] * W_y[hidden, out];
```

Three parameter gradients to derive: `dW_h`, `dW_x`, `dW_y`, `db`. For each one, ask: what coordinate does the parameter omit that the output has? That coordinate is the path coordinate — sum over it.

Before reading the answer, write each gradient. Use the five steps. The coordinate sets tell you what to sum. The forward expression tells you what to multiply.

---

The pattern from recurrent gradients is the same pattern from every pullback: hold one cell, trace its contribution to every output cell, sum over the path coordinates. The recurrence adds a dependence chain through time. The coordinate accounting does not change. `t` is a coordinate. The output has it. The parameter doesn't. Sum over `t`.

The pullback procedure was always coordinate set subtraction. Recurrence proved it survives the hardest test.

---

## Fancy Indexing: Where the Gradient Scatters

Chapter 7 distinguished pairwise indexing from outer-product indexing by coordinate name: `k` in both positions means pairwise; `i` and `j` different means outer-product. The gradient behavior differs in exactly the way the names predict.

**Pairwise indexing.** Forward:

```rust
let gathered[k] = matrix[idx[k], col_idx[k]];
```

The output has `{k}`. The operand `matrix` has `{i, j}`. Path coordinate: `{k}`. The gradient must sum over `k` and restore the contribution to the `{i, j}` shape of `matrix`. But each `k` contributes to exactly one `(i, j)` position — `(idx[k], col_idx[k])`. The sum over `k` is a **scatter-add**: accumulate `d_gathered[k]` into `d_matrix[idx[k], col_idx[k]]` for each `k`.

No dense sum. No broadcast. A sparse scatter, indexed by the same index arrays as the forward pass, accumulating at the gathered positions. The coordinate name `k` tells you the path coordinate. The index arrays tell you where each `k` maps.

**Outer-product indexing.** Forward:

```rust
let outer[i, j] = matrix[idx[i], col_idx[j]];
```

The output has `{i, j}`. The operand `matrix` has the same `{i, j}` in its index pattern (via `idx` and `col_idx`). No coordinate to sum — the gradient flows directly: `d_matrix[idx[i], col_idx[j]] += d_outer[i, j]`.

But here is the difference from pairwise: if `idx[i] = 3` and `col_idx[j] = 3` for multiple `(i, j)` pairs, the gradient at `matrix[3, 3]` accumulates contributions from every such pair. The outer-product scatter can have collisions. The pairwise scatter cannot — each `k` maps to a distinct `(idx[k], col_idx[k])` by construction.

The coordinate names distinguish the two indexing modes. The same names determine the gradient accumulation pattern. Pairwise: sum over `k`, no collisions. Outer-product: no sum, collisions possible. One rule. Two behaviors. The names tell you which.

The compiler does not have a special scatter-add pass for indexing gradients. It inlines the gather body into Einstein notation (`result[i,j] = data[indices[i], j]`) and differentiates through the `RectangularAccessIR` node: the differential of the array is wrapped in the same index access pattern. This general autodiff produces the correct gradient — `d_data[indices[i], j] += d_result[i, j]` — without recognizing it as a scatter. The coordinate accounting is a conceptual framework that predicts what the autodiff will produce. The autodiff is the mechanism that produces it.

---

## Where Set Subtraction Stops

The coordinate set subtraction rule — sum over `{output coords} \ {operand coords}` — derives the gradient for most tensor operations: reductions, broadcasts, matmul, convolutions, recurrences, where clauses. Three operations require a different treatment.

**Non-differentiable operations.** `argmax[j](x[i, j])` returns an index, not a continuous value. The output is an integer coordinate position. Perturbing `x` by ε does not change the argmax except at boundaries where two values are exactly equal. The gradient is zero almost everywhere, undefined at ties. The autodiff pass has no branch for `ReductionOp.ARGMAX` — the operation is not differentiable. The compiler leaves a zero gradient, but this is a silent default, not a checked error. A correct compiler should report `error: gradient of argmax is not defined` at the point where `@loss / @argmax_result` is written. Downstream code that needs a gradient through a selection must use a soft approximation (softmax with temperature) or a straight-through estimator wrapped in `@fn`. The `max` reduction *is* differentiable — the compiler uses `SelectAtArgmaxIR` to route the cotangent to the winning index. `argmax` returning an integer index is not. The distinction between `max` (differentiable, gradient routes to argmax position) and `argmax` (non-differentiable, returns integer) is a compiler-semantic distinction, not a syntactic one.

**`prod`.** `result = prod[i](data[i])`. The gradient of `data[k]` is `d_result * result / data[k]` — the product of all elements EXCEPT `data[k]`. This does not fit the pure coordinate-set-subtraction pattern because the local derivative of `data[k]` depends on every other element's value. The compiler handles `prod` with a built-in gradient rule: the autodiff pass recognizes `ReductionOp.PROD` and emits `sum((total_product / body) * d_body)`. Unlike `sum` (which distributes the tangent as-is) or `max` (which routes to the winning index via `SelectAtArgmaxIR`), `prod` requires the full product to compute the local derivative. The coordinate accounting still works — `i` is the consumed coordinate, the gradient restores it — but the *value* of the local derivative requires knowing the total product, not just the coordinate structure.

**Reshape and permute.** These operations change coordinate layout without changing values. The gradient is the inverse operation: a forward `permute` of `(batch, feature)` to `(feature, batch)` has a backward `permute` of `(feature, batch)` to `(batch, feature)`. The coordinate accounting is trivial — no sum, no path coordinate — because every output cell reads exactly one input cell. The gradient flows through the inverse layout change.

For everything else — reductions, broadcasts, matmuls, convolutions, recurrences, where clauses, pack polymorphism, cumulative scans, normalization, attention — the five-step pullback is coordinate set subtraction. The names tell you what to sum. The forward expression tells you what to multiply. The rest is accounting.

---

## The Pullback as Coordinate Set Subtraction: A Summary

Every pullback you have seen follows the same rule. Given a forward expression and the operand whose gradient you want:

1. Write the operand's coordinate set. `A` has `{i, k}`.
2. Write the output's coordinate set. `C` has `{i, j}`.
3. The path coordinates = output set minus operand set. `{j}`.
4. Sum over the path coordinates. `dA[i, k] = sum[j](...)`.

The other operand (`B[k, j]`) provides the missing coordinate `k`. The local derivative comes from differentiating the forward operation with respect to the operand. The path-sum structure comes from the coordinate sets.

This is the pullback rule as coordinate accounting. No transpose rules memorized. No Jacobian dimensions counted. Just set subtraction, applied to coordinate names.

In a single equation—carry this with you:

```
dA = Σ_{paths(A, C)}  dC · d(forward)/dA

where paths(A, C) = {coordinates in C} \ {coordinates in A}
```

`paths(A, C)` is the set of coordinates in the output `C` that are NOT in the operand `A`. Sum over them. Multiply by the local derivative. Done. It is the formula behind every gradient derived from a tensor expression. The rest is accounting.

---

The gradient is not a separate computation from the forward pass. It is the forward pass, read backward. The coordinate names that organize the forward pass—which survive, which are consumed, which are omitted—organize the backward pass in exactly the same way. A coordinate eliminated by a forward sum becomes a coordinate introduced by a backward broadcast. A coordinate omitted by a forward broadcast becomes a coordinate summed by a backward reduction.

The names are the same. The direction is reversed. The principle is symmetric.

---

## The Gradient of a Coordinate-Aware Function Call

When a function call appears in the forward pass, its backward pass is determined by the function's coordinate contract. The gradient does not need to re-verify the contract—the forward call already did. The gradient only needs to apply the Inversion Rule to each primitive operation inside the function body.

Consider `softmax[class](logits)`. Forward: the function body contains `max[class]`, `sum[class]`, elementwise operations, and a division. Backward: `max[class]` becomes a broadcast (only the argmax position receives the gradient), `sum[class]` becomes a broadcast (every summed element receives the gradient), the division's gradient is elementwise.

The coordinate `class` is the reduction axis in both `max` and `sum`. The backward pass broadcasts over `class` for both. The gradient flows through the function's coordinate parameter—`class` in, `class` out. The caller never sees the internal backward operations. The coordinate contract at the call site guarantees that `class` exists on the input, so the gradient exists on the same coordinate.

A custom `@fn` rule for softmax overrides this automatic derivation—but the coordinate parameter in the custom rule must match the primal. The compiler checks this:

```rust
fn softmax[j](x: [f32; ..left, j, ..right]) -> [f32; ..left, j, ..right] { ... }

@fn softmax[j](x: [f32; ..left, j, ..right]) {
    // Custom backward: softmax_tangent[j](x, @x)
    // j must match the primal's coordinate parameter
}
```

If the custom rule declares `@fn softmax[k]` (different coordinate parameter name), the compiler reports a mismatch. If the custom rule returns a gradient with different coordinates than the primal's input, the compiler reports a mismatch. The coordinate contract extends from the forward pass into the backward pass. The same names, checked in both directions.


---

*Revisit the last gradient you debugged. Write the forward expression with coordinate names. Derive the backward expression using coordinate set subtraction. Does the backward sum match the forward broadcast? The answer to that question is the answer to whether your gradient was correct.*

---

### The Pullback in Practice

The five-step pullback procedure applies to matmul, broadcast-add, and sum-to-scalar. It also applies to convolution, normalization, and attention. The procedure is always the same: hold one cell, list every output cell that reads it, attach the incoming gradient, multiply by the local derivative, sum over the path coordinates.

A gradient written or debugged by hand—a custom backward pass, a `torch.autograd.Function`, a manual gradient check—is a forward expression read backward. Writing the forward expression with coordinate names, even if the original code is in PyTorch, makes the backward derivation mechanical. What coordinates does the output have? What coordinates does the operand have? What's the difference? Sum over the difference. No insight required. Coordinate set subtraction, applied to the forward expression, produces the backward sum.

When the derived gradient matches the original, the coordinate accounting confirms the manual derivation—both arrived at the same result through different paths. When they differ, either the coordinate accounting was wrong, or the original gradient was. Both possibilities are worth checking. The same procedure works for convolution, normalization, and attention. The coordinate sets are larger. The procedure doesn't change.

The pullback is not a separate computation from the forward pass. It is the forward pass, read backward, through the lens of the Inversion Rule. Every forward reduction becomes a backward broadcast. Every forward broadcast becomes a backward reduction. The coordinate names are the bridge between the two directions. The five steps are the procedure for crossing it.

---

## Return to the Transformer

Eight chapters ago, the transformer block was a teaser. You could point to the attention and say "that sums over the sequence." You could point to the MLP and say "that projects the features." But you could not audit a single bracket. Now return to it and read it with the eyes you have built.

You can audit every broadcast in the attention block. Chapter 4 gave you the three-question self-audit: which coordinate does this broadcast copy over, is independence justified, what will the gradient do? `V[head, seq_k, d]` omits `seq_q` — the value does not depend on which query position is asking. Is that claim correct? Only if the value projections are shared across query positions — which they are, by design. The bracket makes the design visible. `weights[head, seq_q, seq_k]` omits `d` — the attention pattern is the same for every channel. Is that claim correct? Only if the attention is not channel-specific. The bracket makes the question askable. The answers are not trivial. They are the architectural decisions of the transformer, surfaced in the notation.

You can extract its skeleton. Chapter 5 showed that four normalization variants share one structural core — just as masked self-attention, cross-attention, and multi-query attention share the same attention core. The difference between attending to yourself and attending to another sequence is a coordinate name: `seq_q == seq_k` or `seq_q != seq_k`. The code change is a name swap. The skeleton stays the same.

You can verify its causality. Chapter 6 taught you that `t-1` in a bracket tells the compiler that time flows forward. In decoder self-attention, the constraint `seq_k <= seq_q` is a statement about the coordinate domain — not a separate mask tensor you create and multiply. The compiler can check it. The minus sign that organized recurrence organizes causality here as well.

You can handle its head-splitting coordinate arithmetic. Chapter 7 worked through the split of one coordinate into two roles: `d` becomes `head` and `d_head`. The query, key, and value projections each split differently but compatibly. The names record which part is which. If the split is inconsistent across projections, the compiler reports a shape mismatch.

You can derive its backward pass through set subtraction. Chapter 8 gave you the five-step pullback, and the procedure works on attention exactly as it works on matmul. The forward sum over `seq_k` becomes a backward broadcast over `seq_k` for the query gradient. The forward broadcast over `seq_q` on `V` becomes a backward sum over `seq_q` for the value gradient. The Inversion Rule, applied to the attention block. Not a formula you memorized. A consequence of the coordinate sets.

The block hasn't changed. But you can now read every bracket in it, verify every broadcast, extract its structural skeleton, check its causality constraint, handle its coordinate arithmetic, and derive its backward pass. It was a teaser in Chapter 1. It is a test now — and you can pass it.

---

Part II began with a question: when operations compose, does the coordinate story survive? You tested that question against every operation in tensor programming. Functions (Chapter 3): yes, the contract is checked at the call site. Broadcasts (Chapter 4): yes, the three-question audit reveals unwarranted independence claims. Normalization (Chapter 5): yes, four variants share one skeleton, and the coordinate name absorbs all layout changes. Recurrence (Chapter 6): yes, the compiler detects the direction of time from a minus sign. Complex terrain (Chapter 7): yes, the split of one coordinate into two roles is the operation, and the names record it. Differentiation (this chapter): yes, the gradient is coordinate set subtraction, read in reverse — whether the forward pass is a matmul, a convolution, a recurrence, or an RNN.

The five-step pullback is coordinate accounting done by hand. The question that opens Part III: can a machine do it? Chapter 9 builds a compiler frontend that reads names from source, checks them against five rules, and lowers them to integers — all without ever running the program. The answer turns out to be the same question you have been asking since Chapter 2: *which coordinates survive?*
