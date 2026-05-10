---
layout: book
title: "Chapter 7 · Names Through Differentiation"
---

# Chapter 7 · Names Through Differentiation

> "In the forward pass, you eliminate information. In the backward pass, you guess."
>
> — The author, after a long debugging session

*Combinations · Automatic differentiation and the pullback*

---

A derivative measures sensitivity. `@loss / @W` asks: if I perturb `W` by a small amount, how much does `loss` change? For scalar `loss` and scalar `W`, the answer is a single number. For tensor `W`, the answer is a tensor of the same shape as `W`—each element says how `loss` responds to perturbing that specific element.

This chapter traces gradients through matrix multiplication, convolution, and our own functions. At each step, coordinate names tell the story of what the gradient must sum over—and what happens when the names are wrong.

---

## The Shopping Cart and the Restocking Run

Before we dive into the math, an intuition model that will carry us through the entire chapter.

**Forward pass = shopping.** You walk through the aisles with a cart. From each shelf (input tensor), you take items (values) and put them in your cart. When you reach the register, some items have been consumed along the way—a reduction ate them (`sum[k]` consumed coordinate `k`). Other items were never on your list at all—broadcast items that were silently copied along a coordinate you didn't care about (`bias[j]` was silent on `i`).

The forward pass leaves a record: which coordinates were consumed, and which were copied.

**Backward pass = restocking.** The store manager takes your shopping record and works backward. For every item you consumed (every reduction), the manager must broadcast replenishment stock back to the shelf. For every item that was silently copied (every broadcast), the manager must collect all the copies and *reduce* them back to a single restocking order.

The forward record and the backward record are the same document, read in opposite directions. What was consumed becomes broadcast. What was broadcast becomes consumed. The names of the coordinates tell you which is which.

This is the pullback rule, stated as a shopping trip. Now let's see it in coordinates.

---

## The Gradient as Coordinate Subtraction

`@loss / @W` has the same shape as `W`. The set of coordinates on the result of differentiation is exactly the set of coordinates on the denominator.

But `loss` is a scalar—it has no coordinates. So the path from `W` to `loss` must eliminate every coordinate that `W` carries. The gradient computation must reconstruct those eliminations in reverse: every forward reduction becomes a backward broadcast, and every forward broadcast becomes a backward reduction.

Let's trace this for matrix multiplication.

![Forward and backward: the gradient sums over coordinates in C but not in A](figures/gradient_pullback.svg)

Forward pass:

```rust
let C[i, j] = sum[k](A[i, k] * B[k, j]);
```

`C` has coordinates `[i, j]`. `A` has `[i, k]`. `B` has `[k, j]`. The coordinate `k` is consumed by the reduction.

Now suppose we have `dC[i, j]`—the gradient of the loss with respect to `C`. We want `@C / @A`.

The result must have coordinates `[i, k]` (the coordinates of `A`). The gradient signal `dC` has coordinates `[i, j]`. We need to eliminate `j` and introduce `k`. Multiply `dC[i, j]` by the *other* input `B[k, j]`, and sum over `j`:

```rust
let dA[i, k] = sum[j](dC[i, j] * B[k, j]);
```

The sum coordinate is `j`—the coordinate that `A` does not have but `C` does. The surviving coordinates are `i` (shared between `A` and `C`) and `k` (from `A`, reintroduced by `B`).

This is the pullback rule: **the gradient sums over the set-difference of coordinates between the output and the operand.** `C` has `{i, j}`. `A` has `{i, k}`. The difference is `{j}`—sum over `j`. The missing coordinate `k` is provided by `B`.

You don't need to memorize the formula for a matmul pullback. You need the coordinate accounting. Which coordinates does the operand carry? Which does the output carry? The reduction coordinates in the gradient are exactly the coordinates present in the output but absent from the operand.

---

## The Five-Step Pullback Procedure

Given a forward expression and a target operand, derive the gradient:

1. **Hold one cell** of the target operand. Choose a specific element—say `A[i₀, k₀]`.
2. **List every output cell that reads it.** For `C[i, j] = sum[k](A[i, k] * B[k, j])`, the held cell is read by every output where `i = i₀` and the sum includes `k = k₀`. That means `C[i₀, j]` for *all* `j`.
3. **Attach the incoming gradient.** Each output cell `C[i₀, j]` carries a gradient signal `dC[i₀, j]`. The contribution from the path through `A[i₀, k₀]` is `dC[i₀, j] * B[k₀, j]`.
4. **Multiply by the local derivative.** For elementwise multiplication inside the sum, the local derivative of `A[i, k] * B[k, j]` with respect to `A[i, k]` is `B[k, j]`.
5. **Sum the routes.** The path coordinate—the coordinate in `C` but not in `A`—is `j`. Sum over it.

The result: `dA[i, k] = sum[j](dC[i, j] * B[k, j])`. No calculus memorization. No transpose rules. Just coordinate accounting.

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

The gradient sums over `b`. Why? Because in the forward pass, `bias[out]` was replicated across every batch element. Every batch element carries a piece of the gradient signal. To update `bias`, we must collect all those pieces. The omitted coordinate becomes the reduced coordinate in the backward pass.

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

---

The gradient is not a separate computation from the forward pass. It is the forward pass, read backward. The coordinate names that organize the forward pass—which survive, which are consumed, which are omitted—organize the backward pass in exactly the same way. A coordinate eliminated by a forward sum becomes a coordinate introduced by a backward broadcast. A coordinate omitted by a forward broadcast becomes a coordinate summed by a backward reduction.

The names are the same. The direction is reversed. The principle is symmetric.

The shopping cart record, read forward, tells you what you bought. Read backward, it tells the store what to restock. The names of the items are on both sides of the receipt.

In the next chapter, we leave the single-language perspective and do something new. We put einlang side by side with PyTorch—same computation, two notations—and ask what the names let you see that positions hide.
