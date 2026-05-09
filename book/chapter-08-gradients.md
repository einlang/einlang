---
layout: book
title: "Chapter 8 · The Challenge of Walking Backward"
---

# Chapter 8 · The Challenge of Walking Backward

> "In the forward pass, you eliminate information. In the backward pass, you guess."
>
> — The author, after a long debugging session

*Combinations, mirrored · Automatic differentiation*

---

Chapter 5 introduced `@y / @x` as a diagnostic tool for broadcast contracts. We used it in the simplest possible setting—scalar derivatives—to verify that our forward broadcasts produced the expected backward reductions.

This chapter goes deeper. We trace gradients through matrix multiplication, through convolution, through functions we define ourselves. At each step, the coordinate names tell the story of what the gradient must sum over—and what happens when the names are wrong.

---

## The Gradient as Coordinate Subtraction

A derivative measures sensitivity. `@loss / @W` asks: if I perturb `W` by a small amount, how much does `loss` change? For scalar `loss` and scalar `W`, the answer is a single number. For tensor `W`, the answer is a tensor of the same shape as `W`—each element says how `loss` responds to perturbing that specific element.

This means the shape of `@loss / @W` is the shape of `W`. The set of coordinates on the result of differentiation is exactly the set of coordinates on the denominator.

But `loss` is a scalar—it has no coordinates. So the path from `W` to `loss` must eliminate every coordinate that `W` carries. The gradient computation must reconstruct those eliminations in reverse: every forward reduction becomes a backward broadcast, and every forward broadcast becomes a backward reduction.

Let's trace this for matrix multiplication.

---

![Forward and backward: the gradient sums over coordinates in C but not in A](figures/gradient_pullback.svg)

The figure traces the full round trip. Top half (Forward): matrices $\mathbf{A}$ and $\mathbf{B}$ feed into $\mathbf{C}$ through a `sum[k]` reduction. The coordinate sets are annotated: $\mathbf{A} : \{i,k\}$, $\mathbf{B} : \{k,j\}$, $\mathbf{C} : \{i,j\}$. The divider marks the reversal. Bottom half (Backward): the incoming gradient $\partial\mathbf{C}$ (with coordinates $\{i,j\}$) and the operand $\mathbf{B}$ (with $\{k,j\}$) feed into $\partial\mathbf{A}$ (with $\{i,k\}$). The set subtraction rule is written out: $\mathbf{C}\{i,j\} \setminus \mathbf{A}\{i,k\} = \{j\}$. The coordinate $j$—present in $\mathbf{C}$ but absent from $\mathbf{A}$—is the one summed over. The coordinate $k$—absent from $\mathbf{C}$ but present in $\mathbf{A}$—is reintroduced by $\mathbf{B}$. This is not a special case for matrix multiplication. It is the universal pullback rule for sum-of-products: sum over what the output has that the operand lacks.

## The Matmul Pullback

Forward pass:

```rust
let C[i, j] = sum[k](A[i, k] * B[k, j]);
```

`C` has coordinates `[i, j]`. `A` has `[i, k]`. `B` has `[k, j]`. The coordinate `k` is consumed by the reduction.

Now suppose we have `dC[i, j]`—the gradient of the loss with respect to `C`. We want `@C / @A` and `@C / @B`.

For `@C / @A`: the result must have coordinates `[i, k]` (the coordinates of `A`). The gradient signal `dC` has coordinates `[i, j]`. We need to eliminate `j` and introduce `k`. Which operation does that? Multiply `dC[i, j]` by the *other* input `B[k, j]`, and sum over `j`:

```rust
let dA[i, k] = sum[j](dC[i, j] * B[k, j]);
```

The sum coordinate is `j`—the coordinate that `A` does not have but `C` does. The surviving coordinates are `i` (shared between `A` and `C`) and `k` (from `A`, reintroduced by `B`).

This is the pullback rule: **the gradient sums over the set-difference of coordinates between the output and the operand.** `C` has `{i, j}`. `A` has `{i, k}`. The difference is `{j}` minus `{k}`—we sum over `j` and `k` is provided by `B`.

You don't need to memorize the formula for a matmul pullback. You need the coordinate accounting. Which coordinates does the operand carry? Which does the output carry? The reduction coordinates in the gradient are exactly the coordinates present in the output but absent from the operand.

This rule is general. It works for any sum-of-products expression, not just matrix multiplication. It works for convolutions, for bilinear layers, for attention score computations. The coordinates tell you what to sum over. The rest is just multiplication with the other operands.

---

## The Five-Step Pullback Procedure

The pullback rule of Chapter 7 can be turned into a mechanical procedure. Given a forward expression and a target operand, derive the gradient:

1. **Hold one cell** of the target operand. Choose a specific element—say `A[i₀, k₀]`.
2. **List every output cell that reads it.** For `C[i, j] = sum[k](A[i, k] * B[k, j])`, the held cell `A[i₀, k₀]` is read by every output where `i = i₀` and the sum includes `k = k₀`. That means `C[i₀, j]` for *all* `j`.
3. **Attach the incoming gradient.** Each output cell `C[i₀, j]` carries a gradient signal `dC[i₀, j]`. The contribution from the path through `A[i₀, k₀]` is `dC[i₀, j] * B[k₀, j]`.
4. **Multiply by the local derivative.** For elementwise multiplication inside the sum, the local derivative of `A[i, k] * B[k, j]` with respect to `A[i, k]` is `B[k, j]`. So the total sensitivity at `A[i₀, k₀]` is `sum[j](dC[i₀, j] * B[k₀, j])`.
5. **Sum the routes.** The path coordinate—the coordinate in `C` but not in `A`—is `j`. Sum over it.

The result: `dA[i, k] = sum[j](dC[i, j] * B[k, j])`. No calculus memorization. No transpose rules. Just coordinate accounting.

This procedure works for *any* sum-of-products expression in einlang. For convolution, the same five steps produce the weight gradient with the correct index arithmetic. The coordinate set subtraction is the engine. The five steps are the manual.

---

## Convolution Gradients

A convolution is a sum of products, just like matrix multiplication, but with index arithmetic:

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

Again: set subtraction. The formula is mechanically derivable from the coordinate sets.

---

## Custom Differentiation with `@fn`

Some functions have derivatives that are better expressed directly than derived by the compiler. Einlang supports custom derivative rules with `@fn`:

```rust
fn relu(x) { if x > 0.0 { x } else { 0.0 } }

@fn relu(x) {
    if x > 0.0 { @x } else { 0.0 }
}
```

The `@fn` declaration shares the function's name and parameter list. Inside the body, `@x` refers to the tangent flowing through parameter `x`. The body describes how to assemble the output tangent from the input tangents.

Custom rules can also be coordinate-aware:

```rust
@fn softmax[j](x: [f32; ..left, j, ..right]) {
    softmax_tangent[j](x, @x)
}
```

The coordinate parameter `j` appears in both the primal function and its derivative rule. The tangent computation follows the same coordinate contract as the primal.

Custom rules are useful when:
- The function calls external code (Python/NumPy) whose internals the compiler can't trace.
- You want a surrogate derivative—for example, a straight-through estimator that passes the gradient through a discrete operation.
- The derivative is simpler to write by hand than to let the compiler derive (rare, but it happens).

---

## Recurrence and Gradients

A recurrence defines a sequence where later elements depend on earlier ones:

```rust
let fib[0] = 0;
let fib[1] = 1;
let fib[n in 2..8] = fib[n-1] + fib[n-2];
```

The gradient of a recurrence is itself a recurrence, running backward in time. If you differentiate `fib[7]` with respect to `fib[0]`, the compiler automatically constructs the backward recurrence that propagates the tangent from `n=7` back to `n=0`. The same coordinate names serve both directions—`n` steps forward, and the gradient steps backward along `n`.

We'll explore recurrence in depth in Chapter 11. For now, the key insight: **the coordinate structure of a recurrence determines the coordinate structure of its gradient.** Forward and backward share the same index domain. The same bracket syntax serves both.

---

## Where Clauses in the Backward Pass

A where clause in the forward pass affects the backward pass. Consider:

```rust
let pos_sum = sum[i](data[i]) where data[i] > 0;
```

In the forward pass, only positive elements are summed. In the backward pass, the gradient signal is distributed only to the positive elements. Elements that were filtered out receive zero gradient. The where clause acts as a gate in both directions—forward filtering, backward masking.

The consistency is automatic. You don't write a separate backward filter. The where clause defines the domain of the operation, and the domain applies in both directions.

---

The gradient is not a separate computation from the forward pass. It is the forward pass, read backward. The coordinate names that organize the forward pass—which survive, which are consumed, which are omitted—organize the backward pass in exactly the same way. A coordinate eliminated by a forward sum becomes a coordinate introduced by a backward broadcast. A coordinate omitted by a forward broadcast becomes a coordinate summed by a backward reduction.

The names are the same. The direction is reversed. The principle is symmetric.

Given `let C[i, j] = sum[k](A[i, k] * B[k, j])`, the gradient `@C / @B` is derived by set subtraction: the forward coordinates are `[i, j]`, the output consumes `[k]`. Applying the Inversion Rule, the backward pass must broadcast `[k]` and sum over `[i, j]`. The result has coordinates `[k, j]`, matching `B`'s shape—the compiler verifies this match automatically. For a custom `@fn` rule, consider `fn square(x) { x * x }`. Its derivative is `2.0 * x * @x`, and at `x = 3.0`, `@square(x) / @x = 6.0`. For a convolution `let y[b, o, i, j] = sum[c, kh, kw](x[b, c, i+kh, j+kw] * W[o, c, kh, kw])`, the forward output has coordinates `[b, o, i, j]` while the weight has coordinates `[o, c, kh, kw]`. The weight gradient must sum over `[b, i, j]`—the coordinates in the forward output that are not in the weight. Set subtraction, applied to coordinate names, derives this automatically. The pattern is mechanical: every forward reduction becomes a backward broadcast, every forward broadcast becomes a backward reduction, and the names stay the same.?
