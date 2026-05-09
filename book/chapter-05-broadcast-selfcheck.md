---
layout: book
title: "Chapter 5 · When Promises Chain Together"
---

# Chapter 5 · When Promises Chain Together

> "Trust, but verify."
>
> — Russian proverb

*Primitives to Combinations · Broadcast self-audit, where clauses, and the first look at gradients*

---

Chapter 4 ended with a principle: when a term omits a coordinate, it promises independence from that coordinate. This chapter asks what happens when those promises interact—when one operation's broadcast feeds into another operation's reduction, when gradients flow backward through a broadcast, and when a conditional guard depends on the very values being combined.

We are crossing the bridge from primitives to combinations. The operations are still elementary, but we are about to compose them in ways that reveal hidden structure.

---

## The Broadcast Self-Audit

A forward broadcast makes a promise. A backward gradient collects dependence. If the two disagree, the gradient is silently wrong.

Consider a linear layer with a bias:

```rust
let z[b, out] = sum[in](x[b, in] * W[out, in]) + bias[out];
```

The bias term omits `b`—it promises that the bias does not depend on the batch element. The forward pass is correct. Now compute the gradient of the loss with respect to the bias:

```rust
let d_bias[out] = sum[b](d_loss[b, out]);
```

The gradient sums over `b`. Why? Because in the forward pass, `bias[out]` was replicated across every batch element. Every batch element carries a piece of the gradient signal. To update `bias`, we must collect all those pieces. The omitted coordinate becomes the reduced coordinate in the backward pass.

This is the broadcast-reduce duality: **what you broadcast over in the forward pass, you reduce over in the backward pass.**

Now ask yourself three questions about any broadcast in your code:

1. **What coordinate am I broadcasting over?** Is the name visible in the code, or is it inferred from position?
2. **Is independence truly justified?** Does the broadcast value genuinely not depend on that coordinate, or is this a shape coincidence?
3. **What will the gradient do?** In the backward pass, the broadcast coordinate becomes a reduction coordinate. Does that reduction produce the right shape for the parameter update?

These three questions are the broadcast self-audit. They cost nothing to ask. They catch the class of bugs where a broadcast is shape-correct but semantically wrong.

---

## Introducing `@y / @x`

Einlang provides built-in automatic differentiation. You don't call `loss.backward()`. You don't register hooks. You write the derivative you want directly:

```rust
let x = 1.0;
let y = 2.0;
let z = x + y;
let dz_dx = @z / @x;   // 1.0
let dz_dy = @z / @y;   // 1.0
```

`@z / @x` is an expression, not a framework call. It means "the derivative of the named binding `z` with respect to the named binding `x`." For scalars, it evaluates to a number. The compiler computes it through the chain rule, tracing the dependency graph you constructed with `let` bindings.

We will explore automatic differentiation in depth in Chapter 8. For now, we use it in the simplest possible way: as a tool for verifying that our broadcast contracts are self-consistent. If the gradient has the wrong shape, the broadcast was wrong. If the gradient sums over a coordinate you didn't expect, your broadcast omitted a coordinate you didn't intend.

`@z / @x` is a diagnostic instrument before it is an optimization tool. Point it at any broadcast. The shape of the result will tell you whether the broadcast contract holds.

---

## The `where` Clause

Sometimes a computation should only apply to a subset of coordinate values. In a positional API, you'd create a mask tensor, multiply, and hope the mask doesn't silently broadcast into the wrong dimension. In einlang, you attach a **`where` clause** directly to the declaration:

```rust
let pos_sum = sum[i](data[i]) where data[i] > 0;
```

The where clause is evaluated for each combination of the enclosing index variables. For reductions, elements where the guard is false are skipped—the reduction's identity element is used instead. For rectangular declarations, filtered-out positions receive the default value (zero for numeric types).

A where clause can also bind intermediate variables to avoid recomputation:

```rust
let output[i, j] = activated
    where z = sum[k](input[i, k] * weight[k, j]) + bias[j],
          activated = if z > 0.0 { z } else { 0.0 };
```

Without the where clause, you'd write the `sum[k](...)` expression twice—once for the comparison and once for the value. With the where clause, you name the shared subexpression `z` and refer to it in `activated`. The bindings are evaluated in order; later bindings can reference earlier ones.

The where clause is not a separate language feature bolted onto tensor operations. It is the natural extension of the idea that declarations state facts over coordinate domains. A where clause narrows the domain over which the fact holds. The syntax reflects the semantics directly.

---

## From Primitives to Combinations

We have now covered the four primitives of tensor computation:
- **Naming** (Chapter 1): every dimension gets a name tag.
- **Permutation** (Chapter 2): coordinate movement is stated in terms of names, not positions.
- **Reduction** (Chapter 3): a consumed coordinate is named in the reduction bracket.
- **Broadcasting** (Chapter 4): an omitted coordinate is visible by its absence from an index list.

We have also introduced our first means of combining these primitives: the where clause (which scopes a computation over a restricted coordinate domain) and the derivative operator (which reverses the flow of dependence).

In the next chapter, we give the language a name and learn how to combine primitives not just in single declarations, but in functions that carry coordinate information across call boundaries.

Given `let z[b, o] = sum[i](x[b, i] * W[o, i]) + bias[o]`, the gradient `@z / @bias` sums over `[b, o]`—those are the coordinates of `z`. The Inversion Rule tells us that what was omitted in the forward pass (`b` in `bias[o]`) becomes a reduction coordinate in the backward pass. A where clause keeps this machinery in one place: `let pos_mean = sum[i](data[i] where data[i] > 0) / sum[i](1 where data[i] > 0)` computes the mean of positive elements in one line. Compare to the PyTorch equivalent—it's four lines with masks and at least one `sum` whose axis you have to count. But the compiler does not read intent: `let y[b, f] = x[b, f] + scale[f] where scale[f] > 0` broadcasts `scale[f]` over `b` only when `scale[f] > 0`—a boolean guard that gates the *values*, not the coordinate structure. The coordinate contract stays intact; the where clause scopes the domain, not the shape.
