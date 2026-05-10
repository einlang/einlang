---
layout: book
title: "Chapter 3 · Names as Contracts"
---

# Chapter 3 · Names as Contracts

> "The art of programming is the art of organizing complexity."
>
> — Edsger Dijkstra

*Combinations · Coordinate-aware functions*

---

We have spent two chapters learning to name coordinates. We name them when they survive, when they are consumed, when they are copied, and when they are rearranged. Every operation so far has been a single statement—a `let` declaration with brackets.

Real programs are not single statements. They are compositions. A softmax is not one operation but three—a max reduction, a broadcast subtraction, an exponentiation, a sum reduction, and a division. Each step involves coordinates with distinct roles. Composing them naively, with intermediate `let` bindings for every step, produces working code. But it scatters the coordinate story across multiple lines, and it gives the reader no way to see, at a glance, which coordinates are structurally load-bearing and which are just passing through.

This chapter introduces the mechanism that turns scattered primitives into a coherent composition: the **coordinate-aware function**.

---

## The Softmax Decomposition

Softmax is the workhorse of classification. It takes a vector of logits and returns a probability distribution:

$$\text{softmax}(x)_j = \frac{e^{x_j}}{\sum_k e^{x_k}}$$

Decomposed into primitives:

```rust
let m = max[j](logits[j]);
let e[j] = exp(logits[j] - m);
let z = sum[k](e[k]);
let probs[j] = e[j] / z;
```

This works. Every coordinate is named. Every reduction states what it consumes. But notice the subtle thing that happened: the stability scan uses `j` as the scanned coordinate, and the normalization uses `k`. These are the *same* coordinate—the class dimension—used in two different roles. The notation doesn't capture that identity because the names are local to each statement.

---

## The Coordinate-Aware Function

What if we could package this pattern—"normalize over the coordinate named in brackets"—as a single reusable operation, and make the coordinate name part of its **contract**?

```rust
fn softmax[j](x: [f32; ..left, j, ..right])
    -> [f32; ..left, j, ..right]
{
    let m[..left, ..right] = max[j](x[..left, j, ..right]);
    let e[..left, j, ..right] = exp(x[..left, j, ..right] - m[..left, ..right]);
    let z[..left, ..right] = sum[j](e[..left, j, ..right]);
    e[..left, j, ..right] / z[..left, ..right]
}
```

Let's read this carefully. It is the most important code block in the book.

`fn softmax[j]`—the `j` in brackets after the function name is a **coordinate parameter**. It is not a value. It is not a number. It is a coordinate name that the caller supplies at the call site.

`x: [f32; ..left, j, ..right]`—the parameter `x` is a tensor whose shape includes the coordinate `j`, plus zero or more leading coordinates (`..left`) and zero or more trailing coordinates (`..right`). These are **packs**—they stand for whatever coordinates surround `j` in the actual argument.

`-> [f32; ..left, j, ..right]`—the return type has the same coordinate structure as the input. Normalization preserves the shape.

Now the call:

```rust
let logits[b, class] = model(x[b, feature]);
let p[b, class] = softmax[class](logits[b, class]);
```

The caller writes `softmax[class](...)`. The `class` in brackets is not a comment. It is not a string. It is a coordinate argument—the name of the dimension the function will normalize over. That `logits` has a `class` coordinate is checked. If it doesn't, the call is a compile error.

Compare to the standard API:

```python
p = torch.softmax(logits, dim=-1)
```

`dim=-1` says "the last one." If the last dimension is `class`, this is correct. If upstream changes the dimension order, `dim=-1` silently begins normalizing over `batch`, or `feature`, or whatever happens to be last. The code runs. The output is a valid probability distribution—just over the wrong coordinate.

Now compare to a different einlang call:

```rust
let p[b, class] = softmax[b](logits[b, class]);
```

This normalizes over `b`—the batch dimension. It is a one-character bug (`b` instead of `class`). It is also a compile error, because `softmax[b]` would attempt to consume `b`, and the function signature says `b` should survive in `..left`. The coordinate contract catches the error at the call site.

---

## One Coordinate, Three Jobs

Inside `softmax[j]`, the coordinate `j` plays three distinct roles:

```rust
let m[..left, ..right] = max[q](x[..left, q, ..right]);    // Role 1: stability scan
let e[..left, k, ..right] = exp(x[..left, k, ..right] - m); // Role 2: exponentiate
let z[..left, ..right] = sum[j](e[..left, j, ..right]);     // Role 3: normalize
```

All three—`q`, `k`, `j`—range over the same domain (the class axis). But each carries a different **gradient contract**. The stability scan passes a sparse gradient: only the maximum element receives a signal. The denominator scan passes a dense gradient: every element contributes to the sum. The output has a diagonal-plus-off-diagonal structure: each element's gradient depends on itself and on every other element. The coordinate `j` is consumed by `max`, consumed again by `sum`, and reconstructed by the division—two different consumption events on the same coordinate, with the coordinate surviving both.

In a `dim=-1` API, these three roles collapse into a single integer. The reader cannot see which role `dim=-1` plays at each step. In the named-coordinate version, the roles are given distinct letters, and a reader can audit whether the gradient contracts are satisfied.

---

## The Square Matrix Test

There is a simple, brutal test for whether a piece of tensor code is robust to coordinate swaps. Set all dimension sizes equal. Swap two axes. Ask: does the program still mean the same thing?

For a square input where `batch_size == num_classes == 128`:

```rust
let probs[batch, class] = softmax[class](logits[batch, class]);   // correct
let probs[class, batch] = softmax[batch](logits[class, batch]);   // bug
```

Both lines produce a `(128, 128)` matrix where every row sums to 1. The cross-entropy loss descends identically. The training curves overlay perfectly. But the first normalizes classes against each other. The second normalizes examples against each other.

When `batch_size == num_classes`, the probability matrix is square. Softmax over rows and softmax over columns produce the same numbers when the matrix has symmetric structure. The loss curves overlay. The calibration reports pass. Six weeks later, a deployed model silently normalizes examples against each other instead of classes against each other.

![The Square Matrix Test: when dimensions have equal extent, only the coordinate name records which meaning was intended](figures/softmax_roles.svg)

No shape checker catches this. No gradient check catches this. Only a notation that records *which coordinate is the distribution* catches this. `softmax[class](logits)` records it. `softmax(logits, dim=-1)` does not.

The Square Matrix Test is named after this property: when all extents are equal, a coordinate swap can hide inside shape compatibility. If square matrices fool shape checkers—and they do, routinely—what can prevent this class of error?

---

## The Language Gets a Name

We have been writing in a notation that puts coordinate names in brackets, that requires reductions to state what they consume, that makes broadcasting explicit in the indexing pattern. This notation needs a name.

It is called **einlang**—a contraction of "Einstein" and "language," acknowledging the debt to Einstein summation notation while distinguishing itself as a full programming language rather than a string-based convention.

The name is not the point. The point is what the name represents: a language where coordinates are first-class syntactic entities, not comments embedded in variable names. Where coordinate contracts are statically checked. Where the reader can audit coordinate flow without reconstructing it from shape arithmetic.

---

A coordinate-aware function does something that no positional API can do: it makes the **identity** of the operated-on coordinate part of the function's type-level contract. The caller must name the coordinate. The name is checked against the argument's layout. The function body uses the name without knowing its position.

This is the combination layer. The primitives—naming, permuting, reducing, broadcasting—are composed into a function whose coordinate behavior is specified in its signature. The function can be called, passed around, and composed further, without losing the coordinate information that the primitives established.

In the next chapter, we explore how coordinate-aware functions compose into reusable skeletons—patterns that are identical across softmax, LayerNorm, RMSNorm, and GroupNorm, differing only in which coordinates play which roles.
