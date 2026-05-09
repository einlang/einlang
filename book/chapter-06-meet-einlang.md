---
layout: book
title: "Chapter 6 · Putting the Pieces Together"
---

# Chapter 6 · Putting the Pieces Together

> "The art of programming is the art of organizing complexity."
>
> — Edsger Dijkstra

*Combinations · Meeting einlang*

---

We have spent five chapters learning to name coordinates. We name them when they survive, when they are consumed, when they are copied, and when they are rearranged. Every operation so far has been a single statement—a `let` declaration with brackets.

Real programs are not single statements. They are compositions. A softmax is not one operation but three—a max reduction, a broadcast subtraction, an exponentiation, a sum reduction, and a division. Each step involves coordinates with distinct roles. Composing them naively, with intermediate `let` bindings for every step, produces working code. But it scatters the coordinate story across multiple lines, and it gives the reader no way to see, at a glance, which coordinates are structurally load-bearing and which are just passing through.

This chapter introduces the mechanism that turns scattered primitives into a coherent composition: the **coordinate-aware function**. It is the pivotal idea of the book. Everything before this chapter has been preparation for it. Everything after this chapter builds on it.

And at the end of the chapter, we give the language a name.

---

## The Softmax Decomposition

Softmax is the workhorse of classification. It takes a vector of logits and returns a probability distribution:

$$\text{softmax}(x)_j = \frac{e^{x_j}}{\sum_k e^{x_k}}$$

Decomposed into primitives, it is three steps:

```rust
// Step 1: find the maximum for numerical stability
let m = max[j](logits[j]);

// Step 2: exponentiate the shifted values
let e[j] = exp(logits[j] - m);

// Step 3: normalize
let z = sum[k](e[k]);
let probs[j] = e[j] / z;
```

This works. Every coordinate is named. Every reduction states what it consumes. But notice the subtle thing that happened: the stability scan uses `j` as the scanned coordinate, and the normalization uses `k`. These are the *same* coordinate—the class dimension—used in two different roles. The notation doesn't capture that identity because the names are local to each statement. A reader has to check that `j` in the `max` and `k` in the `sum` refer to the same domain.

This is not wrong. It is just local. The coordinate story is told one statement at a time.

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

`-> [f32; ..left, j, ..right]`—the return type has the same coordinate structure as the input. Normalization preserves the shape. What gets consumed and reconstructed inside the function is invisible to the caller.

The function body is the same three-step decomposition we wrote before, but with rest packs instead of concrete single coordinates. The packs `..left` and `..right` thread through the computation untouched, carrying whatever batch-like dimensions the caller provides.

Now the call:

```rust
let logits[b, class] = model(x[b, feature]);
let p[b, class] = softmax[class](logits[b, class]);
```

The caller writes `softmax[class](...)`. The `class` in brackets is not a comment. It is not a string. It is a coordinate argument—the name of the dimension the function will normalize over. The compiler checks that `logits` indeed has a `class` coordinate. If it doesn't, the call is a compile error.

Compare this to the standard API:

```python
p = torch.softmax(logits, dim=-1)
```

`dim=-1` says "the last one." If the last dimension is `class`, this is correct. If upstream changes the dimension order, `dim=-1` silently begins normalizing over `batch`, or `feature`, or whatever happens to be last. The code runs. The output is a valid probability distribution—just over the wrong coordinate.

Now compare to a different einlang call:

```rust
let p[b, class] = softmax[b](logits[b, class]);
```

This normalizes over `b`—the batch dimension. Every batch element sums to 1 individually, rather than every class summing to 1 per batch element. This is a one-character bug (`b` instead of `class`). It is also a compile error, because `softmax[b]` would attempt to consume `b`, and the function signature says `b` should survive in `..left`. The coordinate contract catches the error at the call site.

---

## What Just Happened

A coordinate-aware function does something that no positional API can do: it makes the **identity** of the operated-on coordinate part of the function's type-level contract. The caller must name the coordinate. The compiler checks the name against the argument's layout. The function body uses the name without knowing its position.

This is the combination layer. The primitives—naming, permuting, reducing, broadcasting—are composed into a function whose coordinate behavior is specified in its signature. The function can be called, passed around, and composed further, without losing the coordinate information that the primitives established.

The rest of this chapter names the language and explores what coordinate-aware functions unlock.

---

## One Coordinate, Three Jobs

Softmax is not one operation. It is three, sharing the same coordinate domain but playing different roles:

```rust
// Inside softmax[j], the coordinate j plays three distinct roles:
let m[..left, ..right] = max[q](x[..left, q, ..right]);    // Role 1: stability scan (q)
let e[..left, k, ..right] = exp(x[..left, k, ..right] - m); // Role 2: exponentiate (k)
let z[..left, ..right] = sum[j](e[..left, j, ..right]);     // Role 3: normalize (j)
```

All three—`q`, `k`, `j`—range over the same domain (the class axis). But each carries a different **gradient contract**. The stability scan (`q`) passes a sparse gradient: only the maximum element receives a signal. The denominator scan (`j`) passes a dense gradient: every element contributes to the sum. The output (`k` after division) has a diagonal-plus-off-diagonal structure: each element's gradient depends on itself (through the numerator) and on every other element (through the denominator).

![The Square Matrix Test: when dimensions have equal extent, only the coordinate name records which meaning was intended](figures/softmax_roles.svg)

The figure shows the Square Matrix Test applied to softmax. Two $3 \times 3$ probability matrices sit side by side, with row labels (img1, img2, img3) and column labels (cat, dog, fish). The left matrix was produced by `softmax[class]`—every row sums to 1, meaning each image's class probabilities are normalized against each other. The right matrix was produced by `softmax[batch]`—every row also sums to 1, but the normalization ran over examples instead of classes. When `batch_size == num_classes == 3`, the two matrices are numerically identical. The numbers in every cell match. No shape checker catches this. No gradient check catches this. Only a notation that records *which coordinate is the distribution* catches this.

Here is softmax over class vs. softmax over batch, with 3 classes and 3 examples:

```text
Logits (correct layout [batch, class]):
         cat  dog  fish
  img1: [2.0, 1.0, 0.1]  →  softmax over class → [0.66, 0.24, 0.10]
  img2: [0.5, 2.0, 0.3]  →  softmax over class → [0.14, 0.63, 0.23]
  img3: [0.1, 0.2, 2.0]  →  softmax over class → [0.10, 0.12, 0.78]
  Each row sums to 1.0. This is correct.

Logits (swapped layout [class, batch]):
  img1: [2.0, 1.0, 0.1]  →  softmax over batch → [0.66, 0.24, 0.10]
  img2: [0.5, 2.0, 0.3]  →  softmax over batch → [0.14, 0.63, 0.23]
  img3: [0.1, 0.2, 2.0]  →  softmax over batch → [0.10, 0.12, 0.78]
  Each row sums to 1.0. The numbers are IDENTICAL.
```

When `batch_size == num_classes`, the probability matrix is square. Softmax over rows and softmax over columns produce the same numbers when the matrix has symmetric structure. The loss curves overlay. The calibration reports pass. Six weeks later, a deployed model silently normalizes examples against each other instead of classes against each other.

No shape checker catches this. No gradient check catches this. Only a notation that records *which coordinate is the distribution* catches this. `softmax[class](logits)` records it. `softmax(logits, dim=-1)` does not.

In a `dim=-1` API, these three roles collapse into a single integer. The reader cannot see which role `dim=-1` plays at each step of the decomposition. In the named-coordinate version, the roles are given distinct letters, and a reader can audit whether the gradient contracts are satisfied.

## The Square Matrix Test

There is a simple, brutal test for whether a piece of tensor code is robust to coordinate swaps. Set all dimension sizes equal. Swap two axes. Ask: does the program still mean the same thing?

For a square input where `batch_size == num_classes == 128`:

```rust
let probs[batch, class] = softmax[class](logits[batch, class]);   // correct
let probs[class, batch] = softmax[batch](logits[class, batch]);   // bug
```

Both lines produce a `(128, 128)` matrix where every row sums to 1. The cross-entropy loss descends identically. The training curves overlay perfectly. But the first normalizes classes against each other. The second normalizes examples against each other. A deployed model using the second version will produce confidence scores that drift with batch size.

The Square Matrix Test is named after this property: when all extents are equal, a coordinate swap can hide inside shape compatibility. The test asks: if I rename the axes, does the program's *meaning* change? If so, the renamed axes should be reflected in the source—and the source should make them visible enough that a rename would break compilation or at minimum readability.

Apply the Square Matrix Test to any softmax call you write. If `softmax[class](logits)` would silently become wrong when `class` and `batch` swap, your notation is not recording the necessary fact.

## The Language Gets a Name

We have been writing in a notation that puts coordinate names in brackets, that requires reductions to state what they consume, that makes broadcasting explicit in the indexing pattern. This notation needs a name.

It is called **einlang**—a contraction of "Einstein" and "language," acknowledging the debt to Einstein summation notation while distinguishing itself as a full programming language rather than a string-based convention.

The name is not the point. The point is what the name represents: a language where coordinates are first-class syntactic entities, not comments embedded in variable names. Where the compiler checks coordinate contracts. Where the reader can audit coordinate flow without reconstructing it from shape arithmetic.

Einlang is the microscope. The coordinate habit is the specimen. This book teaches both.

---

## Composition Without Pipelines

Some languages use a pipeline operator (`|>`) to thread data through a sequence of transformations. Einlang does not need one. `let` bindings already express data-flow composition:

```rust
let step1 = normalize[feature](raw);
let step2 = activate(step1);
let step3 = project[out](step2, weight);
```

Each step names its result. The names serve as documentation. The data flow is visible in the sequence of bindings. No new syntax is required because the language's existing means of combination—`let` declarations, function calls, coordinate arguments—already compose cleanly.

This is a deliberate choice. Einlang is a small language. Every syntactic addition must justify itself against the alternative of using what's already there. Pipeline operators, in particular, add a new control-flow concept for a benefit—reading transformations in execution order—that named `let` bindings already provide.

---

## What We Carry Forward

This chapter crossed a threshold. Before it, we worked with primitives—single statements that name, permute, reduce, or broadcast coordinates. After it, we have a means of combination: the coordinate-aware function, whose signature declares which coordinates it consumes, which it preserves, and which it is polymorphic over.

The next three chapters explore what happens when combinations interact: the traps that refactoring creates (Chapter 7), the mirror world of gradients (Chapter 8), and the challenge of updating parameters across time (Chapter 9).

Consider writing the softmax decomposition as a coordinate-aware function `fn my_softmax[coord](x: [f32; ..left, coord, ..right]) -> [f32; ..left, coord, ..right]`. The rest packs make it work identically on a 2D tensor `logits[b, c]` and a 3D tensor `scores[b, h, c]`—the function is polymorphic over the structure surrounding `coord`. But coordinate names must be grounded: if you call `softmax[class](logits)` and `logits` has never been bound with coordinate names, the compiler rejects it. A shape `(32, 10)` is not a coordinate story. You must first bind `logits[b, c]` to ground the names, then `softmax[c]` works. In the same vein, a function `fn top1[class](x: [f32; ..left, class, ..right]) -> [i32; ..left, ..right]` that uses `argmax[class]` consumes the `class` coordinate—it is absent from the return type—while `..left` and `..right` pass through unchanged. The type signature is the contract, and the compiler enforces it at every call site.
