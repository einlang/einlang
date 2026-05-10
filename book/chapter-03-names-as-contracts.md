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

Now pause and think about a different kind of failure. You write `softmax[class](logits)` and it works. Three months later, a colleague refactors the model. They rename the `class` coordinate to `category`—a better name, more consistent with the rest of the codebase. They update twenty-three files. They miss one: the call to `softmax[class]`. What happens?

The call is now `softmax[class](logits[b, category])`. The compiler checks: does `logits` have a `class` coordinate? No—it has `batch` and `category`. The error is not "shape mismatch." It is not "dimension 1 out of bounds." It is: **`logits` has no coordinate named `class`.** The error message names the missing coordinate. The fix is to change one character in the brackets: `softmax[category](...)`.

Now consider the same scenario with `dim`. The old code is `softmax(logits, dim=1)`. The colleague changes the coordinate name, not the position. `dim=1` remains correct. The code compiles. The code runs. Everything is fine—*this time.* The position didn't change, only the name did.

But what if the colleague had also changed the dimension order? What if `category` moved to position 2? `dim=1` would silently begin normalizing over the wrong coordinate. No error. No warning. The loss goes down. The model deploys.

Here is the distinction in one sentence: **when positions change without names changing, `dim=-1` silently becomes wrong. When names change without positions changing, `dim=-1` silently stays correct. A named coordinate gets the first case wrong and the second right. A positional API gets the first case wrong and the second accidentally right.** The named coordinate is predictably correct: it fails when the name is wrong, and it fails loudly. The positional API is unpredictably correct: it fails silently when the name drifts from the position, and it succeeds by luck when they stay aligned.

Neither notation prevents all errors. But named coordinates make the errors *visible*. A compile error is visible. A silent semantic drift is not.

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

Now ask yourself: why do we use three different letters (`q`, `k`, `j`) for the same coordinate? Why not just `j` everywhere? Because the *binding site* of each occurrence carries different gradient implications. `max[q]` says: "I consume `q` and return a scalar per batch element. The backward pass through me will broadcast the gradient signal to only the maximum element." `sum[j]` says: "I consume `j` and return a scalar per batch element. The backward pass through me will broadcast the gradient signal to *all* elements." Same coordinate domain. Different gradient contracts. Different letters make each contract's scope visible: `q` is consumed by `max` and never seen again. `k` is used in the exponent and survives. `j` is consumed by `sum` and reconstructed by the division.

The letters are not decoration. They are the scope markers for the coordinate's three lives. In `dim=-1`, all three are the same integer. The scopes are invisible.

---

## What Gets Checked at the Call Site

Let's trace exactly what happens when you call `softmax[class](logits)`.

1. The compiler looks up `softmax`'s signature: coordinate parameter `j`, value parameter `x` with layout `[..left, j, ..right]`.
2. The coordinate argument `class` is bound to `j`.
3. The value argument `logits` is checked: does it have a coordinate named `class`? Yes—`logits` was declared as `logits[batch, class]`.
4. The packs are bound: `..left` = `[batch]`, `..right` = `[]` (nothing on the right).
5. The return type is instantiated: `[f32; batch, class]`.
6. The call is valid.

Now consider five wrong calls:

| Call | What goes wrong | Caught by |
|---|---|---|
| `softmax[class](logits[batch, feature])` | `logits` has no `class` | Step 3: index existence |
| `softmax[batch](logits[batch, class])` | `batch` would be consumed, but `softmax` preserves `j` in the return type—`batch` would be consumed and returned, which is a contract violation | Step 5: return type enforcement |
| `softmax[class](logits)` where `logits` is 1D | `class` exists but `..left` and `..right` are both empty—valid | No error (this is correct) |
| `softmax[class](logits[batch, class, extra])` | `..right` binds to `[extra]`, which is fine—`softmax` is polymorphic over trailing dims | No error (this is correct) |
| `softmax[class](logits[batch, class], wrong_arg)` | Wrong number of value arguments | Step 1: arity check |

Each check is a single, mechanical verification. No shape guessing. No positional arithmetic. The coordinate names and the function signature together form the contract. The call site satisfies it—or doesn't.

---

## Function Composition

Coordinate-aware functions compose. The output of one becomes the input of another, and the coordinate contracts chain.

Consider a pipeline: linear layer, then softmax:

```rust
fn linear[in, out](x: [f32; ..batch, in], W: [f32; out, in], b: [f32; out])
    -> [f32; ..batch, out]
{
    sum[in](x[..batch, in] * W[out, in]) + b[out]
}

fn pipeline[in, class](x: [f32; ..batch, in], W: [f32; class, in], b: [f32; class])
    -> [f32; ..batch, class]
{
    let logits[..batch, class] = linear[in=in, out=class](x[..batch, in], W[class, in], b[class]);
    softmax[class](logits[..batch, class])
}
```

The coordinate `class` flows from the pipeline's signature through `linear[out=class]` into the result `logits`, then into `softmax[class]`. At each step, the compiler checks: does the argument carry the coordinate the function expects? `linear` expects `out`—the caller binds `class` to `out`. `logits` now carries `class`. `softmax[class]` expects `class` on its argument—`logits` has it. The chain is verified.

If a refactoring changes `linear`'s output coordinate from `class` to `category`, the pipeline still compiles—`linear[out=category]` produces a tensor with `category`, and `softmax[class]` complains that `category` is not `class`. The error is at the composition boundary. The compiler names both coordinates. The mismatch is visible.

Positional composition has no such check. `logits = linear(x); softmax(logits, dim=-1)`. If `linear`'s output layout changes, `dim=-1` silently normalizes over a different coordinate. The chain is unverified.

---

## The Contract in One Question

Every coordinate-aware function can be audited with a single question: **does the caller's argument carry the coordinate that the function claims to operate on?**

If yes: the call compiles, and the coordinate flow is guaranteed consistent. If no: the call is rejected, and the error message names the missing coordinate.

This is the difference between a type-level contract and a documentation-level contract. `def softmax(logits, dim=-1)` has a documentation-level contract: the docstring says `dim` is the class dimension. But nothing checks it. `fn softmax[j](x)` has a type-level contract: `j` is a coordinate parameter, and the compiler verifies that the argument carries it. The contract is not a hope. It is a check.

The rest of this book builds on this distinction. Every chapter from here forward—skeletons, recurrences, gradients, comparisons—assumes that coordinate identities can be checked. The coordinate-aware function is the mechanism that makes checking possible.

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

But there is a subtler consequence. When coordinates are part of the type-level contract, refactoring becomes checkable. Rename a coordinate from `class` to `category`, and every call site that passes `class` becomes a compile error. The error message names the call site and the missing coordinate. The refactoring is systematic: change all `class` to `category`, and the errors disappear one by one. No silent breakage. No "hope I found all the places." The compiler is the audit trail.

In a positional API, renaming a dimension's role (e.g., changing what "position 1" means) produces no errors. The code compiles. The integer didn't change. Only the meaning changed. The compiler can't track meaning. It tracks integers. The refactoring is silent—and its bugs are silent with it.

This is the argument of the book, stated at the combination layer: when the coordinate identity is part of the function's contract, the compiler can audit every use of that identity. When the identity is in a comment, the audit is yours to perform manually. The difference between the two is the difference between a check and a hope.


---

*Design a coordinate-aware function. Write its signature. Now imagine a colleague calls it with the wrong coordinate name. What happens? In `dim=-1`, nothing—the call succeeds silently. In your function, the compiler stops and says: "this tensor has no coordinate named X." The difference is a bracket with a name in it.*

---

## Coordinate Parameters vs. Value Parameters

A coordinate-aware function has two kinds of parameters, and the distinction matters.

**Coordinate parameters** (in brackets after the function name: `fn softmax[j]`) name the coordinate the function operates on. They control the structure of the computation—which axis is reduced, which axis is broadcast, which axis is preserved. They are not values; you cannot do arithmetic on them. They are compile-time identities.

**Value parameters** (in the parenthesized argument list: `x: [f32; ..left, j, ..right]`) hold the data. Their shapes reference the coordinate parameters. They are the tensors the function transforms.

The separation is syntactic but the implication is semantic. The coordinate parameter says: "this function's behavior depends on the identity of this coordinate, not its position." The value parameter says: "this function consumes this tensor." A function that takes `j` as a coordinate parameter and `x` as a value parameter cannot confuse which is which. The brackets hold the coordinate. The parentheses hold the value.

At the call site, the same separation applies: `softmax[class](logits)`. `class` is in brackets—it is a coordinate argument. `logits` is in parentheses—it is a value argument. The syntax tells the reader which is which. In `softmax(logits, dim=-1)`, `dim=-1` is syntactically identical to any other integer argument. The syntax does not distinguish "this integer controls the axis of reduction" from "this integer is a value being passed to the function." The distinction lives in the documentation. The named version puts it in the syntax.

This separation between coordinate parameters and value parameters is what enables the compiler to check the coordinate contract. Without it, a coordinate name passed to a function would be just another string or symbol—the compiler couldn't know it was supposed to be verified. The brackets create a syntactic position that the compiler recognizes as "coordinate argument." Everything in that position gets checked. Everything outside it gets normal type-checking. The syntax carves out a space for coordinate verification.

In the next chapter, we explore how coordinate-aware functions compose into reusable skeletons—patterns that are identical across softmax, LayerNorm, RMSNorm, and GroupNorm, differing only in which coordinates play which roles.
