---
layout: book
title: "Chapter 7 · The Refactoring Trap"
---

# Chapter 7 · The Refactoring Trap

> "The first principle is that you must not fool yourself—and you are the easiest person to fool."
>
> — Richard Feynman

*Combinations · Immutability, where-clause bindings, and boolean guards*

---

There is a species of refactoring that makes code shorter, cleaner, and wrong.

It happens when you notice that two branches of an `if` share a structure, so you merge them into a single loop. Or when you see a repeated subexpression and factor it out. Or when you inline a variable that "isn't doing anything."

Each of these is a good instinct. Each of them, applied to tensor code without coordinate awareness, can silently delete the information that made the code correct.

This chapter is about the traps that sit between "it works" and "it's clean." The traps are not bugs in the language. They are bugs in the assumption that shorter code preserves meaning. In a positional notation, that assumption is frequently false. In a named-coordinate notation, the syntax itself prevents the most common forms of semantic collapse.

---

## Immutability: The Binding That Doesn't Change

Every `let` binding in einlang is immutable. Once a name is bound to a value, that binding cannot be changed within its scope:

```rust
let x = 10;
let x = x + 1;   // ERROR: redefinition of `x` in the same scope
let x_next = x + 1;   // OK: new name
```

This is not a stylistic preference. It is a semantic guarantee that serves the coordinate audit. When you see `x[b, c]` on line 50 and trace it back to its definition on line 10, you know that `x` still means what it meant on line 10. There is no mutation that could have changed its shape, its coordinates, or its values between definition and use.

In a language with mutable variables, refactoring often introduces bugs by accidentally rebinding a name to a tensor with a different coordinate layout. You had `x` as `(batch, channel, spatial)`; later someone inserts `x = x.transpose(1, 2)`. The name `x` now points to a tensor with a different axis order. All downstream code that assumed the original order is now silently wrong. The shapes might even match, if `channel` and `spatial` happen to have the same size.

Immutability eliminates this class of bug entirely. A `let` binding is a definition, not an assignment. It states a fact. Facts don't change.

---

## The Where Clause: Naming Without Cluttering

Chapter 5 introduced the `where` clause for boolean guards. But the where clause has a second, equally important role: it provides a scoped space for intermediate variables that are only relevant to a single declaration:

```rust
let output[i, j] = activated
    where z = sum[k](input[i, k] * weight[k, j]) + bias[j],
          activated = if z > 0.0 { z } else { 0.0 };
```

Without the where clause, you face a choice. Either you repeat the `sum[k](...)` expression:

```rust
// Bad: duplicated computation, and the two copies can drift apart
let output[i, j] = if sum[k](input[i, k] * weight[k, j]) + bias[j] > 0.0
    { sum[k](input[i, k] * weight[k, j]) + bias[j] }
    else { 0.0 };
```

Or you introduce a temporary tensor that exists at the scope level:

```rust
// Better, but `z` now lives in the outer scope
let z[i, j] = sum[k](input[i, k] * weight[k, j]) + bias[j];
let output[i, j] = if z[i, j] > 0.0 { z[i, j] } else { 0.0 };
```

The where clause gives you a third option: name the intermediate value, but keep it lexically scoped to the declaration that needs it. `z` and `activated` exist only within the where clause. They do not clutter the outer scope. The reader knows they are implementation details of this one declaration, not structural values that other code depends on.

This is a small syntactic feature with an outsized impact on refactoring safety. When you extract a computation into a where-clause binding, you are not changing the scope of any name visible to other code. You are localizing the change. Local changes are easier to audit.

---

## Boolean Guards: Filtering Without Masks

A boolean guard in a where clause filters the domain over which the declaration applies:

```rust
let pos_sum = sum[i](data[i]) where data[i] > 0;
let upper[i, j] = matrix[i, j] where i <= j;
```

These read like set comprehensions: "sum over `i` such that `data[i]` is positive." No mask tensor is created. No broadcasting semantics need to be verified. The guard operates on the index variables directly.

In a positional framework, you'd write:

```python
mask = data > 0
pos_sum = (data * mask).sum()
```

This creates a temporary tensor `mask`. It relies on broadcasting to align `mask` with `data`. If `data` has an unexpected batch dimension, the mask silently broadcasts into it. If you later refactor to remove the temporary, you might accidentally change the broadcasting behavior.

The where clause avoids all of this by attaching the filter to the reduction syntactically. The filter's domain is the reduction's domain. There is no separate mask whose shape must be audited.

---

## Refactoring with Names

Let's put these pieces together. A common refactoring is to take a repeated pattern and extract it into a function. Here is the before, in positional PyTorch:

```python
# Before: two instances of the same pattern
h1 = torch.relu(torch.matmul(x, W1) + b1)
h2 = torch.relu(torch.matmul(x, W2) + b2)
```

A clean refactoring would extract the linear-relu pattern:

```python
def linear_relu(x, W, b):
    return torch.relu(torch.matmul(x, W) + b)
```

What information was lost? The coordinate structure. `torch.matmul(x, W)` contracts the last dimension of `x` with the second-to-last of `W`. Which coordinate is that? The function signature doesn't say. The caller has to know that `x` must end with the input-feature dimension and `W` must have output-feature followed by input-feature. If either convention changes, every call site silently breaks.

Now in einlang:

```rust
fn linear_relu[in](x: [f32; ..batch, in], W: [f32; out, in], b: [f32; out])
    -> [f32; ..batch, out]
{
    let z[..batch, out] = sum[in](x[..batch, in] * W[out, in]) + b[out];
    if z[..batch, out] > 0.0 { z[..batch, out] } else { 0.0 }
}
```

The coordinate `in` is a coordinate parameter—it names the contracted dimension. The coordinate `out` appears in `W`'s shape and the return type. The rest pack `..batch` handles whatever batch structure the caller provides. The coordinate contract is in the type signature. The compiler checks it at every call site. The refactoring did not hide anything, because the syntax had a place to put the facts.

---

When you refactor, you are making a bet: that the new, shorter code is equivalent to the old, longer code. In a positional notation, verifying that bet requires holding the coordinate story in your head while comparing two versions of the code. In a named-coordinate notation, the coordinate story is part of the text you're comparing. The bet is easier to verify because the evidence is on the page.

Next chapter: the gradient. When the forward pass eliminates a coordinate, the backward pass must recover it. We'll see how the coordinate contract extends naturally into differentiation.

Refactor the attention computation:

```rust
let scores[b, i, j] = sum[d](Q[b, i, d] * K[b, j, d]);
let weights[b, i, j] = softmax[j](scores[b, i, j]);
let output[b, i, d] = sum[j](weights[b, i, j] * V[b, j, d]);
```

into a single declaration with a where clause. Does your version still clearly name which coordinates survive and which are consumed? Now imagine the coordinate layout changes upstream—`Q` arrives as `[b, d, i]` instead of `[b, i, d]`. In a positional notation, the refactor silently breaks. In the named-coordinate version, the compiler catches the mismatch. This is what `let` immutability buys you: each intermediate tensor carries an explicit coordinate binding, and the compiler checks each one against every consumer. During a refactoring, those bindings are the guard rails—they fail fast when the layout drifts. A boolean guard like `let upper[i, j] = matrix[i, j] where i <= j` introduces another consideration: positions where `i > j` are not present in the output, so the compiler must define a fill value for downstream consumers. The coordinate set shrinks by value, not by structure, and the compiler tracks that difference.
