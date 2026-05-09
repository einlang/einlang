---
layout: book
title: "Chapter 9 · Updates, with Names"
---

# Chapter 9 · Updates, with Names

> "The only way to make sense of change is to plunge into it, move with it, and join the dance."
>
> — Alan Watts

*Combinations · Parameter updates and recurrence syntax*

---

A trained model is a collection of parameter tensors whose values have been adjusted by an optimizer. The optimizer's job is to nudge each parameter in the direction that reduces the loss. Which direction? That depends on the parameter's shape, its coordinate names, and the gradient's coordinate structure.

This chapter is about the intersection of optimization and coordinate names. When a weight decay regularizer pushes all values toward zero, it should push uniformly across the *feature* dimension but not across the *output* dimension. When an optimizer normalizes gradients by their variance, it should normalize within each *layer*, not across layers. These distinctions are invisible in a parameter tensor named `w` with shape `(128, 64)`. They become visible when the tensor carries coordinate names.

---

## Parameters vs. Hyperparameters

Einlang distinguishes between two kinds of named values:

```rust
let weight: [f32; out, in] = init_random(out, in);
let learning_rate: f32 = 0.001;
```

Both are immutable bindings. Both are in scope for subsequent code. But they have different roles in the optimization story. `weight` is a parameter—its value changes during training, driven by gradients. `learning_rate` is a hyperparameter—it controls the optimizer's behavior but is not itself updated by gradients.

In einlang, the distinction is semantic rather than syntactic. Both use `let`. The convention is that parameters carry tensor types with coordinate names, while hyperparameters are scalars or small configuration tensors. The coordinate names on parameters serve as documentation for the optimizer: `weight[out, in]` tells you that the first axis corresponds to output neurons and the second to input neurons. A weight decay regularizer that treats all elements uniformly can apply without coordinate awareness, but a per-layer or per-neuron regularization policy needs to know which axis is which.

This is not a language feature. It is a naming discipline. But the discipline is only possible because the language provides a place to put the names. When your tensor framework only records shapes, you can't name the axes; when you can't name the axes, you can't write a regularizer that distinguishes them.

---

## Recurrence for Parameter Updates

Training is inherently sequential. You compute a forward pass, compute a loss, compute gradients, and update parameters—then repeat. The repetition has a time structure. In einlang, that structure is expressed with **recurrence declarations**:

```rust
let w[0, out, in] = init_random(out, in);
let w[t in 1..T, out, in] = w[t-1, out, in] - lr * grad[t-1, out, in];
```

This defines `w` over a time coordinate `t`. At `t=0`, `w` is the random initialization. At each subsequent step, `w` is the previous `w` minus a gradient step. The recurrence reads backward in time (`t-1`)—a fundamental constraint that we'll explore in Chapter 11.

The coordinate `t` makes the training trajectory explicit. You can inspect `w[10, out, in]` to see the weights after 10 steps. You can compute `w[T-1, out, in] - w[0, out, in]` to see the total change. The time dimension is not hidden inside a mutable variable—it is a coordinate like any other.

---

## Declaration Bracket Rules

Recurrences introduce an important syntactic constraint: what can appear in the declaration bracket.

Every index slot in a `let` declaration's left-hand-side bracket must be one of:

- An **identifier**: `i`, `t`, `batch`, `channel`—a name for the index variable.
- A **literal**: `0`, `1`—used for base cases in recurrences.
- A **named rest**: `..batch`—standing for zero or more axes.

Expressions are **not** allowed in the declaration bracket. You cannot write:

```rust
let fib[n-1] = ...;   // ERROR: n-1 is an expression, not an identifier
```

The declaration bracket names *what* is being defined. The body says *how* it is computed. This separation keeps the declaration side simple and declarative, while the computation side can use arbitrary index expressions.

In a recurrence, the body references earlier elements using index arithmetic:

```rust
let fib[n in 2..8] = fib[n-1] + fib[n-2];   // OK: n-1, n-2 are in the body
```

The recurrence index range `n in 2..8` goes in the declaration bracket—it defines the domain over which the recurrence holds. The expressions `n-1` and `n-2` go in the body—they compute the value by referencing earlier elements.

The constraint that recurrences may only reference **backward** along every dimension is enforced by the compiler. You cannot read `fib[n+1]` when defining `fib[n]`—that value hasn't been computed yet. This is not an arbitrary restriction. It is the mathematical definition of a recurrence: each value depends only on values with strictly smaller indices.

---

## The Shape of Optimization

Let's put these pieces together. A full optimization step in einlang:

```rust
// Forward pass
let logits[t, b, class] = model(x[t, b, feature], w[t, out, feature]);
let loss[t] = cross_entropy[class](logits[t, b, class], labels[t, b]);

// Gradient
let grad[t, out, feature] = @loss[t] / @w[t, out, feature];

// Update
let w[t+1, out, feature] = w[t, out, feature] - lr * grad[t, out, feature];
```

The time coordinate `t` threads through forward, loss, gradient, and update. Every tensor knows its temporal position. The gradient `@loss[t] / @w[t, out, feature]` is explicitly anchored to time step `t`. The optimizer step defines `w[t+1]` in terms of `w[t]` and `grad[t]`.

This is not the only way to write optimization in einlang. You can use a simpler style with separate `let` bindings per step. But the recurrence form makes the temporal structure explicit, and explicitness is what the coordinate habit demands.

---

We have now covered the combination layer. We can write functions that carry coordinate contracts (Chapter 6), refactor them safely (Chapter 7), differentiate them (Chapter 8), and optimize them across time (Chapter 9). The next layer is abstraction: building our own primitives.

An exponential moving average recurrence—`ema[0] = x[0]; ema[t in 1..T] = alpha * x[t] + (1 - alpha) * ema[t-1]`—makes the compiler enforce three constraints: the base case `ema[0]` must exist, every recursive step must reference a strictly earlier time step, and the coordinate sets must be consistent across all branches. You cannot write `let w[t+1] = ...` in the declaration bracket because a forward reference would create a circular dependency: the value at `t+1` depends on itself. The recurrence syntax enforces causality by construction. A colleague who writes `let h[t in 0..T] = step(h[t+1], x[t])` will get a compile error—it violates the rule that time flows forward and the backward reference must be strictly negative. The compiler catches what the position-based index would silently accept.
