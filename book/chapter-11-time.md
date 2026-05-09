---
layout: book
title: "Chapter 11 · Names in Time"
---

# Chapter 11 · Names in Time

> "Time is what prevents everything from happening at once."
>
> — John Archibald Wheeler

*Abstraction extended · Recurrence relations, directional constraints, and gradients through time*

---

A tensor is a map from coordinates to values. When one of those coordinates represents time, the map acquires a direction. The value at time `t` can depend on the value at time `t-1`. It cannot depend on the value at time `t+1`—that value does not exist yet.

This is intuitively obvious when stated as a sentence. It is easy to violate when writing code. A `for` loop that mutates a state variable has no built-in protection against accidentally reading a future value through an aliased reference. A positional recurrence like `h[t] = f(h[t+1])` is accepted by most tensor frameworks as a valid indexing operation—the array exists, so the read succeeds—even though it is mathematically nonsensical as a causal definition.

Einlang's recurrence syntax is designed to make the direction of time visible in the source and enforceable by the compiler.

---

## Recurrence Syntax, Revisited

A recurrence in einlang is a self-referential rectangular declaration with an explicit domain:

```rust
let fib[0] = 0;
let fib[1] = 1;
let fib[n in 2..8] = fib[n-1] + fib[n-2];
```

The base cases (`fib[0]`, `fib[1]`) are evaluated first. The recursive case is evaluated in increasing order of `n`, so that `fib[n-1]` and `fib[n-2]` are already computed when `fib[n]` is evaluated.

The key syntactic constraint: **the index slot in the declaration bracket may only contain an identifier, a literal, or a named rest. No expressions.** You write `fib[n in 2..8]`, not `fib[n-1]`. The expression `n-1` belongs in the body, where it references a value already defined:

```rust
let fib[n in 2..8] = fib[n-1] + fib[n-2];
//  ^^^^^^^^^^^^^^   ^^^  ^^^
//  declaration      body index expressions (backward references)
//  bracket
```

This separation is not arbitrary. The declaration bracket names *what* is being defined—the element at index `n` over the domain `2..8`. The body says *how* to compute it—by referencing elements at indices `n-1` and `n-2`. The left side is a definition. The right side is a computation.

---

## The Backward-Only Constraint

The compiler enforces a simple rule: **when defining an element of a recurrence array, you may only read elements at strictly smaller indices along every dimension.**

For a 1D recurrence, this means `h[t-1]` is allowed and `h[t+1]` is an error. For a 2D recurrence with time and space dimensions, the rule applies per dimension:

```rust
// Valid: backward in time, backward in space
let h[t in 1..T, i in 1..N-1, j in 1..N-1] =
    f(h[t-1, i, j], h[t, i-1, j], h[t, i, j-1]);

// Error: reads the future in time
let h[t in 0..T-1, i, j] = f(h[t+1, i, j]);
//                               ^^^^^^ ERROR

// Error: reads the future in space
let h[t, i, j] = f(h[t, i+1, j]);
//                      ^^^^^^ ERROR
```

This constraint has two justifications. The practical one: recurrences are evaluated in increasing index order, so future values don't exist yet. The mathematical one: a recurrence is a causal definition. The value at a point in the index space is determined by values that are strictly "before" it in the partial order induced by the coordinate axes.

The compiler checks the constraint statically. The index expressions in the body are compared against the declaration's index variables. Any read at an index greater than or equal to the declaration index (along matching dimensions) is rejected. This catches the "read the future" bug at compile time, rather than silently producing a value that happens to be in memory from a previous run.

---

## Recurrence and Autodiff

A recurrence defines a computation graph that unfolds along the time axis. The gradient of a recurrence is itself a recurrence, running backward in time.

Consider:

```rust
let h[0] = init;
let h[t in 1..T] = step(h[t-1], x[t]);
let loss = loss_fn(h[T-1]);
```

The gradient `@loss / @h[0]` requires propagating the tangent from `t=T-1` backward to `t=0`, through each step of the recurrence. The compiler constructs this backward recurrence automatically. The user does not write a separate backward pass. The same coordinate `t` that organized the forward computation organizes the backward computation, running in reverse.

This is not a special case. It is the same pullback rule from Chapter 8—the gradient sums over the set-difference of coordinates—applied to a graph that happens to be sequential. The coordinate `t` is a coordinate like any other. The fact that it represents time, with a directional constraint, is additional structure that the compiler uses to order computation, but it does not change the differentiation rules.

---

## Time Steps Are Not Loops

A `for` loop is an execution story. It says: "do this, then this, then this." A recurrence is a dependency story. It says: "the value at time `t` depends on the value at time `t-1` in this specific way."

The difference matters when you ask questions that are not about execution order:

- **Storage**: which time steps must be kept in memory? The recurrence declaration does not specify—it only states dependencies. The compiler can allocate storage based on which time steps are actually observed by later code (Chapter 11 of the existing book, now folded into this discussion).

- **Parallelism**: can time steps be computed in parallel? A `for` loop says no (by fiat). A recurrence says: it depends on the dependency graph. If the recurrence only references `t-1`, it is inherently sequential. If it references `t-k` for `k > 1`, there may be pipeline parallelism. The dependency structure, not the loop syntax, determines what can be parallel.

- **Gradient checkpointing**: which intermediate values must be saved for the backward pass, vs. recomputed? The recurrence's dependency graph answers this question mechanically.

The recurrence syntax separates these concerns. The declaration states the dependency. The compiler decides the execution. The human reasons about the dependency. The compiler reasons about the execution.

A recurrence makes three causal claims. Each is checkable at the declaration site:

1. **Monotonicity.** The index only moves forward. `t` goes from `0` to `T-1`. The body references `t-1`, `t-2`, `t-k`—strictly smaller indices. A reference to `t+1` is a compile error. This is the syntactic embodiment of "the future hasn't happened yet."

2. **Bounded memory.** The largest backward offset determines the recurrence's memory window. If the body only reads `t-1`, the compiler knows a rolling buffer of size 1 suffices. If it reads `t-k`, the window is `k`. The declaration itself advertises the memory requirement.

3. **Acyclicity.** Every dimension of the recurrence must be strictly monotonic in its own index. You cannot have `let h[t, i] = f(h[t, i-1], h[t-1, i])` where two dimensions chase each other in a cycle. The compiler checks that each dimension's backward references form a DAG — a directed acyclic graph across the coordinate space.

These claims are not enforced by a `for` loop. A loop lets you read any element of any array at any time. The loop's causal structure is whatever the loop body does—invisible to the compiler, discoverable only by running the code or reasoning about it carefully. The recurrence syntax makes the causal structure visible in the declaration.

---

## Storage Follows Observation

A recurrence *defines* a sequence of values. It does not *materialize* all of them.

```rust
let h[0] = init;
let h[t in 1..T] = step(h[t-1], x[t]);
```

These two lines define `h` for every `t` from `0` to `T-1`. But which values actually occupy memory depends on which values are *observed* by later code:

```rust
let final = h[T-1];              // Only need the last → rolling buffer of 1
let trace[t] = h[t];             // Need every value → full materialization
let every_other[u] = h[2 * u];   // Need every other → partial materialization
```

The same recurrence, three different storage plans. The compiler decides based on the observation pattern, not the definition. The human states what they need. The compiler allocates accordingly.

This separation—definition, storage, observation—is impossible in a `for` loop. A loop fuses all three. You cannot "observe every other time step" without rewriting the loop. You cannot "checkpoint and recompute" without inserting framework-specific calls. The recurrence syntax keeps these concerns distinct.

Gradient checkpointing is the same idea in reverse. The forward recurrence produces values the backward recurrence reads. Which intermediate values should be saved, and which recomputed? The answer depends on memory budget and compute cost. But the *possibility* of checkpointing comes from the dependency graph being visible in the source.

---

## What We've Built

We now have all three layers:

**Primitives** (Chapters 1–5): naming coordinates, permuting them, reducing them, broadcasting them. The fundamental operations that manipulate individual coordinates.

**Combinations** (Chapters 6–9): composing primitives into functions with coordinate contracts, refactoring without semantic collapse, differentiating through compositions, updating parameters across time.

**Abstractions** (Chapters 10–11): naming coordinate patterns so they can be used as primitives—coordinate-aware functions with packs and selection reductions, and recurrences that make time a first-class coordinate with directional constraints.

---

## Patterns of Recurrence

The simple `h[t] = f(h[t-1], x[t])` pattern generalizes in two important directions.

### Multiple Gates: The LSTM

An LSTM cell has four gates, each scanning the previous hidden state with its own weight matrix. The coordinate pattern `h` (current output) vs `h_prev` (previous output) keeps the roles distinct:

```rust
let hidden[t in 1..T, b, h] = {
    let g_input = sigmoid(sum[i](x[t, b, i] * Wi[h, i])
                        + sum[h_prev](hidden[t-1, b, h_prev] * Ui[h, h_prev])
                        + bi[h]);
    let g_forget = sigmoid(sum[i](x[t, b, i] * Wf[h, i])
                         + sum[h_prev](hidden[t-1, b, h_prev] * Uf[h, h_prev])
                         + bf[h]);
    let g_output = sigmoid(sum[i](x[t, b, i] * Wo[h, i])
                         + sum[h_prev](hidden[t-1, b, h_prev] * Uo[h, h_prev])
                         + bo[h]);
    let candidate = tanh(sum[i](x[t, b, i] * Wc[h, i])
                       + sum[h_prev](hidden[t-1, b, h_prev] * Uc[h, h_prev])
                       + bc[h]);

    let cell[t, b, h] = g_forget * cell[t-1, b, h] + g_input * candidate;
    g_output * tanh(cell[t, b, h])
};
```

The same pattern from the simple RNN—`h` for the current output unit, `h_prev` for the previous unit that is being scanned—appears four times, once per gate. Each gate has independent weights. The coordinate names make the roles visible: `Wi[h, i]` projects input to hidden, `Ui[h, h_prev]` projects previous hidden to current. The distinction between `h` and `h_prev` is not a comment. It is in the indices. A reader who sees `Ui[h_prev, h]` instead of `Ui[h, h_prev]` knows immediately that the recurrence matrix has been transposed—even though a square matrix produces the same output shape either way.

### Two Directions: Bidirectional Recurrence

Some sequences depend on both past and future context. A bidirectional recurrence runs two passes—one forward, one backward—over the same time coordinate:

```rust
let h_f[t in 1..T, b, hidden] = step_f(h_f[t-1, b, hidden], x[t, b, feature]);
let h_b[t in 0..T-1, b, hidden] = step_b(h_b[t+1, b, hidden], x[t, b, feature]);
let h[t, b, hidden * 2] = concat[hidden](h_f[t, b, hidden], h_b[t, b, hidden]);
```

The forward pass reads `t-1`. The backward pass reads `t+1`. Both are valid recurrences—the compiler accepts both because each body only references indices that are already computed given the enumeration order. The forward recurrence enumerates `t` from `1` to `T`; `t-1` is available. The backward recurrence enumerates from `T-1` down to `0`; `t+1` is available. The compiler infers the enumeration direction from the index offsets.

The key insight: forward and backward are the same syntactic construct. They differ only in what the body reads. A `for` loop obscures this symmetry by hardcoding direction into control flow. A recurrence makes direction a property of the index expression.

---

The next chapter is a hinge. We step back from the syntax and ask: what habits has this journey taught us, independent of any particular language?

A simple RNN hidden state recurrence—`h[t in 1..T, b, hidden] = tanh(sum[in](x[t, b, in] * Wxh[hidden, in]) + sum[h_prev](h[t-1, b, h_prev] * Whh[hidden, h_prev]))`—has exactly one backward reference: `h[t-1]`. The compiler verifies that every reference to `h` at time `t` reads only from earlier steps. If you try `let h[t in 0..T] = h[t+1] + 1`, the compiler rejects it—`t+1` is a forward reference, and the recurrence syntax enforces backward-only temporal flow. A colleague who writes `let x[t in 0..T] = if t == 0 { init } else { f(x[t-1]) }` is writing a valid recurrence in a single-branch style. It differs from the two-line form only in syntax: the base case and recursive case share a single declaration bracket, and the compiler still checks that the coordinate sets and backward references are sound.
