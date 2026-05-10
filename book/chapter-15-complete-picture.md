---
layout: book
title: "Chapter 15 · The Complete Picture"
---

# Chapter 15 · The Complete Picture

> "The purpose of computing is insight, not numbers."
>
> — Richard Hamming

*Reference · Thought map and syntax at a glance*

---

This chapter contains no new syntax. It is a map of territory already explored.

The preceding fourteen chapters introduced einlang's grammar piece by piece, each piece arriving when the concept it served had earned its introduction. The result is a working knowledge of the language, but the pieces are scattered across chapters. This chapter assembles them into one place, organized by category rather than by pedagogical necessity.

Think of it as the view from the summit. You climbed the mountain one trail at a time. Now you can see the whole range.

---

## The Thought Map

Before the syntax reference, a map of how the ideas connect. Each arrow is a dependency: the idea at the tail must be understood before the idea at the head.

```
dim=1 bug (Prologue)
    │
    ▼
A coordinate has a name, a domain, a position (Ch1)
    │
    ├──► Permutation: names survive position changes (Ch1)
    │
    ├──► Reduction: the consumed coordinate is named (Ch2)
    │       │
    │       └──► Broadcasting: the omitted coordinate is visible (Ch2)
    │               │
    │               └──► Inversion Rule: broadcast ↔ reduction dual (Ch2, Ch7)
    │
    ├──► Coordinate-aware functions: names as type-level contracts (Ch3)
    │       │
    │       ├──► Square Matrix Test: when extents equal, only names differ (Ch3)
    │       │
    │       ├──► Pack polymorphism: ..batch absorbs unknown leading dims (Ch4)
    │       │
    │       └──► Normalization skeleton: one pattern, four functions (Ch4)
    │
    ├──► Recurrence: time as a directional coordinate (Ch5)
    │       │
    │       └──► Causality constraint: t-1 valid, t+1 rejected (Ch5)
    │
    ├──► Complex terrain: splits, arithmetic, disambiguation (Ch6)
    │
    ├──► Differentiation: the pullback reads the forward pass backward (Ch7)
    │       │
    │       └──► @fn: custom derivative rules carry coordinate contracts (Ch7)
    │
    ├──► Comparisons: same computation, two notations (Ch8–10)
    │       │
    │       ├──► Normalization: GroupNorm reshape chain vs named groups (Ch8)
    │       ├──► Attention: identical PyTorch, distinct einlang signatures (Ch9)
    │       └──► Physics: integer field indices vs named field coordinates (Ch10)
    │
    └──► Compiler construction (Ch11–13)
            │
            ├──► IR: S-expressions preserve every name (Ch11)
            │
            ├──► Analysis: range → shape → type, five check rules (Ch12)
            │
            └──► Lowering: names → integers, three strategies (Ch13)
                    │
                    └──► Firewood: names burn, heat remains (Ch13)
```

Every path begins at the `dim=1` bug. Every arrow is a question the bug forced us to ask. The map is not the territory—but it shows how the trails connect.

---

## Declarations

**`let`** binds an immutable name to a value. *Introduced in Chapter 1.*

```rust
let x = 42;
let pi: f64 = 3.141592653589793;
let matrix: [f32; 2, 3] = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
```

Type annotations are optional. When present, the value must be compatible. All `let` bindings are immutable.

---

## Rectangular Declarations

A rectangular declaration binds a tensor by naming its coordinates. *Introduced in Chapter 1; extended with domains in Chapter 5.*

```rust
let C[i, j] = sum[k](A[i, k] * B[k, j]);
```

Index slots in the declaration bracket may be:
- A name: `i`, `j`, `batch` — the standard case.
- A name with an explicit domain: `t in 0..T` — for recurrences.
- A literal: `0` — used for base cases.
- A named rest: `..batch` — absorbs zero or more adjacent axes.

Expressions are not allowed in the declaration bracket. `let fib[n-1] = ...` is an error. The left side names what is being defined. The right side computes it.

---

## Reductions

A reduction consumes a coordinate. *Introduced in Chapter 2; selection reductions in Chapter 4.*

Operations: `sum`, `max`, `min`, `prod`.

```rust
let total = sum[i](data[i]);
let row_sums[i] = sum[j](matrix[i, j]);
```

Selection reductions return addresses rather than values:

```rust
let pred[b] = argmax[class](logits[b, class]);
```

The consumed coordinate is eliminated from the result shape. The reduction bracket names it explicitly—the reader does not need to infer which coordinate disappeared.

---

## Broadcasting

Broadcasting is an omission in the indexing pattern. *Introduced in Chapter 2; self-audit in Chapter 7.*

```rust
let out[i, j] = A[i, j] + bias[j];   // bias omits i → broadcast over i
```

The omitted coordinate is the one being broadcast over. The megaphone model: `bias` is silent on `i`, so the compiler copies it across all values of `i`. The silence is a semantic claim: `bias` does not depend on `i`.

The Inversion Rule: what broadcasts in the forward pass is reduced in the backward pass. `bias[j]` omits `i` forward → `d_bias[j] = sum[i](d_out[i, j])` backward.

---

## Named Rest Indices

`..name` stands for zero or more adjacent axes, collectively named. *Introduced in Chapter 2; pack polymorphism in Chapter 4.*

```rust
let result[..batch, j] = x[..batch, j] + bias[j];
let row_sum[..batch] = sum[j](x[..batch, j]);
```

The same rest name must describe the same axis span within an expression. Packs make functions rank-polymorphic: the same `layer_norm[feature]` works on 2D, 3D, or 4D inputs.

---

## Where Clauses

A where clause filters or binds. *Introduced in Chapter 2; backward behavior in Chapter 7.*

Boolean guards narrow the domain:

```rust
let pos_sum = sum[i](data[i]) where data[i] > 0;
let upper[i, j] = matrix[i, j] where i <= j;
```

Variable bindings name intermediate values:

```rust
let output[i, j] = activated
    where z = sum[k](input[i, k] * weight[k, j]) + bias[j],
          activated = if z > 0.0 { z } else { 0.0 };
```

In the backward pass, filtered elements receive zero gradient. The domain constraint applies symmetrically in both directions.

---

## Coordinate-Aware Functions

A function may declare coordinate parameters. *Introduced in Chapter 3; pack parameters in Chapter 4.*

```rust
fn softmax[j](x: [f32; ..left, j, ..right])
    -> [f32; ..left, j, ..right]
{ ... }
```

Call sites pass coordinate arguments in the bracket position:

```rust
let p[b, class] = softmax[class](logits[b, class]);
```

The compiler checks that `class` exists on `logits` and that the coordinate contract is satisfied. The bracketed name is part of the call contract, not a comment.

Packs (`..left`, `..right`, `..spatial`) make functions polymorphic over surrounding structure. A caller disambiguates by grouping: `softmax[(height, width)](x)`.

---

## Recurrence Relations

Self-referential declarations define sequences over time. *Introduced in Chapter 5.*

```rust
let u[t in 0..T, i] = initial[i];
let u[t in 1..T, i] = u[t-1, i] + f(u[t-1, i]);
```

Backward references only. `u[t+1, i]` on the right-hand side with declaration index `t` is a compile error. Causality is a syntactic constraint, not a convention.

The optimizer is a recurrence:

```rust
let w[t in 1..T, out, in] = w[t-1, out, in] - lr * grad[t-1, out, in];
```

---

## Automatic Differentiation

`@loss / @W` computes the gradient. *Introduced in Chapter 7.*

```rust
let dW = @loss / @W;
```

The gradient has the same shape as the denominator. The pullback is computed by reversing the forward graph: every forward reduction becomes a backward broadcast; every forward broadcast becomes a backward reduction. The shopping cart record, read in reverse.

Custom rules use `@fn`:

```rust
@fn relu(x) {
    if x > 0.0 { @x } else { 0.0 }
}
```

Coordinate-aware custom rules carry the same bracketed parameters as the primal function.

---

## Why the Compiler Reads Coordinates Too

The preceding sections catalogued syntax. But syntax is only half the story. Each compiler pass depends on coordinate names to do its job. *These passes are described in Chapters 11–13.*

**Shape inference** (Ch11–12) reads coordinate names to decide whether an expression is legal before it runs. `sum[k](A[i, k] * B[k, j])` succeeds if `k` appears in both `A` and `B`. Under names, the contract is: `i` survives from `A`, `j` survives from `B`, `k` appears in both and is consumed.

**Range analysis** (Ch12) finds the domain of every axis: from array shapes, from literals, or from explicit declarations. Every coordinate gets a concrete range before code generation.

**Five check rules** (Ch12) verify the IR: index existence, reduction consistency, broadcast recording, causality, and coordinate contract at call sites. Each catches a class of bug that positional notation silently accepts.

**Gradient lowering** (Ch13) reads coordinate names to build the backward pass. The rule: preserve the coordinates of `W`, sum over everything else. Set subtraction, applied to coordinate names, derives the pullback.

**Storage planning** (Ch13) reads coordinate names to decide which tensors can share memory. A recurrence creates a dependency chain; the compiler allocates a rolling buffer.

**Kernel fusion** (Ch13) reads coordinate names to decide which operations can be merged. Operations that share surviving coordinates can fuse; operations across a reduction boundary cannot.

---

## Error Codes

Two errors are especially relevant to the coordinate habit:

- **E003 (Undefined Coordinate)**: a coordinate name is referenced but does not exist on the tensor. `softmax[nonexistent](logits)` — caught at the call site.
- **E004 (Coordinate Range Mismatch)**: two uses of the same coordinate name infer incompatible ranges. `A[i, k] * B[k, j]` where `k` has range 64 in `A` but 128 in `B`.

These errors catch the bugs that positional APIs leave to runtime or to silence.

---

## One Table: The Coordinate Audit

Every tensor operation can be audited with four questions. They are not einlang-specific. They work in any framework because they are questions about meaning, not syntax.

| Question | What it catches | Chapter |
|---|---|---|
| Which coordinate is consumed? | Reduction over wrong axis | 2, 8 |
| Which coordinate is copied along? | Broadcast over wrong axis | 2, 7 |
| Can you trace a coordinate from source to destination? | Silent permutation/transpose | 1, 6, 9 |
| Does the backward reduction match the forward broadcast? | Gradient shape mismatch | 7 |

Ask these four questions of any tensor line. The answers tell you whether the notation preserved the facts that correctness depends on.

---

The syntax has a small surface area. Once you internalize the primitives—naming, reducing, broadcasting, recurring, differentiating—you can regenerate most of what you need from first principles. The thought map above shows how they connect. The syntax reference records what they are.

The syntax will evolve. The thought map will grow. The habit—write the coordinate names, make the omissions explicit, let the compiler check the contracts—will outlast any particular syntax.

Turn the page.
