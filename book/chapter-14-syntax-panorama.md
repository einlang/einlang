---
layout: book
title: "Chapter 14 · The Complete Picture"
---

# Chapter 14 · The Complete Picture

> "The whole is greater than the sum of its parts."
>
> — Aristotle

*Formalization · A systematic review of einlang syntax*

---

This chapter contains no new syntax. It is a map of territory we have already explored.

The preceding thirteen chapters introduced einlang's grammar piece by piece, each piece arriving when the concept it serves had earned its introduction. The result is a working knowledge of the language, but the pieces are scattered across chapters. This chapter assembles them into one place, organized by category rather than by pedagogical necessity.

Think of it as the view from the summit. You climbed the mountain one trail at a time. Now you can see the whole range.

---

## Declarations

**`let`** binds an immutable name to a value:

```rust
let x = 42;
let pi: f64 = 3.141592653589793;
let matrix: [f32; 2, 3] = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
```

Type annotations are optional. When present, the value must be compatible—same type, or a literal coercible to the annotated type.

All `let` bindings are immutable. A name cannot be rebound in the same scope:

```rust
let x = 10;
let x = x + 1;       // ERROR: redefinition of `x`
let x_next = x + 1;  // OK: new name
```

---

## Rectangular Declarations

A rectangular declaration binds a tensor by iterating over index variables:

```rust
let C[i, j] = sum[k](A[i, k] * B[k, j]);
```

The index variables on the left-hand side define the output tensor's coordinates. The compiler infers each variable's range by examining how it indexes arrays in the body. The index variables are in scope for the body and any attached `where` clause, but not outside the statement.

Index slots in the declaration bracket may be:
- A name: `i`, `j`, `batch`—the standard case.
- A name with an explicit domain: `i in 0..n`.
- A literal: `0`—used for base cases in recurrences.
- A named rest: `..batch`—standing for zero or more adjacent axes.

Expressions are not allowed in the declaration bracket. `let fib[n-1] = ...` is an error.

---

## Reductions

A reduction consumes a coordinate:

```rust
let total = sum[i](data[i]);
let row_sums[i] = sum[j](matrix[i, j]);
let col_sums[j] = sum[i](matrix[i, j]);
```

Available operations: `sum`, `max`, `min`, `prod`.

Selection reductions return addresses rather than values:

```rust
let pred[b] = argmax[class](logits[b, class]);
let route[b, t] = argmax[expert](gate_prob[b, t, expert]);
```

The selected coordinate is consumed from the result shape. The address domain is tracked by the compiler for later indexed reads.

---

## Broadcasting

Einlang supports broadcasting in exactly two cases:
- **Same rank**: two tensors with identical shapes can be combined elementwise.
- **Tensor vs. scalar**: a scalar can be combined with any tensor.

For different-rank combinations, use explicit indexing. The coordinate omitted from a term is the coordinate being broadcast over:

```rust
let out[i, j] = A[i, j] + bias[j];   // bias is broadcast over i
```

---

## Named Rest Indices

`..name` stands for zero or more adjacent axes, collectively given a name:

```rust
let result[..batch, j] = x[..batch, j] + bias[j];
let row_sum[..batch] = sum[j](x[..batch, j]);
```

The same rest name must describe the same axis span within an expression.

---

## Where Clauses

A where clause attaches filters and intermediate bindings to a declaration:

**Boolean guards** filter the domain:

```rust
let pos_sum = sum[i](data[i]) where data[i] > 0;
let upper[i, j] = matrix[i, j] where i <= j;
```

**Variable bindings** name intermediate values, scoped to the declaration:

```rust
let output[i, j] = activated
    where z = sum[k](input[i, k] * weight[k, j]) + bias[j],
          activated = if z > 0.0 { z } else { 0.0 };
```

Bindings are evaluated in order; later bindings can reference earlier ones.

**Index arithmetic** in where-clause index positions computes derived coordinates:

```rust
let conv[b, oc, oh, ow] = sum[ic, kh, kw](
    input[b, ic, oh + kh, ow + kw] * weight[oc, ic, kh, kw]
);
```

---

## Coordinate-Aware Functions

A function may declare coordinate parameters in brackets after its name:

```rust
fn softmax[j](x: [f32; ..left, j, ..right])
    -> [f32; ..left, j, ..right]
{ ... }
```

The caller passes coordinate arguments in the same bracket position:

```rust
let p[b, class] = softmax[class](logits[b, class]);
```

Coordinate arguments must be grounded in the argument's layout. The bracketed name is part of the call contract, not a comment.

Packs (`..left`, `..right`, `..spatial`) make functions polymorphic over surrounding coordinate structure. A pack parameter declared with `..` in the function signature accepts a parenthesized coordinate group from the caller when disambiguation is needed.

Coordinate facts flow through pointwise expressions automatically. A tensor bound with coordinate names carries those names through arithmetic, function calls, and control flow.

---

## Recurrence Relations

Self-referential rectangular declarations define sequences:

```rust
let fib[0] = 0;
let fib[1] = 1;
let fib[n in 2..8] = fib[n-1] + fib[n-2];
```

The recurrence index range goes in the declaration bracket (`n in 2..8`). Backward references only—the compiler rejects reads at indices greater than or equal to the declaration index along any dimension.

---

## Automatic Differentiation

`@x` is the identity tangent seed of a named binding. `@y / @x` is the derivative of `y` with respect to `x`:

```rust
let dz_dx = @z / @x;
```

For sum-of-products declarations (matmul, convolution), the compiler derives pullbacks automatically using coordinate set subtraction. The gradient of a recurrence is itself a recurrence, running backward along the time coordinate.

Custom derivative rules use `@fn`:

```rust
@fn relu(x) {
    if x > 0.0 { @x } else { 0.0 }
}
```

Coordinate-aware custom rules carry the same bracketed parameters as the primal function.

---

## Why the Compiler Reads Coordinates Too

The preceding sections catalogued syntax. But syntax is only half the story. The other half is who reads it.

The einlang compiler is not a single pass. It is a chain of readers, and each reader depends on coordinate names to do its job:

**Shape inference** reads coordinate names to decide whether an expression is legal before it runs. `sum[k](A[i, k] * B[k, j])` succeeds if `k` appears in both `A` and `B` and the non-reduced axes align. Without names, the check is integer matching: shape `[32, 64]` times shape `[64, 128]` is only legal if dimension 1 of the first equals dimension 0 of the second. Under names, the contract is: `i` survives from `A`, `j` survives from `B`, `k` appears in both and is consumed. The rule is the same rule a human uses to read the expression. The compiler becomes a reader of the same notation the programmer reads.

**Gradient lowering** reads coordinate names to build the backward pass. When the compiler sees `@loss / @W[out, in]`, it must decide which coordinates to sum over. The rule: preserve the coordinates of `W`, sum over everything else. That rule works because the forward pass already recorded which coordinates survive, which are consumed, and which are omitted. The gradient passer reads that record backward.

**Storage planning** reads coordinate names to decide which tensors can share memory. A recurrence `fib[n in 2..N] = fib[n-1] + fib[n-2]` creates a dependency along `n`. The compiler sees the index offset `n-1`, recognizes the backward edge, and allocates a rolling buffer—no materialization of the full sequence unless the full sequence is observed. The same logic applies to any recurrent coordinate: `t` in a time series, `layer` in a deep network unrolled as a recurrence.

**Kernel fusion** reads coordinate names to decide which operations can be merged. Two elementwise operations on `x[b, t, d]` can fuse into one pass. A reduction over `d` cannot fuse with an operation that preserves `d`, because the coordinate disappears. The names draw the boundary: operations that share surviving coordinates can fuse; operations across a reduction boundary cannot.

This is the design principle the compiler is built on. Every pass consumes a coordinate fact that the source makes explicit. No pass must reconstruct a coordinate fact from shapes, access patterns, or tracing. The source says it; the compiler reads it; the pass obeys it.

The practical consequence: because the compiler reads coordinate names and not positional shapes, the execution strategy can change while the coordinate contract remains stable. The same program can run on a NumPy eager backend, an IREE compiled backend, or a future scheduler that fuses the sum and broadcast into a single kernel. The backends differ. The coordinate audit is constant.

---

## Error Codes (Coordinate-Relevant Subset)

Among einlang's error codes, two are especially relevant to the coordinate habit:

- **E003 (Undefined Variable)**: a coordinate name is referenced but not declared. This catches typos in coordinate names that would otherwise silently refer to the wrong dimension.
- **E004 (Shape Mismatch)**: two uses of the same coordinate name infer incompatible ranges. This catches the case where `A[i, k] * B[k, j]` has mismatched `k` sizes.

These two errors are the compiler's contribution to the coordinate audit. They catch the bugs that positional APIs leave to runtime or to silence.

---

## What the Syntax Serves

A grammar is not an end in itself. The einlang syntax exists to serve the coordinate habit:

- **Eliminate with a name**: `sum[channel](...)` puts the eliminated coordinate in the syntax.
- **Copy with a signature**: `out[i, j] = A[i, j] + bias[j]` puts the broadcast coordinate in the indexing pattern.
- **Permute with a source**: `y[b, c, h, w] = x[b, h, w, c]` puts the permutation in the coordinate correspondence.
- **Forward and backward, symmetric**: `@loss / @W` preserves coordinate structure through differentiation.

The syntax is the delivery mechanism. The habit is the payload.

Without looking back at earlier chapters, try to write the einlang expression for: a matrix-vector product, a batched dot product, a softmax with numerical stability, a 1D convolution, a recurrent neural network cell. What did you remember? What did you reconstruct? The syntax has a small surface area—once you internalize the primitives, you can regenerate most of what you need from first principles. For each error you have seen in your own tensor programming, ask: would a named-coordinate compiler have caught this earlier? At what stage—parsing, shape inference, gradient lowering, runtime? The answer is almost always: earlier than where you caught it. This chapter is a map, not a reference manual. The syntax will evolve. The habit—write the coordinate names, make them explicit, let the compiler check them—will outlast any particular syntax.
