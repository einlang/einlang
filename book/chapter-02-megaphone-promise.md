---
layout: book
title: "Chapter 2 · The Megaphone's Promise"
---

# Chapter 2 · The Megaphone's Promise

> "The absence of a signal is itself a signal."
>
> — Geoffrey Hinton (apocryphal)

*Primitives · Reduction and broadcasting*

---

A permutation moves coordinates around. A reduction makes one disappear. A broadcast makes one appear where it wasn't.

Reduction and broadcasting are inverses. They govern which coordinates a value depends on—and which it doesn't. This chapter introduces both, unified by a single intuition: **a tensor is a speaker that speaks on some coordinates and stays silent on others.** The ones it stays silent on, it gets copied. The ones it speaks on, it can be summed away.

We'll call this the megaphone model. Once you have it, you have the core of every tensor computation.

---

## The Megaphone

Imagine a tensor `bias[j]` as a person holding a megaphone. The megaphone is pointed at coordinate `j`. On `j`, the value speaks—`bias[0]` is one number, `bias[1]` is another, each position carries its own meaning. On every other coordinate—coordinates not in the bracket—the megaphone is silent.

What happens when silence meets a coordinate? The value gets copied. If you write:

```rust
let out[i, j] = A[i, j] + bias[j];
```

`bias` has no `i` in its brackets. It is silent on `i`. This silence is a declaration: "the value of `bias` does not depend on `i`. Whatever `i` you ask for, the answer is the same." So the value is copied across all 32 values of `i`—not because it saves keystrokes, but because the indexing pattern makes a semantic claim: `bias` is independent of batch identity.

This is broadcasting. Not a shape-compatibility hack. A semantic declaration: "this value does not depend on that coordinate." The claim is **statically verifiable**: every use of `bias` is traced, and if any context requires `bias` to vary with `i`, the omission is flagged. Broadcasting is a promise, and the promise is checked.

Now the inverse. If broadcasting is silence—staying quiet on a coordinate so you are copied along it—reduction is speaking: naming the coordinate you consume, marking what disappeared.

```rust
let total = sum[i](data[i]);
```

`sum[i]` picks up the megaphone and points it at `i`. "I am going to speak on `i`—by summing over it. After this line, `i` is consumed." The coordinate `i` appears in the reduction bracket and is absent from the result. `total` is a scalar.

Reduction and broadcasting are the same megaphone, pointed in opposite directions. Broadcasting says "I am silent on `i`—copy me." Reduction says "I am speaking on `i`—consume me."

---

## Rectangular Declarations

Before we can eliminate or broadcast a coordinate, we need to name the ones we're keeping. In einlang, you name coordinates with a **rectangular declaration**:

```rust
let doubled[i, j] = matrix[i, j] * 2.0;
```

The `let` binds a new, immutable tensor. The `[i, j]` on the left declares the output coordinates—the new tensor will have two dimensions, and we are naming them `i` and `j`. The `matrix[i, j]` on the right indexes the input tensor `matrix` by those same coordinates. It is inferred that `i` ranges from `0` to `matrix.shape[0]` and `j` from `0` to `matrix.shape[1]`.

This is not a loop. It is a declaration. You are stating a fact: "for all `i` and `j` in their respective domains, `doubled[i, j]` equals `matrix[i, j] * 2.0`." Iteration is handled automatically. You handle the meaning.

---

## Reduction

Now the main event. A reduction iterates over a coordinate and combines all the values along it using an associative operation. The coordinate appears in the reduction bracket—and then it is gone from the result:

```rust
let total = sum[i](data[i]);
```

`sum[i](...)` says: for every value of `i`, evaluate the body `data[i]`, and sum the results. The coordinate `i` is introduced by the `sum`, used in the body, and consumed by the reduction. It does not appear in `total`—`total` is a scalar.

The four reduction operations are `sum`, `max`, `min`, and `prod`. Each has an identity element: `sum` starts from `0`, `prod` from `1`, `max` from negative infinity, `min` from positive infinity.

A reduction can leave some coordinates intact—producing a tensor rather than a scalar:

```rust
let row_sums[i] = sum[j](matrix[i, j]);
let col_sums[j] = sum[i](matrix[i, j]);
```

These two lines produce the same output shape (a 1D tensor of length equal to the surviving coordinate). But they mean completely different things. `row_sums[i]` sums over columns, leaving rows. `col_sums[j]` sums over rows, leaving columns. The difference is entirely in the bracket after `sum`—one character, carrying the full semantic weight of the operation.

In a positional API, these would be `matrix.sum(dim=1)` and `matrix.sum(dim=0)`. The reader must remember which position is rows and which is columns. The code does not help.

---

## The Two-Column Ledger

A reduction is the most semantically loaded operation in tensor programming. Every reduction makes two claims: which coordinate is being *consumed*, and which coordinates are *surviving*. When you read a reduction, draw an imaginary line down the middle of the page:

| Survivors | Consumed |
|-----------|----------|
| Coordinates that appear in the result | Coordinates introduced in the reduction bracket |
| They keep their identity | They are gone from the output |
| They can be used by later operations | They exist only within the reduction body |

For `let row_sums[i] = sum[j](matrix[i, j]);`:

- **Survivors**: `i` (appears on the left-hand side)
- **Consumed**: `j` (introduced by `sum[j]`, gone)

Five steps for reading any reduction:

1. **Identify the operation**: `sum`, `max`, `min`, or `prod`—and which coordinates are in its bracket?
2. **Identify the survivors**: which coordinates appear on the left-hand side of the `let`?
3. **Identify the consumed**: which coordinates appear in the reduction bracket but not on the left?
4. **Verify alignment**: do the consumed coordinates index matching positions across all terms in the body?
5. **State the claim**: in one sentence, what does this reduction assert?

This takes five seconds. It catches the bug where `sum[class]` silently became `sum[batch]` after a refactoring.

---

## Matrix Multiplication

With rectangular declarations and reductions, we have enough machinery to write matrix multiplication:

```rust
let C[i, j] = sum[k](A[i, k] * B[k, j]);
```

![Coordinates flow through matmul: i and j survive, k is consumed](figures/matmul_coords.svg)

If you've taken a linear algebra course, this line needs no explanation. It is the mathematical definition, transcribed character for character:

$$C_{ij} = \sum_k A_{ik} B_{kj}$$

The coordinate `k` is introduced by `sum`, used to index both `A` and `B` (forcing their shared dimension to agree—this is checked), and consumed. The coordinates `i` and `j` survive into the result.

Notice what this line does *not* contain: no `transpose`, no `matmul` function name, no `@` operator. The line says exactly what it means. And if you get the coordinates wrong—if you write `B[j, k]` instead of `B[k, j]`—it is caught, because `k`'s range from `A` won't match `j`'s range from `B`.

---

## Broadcasting: The Explicit Omission

Broadcasting is the inverse of reduction. A reduction consumes a coordinate. A broadcast copies along one.

In einlang, broadcasting is not a shape-compatibility rule that triggers automatically when dimensions happen to align. It is a visible omission in the indexing pattern:

```rust
let out[i, j] = A[i, j] + bias[j];
```

The coordinate `i` appears on `A` and `out`, but not on `bias`. Its absence from `bias` is the notation for broadcasting: `bias` is indexed only by `j`, so it is replicated across all values of `i`. The code states the semantic claim directly—`bias` does not depend on `i`—and the claim is visible to both the reader and static analysis.

Compare this to the implicit version: `A + bias`. The shapes match. The broadcast happens. But *which coordinate was broadcast over*? You have to know the shapes to answer that. And if the shapes change upstream, the answer changes with them, silently.

This is the principle of explicit omission: **if a term is independent of a coordinate, the indexing should show it.** When the indexing shows it, the reader can audit it. When the indexing hides it, the reader must guess.

The verification follows a mechanical procedure. Take the output coordinate set. Subtract each operand's coordinate set. The difference for each operand is the set of coordinates that operand broadcasts over:

```
Output coordinates: {i, j}
A's coordinates:    {i, j}  → broadcasts over: {}     (no omission)
bias's coordinates: {j}     → broadcasts over: {i}    (omitted i)
```

This set subtraction is statically computable from the indexing patterns alone—no execution required. The brackets are read, the differences are computed, and every broadcast is verified consistent across all uses of the broadcast value. If one expression claims `bias` is independent of `i` and another expression requires it to vary with `i`, it is a coordinate contract violation—caught before a single value is computed.

---

## Named Rest: `..batch`

So far our coordinates have been single, explicit names. But real tensor code often needs to be polymorphic over how many batch dimensions there are. A normalization function shouldn't care whether the input is `(batch, feature)`, `(batch, time, feature)`, or `(batch, head, time, feature)`. It only cares that `feature` is the last dimension and everything else is batch-like.

Einlang provides **named rest indices** for this:

```rust
let result[..batch, j] = x[..batch, j] + bias[j];
let row_sum[..batch] = sum[j](x[..batch, j]);
```

The notation `..batch` stands for zero or more adjacent axes, collectively referred to as `batch`. The same rest name must describe the same axis span everywhere it appears within an expression. Which concrete axes `..batch` covers is inferred from the shape of `x`.

This is not a wildcard. It is a named group. The name `batch` carries semantic weight—it says "these leading dimensions are all batch-like, and the operation treats them uniformly." If upstream adds a `head` dimension between `batch` and `time`, `..batch` absorbs it automatically.

---

## The Where Clause

Sometimes a computation should only apply to a subset of coordinate values. In a positional API, you'd create a mask tensor, multiply, and hope the mask doesn't silently broadcast into the wrong dimension. In einlang, you attach a **where clause** directly to the declaration:

```rust
let pos_sum = sum[i](data[i]) where data[i] > 0;
```

The where clause is evaluated for each combination of the enclosing index variables. For reductions, elements where the guard is false are skipped—the reduction's identity element is used instead.

A where clause can also bind intermediate variables to avoid recomputation:

```rust
let output[i, j] = activated
    where z = sum[k](input[i, k] * weight[k, j]) + bias[j],
          activated = if z > 0.0 { z } else { 0.0 };
```

Without the where clause, you'd write the `sum[k](...)` expression twice. With the where clause, you name the shared subexpression `z` and refer to it in `activated`. The bindings are evaluated in order; later bindings can reference earlier ones.

The where clause is not bolted onto tensor operations. It is the natural extension of the idea that declarations state facts over coordinate domains. A where clause narrows the domain over which the fact holds.

---

## The Inversion Rule

Broadcast and reduction are inverses. What you broadcast over in the forward pass, you reduce over in the backward pass. `bias[j]` omits `i` in the forward direction—broadcast. The gradient `dbias[j] = sum[i](dy[i, j])` consumes `i` in the backward direction—reduction. The omitted coordinate and the consumed coordinate are the same coordinate.

This pairing catches more bugs than any other single rule in this book. If a broadcast is shape-correct but semantically wrong, the gradient will sum over the wrong coordinate. If a reduction consumes `class` but the broadcast was over `batch`, the shapes might still match—but the gradient will silently compute a different quantity.

In PyTorch, write `x = torch.randn(8, 10); b = torch.randn(10); y = x + b`. The bias `b` broadcasts over the first dimension—but nothing in the code says so. If `x` is transposed upstream to shape `(10, 8)`, `x + b` still runs, but now broadcasts over the *second* dimension silently. In einlang, `let out[b, c] = logits[b, c] + class_bias[c]` makes the broadcast visible: `class_bias` depends only on `c`, so the reader knows it is independent of `b`. The code says what it means.

---

We have now covered the four primitives of tensor computation: naming (a coordinate has an identity), permutation (coordinates move by name, not position), reduction (the consumed coordinate is named in the bracket), and broadcasting (the omitted coordinate is visible by its absence). They all flow from a single idea: **the megaphone.** A tensor speaks on some coordinates and stays silent on others. Reduction silences a coordinate. Broadcast copies along one the tensor was already silent on. Naming makes the coordinates audible. Permutation moves them without changing who speaks.

In the next chapter, we begin composing these primitives into functions—functions whose coordinate contracts are part of their type, statically checked at every call site.
