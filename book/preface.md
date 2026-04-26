---
layout: book
title: "Preface"
---

# Preface

Tensor programs often begin as equations and end as scattered machinery:
indexing in one API, reductions in another, loops in the host language, and
derivatives wrapped around a function boundary. Einlang starts from a different
question:

```text
What if the structures in the formula were source-language constructs?
```

This book answers that question gradually. We will not begin with the compiler.
We will begin with names, subscripts, reductions, recurrence equations, and
derivative requests. Each chapter asks the same questions:

- What names does this construct introduce?
- Which indices survive into the result?
- Which indices are consumed?
- What shape can the compiler infer?
- What dependency or derivative fact becomes visible?

This is a book about designing a notation. It treats programs as objects that
can be read by two audiences at once: a person and a compiler. The person wants
the program to resemble the equation they already understand. The compiler wants
the program to reveal enough structure to check shapes, infer ranges, schedule
recurrences, and lower derivative requests. Einlang's bet is that these goals
can reinforce each other.

Einlang is not presented here as a universal programming language. It has
functions, modules, blocks, `if`, `match`, comprehensions, and Python interop,
but its center is tensor computation. The language is strongest when the source
program can keep the formula's structure intact.

## How to Read a Program

A small example already contains the main discipline:

```rust
let C[i, j] = sum[k](A[i, k] * B[k, j]);
```

Read it in layers. First, `C` is a new binding. Second, `i` and `j` are output
indices, so they define the axes of `C`. Third, `k` is introduced by `sum[k]`,
so it is local to the reduction and disappears afterward. Fourth, the reads
`A[i, k]` and `B[k, j]` force the `k` dimension of `A` to match the `k`
dimension of `B`.

That little reading gives a shape rule, a scope rule, and a runtime strategy.
The shape rule says `C` has the outer dimensions of `A` and `B`. The scope rule
says `k` cannot be used outside the reduction. The runtime strategy can be a
loop nest, an einsum-like lowering, a matrix multiply fast path, or something
else. The source does not choose the schedule too early; it states the
structure.

This style of reading carries through the whole book.

## A Note on Syntax

The examples use current Einlang syntax. A few choices are worth keeping in
mind:

```rust
let C[i, j] = sum[k](A[i, k] * B[k, j]);
let tail[..batch] = sum[k](x[..batch, k]);
let dy_dx = @y / @x;
let data = python::numpy::load("x.npy") as [f32; 10, 20];
```

Ranges are end-exclusive by default:

```rust
let x[i in 0..5] = i;   // i = 0, 1, 2, 3, 4
```

Use `..=` for an inclusive end:

```rust
let x[i in 0..=5] = i;  // i = 0, 1, 2, 3, 4, 5
```

## How to Run Examples

Standalone files can be run from the repository root:

```bash
python3 -m einlang path/to/file.ein
```

Many book snippets are intentionally partial. Their purpose is to explain the
notation and the compiler-visible structure, not to serve as complete programs
with data loading and assertions.
