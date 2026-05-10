---
layout: book
title: "Chapter 11 · The Shape of Thought"
---

# Chapter 11 · The Shape of Thought

> "A compiler is a program that reads a program written in one language—and says it again in another."
>
> — Adapted from SICP

*Construction · Intermediate representation*

---

A compiler cannot operate on source code strings directly. It needs an internal language. This language must satisfy two conditions. First, it must be able to express every structure in the source language with perfect fidelity—no information lost in translation. Second, it must be simple enough that checker passes can traverse it mechanically, without parsing or disambiguation.

We use S-expressions—parenthesized lists—as this internal language. In this language, coordinate names are preserved exactly as written. They are not pre-resolved to integers. Model names remain model names. The structure mirrors the source, stripped of syntactic sugar, with all semantic relationships intact.

The compiler's first job is translation: read einlang, write parentheses. Here is what that translation looks like.

---

You just wrote a short einlang program.

```
let x = softmax[i class](logits[i class]);
let y = sum[class](x[i class] * labels[i class]);
```

Someone asks: what is the shape of `y`?

You cannot run this code to find out—you have no data. But you need to know, because if shapes don't match, you want to catch it before training runs for an hour.

Pick up a pen. Trace: `softmax[i class]` produces something of shape `[batch, class]`. Its 0th dimension is named `i`, its 1st is named `class`. Then `sum[class]` reduces along the `class` axis, eliminating dimension 1. What remains is `[batch]`.

You just did by hand what a compiler does automatically. The compiler needs a way to write your program in a form that can be traced. A form stripped of syntax—only the relationships preserved. That form is the **intermediate representation**.

---

## What You Write, and What the Compiler Sees

```
A[i, j] = B[i, k] * C[k, j]
```

After parsing, the compiler sees a tree:

```lisp
(let-decl (output A (i j))
  (* (index B (i k)) (index C (k j))))
```

`A[i, j]` becomes `(let-decl (output A (i j)) ...)`. `B[i, k]` becomes `(index B (i k))`. `*` becomes `(*)`.

One-to-one. On the left is source code; on the right, parentheses. Commas, equals signs, brackets—all gone. But every relationship remains. `i` and `j` are names. `k` is a name. Nothing has been turned into a number.

The compiler, at this stage, does not speak in numbers. It speaks in names. Because names carry information—numbers come later.

---

## Reduction

```
sum[i](A[i] * B[i])
```

Becomes:

```lisp
(reduction sum (i)
  (* (index A (i)) (index B (i))))
```

`sum[i](...)` becomes `(reduction sum (i) ...)`. The bracket holds the name of the reduction axis; the body sits one level below. The structure mirrors what you write in mathematical notation: Σ, then the index variable, then the expression.

Reduction with a where clause:

```
sum[i](A[i] * B[i]) where i > 5
```

```lisp
(reduction sum (i)
  (* (index A (i)) (index B (i)))
  (where (> i (literal 5))))
```

`where` becomes the third child of `reduction`. It filters the reduction variable `i`—each iteration accumulates only elements that satisfy the condition. Not a post-hoc mask. A gate on the reduction itself.

---

## Broadcasting

```
A[i, j] + bias[j]
```

Becomes:

```lisp
(+ (index A (i j)) (index bias (j)))
```

`bias` is indexed only by `j`. `A` is indexed by `i` and `j`. The coordinate `i` is absent from `bias`'s index list. The compiler records this absence—not as a shape fact, but as a coordinate fact: `bias` declares independence from `i`. Broadcasting is not a shape-compatibility rule. It is a coordinate omission, visible in the index list.

---

## Multiple Clauses

Einlang allows multiple clauses for the same tensor:

```
A[i, j] = B[i, j];
A[i, j] += C[i, j];
```

Underneath:

```lisp
(let-decl (output A (i j))
  (clause (=) (index B (i j)))
  (clause (+=) (+ (index A (i j)) (index C (i j)))))
```

One `let-decl` holds two clauses. They share axis names—`i` and `j` mean the same thing in both. `(=)` says "initialize." `(+=)` says "accumulate." The IR makes the relationship explicit.

---

## Coordinate-Aware Functions

```
fn softmax[j](x: [f32; ..left, j, ..right])
    -> [f32; ..left, j, ..right] { ... }
```

```lisp
(define-fn softmax (coord-params j) (value-params x)
  (type-params (..left) (..right))
  (body ...))
```

The coordinate parameter `j` is a child of `define-fn`. The packs `..left` and `..right` are type parameters—they represent coordinate structure, not values.

At the call site:

```
softmax[class](logits)
```

```lisp
(call softmax (index logits (..batch class))
  (coord-args class))
```

The coordinate argument `class` is transmitted separately from the value argument `logits`. The compiler later verifies that `class` exists on `logits` and that the coordinate contract of `softmax` is satisfied.

---

## Recurrence

```
let u[t in 0..T, i] = initial[i];
let u[t in 1..T, i] = u[t-1, i] + f(u[t-1, i]);
```

```lisp
(recurrence u (index-var t) (var i)
  (clause (domain 0) (index initial (i)))
  (clause (domain 1 T)
    (+ (index u ((- t 1) i))
       (call f (index u ((- t 1) i))))))
```

The time axis `t` is marked as `index-var`. Its domain is split: `(domain 0)` for the initial condition, `(domain 1 T)` for the recurrence body. The backward reference `t-1` becomes `(- t 1)` in the index position. The compiler sees the subtraction and marks `t` as a time axis—a coordinate with a direction.

---

## Gradient

```
@loss / @w
```

```lisp
(gradient (numerator loss) (denominator w))
```

The gradient node records what it differentiates. The compiler computes the pullback by reversing the forward graph—every forward reduction becomes a backward broadcast, every forward broadcast becomes a backward reduction. The shopping cart record, read in reverse.

---

## A Complete Program

Put it all together:

```
let data = random([batch, class]);
let result = softmax[class](data);
```

```lisp
(program
  (let-decl (output data (batch class))
    (call random (literal (batch class))))
  (let-decl (output result (batch class))
    (call softmax (index data (batch class))
      (coord-args class))))
```

`program` is the outermost node—the container for the entire compilation unit. Inside are `let-decl` trees. Everything is in parentheses. There are no exceptions.

---

## What the IR Preserves

Read the softmax IR tree and notice three things.

1. **Names are everywhere.** `i` and `class` remain names from start to finish. Nothing has become axis=0, axis=1. The IR preserves names.

2. **Reductions name their axis.** `(reduction max (class) ...)` and `(reduction sum (class) ...)`—the reduction operates on `class`, not "axis 1."

3. **Array access uses names.** `(index logits (i class))`—the indices are `i` and `class`, matching what the source wrote.

This tree is the skeleton of your program. No syntactic sugar—no colons, no equals signs, no commas. But every semantic relationship is present. The names you wrote are still names. The coordinates you named are still coordinates. The IR has not *translated* your program. It has *said it again*, in parentheses.

---

The compiler's native tongue does not perform magic. It simply restates your einlang program in a different notation—syntactic sugar removed, core information preserved. Axis names remain. Reduction names remain. Clause relationships remain. The compiler has not *translated* your program. It has *said it again*, in parentheses.

Now the tree exists. But the names on the tree are still just names—they carry no range, no shape, no type. The next chapter asks: what can the compiler *derive* from those names, without running the program?
