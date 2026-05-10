---
layout: book
title: "Chapter 13 · Firewood"
---

# Chapter 13 · Firewood

> "A ship in harbor is safe, but that is not what ships are built for."
>
> — John A. Shedd

*Construction · Lowering: names become numbers*

---

When Chapter 12 ended, your softmax looked like this:

```lisp
(let-decl (output probs (i class))
  (/ (call exp (- (index logits (i class))
                  (reduction max (class)
                    (index logits (i class)))))
     (reduction sum (class)
       (call exp (- (index logits (i class))
                    (reduction max (class)
                      (index logits (i class))))))))
;; type: f64, shape: (batch n_class)
;; i: (range 0 batch), class: (range 0 n_class)
;; five rules: all passed
```

The tree is complete. It knows everything it needs to know. But it cannot be handed to NumPy. NumPy does not understand what `class` means. It needs `axis=1`.

Translating the analyzed tree into instructions a machine can execute—this process is called **lowering**. This is the chapter where names are burned.

---

## Machines Don't Read Names

You wrote names. `i`, `class`, `sum`, `max`.

The machine needs integers. axis 0, axis 1. Loop ranges. Execution strategy.

```lisp
;; (i class) → output axes: axis=0 is i, axis=1 is class
;; (reduction sum (class) ...) → reduction on axis=1
;; (index logits (i class)) → read whole logits, both axes present
```

The translation rules are deterministic:
- Every axis name maps to its position in the declaration order.
- Every reduction axis maps to the integer position it occupies.
- Every index list maps to a slice or full-array access.

Lowering is the systematic translation of every name into a number.

---

## The Lowered Form

After lowering, the tree gains execution annotations:

```lisp
(let-decl (output probs (i class))
  (/ (call exp ...) (reduction sum (class) ...))
  (loops (i (range 0 batch)))
  (reduction-axes (class 1))
  (strategy vectorized))
```

`(loops ...)` — the output axes to iterate. `i` needs a loop. `class` is not here—it's consumed by the reduction.

`(reduction-axes ...)` — `class` maps to axis 1. Code generation turns this into `axis=1`.

`(strategy vectorized)` — chosen automatically by the compiler based on shapes.

---

## Three Strategies

After lowering, all axis ranges are known. The compiler chooses:

**vectorized** — hand everything to NumPy's C kernels:

```python
m = np.max(logits, axis=1, keepdims=True)
e = np.exp(logits - m)
return e / np.sum(e, axis=1, keepdims=True)
```

`class` → `axis=1`. `keepdims=True` because the analyzed shapes showed that `logits - max` requires broadcasting.

**scalar** — pure Python for loops. Used when arrays are jagged or when a dependency chain prevents vectorization:

```python
for i in range(batch):
    row_max = max(logits[i])
    row_exp = [exp(x - row_max) for x in logits[i]]
    row_sum = sum(row_exp)
    result[i] = [e / row_sum for e in row_exp]
```

**hybrid** — some dimensions use NumPy, others use Python loops.

The strategy is not chosen by the user. The compiler decides. You write the einstein clause. The compiler chooses how to run it.

---

## Matrix Multiplication: Read from Axis Sharing

This is where lowering shows its elegance:

```lisp
(let-decl (output C (i j))
  (reduction sum (k)
    (* (index A (i k)) (index B (k j)))))
```

Lowered:

```lisp
(let-decl (output C (i j)) ...
  (loops (i (range 0 n)) (j (range 0 m)))
  (reduction-axes (k 1))
  (strategy vectorized))
```

Code generation:

```python
C = A @ B
```

`k` appears in both A and B, but not in the output axes `(i j)` → a contracting dimension. The lowering recognizes this as matrix multiplication—no user annotation, no `np.einsum('ik,kj->ij', A, B)`. The axis sharing pattern **is** the expression of intent.

If the same clause uses jagged arrays, lowering selects scalar strategy instead. Same einstein clause, different execution path. The names are the same. The strategy is the compiler's decision.

---

## Lowering Multiple Clauses

Multi-clause declarations each receive their own annotations:

```lisp
(let-decl (output A (i j))
  (clause (=) (index B (i j)))
  (clause (+=) (+ (index A (i j)) (index C (i j)))))
```

After lowering:

```lisp
(let-decl (output A (i j))
  (clause (=) (index B (i j))
    (kind init)
    (loops (i (range 0 n)) (j (range 0 m))))
  (clause (+=) (+ (index A (i j)) (index C (i j)))
    (kind accum)
    (loops (i (range 0 n)) (j (range 0 m))))
  (strategy vectorized))
```

Generated code:

```python
result = B.copy()
result += C
```

`(kind init)` → `=`. `(kind accum)` → `+=`. The lowering preserves the distinction.

---

## Softmax End-to-End

From source to execution, one unbroken thread.

**Source** (Chapter 3):

```
softmax[i class](logits[i class]) =
    exp(logits[i class] - max[class](logits[i class]))
    / sum[class](exp(logits[i class] - max[class](logits[i class])));
```

**IR** (Chapter 11):

```lisp
(define-fn softmax (coord-params class) (value-params logits) ...)
```

All names present. The structure is the skeleton of the source.

**After analysis** (Chapter 12):

```
;; shape: (batch n_class), type: f64
;; i: (range 0 batch), class: (range 0 n_class)
;; five rules: all passed
```

All annotations present. Every name has a range and a type.

**After lowering** (this chapter):

```
;; class → axis=1, reduction
;; i → loop axis
;; strategy: vectorized
```

Names have become numbers and loop descriptions.

**Code**:

```python
m = np.max(logits, axis=1, keepdims=True)
e = np.exp(logits - m)
return e / np.sum(e, axis=1, keepdims=True)
```

`class`—a word you typed by hand—was lowered into `axis=1`. Not magic. Deterministic translation. You can see every step.

---

## Firewood

Names are a good thing. They let you write `class` instead of `1`. They let the compiler check coordinate alignment—by name, not position. They make your code read like mathematical notes. They survive the five check rules.

But names must eventually be burned. Because the CPU does not know `class`. It knows axis 1, loop range n_class, and floating-point instructions.

Lowering is that fire. `(reduction sum (class) ...)` goes in; `axis=1, reduction` comes out. `(i class)` goes in; `axis=0, axis=1, output` comes out. `(index logits (i class))` goes in; `keepdims=True` comes out—because the analyzed shapes tell lowering that the division and subtraction require broadcasting.

The names burn. The heat remains. And the heat is a computation that can be handed directly to NumPy's C kernels.

A good abstraction is not a good painting. A good abstraction is good firewood. Its beauty is not in its surface—but in the light the flame casts when it burns.

---

You have now seen the complete life cycle of a name in einlang: it is written in source, preserved in IR, verified by analysis, and burned in lowering. The next chapter looks back at that life cycle and asks: what did the names do for us—and what would have happened without them?
