---
layout: book
title: "Chapter 14 · Firewood"
---

# Chapter 14 · Firewood

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

## The Panorama: One Name, Five Forms

You have now traveled through all four parts of this book. Part I gave you the primitives—naming, reducing, broadcasting. Part II composed them—functions, recurrences, gradients. Part III put them next to the notations you use every day. Part IV built the compiler that reads them.

Now stop. Don't turn the page yet. We are going to do something we haven't done in any previous chapter: look at the same program in five forms, simultaneously. Like a museum exhibit with a gem displayed under five different lights, each revealing a different facet.

The program is softmax. You first saw it in Chapter 3. It has traveled with you through every chapter since. It is the simplest non-trivial coordinate-aware function in the language—six lines, one coordinate parameter, one reduction. And in those six lines, the entire life cycle of a name is visible.

Here it is. Read it slowly. Then we will unfold it.

---

### Form 1: The Source

This is what you wrote. The ink on your screen. The letters you typed.

```
softmax[i class](logits[i class]) =
    exp(logits[i class] - max[class](logits[i class]))
    / sum[class](exp(logits[i class] - max[class](logits[i class])));
```

Six lines. Two reductions named `[class]`. Three occurrences of the index pattern `[i class]`. One coordinate parameter `class` in the function bracket.

You read this and you see: *softmax normalizes over `class`, preserves `i`.* The name `class` appears five times—in the function bracket, in every index pattern, in both reduction brackets. It is impossible to read this code and not know which coordinate is being normalized.

Now look at the positional equivalent:

```python
def softmax(logits, dim=-1):
    m = np.max(logits, axis=dim, keepdims=True)
    e = np.exp(logits - m)
    return e / np.sum(e, axis=dim, keepdims=True)
```

`dim=-1`. That's it. That's the only record of which coordinate is being normalized. If `logits` changes shape, `-1` silently normalizes over whatever coordinate happens to be last.

The Einlang source records the name `class` five times. The positional source records the position `dim=-1` once. The ratio of recorded intent is 5:1. And yet the Einlang version is the one that looks simpler—because the coordinate structure is in the brackets, not in the reader's head.

---

### Form 2: The IR Tree

The compiler reads your source. It strips the syntax—no more `=`, no more `/`, no more line breaks. What remains is a tree of parentheses. Every name you wrote is still a name.

```lisp
(define-fn softmax (coord-params class) (value-params logits)
  (type-params (i))
  (body
    (let-decl (output probs (i class))
      (/ (call exp
           (- (index logits (i class))
              (reduction max (class)
                (index logits (i class)))))
         (reduction sum (class)
           (call exp
             (- (index logits (i class))
                (reduction max (class)
                  (index logits (i class))))))))))
```

No information lost. The structure mirrors the source exactly. `(reduction max (class) ...)` — the reduction still names its axis. `(index logits (i class))` — the index list still names its coordinates. `(coord-params class)` — the coordinate parameter is still separate from the value parameter.

The IR is not a translation. It is a *restatement*. Your source code, spoken in the compiler's native tongue. The names are the same. The structure is the same. Only the punctuation has changed.

---

### Form 3: After Analysis

Chapter 13's analysis passes walk the IR tree. They ask: what is the range of `i`? Where does `class` come from? Do the reductions consume what they claim to consume? What is the output shape? What is the type?

When the passes finish, the tree carries answers:

```lisp
(define-fn softmax (coord-params class) (value-params logits)
  (type-params (i))
  ;; ── analysis annotations ──
  ;; i:     (range 0 batch_size)
  ;; class: (range 0 n_class), reduction axis
  ;; output shape: (batch_size n_class)
  ;; output type:  f64
  ;; five rules:
  ;;   Rule 1 ✓ — class exists on logits
  ;;   Rule 2 ✓ — class appears in all operands of both reductions
  ;;   Rule 3 ✓ — broadcast recorded: (reduction max ...) omits class,
  ;;              broadcasts back in subtraction
  ;;   Rule 4 N/A — no recurrence
  ;;   Rule 5 ✓ — coordinate contract satisfied
  (body
    (let-decl (output probs (i class))
      ;; surviving coordinates: {i, class}
      (/ (call exp ...) (reduction sum (class) ...)))))
```

Every question that can be answered from the names alone has been answered. Range, shape, type, coordinate contract—all derived from the names you wrote. The compiler did not need data. It did not need to run the program. It needed the names, and the names were there.

Notice something: the positional softmax passes zero of these checks. There is no `class` to verify. There is no coordinate contract to check. The positional compiler sees `axis=-1` and says: "valid integer." It does not say: "this integer refers to the coordinate you intend to normalize over." The name-based compiler asks that question. The positional compiler cannot ask it. Not because it is a worse compiler—because it has less information.

---

### Form 4: After Lowering

Now the names begin their transformation. Chapter 14's lowering pass maps every name to a number, every coordinate to a loop or a reduction axis.

```lisp
(define-fn softmax (coord-params class) (value-params logits)
  ;; ── lowering annotations ──
  ;; i     → axis=0, loop over 0..batch_size
  ;; class → axis=1, reduction
  ;; strategy: vectorized (all shapes known, no jagged arrays)
  ;; keepdims: required for (logits - max) broadcast
  (body
    (let-decl (output probs (i class))
      ;; loops: (i (range 0 batch_size))
      ;; reduction-axes: (class 1)
      ;; generated call: np.max(logits, axis=1, keepdims=True)
      (/ (call exp ...) (reduction sum (class) ...)))))
```

`class → axis=1`. The name becomes a number. But it becomes the right number—because the compiler mapped the declared coordinate order to integer positions. `i` is first → axis 0. `class` is second → axis 1. The mapping is deterministic. You can trace it by hand.

`keepdims: required` — this annotation is not in the source. The compiler deduced it. The subtraction `logits - max` requires the max to broadcast back over `class`. The analyzed shapes showed that `logits` has `(i class)` and `max` reduces `class` to produce `(i)`. The shapes differ. `keepdims=True` bridges the gap. The programmer didn't write `keepdims=True`. The compiler inferred it from the coordinate structure.

---

### Form 5: The Generated Code

The lowering annotations are complete. Every name has a number. Every reduction has an axis. Every broadcast has a `keepdims`. The code generator reads the annotations and emits NumPy:

```python
def softmax(logits):
    m = np.max(logits, axis=1, keepdims=True)
    e = np.exp(logits - m)
    return e / np.sum(e, axis=1, keepdims=True)
```

This is the same NumPy code a human would write. `axis=1` came from `class → axis 1`. `keepdims=True` came from the broadcast requirement the compiler deduced. The code is clean, vectorized, and identical to what you'd find in any well-written ML library.

But it was not written by a human. It was generated from a source program where `class` was a name and `max[class]` was an expression of intent. The human wrote the names. The compiler burned them into numbers.

---

### The Pause

Now look at all five forms together. Not sequentially—simultaneously. Let your eye move from the source `max[class]` to the IR `(reduction max (class) ...)` to the analysis annotation `class: (range 0 n_class), reduction axis` to the lowering annotation `class → axis=1` to the generated code `np.max(logits, axis=1, keepdims=True)`.

```
 max[class](logits[i class])                          ← what you wrote
 (reduction max (class) (index logits (i class)))     ← what the compiler sees
 class: (range 0 n_class), reduction axis, Rule 2 ✓   ← what the compiler derives
 class → axis=1, reduction, keepdims=True              ← how the name becomes a number
 np.max(logits, axis=1, keepdims=True)                 ← what executes
```

Five forms. One name. The name `class` traveled through all five without changing its identity. It was written as `class`, preserved as `class`, verified as `class`, mapped from `class` to `axis=1`. At no point was it guesswork. At no point did the compiler wonder "is axis 1 really the class dimension?" The information was in the source. The compiler carried it forward. The number at the end is the correct number because the name at the beginning was the correct name.

Now ask yourself: if the positional version had a bug—if `dim=-1` was normalizing over the wrong axis—at which of the five stages would that bug be caught?

Source: not caught. `dim=-1` is a valid integer.
IR: not caught. The IR for positional code has no names to verify.
Analysis: not caught. There is no coordinate contract to check.
Lowering: not caught. `-1` maps to the last axis correctly—it's the *choice* of the last axis that is wrong.
Generated code: not caught. `np.max(logits, axis=-1)` is valid NumPy.

The answer: **none of them.** The positional bug is invisible to all five stages because the information that would expose it—the name of the coordinate—was never written down.

The Einlang bug is caught at Form 3. Analysis checks Rule 2: does `class` appear in every operand of the reduction? If `class` doesn't exist on `logits`—if the tensor was declared with a different coordinate name—Rule 1 catches it. If the coordinate contract at the call site doesn't match the function declaration, Rule 5 catches it. The bug surfaces before the program runs, at the stage where names are still names and the compiler can still reason about them.

You have now seen the complete life of a name. It is written. It is preserved. It is verified. It is burned.

The name exists so that the number can be correct. The name is the guarantee. The number is the execution. You cannot have the second without the first—not if you want the number to mean what you think it means.

---

---

You wrote `class`. Five characters. They survived parsing, resolution, analysis, lowering—each stage asking a question that a number could not. At the end, they became `axis=1` and were burned. But the burn was correct because the name was verified. The name is gone from the execution but not from the source—and the source is where questions are asked.

Form 0 is the alternative: `dim=-1`, where no name enters the system and no question is asked. Compare the two: five keystrokes that enable five checks, or three keystrokes that enable none. The ratio is the distance between correct-by-construction and correct-by-coincidence.


Names are a good thing. They let you write `class` instead of `1`. They let the compiler check coordinate alignment—by name, not position. They survive the five check rules.

But names must eventually be burned. The CPU does not know `class`. It knows axis 1, loop range n_class, and floating-point instructions. Lowering is that fire. `(reduction sum (class) ...)` goes in; `axis=1, reduction` comes out. The backward pass reads the ash: every forward reduction becomes a backward broadcast, every forward silence becomes a backward sum. The names burn. The heat remains—a computation that can be handed directly to NumPy's C kernels.

A good abstraction is not a good painting. A good abstraction is good firewood. Its beauty is not in its surface—but in the light the flame casts when it burns.

---

The name is burned. `class` became `axis=1`. The word is not in the generated code. The CPU will never read it.

But the name is not lost. It is in the source code, where you can read it. It is in the compiler's analysis log, where the checks are recorded. It is in the error message, if a future refactoring breaks the contract. It is in the lowering annotations, documenting why `axis=1` was chosen.

The name burned. The heat remains. And the heat is a program whose axes are correct—not because the programmer memorized the dimension order, but because the compiler verified the coordinate names.

*Look at any `axis=1` in your generated or handwritten NumPy code. Trace it backward: what name did it come from? If you cannot answer—if the integer has no recoverable identity—you are looking at a number that was never a name. The fire burned nothing, because there was nothing to burn. That is the positional default.*

*You first wrote `class` in Chapter 3. It has now traveled through every chapter of this book—through primitives, combinations, comparisons, construction. It was written, preserved, verified, and burned. Close your eyes. Trace its journey in your mind. The path you trace is the coordinate habit.*
