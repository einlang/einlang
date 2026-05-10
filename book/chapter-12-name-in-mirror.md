---
layout: book
title: "Chapter 12 · The Name in the Mirror"
---

# Chapter 12 · The Name in the Mirror

> "The purpose of analysis is not to add information, but to make explicit the information that was already implicit in the names."
>
> — The compiler writer's maxim

*Construction · Range, shape, type, and five check rules*

---

The IR tree from Chapter 11 is full of names—`i`, `class`, `batch`, `sum`. But names *said* and names *checked* are two different things.

Look at this tree:

```lisp
(let-decl (output C (i j))
  (reduction sum (k)
    (* (index A (i k)) (index B (k j)))))
```

The names have already said: `i` and `j` survive, `k` is consumed. But they haven't said:
- Where does `i` range? 0..100? 0..batch_size?
- What is the output shape?
- What are the types?
- Do the coordinate contracts match at every call site?

The compiler must answer all four before it can generate code. This chapter is about how it answers them—and about five check rules that turn names from documentation into verification.

---

## Range Analysis

RangeAnalysis walks the tree and finds a range source for every axis.

1. **Array shapes.** `(index logits (i class))`—`i` appears at logits' 0th position, so `i`'s range is logits' 0th dimension length.
2. **Literals.** If code says `A[i]` and `A` is defined with shape `(10)`, then `i`'s range is 0..10.
3. **Explicit declarations.** `(reduction sum (i) ... (domain 0..n))`—the range is given directly.

At the outermost node, every axis has an annotation:

```lisp
;; i     → (range 0 batch_size)
;; class → (range 0 num_classes)
```

---

## Shape Analysis

With ranges, shapes follow mechanically. The output shape is all declared axes minus reduction axes, each remaining axis taking its range length.

```lisp
(let-decl (output C (i j)) ...)
;; shape: (batch_size num_classes)
```

With a reduction:

```lisp
(reduction sum (class) body)
;; body has (batch n_class), reduction consumes class
;; output shape: (batch)
```

If the shapes don't match—if a clause declares `(class)` but the body accesses an array requiring `k`—the compiler reports it here. Not at runtime.

---

## Type Propagation

Types flow from leaves to root:

```lisp
(literal 3.14)   → f64
(literal 5)      → i64
(index A (i))    → A's element type
(+ f64 f64)      → f64
(call exp f64)   → f64
(reduction sum ... f64) → f64
```

Leaves know their types. Operators know input→output rules. Propagate upward. If `(+ f64 i64)` appears, the compiler promotes i64 to f64.

---

## The Five Check Rules

Analysis culminates in five verification rules. Each catches a class of bug that positional notation silently accepts.

### Rule 1: Index Existence

Every name in an `(index T (axes...))` must exist in `T`'s declared axis list.

```
// Correct
(index logits (batch class))  // logits has axes (batch class)

// Error
(index logits (batch time))   // logits has no time axis
```

**Catches**: renamed a dimension upstream but missed one index site. *If you have ever refactored a data pipeline and spent an afternoon wondering why one operation silently started reading the wrong axis, you have met this bug.*

### Rule 2: Reduction Consistency

The consumed coordinate must appear in every operand's index list within the reduction body.

```
// Correct
(reduction sum (k)
  (* (index A (i k)) (index B (k j))))  // k appears in both

// Error
(reduction sum (k)
  (* (index A (i k)) (index B (j q))))  // k missing from B
```

**Catches**: reduction bracket says `k` but one index reference uses `q` by mistake. *If you have ever written `sum[class]` but one operand's index used `batch` instead—the result had the right shape but the wrong coordinate was consumed—this rule would have caught it. Every chapter of this book has used the same word: consume. A reduction consumes a coordinate. A broadcast omits one so the backward pass can consume it. A gradient collects over what was consumed. This rule is where that thread lands: did you actually consume what you claimed to consume?*

### Rule 3: Broadcast Recording

For every term in an expression, compute its coordinate set. The output coordinate set minus the term's coordinate set = the coordinates that term broadcasts over. Record this. It is not an error check—it is the foundation of the backward pass. What was broadcast forward becomes reduced backward.

```lisp
;; (+ (index A (i j)) (index bias (j)))
;; A coordinates: {i, j}
;; bias coordinates: {j}
;; bias broadcasts over: {i}  ← recorded for gradient
```

*If you have ever traced a gradient that was silently summed over the wrong dimension because a forward broadcast was implicit, you know why this record must exist before the backward pass runs.*
```

### Rule 4: Causality Verification

In a recurrence body, every reference to the time variable must be strictly less than the declared index.

```
// Correct
(recurrence u (index-var t) ...
  (index u ((- t 1) i)))  // t-1 < t, valid

// Error
(recurrence u (index-var t) ...
  (index u ((+ t 1) i)))  // t+1 > t, invalid
```

**Catches**: accidentally wrote `t+1` instead of `t-1` in a recurrence. *If you have ever implemented an RNN and watched the loss diverge because a forward reference read uninitialized memory instead of the previous hidden state, this rule catches that bug at compile time.*

### Rule 5: Coordinate Contract at Call Sites

When calling a coordinate-aware function, the coordinate arguments must exist on the value arguments, and the coordinate layout must match the function's declaration.

```
// fn softmax[j](x: [f32; ..left, j, ..right]) -> ...

// Correct
(call softmax (index logits (batch class)) (coord-args class))
// logits has class, softmax expects j ← class

// Error
(call softmax (index logits (batch class)) (coord-args nonexistent))
// logits has no nonexistent axis
```

**Catches**: `softmax[nonexistent](logits)`—a one-character typo caught at compile time. *If you have ever written `softmax(logits, dim=2)` and spent thirty minutes trying to understand why the output wasn't normalized along the dimension you intended, this rule would have told you immediately.*

---

## A Filled-In Tree

Before analysis, the tree is bare parentheses with names:

```lisp
(define-fn softmax (coord-params j) (value-params x)
  (type-params (..left) (..right))
  (body
    (let-decl (output e (..left j ..right))
      (call exp (- (index x (..left j ..right))
                   (reduction max (j)
                     (index x (..left j ..right))))))
    ...))
```

After analysis:

```lisp
(define-fn softmax (coord-params j) (value-params x)
  (type-params (..left) (..right))
  ;; j: (range 0 n_class), type: f64
  ;; output shape: (..batch_size n_class ..rest)
  ;; coordinate layout: (..left j ..right)
  ;; five rules: all passed
  (body ...))
```

Every question mark is gone. Every parenthesis is annotated with the answers the compiler derived. Range, shape, type, coordinate layout—they flowed out of the names, and now sit beside the names.

---

Names are not a convenience. They are information the compiler can reason about. You wrote `(i class)` and `(sum (class) ...)`. From that, the compiler derived shape, range, type, and coordinate layout—and verified five rules that catch the bugs positional notation leaves to comments.

Throughout this book, a single word has threaded through the chapters: **consume**. A reduction consumes a coordinate—the bracket names what disappears. A broadcast omits a coordinate—the omission records what the value is independent of. A gradient collects over what was broadcast and broadcasts over what was consumed. The five check rules, taken together, are the compiler's answer to the question: *did you actually consume what you claimed to consume?* Rule 2 checks it directly—the consumed coordinate must appear in every operand's index list. Rule 3 records what was broadcast so the gradient can consume it. Rule 4 checks that time steps only consume the past. The word "consume" is not decorative. It is the verb that links the forward pass to the backward pass, the reduction bracket to the gradient sum, the where clause to the filtered backward.

The tree is now complete. Every slot that needed an answer has one. It can be safely handed to the next chapter—where the names are burned.
