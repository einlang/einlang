---
layout: book
title: "Chapter 12 · The Shape of Thought"
---

# Chapter 12 · The Shape of Thought

> "A compiler is a program that reads a program written in one language—and says it again in another."
>
> — Adapted from SICP

*Construction · Intermediate representation*

---

A compiler cannot operate on source code strings directly. It needs an internal language. This language must satisfy two conditions. First, it must be able to express every structure in the source language with perfect fidelity—no information lost in translation. Second, it must be simple enough that checker passes can traverse it mechanically, without parsing or disambiguation.

We use S-expressions—parenthesized lists—as this internal language. In this language, coordinate names are preserved exactly as written. They are not pre-resolved to integers. Model names remain model names. The structure mirrors the source, stripped of syntactic sugar, with all semantic relationships intact.

The compiler's first job is translation: read Einlang, write parentheses. Here is what that translation looks like.

---

You just wrote a short Einlang program.

```
let x = softmax[class](logits[batch, class]);
let y = sum[class](x[batch, class] * labels[batch, class]);
```

Someone asks: what is the shape of `y`? You cannot run this code—there is no data. But you can still answer. How?

Stop and actually do it. Trace the coordinates in your head. `logits` has `[batch, class]`. `softmax[class]` preserves the shape—output is still `[batch, class]`, bound to `x`. Then `sum[class]` consumes `class`. The surviving coordinate is `batch`. `y` has shape `[batch]`.

You just performed coordinate propagation in your head. You tracked each name—where it was introduced, how it flowed through the function call, whether it survived the reduction. You did not need data. You did not need to run the program. You needed only the names.

Now ask: what exactly did your brain do? It maintained a mental table:

```
logits:  coordinates {batch, class}
x:       coordinates {batch, class}   ← softmax preserves
labels:  coordinates {batch, class}
y:       coordinates {batch}           ← sum consumes class
```

At each step, you looked at the operation and asked: which coordinates go in, which come out? The answer was always in the brackets. `softmax[class]` says `class` is the normalized coordinate—preserved. `sum[class]` says `class` is the reduced coordinate—consumed.

You did not think about positions. You did not count axes. You did not check shapes. You checked names. And you got the answer instantly.

This—exactly this—is what a compiler must do. It must answer the same questions you just answered: what is the shape of every tensor? Which coordinates survive each operation? Does the coordinate contract at each call site match the function's declaration? The compiler needs to do this for every line of every program, without data, without execution. It needs an internal form in which these questions have mechanical answers.

That form is the **intermediate representation**. And the first decision in designing it is the most important one: **names must survive translation.** If the IR replaced `class` with `axis=1`, the compiler would have to reconstruct the identity you just effortlessly traced. Reconstruction is brittle—it depends on conventions, on consistent naming, on the very human discipline that positional notation abandons. Preservation is mechanical—if the name is in the IR, the compiler can check it. If the name is gone, it cannot.

A compiler cannot operate on source code strings directly. It needs an internal language. This language must satisfy two conditions. First, it must express every structure in the source language with perfect fidelity—no information lost in translation. Second, it must be simple enough that checker passes can traverse it mechanically, without parsing or disambiguation.

We use S-expressions—parenthesized lists—as this internal language. In this language, coordinate names are preserved exactly as written. They are not pre-resolved to integers. Model names remain model names. The structure mirrors the source, stripped of syntactic sugar, with all semantic relationships intact.

The compiler's first job is translation: read Einlang, write parentheses. Here is what that translation looks like.

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

## Why Parentheses?

Why S-expressions? Why not a JSON AST? Why not protobufs?

Because parentheses are homoiconic: the IR is both the data structure the compiler manipulates and the text format a human can read. A compiler pass can traverse the tree with `car` and `cdr`. A programmer can read the same tree in a debug log. There is no gap between the internal representation and its printed form.

Because parentheses are uniform. Every node is `(operator children...)`. There are no special cases. `(reduction sum (class) ...)` looks like `(let-decl (output C (i j)) ...)` looks like `(+ (index A (i)) (index B (i)))`. The uniformity makes passes easy to write: every pass is a tree walk with a case for each operator. Adding a new operator means adding one case.

Because parentheses have no ambiguity. In `x[i, j]`, the comma and brackets are syntax—they must be parsed, their precedence resolved against other operators. In `(index x (i j))`, the structure is explicit: `index` is the operator, `x` is the tensor, `(i j)` is the index list. No precedence to resolve. No grammar to extend. The parentheses *are* the parse tree.

The IR's job is to be the simplest possible form that preserves all source-level information. Parenthesized prefix notation is that form. It has been that form since Lisp discovered it in 1958. The Einlang IR does not innovate on representation. It inherits.

---

The compiler's native tongue does not perform magic. It simply restates your Einlang program in a different notation—syntactic sugar removed, core information preserved. Axis names remain. Reduction names remain. Clause relationships remain. The compiler has not *translated* your program. It has *said it again*, in parentheses.


---

*Take a single line of your own tensor code. Translate it into S-expressions by hand—parenthesize every operation, preserve every name. What information survived? What was lost? The IR is not magic. It is your program, said again, in a form that a machine can query.*

Now the tree exists. But the names on the tree are still just names—they carry no range, no shape, no type. The next chapter asks: what can the compiler *derive* from those names, without running the program?

---

## You Are the Compiler

Before we hand the tree to Chapter 13's analysis passes, let's do one ourselves. Here is a small Einlang program:

```
let result[i, j] = sum[k](A[i, k] * B[k, j]) + bias[j];
```

Translate it into IR. Don't look at the answer below. Actually do it. Write it in parentheses.

---

Done? Compare:

```lisp
(let-decl (output result (i j))
  (+ (reduction sum (k)
       (* (index A (i k)) (index B (k j))))
     (index bias (j))))
```

Every construct becomes a parenthesized form. `sum[k]` → `(reduction sum (k) ...)`. `A[i, k]` → `(index A (i k))`. `+` → `(+)`. The `let` declaration wraps the entire expression.

Now ask yourself: what can you determine about this program just by reading the parentheses? No runtime. No data. Only the tree.

You can determine:
- `result` has coordinates `(i, j)`. Those are the survivor names in `let-decl`.
- `k` is consumed inside the reduction. `sum` eliminates it.
- `bias` is indexed only by `j`. It omits `i`—a broadcast.
- `A` and `B` share `k`. The multiplication inside the sum operates on matching `k` positions.
- The addition of `bias[j]` to the reduction result requires broadcasting `bias` along the `i` dimension—because the reduction output has `(i, j)` and `bias` only has `(j)`.

All of this is mechanically derivable from the tree. No guessing. No shape arithmetic. Just coordinate names and their positions in index lists. The tree preserves everything the source said—and makes it queryable by compiler passes that only understand parentheses.

This is why the IR must preserve names. If the IR replaced `k` with `axis=1`, the question "which coordinate is consumed?" would have an integer answer but no identity answer. The compiler could still generate code—`axis=1` is all NumPy needs. But it could not answer the *check* questions: does `k` appear in both `A` and `B`? Is `bias[j]` independent of `i`? These questions require names, not numbers. The IR preserves names so that the analysis passes in Chapter 13 can ask these questions—and get answers that are checkable facts, not deduced conventions.

Now let's test that claim with a harder case—one where the positional IR would be silent:

```
let y[b, s] = mean[channel](x[b, channel, s]);
```

IR:

```lisp
(let-decl (output y (b s))
  (reduction mean (channel)
    (index x (b channel s))))
```

What can you determine from this tree?
- `y` has `(b, s)`. The declaration says so.
- `channel` is consumed by `mean`. It's in the reduction bracket and absent from the output.
- `x` has `(b, channel, s)`. All three are in the index list.

Now ask: what if `x` had `(b, time, s)` instead? The IR would be `(index x (b time s))` and the reduction bracket would be `(reduction mean (channel) ...)`. The compiler would check: does `channel` appear in `(b time s)`? No. Error. The name `channel` caught the mismatch. A positional IR with `axis=1` would ask: is axis 1 valid on `x`? Yes—`x` has 3 axes. No error. Silent consumption of the wrong coordinate.

The tree preserves what the positional IR loses: the identity of the consumed coordinate.

---

## Writing a Simple Analysis Pass

Let's write a pass together. Not in real code—in pseudocode, the way you'd sketch it on a whiteboard. The pass checks Rule 1: Index Existence. Every coordinate that appears in an `(index ...)` or `(reduction ...)` must exist on the tensor being indexed.

```lisp
;; PASS: check-index-existence
;; Walk every (index T (c1 c2 ...)) node.
;; For each coordinate ci:
;;   1. Resolve T to its declaration.
;;   2. Check that ci is in T's declared coordinate list.
;;   3. If not, emit E003 with ci and T's name.

;; Example tree:
;; (index logits (batch class))  -- logits declared with (batch class): pass
;; (index logits (batch channel)) -- channel not on logits: E003
```

That's the pass. Four lines of logic. The same structure works for `(reduction sum (k) (index A (i k)))`—`k` must exist on every tensor indexed inside the reduction body. `i` must exist on `A`.

Now a second pass, checking Rule 5: Coordinate Contract at call sites:

```lisp
;; PASS: check-coordinate-contract
;; Walk every (call fn ...) node.
;; For each coordinate argument ci:
;;   1. Resolve fn to its declaration.
;;   2. Get fn's coordinate parameter layout (coord-params ...).
;;   3. Bind ci to the corresponding parameter.
;;   4. Walk the value arguments. For each, check that ci exists on the tensor.
;;   5. If the call site passes ci but the tensor lacks ci, emit E006.

;; Example:
;; (call softmax (index logits (batch class)) (coord-args class))
;; softmax expects j, class is bound to j.
;; logits has (batch class): class exists. Pass.
;;
;; (call softmax (index logits (batch feature)) (coord-args class))
;; logits has (batch feature): class does not exist. E006.
```

Again, four lines of logic. The pass is small because the information it needs is in the tree. The coordinate argument `class` is a symbol in `(coord-args class)`. The tensor's declared coordinates are symbols in the declaration node. The check is symbol comparison. No shapes. No positions. No inference. Just: "does this name appear in this list?"

The simplicity of the passes is the IR's justification. If the IR were a complex data structure—nested records with optional fields, implicit defaults, position-dependent layout—each pass would need to unpack, normalize, and reconstruct before it could ask its one question. The S-expression IR has no defaults, no implicit fields, no position-dependent layout. Every fact is a symbol in a known position in a known list. The pass asks its question. The answer is in the tree.

This is the point of homoiconicity from Section 7. The IR is both the text the compiler reads and the data structure it queries. There is no AST → IR translation step. There is no AST. The parse tree IS the IR. What you wrote by hand in the "You Are the Compiler" exercise is exactly what the parser produces. The passes walk what you wrote.

---

### Why Names Must Survive Into the IR

There is an alternative design: translate names to integers at parse time. `class` becomes `axis=1`. The IR has no names, only positions. Simpler—but wrong. If names are gone before analysis, every check rule asks "is position *p* the same as position *p*?" The answer is always yes. The rule that should catch `channel`-vs-`class` becomes a vacuous integer comparison. The IR preserves names not as a convenience, but because the five check rules require identities, not positions. Lowering burns names into integers only after every identity-based check has passed—at which point the number is guaranteed correct.


### Stop and Think: Your Own IR

Take the most recent tensor operation you wrote. Translate it into S-expression IR by hand:

1. Write the source in Einlang: `let result[i, j] = sum[k](A[i, k] * B[k, j])`.
2. Translate: `(let-decl (output result (i j)) (reduction sum (k) (* (index A (i k)) (index B (k j)))))`.
3. Now ask: what can you determine from the IR alone? Survivors: `{i, j}`. Consumed: `{k}`. Shared: `k` appears on both `A` and `B`. Broadcasts: none — both operands index all output coordinates except `k`.
4. Now ask: if the IR had integers instead of names — `(let-decl (output result (0 1)) (reduction sum (2) (* (index A (0 2)) (index B (2 1)))))` — could you determine which coordinate is shared? You could determine that axis 2 is contracted. You could not determine whether axis 2 is `k` or `feature` or `inner`. The identity is gone. The checks that depend on identity are impossible.

The exercise takes two minutes. It shows why the IR preserves names. Not for elegance. For correctness.
