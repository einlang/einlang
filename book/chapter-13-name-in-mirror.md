---
layout: book
title: "Chapter 13 · The Name in the Mirror"
---

# Chapter 13 · The Name in the Mirror

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

## The Error Map: A Detective Walkthrough

You have seen the five rules. You have read what each catches. But reading a list of rules and applying them are different things. Let's apply them together.

Here is a broken einlang program. It was written by a programmer who intended to implement a simple linear layer with a bias, followed by a softmax over classes. It compiles. It runs. It produces wrong results. The programmer has been debugging for two hours.

```
fn predict[class](x: [f32; batch, in], W: [f32; out, in], bias: [f32; out])
    -> [f32; batch, out]
{
    let logits[batch, class] = sum[k](x[batch, k] * W[out, k]) + bias[out];
    softmax[class](logits[batch, class])
}
```

Stop. Before reading further, find the bugs yourself. There are multiple. Don't just glance—trace each coordinate. Which names appear where? Which names are missing? Which operations consume the right coordinate? Take two minutes. Write down every bug you find.

---

Done? Good. Now let's hunt together.

### The Wall

Imagine a detective's wall. Five empty slots, each labeled with one rule. Every time we find a bug, we pin it to the slot of the rule that caught it. By the end, we will see whether these five rules are enough—or whether some bugs slip through.

The slots:

```
[ Rule 1: Index Existence  ]   [ Rule 2: Reduction Consistency ]   [ Rule 3: Broadcast Recording ]
[ Rule 4: Causality        ]   [ Rule 5: Coordinate Contract    ]
```

### Round 1: Rule 1 — Index Existence

Rule 1 says: every name in an index list must exist on that tensor's declared axes. Let's check.

First line: `fn predict[class](x: [f32; batch, in], W: [f32; out, in], bias: [f32; out])`.

Parameter `x` declares axes `(batch, in)`. Parameter `W` declares axes `(out, in)`. `bias` declares axes `(out)`.

Second line: `let logits[batch, class] = sum[k](x[batch, k] * W[out, k]) + bias[out]`.

- `x[batch, k]`: `x` has `(batch, in)`. `batch` exists. `k`—does `k` exist on `x`? `x` has `in`, not `k`. **First bug.** `k` is not declared on `x`. The writer meant to use `in` but wrote `k`.

Pin it: **[Rule 1 — Index Existence]** `k` referenced on `x` but `x` declares `in`.

What about `W[out, k]`? `W` has `(out, in)`. `out` exists. `k`—same bug. `W` declares `in`, not `k`. Two pins on the same rule.

The reduction variable `k` in `sum[k]` is introduced by the reduction bracket—it is the coordinate being consumed. But it must match the coordinate that exists on the tensors inside the body. Here, `k` is a ghost coordinate—it appears in the index lists without being a declared coordinate of those tensors. The writer renamed the coordinate in their head but not in the code.

Now check `bias[out]`: `bias` declares `(out)`. `out` exists. Correct.

`softmax[class](logits[batch, class])` on line 3: `logits` was declared with `(batch, class)`. `class` exists on `logits`. Correct.

**Pins so far**: Rule 1 — two occurrences of the same rename bug.

### Round 2: Rule 2 — Reduction Consistency

Rule 2 says: the consumed coordinate must appear in every operand's index list within the reduction body.

The reduction is `sum[k]`. The body is `x[batch, k] * W[out, k]`.

Does `k` appear in every operand? `x[batch, k]` — yes, `k` is there. `W[out, k]` — yes, `k` is there. All operands reference `k`.

Syntactically, Rule 2 passes. But semantically? The reduction coordinate should be `in`—the coordinate that `x` and `W` both declare. `k` is a name the writer invented on the spot. The fact that `k` appears in both operands means Rule 2 is satisfied, but the *wrong coordinate* is being consumed. This is a near-miss: Rule 2 catches absence, not misidentification. The misidentification is caught by Rule 1 (index existence) when `k` doesn't match any declared coordinate. Together, Rules 1 and 2 cover both "the coordinate doesn't exist" and "the coordinate exists but isn't used consistently."

Now look at the output declaration: `logits[batch, class]`. The reduction consumed `k`. The surviving coordinates from the body are `batch` (from `x`) and `out` (from `W`). But the output declares `(batch, class)`. `class` appears in the output but was never in the body. `out` appears in the body but is not in the output. This is a **shape mismatch** between declared output coordinates and the coordinates that actually survive the reduction. The shape analysis from the previous section catches this—the declared coordinates don't match the body's surviving coordinates. It's not one of the five rules per se, but it falls out of the analysis the rules feed.

**Note it**: the output declaration says `class` but the body produces `out`. Caught by shape analysis.

### Round 3: Rule 3 — Broadcast Recording

Rule 3 records what each term broadcasts over. In `let logits[batch, class] = sum[k](x[batch, k] * W[out, k]) + bias[out]`:

The reduction `sum[k]` produces surviving coordinates `{batch, out}`. Then `+ bias[out]`. `bias` has coordinates `{out}`. The full output set is `{batch, out}`. `bias` is missing `batch`. So `bias` broadcasts over `batch`.

Is this correct? A bias should be independent of batch—that's the definition of a bias term. This broadcast is intentional. Rule 3 records it: `bias` broadcasts over `batch`. The backward pass will sum over `batch` to produce `d_bias[out]`. No bug here. Rule 3's job is not to flag errors—it's to record facts for the gradient. And the fact it records is correct.

### Round 4: Rule 4 — Causality

There is no recurrence in this program. No `t`, no `t-1`, no `t+1`. Rule 4 is silent. Move on.

### Round 5: Rule 5 — Coordinate Contract at Call Sites

Rule 5 says: when calling a coordinate-aware function, the coordinate argument must exist on the value argument, and the call must match the function's declared contract.

The call is `softmax[class](logits[batch, class])`. `softmax`'s declaration (from Chapter 3) is `fn softmax[j](x: [f32; ..left, j, ..right]) -> [f32; ..left, j, ..right]`. The call binds `j` to `class`. `logits` has `(batch, class)`. `class` exists on `logits`. This passes.

But now look at the function `predict` itself. Its return type is `[f32; batch, out]`. The body's last expression is `softmax[class](logits[batch, class])`, which returns `[f32; batch, class]`. The return type says `out`; the body produces `class`. **Bug.** The coordinate `out` in the return type doesn't match the coordinate `class` in the return value.

Pin it: **[Rule 5 — Coordinate Contract]** Return type declares `out` but body produces `class`.

### The Wall, After the Hunt

Let's look at our wall:

```
[ Rule 1: Index Existence   ]  [ Rule 2: Reduction Consistency ]  [ Rule 3: Broadcast Recording ]
  ✗ k on x (meant "in")         ~ k appears in all operands         ✓ bias broadcasts over batch
  ✗ k on W (meant "in")         (syntactic pass, semantic miss)    (intentional, recorded)

[ Rule 4: Causality          ]  [ Rule 5: Coordinate Contract    ]
  N/A (no recurrence)            ✗ Return type says "out", body
                                    produces "class"
```

Four bugs pinned on the wall. Two caught by Rule 1. One caught by Rule 5. One caught by shape analysis (the output coordinate mismatch between declaration and expression). Rule 2 had a near-miss—the reduction coordinate was wrong but syntactically present. Rule 3 recorded a correct broadcast. Rule 4 was not applicable.

Now rewrite the program correctly:

```
fn predict[class](x: [f32; batch, in], W: [f32; class, in], bias: [f32; class])
    -> [f32; batch, class]
{
    let logits[batch, class] = sum[in](x[batch, in] * W[class, in]) + bias[class];
    softmax[class](logits[batch, class])
}
```

Three changes. `sum[k]` → `sum[in]`: the reduction now consumes the coordinate that both `x` and `W` declare. `W[out, k]` → `W[class, in]` (and `W`'s declaration from `out` to `class`): the output coordinate `class` now flows from `W`'s declaration through the reduction body to the output. Return type `out` → `class`: the declared return coordinate matches the body. Every coordinate flows from declaration to use. Every name matches.

### What the Wall Teaches

Every bug we found was a name that didn't match. `k` was supposed to be `in`. `class` was declared in the output declaration but came from nowhere in the body. `out` was in the return type but not in the return value. Not one of these bugs was a shape error—the shapes would have been correct in a positional version. `sum[k]` over the second axis of `(batch, in)` and second axis of `(out, in)` would produce `(batch, out)`—a valid shape. The code would run. The loss would descend. And the model would be computing a meaningless function.

Now replay the debugging session the programmer spent two hours on. In a positional framework, the first sign of trouble would be a runtime shape mismatch—maybe at the loss computation, maybe at the gradient step, maybe not at all if all shapes happened to align. The programmer would trace shapes backward, print `x.shape` and `logits.shape`, and eventually deduce that an axis was misnamed. The deduction would rely on the programmer's understanding of what each axis *should* be—an understanding not recorded anywhere in the code.

In the named version, the compiler pins the bugs before the program runs. Rule 1: `k` is not a coordinate of `x`. Rule 5: return type mismatch. The programmer sees the errors, fixes the names, and moves on.

The wall is not a metaphor. It is a design claim. The claim is: **every coordinate bug that can exist in a tensor program is caught by one of these five rules, or by the shape analysis that tracks whether declared coordinates match the coordinates that actually flow through expressions.** The five rules are not arbitrary. They are the five ways a name can be wrong: it can refer to a non-existent coordinate (Rule 1), it can fail to appear where the operation requires it (Rule 2), it can broadcast silently where it shouldn't—or worse, broadcast correctly but without the record the backward pass needs (Rule 3), it can reference the future in a recurrence (Rule 4), or it can violate the contract of a function call (Rule 5). That's it. Those are all the ways.

Every rule was born from a night someone spent staring at a tensor shape that was correct and a computation that was wrong. The night that became Rule 1: a renamed column in a CSV that silently shifted every downstream index. The night that became Rule 2: a `sum[class]` where one operand accidentally used `batch` instead, producing the right shape with the wrong reduction axis. The night that became Rule 3: a broadcast that flipped direction after a refactoring, and a gradient that silently summed over the wrong dimension. The night that became Rule 4: an RNN that read from `t+1` instead of `t-1` and converged to garbage. The night that became Rule 5: a `softmax[class]` call where `class` didn't exist on the tensor anymore, and the code ran anyway because `dim=-1` still pointed to something.

Five rules. Five nights. One wall.

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
