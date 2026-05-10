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

**Catches**: reduction bracket says `k` but one index reference uses `q` by mistake. *If you have ever written `sum[class]` but one operand's index used `batch` instead—the result had the right shape but the wrong coordinate was consumed—this rule would have caught it. A reduction consumes a coordinate. The bracket names what disappears. Rule 2 checks: did you actually consume what you claimed to consume?*

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

Here is a broken Einlang program. It was written by a programmer who intended to implement a simple linear layer with a bias, followed by a softmax over classes. It compiles. It runs. It produces wrong results. The programmer has been debugging for two hours.

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

## Five Micro-Scenarios

The wall was one program, broken in multiple ways. Now let's isolate each rule with its own scenario—one bug, one rule, one line. These are the shortest Einlang programs that trigger each check.

### Micro-Scenario 1: Index Existence (Rule 1)

A programmer refactors a data loader. The CSV column `temperature` is renamed to `temp`. The model code is updated—mostly:

```
let readings[station, temperature] = load_csv("weather.csv");
let anomalies[station] = mean[temperature](readings[station, temperature]);
//                                                    ^^^^^^^^^^^^^^^^^^^^^^^^^
//                                                    ✓ temperature exists
```

But the normalization function still references the old name:

```
fn normalize[feature](x: [f32; station, temperature]) -> [f32; station, temperature] {
    let stats[station, channel] = mean[temperature](x[station, temperature]);
    //                          ^^^^^^^ 坐标 channel doesn't exist in output
}
// wait—the bug is subtler
```

Better example. The data pipeline changed, but one file wasn't updated:

```
// file: data.ein — updated
let readings[station, temp, hour] = load_csv("weather.csv");

// file: model.ein — NOT updated
let avg[station, temperature] = mean[temperature](readings[station, temperature, hour]);
//                                            ^^^^^^^^^^^
//                                            ERROR: 'temperature' not in (station, temp, hour)
//                                            did you mean 'temp'?
```

The error message names the missing coordinate and suggests the closest match. The programmer sees it, fixes `temperature` → `temp`, and the code compiles. In a positional framework, `temperature` was axis 1, `temp` is axis 1—the integer hasn't changed. The bug would compile and run. At runtime, it would silently average over the wrong thing.

**What Rule 1 catches**: a coordinate rename that the positional equivalent would silently absorb.

### Micro-Scenario 2: Reduction Consistency (Rule 2)

A programmer writes a bilinear form. Two matrices, one reduction:

```
let bilinear[i, j] = sum[k](A[i, k] * B[q, j]);
//                                      ^
//                                      ERROR: reduction consumes 'k', but 'B' uses 'q'
```

`k` is the reduction coordinate. It appears in `A[i, k]`. It does not appear in `B[q, j]`. The result shape would be `(i, q, j)`—three surviving coordinates from two-indexed tensors. The programmer meant `B[k, j]` but typed `q` by habit (`q` is used elsewhere in the function as a query coordinate).

Rule 2 catches this: the reduction coordinate `k` does not appear in every operand. The reduction cannot consume what isn't there. The compiler reports: *reduction coordinate `k` missing from operand `B`*.

In a positional framework: `A.shape = (n, m)`, `B.shape = (p, m)`. If `n == p`, the matmul produces a valid shape. The bug is that `B`'s first axis should contract with `A`'s second, but the shapes happen to allow a different contraction. The result has the right dimensions, the wrong values, and no error.

**What Rule 2 catches**: a coordinate in the reduction bracket that doesn't appear in every operand—a bug that shape-compatible tensors would hide.

### Micro-Scenario 3: Broadcast Recording (Rule 3)

Rule 3 is not an error check—it records facts. But the facts it records prevent errors downstream. Consider:

```
let scaled[i, j] = x[i, j] * factor[j];
//  factor omits i → broadcasts over i
```

The compiler records: `factor` broadcasts over `{i}`. Now consider the gradient:

```
// backward pass (automatic, from Rule 3's record)
let d_factor[j] = sum[i](d_scaled[i, j] * x[i, j]);
//                   ^^^ sum over i because factor broadcast over i forward
```

Without the record, the backward pass must infer the sum from shapes. If `factor` changes from a 1D `(j,)` to a 2D `(i, j)` during a refactoring, the shape-based inference changes silently—the backward sum might sum over different axes. With the record, the backward pass reads `broadcast over {i}` and sums over `{i}` regardless of shape.

Now the scenario where it matters. A programmer adds a time dimension to `factor`:

```
// Before: factor[j] — static per-class weight
let scaled[i, j] = x[i, j] * factor[j];

// After: factor[t, j] — time-varying per-class weight
let scaled[t, i, j] = x[t, i, j] * factor[t, j];
//                                   factor omits i → broadcasts over {i}
//                                   factor has {t, j} — t is present, i is omitted
```

The broadcast record updates: `factor` broadcasts over `{i}` (still). The backward pass continues to sum over `i`. The time dimension `t` passes through undisturbed—because it was never in the broadcast record. The gradient comes out correct without the programmer touching the backward pass.

**What Rule 3 catches**: a backward pass that would silently change behavior when a forward broadcast's coordinate structure changes. The record survives shape changes.

### Micro-Scenario 4: Causality (Rule 4)

The simplest micro-scenario in the book. Two lines:

```
let h[t in 1..T, d] = activation(h[t-1, d] * W[d, d] + b[d]);
// ✓ t-1 < t, valid backward reference

let h[t in 1..T, d] = activation(h[t+1, d] * W[d, d] + b[d]);
// ERROR: t+1 > t, forward reference
```

The second line is legal Python. In a positional loop:

```python
h = torch.zeros(T, D)
for t in range(1, T):
    h[t] = activation(h[t+1] @ W + b)  # reads uninitialized memory at first iteration, IndexError at last
```

The Python loop either reads garbage (if `h` is pre-allocated) or raises an IndexError (if `t+1 >= T`). Either way, the error is at runtime—possibly after thousands of correct iterations. The Einlang compiler rejects the second line at analysis time. `t+1` references a future that has not been computed.

**What Rule 4 catches**: a forward reference in a recurrence. One character (`+` vs `-`) changes a correct program into a silent runtime error. Rule 4 catches it before the loop runs.

### Micro-Scenario 5: Coordinate Contract (Rule 5)

A colleague refactors the data pipeline. The CSV column `class` is renamed `category` for clarity. The colleague updates the data loader, the model declaration, and every call site they can find. But one call site is missed—it's in a different file, in a utility function that's rarely touched:

```
// file: pipeline.ein — updated by colleague
fn pipeline[in, category](x: [f32; ..batch, in], W: [f32; category, in])
    -> [f32; ..batch, category]
{
    let logits[..batch, category] = ...
    softmax[category](logits[..batch, category])  // ✓ updated
}

// file: utils.ein — NOT updated
fn debug_probs(x: [f32; ..batch, class]) -> [f32; ..batch, class] {
    let p[..batch, class] = softmax[class](x[..batch, class]);
    //                                ^^^^^
    //                                ERROR: 'class' not found on 'x'
    //                                x has coordinates (..batch, category)
    //                                did you mean 'category'?
}
```

The compiler catches it. The error message names the missing coordinate (`class`), the tensor it was expected on (`x`), the tensor's actual coordinates (`batch, category`), and a suggestion (`did you mean 'category'?`). The programmer sees it, fixes `class` → `category`, and the code compiles.

In the positional equivalent, `debug_probs` calls `softmax(logits, dim=-1)`. The colleague changed the data layout but `dim=-1` is still valid—it just silently refers to a different coordinate now. The code runs. The output has the right shape. The loss looks fine. The bug survives. Rule 5 would have caught it at compile time, but Rule 5 requires a coordinate name to check.

**What Rule 5 catches**: a coordinate rename that propagates incompletely through the codebase—the most common refactoring bug in large tensor programs.

---

### The Five, Separated

Each rule catches a different shape of error. Together they cover the space of coordinate bugs:

| Rule | Bug shape | Positional behavior |
|---|---|---|
| 1: Index Existence | Coordinate renamed upstream, reference lingers | Silent axis shift |
| 2: Reduction Consistency | Reduction coordinate missing from one operand | Shape-compatible wrong contraction |
| 3: Broadcast Recording | Broadcast structure changes, backward pass drifts | Silent gradient shape change |
| 4: Causality | Forward reference in recurrence | Runtime garbage or IndexError |
| 5: Coordinate Contract | Wrong coordinate argument at call site | Silent normalization over wrong axis |

Not every tensor bug is a coordinate bug. But every coordinate bug is one of these five shapes. And every one of these shapes is silent in a positional framework—because the positional framework has no place to record the coordinate identity that would expose the mismatch.

Five rules. Five micro-scenarios. Five compiler errors that replace five runtime silences. The wall and the micro-scenarios together are the compiler's argument that *the names are sufficient*—not sufficient to catch every bug, but sufficient to catch every bug whose root cause is a coordinate identity that was written in one place and wrong in another.

---

## Design a Rule: Concat

The five rules were not handed down from above. Each was designed to catch a specific class of coordinate bug. Now you design one yourself.

Here is a new operation: `concat`. It takes two tensors and a coordinate, and concatenates them along that coordinate:

```
let combined[batch, feature] = concat[feature](A[batch, feature_a], B[batch, feature_b]);
```

`feature_a` and `feature_b` are different coordinate names—they represent different extents of the same semantic dimension (e.g., two feature sets being joined). The result has coordinate `feature`, whose range is `range(feature_a) + range(feature_b)`.

Take five minutes. Write down:

1. **What must be checked?** The operation `concat[feature]` consumes no coordinates—`feature` survives in the output. But what must be true of the coordinates that are *not* being concatenated over? If `A` has `(batch, feature_a)` and `B` has `(batch, feature_b)`, the `batch` coordinate must match. If `A` had `(batch_a, feature_a)` and `B` had `(batch_b, feature_b)`, the concatenation would produce a result with two unrelated batch axes—nonsense.

2. **Write the rule.** In one sentence: what does the compiler check before allowing `concat[feature](A, B)`?

3. **What's the positional equivalent?** PyTorch's `torch.cat([A, B], dim=1)` checks that `A.shape[0] == B.shape[0]` (all non-concatenated dimensions match). It catches the shape mismatch. What does it NOT catch that a name-based check could?

Don't scroll down until you've written your answers.

---

Done? Compare:

**What must be checked.** Every coordinate that is NOT the concatenation coordinate must have the same name and same range in both operands. If `A` has `(batch, feature_a)` and `B` has `(seq, feature_b)`, the compiler should reject the concat—`batch` and `seq` are different coordinates, and concatenating across them would silently mix batch and sequence elements.

**Write the rule.** *For every coordinate in the output that is not the concatenation coordinate, both operands must carry that coordinate with the same name and compatible range.* (Ranges must be equal, not just compatible—unlike broadcasting where omission is allowed.)

**What's the positional equivalent miss?** `torch.cat([A, B], dim=1)` checks that `A.shape[0] == B.shape[0]`. If both have shape `(32, 64)`, the check passes. But `A` might have `(batch=32, feature_a=64)` and `B` might have `(seq=32, feature_b=64)`. The shapes match. The coordinates don't. The positional check verifies only that the numbers are equal. It cannot verify that the identities are consistent. The name-based check catches the identity mismatch before the tensors are joined.

---

You just designed a check rule. The five rules in this chapter were designed the same way: start with an operation, ask what can go wrong, write a rule that catches it. The rules are not magic. They are engineering. And they all depend on the same thing: coordinates that have names.

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

A reduction consumes a coordinate—the bracket names what disappears. A broadcast omits a coordinate—the omission records what the value is independent of. A gradient collects over what was broadcast and broadcasts over what was consumed. The five check rules are the compiler's answer to a single question: *did you actually consume what you claimed to consume?* Rule 2 checks it directly. Rule 3 records what was broadcast so the gradient can consume it. Rule 4 checks that time steps only consume the past. Rule 1 verifies the coordinate existed to be consumed. Rule 5 verifies the contract that binds consumed coordinates across function boundaries.

The tree is now complete. Every slot that needed an answer has one. It can be safely handed to the next chapter—where the names are burned.
