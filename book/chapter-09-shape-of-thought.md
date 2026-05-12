---
layout: book
title: "Chapter 9 · The Shape of Thought"
---

# Chapter 9 · The Shape of Thought

> "A compiler is a program that reads a program written in one language—and says it again in another."
>
> — Adapted from SICP

*Construction · How the compiler reads names*

---

You just wrote a short Einlang program:

```
let x = softmax[class](logits[batch, class]);
let y = sum[class](x[batch, class] * labels[batch, class]);
```

Someone asks: what is the shape of `y`? You cannot run this code—there is no data. But you can still answer.

Trace the coordinates. `logits` has `[batch, class]`. `softmax[class]` preserves the shape—output is `[batch, class]`, bound to `x`. Then `sum[class]` consumes `class`. The surviving coordinate is `batch`. `y` has shape `[batch]`.

You just performed coordinate propagation in your head. You tracked each name—where it was introduced, how it flowed through the function call, whether it survived the reduction. You did not need data. You did not need to run the program. You needed only the names.

This—exactly this—is what a compiler must do. For every line, without data, without execution, it must answer: what is the shape of every tensor? Which coordinates survive each operation? Does the coordinate contract at each call site match the function's declaration?

But before we see the machinery that answers these questions, let's watch the questions themselves catch bugs. Here is a broken program. It was written by a programmer who intended a linear layer with bias followed by softmax over classes. It compiles. It runs. It produces wrong results.

---

## The Wall

Imagine a detective's wall. Five slots, each labeled with one rule. When a bug is found, it gets pinned to the slot of the rule that caught it.

Here is a broken program. It was written by a programmer who intended a linear layer with bias followed by softmax over classes. It compiles. It runs. It produces wrong results.

```
fn predict[class](x: [f32; batch, in], W: [f32; out, in], bias: [f32; out])
    -> [f32; batch, out]
{
    let logits[batch, class] = sum[k](x[batch, k] * W[out, k]) + bias[out];
    softmax[class](logits[batch, class])
}
```

Look at the program. Which names don't match?

**Rule 1.** `k` is referenced on `x` and `W`, but both declare `in`, not `k`. The reduction introduces a ghost coordinate. Rule 1 pins: `k` is not a declared coordinate of `x` or `W`. The writer meant `in`.

**Rule 2.** Syntactically, `k` appears in all operands. Rule 2 passes—but the wrong coordinate is being consumed. Rules 1 and 2 together cover both "coordinate doesn't exist" and "coordinate exists but isn't used consistently."

**Shape analysis.** The output declaration says `(batch, class)` but the reduction body produces `(batch, out)`. `class` appears from nowhere. `out` is produced but not declared. The declared coordinates don't match the coordinates that actually flow through the expression.

**Rule 5.** Return type says `[f32; batch, out]` but the body's `softmax[class](logits[batch, class])` returns `[f32; batch, class]`. The coordinate `out` in the return type doesn't match `class` in the return value. Pinned.

**Rule 3.** `bias[out]` omits `batch`—a correct broadcast (bias is independent of batch). Recorded for the gradient. No bug.

**Rule 4.** No recurrence. Silent.

The corrected program:

```
fn predict[class](x: [f32; batch, in], W: [f32; class, in], bias: [f32; class])
    -> [f32; batch, class]
{
    let logits[batch, class] = sum[in](x[batch, in] * W[class, in]) + bias[class];
    softmax[class](logits[batch, class])
}
```

Three changes. `sum[k]` → `sum[in]`. `W[out, k]` → `W[class, in]`. Return type `out` → `class`. Every coordinate flows from declaration to use. Every name matches.

Not one of these bugs was a shape error. In a positional framework, `sum(axis=1)` on `(batch, in)` and `(out, in)` would produce `(batch, out)`—a valid shape. The code would run. The loss would descend. And the model would be computing a meaningless function.

The five rules are the five ways a name can be wrong: it can refer to a non-existent coordinate (Rule 1), it can fail to appear where the operation requires it (Rule 2), it can broadcast silently without the record the backward pass needs (Rule 3), it can reference the future in a recurrence (Rule 4), or it can violate the contract of a function call (Rule 5). That's it. Those are all the ways.

But to apply these rules mechanically, the compiler needs a representation where names are preserved and operations are explicit. It needs an **intermediate representation**.

---

## The IR Tree

The compiler translates Einlang into S-expressions—parenthesized lists where every name is preserved:

```
A[i, j] + B[i, j]
```

```lisp
(+ (index A (i j)) (index B (i j)))
```

Add a reduction:

```
sum[k](A[i, k] * B[k, j])
```

```lisp
(reduction sum (k)
  (* (index A (i k)) (index B (k j))))
```

The reduction coordinate `k` is named, not numbered. A full declaration:

```
C[i, j] = sum[k](A[i, k] * B[k, j])
```

```lisp
(let-decl (output C (i j))
  (reduction sum (k)
    (* (index A (i k)) (index B (k j)))))
```

`(output C (i j))` declares the surviving coordinates: `i` and `j` survive, `k` is consumed.

Three things the IR preserves: **names** (`i` and `k` remain names, never become `axis=0`), **reduction targets** (`(reduction sum (k) ...)` operates on `k`), and **index patterns** (`(index A (i k))` matches what the source wrote). The IR has not *translated* your program. It has *said it again*, in parentheses.

---

## Lowering: Names Become Numbers

The tree passed every check. But it cannot be handed to NumPy. NumPy does not understand `class`. It needs `axis=1`.

Translating the analyzed tree into executable instructions is **lowering**. The mapping is deterministic: every axis name maps to its position in declaration order. `i` is first → axis 0. `class` is second → axis 1. The name is burned.

After lowering, the softmax IR becomes:

```python
def softmax(logits):
    m = np.max(logits, axis=1, keepdims=True)
    e = np.exp(logits - m)
    return e / np.sum(e, axis=1, keepdims=True)
```

`keepdims=True` was not in the source. The compiler inferred it: `logits - max` requires the max result to broadcast back over `class`. The analyzed shapes showed the mismatch—max produces `(batch)`, but the subtraction expects `(batch, class)`. `keepdims=True` bridges the gap. The programmer didn't write it. The compiler deduced it from the coordinate structure.

Matrix multiplication is a special case. `C[i, j] = sum[k](A[i, k] * B[k, j])` — `k` appears in both operands but not in the output. The compiler recognizes this as a contracting dimension and emits `C = A @ B`. No user annotation. The axis sharing pattern is the expression of intent.

---

## The Panorama: One Name, Five Forms

Here is softmax, in five simultaneous forms:

```
 max[class](logits[i class])                          ← what you wrote
 (reduction max (class) (index logits (i class)))     ← what the compiler sees
 class: (range 0 n_class), reduction axis, Rule 2 ✓   ← what the compiler derives
 class → axis=1, reduction, keepdims=True              ← how the name becomes a number
 np.max(logits, axis=1, keepdims=True)                 ← what executes
```

Five forms. One name. The name `class` traveled through all five without changing its identity. It was written as `class`, preserved as `class`, verified as `class`, mapped from `class` to `axis=1`. At no point was it guesswork.

Now ask: if the positional version had a bug—if `dim=-1` was normalizing over the wrong axis—at which of the five stages would that bug be caught?

Source: not caught. `dim=-1` is a valid integer.
IR: not caught. No names to verify.
Analysis: not caught. No coordinate contract to check.
Lowering: not caught. `-1` maps correctly—it's the *choice* of `-1` that is wrong.
Generated code: not caught. `np.max(logits, axis=-1)` is valid NumPy.

The answer: **none of them.** The positional bug is invisible to all five stages because the information that would expose it—the name of the coordinate—was never written down.

The Einlang bug is caught at Form 3. Analysis checks: does `class` appear in every operand of the reduction? Does it exist on the tensor? The bug surfaces before the program runs, at the stage where names are still names and the compiler can still reason about them.

---

## The Core Loop

The entire compiler frontend fits in fifteen lines:

```
check(expr, env, errors):
  match expr:
    case Index(T, coords):
      for c in coords:
        if not declared(c, T, env):
          errors.push("undeclared", c, "in", T)
      return coords

    case Reduction(op, c, body):
      out = check(body, env + [c], errors)
      if c not in out:
        errors.push(c, "not consumed")
      return out - {c}

    case Add(left, right):
      L = check(left, env, errors)
      R = check(right, env, errors)
      for c in R:
        if c not in L:
          errors.push("broadcast", c, "into left")
      for c in L:
        if c not in R:
          errors.push("broadcast", c, "into right")
      return L | R

    case LetDecl(output, T, coords, body):
      check(body, env + coords, errors)
      return coords

    default:
      errors.push("unknown node", expr)
      return {}
```

Walk the tree. At each node, ask one question. If wrong, record it. If right, return the coordinate set.

Lines 11–16 are the broadcast merge. When `Add(left, right)` is checked, every coordinate on the right absent on the left means the left operand must broadcast into it—and vice versa. The `Add` node doesn't need to know what operation it's checking—only that both sides contribute coordinate sets and the broadcast relationship must be recorded.

The complexity is in the details—type inference, pack resolution, error message formatting. The structure is fifteen lines. You can hold the entire thing in your head.

---

You wrote `class`. Five characters. They survived parsing, analysis, lowering—each stage asking a question that a number could not. At the end, they became `axis=1` and were burned. But the burn was correct because the name was verified.

The positional alternative is `dim=-1`: three keystrokes that enable zero checks. The ratio is the distance between correct-by-construction and correct-by-coincidence.

Consume—that word has appeared in every chapter since Chapter 2. A reduction consumes a coordinate. A broadcast consumes silence. A gradient consumes the broadcast set. And now the compiler consumes the name itself. `class` goes in. `axis=1` comes out. A good abstraction is good firewood. Its beauty is not in its surface—but in the light the flame casts when it burns.
