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

Part II showed that coordinate names survive composition—through functions, through time, through differentiation. Part III asks: can this be automated? This chapter constructs a compiler frontend that reads named coordinates, checks them against five rules, and lowers them to integers. It is the proof that the checks you have been doing by hand—tracing `batch` and `class` through softmax, subtracting coordinate sets at broadcast sites, verifying that function contracts match—can be done by a machine.

You just wrote a short Einlang program:

```
let x[batch, class] = softmax[class](logits[batch, class]);
let y[batch] = sum[class](x[batch, class] * labels[batch, class]);
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

Every one of these checks reads the same fact from every tensor: its **coordinate layout**—the ordered list of names the tensor carries. When you write `x: [f32; batch, in]`, the compiler records `layout(x) = [batch, in]`. Rule 1 checks whether `k` appears in that list. Rule 3 subtracts lists to find broadcast axes. Rule 5 compares the call-site list against the function's declared parameter list. The five rules are five queries against one data structure. The coordinate layout is the one fact every rule reads.

The five rules are the five ways a name can be wrong: it can refer to a non-existent coordinate (Rule 1), it can fail to appear where the operation requires it (Rule 2), it can broadcast silently without the record the backward pass needs (Rule 3), it can reference the future in a recurrence (Rule 4), or it can violate the contract of a function call (Rule 5). That's it. Those are all the ways.

---

**Derive it yourself.** Here is a buggy program. Apply the five rules. Which rules fire? What does each rule catch?

```rust
fn forward[batch, out](x: [f32; batch, in], W: [f32; out, in], b: [f32; out])
    -> [f32; batch, out]
{
    sum[k](x[batch, k] * W[out, k]) + b[out]
}
```

Rules 1–5. Go.

---

But to apply these rules mechanically, the compiler needs a representation where names are preserved and operations are explicit. It needs an **intermediate representation**—a tree where every name is visible.

Before we build the tree, try this: write `sum[k](A[i, k] * B[k, j])` on a piece of paper. Circle every name. Draw an arrow from each name to where it appears. Now erase the brackets and the `sum`. What's left? The structure of the computation—two indices multiplied, one coordinate reduced away, two coordinates surviving. That structure is the IR. The names are its bones.

---

## The IR Tree

Conceptually, the compiler's IR can be understood as a tree of expressions where names are first-class:

```
A[i, j] + B[i, j]
```

```
(+ (index A (i j)) (index B (i j)))
```

Add a reduction:

```
sum[k](A[i, k] * B[k, j])
```

```
(reduction sum (k)
  (* (index A (i k)) (index B (k j))))
```

The reduction coordinate `k` is named, not numbered. A full declaration:

```
let C[i, j] = sum[k](A[i, k] * B[k, j])
```

```
(let-decl (output C (i j))
  (reduction sum (k)
    (* (index A (i k)) (index B (k j)))))
```

`(output C (i j))` declares the surviving coordinates: `i` and `j` survive, `k` is consumed.

Three things the IR preserves: **names** (`i` and `k` remain names, never become `axis=0`), **reduction targets** (`(reduction sum (k) ...)` operates on `k`), and **index patterns** (`(index A (i k))` matches what the source wrote). The IR has not *translated* your program. It has *said it again*, as a tree.

But the IR preserves more than the tree shape. Each node carries its **coordinate set**—the names that survive at that point. `(index A (i k))` carries `{i, k}`. `(reduction sum (k) ...)` carries `{i}`—`k` was consumed. The compiler computes these sets by walking the tree: index nodes introduce coordinates, reductions subtract them, additions merge them. This is exactly what you did by hand at the start of this chapter—tracing `batch` and `class` through softmax and sum, without data, using only the names. The compiler does it mechanically, node by node, for every expression in the program.

---

## Lowering: Names Become Numbers

The tree passed every check. But it cannot be handed to a numerical backend. NumPy does not understand `class`. It needs `axis=1`.

Translating the analyzed tree into executable instructions is **lowering**. The mapping is deterministic: the compiler reads the coordinate layout stored on each tensor's IR node—the same layout that the five rules consulted during checking. `logits` was declared `[f32; batch, class]`, so its layout is `[batch, class]`. At lowering, the compiler walks the layout and assigns positions: `batch → 0`, `class → 1`. Then every index expression in the tree is rewritten: `(index logits (batch class))` touches both axes; `(reduction max (class) ...)` becomes a reduction over axis 1. The layout is the map. Lowering is the lookup. The name is burned.

After lowering, the softmax conceptually becomes:

```python
def softmax(logits):
    m = np.max(logits, axis=1, keepdims=True)
    e = np.exp(logits - m)
    return e / np.sum(e, axis=1, keepdims=True)
```

`keepdims=True` was not in the source. The compiler inferred it. Here is the principle, visible in the core loop at the end of this chapter:

The subtraction `logits - max_result` has two operands. `logits` carries `{batch, class}`. `max_result`—the output of `max[class](logits)`—carries `{batch}` because `class` was consumed. The coordinate sets differ: left has `class`, right does not. Set difference: `{batch, class} - {batch} = {class}`. The missing coordinate is `class`. The compiler records the broadcast—not as a flag the programmer writes, but as a requirement the coordinate structure demands. This is not a heuristic. It is the same set subtraction from Chapter 2, applied to the operand coordinate sets at every binary operation.

In the actual compiler, this check lives in the `CoordinateGroundingPass`: at every `BinaryOpIR` node, the left and right operand layouts are compared. When they are equal, the shared layout is stamped on the result. When they differ, no layout can be stamped—the coordinate structures are incompatible, and the mismatch is recorded as a broadcast that the lowering pass must resolve. The core loop at the end of this chapter reduces this logic to four lines: compare the sets, record the differences, return the union. The principle is the same. The implementation is more careful.

The same set-difference logic applies to reductions. `sum[k](A[i, k] * B[k, j])` — `k` appears in both operands but not in the output `C[i, j]`. The compiler subtracts `{k}` from the body's coordinate set to produce the output layout `{i, j}`. The coordinate `k` is a contracting dimension—shared by both operands, consumed by the reduction, absent from the output. The pattern `(i, k) × (k, j) → (i, j)` follows from the coordinate structure alone. No annotation needed. The compiler lowers this to a nested loop with a reduction over `k`. The names told it which axis to contract.

---

## The Panorama: One Name, Five Forms

Here is softmax, in five simultaneous forms:

```
 max[class](logits[i, class])                          ← what you wrote
 (reduction max (class) (index logits (i class)))     ← what the compiler sees
 class: (range 0 n_class), reduction axis, Rule 2 ✓   ← what the compiler derives
 class → axis=1, reduction                                ← how the name becomes a number
 np.max(logits, axis=1, keepdims=True)                 ← what executes (conceptually)
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

## Range Inference: Where Domains Come From

The constraint solver needs ranges. `oh` ranges over `0..output_height`. Where does that range come from?

Sometimes the user declares it: `oh in 0..output_height`. But in the common case, they don't. When you write `let result[i] = data[i] * 2`, the range of `i` is never stated. The compiler infers it.

The algorithm: find every array access where `i` appears as an index. For each access, look at the array's declared shape at the position where `i` appears. If `data` has shape `[N]`, then `i < N`. If `i` also appears in `arr[i+1]` and `arr` has shape `[M]`, then `i+1 < M`, so `i < M-1`. Collect all inferred upper bounds. Take the minimum — the most restrictive one. That is `i`'s range.

When you write `let result[i] = arr[i] + arr[i+1]`, the compiler finds two accesses:
- `arr[i]` → `i < len(arr)` → `i in [0, N)`
- `arr[i+1]` → `i+1 < len(arr)` → `i in [0, N-1)`

Intersection: `i in [0, N-1)`. The compiler inferred that `i` cannot reach `N-1` because `arr[i+1]` would be out of bounds. No annotation needed. The name `i`, appearing in two positions, carries enough information.

The entire algorithm is: sort accesses by expression complexity (direct `i` before `i+1`), back-compute a range from each, intersect. The coordinate-name philosophy makes this possible — every index variable appears literally in array accesses, so every range is inferrable from the shapes those arrays were declared with.

---

## Index Arithmetic: The Constraint Solver

With ranges in hand, the compiler can check a harder problem: index arithmetic. `input[b, ic, oh + kh, ow + kw]` — the expression `oh + kh` must not exceed the input's spatial extent. Given the ranges for `oh` and `kh`, is this checkable at compile time?

When the domain sizes are known statically, the answer is yes. The compiler's constraint solver reads the index expression and the declared bounds, then solves for each index variable. The algorithm is pattern-matching on the expression tree.

Take the convolution index `oh + kh`. The compiler knows:
- `oh` ranges over `0..output_height` (from the output declaration)
- `kh` ranges over `0..kernel_height` (from the weight declaration)
- The input has `input_height` (from the input declaration)
- Constraint: `oh + kh < input_height`

The solver must verify that `oh + kh < input_height` for all valid `oh` and `kh`. It does this by solving for each variable in the worst case. For `oh + kh`, the maximum value is `(output_height - 1) + (kernel_height - 1)`. If this maximum is less than `input_height`, the constraint holds. If the domains are known, the check is a single comparison.

But the solver handles more complex expressions. Consider `(i * 2 + offset) < N`. The solver pattern-matches the expression tree:

1. **Top level is addition** `(i * 2) + offset < N`. Isolate the target: `i * 2 < N - offset`.
2. **Recurse into multiplication** `i * 2 < N - offset`. Isolate: `i < (N - offset) / 2`.
3. **Base case** `i < bound`. The range is `[0, ceil(bound))`.

The solver chains these rewrites automatically. Addition adjusts the bound by subtraction. Multiplication adjusts by division (with ceiling, so `(a + b - 1) / b` for safety). Division by a constant adjusts by multiplication. When the target appears with a negative coefficient—`k - target < bound`—the solver returns nothing: negative coefficients require a lower bound, and the solver only computes upper bounds. The pattern is recognized. It is declared unsolvable by the current pass.

The solver operates entirely on IR nodes, not Python integers. The bound `N` can be a dynamic expression like `image.shape[0]` rather than a compile-time constant. The ceiling division `ceil(a / b)` is implemented as `(a + b - 1) / b` at the IR level, so it works for any input size. This means the bounds inference produces symbolic ranges that are safe for any runtime shape.

The constraint solver is not a general-purpose theorem prover. It handles the patterns that appear in real tensor index expressions: linear combinations with positive coefficients, simple arithmetic (`i*2`, `i/2`, `i+1`, `i-1`), and the common stencil patterns from Chapter 7. For patterns it cannot solve—modulo, negative coefficients, non-linear expressions—the solver returns no range, and lowering fails with a compile error. The failure is a compile error, not a runtime surprise. The name is attached to the error, as it is to every error in this compiler.

What the solver proves: that a coordinate arithmetic expression, evaluated over its declared ranges, stays within the declared bounds of the tensor it indexes. What it cannot prove—correctness of the arithmetic itself—is not a failure of the solver. It is the boundary from Chapter 14. The solver narrows that boundary. Every expression it proves safe is one less degree of freedom for silent errors. Every expression it cannot prove is flagged before the program runs.

---

## The Core Loop

The Wall presented five rules. Before reading the implementation: Rules 1 and 2 (undeclared coordinates, missing operands) require checking every index expression against its tensor's declaration. Rule 3 (broadcast recording) requires comparing operand coordinate sets at every binary operation. Rule 4 (causality) applies only to recurrences. Rule 5 (contract matching) applies only at function call boundaries. Which of these does a tree-walking checker handle naturally? Which require additional machinery?

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

Rules 1 and 2 are the `Index` case: check every coordinate against its tensor's declaration. Rule 3 is the `Add` case: compare left and right coordinate sets, record the differences as broadcasts. Rule 5 is the `LetDecl` case: verify the function body's coordinate set against the declared output. Rule 4—causality—is not in this loop; recurrence checking is a separate pass with its own fifteen lines. The five rules from the Wall map to five cases in the tree walk. Four of them are above.

Lines 11–16 are the broadcast merge. When `Add(left, right)` is checked, every coordinate on the right absent on the left means the left operand must broadcast into it—and vice versa. The `Add` node doesn't need to know what operation it's checking—only that both sides contribute coordinate sets and the broadcast relationship must be recorded.

The complexity is in the details—type inference, pack resolution, error message formatting. The structure is fifteen lines. You can hold the entire thing in your head.

If you implement these fifteen lines and add three things: **type inference** for scalar expressions (every literal is `f32`, every variable inherits from its declaration), **pack resolution** for `..rest` coordinates (unroll the pack into its concrete members at each call site), and **error formatting** that prints the coordinate name, the file, and the line—you have a working coordinate checker. Total: roughly three hundred lines. The fifteen lines above are the skeleton. The three additions are the flesh. The names are the firewood.

---

You wrote `class`. Five characters. They survived parsing, analysis, lowering—each stage asking a question that a number could not. At the end, they became `axis=1` and were burned. But the burn was correct because the name was verified.

The positional alternative is `dim=-1`: three keystrokes that enable zero checks. The ratio is the distance between correct-by-construction and correct-by-coincidence.

Consume—that word has appeared in every chapter since Chapter 2. A reduction consumes a coordinate. A broadcast consumes silence. A gradient consumes the broadcast set. And now the compiler consumes the name itself. `class` goes in. `axis=1` comes out. A good abstraction is good firewood. Its beauty is not in its surface—but in the light the flame casts when it burns.

The compiler proved that names can be checked mechanically. But do they matter in practice? Chapters 11 through 13 put Einlang side by side with PyTorch and NumPy on real code: LayerNorm, multi-head attention, Flash Attention, and physical simulation. No arguments. Just code, in two notations, side by side. The question is not "which is better." It is what each notation makes visible—and what each notation hides.
