---
layout: book
title: "Chapter 14 · The Edge of the Name"
---

# Chapter 14 · The Edge of the Name

> "The map is not the territory—but a good map tells you where the boundaries are."
>
> — Adapted from Alfred Korzybski

*Boundaries · What names can check, and what they cannot*

---

Suppose the programmer writes:

```rust
softmax[batch](logits[batch, class])
```

It compiles. Every name matches. `batch` exists on `logits`. The reduction consumes a declared coordinate. The contract is satisfied. The shapes are correct.

But the programmer meant `softmax[class]`. Normalizing over batch instead of class is a bug—a model that silently computes the wrong probabilities, descends a valid loss, and deploys with a semantic error no tool can detect.

The compiler cannot catch this. And it was never intended to.

---

## Consistency and Correctness

The distinction that governs everything the compiler can and cannot do:

**Consistency** is internal. Does the coordinate story cohere? Does `channel` appear wherever the reduction claims it does? Does the function signature match the call site? The compiler can check consistency, because consistency is a relationship between declarations—and declarations are all in the source.

**Correctness** is external. Does the computation achieve what the programmer intended? Does `mean[channel](x)` express the right thing? The compiler cannot check correctness, because correctness is a relationship between the source and the programmer's intent—and intent is not in the source.

This distinction is not a weakness of named coordinates. It is the same distinction that separates type-checking from verification in every programming language. A type checker verifies that `int` and `string` are used consistently. It does not verify that the function computes the right answer. The compiler catches `"hello" + 3`. It does not catch `interest = principal * (1 - rate)` when the formula should be `interest = principal * (1 + rate)`. One is a type error. The other is a formula error. Both produce wrong results. Only one is caught.

Coordinate names extend this boundary. A reduction over `channel` where `spatial` was intended is a formula error—internally consistent, semantically wrong. A reduction over `class` where the tensor has no `class` is a consistency error—caught at compile time. The name makes the second kind of error visible. It does not prevent the first.

Return to the opening example. `batch` exists on `logits`. The reduction consumes it. Every check passes. The program compiles. It produces a valid probability distribution—over the batch dimension, not the class dimension. The name was wrong. The check passed. The program is incorrect.

But the name `batch` is visible. When the next programmer reads `softmax[batch](logits)`, they see the error immediately. The positional equivalent `softmax(logits, dim=0)` hides the error behind a number. The reader sees `dim=0` and must reconstruct whether axis 0 is batch or class. The reconstruction may be wrong.

A wrong name is a visible error. A missing name is an invisible one.

---

## What Names Check (Recap)

Before listing what names cannot check, recall what they can. The five rules from Chapter 9:

1. **Index Existence.** Every coordinate in an index list must exist on the tensor.
2. **Reduction Consistency.** The consumed coordinate must appear in every operand.
3. **Broadcast Recording.** Every omission is recorded for the backward pass.
4. **Causality.** Recurrence references must be strictly backward in time.
5. **Coordinate Contract.** Function call coordinate arguments must match the declaration.

Every one of these checks is a consistency check. Every one operates on names declared in the source. None requires data. None requires execution. All five together define the boundary: *if a bug can be expressed as a mismatch between two declarations in the source, the compiler catches it. If the bug lives entirely in the gap between the source and the programmer's intent, the compiler cannot.*

---

## What Names Cannot Check

Three categories. Each one is a different kind of silence between what the code says and what the programmer meant.

### 1. The Wrong Name

A programmer writes:

```rust
let result[..b, feature] = softmax[batch](logits[..b, feature]);
```

`batch` exists on `logits`. The reduction consumes it. The gradient sums over it. Every check passes. The program compiles. It produces a valid probability distribution — over the batch dimension, not the feature dimension. The name was wrong. The check passed. The program is incorrect. But the wrong name sits in the bracket, visible to the next programmer who reads `softmax[batch](logits)` and immediately asks: "should this be `feature`?"

Now the positional version:

```python
result = torch.softmax(logits, dim=0)
```

`dim=0` is valid. The shapes match. The output is a valid probability distribution — over axis 0, which might be batch, might be feature, might be something else entirely. The reader sees `dim=0` and must reconstruct which coordinate axis 0 maps to. That reconstruction may be wrong. And even if it's right, the code records nothing to confirm it.

A wrong name is a visible mistake. A missing name is an invisible one. The compiler cannot prevent you from choosing the wrong coordinate — that would require reading your mind. But it can make the choice visible. And visibility is the difference between a bug found in review and a bug found in production.

### 2. What Arithmetic Cannot Promise

You write a convolution:

```rust
let conv[b, oc, oh, ow] = sum[ic, kh, kw](
    input[b, ic, oh + kh, ow + kw] * weight[oc, ic, kh, kw]
);
```

The compiler checks that `oh`, `kh`, `ow`, `kw` are declared coordinates. It knows their domains. If `oh` ranges over `0..32`, `kh` over `0..3`, and `input` has width 35, the compiler proves: worst case `oh + kh = 31 + 3 = 34 < 35`. Safe. The check is compile-time.

Now the same expression with runtime dimensions. The image is read from a file. The input width is a variable. The compiler cannot prove `oh + kh < input_width` because `input_width` is not a constant. It emits a runtime guard: `assert oh + kh < input_width`. The name is in the guard: `IndexError: oh + kh = 67 exceeds input width 64`. The coordinate names the expression that overflowed. The positional equivalent: `IndexError: dimension 3 out of bounds`. Which dimension is 3? What was supposed to be in bounds? The number answers neither question.

The boundary here is not about names. It is about static vs. dynamic knowledge. The compiler proves everything provable from declarations. What it cannot prove, it guards. The name is in the error either way.

### 3. The Shape That Isn't There Yet

A beam search decoder. The output sequence length depends on the input — sentences are different lengths. The compiler knows `seq` is a coordinate. It does not know the length of `seq` for any particular input. That length is the answer.

This is not a failure of the compiler. It is the nature of the computation. The output shape is data-dependent. No static system can determine it. The name is still there — `seq` names the coordinate whose length varies. The runtime code that handles the dynamic length reads `seq_len = lengths[b]`. The name connects the declaration to the dynamic bound. The positional equivalent: `axis 1 has dynamic extent`. Axis 1 might be `seq`, `class`, `head`, or `feature`. The number doesn't know. The name does.

---

## Three Kinds of Silence

Step back from the three categories and they resolve into a single taxonomy of what the compiler cannot say.

**The silence of intent.** You chose `batch` when you meant `feature`. The code is consistent. It is wrong. No static check can close the gap between what you wrote and what you meant. What the name provides: the gap is visible. `softmax[batch](logits)` wears its intent in a bracket. `softmax(logits, dim=0)` wears it in a convention. Conventions can be forgotten. Brackets cannot.

**The silence of data.** The input width is a variable. The sequence length varies per sample. The output shape is the answer. These facts are not known at compile time because they are not knowable at compile time. The compiler checks everything that can be checked from declarations alone. What remains is guarded at runtime — with names attached. The name doesn't prevent the runtime check. It makes the runtime check readable.

**The silence of the world.** You are simulating the Navier-Stokes equations. The code compiles. The shapes match. The simulation runs. It produces negative absolute temperatures. The coordinate names are correct — `u`, `v`, `p` are different tensors, `i` is x, `j` is y. Every check passes. And the physics is wrong because the discretization is unstable at the chosen Reynolds number. No programming language catches this. The boundary is not between named and positional notation. It is between computation and reality. Names record identities. They do not replace physics.

The first silence is the one the book has spent fourteen chapters arguing against — the silence where the notation refuses to record a fact the programmer knows. The second silence is honest: the compiler checks what it can, guards what it can't, and names both. The third silence is the edge of the map. Here there be dragons — and no notation, named or positional, will chart them for you. But the map that marks the boundary honestly is better than the map that pretends the boundary does not exist.

---

## The Middle Ground

Between a purely positional notation and a complete named-coordinate compiler lie intermediate solutions. Each records more than integers. None records everything.

**Defensive assertions.** `assert x.shape[1] == channel_size` catches a refactoring when the sizes differ. But if `channel` and `spatial` happen to have the same extent, the assertion passes. Assertions check shapes, not identities. They protect against size mismatches, not semantic drift.

**Einops.** `reduce(x, "batch channel spatial -> batch spatial", "mean")` names the reduced coordinate at the call site. The string notation is minimal, library-level, and works in PyTorch today. For many projects, einops alone eliminates the most common positional bugs. But the names in the string are not checked against any declaration. If the tensor actually contains `time` rather than `channel`, the string won't catch it. And einops names die at the edge of the expression — the next function receives an anonymous tensor and must re-discover what the dimensions are called.

**PyTorch Named Tensors.** `x.refine_names("batch", "channel", "spatial").mean("channel")` checks that `channel` exists. But many operations strip names silently — `torch.matmul`, `torch.cat`, most `nn` layers. When a name is stripped, the protection vanishes without warning.

**Einlang.** The coordinate contract is part of the function type. Every call site is checked. Every operation preserves names or explicitly consumes them. The contract is global. The check is complete. The cost is a compiler.

The distance between *no checking* and *complete checking* is measurable. Einops catches local bugs. Named tensors catch the ones that survive through supported operations. Einlang catches all of them. Which step you take depends on how much correctness you need and how much infrastructure you can afford. The coordinate habit works at every step. It only asks: *is the name in the code?* In an einops string, that's a name. In a bracket a compiler checks, that's a name with a guarantee. The habit does not prescribe the tool. It prescribes the information.

---

Five bugs. Which ones does the compiler catch?

`softmax[class](logits[batch, channel])` — `class` is not on `logits`. Caught.

`softmax[batch](logits[batch, class])` — `batch` exists, every check passes. Not caught. The wrong name is visible in the bracket, but the compiler cannot read intent.

`sum[head](x[batch, head, d])` where the programmer meant `sum[d]`. `head` exists. Not caught. Same structure: the name is wrong, the check passes.

A convolution where `oh + kh` exceeds the input width, width is a runtime variable. The compiler can't prove the bound statically. It emits a runtime guard with the coordinate name in the error. Not caught at compile time — guarded at runtime, with the name attached.

`mean[spatial](x[batch, channel, spatial])` where `x` actually has `{batch, channel, time}` — `spatial` was renamed in a refactoring. Caught. Missing coordinate.

The compiler catches two, guards one, and is silent on two. The two it misses are the ones where the name is consistent but wrong. No static system catches those. But in the named versions the wrong name is in the bracket. In the positional versions it is not in the code at all.

---

## What Survives

Names do not eliminate runtime shape errors. They do not replace testing. They do not guarantee correctness.

They do one thing: they prevent the class of errors where the coordinate identity exists in the programmer's head but not in the source text, because the notation provides no place to record it. The silent axis swap. The broadcast that drifts with the layout. The reduction that changes meaning without changing syntax. For that class, names are the only defense.

The boundary is not a flaw. It is a map of what is statically knowable. Every fact names cannot check is a fact no purely static system can check — the programmer's intent, the runtime data, the physics of the world. The names don't fail at these boundaries. They mark them. And a marked boundary is one you don't cross by accident.

This is the argument the book has been making since Chapter 1. Not that names make programming safe. That names make identity visible. And visibility is the difference between a bug found in review and a bug found at 3 AM, three weeks after it shipped, because the notation had no slot for the fact that would have caught it.

---

But before you close the cover, the Appendix assembles the complete grammar and every check rule in one place — the view from the summit after fourteen chapters of climbing.

Before you close the book, open your most recent tensor code. Find a `dim=` argument. Ask: which coordinate is that? If you can't answer from the code alone, the coordinate isn't in the code. It's in your head. The name is missing. And the name is the only thing that can go in the bracket.

The Appendix collects every check rule, every error code, and a complete program. The Epilogue returns to the Tuesday bug from Chapter 1 — replays Day 100 with a working compiler — and asks the question the book was always asking: what changes when a name has a place to live?
