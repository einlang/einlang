---
layout: book
title: "Chapter 13 · The Edge of the Name"
---

# Chapter 13 · The Edge of the Name

> "The map is not the territory—but a good map tells you where the boundaries are."
>
> — Adapted from Alfred Korzybski

*Boundaries · What names can check, and what they cannot*

---

Every chapter so far has argued for putting the name in the bracket. This chapter argues for knowing when the name is not enough.

A coordinate name records identity. The compiler checks that identity is respected—that `channel` means the same thing everywhere it appears, that reductions consume declared coordinates, that function contracts match call sites. These checks prevent an entire class of bugs: the silent axis swap, the broadcast that drifts with the layout, the reduction that changes meaning without changing syntax.

They do not prevent every bug. And they were never intended to.

---

## Consistency and Correctness

The distinction that governs everything the compiler can and cannot do:

**Consistency** is internal. Does the coordinate story cohere? Does `channel` appear wherever the reduction claims it does? Does the function signature match the call site? The compiler can check consistency, because consistency is a relationship between declarations—and declarations are all in the source.

**Correctness** is external. Does the computation achieve what the programmer intended? Does `mean[channel](x)` express the right thing? The compiler cannot check correctness, because correctness is a relationship between the source and the programmer's intent—and intent is not in the source.

This distinction is not a weakness of named coordinates. It is the same distinction that separates type-checking from verification in every programming language. A type checker verifies that `int` and `string` are used consistently. It does not verify that the function computes the right answer. The compiler catches `"hello" + 3`. It does not catch `interest = principal * (1 - rate)` when the formula should be `interest = principal * (1 + rate)`. One is a type error. The other is a formula error. Both produce wrong results. Only one is caught.

Coordinate names extend this boundary. A reduction over `channel` where `spatial` was intended is a formula error—internally consistent, semantically wrong. A reduction over `class` where the tensor has no `class` is a consistency error—caught at compile time. The name makes the second kind of error visible. It does not prevent the first.

---

## The Wrong Name That's Consistent

Suppose the programmer writes:

```rust
softmax[batch](logits[batch, class])
```

`batch` is the coordinate argument. It exists on `logits`. The reduction consumes it. The gradient sums over it. Every check passes. The program compiles.

It produces a valid probability distribution—over the batch dimension, not the class dimension. The name was wrong. The check passed. The program is incorrect.

But the name `batch` is visible. When the next programmer reads `softmax[batch](logits)`, they see the error immediately: "this normalizes over batch, not class." The positional equivalent `softmax(logits, dim=0)` hides the error behind a number. The reader sees `dim=0` and must reconstruct whether axis 0 is batch or class. The reconstruction may be wrong.

A wrong name is a visible error. A missing name is an invisible one.

The compiler cannot read the programmer's mind. `softmax[batch]` is consistent and wrong. `softmax(logits, dim=0)` is also consistent and wrong—but it is additionally anonymous. The named version gives the reviewer a handle. The positional version gives them an integer. The integer is always correct as an integer. It is wrong as a coordinate reference. But the notation has no slot for that fact.

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

Three categories exhaust the space beyond the boundary.

**1. The wrong coordinate.** `mean[channel](x)` where `spatial` was intended. The name exists. The reduction is well-formed. The coordinates are consistent. The programmer chose the wrong one. No static system can catch this—it would need to read the programmer's mind. What the name provides: visibility. A reviewer reading `mean[channel](x)` can ask "should this be `spatial`?" A reviewer reading `x.mean(dim=1)` must first deduce which coordinate `dim=1` refers to, then ask the same question. The name shortens the distance between the code and the question.

**2. Index arithmetic bounds — when ranges are dynamic.** `input[b, c, oh + kh, ow + kw]`—the compiler checks that `oh`, `kh`, `ow`, `kw` are declared coordinates. When the coordinate domains are statically known (e.g., `oh` in `0..32`, `kh` in `0..3`, input width = 35), the compiler can verify that `oh + kh` stays within bounds: it solves the constraint `oh + kh < 35` given `kh_max = 3`, yielding `oh < 32`, which matches the declared domain of `oh`. The check is compile-time.

When domain sizes are dynamic—unknown until runtime—the compiler cannot fully verify the bound. It still records the constraint and emits a runtime guard. The guarantee in both cases is that the coordinate names are in the error: no matter whether the bound is checked at compile time or runtime, an overflow reports which coordinate expression overflowed (`oh + kh`), which operands contributed, and what the bound was. The positional equivalent—`IndexError: dimension 3 out of bounds`—names a position. You then count dimensions, reconstruct the index arithmetic, and solve for which coordinate caused the overflow. The name doesn't always prevent the error. It always names it.

**3. Runtime-dependent shapes.** Dynamic dimensions—sequence lengths that vary per batch element, numbers of detected objects per image—cannot be fully verified at compile time. The compiler can check that `seq` is a declared coordinate and that functions consuming it have consistent contracts. It cannot check that `seq` has range 0..15 in one batch element and 0..23 in the next. That check lives at runtime, in an assertion or a shape guard. The name is there when the runtime check fires: `AssertionError: seq coordinate has range 0..23, expected 0..15`. The integer is there: `AssertionError: axis 1 out of bounds`. The name makes the error message searchable.

---

## The Positive Framing

The boundary is often framed as a limitation: what names *can't* do. Turn it around. What the compiler *can* check, it checks exhaustively. What it cannot check, it makes visible. The coordinate names are in the index expressions when bounds are checked at runtime, in the declaration when the shape is resolved, in the function signature when the meaning is asserted. The names are present at every check—compile-time or runtime.

This is what "everything checkable is checked" means. The compiler verifies every fact derivable from declarations. The facts it cannot verify, it leaves visible—with names attached—so that runtime checks and human reviewers know what to look for. A name is more useful when checked at compile time, but it is still useful when checked only at runtime.

---

## The Middle Ground

Between pure positional notation and a complete named-coordinate compiler lie several intermediate solutions. Each catches some bugs. None catches all.

**Defensive assertions.** `assert x.shape[1] == channel_size` catches a refactoring when the sizes differ. But if `channel` and `spatial` happen to have the same extent, the assertion passes silently. Assertions check shapes, not identities. They protect against size mismatches, not semantic drift.

**Einops.** `reduce(x, "batch channel spatial -> batch spatial", "mean")` names the reduced coordinate at the call site. Einops is locally excellent—within a single expression, it records coordinate identities with clarity comparable to Einlang. The string notation is minimal, library-level, and works in today's PyTorch/JAX/NumPy without a custom compiler. For many projects, einops alone eliminates the most common class of positional bugs.

But einops strings are strings. The names `batch`, `channel`, `spatial` are not checked against any declaration. If the tensor actually contains `time` rather than `channel`, the string won't catch it—it will treat `time` as if it were `channel`. And einops names do not propagate across function boundaries. A function that receives an einops-rearranged tensor has no way to know what the dimensions are called. The name dies at the edge of the expression. Every function that receives the tensor must re-discover or re-declare the coordinate identities. Einlang makes the names part of the function's type-level contract, so they survive composition.

**PyTorch Named Tensors.** `x.refine_names("batch", "channel", "spatial").mean("channel")` checks that `channel` exists and eliminates it. But many operations strip names silently—`torch.matmul`, `torch.cat`, most `torch.nn` layers. When a name is stripped, the protection vanishes without warning. The contract is partial. The check is partial. The name can fall off without the programmer knowing.

**Einlang's compiler.** The coordinate contract is part of the function type. Every call site is checked. Every operation preserves names or explicitly consumes them. If an operation strips a name, the compiler reports a contract violation at the call site. The contract is global. The check is complete.

The distance between *no checking* and *complete checking* is measurable. Every step along that distance catches more bugs. Einops catches the local ones. Named tensors catch the ones that survive through supported operations. Einlang catches all of them—at the cost of a compiler that must be built. Which step you take depends on how much correctness you need and how much infrastructure you can afford.

The coordinate habit works at every step. It only asks: *is the name in the code?* In an einops string, that's a name. In a PyTorch named tensor, that's a name. In a bracket that a compiler checks, that's a name with a guarantee. The habit does not prescribe the tool. It prescribes the information.

---

## What Survives

Names do not eliminate runtime shape errors. They do not replace testing. They do not guarantee correctness. They do not cost zero keystrokes.

They prevent one class of error: the error where the coordinate identity exists in the programmer's head but not in the source text, and the notation provides no place to record it. For that class—the silent axis swap, the broadcast that drifts with the layout, the reduction that changes meaning without changing syntax—names are the only defense. For errors outside that class, other defenses apply.

The boundary is not a flaw. It is a map of what is statically knowable. Every fact the names cannot check is a fact that no purely static system can check. The names don't fail at these boundaries. They mark them.

The coordinate you name in the bracket is the coordinate the compiler checks. The coordinate you don't name is the coordinate the compiler can't. The difference is not between safety and danger. It is between checked and unchecked. Between visible and invisible. The bracket makes the boundary explicit. The integer leaves it implied. And implied boundaries are the ones that get crossed at 3 AM, by a bug that survived three weeks because the notation had no slot for the fact that would have caught it.

The boundary is drawn. What remains is to look back at the whole—the primitives, the combinations, the construction, the comparisons—and ask what was built. The Appendix collects every error code, every check rule, and a complete program. The Epilogue returns to the Tuesday bug from Chapter 1, replays Day 100 with a working compiler, and asks the question the book was always asking: what changes when a name has a place to live?
