---
layout: book
title: "Chapter 12 · The Guardian's Booklet"
---

# Chapter 12 · The Guardian's Booklet

> "We are what we repeatedly do. Excellence, then, is not an act, but a habit."
>
> — Will Durant, paraphrasing Aristotle

*Meta-cognition · Four habits, independent of syntax*

---

This chapter contains no new syntax.

It contains something more portable: four questions. You can ask them in any tensor framework, in any language, with any notation. They require no tooling beyond your attention. They catch the class of bugs that shape checks cannot.

The preceding eleven chapters earned these questions one at a time. Each question emerged from a concrete bug story—a silent swap, an invisible broadcast, a reduction that consumed the wrong coordinate. The questions are the permanent residue of those stories. The stories are evidence. The questions are the habit.

---

## Habit 1: Eliminate with a Name

**The question:** *Which coordinate did this operation eliminate? Is the name still in the code?*

Every reduction—`sum`, `mean`, `max`, `min`—destroys a coordinate. The coordinate's variation is collapsed into a single value. That value is then used downstream, where its meaning depends entirely on which coordinate was collapsed.

In a positional API, the eliminated coordinate is identified by a number: `dim=1`, `axis=-1`, `keepdims=False`. The number identifies a position. The position is a function of the current tensor shape. When the shape changes—as it does during refactoring, pipeline updates, or model architecture experiments—the number silently refers to a different coordinate. The code is still correct. The meaning is wrong.

In a named-coordinate notation, the eliminated coordinate is identified by name: `sum[channel](...)`, `mean[time](...)`, `max[class](...)`. The name is stable under shape changes. If the coordinate doesn't exist, the compiler reports an error. If the coordinate exists but shouldn't be eliminated, the reader can audit it.

**The habit:** When you see a reduction, pause. Say aloud (or in your head) which coordinate is being eliminated. If the answer is "the one at position 2," ask: why position 2? What is its name? If the name is not in the code, write it in a comment. Better: find a way to put it in the syntax. If your framework has named dimensions (einops, xarray, etc.), use them. If not, the comment is your bridge.

---

## Habit 2: Copy with a Signature

**The question:** *Which coordinate is this operation copying along? Is the copy explicit in the code?*

Every broadcast—implicit or explicit—copies a value along a coordinate. The copy is a semantic claim: the value is independent of that coordinate. The claim is checked by the shape system (the sizes must align) but not by the semantic system (the value must genuinely be independent).

In a positional API, broadcasting is inferred from shape compatibility. `x + bias` where `x` is `(32, 64)` and `bias` is `(64,)` broadcasts `bias` along dimension 0. Which coordinate is that? You need to know the shapes to answer. If the shapes change upstream, the broadcast target changes silently.

In a named-coordinate notation, broadcasting is visible in the indexing pattern: `let out[i, j] = A[i, j] + bias[j]`. The coordinate `i` appears on `A` and `out` but not on `bias`. The absence is the broadcast. The coordinate being copied along is `i`—visible, named, auditable.

**The habit:** When you see a binary operation between tensors of different shapes, pause. Identify which coordinate is being broadcast over. Ask: is independence from that coordinate genuinely justified? If the broadcast is wrong, will the code error, or will it silently produce a valid-but-wrong result? If the latter, make the broadcast explicit—with a comment, a named dimension, or an explicit `expand` call that names the axis.

---

## Habit 3: Permute with a Source

**The question:** *Where did this coordinate come from, and where is it going? Can you trace the route without reconstructing position numbers?*

Every permutation—`transpose`, `permute`, `reshape`, `rearrange`—moves coordinates. The movement is a semantic claim: "this coordinate, which used to mean X, now means Y." The claim is stated in the permutation arguments: `permute(0, 3, 1, 2)`. The arguments describe how positions change. They do not describe what the positions mean.

In a named-coordinate notation, permutations are implicit in the indexing pattern: `let y[b, c, h, w] = x[b, h, w, c]`. The coordinate `c` appears at position 1 on the left and position 3 on the right. The movement is visible. No `permute` call is needed.

**The habit:** When you see a `permute`, `transpose`, or `reshape` chain, pause. Trace one coordinate from source to destination. Can you do it without writing down the intermediate shapes? If not, the code is hiding information that should be visible. Write a comment showing the coordinate map. Better: use einops `rearrange` with named axes. Best: use a notation where the coordinate names are the primary addressing mechanism, and positions are derived from them.

---

## Habit 4: Forward and Backward, Symmetric

**The question:** *The forward pass eliminated a coordinate—how does the backward gradient handle it? Are the two directions symmetric?*

Every forward broadcast becomes a backward reduction. Every forward reduction becomes a backward broadcast. The coordinates that were consumed in the forward pass reappear in the backward pass as the dimensions over which gradients are summed. If the forward pass consumed the wrong coordinate, the backward pass sums over the wrong dimension—and the gradient has the correct shape but the wrong values.

In a positional API, the symmetry is invisible. You call `loss.backward()` and trust that the autodiff engine computed the right thing. If it computed the wrong thing because a coordinate was silently swapped upstream, you discover the error through anomalous gradients—small but systematic, easy to dismiss as "noisy training."

In a named-coordinate notation, the gradient request names the coordinates it differentiates with respect to: `@loss / @W[out, in]`. The denominator's coordinates determine the result's coordinates. The compiler derives the reduction pattern from the coordinate sets, using the same rule from Chapter 8.

**The habit:** When you inspect a gradient, check its shape against the parameter it updates. If the shapes match, check the coordinates they represent. A gradient summed over `batch` when it should have summed over `feature` has the right shape but wrong semantics. Ask: which coordinates were consumed in the forward pass? Which are being summed in the backward pass? Are they the same?

---

## These Habits Are Not About einlang

Einlang is the microscope we used to examine the specimens. The habits are what you take home.

You can practice them in PyTorch:

```python
# Instead of:
x.mean(dim=1)

# Write:
x.mean(dim=1)  # dim 1 = channel
```

The comment is not as good as compiler-checked syntax. But it is better than nothing. It is a commitment, written down, that `dim=1` means `channel`. When upstream changes the dimension order and you update the code, the comment becomes a lie—and a lie in a comment is a bug you can notice, whereas a silent drift in a positional argument is not.

You can practice them in JAX with einops. You can practice them in NumPy with docstrings. You can practice them in any framework that gives you a place to put a name and a discipline to keep it honest.

The habit is free. The habit is portable. The habit catches bugs that shape checks miss.

---

## The Principles Beneath the Habits

The four habits are not arbitrary rules. They follow from a single principle that has guided the entire book:

> **The Hiding Law.** Do not hide a fact that later reasoning must recover.

Eliminated coordinates, broadcast targets, permutation routes, and gradient addresses all pass this test. They are easy to state when the formula is written. They are expensive—sometimes impossible—to rediscover after the formula has been lowered into axis numbers, layout operations, or execution traces.

Other facts are good candidates for hiding. Register allocation. Temporary buffer reuse. Device placement. Kernel fusion order. Tiling shape. Vector width. These matter for performance, but they should not be facts a reader must recover to know whether `class` or `batch` was normalized.

The boundary: hide implementation. Show semantics. The hard part is knowing which is which.

**The Deletion Test.** Delete every comment and every variable name from a piece of tensor code. What facts remain in the syntax itself? If the answer does not include "which coordinate was reduced," "which coordinate was broadcast," and "which coordinate was the gradient taken with respect to," the notation is hiding facts.

Apply the Deletion Test to `x.mean(dim=1)`: only the position of the eliminated coordinate survives. The name is gone. Apply it to `mean[channel](x)`: the name survives. The test is not about comments being bad. It is about syntax being the only channel that the compiler can enforce and that refactoring tools preserve.

**The Placement Rule.** The programmer owns coordinates. The compiler owns lowering. When coordinate roles are recorded in the source, the compiler can read them for shape inference, gradient lowering, storage planning, and kernel fusion—each pass recovering the same facts from the same names, rather than reconstructing them from access patterns. The names are the single source of truth. The compiler passes are readers of that truth.

This is not a theoretical distinction. It is a concrete design choice for any tensor notation: put the coordinate facts in the syntax, or put them in the documentation. The former makes the compiler a co-reader. The latter makes the compiler a blind executor and the human the sole bearer of meaning.

---

The remaining chapters are exercises in applying the four habits with these principles as your compass.

Take a piece of tensor code you wrote recently and audit it with the four questions. You will likely find a broadcast whose target coordinate wasn't obvious, a reduction whose eliminated coordinate you had to reconstruct from the shape, a permutation you traced by writing down intermediate positions on a scratch pad. The four habits are interdependent: a broadcast made explicit (Habit 2) makes the backward reduction explicit (Habit 4); a reduction named (Habit 1) makes the backward broadcast named (Habit 4). Fix one, and another often fixes itself—because they are not four separate rules so much as four faces of the same idea. A skeptic says: "I just use einops for everything. Problem solved." Einops gives you named permutation and the string as a readability aid. But it does not check coordinate consistency across operations, it does not ground broadcast or reduction in coordinate names, and it leaves the semantic story—which coordinate means what, which is consumed, which is omitted—to comments and memory. The coordinate habit is not a library. It's an expectation that the notation carries the meaning, and that the compiler checks it.
