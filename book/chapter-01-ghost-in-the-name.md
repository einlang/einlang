---
layout: book
title: "Chapter 1 · The Ghost in the Name"
---

# Chapter 1 · The Ghost in the Name

*Primitives · Naming and permutation*

---

> Every time you write `dim=-1`, you know what it means. The compiler doesn't. This is about what happens when it does.

Here is a story about a bug.

A tensor has shape `(32, 64, 256)`. The data loader author knows these dimensions are `batch`, `channel`, and `spatial`. There is a comment. There is a variable name: `spatial_features`. Then:

```python
x = x.mean(dim=1)
```

`dim=1` erases a dimension. At the time of writing, position 1 holds `channel`. The intent is "average over channels." The text says `dim=1`. A position. A number.

Three months later, the data pipeline is refactored. Channel moves to position 2. The new shape is `(32, 256, 64)`. `mean(dim=1)` now silently erases `spatial`. No errors. No warnings. The loss descends. The model deploys. The customer complaint arrives on Thursday.

**The notation had no slot for the fact that would have caught it.** The fact—"erasing `channel`, not `spatial`"—was in a comment, in a variable name, in the author's head. It was absent from the one place the compiler could see: the source text of the operation itself.

Positional notation is not wrong. It is insufficient. It records the arithmetic of shapes. It does not record the identity of coordinates. When a shape is correct but a coordinate is wrong, positional notation gives no place to notice. When shapes and types were both correct, what information was missing?

---

A coordinate has three properties. The framework records two of them. The third—the name—exists only in the programmer's head, in comments, or in variable naming conventions.

![Three properties of a coordinate: domain and position are checked; the name is not](figures/shape_meanings_gap.svg)

This is the shape-meanings gap. The shape says *how many*. The role says *which one*. Every framework knows the shape. None of them know the role.

Now imagine a different notation:

```rust
let y[b, s] = mean[channel](x[b, channel, s]);
```

The bracket after `mean` names the coordinate being **consumed**—eliminated from the output, its values collapsed into a single number. The brackets after `y` and `x` name the coordinates that survive. That `channel` exists on `x` is statically checked. The reader sees the consumption without reconstructing it. The fact that was previously in a comment—"average over channels"—is now in the syntax, where it can be enforced and the reader can audit it.

Now consider the reverse situation. Instead of eliminating a coordinate, suppose a value needs to be copied *along* one:

```rust
let out[b, c, s] = x[b, c, s] + bias[c];
```

`bias` is indexed only by `c`. It has no `b` and no `s` in its brackets. The absence declares: `bias` is silent on `b` and `s`. Its value is copied across every batch element and every spatial position. Not because broadcasting is a convenient default. Because the indexing pattern makes a semantic claim—`bias` does not depend on the batch or the spatial coordinate—and that claim is honored.

Think of a tensor as a person holding a **repeater** (a megaphone). `bias[c]` speaks on coordinate `c`: at `c=0` it says one value, at `c=1` another. On every coordinate not in its brackets—`b`, `s`—it says nothing. Silence. The notation, encountering this silence in the indexing pattern, fills it by repeating the value. The repetition is not a convenience feature. It is the notation honoring the promise that `bias` made by omitting those coordinates: "my value is independent of `b` and `s`. Ask me a thousand times with different `b`, and I give the same answer."

This is the megaphone model: a tensor speaks on the coordinates in its brackets, and stays silent on all others. Broadcasting is repeating the silent message wherever it is asked. Reduction is the inverse—pointing the megaphone at a coordinate and speaking it out of existence.

A coordinate has three properties. First, a **name**: `batch`, `channel`, `time`, `feature`. The name carries the semantic role. Second, a **domain**: the set of values the coordinate can take. For a tensor of shape `(32, 64, 256)`, the `batch` coordinate ranges from `0` to `31`, `channel` from `0` to `63`, and `spatial` from `0` to `255`. Third, a **position**: where this coordinate sits in the tensor's shape tuple. In `(32, 64, 256)`, `batch` is at position 0, `channel` at position 1, `spatial` at position 2.

Positional notation records only the domain and the position—`(32, 64, 256)` tells you the sizes and their order, but not their names. Named notation records all three: `[batch: 32, channel: 64, spatial: 256]`.

When you write `x.mean(dim=1)`, you are asking the position to stand in for the name. It works until the position changes. When you write `mean[channel](x)`, you are using the name directly. The position becomes an implementation detail—the compiler's problem, not yours.

---

## An Analogy: The Parking Lot

You park your car in Row D, Slot 7. The ticket in your pocket says "D-7." You return after dinner to find the lot has been repainted. The rows now run perpendicular to their old orientation. Row D is now somewhere else entirely. Your ticket, which records a *position* in a fixed coordinate system, sends you to the wrong car.

The lot's *shape* hasn't changed. It is still an 8 × 20 grid. A shape checker would tell you everything is fine. But the *role* of each row—which row is "D"—has moved.

This is what happens when you write `x.transpose(1, 2)`. The shape is still `(32, 256, 64)`. A shape checker sees the same three numbers. But the positions have been reassigned. Dimension 1 is no longer `channel`. Dimension 2 is no longer `spatial`. The ticket in your pocket—`dim=1`—now points to the wrong car.

A named-coordinate notation is like a ticket that says "the blue Honda Civic" instead of "D-7." The car may move, but the description finds it.

Now extend this analogy. Imagine the parking lot has three underground levels—B1, B2, B3—each an 8×20 grid. Your ticket says "D-7-B1": row D, slot 7, basement level 1. You return to find the lot has been renovated. The levels have been renumbered—B1 is now B3. The rows on each level have been rotated 90 degrees. Slot numbers run in the opposite direction. Your ticket now points to a space that doesn't exist, or worse, to someone else's car.

A shape checker would tell you the lot still has three levels of 8×20. Correct. The shape is the same. A position checker would tell you "D-7-B1" is a valid coordinate in the new system—it's just a *different* space. Also correct. Neither checker can tell you that your ticket describes the wrong space, because neither checker knows which space is yours.

This is what happens when you write `x.permute(1, 2, 0)` on a tensor with shape `(32, 64, 256)`. The shape after permutation is `(64, 256, 32)`. A shape checker sees three numbers and approves. But which dimension is batch now? Which is channel? The shape doesn't tell you. The permutation numbers don't tell you. They tell you that dimension 1 moved to position 0, dimension 2 to position 1, dimension 0 to position 2. They don't tell you that `channel` moved to the front, that `spatial` is now in the middle, that `batch` is now last.

A named permutation tells you all of that:

```rust
let y[c, s, b] = x[b, c, s];
```

You don't need to decode `(1, 2, 0)`. You read `y[c, s, b]` and you know: channel first, then spatial, then batch. If the lot gets renovated—if the upstream tensor changes its internal order—the named permutation still finds your car, because it looks for the blue Honda Civic, not D-7-B1.

The 3D parking lot teaches something the 2D version couldn't: when you compose multiple layout changes—renumbering levels AND rotating rows AND reversing slots—the positional ticket becomes wrong in multiple independent ways. Each way must be diagnosed separately. The named ticket is right about all of them simultaneously, because it never depended on any of them to begin with.

---

## Permutation: Moving Without Losing

A permutation changes the order of dimensions without changing any values. It is the simplest tensor operation there is—no arithmetic, no reduction, just relabeling positions.

It is also a reliable source of 11 PM debugging sessions.

The problem is not that permutation is hard. The problem is that positional permutation describes *mechanics* rather than *intent*. Here is a concrete example. An image-processing pipeline takes input in `(batch, height, width, channel)` and needs it in `(batch, channel, height, width)`:

```python
x = x.permute(0, 3, 1, 2)
```

The programmer writes this while looking at a diagram that says "channel moves from position 3 to position 1." The diagram is correct. The code is correct. Six months later, upstream changes its output convention to `(batch, width, height, channel)`. Height and width have swapped. `permute(0, 3, 1, 2)` still executes without complaint. Channel still ends up at position 1—correct. But height and width are now in positions the programmer did not intend. The shapes are identical. The values are wrong.

No shape checker catches this. No type checker catches this. The bug will surface in production as "the model is slightly worse on images with non-square aspect ratios," and it will take a human being several hours to trace the silent swap back to this one line.

The root cause: `(0, 3, 1, 2)` describes a rearrangement of *positions*. What the programmer needed to describe was a rearrangement of *identities*—"move the dimension called `channel` to the front, and keep everything else in order."

---

![Positional permute silently breaks when upstream changes; named permute survives](figures/permute_survival.svg)

The figure tests both notations against a common refactoring. Top row: the original pipeline maps BHWC to BCHW. On the left, `permute(0,3,1,2)`—read "old axis 0 stays at 0, old axis 3 moves to 1, old axis 1 moves to 2, old axis 2 moves to 3"—produces the correct result. On the right, the named expression `y[b,c,h,w] = x[b,h,w,c]` produces the same correct result. Both pass. Bottom row: upstream swaps height and width, so the input is now BWHC. The positional instruction executes identically—`permute(0,3,1,2)` is still the same four numbers—but the output is now B,C,W,H. Height and width are silently exchanged. The named expression `y[b,c,h,w] = x[b,h,w,c]` adapts automatically: `h` maps to the second axis in the input regardless of where height landed, `w` maps to the third. The instruction did not change. The meaning did.

Einops addresses this with a string-based notation:

```python
y = rearrange(x, "batch height width channel -> batch channel height width")
```

This is better. The names survive renaming of upstream positions, because `rearrange` matches by name, not by index. But the string is still a string. The names `height` and `width` are not checked against any declaration. They are local to this one call. If the tensor actually contains `time` rather than `height`, the string won't catch it—it will happily treat `time` as if it were `height`, because the names in the string are just pattern variables, not coordinate declarations.

What we want is for the coordinate names to be **checked facts**, not comments embedded in syntax:

```rust
let y[b, c, h, w] = x[b, h, w, c];
```

This is an Einlang rectangular declaration. The left-hand side declares the output coordinates. The right-hand side indexes the input by those same coordinate names. `b` appears on both sides in the same position—it survives unchanged. `h` appears on the left at position 2 and on the right at position 1—it has been moved. Every coordinate on the right is checked to exist on `x`, and every coordinate on the left must appear somewhere on the right.

You don't need a `permute` function. You don't need a `rearrange` string. You just write where each coordinate goes, and the movement is inferred. The code says *what you want*, not *how to achieve it*.

This is a pattern that will recur through the entire book: **when coordinate names appear in the syntax, operations become self-documenting.** The same line of code that instructs the machine also informs the reader. There is no separate channel of documentation that can drift out of sync.

---

Positional permutation is not evil. It is the right abstraction for a compiler pass that only needs to know "move this stride to that position." But source code is not written for compilers. It is written for the human who will debug it at 11 PM, three months after the original author left the team. That human needs to know *what moved where and why*. Position numbers answer the first question, but not the second. Names answer both.

---

---

> Silence is not absence. Silence is a claim. And claims can be checked.

Chapter 2 gives this claim its machinery: the megaphone model. A tensor speaks on some coordinates and stays silent on others. Reduction silences a coordinate. Broadcast copies along one the tensor was already silent on. Together they form a single rule—and the next chapter shows why that rule governs every tensor computation.
