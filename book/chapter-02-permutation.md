---
layout: book
title: "Chapter 2 · Moving Flowers Without Losing the Trail"
---

# Chapter 2 · Moving Flowers Without Losing the Trail

> "The map is not the territory."
>
> — Alfred Korzybski

*Primitives · Coordinate permutation*

---

A permutation is an operation that changes the order of dimensions without changing any values. It is the simplest tensor operation there is—no arithmetic, no reduction, just relabeling positions.

It is also a reliable source of 11 PM debugging sessions.

The problem is not that permutation is hard. The problem is that positional permutation is a description of *mechanics* rather than *intent*. The mechanics are correct. The intent is invisible. When upstream changes the mechanics—same operation, different positions—the intent silently drifts.

Here is a concrete example. An image-processing pipeline takes input in `(batch, height, width, channel)` and needs it in `(batch, channel, height, width)`:

```python
x = x.permute(0, 3, 1, 2)
```

The programmer writes this while looking at a diagram that says "channel moves from position 3 to position 1." The diagram is correct. The code is correct. Six months later, upstream changes its output convention to `(batch, width, height, channel)`. Height and width have swapped. `permute(0, 3, 1, 2)` still executes without complaint. Channel still ends up at position 1—correct. But height and width are now in positions the programmer did not intend. The shapes are identical. The values are wrong.

No shape checker catches this. No type checker catches this. The bug will surface in production as "the model is slightly worse on images with non-square aspect ratios," and it will take a human being several hours to trace the silent swap back to this one line.

The root cause: `(0, 3, 1, 2)` describes a rearrangement of *positions*. What the programmer needed to describe was a rearrangement of *identities*—"move the dimension called `channel` to the front, and keep everything else in order."

---

## An Analogy: Driving Directions

Two sets of driving directions get you from the library to the bakery.

Set A: "Turn left at the third light. Turn right at the second stop sign. It's the fourth building on the right."

Set B: "Turn left from Main Street onto Oak Avenue. Turn right from Oak onto Elm. It's the building between the post office and the park."

Set A is positional. It works until a road closes, a light is removed, or the city renumbers the blocks. Then "the third light" is a different intersection. Set B is named. It works regardless of how the roads are numbered, because it references identities that are stable under infrastructure changes.

![Positional permute silently breaks when upstream changes; named permute survives](figures/permute_survival.svg)

The figure tests both notations against a common refactoring. Top row: the original pipeline maps BHWC to BCHW. On the left, `permute(0,3,1,2)`—read "old axis 0 stays at 0, old axis 3 moves to 1, old axis 1 moves to 2, old axis 2 moves to 3"—produces the correct result. Marked OK. On the right, the named expression `y[b,c,h,w] = x[b,h,w,c]` produces the same correct result. Both pass. Bottom row: upstream swaps height and width, so the input is now BWHC. The positional instruction executes identically—`permute(0,3,1,2)` is still the same four numbers—but the output is now B,C,W,H. Height and width are silently exchanged. Marked Bug. The named expression `y[b,c,h,w] = x[b,h,w,c]` adapts automatically: `h` maps to the second axis in the input regardless of where height landed, `w` maps to the third. Marked OK. The instruction did not change. The meaning did.

`permute(0, 3, 1, 2)` is Set A. It says "take the dimension at old position 0 and put it at new position 0, take old position 3 to new position 1..." When upstream changes the dimension order, the positions shift, and the same permute arguments produce a different semantic result.

`y[b, c, h, w] = x[b, h, w, c]` is Set B. It says "put the coordinate named `c` at position 1, regardless of where it was before." When upstream changes, the coordinate name finds its target.

The difference is not syntax. It is durability.

Einops addresses this with a string-based notation:

```python
y = rearrange(x, "batch height width channel -> batch channel height width")
```

This is better. The names survive renaming of upstream positions, because `rearrange` matches by name, not by index. A reader can see what moved without reconstructing the position map.

But the string is still a string. The names `height` and `width` are not checked against any declaration. They are local to this one call. If the tensor actually contains `time` rather than `height`, the string won't catch it—it will happily treat `time` as if it were `height`, because the names in the string are just pattern variables, not coordinate declarations.

What we want is for the coordinate names to be **checked facts**, not comments embedded in syntax:

```rust
let y[b, c, h, w] = x[b, h, w, c];
```

This is an einlang rectangular declaration. The left-hand side declares the output coordinates. The right-hand side indexes the input by those same coordinate names. `b` appears on both sides in the same position—it survives unchanged. `h` appears on the left at position 2 and on the right at position 1—it has been moved. The compiler checks that every coordinate on the right actually exists on `x`, and that every coordinate on the left appears somewhere on the right.

You don't need a `permute` function. You don't need a `rearrange` string. You just write where each coordinate goes, and the compiler figures out the movement. The code says *what you want*, not *how to achieve it*.

This is our first encounter with a pattern that will recur through the entire book: **when coordinate names appear in the syntax, operations become self-documenting.** The same line of code that instructs the compiler also informs the reader. There is no separate channel of documentation that can drift out of sync.

---

Positional permutation is not evil. It is the right abstraction for a compiler pass that only needs to know "move this stride to that position." But source code is not written for compilers. It is written for the human who will debug it at 11 PM, three months after the original author left the team. That human needs to know *what moved where and why*. Position numbers answer the first question, but not the second. Names answer both.

In the next chapter, we introduce a more dramatic operation: reduction. A reduction does not merely rearrange coordinates. It eliminates one. And the name of the eliminated coordinate matters more than any other.

Take the permute bug from this chapter and write down a convention for your own code that would have prevented it. Does your convention rely on variable names? Comments? Assertions? Each of those decays over time—variables get renamed, comments rot, assertions get deleted when they fire too often. Now consider the einops equivalent: `rearrange(x, "batch time feature -> batch feature time")`. If `x` arrives with shape `(batch, feature, time)`, the rearrange still produces the correct result because it matches by name, not by position. The convention is enforced by the code, not by discipline.
