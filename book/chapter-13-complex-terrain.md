---
layout: book
title: "Chapter 13 · Complex Terrain"
---

# Chapter 13 · Complex Terrain

> "Everything should be made as simple as possible, but not simpler."
>
> — Albert Einstein

*Applications · Splitting, merging, and self-interaction of coordinates*

---

A coordinate is not a fixed, atomic thing. It can split into two. Two can merge into one. One coordinate can interact with another copy of itself, playing two roles in the same expression. These operations are common in real tensor code—distance matrices, image patches, self-attention—and they are where the simple primitives of reduction and broadcasting meet the complex reality of tensor shapes.

This chapter is organized as a tour of coordinate interactions, from simple to complex. Each section introduces a pattern, shows the positional version (to see what gets hidden), and then shows the named-coordinate version (to see what becomes visible).

---

## Splitting Coordinates

A coordinate can be split into two. The classic case is a distance matrix: given `n` points, produce an `n × n` matrix of pairwise distances. The input coordinate `point` splits into `point_i` and `point_j`:

```rust
let diff[point_i, point_j, d] = points[point_i, d] - points[point_j, d];
let dist[point_i, point_j] = sum[d](diff[point_i, point_j, d] ** 2.0) ** 0.5;
```

`point_i` and `point_j` range over the same domain (the `n` points) but are independent coordinates. The subtraction `points[point_i, d] - points[point_j, d]` broadcasts naturally—`point_i` appears on the first term, `point_j` on the second, so the result is a 3D tensor indexed by `(point_i, point_j, d)`.

In a positional API, this requires `points.unsqueeze(1) - points.unsqueeze(0)` or `points[:, None, :] - points[None, :, :]`. The reader must infer from the `None` axes which coordinate is being split. The names `point_i` and `point_j` live in a comment.

The split-coordinate pattern recurs in attention (query `i` vs. key `j`), in graph neural networks (source node vs. target node), and in any computation that compares every element of a set to every other element.

---

## Merging Coordinates

Two coordinates can merge into one. A flattening of a 2D image grid into a 1D sequence is a merge of `height` and `width` into `position`:

```rust
let flat[b, pos] = image[b, pos / width, pos % width];
let image[b, h, w] = flat[b, h * width + w];
```

The expression `pos / width` and `pos % width` in the index position compute the original coordinates from the flattened one. The compiler verifies that these index expressions stay within bounds.

The reverse—unflattening—uses `h * width + w` to compute the flattened index from the original coordinates.

These index arithmetic expressions are the bridge between coordinate worlds. They are necessary because not all tensor layouts are simple rectangular grids. Sometimes a coordinate in one representation corresponds to a function of coordinates in another. The expressions make that function explicit.

---

## Self-Interaction: The `k` in Matrix Multiplication

Matrix multiplication `let C[i, j] = sum[k](A[i, k] * B[k, j])` involves a coordinate `k` that appears on both `A` and `B`. The compiler checks that `k` has the same range in both positions. This is the coordination contract: the two operands must agree on the size of the shared dimension.

This pattern generalizes. Any time a reduction coordinate indexes multiple tensors, the compiler enforces range agreement. This catches the class of bugs where two tensors are meant to share a coordinate domain but accidentally have different sizes. In a positional API, this is a runtime error (`RuntimeError: mat1 and mat2 shapes cannot be multiplied`). In a named-coordinate API, it is a compile error with the coordinate name in the message: "coordinate `k` has range 64 in A but range 128 in B."

---

## Depth-to-Space: Unpacking Coordinates

A striking example of coordinate restructuring is depth-to-space—the operation that rearranges a tensor from `(batch, channel * r * r, height, width)` to `(batch, channel, height * r, width * r)`. In a positional API, it's a chain of reshape and permute that is nearly impossible to read without a whiteboard:

```python
x = x.reshape(b, c, r, r, h, w)
x = x.permute(0, 1, 4, 2, 5, 3)
x = x.reshape(b, c, h * r, w * r)
```

In einlang, it's an index expression that states directly where each output cell reads its input:

```rust
let result[b, c, i, j] = input[b, c * (r*r) + (i % r) * r + (j % r), i / r, j / r];
```

`i / r` and `j / r` are the coarse spatial coordinates (which output cell). `i % r` and `j % r` are the fine spatial coordinates (which sub-pixel within the cell). The channel index in the input selects the right sub-pixel channel. All the information that required three operations and a whiteboard diagram is now in a single address equation.

This is not always the most readable form—the index arithmetic is dense. But it is *explicit*. The coordinate relationships are stated, not implied by a sequence of positional transforms. A reader can verify that every output address maps to a unique input address. A compiler can check that the indices stay within bounds.

The right notation for a given task depends on the task. Sometimes a named `rearrange` string is clearer. Sometimes explicit index arithmetic is clearer. The principle is the same: **the coordinate map should be visible in the source, not reconstructed by the reader.**

---

## Convolution as Index Arithmetic

A convolution is matrix multiplication with spatial structure:

```rust
let conv[b, oc, oh, ow] = sum[ic, kh, kw](
    input[b, ic, oh + kh, ow + kw] * weight[oc, ic, kh, kw]
);
```

The index expressions `oh + kh` and `ow + kw` slide the kernel over the input. The compiler infers the valid ranges of `oh` and `ow` from the constraint that `oh + kh` must stay within the input's spatial bounds.

Notice where the expressions live: in the body index positions, not in the declaration bracket. The declaration bracket names the output coordinates `b, oc, oh, ow`—simple identifiers. The body index expressions `oh + kh` describe how to compute the input indices from the output indices and the kernel coordinates. This is the same separation we saw in Chapter 11 for recurrences: the left side is a definition; the right side is a computation.

---

## The Declaration Bracket vs. the Body

By now a pattern should be clear. The declaration bracket (the left side of `let`) and the body index positions (the right side) have different rules:

**Declaration bracket**: only identifiers, literals, and named rests. These *name* what is being defined. They are the coordinates of the output tensor.

**Body index positions**: arbitrary expressions involving those identifiers and other coordinates. These *compute* the input address from the output address.

This separation is one of the most important design decisions in einlang. It keeps the declaration side simple—a list of coordinate names, each with an optional explicit domain. It puts all the complexity of index arithmetic on the computation side. The reader sees, at a glance, the coordinate structure of the result. The reader drills into the body only when they need to understand how that result is computed from the inputs.

---

## Packs for Batch Polymorphism, Revisited

All the patterns in this chapter—splitting, merging, self-interaction, convolution—work with arbitrary batch structure thanks to rest packs:

```rust
let dist[..batch, i, j] = sum[d](points[..batch, i, d] - points[..batch, j, d]) ** 2.0;
let conv[..batch, oc, oh, ow] = sum[ic, kh, kw](
    input[..batch, ic, oh + kh, ow + kw] * weight[oc, ic, kh, kw]
);
```

The `..batch` pack absorbs whatever batch dimensions the caller provides, leaving the operation's core coordinate logic unchanged. This is polymorphism through naming: the function doesn't need to know how many batch dimensions there are, only that they exist and should be preserved.

---

The patterns in this chapter are not exotic. They are the daily work of tensor programming. Every one of them can be written in positional notation. The argument of this book is not that positional notation is incapable. It is that positional notation hides the coordinate story, and the coordinate story is what catches the bugs.

A batched pairwise distance computation—`points[b, n, d]` to `dist[b, i, j]`—demonstrates how the batch coordinate `b` flows through the computation untouched by the pairwise logic, carried implicitly by rest packs. A 2D max-pooling on `input[b, c, h, w]` produces `output[b, c, oh, ow]` with `max[kh, kw]` and index arithmetic that names the pooling window explicitly. In both cases, the coordinate story is visible on the page—you can see which coordinates are consumed, which survive, and which are broadcast. Now consider a colleague who writes `let C[i, j] = sum[k](A[i, k] * B[k, i])`. The compiler accepts it—the shapes are valid—but `B[k, i]` contracts `k` with `A[i, k]` while keeping `i` as a surviving coordinate on both sides. The result is not the intended contraction. The bug is invisible to shape check but visible to coordinate check: your eye expects `B[k, j]` because the result promises `j`.
