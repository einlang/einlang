---
layout: book
title: "Chapter 7 · Complex Terrain"
---

# Chapter 7 · Complex Terrain

> "The map is not the territory—but a good map tells you where the boundaries are."
>
> — Adapted from Alfred Korzybski

*Combinations · Distance matrices, convolution, and fancy indexing*

---

So far our coordinates have been simple: one name per axis, each axis independent. Real tensor programs are messier. A coordinate can split in two. It can carry arithmetic. It can appear in patterns where the same name used twice means something different from two names used once.

---

## Distance Matrix: When One Coordinate Becomes Two

Given a set of points, compute the pairwise distance between every pair. The same coordinate—`point`—appears twice, playing two roles: once as the source, once as the target.

```rust
let dist[point_i, point_j] =
    sum[dim]( (points[point_i, dim] - points[point_j, dim]) ** 2.0 ) ** 0.5;
```

`point` has been split into `point_i` and `point_j`. Both index the same underlying domain—the set of points. But `point_i` walks the rows of the distance matrix and `point_j` walks the columns. The naming makes the split visible. That both index into the same coordinate domain is checked. The reader sees that `point` was duplicated, not two unrelated coordinates that happen to share a prefix.

In a positional framework, this split is invisible:

```python
dist = ((points[:, None, :] - points[None, :, :]) ** 2).sum(-1) ** 0.5
```

`None` inserts a dimension. `:` slices everything. Which dimension is `point_i` and which is `point_j`? You have to count positions. If `points` changes from `(N, D)` to `(D, N)`, the positional code silently computes the wrong matrix.

The coordinate-split pattern appears in many domains. Once you recognize it, you see it everywhere:

**Graph neural networks.** `source_node` and `target_node` split from the same node domain. Edges connect `source_node` to `target_node`. The adjacency matrix has `(source_node, target_node)` coordinates. A message-passing step gathers from `target_node` and scatters to `source_node`. In positional code, both are axis 0 and axis 1. In named code, they carry different identities even when they index the same domain.

**Contrastive learning.** `anchor` and `positive` index the same set of samples. The similarity matrix has `(anchor, positive)` coordinates. The loss pulls `anchor[i]` close to `positive[i]` (pairwise) and pushes `anchor[i]` away from `positive[j]` for `i != j`. The coordinate split `anchor`/`positive` distinguishes "same sample, different view" from "different sample." In positional code, the distinction is in the mask tensor, not in the coordinate structure.

**Collaborative filtering.** `user` and `item` share a latent factor `k` through a matrix factorization: `ratings[user, item] = sum[k](U[user, k] * V[item, k])`. The inner dimension `k` is shared and consumed. `user` and `item` are different coordinates indexing different domains. In positional code, `U @ V.T`—all three identities (`user`, `item`, `k`) collapsed into positions.

The names record what was split. The positional code records only the mechanic of `unsqueeze`.

---

## When Names Collide: Renaming Across Conventions

Naming coordinates prevents one class of error—the silent axis swap. But it introduces a new problem: what happens when two tensors use different names for the same coordinate?

A tensor from the data loader uses `[batch, class]`. A tensor from a pre-trained model uses `[sample, category]`. They describe the same dimensions—`batch` and `sample` both mean "which example"; `class` and `category` both mean "which output label." But the names differ. If you try to add them, the compiler reports a coordinate mismatch. The names that were supposed to prevent errors are now preventing a valid operation.

This is not a flaw in named coordinates. It is the same problem that variable names have always had: two programmers call the same thing by different names. The solution is the same: a **rename** operation.

```rust
// y uses [sample, category]; we need [batch, class]
let aligned[batch, class] = y[sample -> batch, category -> class];
```

The arrow `->` reads as "renamed to." The compiler checks that `sample` exists on `y`, that `batch` is the new name, and that the domains have matching sizes. After this line, `aligned` carries `[batch, class]` and can be used with any other tensor that shares those names.

A rename is not a permutation—the coordinates stay in the same positions. It is not a reshape—the shapes are unchanged. It is a name change, and only a name change. The compiler verifies that every old name exists on the source and every new name is distinct from the others. After renaming, the new names participate in coordinate contracts exactly as if they had been declared from the start.

This pattern appears wherever tensors cross a boundary between naming conventions. A data loader produces `[N, C, H, W]` but a vision model expects `[batch, channel, height, width]`. A library returns `[query, key]` but the calling code uses `[seq_q, seq_k]`. In each case, a rename at the boundary is a one-line bridge. The cost is a single line. The benefit is that the compiler checks the bridge—it cannot silently map the wrong coordinate to the wrong name.

---

## Convolution: Coordinates with Arithmetic

A convolution is a sum of products with index arithmetic:

```rust
let conv[b, oc, oh, ow] = sum[ic, kh, kw](
    input[b, ic, oh + kh, ow + kw] * weight[oc, ic, kh, kw]
);
```

The novelty is `oh + kh` and `ow + kw`. These are index expressions—arithmetic on coordinate variables. `oh + kh` says: to read from the input at spatial position `oh + kh`, take the output position `oh` and add the kernel offset `kh`. Bounds of this arithmetic are not verified (that's a runtime check), but every coordinate in the index expression is verified to be in scope and to have a known domain.

Notice where the expressions live: in the body index positions, not in the declaration bracket. The declaration bracket names the output coordinates `b, oc, oh, ow`—simple identifiers. The body index expressions `oh + kh` describe how to compute the input indices from the output indices and the kernel coordinates. Left side: definition. Right side: computation.

The gradient with respect to `weight` follows mechanically from the coordinate sets:

```rust
let dW[oc, ic, kh, kw] = sum[b, oh, ow](
    dConv[b, oc, oh, ow] * input[b, ic, oh + kh, ow + kw]
);
```

The coordinates `b`, `oh`, `ow` are summed away because they appear in the output but not in `weight`. The surviving coordinates `oc`, `ic`, `kh`, `kw` are exactly `weight`'s coordinates. Set subtraction, applied to coordinate names, derives the formula. No memorization of transpose rules needed.

---

## Depth-to-Space: One Line Instead of Three

A depth-to-space operation reshapes a tensor by moving pixels from the channel dimension into the spatial dimensions, increasing resolution. In a positional framework:

```python
b, c, h, w = x.shape
x = x.reshape(b, c // 4, 2, 2, h, w)
x = x.permute(0, 1, 4, 2, 5, 3)
x = x.reshape(b, c // 4, h * 2, w * 2)
```

Three operations. Six position numbers to track. The semantic claim—"channel pixels become spatial neighbors"—is invisible.

In Einlang:

```rust
let y[b, c_out, h * 2 + dy, w * 2 + dx] =
    x[b, c_out * 4 + dy * 2 + dx, h, w];
```

One line. The index arithmetic says exactly what moved where. The names `dy` and `dx` declare the sub-pixel offsets. The multiplication `h * 2 + dy` expresses the output spatial coordinate in terms of the input. The coordinate names carry the story.

---

## Fancy Indexing: Names Disambiguate

Fancy indexing—using arrays of indices rather than slices—is where positional notation becomes most ambiguous. Consider gathering elements from a matrix:

```python
# Positional: what does this gather?
result = matrix[indices_i, indices_j]
```

In NumPy, this does *pairwise* indexing: `result[k] = matrix[indices_i[k], indices_j[k]]`. But if `indices_i` is a row vector and `indices_j` is a column vector, broadcasting produces *outer-product* indexing instead. The behavior depends on the shapes of the index arrays, not on anything written in the code.

In Einlang, the coordinate names disambiguate:

```rust
// Pairwise gather: same coordinate name appears in both index positions
let gathered[k] = matrix[indices_i[k], indices_j[k]];

// Outer-product gather: different coordinate names → Cartesian product
let outer[i, j] = matrix[indices_i[i], indices_j[j]];
```

When `k` appears in both index positions, pairwise indexing is inferred: `indices_i` and `indices_j` are traversed together. When `i` and `j` are different coordinates, outer-product indexing is inferred: every combination is produced. The distinction is visible in the coordinate names, not hidden in the shapes of the index arrays.

In NumPy, you need `np.ix_` or NEP 21's `oindex`/`vindex` to explicitly declare which behavior you want. In Einlang, the names do it.

Consider the indexing distinction in NumPy. Suppose you have two index arrays: `i = np.array([0, 1, 2])` and `j = np.array([0, 1, 3])`, and a 2D array `A`. To get elements at `A[0,0], A[1,1], A[2,3]` (pairwise), you write `A[i, j]`. To get all 3×3 combinations (outer-product), you write `A[i[:, None], j[None, :]]`. Compare the two expressions. Which one required you to create a dummy axis? The outer-product version did—you had to broadcast `i` and `j` into mutually orthogonal shapes with `None`/`np.newaxis`.

The pattern is visible: NumPy uses shape manipulation to disambiguate pairwise from outer-product indexing. The coordinate-sharing approach does it with names. Coordinate-sharing (`k` in both index positions) means pairwise; different names mean outer-product. The coordinate name carries the distinction. No shape manipulation needed.

---

## When One Coordinate Becomes Two: Gather vs. Scatter

Fancy indexing isn't the only place where coordinates split. Consider gathering rows from a matrix by index:

```python
# Positional: gather rows 0, 3, 2 from a matrix
rows = matrix[[0, 3, 2], :]
```

The index list `[0, 3, 2]` selects specific positions along axis 0. But what if you also want to select specific columns? And what if those column indices depend on the row?

```python
# Positional: for each selected row, pick a different column
idx = np.array([0, 3, 2])   # row indices
col_idx = np.array([1, 0, 2])  # column index for each row
result = matrix[idx, col_idx]  # pairwise: result[k] = matrix[idx[k], col_idx[k]]
```

This is pairwise indexing—`k` walks through both index arrays together. But what if you wanted outer-product: every combination of `idx` and `col_idx`? You'd need:

```python
result = matrix[idx[:, None], col_idx[None, :]]  # outer-product via broadcast
```

Two different behaviors, two different indexing patterns, one API. The difference is in the shapes of the index arrays (1D vs 2D after broadcasting), not in any semantic marker. If `idx` and `col_idx` happen to have the same length, the pairwise version runs and produces a result—just not the one you wanted if you intended outer-product.

**Einlang:**

```rust
// Pairwise: same coordinate name in both index positions
let gathered[k] = matrix[idx[k], col_idx[k]];

// Outer-product: different coordinate names → Cartesian product
let gathered[i, j] = matrix[idx[i], col_idx[j]];
```

When `k` appears in both index positions, the compiler infers pairwise indexing. When `i` and `j` are different, outer-product is inferred. The coordinate name *is* the disambiguation. No shape-dependent behavior. No manual broadcasting.

The named notation records the mental model ("these index together" vs "these index independently"). The positional notation buries it in shapes.

---

## The Coordinate Collision Test

There is a simple test for whether your notation disambiguates well. Consider two operations that have the same shape but different coordinate semantics. Can a colleague tell which is which just by looking at the code?

For fancy indexing:

```python
# Operation A
result = matrix[idx, col_idx]

# Operation B
result = matrix[idx[:, None], col_idx[None, :]]
```

If `idx` has length 5 and `col_idx` has length 5, both operations produce a `(5, 5)` result. But Operation A produces the diagonal (pairwise), while Operation B produces the full Cartesian product. From the code alone—without printing shapes or reading comments—can your colleague tell which is which?

Now the Einlang versions:

```rust
// Operation A: pairwise
let result[k] = matrix[idx[k], col_idx[k]];

// Operation B: outer-product
let result[i, j] = matrix[idx[i], col_idx[j]];
```

The difference is in the coordinate names. `k` vs `(i, j)`. One coordinate means pairwise. Two means outer-product. Your colleague reads the code and sees the difference. The code records the intent.

This is the Coordinate Collision Test: when two operations produce the same shape but different semantics, does your notation distinguish them?



---

## Convolution Backward: Gradient as Index Arithmetic

The input gradient—the one that backpropagates through the network—shows how index arithmetic survives differentiation.

Forward: `conv[b, oc, oh, ow] = sum[ic, kh, kw](input[b, ic, oh + kh, ow + kw] * weight[oc, ic, kh, kw])`.

To find `d_input[b, ic, ih, iw]`, apply the five-step procedure from Chapter 7:

1. **Hold one cell** of `input`: `input[b0, ic0, ih0, iw0]`.
2. **List every output cell that reads it.** The held cell is read by every `conv[b0, oc, oh, ow]` where `oh + kh = ih0` and `ow + kw = iw0`, for all `oc`, all `kh`, all `kw`. That means `oh = ih0 - kh` and `ow = iw0 - kw`. For each `(kh, kw)`, the output at position `(oh, ow) = (ih0 - kh, iw0 - kw)` receives a contribution.
3. **Attach the incoming gradient.** Each output cell `conv[b0, oc, oh, ow]` carries `d_conv[b0, oc, oh, ow]`. The contribution through the held input cell is `d_conv[b0, oc, ih0 - kh, iw0 - kw] * weight[oc, ic0, kh, kw]`.
4. **Multiply by the local derivative.** The derivative of `input * weight` with respect to `input` is `weight`.
5. **Sum over the path coordinates.** The output has `{b, oc, oh, ow}`. The input has `{b, ic, ih, iw}`. The path coordinates are `{oc, kh, kw}`—they appear in the output (via `oh, ow`) but are absorbed into `ih, iw` through the index arithmetic. But `oh, ow` are not independent—they are coupled to `ih, iw` through `kh, kw`. The sum is over `{oc, kh, kw}`, with the index relationship inverted: `oh → ih - kh`, `ow → iw - kw`.

Result:

```rust
let d_input[b, ic, ih, iw] = sum[oc, kh, kw](
    d_conv[b, oc, ih - kh, iw - kw] * weight[oc, ic, kh, kw]
);
```

This is the convolution transpose—a transposed convolution with flipped kernel indices. You derived it without memorizing "the gradient of convolution is a transposed convolution." You did coordinate accounting: which coordinates is the output indexed by that the input is not? The output uses `oh, ow` where the input uses `ih, iw`. The kernel indices `kh, kw` are shared—they appear in both the forward and the gradient. The output channel `oc` is in the output but not the input—sum over it.

The index arithmetic `ih - kh` and `iw - kw` comes from inverting the forward relationship `oh + kh → ih`. The gradient "reads" from the output at the position where the input contributed. The inversion is mechanical: solve `oh + kh = ih` for `oh`, giving `oh = ih - kh`.

---

## The Boundary: What Names Can't Check

Named coordinates are powerful, but they have a boundary. That a coordinate *exists* on a tensor is verified. Index arithmetic bounds are not verified. `oh + kh` is syntactically checked—`oh` and `kh` must be in scope—but whether `oh + kh` exceeds the input's spatial extent is a runtime question.

This boundary is not a flaw. It is a design choice. What can be proven from names and domains alone is checked. Bounds checking is the runtime's job. Semantic correctness—whether the formula means what you think it means—is yours.

Arithmetic cannot be verified. But the guarantee is: **everything that CAN be automatically checked IS automatically checked. Everything that CANNOT be automatically checked is made explicitly visible for you to review.** The coordinate names are the bridge between the two categories. They make the checkable parts machine-verifiable and the uncheckable parts human-visible.

The names are there so that when the runtime does report an out-of-bounds error, the error message can say which coordinate overflowed. `IndexError: oh + kh = 67 exceeds input width 64` is a better error than `IndexError: dimension 3 out of bounds`.

---

Named coordinates handle the complex terrain by giving each coordinate a persistent identity. The notation scales because the names scale.

**When do names fail?**

Names fail when the coordinate structure is truly unknown at compile time. A fully dynamic computation graph where shapes depend on runtime values—the number of detected objects in an image, the length of a generated sequence—cannot be fully verified by coordinate names alone. The names can check that `obj` is a declared coordinate and that functions consuming it have consistent contracts. They cannot check that `obj` has range 0..7 in one run and 0..12 in the next.

This is the boundary from Section 5, restated: names check consistency, not correctness. The compiler verifies that the coordinate story is internally coherent. It does not verify that the story matches reality. For that, you need runtime assertions. For that, you need tests. The names reduce the surface area of things that can go wrong silently. They do not eliminate the need for vigilance.

But vigilance is easier when the code records what you were being vigilant about. A name is a note to your future self: *this coordinate matters, I checked it, and the compiler checks that I checked it.*

The boundary between what names can and cannot check is not a weakness of the naming approach. It is a precise map of what is statically knowable. Every fact the names cannot check—bounds of index arithmetic, runtime-dependent shapes, semantic correctness of the formula itself—is a fact that *no* purely static system can check. The names don't fail at these boundaries. They mark them. The coordinate names are in the index expressions when the bounds are checked at runtime, in the declaration when the shape is resolved, in the function signature when the meaning is asserted. The names are there when the check happens—even if the check is runtime, not compile time.

This is what "everything checkable is checked" means in practice. The compiler checks what it can from declarations alone. What it cannot check, it leaves visible—with names attached—so that runtime checks and human reviewers know what to look for. A name is more useful when it's checked at compile time, but it's still useful when it's only checked at runtime. `IndexError: oh + kh = 67 exceeds input width 64` names the coordinate that overflowed. `IndexError: dimension 3 out of bounds` names a position. Which error would you rather debug at 3 AM?

---

### Stop and Think: The Splits in Your Code

Every time a coordinate splits into two roles—source and target, anchor and positive, user and item—you are doing coordinate splitting. The split is the operation. The names record it.

Open your codebase. Find a place where you inserted a dimension with `None`, `unsqueeze`, or `np.newaxis` to create a pairwise matrix. For each one:

1. **What coordinate split?** What single coordinate were you splitting into two? Write the two new names.
2. **Why did you split it?** Was it for a distance matrix? A similarity matrix? An attention mask? An outer product?
3. **Would a reader know which is which?** If they see `matrix[i]` and `matrix[j]` with `i` and `j` as different names, they know it's a split. If they see `matrix[:, None]` and `matrix[None, :]`, they see shape manipulation. The same intent. Different visibility.

The splits are there. The names make them readable.


---

*Find the most complex tensor indexing line in your codebase—the one with multiple `unsqueeze`/`expand`/`permute` calls chained together. Rewrite it with coordinate names, even just in a comment. Notice how many of those operations disappear when dimensions have identities.*

---

## When Index Arithmetic Meets Gradients

The convolution backward example earlier in this chapter derived the input gradient: `d_input[b, ic, ih, iw] = sum[oc, kh, kw](d_conv[b, oc, ih - kh, iw - kw] * weight[oc, ic, kh, kw])`. The index arithmetic `ih - kh` and `iw - kw` comes from inverting the forward relationship `oh + kh = ih`, solved for `oh → ih - kh`.

This is a general pattern: index arithmetic in the forward pass produces inverted index arithmetic in the backward pass. If the forward pass has `input[..., oh + kh, ...]`, the backward pass has `d_conv[..., ih - kh, ...]`. The subtraction inverts the addition. The coordinates involved in the arithmetic—`kh`, `kw`—become reduction axes in the backward sum. The coordinates that were loop axes—`oh`, `ow`—become the path coordinates that get summed.

In a positional autodiff engine like PyTorch's, this inversion is computed by the engine's internal graph. The programmer never writes `ih - kh`. The engine traces the forward operations, records the index arithmetic as node dependencies, and derives the backward computation automatically. The programmer writes `conv2d(input, weight)` and the gradient is handled by autograd.

The difference is not correctness—both derive the same result. The difference is auditability. When the backward pass is written explicitly with coordinate names, a reader can trace `ih - kh` back to the forward `oh + kh` and verify that the inversion is correct. When the backward pass is generated by the autodiff engine, the inversion is a black box. It is correct until it isn't—and when it isn't (because a custom kernel was written with the wrong index arithmetic, or because a `@tf.custom_gradient` rule has a bug), the reader has no source-level path from the forward expression to the backward expression.

Named coordinates don't replace autodiff. They give autodiff's output a form that a human can read, verify, and debug. The index arithmetic is in the source. The coordinate names tell you what is being inverted. The reader can trace the thread.
