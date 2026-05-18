---
layout: book
title: "Chapter 7 · Complex Terrain"
---

# Chapter 7 · Complex Terrain

> "The shortest path between two truths in the real domain passes through the complex domain."
>
> — Jacques Hadamard

*Combinations · Distance matrices, convolution, and fancy indexing*

---

You have a distance matrix. Shape `(N, N)`. Axis 0 and axis 1 both index over the same set of points. Which one is the source point, and which is the target? The shape does not tell you. The positions do not tell you. If you wrote a comment, the comment tells you—for now, until the comment rots.

The matrix is square because the same coordinate appears twice, playing two different roles. The coordinate has split. `point` becomes `point_i` and `point_j`. The split is invisible to every positional tool you have. It is visible to the reader only if the notation records which copy is which.

This chapter is about the split—one phenomenon seen through several lenses. Distance matrices, where one coordinate turns into two. Convolution index arithmetic, where coordinates carry formulas (`oh + kh`). Fancy indexing, where coordinate collisions are bugs or features depending on intent. Every section asks the same question: when a coordinate's role is more complex than "this axis exists," does your notation record the complexity—or bury it in shape arithmetic?

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

The split was spatial—one coordinate name, two roles, same domain. Next: what happens when coordinates carry arithmetic, and a coordinate from the output combines with a coordinate from the kernel to produce an index?

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

Index arithmetic reshapes coordinates. Next: when coordinates determine *which* elements you read from a tensor, and the difference between "read together" and "read independently" depends entirely on whether the coordinate name is shared.

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

This is the Coordinate Collision Test: when two operations produce the same shape but different semantics, does your notation distinguish them? If the answer is no, the semantics are in your head. If the answer is yes, they are in the code. The distance between those two places is the distance between a bug found in review and a bug found in production.

**Derive it yourself: Collision-proof your notation.** Here are three index-gathering operations. For each one, write both the pairwise and outer-product versions by choosing coordinates. Then check: would a reader see the difference?

1. A lookup table `L` maps `(token, feature)` pairs to values. Gather from it using a list of indices. Write the pairwise version (index `k` selects tokens and features together) and the outer-product version (indices `i` and `j` select tokens and features independently).
2. A routing operation sends each item `item[i]` to a destination `dest[j]` using a routing table `route`. Write the pairwise version and the outer-product version.
3. A graph convolution gathers features from neighbors: `adj[node, neighbor]` selects which neighbors to aggregate. Write the pairwise version (each `edge` selects a specific `(node, neighbor)` pair) and the outer-product version.

The answer in each case: one coordinate means pairwise. Two means outer-product. The names carry the semantics. The shape alone—`(N,)` in both cases—carries none.

Every section so far has been about the forward pass. Now: what happens when differentiation runs backward through these same patterns? Index arithmetic in the forward pass becomes inverted index arithmetic in the backward pass. Coordinate splits in the forward pass determine which coordinates get summed in the gradient. The names don't change. The direction does.

---

## Convolution Backward: Gradient as Index Arithmetic

The input gradient—the one that backpropagates through the network—shows how index arithmetic survives differentiation.

Forward: `conv[b, oc, oh, ow] = sum[ic, kh, kw](input[b, ic, oh + kh, ow + kw] * weight[oc, ic, kh, kw])`.

To find `d_input[b, ic, ih, iw]`, apply the five-step procedure from Chapter 8:

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

Step back. You just derived a transposed convolution—the gradient of convolution—without memorizing a formula. The steps were: hold one input cell, list every output cell that reads it, invert the index relationship, sum over the path coordinates. These are the same five steps from Chapter 8's pullback procedure. The names `ih`, `iw`, `kh`, `kw`, `oc` stayed constant. The operation changed: forward addition (`oh + kh`) became backward subtraction (`ih - kh`). The structure—coordinate accounting—stayed the same.

This is the central claim of Part II: coordinate names survive composition. They survive function boundaries (Chapter 3), broadcast (Chapter 4), normalization skeletons (Chapter 5), recurrence (Chapter 6), complex index arithmetic (this chapter), and differentiation (Chapter 8). The name is the invariant. The operation around it changes. The name remains.

---

## The Boundary

Index arithmetic like `oh + kh` is syntactically checked—`oh` and `kh` must be in scope. Whether `oh + kh < input_height` can be proved depends on what the compiler knows.

When the domain sizes are known at compile time—`oh` ranges over `0..output_height`, `kh` over `0..kernel_height`, both static—the constraint solver can prove the bound. `(output_height - 1) + (kernel_height - 1) < input_height` is a single comparison. The proof is mechanical. Chapters 9 and 10 show how.

When the domain sizes are runtime values—image dimensions read from a file, sequence lengths that vary per batch—the compiler cannot prove the bound. The names are still there. The error message names the coordinate: `IndexError: oh + kh = 67 exceeds input width 64`. The positional equivalent names a position: `IndexError: dimension 3 out of bounds`. The difference is the distance between the two.

What names can and cannot check is the subject of Chapter 14. For now: the compiler checks every fact derivable from declarations and proves every bound that static ranges permit. What it cannot prove—runtime-dependent bounds, data-dependent shapes, semantic correctness—it leaves visible, with names attached.

---

## When Index Arithmetic Meets Gradients

The general pattern: index arithmetic in the forward pass produces inverted index arithmetic in the backward pass. Forward `input[..., oh + kh, ...]` becomes backward `d_conv[..., ih - kh, ...]`. The subtraction inverts the addition. The coordinates involved in the arithmetic—`kh`, `kw`—become reduction axes in the backward sum. The coordinates that were loop axes—`oh`, `ow`—become the path coordinates that get summed.

In a positional autodiff engine like PyTorch's, this inversion is computed by the engine's internal graph. The programmer never writes `ih - kh`. The engine traces the forward operations, records the index arithmetic as node dependencies, and derives the backward computation automatically. The programmer writes `conv2d(input, weight)` and the gradient is handled by autograd.

The difference is not correctness—both derive the same result. The difference is auditability. When the backward pass is written explicitly with coordinate names, a reader can trace `ih - kh` back to the forward `oh + kh` and verify that the inversion is correct. When the backward pass is generated by the autodiff engine, the inversion is a black box. It is correct until it isn't—and when it isn't (because a custom kernel was written with the wrong index arithmetic, or because a `@tf.custom_gradient` rule has a bug), the reader has no source-level path from the forward expression to the backward expression.

Named coordinates don't replace autodiff. They give autodiff's output a form that a human can read, verify, and debug. The index arithmetic is in the source. The coordinate names tell you what is being inverted. The reader can trace the thread.

---

## Ranges Are Expressions

Every reduction so far has used a fixed range—`sum[i](data[i])` sums over all of `i`. But the range can be an expression. And the expression can reference another coordinate:

```rust
let cumsum[i] = sum[k in 0..i+1](data[k]);
```

The coordinate `i` appears in two places: on the left-hand side (the output index) and in the reduction range `k in 0..i+1`. Each output position `cumsum[0]`, `cumsum[1]`, `cumsum[2]` sums a different prefix of `data`.

This is not a primitive `scan` operation. It is a reduction. The only difference from `sum[k](data[k])` is that the range depends on `i`. The coordinate `i` does double duty—it tells you which output cell you are computing, and it sets the boundary for how many inputs that cell sees.

The same pattern produces every cumulative operation:

```rust
let cumprod[i] = prod[k in 0..i+1](data[k]);     // cumulative product
let cummax[i] = max[k in 0..i+1](data[k]);        // cumulative maximum
let running_avg[i] = sum[k in 0..i+1](data[k]) / (i + 1) as f32;  // running average
```

And it composes with other coordinates:

```rust
// Cumulative energy along time, per batch element
let energy[b, t] = sum[k in 0..t+1](signal[b, k] * signal[b, k]);
```

`b` is a survivor—it passes through untouched. `t` is the scan axis—it sets the range for `k`. `k` is consumed by the sum. Three coordinates, three roles, one line.

In a positional framework, `np.cumsum(data, axis=0)` gives you the cumulative sum. But `axis=0` tells you nothing about which coordinate is being scanned. If the layout changes—a temporal dimension is inserted, or the batch and time axes swap—`axis=0` silently scans the wrong coordinate. The named version says `cumsum[t]`. The scan is over `t`, regardless of where `t` sits in the layout.

Ranges are expressions. Expressions can reference coordinates. Coordinates can appear in both the output declaration and the reduction range. The three roles—output index, range parameter, consumed variable—are all visible in the notation, all carried by names.

---

### Ranges that Shape a Neighborhood

The ranges so far have been one-sided: `k in 0..i+1` says "stop at the current position." But ranges can also exclude boundaries, and—combined with offset index expressions—they can define a spatial neighborhood that moves across a grid.

Consider the 1D heat equation, discretized on a grid of 41 spatial points evolved over 80 time steps. The continuous equation is `∂u/∂t = α ∂²u/∂x²`. The discrete version uses a three-point centered difference:

```rust
let r = 0.2;
let n = 41;

let u[0, i in 0..41] = 0.0;
let u[0, 20] = 10.0;                          // initial bump

let u[t in 1..80, 0] = 0.0;                   // left boundary — fixed
let u[t in 1..80, 40] = 0.0;                  // right boundary — fixed
let u[t in 1..80, i in 1..40] =               // interior — stencil
    u[t-1, i] + r * (u[t-1, i-1] - 2.0 * u[t-1, i] + u[t-1, i+1]);
```

Read the interior line: at time `t` and spatial position `i`, the new value is the old value plus a diffusion term that reads three neighbors from the previous time step — `i-1`, `i`, and `i+1`. The range `i in 1..40` excludes the boundaries (0 and 40 are handled separately). The range `t in 1..80` starts at 1 because the initial condition occupies `t=0`.

The range expressions on the left side declare *which cells are computed*. The offsets on the right side declare *which already-computed cells are read*. `i in 1..40` says: compute every interior cell. `u[t-1, i-1]` says: read the cell one spatial step left, from the previous time step. The two ranges are different expressions, referencing the same coordinate `i` in two roles. The left range says "compute positions 1 through 40." The right offsets say "read positions 0 through 41." The overlap is exact — the interior reads both boundaries without computing them.

The complete program — initial condition, boundary conditions, interior evolution — is four `let` statements. No loop construct. No time-stepping primitive. The ranges are the loops. The offsets are the stencil. The recurrence across `t` is the same mechanism from Chapter 6, applied to a spatial grid. The coordinate `t` carries time. The coordinate `i` carries space. The names distinguish them.

In a positional framework, the equivalent is nested loops with manual index arithmetic:

```python
u = np.zeros((81, 41))
u[0, 20] = 10.0
for t in range(1, 81):
    u[t, 0] = 0.0
    u[t, 40] = 0.0
    for i in range(1, 40):
        u[t, i] = u[t-1, i] + r * (u[t-1, i-1] - 2*u[t-1, i] + u[t-1, i+1])
```

The loops are explicit. The ranges are numbers. The stencil pattern is visible — but it is visible in the arithmetic, not in the structure. If the grid expands to 2D (heat on a plate) or 3D (heat in a volume), the positional version adds nested loops. The named version adds coordinates:

```rust
let u[t in 1..80, i in 1..40, j in 1..40] =  // 2D interior
    u[t-1, i, j] + r * (u[t-1, i-1, j] + u[t-1, i+1, j]
                       + u[t-1, i, j-1] + u[t-1, i, j+1]
                       - 4.0 * u[t-1, i, j]);
```

Two coordinates added. The stencil grew from three neighbors to five. The range `i in 1..40, j in 1..40` excludes the 2D boundary. The structure — ranges on the left, offsets on the right — is identical in 1D and 2D. The notation scales. The loops scale too, but they scale by nesting. The nests grow. The names don't.

This is the third form of range expression. The first was a fixed range (`k in 0..3`). The second was a range that references a coordinate (`k in 0..i+1` for cumulative operations). The third is a range that excludes boundaries, combined with offsets, to define a moving stencil over a grid. All three use the same mechanism: a coordinate appears in a range expression on the left, and in index expressions (possibly with offsets) on the right. The range says what to compute. The offsets say what to read. The coordinate name ties them together.

Ranges are expressions. Stencils are ranges with offsets. The PDE is the program.

---

Every section in this chapter was a variation on one operation: a single coordinate splitting into two roles.

Distance matrix: `point` becomes `point_i` and `point_j`. Convolution: `oh` and `kh` together index the spatial input — one coordinate from the output, one from the kernel. Depth-to-space: `c_out * 4 + dy * 2 + dx` splits the channel index into a channel group and two sub-pixel offsets. Fancy indexing: `k` appears in both index positions for pairwise, or `i` and `j` are different for outer-product — coordinate-sharing IS the disambiguation. Gather vs. scatter: same split, different direction.

In every case, the split is the operation. The names record it. The positional notation records the mechanic — `unsqueeze`, `permute`, `reshape`, `None`, `:` — and buries the identity. When you read `points[:, None, :] - points[None, :, :]`, you see three dimension manipulations. When you read `points[point_i, dim] - points[point_j, dim]`, you see one split. The difference is the distance between a notation that records what happened and a notation that records why.

A notation that records the split gives the split a name. A notation that records only the mechanic gives the mechanic a position. Positions shift. Names don't.

Index arithmetic was the most complex coordinate manipulation in Part II. Next: what happens when we differentiate through every operation we have built—applying the Inversion Rule systematically to reductions, broadcasts, index arithmetic, and recurrence. The gradient is coordinate set subtraction, applied in reverse. The five-step pullback is the procedure. The coordinate names are the bridge between the forward and backward directions.
