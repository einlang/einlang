---
layout: book
title: "Chapter 6 · Complex Terrain"
---

# Chapter 6 · Complex Terrain

> "The map is not the territory—but a good map tells you where the boundaries are."
>
> — Adapted from Alfred Korzybski

*Combinations · Distance matrices, convolution, and fancy indexing*

---

So far our coordinates have been simple: one name per axis, each axis independent. Real tensor programs are messier. A coordinate can split in two. It can carry arithmetic. It can appear in patterns where the same name used twice means something different from two names used once.

This chapter explores the terrain where naming earns its keep—where the coordinate story is too complex to hold in your head, and the notation either carries it or loses it.

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

In einlang:

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

In einlang, the coordinate names disambiguate:

```rust
// Pairwise gather: same coordinate name appears in both index positions
let gathered[k] = matrix[indices_i[k], indices_j[k]];

// Outer-product gather: different coordinate names → Cartesian product
let outer[i, j] = matrix[indices_i[i], indices_j[j]];
```

When `k` appears in both index positions, pairwise indexing is inferred: `indices_i` and `indices_j` are traversed together. When `i` and `j` are different coordinates, outer-product indexing is inferred: every combination is produced. The distinction is visible in the coordinate names, not hidden in the shapes of the index arrays.

In NumPy, you need `np.ix_` or NEP 21's `oindex`/`vindex` to explicitly declare which behavior you want. In einlang, the names do it.

---

## The Boundary: What Names Can't Check

Named coordinates are powerful, but they have a boundary. That a coordinate *exists* on a tensor is verified. Index arithmetic bounds are not verified. `oh + kh` is syntactically checked—`oh` and `kh` must be in scope—but whether `oh + kh` exceeds the input's spatial extent is a runtime question.

This boundary is not a flaw. It is a design choice. What can be proven from names and domains alone is checked. Bounds checking is the runtime's job. Semantic correctness—whether the formula means what you think it means—is yours.

Arithmetic cannot be verified. But the guarantee is: **everything that CAN be automatically checked IS automatically checked. Everything that CANNOT be automatically checked is made explicitly visible for you to review.** The coordinate names are the bridge between the two categories. They make the checkable parts machine-verifiable and the uncheckable parts human-visible.

The names are there so that when the runtime does report an out-of-bounds error, the error message can say which coordinate overflowed. `IndexError: oh + kh = 67 exceeds input width 64` is a better error than `IndexError: dimension 3 out of bounds`.

---

Named coordinates handle the complex terrain—splits, arithmetic, disambiguation—by giving each coordinate a persistent identity. The same identity that survived a permutation in Chapter 1 survives index arithmetic here. The notation scales because the names scale.

In the next chapter, we turn to the operation that makes all of this learnable: differentiation. The forward pass builds a computation. The backward pass reads it in reverse. And the names are the thread that ties the two directions together.
