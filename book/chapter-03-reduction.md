---
layout: book
title: "Chapter 3 · A Small Farewell"
---

# Chapter 3 · A Small Farewell

> "By relieving the brain of all unnecessary work, a good notation sets it free to concentrate on more advanced problems."
>
> — Alfred North Whitehead

*Primitives · Coordinate elimination and reduction*

---

A permutation moves coordinates around. A reduction makes one disappear.

This is a bigger deal than it sounds. When you eliminate a coordinate, you are making an irreversible decision. All the variation along that coordinate collapses into a single number. If you eliminate the wrong coordinate, every downstream result is contaminated—not with a crash, but with a mathematically valid answer to the wrong question.

And yet, in positional notation, a reduction is almost invisible. `x.mean(dim=1)` is eight characters. The identity of the eliminated coordinate occupies one of those characters: `1`. The other seven are ceremony.

This chapter introduces the first einlang syntax. We're going to write tensor operations where the eliminated coordinate is named explicitly—where the code says *what was lost*, not just *where it used to live*.

---

## Rectangular Declarations

Before we can eliminate a coordinate, we need to name the ones we're keeping. In einlang, you name coordinates with a **rectangular declaration**:

```rust
let doubled[i, j] = matrix[i, j] * 2.0;
```

Let's read this carefully. The `let` binds a new, immutable tensor. The `[i, j]` on the left declares the output coordinates—the new tensor will have two dimensions, and we are naming them `i` and `j`. The `matrix[i, j]` on the right indexes the input tensor `matrix` by those same coordinates. The compiler infers that `i` ranges from `0` to `matrix.shape[0]` and `j` from `0` to `matrix.shape[1]`.

This is not a loop. It is a declaration. You are stating a fact: "for all `i` and `j` in their respective domains, `doubled[i, j]` equals `matrix[i, j] * 2.0`." The compiler handles the iteration. You handle the meaning.

The index slots in the declaration bracket can take four forms:

- **A name**: `i`, `j`, `batch`, `channel`—the standard case. The compiler infers the range from how the name is used in the body.
- **A name with an explicit domain**: `i in 0..n`—when you need to control the range directly.
- **A literal**: `0`—used for base cases in recurrences (we'll see these in Chapter 9).
- **A named rest**: `..batch`—stands for zero or more adjacent axes. We'll meet these in the next chapter.

---

## Reduction

Now the main event. A reduction iterates over a coordinate and combines all the values along it using an associative operation. The coordinate appears in the reduction bracket—and then it is gone from the result:

```rust
let total = sum[i](data[i]);
```

`sum[i](...)` says: for every value of `i`, evaluate the body `data[i]`, and sum the results. The coordinate `i` is introduced by the `sum`, used in the body, and consumed by the reduction. It does not appear in `total`—`total` is a scalar.

The four reduction operations are `sum`, `max`, `min`, and `prod`. Each has an identity element that fills in when the domain is empty: `sum` starts from `0`, `prod` from `1`, `max` from negative infinity, `min` from positive infinity.

A reduction can eliminate multiple coordinates at once:

```rust
let grand_total = sum[i, j](matrix[i, j]);
```

And a reduction can leave some coordinates intact—producing a tensor rather than a scalar:

```rust
let row_sums[i] = sum[j](matrix[i, j]);
let col_sums[j] = sum[i](matrix[i, j]);
```

These two lines produce the same output shape (a 1D tensor of length equal to the surviving coordinate). But they mean completely different things. `row_sums[i]` sums over columns, leaving rows. `col_sums[j]` sums over rows, leaving columns. The difference is entirely in the bracket after `sum`—one character, carrying the full semantic weight of the operation.

In a positional API, these would be `matrix.sum(dim=1)` and `matrix.sum(dim=0)`. The reader must remember which position is rows and which is columns. The code does not help.

---

## Matrix Multiplication

With rectangular declarations and reductions, we have enough machinery to write matrix multiplication:

```rust
let C[i, j] = sum[k](A[i, k] * B[k, j]);
```

![Coordinates flow through matmul: i and j survive, k is consumed](figures/matmul_coords.svg)

The figure lays out the three matrices. $\mathbf{A}$ carries coordinates $\{i, k\}$—rows indexed by $i$, columns by $k$. $\mathbf{B}$ carries $\{k, j\}$—rows indexed by $k$, columns by $j$. The `sum[k]` bracket between them names the coordinate they share and that the reduction consumes. $\mathbf{C}$ carries $\{i, j\}$—only the survivors. On the right, a two-column ledger partitions the coordinates: $i$ and $j$ in the Survivors column (they appear in $\mathbf{C}$), $k$ in the Consumed column (gone from $\mathbf{C}$). This ledger is the visual form of the coordinate-audit instinct we are building.

If you've taken a linear algebra course, this line needs no explanation. It is the mathematical definition of matrix multiplication, transcribed character for character:

$$C_{ij} = \sum_k A_{ik} B_{kj}$$

The coordinate `k` is introduced by `sum`, used to index both `A` and `B` (forcing their shared dimension to agree—the compiler checks this), and consumed. The coordinates `i` and `j` survive into the result.

Notice what this line does *not* contain:
- No `transpose`. The indexing pattern `B[k, j]` naturally expresses that `B`'s rows are `k` and columns are `j`.
- No `matmul` function name. The operation is defined by the coordinates.
- No `@` operator whose meaning you have to look up.

The line says exactly what it means. And if you get the coordinates wrong—if you write `B[j, k]` instead of `B[k, j]`—the compiler catches it, because `k`'s range from `A` won't match `j`'s range from `B`.

---

## How to Read a Reduction: The Two-Column Ledger

A reduction is the most semantically loaded operation in tensor programming. Every reduction is making two claims at once: which coordinate is being *consumed* (gone from the result), and which coordinates are *surviving* (present in the result). When you read a reduction, draw an imaginary line down the middle of the page:

| Survivors | Locals |
|-----------|--------|
| Coordinates that appear in the result | Coordinates introduced in the reduction bracket |
| They keep their identity across the operation | They are consumed—gone from the output |
| They can be used by later operations | They exist only within the reduction body |

For `let row_sums[i] = sum[j](matrix[i, j]);`:

- **Survivors**: `i` (appears on the left-hand side, survives into `row_sums`)
- **Locals**: `j` (introduced by `sum[j]`, consumed, gone)

For `let C[i, j] = sum[k](A[i, k] * B[k, j]);`:

- **Survivors**: `i`, `j` (appear on the left, survive into `C`)
- **Locals**: `k` (introduced by `sum[k]`, aligns `A` and `B`, consumed)

This ledger is the coordinate audit in miniature. Every reduction you encounter, pause and fill in the two columns. If you can't fill them in from the code alone—if you need to print shapes or consult a comment—the code is hiding information that the notation should have recorded.

A five-step rule for reading any reduction:

1. **Identity the reduction**: which operation (`sum`, `max`, `min`, `prod`) and which coordinates are in its bracket?
2. **Identify the survivors**: which coordinates appear on the left-hand side of the `let`?
3. **Identify the consumed**: which coordinates appear in the reduction bracket but not on the left?
4. **Verify alignment**: do the consumed coordinates index matching positions across all terms in the body?
5. **State the claim**: in one sentence, what does this reduction assert? "For each `i`, sum over all `j`." "For each `[b, out]`, contract `[in]` between the input and the weight."

This takes five seconds. It catches the bug where `sum[class]` silently became `sum[batch]` after a refactoring.

---

## What We've Just Learned

This chapter introduced three things that will carry us through the rest of the book:

1. **Rectangular declarations** (`let C[i, j] = ...`) name the coordinates of a new tensor and define its values by indexing existing tensors.
2. **Reductions** (`sum[k](...)`, `max[k](...)`, `min[k](...)`, `prod[k](...)`) consume a coordinate—they introduce it in the reduction bracket, use it in the body, and erase it from the result.
3. **Coordinates are checked, not commented.** When you write `sum[k](A[i, k] * B[k, j])`, the compiler verifies that `A` has a `k` dimension, that `B` has a `k` dimension, and that their ranges agree. A typo in a coordinate name is a compile error, not a runtime mystery.

We are still in the primitives layer. We've learned to name coordinates and to eliminate them. In the next chapter, we learn the third primitive: when a coordinate is *copied* rather than eliminated.

A vector dot product in einlang reads `let dot = sum[i](a[i] * b[i])`. Compare to `torch.dot(a, b)`: the einlang version tells you which coordinate was summed because the name `i` is right there in the reduction bracket. For a linear layer, the expression `let output[b, out] = sum[in](input[b, in] * weight[out, in]) + bias[out]` makes visible that `in` is consumed, `[b, out]` survive, and `bias` omits the `in` coordinate entirely. Now consider a friend who writes `let C[i, j] = sum[k](A[i, k] * B[j, k])` thinking it's matrix multiplication. The compiler accepts it—the shapes are valid—but it's computing something else: the `j` in `B[j, k]` replaced the contraction coordinate when `k` was meant. The names make the mistake legible in a way that `torch.matmul(A[:, :, None] * B[:, None, :]).sum(2)` never would.
