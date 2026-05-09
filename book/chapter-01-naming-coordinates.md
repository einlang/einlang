---
layout: book
title: "Chapter 1 · Every Dimension Deserves a Name"
---

# Chapter 1 · Every Dimension Deserves a Name

> "The act of naming is the great and original instrument of abstraction."
>
> — Ursula K. Le Guin

*Primitives · Naming coordinates*

---

What is a tensor?

Ask a framework documentation and it will tell you: a multidimensional array. Ask a tensor's `.shape` attribute and it will tell you: `(32, 64, 256)`. Ask a compiler and it will tell you: a pointer to a contiguous block of memory with strides and a dtype.

All true. All missing the point.

A tensor is a function from coordinates to values. You give it a `batch` index, a `channel` index, and a `spatial` index; it gives you back a number. The three coordinates together form an address. Every element in the tensor lives at exactly one address.

This definition is not exotic. It is how mathematicians have written tensor operations for a century:

$$C_{ij} = \sum_k A_{ik} B_{kj}$$

The letters `i`, `j`, and `k` are not axis numbers. They are coordinate names. `i` walks the rows of `A`. `j` walks the columns of `B`. `k` walks the dimension they share—the one that gets summed away. You can rename `i` to `row`, `j` to `col`, `k` to `inner`, and the mathematics is unchanged.

Now look at how we write the same operation in a modern framework:

```python
C = torch.matmul(A, B)
```

Where are `i`, `j`, and `k`? They are gone. The names that gave the operation its meaning are not present in the source text. The compiler knows the shapes of `A` and `B`. It checks that the inner dimensions agree. It does not know—cannot know—that `A`'s second axis represents `feature` and not `time`, or that `B`'s first axis represents `feature` and not `vocab_size`. It only knows that both are `64`.

A coordinate has three properties. The framework records two of them. The third—the name—exists only in the programmer's head, in comments, or in variable naming conventions. The figure below shows this asymmetry: each dimension of a `(32, 64, 256)` tensor carries a size (checked by the compiler), a position (checked by the compiler), and a name (checked by no one). The name is the fact that correctness depends on, and it is invisible to every tool in the standard pipeline.

![Three properties of a coordinate: domain and position are checked; the name is not](figures/shape_meanings_gap.svg)

This is the shape-meanings gap. The shape says *how many*. The role says *which one*. Every framework knows the shape. None of them know the role.

---

Let's make this concrete. Here is a reduction:

```python
x = torch.randn(32, 64, 256)
y = x.mean(dim=1)
```

What was eliminated? The code says `dim=1`. If you are the author, and you just wrote this line, you know that position 1 is `channel`. If you are a colleague reading this six months later, you have to reconstruct that fact. You look at the variable name. You look for a comment. You trace the data pipeline backward. You hold your breath and assume.

Now imagine a different notation:

```rust
let y[b, s] = mean[channel](x[b, channel, s]);
```

The bracket after `mean` names the coordinate being eliminated. The brackets after `y` and `x` name the coordinates that survive. The compiler checks that `channel` actually exists on `x`. The reader sees the elimination without reconstructing it. The fact that was previously in a comment—"average over channels"—is now in the syntax, where the compiler can enforce it and the reader can audit it.

This is not a hypothetical language. It is the one we will use for the rest of this book. Its syntax first appears in Chapter 3. But the idea it embodies—**give every dimension a name**—is independent of any particular notation. You can practice it in NumPy docstrings, in einops patterns, in the type annotations of your own code. The habit is the payload. The syntax is only the delivery mechanism.

---

Before we move on, let's be precise about what "naming a coordinate" means.

A coordinate has three properties. First, a **name**: `batch`, `channel`, `time`, `feature`. The name carries the semantic role. Second, a **domain**: the set of values the coordinate can take. For a tensor of shape `(32, 64, 256)`, the `batch` coordinate ranges from `0` to `31`, `channel` from `0` to `63`, and `spatial` from `0` to `255`. Third, a **position**: where this coordinate sits in the tensor's shape tuple. In `(32, 64, 256)`, `batch` is at position 0, `channel` at position 1, `spatial` at position 2.

Positional notation records only the domain and the position—`(32, 64, 256)` tells you the sizes and their order, but not their names. Named notation records all three: `[batch: 32, channel: 64, spatial: 256]`.

When you write `x.mean(dim=1)`, you are asking the position to stand in for the name. It works until the position changes. When you write `mean[channel](x)`, you are using the name directly. The position becomes an implementation detail—the compiler's problem, not yours.

---

## An Analogy: The Parking Lot

You park your car in Row D, Slot 7. The ticket in your pocket says "D-7." You return after dinner to find the lot has been repainted. The rows now run perpendicular to their old orientation. Row D is now somewhere else entirely. Your ticket, which records a *position* in a fixed coordinate system, sends you to the wrong car.

The lot's *shape* hasn't changed. It is still an 8 × 20 grid. A shape checker would tell you everything is fine. But the *role* of each row—which row is "D"—has moved.

This is what happens when you write `x.transpose(1, 2)`. The shape is still `(32, 256, 64)`. A shape checker sees the same three numbers. But the positions have been reassigned. Dimension 1 is no longer `channel`. Dimension 2 is no longer `spatial`. The ticket in your pocket—`dim=1`—now points to the wrong car.

A named-coordinate notation is like a ticket that says "the blue Honda Civic" instead of "D-7." The car may move, but the description finds it.

---

This chapter has said nothing you couldn't have figured out from fifteen minutes with a debugger and a whiteboard. That is by design. The ideas in Part I are simple. Their simplicity is the point.

The next chapter applies the same name-over-position instinct to a second primitive: permutation.

Take a tensor `x = torch.randn(2, 3, 4)` and decide which dimension is `batch`, which is `time`, which is `feature`. Now write `x.mean(dim=1)`. Which dimension did you eliminate? If someone swapped dimensions 1 and 2 upstream, your `mean` call would silently eliminate a different one. The same exercise with `x.permute(0, 2, 1)`: write down what the permutation *means* in terms of your chosen coordinate names. If dimension 0 and dimension 1 are swapped upstream, `permute(0, 2, 1)` no longer means the same thing—the positions marched on, but the meaning got left behind. This is why we name coordinates: so the code says what it means, not where it is.
