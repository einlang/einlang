---
layout: book
title: "Chapter 4 · Blocks and Skeletons"
---

# Chapter 4 · Blocks and Skeletons

> "If you have a procedure with ten parameters, you probably missed some."
>
> — Alan Perlis

*Combinations · Pack polymorphism and normalization skeletons*

---

Primitives let you say simple things. Combinations let you compose simple things into complex ones. Abstraction lets you name the complex things and use them as if they were primitive.

Chapter 3 showed how to wrap a coordinate pattern in a function signature. This chapter shows what happens when you have several such functions—and realize they are all the same skeleton, dressed in different coordinate names.

---

## The Anatomy of a Coordinate-Aware Function

Here is the complete form:

```rust
fn normalize[coord](x: [f32; ..left, coord, ..right])
    -> [f32; ..left, coord, ..right]
{
    let m[..left, ..right] = max[coord](x[..left, coord, ..right]);
    let centered[..left, coord, ..right] = x[..left, coord, ..right] - m[..left, ..right];
    let scale[..left, ..right] = sum[coord](centered[..left, coord, ..right] ** 2.0);
    centered[..left, coord, ..right] / (scale[..left, ..right] ** 0.5 + 1e-5)
}
```

Four things to notice.

**First, the coordinate parameter `coord`.** It appears in brackets after the function name and in the type signatures. It is not a value—you cannot do arithmetic on it. It is a coordinate identity. When the function body uses `max[coord]` and `sum[coord]`, it is verified that `coord` is the same parameter declared in the signature.

**Second, the rest packs `..left` and `..right`.** They stand for whatever coordinates surround `coord` in the actual argument. If the caller passes `x[b, t, f]` and writes `normalize[f](x)`, then `..left` binds to `[b, t]` and `..right` binds to nothing. If the caller passes `x[b, h, f, d]` with `normalize[f](x)`, then `..left` binds to `[b, h]` and `..right` binds to `[d]`. The function body is polymorphic over the surrounding structure.

**Third, coordinate flow.** Inside the function body, the coordinate `coord` is in scope. It can be used in reductions (`max[coord]`, `sum[coord]`), in indexing, and implicitly in the output shape. The packs `..left` and `..right` flow from the input signature to the output signature.

**Fourth, the return type annotation.** The `-> [f32; ..left, coord, ..right]` tells the reader which coordinates survive. If the function body accidentally consumed `coord` without reconstructing it, or dropped a pack, a coordinate mismatch is reported.

---

## Packs and Polymorphism

Packs are what make coordinate-aware functions reusable across different tensor ranks. There are three patterns:

**Leading packs** (`..left` or `..batch`): absorb dimensions that come before the coordinate of interest. Used when the function treats all leading dimensions uniformly.

**Trailing packs** (`..right` or `..rest`): absorb dimensions that come after. Used when the function operates on a coordinate and doesn't care what follows.

**Named spatial packs** (`..spatial`): absorb spatial dimensions as a group. A function that reshapes spatial coordinates can treat them as a unit:

```rust
fn move_channel[channel, ..spatial](x: [f32; channel, ..spatial])
    -> [f32; ..spatial, channel]
{
    x[..spatial, channel]
}
```

When a pack is ambiguous at the call site, the caller disambiguates by grouping:

```rust
id_axes[(height, width)](x)
```

Pack parameters make coordinate-aware functions rank-polymorphic: the same function works on 2D, 3D, 4D, or higher-dimensional tensors, as long as the coordinate of interest exists somewhere in the shape.

---

## Selection Reductions

A normal reduction returns a combined value. A selection reduction returns an address:

```rust
let pred[b] = argmax[class](logits[b, class]);
```

`max[class](logits[b, class])` returns the maximum *value* for each `b`. `argmax[class](logits[b, class])` returns the index within the `class` domain where that maximum occurs. The result has the surviving coordinates—`[b]`—plus an implicit address domain: the integer stored at each `b` is understood to be an address in the `class` coordinate space.

This is more than a convenience. The return value of `argmax[class]` is not just an integer tensor. It is an integer tensor whose values are addresses in the `class` domain. Later code can use this address to index back into the `class` dimension of another tensor.

---

## One Skeleton, Four Normalizations

Every normalization function follows the same coordinate skeleton: reduce to get statistics, broadcast statistics back, apply elementwise. The difference is only which coordinates play which roles:

| Function | Reduction coords | Broadcast params | Survivors |
|---|---|---|---|
| Softmax | `q` (max), `k` (sum) | none | `..batch`, `j` |
| LayerNorm | `f` (mean ×2) | `gamma[f]`, `beta[f]` | `..batch`, `f` |
| RMSNorm | `f` (mean) | `gamma[f]` | `..batch`, `f` |
| GroupNorm | `c_in_group`, `..spatial` | `gamma[g, c_in_group]`, `beta[g, c_in_group]` | `..batch`, `g`, `c_in_group`, `..spatial` |

In a positional API, all four collapse to a single `dim` argument whose meaning shifts with the surrounding layout. `LayerNorm` and `RMSNorm` both use `dim=-1`—but normalize different statistics. `GroupNorm` uses three reduction dimensions buried in a `reshape` chain. The skeleton is invisible.

In a named-coordinate API, the skeleton is a template you can check. The reduction bracket names the consumed coordinates. The indexing pattern names the survivors. The broadcast parameters name the omission. A reviewer can verify that the broadcast coordinate in LayerNorm matches the broadcast coordinate in the gradient without reconstructing both from positional offsets.

This is abstraction: recognizing a pattern, naming it, and reusing it. The pattern is "normalize with named coordinates." Each instance fills in the specific coordinates. The skeleton is constant.

---

The normalization skeleton table reveals something deeper about named coordinates. In a positional API, "feature" is the last axis in a 2D tensor—but in a 4D tensor, what was one axis may now span multiple actual dimensions. Named coordinates handle this naturally: one semantic coordinate (`feature`) can correspond to one or more actual axes in the tensor, depending on the layout. Positional code requires a reshape-permute-reshape dance to group and ungroup those dimensions. Named code just names them.

This is what packs buy you. `..spatial` absorbs however many spatial dimensions there are. The same `GroupNorm` skeleton works whether spatial covers one axis or three.

---

Coordinate facts flow through pointwise operations automatically. If `x` carries the fact that it has coordinates `[b, class]`, then `x ** 2.0` also has coordinates `[b, class]`. You don't need to re-declare them. Coordinate facts are preserved through arithmetic, through function calls that return tensors, through `if` expressions. The only operations that change coordinate facts are those that explicitly manipulate coordinates: reductions (which consume), rectangular declarations (which introduce), and coordinate-aware function calls (which thread them through signatures).

This flow is the foundation of abstraction. When you wrap a computation in a coordinate-aware function, the coordinate facts flow from the caller, through the function body, and out to the result. The function doesn't need to re-derive what the caller already knew. The facts propagate.

In the next chapter, we add a fourth role to our coordinates: time. A coordinate that doesn't just sit there, but flows.
