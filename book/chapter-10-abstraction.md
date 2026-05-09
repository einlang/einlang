---
layout: book
title: "Chapter 10 · Building Your Own Blocks"
---

# Chapter 10 · Building Your Own Blocks

> "If you have a procedure with ten parameters, you probably missed some."
>
> — Alan Perlis

*Abstraction · Coordinate-aware functions, packs, selection reductions*

---

Primitives let you say simple things. Combinations let you compose simple things into complex ones. Abstraction lets you name the complex things and use them as if they were primitive.

This is the third layer of the pyramid we've been building, and it is where the coordinate habit becomes a tool for *design* rather than just *audit*. You are no longer asking whether an existing operation respects coordinate identities. You are building new operations whose coordinate identities are part of their public contract.

---

## The Anatomy of a Coordinate-Aware Function

Chapter 6 introduced the basic form. Here is the complete syntax:

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

Four things to notice:

**First, the coordinate parameter `coord`.** It appears in brackets after the function name and in the type signatures of both the parameter and the return. It is not a value—you cannot do arithmetic on it. It is a coordinate identity, and the compiler treats it as such. When the function body uses `max[coord]` and `sum[coord]`, the compiler verifies that `coord` is the same coordinate parameter declared in the function signature.

**Second, the rest packs `..left` and `..right`.** They stand for whatever coordinates surround `coord` in the actual argument. If the caller passes a tensor `x[b, t, f]` and writes `normalize[f](x)`, then `..left` binds to `[b, t]` and `..right` binds to nothing (empty). If the caller passes `x[b, h, f, d]` with `normalize[f](x)`, then `..left` binds to `[b, h]` and `..right` binds to `[d]`. The function body is polymorphic over the surrounding structure.

**Third, coordinate flow.** Inside the function body, the coordinate `coord` is in scope. It can be used in reductions (`max[coord]`, `sum[coord]`), in indexing, and—implicitly—in the output shape. The packs `..left` and `..right` flow from the input signature to the output signature. The compiler ensures that the function body produces a result with the advertised coordinates.

**Fourth, the return type annotation.** The `-> [f32; ..left, coord, ..right]` tells the compiler (and the reader) which coordinates survive the function. The function body's final expression must have this coordinate structure. If it doesn't—if you accidentally consumed `coord` without reconstructing it, or dropped a pack—the compiler reports a coordinate mismatch.

---

## Packs and Polymorphism

Packs are what make coordinate-aware functions reusable across different tensor ranks. There are three patterns:

**Leading packs** (`..left` or `..batch`): absorb dimensions that come before the coordinate of interest. Used when the function treats all leading dimensions uniformly—batch dimensions, head dimensions, any prefix that the operation is independent of.

**Trailing packs** (`..right` or `..rest`): absorb dimensions that come after. Used when the function operates on a coordinate and doesn't care what follows.

**Named spatial packs** (`..spatial`): absorb spatial dimensions as a group. Useful for functions that move or reshape spatial coordinates as a unit:

```rust
fn move_channel[channel, ..spatial](x: [f32; channel, ..spatial])
    -> [f32; ..spatial, channel]
{
    // channel moves to the end; spatial dimensions move to the front
    x[..spatial, channel]   // not valid syntax, but the intent is clear
}
```

A pack can also be an explicit coordinate parameter:

```rust
fn id_axes[..axes](x: [f32; ..axes]) -> [f32; ..axes] {
    x
}
```

When a pack is ambiguous at the call site—when the compiler can't infer which axes belong to it—the caller disambiguates by grouping:

```rust
id_axes[(height, width)](x)   // pass height and width as a single pack
```

Pack parameters make coordinate-aware functions rank-polymorphic: the same function works on 2D, 3D, 4D, or higher-dimensional tensors, as long as the coordinate of interest exists somewhere in the shape.

---

## Selection Reductions

A normal reduction returns a combined value. A selection reduction returns an address:

```rust
let pred[b] = argmax[class](logits[b, class]);
```

`max[class](logits[b, class])` returns the maximum *value* for each `b`. `argmax[class](logits[b, class])` returns the index within the `class` domain where that maximum occurs. The result has the surviving coordinates—`[b]`—plus an implicit address domain: the integer stored at each `b` is understood to be an address in the `class` coordinate space.

This is more than a convenience. It is a contract. The return value of `argmax[class]` is not just an integer tensor. It is an integer tensor whose values are addresses in the `class` domain. Later code can use this address to index back into the `class` dimension of another tensor, and the compiler can verify that the index is used correctly.

Selection reductions are the natural complement to ordinary reductions. `sum` collapses a coordinate into a combined value. `argmax` pinpoints a location within the coordinate. Both consume the coordinate from the result. Both name it in the same bracket syntax. Both participate in the coordinate contract system.

---

## One Skeleton, Four Normalizations

Every normalization function follows the same coordinate skeleton: reduce to get statistics, broadcast statistics back, apply elementwise. The difference is only which coordinates play which roles:

| Function | Reduction coords | Broadcast coords | Survivors |
|---|---|---|---|
| Softmax | `q` (max), `k` (sum) | `m[b]` (broadcast `j`) | `b`, `j` |
| LayerNorm | `f` (mean ×2) | `gamma[f]`, `beta[f]` | `b`, `t`, `f` |
| RMSNorm | `f` (mean) | `gamma[f]` | `b`, `t`, `f` |
| GroupNorm | `c_in_group`, `i2`, `j2` | `gamma[g, c_in_group]`, `beta[g, c_in_group]` | `b`, `g`, `c_in_group`, `i`, `j` |

In a positional API, all four collapse to a single `dim` argument whose meaning shifts with the surrounding layout. `LayerNorm` and `RMSNorm` both use `dim=-1`—but normalize different statistics. `GroupNorm` uses three reduction dimensions buried in a `reshape` chain. The skeleton is invisible.

In a named-coordinate API, the skeleton is a template you can check. The reduction bracket names the consumed coordinates. The indexing pattern names the survivors. The broadcast parameters name the omission. A reviewer can verify that the broadcast coordinate in LayerNorm matches the broadcast coordinate in the gradient without reconstructing both from positional offsets.

This is abstraction: recognizing a pattern, naming it, and reusing it. The pattern is "normalize with named coordinates." Each instance fills in the specific coordinates. The skeleton is constant.

---

## Coordinate Facts and Their Flow

A coordinate name on a tensor is not just documentation. It is a **coordinate fact**—a piece of type-level information that the compiler tracks through the program.

Coordinate facts flow through pointwise operations automatically. If `x` carries the fact that it has coordinates `[b, class]`, then `x ** 2.0` also has coordinates `[b, class]`. You don't need to re-declare them:

```rust
let x[b, class] = logits[b, class] - max_logit[b];
let pred[b] = argmax[class](x ** 2.0);   // x ** 2.0 still has [b, class]
```

The compiler preserves coordinate facts through arithmetic, through function calls that return tensors, through `if` expressions. The only operations that change coordinate facts are those that explicitly manipulate coordinates: reductions (which consume), rectangular declarations (which introduce), and coordinate-aware function calls (which thread them through signatures).

This flow is the foundation of abstraction. When you wrap a computation in a coordinate-aware function, the coordinate facts flow from the caller, through the function body, and out to the result. The function doesn't need to re-derive what the caller already knew. The facts propagate.

---

## Abstraction as Naming

Abstraction means naming a pattern so you can use it as if it were primitive. In einlang, the patterns you name are coordinate patterns—"normalize over this coordinate," "select the maximum index along this coordinate," "contract these two coordinates and leave the rest."

A coordinate-aware function is a named coordinate pattern. The name goes before the brackets. The pattern's parameters—which coordinates it consumes, which it preserves, which it is polymorphic over—go in the brackets and the type signature. The pattern's implementation goes in the body. The pattern's contract is enforced by the compiler at every call site.

This is the purpose of abstraction: not just factoring out repeated code, but creating a new vocabulary in which to think about the problem. After you define `fn softmax[j](...)`, you stop thinking about max-then-exp-then-sum-then-divide. You think "softmax over `j`." The implementation details recede. The coordinate contract remains visible.

Writing `fn top1[class](x: [f32; ..left, class, ..right]) -> [i32; ..left, ..right]` with `argmax[class]` consumes `class` from the result—it's absent from the return type—while `..left` and `..right` pass through intact. The function body is short: `argmax[class](x)`. The coordinate story is in the signature, not the implementation. For `fn rms_norm[feature](x: [f32; ..batch, feature], gamma: [f32; feature]) -> [f32; ..batch, feature]`, the formula `x * gamma / sqrt(mean(x^2) + eps)` uses rest packs so the function works on input shapes `(batch, feature)`, `(batch, seq, feature)`, or any structure around `feature`. The compiler checks that `gamma` omits every coordinate in `..batch`. And a simple case: if `x[b, c]` has coordinates `[b, c]` and you compute `let pred[b] = argmax[c](x[b, c])`, then `pred` has coordinates `[b]` and `pred[0]` is an index into the `c` dimension. The coordinate that was consumed tells you what `pred` indexes into.
