---
layout: book
title: "Appendix: Coordinate Diagnostics"
---

# Coordinate Diagnostics

This appendix is a quick lookup table for the stubborn failures that appear
throughout the book. The pattern is always the same: a shape can be plausible
while the coordinate story is wrong.

## Common Failure Patterns

```text
1. Same shape, swapped roles
   Symptom: transpose, reshape, or flatten keeps the expected extent.
   Diagnostic: write the output address as a function of named input
   coordinates. Ask which coordinate is slow, fast, packed, or unpacked.

2. Broadcast over the wrong role
   Symptom: a bias, mean, mask, or parameter has a compatible length but is
   reused across the wrong axis.
   Diagnostic: list the coordinates each term mentions. The missing coordinate
   is the broadcast coordinate.

3. Reduction consumes the wrong coordinate
   Symptom: row sum becomes column sum, feature normalization becomes batch
   normalization, or a square matrix hides the error.
   Diagnostic: circle the coordinates on the left-hand side. Anything reduced
   should be absent there for a reason.

4. One logical axis has several local scopes
   Symptom: softmax, normalization, or attention is described as operating
   "over the axis" but the formula confuses output, scan, and stability roles.
   Diagnostic: give each binding site its own name, such as `j`, `k`, and `q`.

5. Pullback sums the wrong route
   Symptom: a gradient has a plausible rank but a surprising magnitude or
   sharing pattern.
   Diagnostic: hold one denominator cell fixed. List every output cell that
   read it. Sum exactly those sensitivity routes.

6. Time is hidden inside mutation
   Symptom: loop code makes it unclear whether a state reads the previous
   point, the future point, or a larger window.
   Diagnostic: rewrite the state as `h[t]` and mark every read of `h[t - n]`
   or `h[t + n]`.

7. Storage is inferred from notation too early
   Symptom: an indexed recurrence is assumed to require a full array, or a
   small dependency window is assumed to require small memory.
   Diagnostic: separate definition, observation, and storage policy.

8. Attention uses the right tensors at the wrong position
   Symptom: `Q`, `K`, `V`, weights, and output all have compatible shapes, but
   values are gathered from the query position instead of the key position.
   Diagnostic: state the communication sentence: `i` asks, `j` answers, and
   `V[j, d]` is carried back.

9. Low-rank attention hides the bottleneck
   Symptom: a linear attention implementation has no explicit `i, j` score
   table, so the approximation looks like an ordinary feature transform.
   Diagnostic: name the bottleneck coordinate `r` and ask where `i` stops
   communicating directly with `j`.

10. Dynamic routing hides dropped or overloaded tokens
    Symptom: an MoE layer scatters tokens into experts, but capacity overflow
    or expert imbalance is visible only as a mask tensor or aggregate metric.
    Diagnostic: name `route[b, t]`, `slot[b, t]`, and `keep[b, t]`; then reduce
    over token coordinates to compute per-expert load.

11. Coordinate function hides the wrong fact
    Symptom: a helper such as normalization, selection, pooling, or routing is
    short enough to trust, but the call site no longer says which coordinate it
    consumes or returns as an address.
    Diagnostic: rewrite the call with bracketed coordinate arguments, such as
    `softmax[class]` or `argmax[expert]`. If the coordinate cannot be grounded
    in the argument, the helper is hiding too much.

12. Coordinate function asks for too much
    Symptom: the call names every surrounding axis even though only one role is
    the choice. The extra names make the abstraction look ceremonial.
    Diagnostic: move the surrounding axes into a pack in the signature. Prefer
    `move_channel[channel](x)` over a call that repeats `height` and `width`
    when those coordinates are already determined by `x`.
```

## A Four-Step Audit

For any suspicious tensor line, use this audit before reading the surrounding
implementation:

```text
1. Name the result coordinates.
2. For each term, list the coordinates it reads.
3. Mark local coordinates introduced by reductions, normalizations, or scans.
4. Ask which coordinates survive, which are omitted, and which are consumed.
```

If the line is differentiable, add one more step:

```text
5. For each requested gradient, write the denominator coordinates first.
```

The denominator address is the shape of the answer. Any other coordinate that
appears in the route has to be reduced, broadcast, or justified by the chain
rule.

## The Fastest Question

When there is time for only one question, ask:

```text
Would the wrong role still have the right shape?
```

If yes, write the coordinates down. That is where the bug usually hides.
