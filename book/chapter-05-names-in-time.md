---
layout: book
title: "Chapter 5 · Names in Time"
---

# Chapter 5 · Names in Time

> "Time is what keeps everything from happening at once."
>
> — Attributed to John Archibald Wheeler

*Combinations · Recurrence and causality*

---

Every coordinate we have met so far is a spatial coordinate. You can sum over it, broadcast along it, permute it. All positions along the coordinate exist simultaneously. No position depends on any other.

Time is different. Time has a direction.

---

## Not All Axes Are the Same

Look at this declaration:

```
u[t, i] = u[t-1, i] + f(u[t-1, i])
```

`t` and `i` are both written in brackets. But `t` does something no spatial coordinate does—it looks backward. `t-1`.

In a spatial expression like `sum[i](A[i, k])`, `i` is just an index. You never write `i-1`. Spatial indices are concurrent—you sum over them, reduce along them, permute them—but you don't *recur* along them.

`t` is different. `t-1` means step one depends on step zero, step two depends on step one. `t` is the direction of recurrence. Not space. Time.

---

## Recurrence Declarations

In einlang, time is just another coordinate—but one that appears in index arithmetic. You declare it with a range:

```
let u[t in 0..T, i] = init_temp(i);
let u[t in 1..T, i] = u[t-1, i] + alpha * (
    u[t-1, i+1] - 2.0 * u[t-1, i] + u[t-1, i-1]
);
```

The first clause defines `u` at `t=0`—the initial condition. The second clause defines `u` at every subsequent time step in terms of the previous step. `t-1` is a backward reference: the value at time `t` depends on the value at time `t-1`.

This is a recurrence. The coordinate `t` carries time's directional structure into the notation. You cannot write `u[t+1, i]` to define `u[t, i]`—that would be a forward reference, and it is rejected as a static error. Causality is not a comment. It is a syntactic constraint. If the index expression references an index greater than or equal to the declared index, it is a static error. This is not philosophy. It is subtraction: `t-1 < t`, valid; `t+1 > t`, rejected.

The declaration bracket names *what* is being defined. The body says *how*. The separation keeps the declaration side simple and declarative, while the body can use arbitrary index arithmetic:

```
let fib[0] = 0;
let fib[1] = 1;
let fib[n in 2..8] = fib[n-1] + fib[n-2];
```

The recurrence index range `n in 2..8` goes in the declaration bracket—it defines the domain. The expressions `n-1` and `n-2` go in the body—they compute the value by referencing earlier elements. Every reference must point strictly backward.

---

## The Optimizer as a Recurrence

Training a model is a recurrence over time:

```
let w[t in 0..T, out, in] = init_random(out, in);
let w[t in 1..T, out, in] = w[t-1, out, in] - lr * grad[t-1, out, in];
```

At `t=0`, `w` is the random initialization. At each subsequent step, `w` is the previous `w` minus a gradient step. The recurrence reads backward in time (`t-1`). The time coordinate `t` makes the training trajectory explicit. You can inspect `w[10, out, in]` to see the weights after 10 steps. You can compute `w[T-1, out, in] - w[0, out, in]` to see the total change. The time dimension is not hidden inside a mutable variable—it is a coordinate like any other.

A full training step:

```
let logits[t, b, class] = model(x[t, b, feature], w[t, out, feature]);
let loss[t] = cross_entropy[class](logits[t, b, class], labels[t, b]);
let grad[t, out, feature] = @loss[t] / @w[t, out, feature];
let w[t+1, out, feature] = w[t, out, feature] - lr * grad[t, out, feature];
```

The time coordinate `t` threads through forward, loss, gradient, and update. Every tensor knows its temporal position. The gradient `@loss[t] / @w[t, out, feature]` is explicitly anchored to time step `t`. The optimizer step defines `w[t+1]` in terms of `w[t]` and `grad[t]`.

---

## Time Is an Axis with a Direction

On a spatial axis, all positions exist concurrently. You can sum over them in any order. On a time axis, position `t` depends on position `t-1`. Not concurrency. Dependency.

This distinction has consequences. A recurrence carries two properties that spatial coordinates don't require:

1. **Causality**: every time-indexed reference in the body must be strictly less than the declared time index. `t-1` is valid. `t+1` is a compile error.
2. **Memory**: only the time steps that are actually referenced backward need to be kept in memory. If every step references only `t-1`, the storage needed is a rolling window of size 2, regardless of whether T is 100 or 100,000. This follows mechanically from the backward references—no annotation needed.

---

## Bidirectional Recurrence

Not all recurrences look only to the past. A bidirectional RNN reads the sequence both ways:

```
let h_forward[t in 1..T, i] = step(h_forward[t-1, i], x[t, i]);
let h_backward[t in 0..T-1, i] = step_back(h_backward[t+1, i], x[t, i]);
```

The forward recurrence reads `t-1`—standard. The backward recurrence reads `t+1`—the future from the perspective of `t`. This is still valid because the backward recurrence iterates from right to left: `t` runs from `T-1` down to `0`, so `t+1` is always already computed. The direction of iteration determines which references are "backward."

The same coordinate domain, two different iteration directions, one linguistic mechanism. The declaration bracket names the domain and direction. The body states the dependency. Consistency is checked.

---

Time is an axis, but not a spatial one. It carries direction. It carries dependency. It carries a constraint: you can only look backward along it. These properties are not metaphorical. They are enforced at the level of index expressions. A colleague who writes `let h[t in 0..T] = step(h[t+1], x[t])` will get a compile error—not a runtime divergence, not a silent wrong answer. The syntax makes the constraint checkable.

In the next chapter, we explore what happens when coordinates split, merge, and carry arithmetic—the complex terrain of distance matrices, convolutions, and fancy indexing.
