---
layout: book
title: "Chapter 6 · Names in Time"
---

# Chapter 6 · Names in Time

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

Now look at this line:

```
let h[t in 0..T] = step(h[t+1], x[t])
```

If someone wrote this, what should happen? Stop and derive the rule yourself. The declaration says `t in 0..T`—the statement defines `h` at time `t`. The body references `h[t+1]`—a value at time `t+1`. At the moment `h[t]` is being computed, `h[t+1]` has not been computed yet. `t+1` is strictly greater than `t`. The rule: **every index reference to the declared variable must be strictly less than the declared index.** `t+1 < t` is false. Error.

The check does not need to know that `t` is "time." It does not need to know what "causality" means. It does exactly one thing: compare the reference index against the declared index, for every reference to the declared variable in the body. Reference index `<` declared index? Valid. Otherwise? Rejected. The coordinate can be called `t`, `x`, or `spatial_index`—the check is the same. Causality is not a name-declared property. It is subtraction.

This has a consequence that spatial coordinates don't require. Only the time steps that are actually referenced backward need to be kept. If every step references only `t-1`, the storage needed is a rolling window of size 2, regardless of whether `T` is 100 or 100,000. This follows mechanically from the backward references—no annotation needed.

Now let's put this mechanism to work.

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

There is a quieter distinction at work here that deserves explicit attention: **parameters versus hyperparameters.** Both are `let` bindings. Both are immutable values in scope for subsequent code. But they have different roles in the optimization story:

```rust
let weight: [f32; out, in] = init_random(out, in);
let learning_rate: f32 = 0.001;
```

`weight` is a parameter—its value changes during training, driven by gradients. Each coordinate on `weight` tells the optimizer something: `out` names the output neurons, `in` names the input connections. A weight decay regularizer that treats all elements uniformly can apply without coordinate awareness, but a per-neuron regularization policy needs to know which axis is `out` and which is `in`.

`learning_rate` is a hyperparameter—it controls the optimizer's behavior but is not itself updated by gradients. It carries no coordinate names because it has no coordinate structure.

This distinction is not a language feature. It is a naming discipline. But the discipline is only possible because the language provides a place to put the coordinate names. When your tensor framework only records shapes, you can't name the axes. When you can't name the axes, you can't write a regularizer that distinguishes them. The optimizer sees `(128, 64)` and doesn't know which number is `out` and which is `in`. The coordinate names on the parameter are the bridge between the optimizer's generic arithmetic and the architecture's specific structure.

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

Now stop. Someone writes this:

```
let h[t in 0..T, i] = step(h[t+1, i], x[t, i]);
```

No iteration direction declared. Just `t in 0..T`. The body references `t+1`. Is this valid or not?

It depends on whether the compiler infers the iteration direction from the reference pattern. If `t+1` references a time step that hasn't been computed yet (because iteration goes left to right), this is a forward reference and should be rejected. But if the compiler can infer that `t` should iterate from `T` down to `0`—making `t+1` already computed—it could be valid.

The einlang rule is conservative: without an explicit reverse-direction declaration, forward references are rejected. `t+1` with `t in 0..T` is an error. The programmer must write `t in T..0` (or equivalent syntax) to declare the reverse iteration. The tool prevents the ambiguous case by default.

This is the same design choice as Chapter 3's Coordinate Contract and Chapter 4's Pack Disambiguation: when a reference pattern is ambiguous, the language requires the programmer to disambiguate. Default deny. Explicit allow.

---

## The Rolling Window: What Causality Buys

Causality is not just a correctness check. It is a memory optimization.

When a recurrence body only references `t-1`, the compiler knows that only one previous time step is needed. It can allocate a rolling window of size 2 rather than storing the entire `(T, ...)` tensor in memory. When `T` is 100,000, this is the difference between allocating gigabytes and allocating megabytes.

This optimization follows mechanically from the backward references. The compiler scans the body for time-indexed references. Every reference to `t - k` (positive `k`) requires storing `k` previous steps. The rolling window size is `max(k)`. No annotation needed. The coordinate names and index arithmetic carry enough information for the compiler to derive the memory plan.

Consider a second-order recurrence:

```
let u[t in 2..T, i] = u[t-1, i] + 0.5 * (u[t-1, i] - u[t-2, i]);
```

References: `t-1` and `t-2`. Maximum offset: 2. Rolling window size: 3 (current, t-1, t-2). The compiler derives this from the index expressions. No `@roll_window(3)` annotation. The information is in the code, not in a compiler directive.

This is the pattern that recurs throughout this book: **make the structural fact visible in the source code, and let the compiler derive the engineering consequence.** The programmer writes `t-2`. The compiler derives window size 3. The programmer writes `sum[class]`. The compiler derives `axis=1`. The programmer writes `bias[j]` omitting `i`. The compiler derives the backward-pass sum over `i`. Source records intent. Compiler derives execution.

---

## Time in the Training Loop

The optimizer recurrence from earlier is worth tracing step by step:

```
// Step 0: random initialization
let w[0, out, in] = init_random(out, in);

// Step 1: forward pass
let logits[1, b, class] = model(x[1, b, feature], w[0, out, feature]);
let loss[1] = cross_entropy[class](logits[1, b, class], labels[1, b]);

// Step 1: backward pass
let grad[1, out, feature] = @loss[1] / @w[0, out, feature];

// Step 1: update
let w[1, out, feature] = w[0, out, feature] - lr * grad[1, out, feature];

// Step 2: forward pass
let logits[2, b, class] = model(x[2, b, feature], w[1, out, feature]);
// ... and so on
```

At each time step, three things happen: forward (model produces output), backward (gradient is computed), update (weights move against the gradient). The time index `t` is explicit on every tensor. You can read the value of `w` after any step. You can read the loss at any step. The training trajectory is a tensor, not a sequence of in-place mutations.

Now compare to the PyTorch equivalent:

```python
w = init_random(out, in)
for t in range(1, T):
    logits = model(x[t], w)
    loss = cross_entropy(logits, labels[t])
    loss.backward()
    with torch.no_grad():
        w -= lr * w.grad
```

`w` is a single mutable tensor. `loss` is a scalar. The time dimension is the loop variable `t`—visible in the Python control flow but absent from the tensor structure. You cannot inspect `w[10]` without checkpointing the value at step 10 yourself. The training trajectory exists in execution time, not in the type system.

The einlang version makes the training trajectory a data structure. The PyTorch version makes it a side effect. The difference is whether you can query the past.

---

## Diffusion Models: Time as a Coordinate with a Schedule

Diffusion models are the most time-intensive architecture in modern ML. A forward process adds noise over `T` timesteps. A backward process learns to reverse the noising. The time coordinate appears in two roles: as a recurrence index for the sampling chain, and as a conditioning signal for the denoising network.

```
let x[t in 0..T, b, c, h, w] = ...;
let eps[t in 1..T, b, c, h, w] = noise_schedule[t] * randn(...);
let x[t in 1..T, b, c, h, w] = sqrt(1 - beta[t]) * x[t-1, ...] + sqrt(beta[t]) * eps[t, ...];
```

The time index `t` threads through the forward noising process. At each step, noise is added. The schedule `beta[t]` controls how much noise—and `beta` is indexed by `t`, making the dependency visible.

In the backward pass (the learned denoising):

```
let x_hat[t in T..1, b, c, h, w] = denoise(x[t, ...], t, model(x[t, ...], t));
```

The iteration runs backward: `T..1`. The model receives `t` as conditioning—it needs to know which timestep it's denoising. In a positional framework, `t` is a positional encoding vector concatenated or added to the input, and the loop runs in Python. In einlang, `t` is a coordinate that flows through the model call: the model's signature can declare `fn denoise[t, ...](x: [f32; t, ...])` and the coordinate `t` is carried alongside the tensor data.

This is the same mechanism that carried `class` through `softmax[class]` in Chapter 3, applied to time. The coordinate is the same kind of thing. The direction—forward or backward—is the only difference.

Time is not "special." It is a coordinate with a direction constraint. The constraint is checked. The coordinate flows through functions. The training loop is a recurrence. The diffusion process is a recurrence. The optimizer is a recurrence. Three domains, one mechanism. The names make them recognizable as the same thing. It carries direction. It carries dependency. It carries a constraint: you can only look backward along it. These properties are not metaphorical. They are enforced at the level of index expressions. A colleague who writes `let h[t in 0..T] = step(h[t+1], x[t])` will get a compile error—not a runtime divergence, not a silent wrong answer. The syntax makes the constraint checkable.

Now pause. Before you move to the next chapter, answer this: what other tensor operations have an implied direction? Think about your own code. Have you ever written a recurrence where the time axis was not the first axis? Where the dependency went both forward and backward? Where the "time" was not time at all—but a layer index in a residual network, an iteration counter in an optimizer, a step in a diffusion process?

Recurrence is not unique to RNNs. Every iterative computation is a recurrence. Every optimizer step is a recurrence. Every diffusion timestep is a recurrence. The coordinate `t` is not "the time axis." It is "the axis along which things depend on earlier things." Causality is the constraint. The directional coordinate is the mechanism that enforces it.

---

## The Gradient of a Recurrence

Recurrences have gradients. And because recurrences are self-referential—each step depends on the previous step—the gradient must flow backward through time. This is Backpropagation Through Time (BPTT), and its coordinate structure is the same recurrence, read in reverse.

Forward: `h[t] = step(h[t-1], x[t])`. The output `h[t]` depends on `h[t-1]`, which depends on `h[t-2]`, and so on back to `h[0]`.

Backward: the gradient `d_loss/d_h[t]` must propagate to `d_loss/d_h[t-1]`, then to `d_loss/d_h[t-2]`, and so on. At each step, the gradient flows through the `step` function's Jacobian with respect to `h[t-1]`. The backward recurrence:

```
let d_h[t in T..0] = @loss[t] / @h[t] + @step(h[t], h[t-1], x[t]) / @h[t] * d_h[t+1];
```

The backward recurrence runs from `T` down to `0`, referencing `t+1` (the future in the backward direction, which has already been computed). This is the same bidirectional mechanism from Section 6, applied to the gradient. The coordinate `t` still carries the causality constraint, but the iteration direction has reversed.

In a positional framework, BPTT is implemented by writing a separate backward loop that iterates in reverse. The relationship between the forward loop `for t in range(T)` and the backward loop `for t in reversed(range(T))` is in the programmer's head—the two loops are separate code blocks. In einlang, the backward recurrence is generated from the forward recurrence by the same Inversion Rule that governs reductions and broadcasts. The forward recurrence declares `t in 1..T` with `t-1` references. The backward recurrence is `t in T..0` with `t+1` references, generated automatically. The time direction flips. The coordinate names stay the same. The compiler generates the backward loop from the forward declaration.

In the next chapter, we explore what happens when coordinates split, merge, and carry arithmetic—the complex terrain of distance matrices, convolutions, and fancy indexing.
