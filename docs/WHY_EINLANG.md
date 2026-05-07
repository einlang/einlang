# Why Einlang?

**Tensor code is often either readable or safe, but not both.** Einlang aims for both. This page covers the main differences from adjacent tools.

**Autodiff note:** We do not bolt on AD with helper APIs; Einlang compiles derivative syntax on expressions directly into executable IR for tensor programs, so supported sensitivities come from the compiler instead of function wrappers or finite-difference probes.

---

## The pitch

- **Math on the page** — Write what you’d write on a whiteboard. Indices, sums, where-clauses, and derivatives are **syntax**, not string APIs or callback libraries. If it type-checks, the shapes are correct.
- **One language for simulation and ML** — ODEs, PDEs, recurrences, gradient descent, calibration, and neural nets use the same notation and the same compiler. No switching between “numerical” and “differentiable” dialects.
- **Gradients without gradient code** — `@loss / @w` is the gradient, `@C / @A` is the derivative tensor, and `@x` exposes a tangent request for a named value. The compiler/runtime derive them where they appear in the program; you never write backprop or VJPs by hand. Same mechanism for training, sensitivity analysis, and adjoints, and a cleaner replacement for finite-difference gradients where the operation is supported.
- **Autodiff as expression syntax, not a function API** — You do not import an AD package, wrap a function, or build a tape. `@` and `/` in `@f / @x` are parsed and lowered like any other operator; the autodiff pass is part of the same compiler that does shapes and Einstein lowering.
- **No stringly-typed einsum** — No `einsum('ik,kj->ij', A, B)`. Indices are first-class; the compiler infers ranges from shapes and catches rank and dimension errors at compile time.
- **Coordinate-aware functions** — Common helpers can stay compact while still naming the coordinate they use: `softmax[class](logits)` normalizes over `class`, and `argmax[class](logits)` returns addresses in the `class` domain.

---

## Core features

| Feature | What you get |
|--------|----------------|
| **Einstein notation as syntax** | `let C[i, j] = sum[k](A[i, k] * B[k, j]);` — indices and shapes checked at compile time. Wrong dimensions → compile error, not a runtime crash. |
| **Where clauses** | Index algebra (`ih = oh + kh`), guards (`where data[i] > 0`), and bindings live next to the computation. Convolutions, stencils, and masks read like the math. |
| **Recurrences as declarations** | Base cases + recursive rule; range in the bracket; compiler handles evaluation order. RNNs, dynamic programming, and time stepping without manual loop wiring. |
| **Coordinate-aware calls** | `softmax[class](logits)` and `argmax[class](logits)` keep axis-sensitive operations short without falling back to `axis=1` convention. |
| **Built-in autodiff** | `@z / @x` is the derivative, `@z` is a tangent request on a named binding, and tensor expressions like `@C / @A` are first-class. One primitive family, quotient form for partials, and `print(@y)` for symbolic debugging. **No tapes, no dual numbers, no `grad(f)(x)` API**: the compiler rewrites autodiff requests to internal runtime intrinsics backed by the same NumPy execution stack as the primal. For supported ops, this replaces finite-difference gradient checks with compiler-derived derivatives. |
| **Same shapes for gradients** | A gradient with respect to a variable has the same shape as that variable. No surprise reshapes or "grad has wrong size" errors at runtime. |
| **300+ stdlib functions** | `use std::math::{sin, sqrt, exp};` — same language, same shape checking. No ad-hoc FFI for basic math. |
| **Real models in one language** | MNIST CNN, quantized CNN, ViT, Whisper, and numerical examples all use the same core language. |

---

## How we’re different

- **Not “einsum in a string”** — NumPy’s `einsum('ik,kj->ij', A, B)` gives you no static checking. Einlang’s indices are part of the language; the compiler sees every index and validates shapes and ranks.
- **Not “gradient as a separate API”** — You don’t call `gradient(f, x)` or `jax.grad(f)(x)`. You write `@loss / @w`, `@state / @dt`, or `@C / @A` exactly where the math is. One mechanism for all derivatives, and usually the first thing to reach for instead of finite differences when you need sensitivities.
- **Not “axis as an integer convention”** — You can write `softmax[class](logits)` instead of remembering whether `class` happens to be axis `1`, `-1`, or something else after a reshape. The same bracket syntax distinguishes three different normalizations that a positional API can only express as three different `dim` integers:

```python
# PyTorch — all produce [128, 128, 128]. Only the dim integer differs.
F.layer_norm(x, (128,))               # dim=-1: normalize over feature
(x - x.mean(1)) / x.std(1)             # dim=1: normalize over time
(x - x.mean(0)) / x.std(0)             # dim=0: normalize over batch
```

```rust
// Einlang — the bracket names the coordinate role.
normalize_over[feature](x[batch, time, feature])
normalize_over[time](x[batch, time, feature])
normalize_over[batch](x[batch, time, feature])
```

When `batch == time == feature == 128`, all six calls produce the same output shape. The PyTorch calls differ by one integer. The Einlang calls differ by one name. Swap `time` and `feature` in the tensor layout, and `dim=1` silently changes meaning. `normalize_over[time]` does not — `time` is still `time`, wherever it lives in the shape.
- **Not “loops + manual indexing”** — Recurrences and reductions are declarations with ranges; the compiler handles order and lowering. You write the recurrence, not the loop.
- **Not “simulation vs ML split”** — One language. ODE/PDE examples and MNIST/ViT/Whisper use the same notation, same autodiff, same stdlib.

---

## Who it’s for

- **Scientists and engineers** who want tensor math that looks like the equation and fails at compile time when shapes are wrong.
- **ML practitioners** who want gradients without writing backprop or depending on a separate AD framework.
- **Teachers and students** who want a single, consistent story: indices, sums, where-clauses, and derivatives in one syntax.
- **Anyone tired of “readable or safe—pick one.”** Einlang is both.

---

## Try it

```bash
git clone https://github.com/einlang/einlang.git
cd einlang
pip install -e .
python3 -m einlang examples/hello.ein
```

**Next:** [Getting started](GETTING_STARTED.md) · [Coordinate functions](COORDINATE_FUNCTIONS.md) · [Autodiff](AUTODIFF.md) · [Examples](../examples/README.md)
