# Why Einlang?

**Tensor code is either readable or safe—usually neither.** Einlang is both. This page brags a bit: what makes Einlang worth your time and how it stands out.

---

## The pitch

- **Math on the page** — Write what you’d write on a whiteboard. Indices, sums, where-clauses, and derivatives are **syntax**, not string APIs or callback libraries. If it type-checks, the shapes are correct.
- **One language for simulation and ML** — ODEs, PDEs, recurrences, gradient descent, calibration, and neural nets use the same notation and the same compiler. No switching between “numerical” and “differentiable” dialects.
- **Gradients without gradient code** — `@loss / @w` is the gradient. The compiler derives it; you never write backprop or VJPs by hand. Same mechanism for training, sensitivity analysis, and adjoints.
- **No stringly-typed einsum** — No `einsum('ik,kj->ij', A, B)`. Indices are first-class; the compiler infers ranges from shapes and catches rank and dimension errors at compile time.

---

## Features that pull their weight

| Feature | What you get |
|--------|----------------|
| **Einstein notation as syntax** | `let C[i, j] = sum[k](A[i, k] * B[k, j]);` — indices and shapes checked at compile time. Wrong dimensions → compile error, not a runtime crash. |
| **Where clauses** | Index algebra (`ih = oh + kh`), guards (`where data[i] > 0`), and bindings live next to the computation. Convolutions, stencils, and masks read like the math. |
| **Recurrences as declarations** | Base cases + recursive rule; range in the bracket; compiler handles evaluation order. RNNs, dynamic programming, and time stepping without manual loop wiring. |
| **Built-in autodiff** | `@z / @x` is the derivative. One primitive (`@`), one quotient form. No tapes, no dual types, no separate AD library. The compiler expands to forward diff and fills gradient slots. |
| **Same shapes for gradients** | Gradient w.r.t. a variable has the same shape as that variable. No surprise reshapes or “grad has wrong size” at runtime. |
| **300+ stdlib functions** | `use std::math::{sin, sqrt, exp};` — same language, same shape checking. No ad-hoc FFI for basic math. |
| **Real models in one language** | MNIST CNN, quantized (int8) CNN, ViT, Whisper — same syntax, same checks. Simulation examples are [accuracy-tested against Julia](https://github.com/einlang/einlang/blob/main/docs/JULIA_DEMOS.md). |

---

## How we’re different

- **Not “einsum in a string”** — NumPy’s `einsum('ik,kj->ij', A, B)` gives you no static checking. Einlang’s indices are part of the language; the compiler sees every index and validates shapes and ranks.
- **Not “gradient as a separate API”** — You don’t call `gradient(f, x)` or `jax.grad(f)(x)`. You write `@loss / @w` where the math is. One mechanism for all derivatives.
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

**Next:** [Getting started](GETTING_STARTED.md) · [Autodiff design](AUTODIFF_DESIGN.md) · [Examples](https://github.com/einlang/einlang/tree/main/examples)
