
# Automatic differentiation: potential future feature

**Status:** Not planned for the current release. Today gradients are **explicit by design**; see [Unsupported by design §12](UNSUPPORTED.md#12-no-automatic-differentiation-autodiff). This document records design notes and constraints **if** the project ever revisits adding autodiff.

---

## Current policy

- **No automatic differentiation** in the language: no `grad(f, x)` or similar where the compiler derives ∂f/∂x.
- **Why:** Einlang aims for clarity and explicitness — no magic in the syntax. What you write is what is computed.
- **Use instead:** Explicit gradients (e.g. [numerics::optim](https://github.com/einlang/einlang/blob/main/docs/stdlib.md), hand-written as in [Rosenbrock](https://github.com/einlang/einlang/tree/main/examples/optimization/rosenbrock.ein), or grid search for calibration).

---

## If we ever add it: design constraints

- **Minimal surface:** One (or very few) primitives, e.g. `grad(expr, x)` or `grad w.r.t. x of { expr }`. No user-visible tapes, dual types, or gradient-specific syntax beyond that.
- **Same shapes:** Gradient of a value w.r.t. a variable has the **same shape** as that variable. The compiler can check this at compile time.
- **Explicit generated code (optional):** Prefer generating the backward pass as normal IR (or even readable Einlang-like code) so the derivative is inspectable rather than opaque.

---

## Possible syntax (exploratory)

- **Expression-level:** `let g = grad(loss, x);` — `loss` is a scalar expression, `x` a variable; `g` has the same shape as `x`.
- **Scoped:** `let g = grad w.r.t. x of { ... };` so the differentiated expression is delimited.
- **Function-level:** `let g = grad_arg0(f)(x, A, b);` — gradient of `f` w.r.t. its first argument, then call.

---

## Possible implementation: source / IR transformation

- **Idea:** The compiler already has IR (expression tree, lowered Einstein, recurrence). Add a **pass** that either:
  - **Forward mode:** Transforms the program so values carry tangents; one forward run yields value + derivative. Or
  - **Reverse mode:** Builds a **backward pass** from the IR — for each op, emit IR that implements the VJP (vector–Jacobian product) and accumulates gradients into inputs. No runtime tape; the backward pass is generated code.
- **Fit:** Einlang’s IR (e.g. `LoweredEinsteinClauseIR` with body, loops, reductions) is explicit; a differentiation pass could walk it and emit derivative IR. Recurrence would require a backward-time loop that applies VJPs step by step.
- **Hard rules for gradients** (any implementation must respect):
  - **Differentiability:** ∇f(x) exists only where f is differentiable w.r.t. x (no kinks, jumps, or undefined derivatives at the point of use).
  - **Shape:** ∇f(x) has the same shape as x (scalar→scalar, vector→vector, matrix→matrix).
  - **Calculus:** Sum, product, and chain rule must hold (and scalar multiple).
  - **Well-defined:** No NaNs/Infs; gradient formula must match the objective.

---

## Real-world use cases (what it would enable)

| Use case | Today in Einlang | With autodiff (hypothetical) |
|----------|------------------|------------------------------|
| **Calibration** | Grid search (e.g. decay_fit) or hand-written ∂loss/∂params | One loss expression; optimizer uses `grad(loss, params)` |
| **Inverse / control** | Forward only; optimizer elsewhere or hand adjoint | Loss over control; minimize with `grad(loss, control)` |
| **Small optimization** | Hand-written gradients (Rosenbrock, numerics::optim) | `grad(f, x)` instead of deriving by hand |
| **Sensitivity** | Finite diff or hand-derived | `grad(y, params)` on same forward code |
| **Train small model** | Forward in Einlang; training in Python | One loss; `grad(loss, weights)` and update in Einlang |

---

## Optimization steps (for reference)

Gradient-based minimization repeats:

1. **Gradient:** Compute g = ∇f(x).
2. **Update:** x_next = x - α * g (for a step size α > 0).
3. Repeat or stop (e.g. when ‖g‖ is small).

So any autodiff feature would need to supply g in step 1; the rest is already expressible with recurrence and existing stdlib (e.g. gradient_descent_step).

---

## Summary

- **Today:** No autodiff; explicit gradients only. See [UNSUPPORTED §12](UNSUPPORTED.md#12-no-automatic-differentiation-autodiff).
- **If revisited:** Keep the surface minimal (one grad primitive, same shapes, optional generated backward as explicit code). Implementation path: source/IR transformation (forward or reverse). This doc is the single place for design and constraints.
