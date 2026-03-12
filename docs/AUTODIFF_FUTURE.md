
# Automatic differentiation: potential future feature

**Status:** Not planned for the current release. Today gradients are **explicit by design**; see [Unsupported by design §12](UNSUPPORTED.md#12-no-automatic-differentiation-autodiff). This document records design notes and constraints **if** the project ever revisits adding autodiff.

---

## Current policy

- **No automatic differentiation** in the language: no `grad(f, x)` or similar where the compiler derives ∂f/∂x.
- **Why:** Einlang aims for clarity and explicitness — no magic in the syntax. What you write is what is computed.
- **Use instead:** Explicit gradients (e.g. [numerics::optim](https://github.com/einlang/einlang/blob/main/docs/stdlib.md), hand-written as in [Rosenbrock](https://github.com/einlang/einlang/tree/main/examples/optimization/rosenbrock.ein), or grid search for calibration).

---

## If we ever add it: design constraints

- **Math formula first:** Syntax must read like the math (ASCII only). The gradient/derivative form in code should be the one you'd write on the board (e.g. dL/dx or ∇_x L). This overrides "looks like a normal function call" — we prefer formula-like even if it needs a small amount of dedicated syntax.
- **Minimal surface:** One (or very few) primitives; no user-visible tapes, dual types, or gradient-specific machinery beyond that.
- **Same shapes:** Gradient of a value w.r.t. a variable has the **same shape** as that variable. The compiler can check this at compile time.
- **Explicit generated code (optional):** Prefer generating the backward pass as normal IR (or readable Einlang-like code) so the derivative is inspectable.

---

## Syntax: three choices (ASCII, no operator reuse)

We do **not** use `d(expr)/d(x)` because `/` is already division. Three options:

---

### Option A: **grad[x](expr)**

- **Form:** `grad` + bracket `[x]` ("with respect to x") + parentheses `(expr)`.
- **Meaning:** Gradient of expr with respect to the variable x. Same shape as x.

### Option B: **@grad(expr)(x)**

- **Form:** `@` + `grad` + `(expr)` + `(x)`. Two argument groups: expression first, then variable(s) w.r.t.
- **Meaning:** Gradient of expr with respect to x. Same shape as x.

### Option C: **grad(expr)(x)**

- **Form:** `grad` + `(expr)` + `(x)` — no @, no bracket. Two paren groups; `grad` is a builtin with two-call semantics.
- **Meaning:** Gradient of expr with respect to x. Same shape as x. (Same as B without the sigil.)

---

### Scorecard (each criterion 1–5, total out of 25)

| Criterion | A: grad[x](expr) | B: @grad(expr)(x) | C: grad(expr)(x) |
|-----------|------------------|-------------------|------------------|
| **Math intuitive** (reads like ∂/∂x or ∇_x) | 5 — variable on operator | 3 — variable in second group | 3 — same as B |
| **No operator reuse** | 5 | 5 | 5 |
| **Einlang consistency** (bracket = structure vs new sigil) | 5 — bracket = w.r.t. | 3 — @ is new | 4 — no new sigil; grad reserved |
| **Parseability** (unambiguous; grad special?) | 5 — [ ] after grad is special | 5 — @ marks special form | 4 — grad(expr)(x) looks like curried call; must reserve grad |
| **Readability** (short, clear for multiple vars) | 5 — grad[x](loss), grad[w](loss) | 4 — @grad(loss)(x) | 4 — grad(loss)(x), grad(loss)(w) |
| **Total** | **25/25** | **20/25** | **20/25** |

**Recommendation:** **Option A: grad[x](expr)**. Best formula match, bracket-as-structure, no new sigil. Option C ties B on score but without @ the two-call form is easier to confuse with a user function or curried call; B makes the special form explicit.

---

## Possible implementation: source / IR transformation

- **Idea:** The compiler already has IR (expression tree, lowered Einstein, recurrence). Add a **pass** that either:
  - **Forward mode:** Transforms the program so values carry tangents; one forward run yields value + derivative. Or
  - **Reverse mode:** Builds a **backward pass** from the IR — for each op, emit IR that implements the VJP (vector–Jacobian product) and accumulates gradients into inputs. No runtime tape; the backward pass is generated code.
- **Fit:** Einlang's IR (e.g. `LoweredEinsteinClauseIR` with body, loops, reductions) is explicit; a differentiation pass could walk it and emit derivative IR. Recurrence would require a backward-time loop that applies VJPs step by step.
- **Hard rules for gradients** (any implementation must respect):
  - **Differentiability:** ∇f(x) exists only where f is differentiable w.r.t. x (no kinks, jumps, or undefined derivatives at the point of use).
  - **Shape:** ∇f(x) has the same shape as x (scalar→scalar, vector→vector, matrix→matrix).
  - **Calculus:** Sum, product, and chain rule must hold (and scalar multiple).
  - **Well-defined:** No NaNs/Infs; gradient formula must match the objective.

---

## Real-world use cases (what it would enable)

| Use case | Today in Einlang | With autodiff (hypothetical) |
|----------|------------------|------------------------------|
| **Calibration** | Grid search (e.g. decay_fit) or hand-written ∂loss/∂params | One loss expression; optimizer uses grad[x](loss) |
| **Inverse / control** | Forward only; optimizer elsewhere or hand adjoint | Loss over control; minimize with grad[control](loss) |
| **Small optimization** | Hand-written gradients (Rosenbrock, numerics::optim) | grad[x](f) instead of deriving by hand |
| **Sensitivity** | Finite diff or hand-derived | grad[params](y) on same forward code |
| **Train small model** | Forward in Einlang; training in Python | One loss; grad[weights](loss) and update in Einlang |

---

## Optimization steps (for reference)

Gradient-based minimization repeats:

1. **Gradient:** Compute g = ∇f(x).
2. **Update:** x_next = x - α * g (for a step size α > 0).
3. Repeat or stop (e.g. when ‖g‖ is small).

So any autodiff feature would need to supply g in step 1; the rest is already expressible with recurrence and existing stdlib (e.g. gradient_descent_step).

---

## Op-level differentiation rules

Concrete VJPs and gradient shapes for each op (elementwise, matmul, conv, reductions, softmax, ReLU, etc.) are specified in **[AUTODIFF_OPS.md](AUTODIFF_OPS.md)**. That document is the reference for implementing a backward pass over the IR.

---

## Summary

- **Today:** No autodiff; explicit gradients only. See [UNSUPPORTED §12](UNSUPPORTED.md#12-no-automatic-differentiation-autodiff).
- **If revisited:** Three syntax choices: **grad[x](expr)** (recommended), **@grad(expr)(x)**, or **grad(expr)(x)**. Same shapes, optional generated backward as explicit code. Design and constraints: this doc; op rules: [AUTODIFF_OPS.md](AUTODIFF_OPS.md).
