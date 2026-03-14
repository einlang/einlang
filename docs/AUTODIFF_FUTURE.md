
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

## Syntax: four choices (ASCII, no operator reuse)

We do **not** use `d(expr)/d(x)` because `/` is already division. Four options:

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

### Option D: **@ as partial / d** (recommended if using a sigil)

Use `@` to mean "∂" (partial) or "d" (derivative), so the syntax reads like the blackboard form ∂(expr)/∂x.

- **Form 1 — @x(expr):** `@` + variable(s) + `(expr)`. Reads as "∂/∂x (expr)" or "d(expr)/dx". Gradient of expr w.r.t. x; same shape as x.
- **Form 2 — @(expr)(x):** `@` + `(expr)` + `(x)`. Same meaning; expression first, then "w.r.t. x".
- **Form 3 — @f / @x:** Fraction-style: `@` + numerator + `/` + `@` + denominator. Reads exactly as **∂f/∂x**. Example: `@loss / @x`, `@(a * b + c) / @w`. The parser treats `@ expr / @ var` as a single partial-derivative expression (not as division); `/` here is part of the derivative notation, so no operator reuse with numeric division. Numerator can be any expression (use `@(expr) / @x` when not a single name).
- **Form 4 — First-class @ bindings:** Allow binding `@`-expressions so the quotient can be written with names, matching the usual calculus style df, dx:
  - `let df = @f; let dx = @x;` — then `df` and `dx` are differential/derivative values (same shape as `f` and `x`; semantics: e.g. "gradient of current scalar w.r.t. f" for df and "w.r.t. x" for dx, or differentials in a forward sense).
  - The derivative is then written as the quotient: `df / dx` (or a dedicated form like `df // dx` if we want to avoid overloading `/` when operands are differential-typed). So you can write:
    - `let df = @f; let dx = @x; let df_dx = df / dx;` — compiler treats division of two @-sourced values as ∂f/∂x.
  - **Design constraint:** If we allow this, we must define (a) what type or tag "differential" has, (b) that `/` between two differentials means "partial of numerator w.r.t. denominator" (not numeric division), and (c) how multiple w.r.t. variables are named (e.g. `@x`, `@w`).
- **Multiple variables:** `@x,w(expr)` or `@(expr)(x, w)` or repeated fraction: `@f/@x`, `@f/@w`. Same shape rules per variable.
- **Grammar note:** Einlang already uses `@` in binding patterns (`name @ pattern`). Prefix derivative form `@ x ( expr )` or `@ expr / @ x` is distinct (starts with `@`); no parse conflict.

- **Minimal example (valid Einlang today; @ forms hypothetical):**
  ```text
  let (x, y) = (1, 2);
  let f = x + y;
  ```
  With autodiff: `@f / @x` and `@f / @y` would both be 1 (scalar). First-class form: `let df = @f; let dx = @x; let df_dx = df / dx;` → `df_dx == 1`.

---

### Scorecard (each criterion 1–5, total out of 25)

| Criterion | A: grad[x](expr) | B: @grad(expr)(x) | C: grad(expr)(x) | D: @x(expr) |
|-----------|------------------|-------------------|------------------|-------------|
| **Math intuitive** (reads like ∂/∂x or ∇_x) | 5 — variable on operator | 3 — variable in second group | 3 — same as B | 5 — @ = ∂/∂, @x(expr) = ∂(expr)/∂x |
| **No operator reuse** | 5 | 5 | 5 | 5 — @ dedicated to partial/d |
| **Einlang consistency** (bracket = structure vs new sigil) | 5 — bracket = w.r.t. | 3 — @ is new | 4 — no new sigil; grad reserved | 4 — @ is new but reads as math |
| **Parseability** (unambiguous; special form clear?) | 5 — [ ] after grad is special | 5 — @ marks special form | 4 — grad(expr)(x) looks like curried call | 5 — @ prefix unambiguously marks derivative |
| **Readability** (short, clear for multiple vars) | 5 — grad[x](loss), grad[w](loss) | 4 — @grad(loss)(x) | 4 — grad(loss)(x), grad(loss)(w) | 5 — @x(loss), @w(loss); @x,w(loss) |
| **Total** | **25/25** | **20/25** | **20/25** | **24/25** |

**Recommendation:** **Option A: grad[x](expr)** remains best if we avoid a new sigil: bracket = "w.r.t.", no new token. **Option D** (with `@`) is best if we adopt a sigil: **@f / @x** (Form 3) is the most blackboard-like (literal ∂f/∂x in ASCII); **@x(expr)** (Form 1) is more compact for multiple vars (`@x(loss)`, `@w(loss)`). Form 2 `@(expr)(x)` is equivalent for expression-first preference.

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

## Internals of @-values (df, dx)

What is **df** (or **dx**) under the hood when we write `let df = @f;`? Possible representations:

- **Reverse-mode (recommended for scalar loss):** There is an ambient scalar “output” (e.g. a loss L). **df** is the gradient of that output w.r.t. **f**: same shape and dtype as **f**, holding ∂L/∂f. So internally **df** is just a tensor (or scalar) of the same shape as **f**, produced by the generated backward pass. **dx** is the gradient w.r.t. **x**, same shape as **x**. The quotient **df / dx** is then interpreted as “∂f/∂x” (e.g. scalar ratio when both are scalars).
- **Forward-mode (tangents):** Values can carry a tangent. **df** could be the tangent of **f** (same shape as **f**). **dx** the tangent of **x**. Then **df/dx** is the ratio of tangents (derivative of f w.r.t. x along that direction). Internally **df** might be stored as a dual (primal, tangent) or the tangent alone when only derivatives are needed.
- **Symbolic / deferred:** **df** and **dx** could be handles (e.g. graph nodes or named gradients) that do not hold numeric arrays until the program actually uses them (e.g. in **df / dx** or in an update). The compiler then generates the backward pass and fills the gradient buffers when the value is demanded. So “internal” is a reference to a gradient slot or a lazy computation, not necessarily a concrete tensor at the moment of `let df = @f;`.

For the minimal example `let (x, y) = (1, 2); let f = x + y; let df = @f; let dx = @x;`: if the ambient output is **f**, then **df** = ∂f/∂f = 1 (scalar), **dx** = ∂f/∂x = 1 (scalar). So internally **df** and **dx** are both the scalar 1; **df / dx** = 1/1 = 1.

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

## How to use it in ML (with @ syntax)

Typical training: **forward → loss → gradients w.r.t. parameters → update**. With the @ notation, the gradient step stays in Einlang and reads like the math.

**1. Forward and loss (unchanged):**

```text
// Weights and batch (shapes as in your IR; example: one linear layer)
let (w, b) = (weights, bias);   // e.g. w[i,j], b[i]
let logits[i] = sum[j](x[j] * w[i, j]) + b[i];
let pred = relu(logits);
let loss = sum(i)( (pred[i] - target[i])^2 );   // scalar
```

**2. Gradients (autodiff with @):**

Using the fraction form or first-class bindings:

```text
// Option A: inline @loss / @var
let dL_dw = @loss / @w;
let dL_db = @loss / @b;

// Option B: first-class differentials, then quotient
let d_loss = @loss;
let d_w   = @w;
let dL_dw = d_loss / d_w;
let dL_db = d_loss / @b;
```

**3. Update (existing style):**

```text
let alpha = 0.01;
let w_next = w - alpha * dL_dw;
let b_next = b - alpha * dL_db;
```

**4. Training loop:**

Wrap the block above in a recurrence or loop over batches; each iteration: forward → loss → `@loss / @w`, `@loss / @b` → update `w`, `b`. No separate “backward” call: the compiler generates the backward pass from the use of `@loss / @w` and `@loss / @b` (or from first-class `@loss`, `@w` and the quotient). Same VJP rules (linear, relu, sum, etc.) as in [AUTODIFF_OPS.md](AUTODIFF_OPS.md); the @ syntax only exposes gradients as expressions.

---

## How to use it in scientific computing (with @ syntax)

Same @ notation; the “output” is a scalar loss (calibration) or a quantity of interest (sensitivity), and you take gradients w.r.t. parameters or inputs.

**1. Calibration (fit params to data):**

Forward model + scalar loss over data; then gradient w.r.t. parameters to drive an optimizer (e.g. gradient descent or L-BFGS).

```text
// Example: decay u(t; k) = u0 * exp(-k * t), fit k to data (t_i, y_i)
let u[i] = u0 * exp(-k * t[i]);
let loss = sum(i)( (u[i] - y[i])^2 );

let dL_dk = @loss / @k;   // sensitivity of loss to k; same shape as k (scalar)
// Then: k_next = k - alpha * dL_dk (or pass dL_dk to numerics::optim)
```

**2. Sensitivity (∂y / ∂params):**

You care about how an output (vector or scalar) changes w.r.t. parameters. Treat that output as the “loss” and take @output / @param.

```text
// Example: y = X * beta (least-squares); how does prediction change w.r.t. beta?
let y[i] = sum[j](X[i, j] * beta[j]);
let dL_dbeta = @y / @beta;   // or @(sum(i)(y[i]))/@beta for a scalar proxy
// Shape of dL_dbeta matches beta; use for uncertainty or optimizer.
```

**3. Inverse / adjoint (e.g. iterative solver):**

Forward: `y = C * x` (e.g. conv). Iterative solvers need the adjoint \(C^\top v\). The backward of conv is conv_transpose; so “gradient of (vᵀy) w.r.t. x” gives \(C^\top v\). With @ you can express that as a gradient of a scalar w.r.t. the input.

```text
// Forward: y = conv(x, w); scalar L = dot(v, y) (for a given v)
// Then @L / @x is the adjoint applied to v (same as conv_transpose in backward)
let y = conv(x, w);
let L = sum(...)( v * y );   // scalar
let adj_x = @L / @x;         // shape of x; use as C^T v in CG or similar
```

So in scientific code you use the same pattern: **forward → scalar (loss or L) → @L / @params or @L / @x**. Calibration and sensitivity use @loss/@param; inverse/adjoint use a dummy scalar (e.g. dot with a vector) and @L/@x. Same VJP table as ML; no separate “science” vs “ML” autodiff.

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
- **If revisited:** Four syntax choices: **grad[x](expr)** (no new sigil), **@grad(expr)(x)**, **grad(expr)(x)**, or **@x(expr)** / **@(expr)(x)** (@ as partial/d). Same shapes, optional generated backward as explicit code. Design and constraints: this doc; op rules: [AUTODIFF_OPS.md](AUTODIFF_OPS.md).
- **Consolidated design (draft):** [AUTODIFF_DESIGN.md](AUTODIFF_DESIGN.md) — single doc with proposed @ syntax, semantics, ML/science usage, implementation approach, and open questions.
