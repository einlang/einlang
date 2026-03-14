# Autodiff design (draft)

**Status:** Design only. No autodiff in the language today; see [UNSUPPORTED §12](UNSUPPORTED.md#12-no-automatic-differentiation-autodiff).

**Summary:** One gradient primitive: **@** produces a differential (e.g. `let df = @f; let dx = @x;`). The **only** derivative form is the quotient **df / dx**. No inline `@loss/@w`, no `@x(expr)`, no `@(loss)(w, b)`. For multiple parameters, write separate bindings and quotients. Same semantics and VJPs for ML and scientific computing. Compiler generates backward pass from IR; no user-visible tape.

---

## 1. Goals and non-goals

**Goals**

- **Math-first syntax:** Gradient in code reads like the blackboard (∂f/∂x in ASCII).
- **Minimal surface:** One (or very few) primitives; no tapes, dual types, or extra machinery in user-facing syntax.
- **Same shapes:** Gradient w.r.t. a variable has the same shape as that variable (compile-time check).
- **One mechanism for ML and science:** Training steps, calibration, sensitivity, and adjoints use the same @ and same VJP table.

**Non-goals**

- Forward-mode as first citizen (reverse-mode suffices for scalar loss and adjoints).
- User-controlled tape or graph; compiler builds backward from IR.
- Higher-order derivatives in v1 (could be added later if needed).

---

## 2. Design constraints

- **Only form df / dx:** Derivatives are written only as the quotient of two @-sourced values (e.g. `d_loss / d_w`). The `/` operator when both operands are differentials means “partial of numerator w.r.t. denominator” (or we use a distinct operator to avoid overloading numeric division).
- **Explicit generated code (optional):** Backward pass can be emitted as normal IR or readable Einlang-like code for inspection.
- **Differentiability:** Gradients are defined only where the program is differentiable w.r.t. the variable (no kinks/jumps at the point of use); subgradients (e.g. ReLU at 0) are defined by convention and documented.

---

## 3. Proposed syntax: only df / dx

**@** produces a differential (same shape as the expression). The **only** way to express a derivative is the quotient **df / dx** of two such values.

**Allowed:**

- `let df = @f;` — bind the differential of `f` (gradient of ambient scalar w.r.t. `f`).
- `let dx = @x;` — bind the differential of `x`.
- `let g = df / dx;` — derivative of (what df refers to) w.r.t. (what dx refers to). `/` when both operands are @-sourced has this meaning (or a dedicated operator if we avoid overloading division).

**Not in scope:**

- No `@x(expr)`, no `@(expr)(x)`.
- No inline `@loss / @w`; use `let d_loss = @loss; let d_w = @w; let dL_dw = d_loss / d_w;`.
- No `@(loss)(w, b)` or multi-gradient form; for several parameters, write separate `let d_w = @w;`, `let d_b = @b;` and separate quotients with `d_loss`.

**Grammar:** Existing `@` is `name @ pattern` (binding). Prefix `@ name` (e.g. `@loss`, `@w`) is distinct; no parse conflict.

**Minimal example**

```text
let (x, y) = (1, 2);
let f = x + y;
let df = @f;
let dx = @x;
let df_dx = df / dx;   // 1
```

---

## 4. Semantics

**Ambient output (reverse-mode):** There is an ambient scalar “output” (e.g. loss L). **@expr** means “gradient of that output w.r.t. expr”. So **@f** is ∂L/∂f (same shape as f), **@x** is ∂L/∂x (same shape as x). The quotient **df / dx** (when both are @-sourced) is interpreted as ∂f/∂x (e.g. scalar ratio when both are scalars).

**Shape rule:** For any expression `x`, the value of `@x` has the same shape (and dtype) as `x`.

**Internals of df / dx:** In reverse-mode, **df** and **dx** are tensors (or scalars) produced by the generated backward pass. No user-visible tape; they are normal values of the same shape as f and x. Division of two @-sourced values is a special meaning of `/` (partial of numerator w.r.t. denominator). Alternative: introduce a distinct operator (e.g. `df // dx`) to avoid overloading `/`.

---

## 5. Use in ML (training step)

**Forward and loss (unchanged):**

```text
let logits[i] = sum[j](x[j] * w[i, j]) + b[i];
let pred = relu(logits);
let loss = sum(i)( (pred[i] - target[i])^2 );
```

**Gradients and update:**

```text
let dL_dw = @loss / @w;
let dL_db = @loss / @b;
let w_next = w - alpha * dL_dw;
let b_next = b - alpha * dL_db;
```

Training loop = repeat over batches: forward → loss → @loss/@w, @loss/@b → update. No separate “backward” call; the compiler generates the backward from the use of @-expressions.

---

## 6. Use in scientific computing

**Calibration:** Forward model + scalar loss; gradient w.r.t. parameters for optimizer.

```text
let u[i] = u0 * exp(-k * t[i]);
let loss = sum(i)( (u[i] - y[i])^2 );
let d_loss = @loss;
let d_k    = @k;
let dL_dk  = d_loss / d_k;
// k_next = k - alpha * dL_dk or pass to numerics::optim
```

**Sensitivity:** Gradient of an output w.r.t. parameters.

```text
let y[i] = sum[j](X[i, j] * beta[j]);
let d_y = @y;
let d_beta = @beta;
let dL_dbeta = d_y / d_beta;
```

**Inverse / adjoint:** Forward y = C*x (e.g. conv). To get Cᵀv, form scalar L = dot(v, y) and take gradient w.r.t. input.

```text
let y = conv(x, w);
let L = sum(...)( v * y );
let d_L = @L;
let d_x = @x;
let adj_x = d_L / d_x;   // shape of x; use as C^T v in CG
```

Same VJP table as ML; conv’s backward (conv_transpose) is the adjoint.

---

## 7. Implementation approach

- **Reverse-mode:** For each op in the forward pass, the compiler (or a dedicated pass) emits backward IR that implements the VJP: given gradient at output, compute and accumulate gradients at inputs. No runtime tape; backward is generated code.
- **IR:** Build on existing IR (expression tree, lowered Einstein, recurrence). Differentiation pass walks IR and emits derivative IR. Recurrence implies a backward-time recurrence that applies VJPs step by step.
- **VJPs:** Op-level rules (elementwise, matmul, conv, reductions, softmax, ReLU, etc.) and gradient shapes are defined in **[AUTODIFF_OPS.md](AUTODIFF_OPS.md)**. Single source of truth for backward formulas.
- **Hard rules:** Differentiability at use site; shape(∇f(x)) = shape(x); sum/product/chain rule and scalar multiple hold; no NaNs/Infs from the gradient formula.

---

## 8. Open questions

- **Ambient output:** How is the “scalar output” (loss) defined when there are several @-expressions? Option: last scalar in scope, or an explicit `loss = ...` that must dominate the block.
- **Quotient operator:** Should `df / dx` when both are @-sourced overload `/` or use a distinct operator (e.g. `//`) to avoid confusion with numeric division?
- **Recurrence and control flow:** Precise rules for differentiating through recurrence and conditionals; subgradients at branches (e.g. ReLU at 0).

---

## 9. References

- **[AUTODIFF_FUTURE.md](AUTODIFF_FUTURE.md)** — Policy, syntax alternatives (grad[x](expr) vs @), scorecard, internals, longer examples.
- **[AUTODIFF_OPS.md](AUTODIFF_OPS.md)** — Op-level VJPs and gradient shapes (single reference for backward pass).
- **[UNSUPPORTED.md §12](UNSUPPORTED.md#12-no-automatic-differentiation-autodiff)** — Current “no autodiff” policy and workarounds.
