# Autodiff design

**Status:** Implemented. The compiler supports `@expr` (differential) and `df / dx` (derivative quotient). See [AUTODIFF_IMPLEMENTATION.md](AUTODIFF_IMPLEMENTATION.md), [AUTODIFF_PIPELINE.md](AUTODIFF_PIPELINE.md), [AUTODIFF_OPS.md](AUTODIFF_OPS.md), and examples [autodiff_small.ein](https://github.com/einlang/einlang/blob/main/examples/autodiff_small.ein), [autodiff_matmul.ein](https://github.com/einlang/einlang/blob/main/examples/autodiff_matmul.ein).

**Summary:** One primitive: **@** produces a **differential** (e.g. `let df = @f; let dx = @x;`). Differentials are **symbolic** — they have no numeric value by themselves. The **only** way to get a numeric value is the quotient **df / dx** (the **derivative**). We use **@fn** to define a function’s diff from the argument diff(s): same args as the primal, **@param** in the body (e.g. @fn exp(x) { exp(x) * @x }); **derivative** only for the numeric quotient @y/@x. No inline `@loss/@w`, no `@x(expr)`, no `@(loss)(w, b)`. For multiple parameters, write separate bindings and quotients. Same semantics and chain rules for ML and scientific computing. **Implementation:** we implement as **forward diff** (propagate differentials / coefficients), not backward (no pullbacks or tape).

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

**@** produces a **differential** (same shape as the expression). Differentials are **symbolic**: `@x` and `@y` do not have a numeric value by themselves. The **only** way to obtain a numeric value is the **quotient** **df / dx**: that is the derivative and has a numeric value.

**Allowed:**

- `let df = @f;` — bind the differential of `f` (symbolic; no numeric value).
- `let dx = @x;` — bind the differential of `x` (symbolic; no numeric value).
- `let g = df / dx;` — **derivative** of (what df refers to) w.r.t. (what dx refers to). Only this quotient has a numeric value. `/` when both operands are @-sourced has this meaning (or a dedicated operator if we avoid overloading division).
- Inline quotient: `@loss / @w` is allowed (derivative of the quantity that `@loss` refers to w.r.t. the quantity that `@w` refers to).
- For several parameters: separate `let d_w = @w;`, `let d_b = @b;` and separate quotients with `d_loss` (e.g. `d_loss / d_w`, `d_loss / d_b`).

**Not in scope:**

- No `@x(expr)`, no `@(expr)(x)`.
- No `@(loss)(w, b)` or single-call multi-gradient form; use separate `@w`, `@b` and separate quotients as above.

**Grammar:** Existing `@` is `name @ pattern` (binding). Prefix `@ name` (e.g. `@loss`, `@w`) is distinct; no parse conflict.

**Minimal example**

```text
let (x, y) = (1, 2);
let f = x + y;
let df = @f;
let dx = @x;
let df_dx = df / dx;   // 1
```

**Small example (only @ variables involved)**

```text
let x = 1.0;
let y = 2.0;
let z = x + y;        // z = 3.0
let dz_dx = @z / @x;  // derivative of z w.r.t. x = 1 (only @z and @x are used)
print(dz_dx);         // 1
```

Here we only use `@z` and `@x`, so the compiler seeds and exposes differentials for `z` and `x` only. `y` is not a target; it may appear internally in the chain rule (dz = dx + dy) but we do not need to handle `@y`.

**Compile-time sugar:** We treat **@z / @x** as syntactic sugar and expand it at compile time. The compiler does not emit “evaluate @z, evaluate @x, then divide”; it emits a single **derivative-quotient** concept (e.g. a dedicated IR node or a marked division) meaning “derivative of (quantity for z) w.r.t. (quantity for x)”. The backend then runs the diff block once and computes that one numeric value (e.g. d_z / d_x from the buffers). So @z and @x are never materialized as first-class values at runtime; only the derivative is.

---

## 4. Semantics (math-first: differentials)

**We use differential semantics.** Notation in code = notation on the blackboard.

**@expr** = the **differential** of `expr`: same shape as `expr`. Differentials are **symbolic** — they do not have a numeric value. Only the **quotient** **df / dx** has a numeric value (the derivative). The compiler uses the symbolic identities to compute that derivative.

Differentials **combine** like in calculus (for the purpose of deriving the quotient):

- If `z = x + y` then **dz = dx + dy** — so `@z` is symbolically `@x + @y`.
- If `z = x * y` then **dz = y dx + x dy** — so `@z` is expressed in terms of `@x`, `@y` and the forward values.

Every expression has its own differential; the compiler propagates them so these identities hold. **Internally we use forward diff:** we build and propagate differential expressions (dy = … dx + … dy), not backward (no pullbacks, no tape). Derivative quotients @y/@x are computed from that forward-diff representation. There is **no “ambient output”** and no single scalar L; differentials are first-class **symbols**.

**df / dx:** When both operands are differentials, the quotient is the **derivative** of (what df is the differential of) w.r.t. (what dx is the differential of). **Only this quotient has a numeric value** at runtime.

**Shape rule:** For any expression `x`, `@x` has the same shape (and dtype) as `x` (as a type; the value is symbolic until used in a quotient).

---

## 5. Use in ML (training step)

**Forward and loss (unchanged):**

```text
let logits[i] = sum[j](x[j] * w[i, j]) + b[i];
let pred = relu(logits);
let loss = sum[i]( (pred[i] - target[i])**2 );
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

**Utilizing derivatives (not just printing):** The quotient `@y / @x` is a normal numeric value. Use it in expressions: gradient steps (`x_next = x - alpha * (@loss / @x)`), first-order sensitivity corrections (`u_corrected = u + (du_dk) * (k_alt - k)`), or refinement steps in calibration (`k_refined = k - step * (@sse / @k)`). See `examples/run_numerics.ein` (one gradient step), `examples/ode/ode_suite.ein` and `examples/pde_1d/heat_1d.ein` (sensitivity correction), and `examples/applications/decay_calibration.ein` (gradient refinement of k).

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

## 7. Diff rules: function-like form (define func diff from argu diff using @)

**Terminology:** We use **@** to define the **function’s diff** (output differential) from the **argument diff(s)** (input differentials). The rule is written as a **function**: it takes the arguments and their differentials (@arg) and returns the expression for the output differential. We use **derivative** only for the *numeric value* of the quotient **@y / @x**.

For opaque or builtin functions (e.g. exp, tanh, Python imports), the user (or stdlib) defines a **diff function**: **same parameter list as the primal** (no @ in the args). In the **body**, **@param_name** refers to the differential of that parameter. The body expression gives dy = (∂f/∂a) da + (∂f/∂b) db + … . The compiler uses this when propagating differentials; it never differentiates the function body for these calls.

**Math:**

- **Single argument:** y = f(x) → **dy = (∂f/∂x) dx**. The diff function has one param (x); in the body, @x is the diff of x. Returns expr × @x with expr = ∂f/∂x. Reverse: d_x += d_out × expr.
- **Multiple arguments:** y = f(a,b) → **dy = (∂f/∂a) da + (∂f/∂b) db**. The diff function has params (a, b); in the body, @a and @b are the diffs of a and b. Returns (expr_a × @a + expr_b × @b).

**Proposed syntax (func-like, same args as primal; @param in body):**

```text
@fn exp(x) {
    exp(x) * @x
}

@fn tanh(x) {
    (1.0 - tanh(x) * tanh(x)) * @x
}

@fn pow(a, b) {
    (b * pow(a, b - 1.0)) * @a + (pow(a, b) * ln(a)) * @b
}
```

Argument list: **only primal parameters** (same as the original function). In the body, **@param** is the differential of that parameter. The body is a single expression (or block with final expr) that computes the output differential. Rules are keyed by function (DefId); the autodiff pass looks up the **@fn** for the callee and uses it in the chain rule.

---

## 8. Comparison with Julia (ChainRules)

| Aspect | Einlang @fn | Julia rrule |
|--------|-------------|-------------|
| **What you write** | Forward diff: output diff as expression in primal args + @params. `@fn exp(x) { exp(x) * @x }` | Pullback: function that takes output tangent (ȳ) and returns input tangents. `function rrule(::typeof(exp), x); y = exp(x); exp_pullback(ȳ) = (NoTangent(), ȳ * y); return y, exp_pullback; end` |
| **Direction** | **Forward:** dy = (∂f/∂x) dx. You give the coefficient(s) of dx (and db, …). | **Reverse:** Pullback(ȳ) → (ā, …). You give how to map output tangent back to input tangents. |
| **Signature** | Same args as primal; @param in body = diff of that param. | Primal: `(::typeof(f), args...)`. Pullback: one argument (output tangent); returns tuple of input tangents. |
| **Keying** | By function (DefId). One @fn per primal. | By function (`typeof(f)`). One rrule per function. |
| **Multi-arg** | One body: sum of (expr_i * @arg_i). Natural for forward diff. | Pullback returns one tangent per argument (e.g. `(NoTangent(), ā, b̄)`). Natural for reverse. |
| **C / opaque** | Same: register a rule for the opaque call; compiler uses it instead of differentiating. | Same: define rrule for the wrapper; AD uses it. |
| **Math on the page** | dy = … × @x + … × @y. Reads like the differential equation. | Pullback: ȳ → (ā, b̄). Reads like “how to backprop”. |

**Summary:** Julia’s rrule is **reverse-mode native** (pullback from output tangent to input tangents). Einlang’s @fn is **forward-diff native** (output diff as a linear combination of input diffs). The compiler can derive the reverse rule from the forward one (contribution to d_x is d_out × coefficient of @x), so both support reverse-mode AD; the difference is whether the **user** writes the forward equation (Einlang) or the backward map (Julia). Einlang’s form matches the usual calculus notation (dy = … dx) and keeps the same parameter list as the primal.

### 8.1 Learning from Julia: reduction autodiff

Julia’s **ChainRules.jl** defines differentiation for reductions in `mapreduce.jl`. Aligning with it keeps semantics and shapes consistent:

- **Sum**  
  - **Julia:** `rrule(sum, x; dims=dims)` returns primal `y = sum(x; dims=dims)` and a pullback that, given output tangent `dy`, fills input tangent by *broadcasting* `dy` to the shape of `x`: `_unsum(x, dy, dims) = broadcast(last∘tuple, x, dy)`. So ∂(sum)/∂x has the same shape as `x`; each element gets the same (or broadcast) sensitivity.  
  - **Einlang:** Same idea: ∂y/∂x_i = 1 (broadcast); gradient w.r.t. `x` has shape of `x`. We implement this as vectorized sum of differentials (no scalar loop).

- **Max / argmax (select-at-argmax)**  
  - **Julia:** The gradient of `maximum(x)` w.r.t. `x` is a **subgradient**: 1 at the argmax index, 0 elsewhere (or normalized if there are ties). So the pullback takes a scalar `dy` and returns a vector `dx` with `dx[argmax] = dy` and `dx[i] = 0` for i ≠ argmax. Shape of `dx` = shape of `x`; the result is **squeezed** (one gradient per parallel position, not a full Jacobian over the reduction dimension).  
  - **Einlang:** We implement “select at argmax” as: primal = value at argmax; differential = **d_body at argmax(primal)**. The gradient w.r.t. the reduction input has shape **parallel_shape** (one selected value per parallel index), not parallel_shape × reduction_size. So the backend should return the **squeezed** result (shape `parallel_shape`) from select-at-argmax/argmin; a full Jacobian is not required for standard reverse-mode and would complicate downstream assignment.

- **Prod**  
  - **Julia (ChainRules, mapreduce.jl):** ∂(prod_i x_i)/∂x_k = prod_{j≠k} x_j = (prod x)/x_k. Gradient has **shape of x** (same as sum). Special handling for **zeros**: (1) **no zero** → gradient at k is (prod x)/x_k; (2) **exactly one zero** at index k₀ → gradient is zero except at k₀, where it is prod_{j≠k₀} x_j (the limit); (3) **multiple zeros** → gradient is zero everywhere. So the pullback returns a full-shaped gradient array, not squeezed.  
  - **Einlang:** We use (prod body) × sum_i (d_body_i / body_i), which equals (prod x)/x_k when body = x and body_i ≠ 0; see [AUTODIFF_OPS.md](AUTODIFF_OPS.md) §6. Gradient has shape of the reduction body (same as input to the reduction). **Zero handling:** current formula is valid when body_i ≠ 0; matching Julia’s one-zero / multi-zero behaviour would require a separate branch (e.g. count zeros, then fill gradient accordingly).

- **General sum(f, xs)**  
  - **Julia:** Either a fast path when `f` only depends on the input (no need to store `f(x_i)`), or stores pullbacks for each `f(x_i)` and in the pullback combines them with the incoming `dy`.  
  - **Einlang:** Reductions are lowered to a single body over indices; we differentiate the body and then apply the reduction’s VJP (sum → broadcast; max → select-at-argmax).

**Takeaway:** For max/argmax, the gradient w.r.t. the reduced array has the same shape as the **output** of the reduction (e.g. `parallel_shape`), not an expanded “full Jacobian”. Returning the squeezed result keeps backend and autodiff semantics aligned with Julia and avoids unnecessary large arrays and slow paths.

### 8.2 Design (Julia-aligned): gradient shape = shape of x

In Julia ChainRules, **the pullback return shape matches the input shape** of the primal: if `x` has shape `(m,)`, the gradient `x̄` has shape `(m,)`. So for sum, max, min, prod the gradient is always **shape of x** — no special “squeezed” array. Our design matches that:

| Reduction | Gradient semantics | Stored gradient shape (Julia and Einlang) |
|-----------|--------------------|-------------------------------------------|
| **Sum**   | ∂y/∂x_i = 1 (broadcast) | shape of x |
| **Max**   | Subgradient: value at argmax, 0 elsewhere | shape of x |
| **Min**   | Subgradient: value at argmin, 0 elsewhere | shape of x |
| **Prod**  | ∂y/∂x_k = (prod x)/x_k | shape of x |

**Backend does not treat autodiff differently.** The backend just assigns clause result to the output buffer. So we need: (1) **Compiler:** the gradient variable has shape of x (quotient shape = denominator in shape analysis; lowering allocates that shape). (2) **Runtime:** select-at-argmax returns a full-shaped array (value at argmax, 0 elsewhere), same shape as the reduction input. Then result.shape == output.shape and the existing `output[:] = result` path applies. No diff-specific backend logic. If the scalar path is used and the body returns an array with shape == output.shape, assigning that to the whole output (instead of a single cell) is a general rule for any such clause, not autodiff-specific.

---

## 9. Plans

- **print(@y) symbolic output:** Use the IR node's `str` (or a dedicated IR visitor that formats expressions for diff source) instead of a custom pretty-printer in the autodiff pass, so one representation is maintained in `ir/nodes` or serialization.

---

## 10. References

- **[AUTODIFF_OPS.md](AUTODIFF_OPS.md)** — Op-level VJPs and differential shapes (single reference for backward pass).
- **[AUTODIFF_IMPLEMENTATION.md](AUTODIFF_IMPLEMENTATION.md)** — Implementation blueprint.
- **[UNSUPPORTED.md §12](UNSUPPORTED.md#12-no-automatic-differentiation-autodiff)** — Current “no autodiff” policy and workarounds.
