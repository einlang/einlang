# Math intuition

**Einlang is built for math intuitiveness:** you write the same notation you use in equations and on the whiteboard. Indices, sums, index relations, and derivatives map directly to code—no string subscripts, no mental translation. If you think in equations and indices, Einlang matches. This page shows how common math notation maps to code. Full semantics: [Language reference](https://github.com/einlang/einlang/blob/main/docs/reference.md).

---

## Sums and tensor expressions


| In math                                      | In Einlang                                 | Reference                                                  |
| -------------------------------------------- | ------------------------------------------ | ---------------------------------------------------------- |
| C_{ij} = Σ_k A_{ik} B_{kj} (matrix multiply) | `let C[i, j] = sum[k](A[i, k] * B[k, j]);` | [Einstein notation](reference.md#einstein-notation)        |
| s_i = Σ_j M_{ij} (row sums)                  | `let s[i] = sum[j](matrix[i, j]);`         | [Reductions](reference.md#einstein-notation)               |
| ‖A‖*F = √(Σ*{ij} A_{ij}²) (Frobenius norm)   | `sqrt(sum[i, j](A[i, j] * A[i, j]))`       | [Reductions](reference.md#einstein-notation)               |
| y_{ij} = 2 x_{ij} (element-wise)             | `let y[i, j] = 2.0 * x[i, j];`             | [Rectangular declarations](reference.md#einstein-notation) |


Indices are **names**, not strings—so the code stays readable and math-aligned. The compiler infers ranges from array shapes and checks that dimensions match.

---

## Index relations (e.g. convolution)


| In math                                                     | In Einlang                                                                                                  | Reference                                                     |
| ----------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------- |
| Output index o, kernel k, input i = o + k                   | `where ih = oh + kh, iw = ow + kw`                                                                          | [Where clauses — Index remapping](reference.md#where-clauses) |
| 2D conv: sum over input channels and kernel, with i = o + k | `let out[b,oc,oh,ow] = sum[ic,kh,kw](input[b,ic,ih,iw] * w[oc,ic,kh,kw]) where ih = oh + kh, iw = ow + kw;` | [Where clauses](reference.md#where-clauses)                   |


The **where** clause is the same index algebra you’d write under the Σ; the compiler checks bounds.

---

## Conditioned sums and guards


| In math                           | In Einlang                                     | Reference                                                    |
| --------------------------------- | ---------------------------------------------- | ------------------------------------------------------------ |
| Σ_i x_i where x_i > 0             | `sum[i](data[i]) where data[i] > 0`            | [Where clauses — Boolean guards](reference.md#where-clauses) |
| M_{ij} for i ≤ j (upper triangle) | `let upper[i, j] = matrix[i, j] where i <= j;` | [Boolean guards](reference.md#where-clauses)                 |


Guards in the **where** clause act as filters; the compiler still sees all indices.

---

## Recurrence equations


| In math                                                     | In Einlang                                                                                                                                                                | Reference                                                 |
| ----------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------- |
| f_0 = 0, f_1 = 1, f_n = f_{n-1} + f_{n-2}                   | `let fib[0] = 0; let fib[1] = 1; let fib[n in 2..N] = fib[n-1] + fib[n-2];`                                                                                               | [Recurrence relations](reference.md#recurrence-relations) |
| h_0 = init, h_{t+1} = φ(h_t + x_t) (RNN step)               | Use backward reference in time: `let h[0, i in 0..H] = init[i]; let h[t in 1..T, i in 0..H] = tanh(h[t-1, i] + x[t-1, i]);` — body reads `h[t-1, i]` only (math-aligned). | [Recurrence relations](reference.md#recurrence-relations) |
| h_{t,i,j} from h_{t-1} and spatial neighbors (e.g. 2D step) | Backward only in every dim: `let h[t in 1..T, i in 1..N-1, j in 1..N-1] = f(h[t-1,i,j], h[t,i-1,j], h[t,i,j-1]);` — no future values (no `h[t,i+1,j]` or `h[t,i,j+1]`).   | [Recurrence relations](reference.md#recurrence-relations) |


Base cases and recursive case in one place; the compiler handles evaluation order.

---

## Derivatives and autodiff


| In math                       | In Einlang                                                  | Reference                                                                                                                                                                               |
| ----------------------------- | ----------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| ∂z/∂x, gradient of z w.r.t. x | `let dz_dx = @z / @x;`                                      | [Automatic differentiation](reference.md#automatic-differentiation)                                                                                                                     |
| Chain rule (compiler-derived) | Write `z = f(x)`; use `@z / @x` — no hand-written gradients | [AUTODIFF_DESIGN](https://github.com/einlang/einlang/blob/main/docs/AUTODIFF_DESIGN.md), [autodiff_small.ein](https://github.com/einlang/einlang/blob/main/examples/autodiff_small.ein) |


The compiler supports **built-in automatic differentiation**: derivatives and gradients from `@expr` and `@a / @b`.

---

## Derivatives of ML and math ops

Mathematical derivatives (∂y/∂x) for common ops. Chain rule: if z = f(y) and y = g(x), then ∂z/∂x = (∂z/∂y)(∂y/∂x).

### Unary and binary (IR-level)

| Op | y = f(x) or expression | ∂y/∂x (or partials) |
|----|------------------------|----------------------|
| **+** | a + b | ∂/∂a = 1, ∂/∂b = 1 |
| **−** | a − b | ∂/∂a = 1, ∂/∂b = −1 |
| **×** | a × b | ∂/∂a = b, ∂/∂b = a |
| **/** | a / b | ∂/∂a = 1/b, ∂/∂b = −a/b² |
| **^** (pow) | a^b | ∂/∂a = b·a^(b−1), ∂/∂b = a^b·ln(a) |
| **neg** | −x | −1 |
| **pos** | +x | 1 |

These are implemented in the autodiff pass for `BinaryOpIR` and `UnaryOpIR` (only NEG and POS; other unary ops are not handled at IR level).

### Standard math (element-wise)

| Op | y = f(x) | ∂y/∂x |
|----|----------|--------|
| **exp** | e^x | e^x |
| **ln / log** | ln(x) | 1/x |
| **log10** | log₁₀(x) | 1/(x ln 10) |
| **log2** | log₂(x) | 1/(x ln 2) |
| **log1p** | ln(1+x) | 1/(1+x) |
| **expm1** | e^x − 1 | e^x |
| **sqrt** | √x | 1/(2√x) |
| **rsqrt** | 1/√x | −1/(2 x^(3/2)) |
| **abs** | \|x\| | sign(x) (subgradient at 0) |
| **sign** | −1/0/+1 | 0 a.e. (undefined at 0) |
| **sin** | sin(x) | cos(x) |
| **cos** | cos(x) | −sin(x) |
| **tan** | tan(x) | 1 + tan²(x) = sec²(x) |
| **sinh** | sinh(x) | cosh(x) |
| **cosh** | cosh(x) | sinh(x) |
| **tanh** | tanh(x) | 1 − tanh²(x) |
| **asin** | asin(x) | 1/√(1−x²) |
| **acos** | acos(x) | −1/√(1−x²) |
| **atan** | atan(x) | 1/(1+x²) |
| **erf** | erf(x) | (2/√π) e^(−x²) |

### Activations (std::ml)

| Op | y = f(x) | ∂y/∂x |
|----|----------|--------|
| **relu** | max(0, x) | 1 if x > 0, 0 if x < 0 (subgradient at 0) |
| **sigmoid** | 1/(1+e^(−x)) | y(1−y) = σ(x)(1−σ(x)) |
| **tanh** | tanh(x) | 1 − y² |
| **softplus** | ln(1+e^x) | sigmoid(x) |
| **leaky_relu** | x if x>0 else αx | 1 if x>0 else α |
| **elu** | x if x>0 else α(e^x−1) | 1 if x>0 else α e^x |
| **gelu** | x·Φ(x) (approx) | (common approx has smooth derivative) |
| **swish** | x·sigmoid(x) | sigmoid(x) + x·σ(x)(1−σ(x)) |
| **softsign** | x/(1+\|x\|) | 1/(1+\|x\|)² (subgradient at 0) |
| **sign** | −1/0/+1 | 0 a.e. (no useful gradient) |

**Note:** `sign` has derivative 0 almost everywhere; for gradient-based training use a smooth surrogate (e.g. tanh) or a straight-through estimator if you keep sign in the forward pass.

In Einlang, **stdlib functions** (e.g. `std::math::sqrt`, `std::ml::relu`) are differentiated by differentiating their **body** when they are linked as user functions. Ops that delegate to NumPy (e.g. `python::numpy::*`) use **custom diff rules** so that autodiff still gets the correct derivatives. **Users can also provide custom diff rules** for their own functions (e.g. via `@fn` or a `custom_diff_body`-style mechanism) so that gradient flow is defined explicitly instead of by differentiating the function body.

---

## Where to go next

- **Full syntax and semantics:** [Language reference](https://github.com/einlang/einlang/blob/main/docs/reference.md)
- **Math and other functions:** [Standard library](https://github.com/einlang/einlang/blob/main/docs/stdlib.md) — `std::math`, `std::ml`
- **Autodiff:** [AUTODIFF_DESIGN](https://github.com/einlang/einlang/blob/main/docs/AUTODIFF_DESIGN.md) · [autodiff_small.ein](https://github.com/einlang/einlang/blob/main/examples/autodiff_small.ein), [autodiff_matmul.ein](https://github.com/einlang/einlang/blob/main/examples/autodiff_matmul.ein)
- **Run an example:** [Getting started](https://github.com/einlang/einlang/blob/main/docs/GETTING_STARTED.md) · [Examples by domain](https://github.com/einlang/einlang/blob/main/examples/README.md) · [README#examples](https://github.com/einlang/einlang/blob/main/README.md#examples)

