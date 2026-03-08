# Math intuition

**If you think in equations and indices,** Einlang is built to match. This page shows how common math notation maps to code. Full semantics: [Language reference](reference.md).

---

## Sums and tensor expressions

| In math | In Einlang | Reference |
|--------|------------|-----------|
| C_{ij} = Σ_k A_{ik} B_{kj} (matrix multiply) | `let C[i, j] = sum[k](A[i, k] * B[k, j]);` | [Einstein notation](reference.md#einstein-notation) |
| s_i = Σ_j M_{ij} (row sums) | `let s[i] = sum[j](matrix[i, j]);` | [Reductions](reference.md#einstein-notation) |
| ‖A‖_F = √(Σ_{ij} A_{ij}²) (Frobenius norm) | `sqrt(sum[i, j](A[i, j] * A[i, j]))` | [Reductions](reference.md#einstein-notation) |
| y_{ij} = 2 x_{ij} (element-wise) | `let y[i, j] = 2.0 * x[i, j];` | [Rectangular declarations](reference.md#einstein-notation) |

Indices are **names**, not strings. The compiler infers ranges from array shapes and checks that dimensions match.

---

## Index relations (e.g. convolution)

| In math | In Einlang | Reference |
|--------|------------|-----------|
| Output index o, kernel k, input i = o + k | `where ih = oh + kh, iw = ow + kw` | [Where clauses — Index remapping](reference.md#where-clauses) |
| 2D conv: sum over input channels and kernel, with \( i = o + k \) | `let out[b,oc,oh,ow] = sum[ic,kh,kw](input[b,ic,ih,iw] * w[oc,ic,kh,kw]) where ih = oh + kh, iw = ow + kw;` | [Where clauses](reference.md#where-clauses) |

The **where** clause is the same index algebra you’d write under the Σ; the compiler checks bounds.

---

## Conditioned sums and guards

| In math | In Einlang | Reference |
|--------|------------|-----------|
| Σ_i x_i where x_i > 0 | `sum[i](data[i]) where data[i] > 0` | [Where clauses — Boolean guards](reference.md#where-clauses) |
| M_{ij} for i ≤ j (upper triangle) | `let upper[i, j] = matrix[i, j] where i <= j;` | [Boolean guards](reference.md#where-clauses) |

Guards in the **where** clause act as filters; the compiler still sees all indices.

---

## Recurrence equations

| In math | In Einlang | Reference |
|--------|------------|-----------|
| f_0 = 0, f_1 = 1, f_n = f_{n-1} + f_{n-2} | `let fib[0] = 0; let fib[1] = 1; let fib[n in 2..N] = fib[n-1] + fib[n-2];` | [Recurrence relations](reference.md#recurrence-relations) |
| h_0 = init, h_{t+1} = φ(h_t + x_t) (RNN step) | Use backward reference in time: `let h[0, i in 0..H] = init[i]; let h[t in 1..T, i in 0..H] = tanh(h[t-1, i] + x[t-1, i]);` — body reads `h[t-1, i]` only (math-aligned). | [Recurrence relations](reference.md#recurrence-relations) |
| h_{t,i,j} from h_{t-1} and spatial neighbors (e.g. 2D step) | Backward only in every dim: `let h[t in 1..T, i in 1..N-1, j in 1..N-1] = f(h[t-1,i,j], h[t,i-1,j], h[t,i,j-1]);` — no future values (no `h[t,i+1,j]` or `h[t,i,j+1]`). | [Recurrence relations](reference.md#recurrence-relations) |

Base cases and recursive case in one place; the compiler handles evaluation order.

---

## Where to go next

- **Full syntax and semantics:** [Language reference](reference.md)
- **Math and other functions:** [Standard library](stdlib.md) — `std::math`, `std::ml`
- **Run an example:** [Getting started](GETTING_STARTED.md) · [Examples](../README.md#examples)
