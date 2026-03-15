# Autodiff: derivative formulas by op

This document lists **derivative formulas** for each op. It extends [AUTODIFF_DESIGN.md](AUTODIFF_DESIGN.md). Notation: partial y / partial x means the derivative of y with respect to x.

**Matmul, conv, and einsum:** The compiler differentiates **high-level Einstein notation** (before lowering). So **matrix multiply** (`let C[i,j] = sum[k](A[i,k]*B[k,j])`), **convolution** expressed as Einstein with a where-clause (e.g. `sum[kh,kw](in[ih,iw]*w[kh,kw]) where ih = oh+kh, iw = ow+kw`), and **any einsum-style sum-of-products** are all supported: use `@C / @A`, `@out / @w`, etc. See [AUTODIFF_EINSTEIN.md](AUTODIFF_EINSTEIN.md) and examples [autodiff_matmul.ein](https://github.com/einlang/einlang/blob/main/examples/autodiff_matmul.ein).

---

## 1. Elementwise unary ops

y = f(x).

| Op | Forward y | partial y / partial x |
|----|-----------|------------------------|
| **neg** | y = -x | -1 |
| **pos** | y = x | 1 |
| **exp** | y = exp(x) | exp(x) = y |
| **ln** | y = ln(x) | 1/x  (x != 0) |
| **sqrt** | y = sqrt(x) | 1 / (2*sqrt(x)) = 1/(2*y) |
| **sin** | y = sin(x) | cos(x) |
| **cos** | y = cos(x) | -sin(x) |
| **tanh** | y = tanh(x) | 1 - y^2 |
| **sigmoid** | y = 1/(1+e^(-x)) | y * (1 - y) |
| **abs** | y = abs(x) | sign(x)  (subgradient at 0) |
| **relu** | y = max(0,x) | 1 where x>0, else 0  (subgradient at 0) |
| **leaky_relu** | y = x if x>0 else alpha*x | 1 where x>0, else alpha |

---

## 2. Elementwise binary ops

y = f(a,b).

| Op | Forward | partial y / partial a | partial y / partial b |
|----|---------|------------------------|------------------------|
| **add** | y = a + b | 1 | 1 |
| **sub** | y = a - b | 1 | -1 |
| **mul** | y = a * b | b | a |
| **div** | y = a / b | 1/b | -a / b^2  (b != 0) |
| **pow** | y = a^b | b * a^(b-1) | a^b * ln(a)  (a > 0 for this one) |
| **mod** (remainder) | y = a % b | 1 (subgradient) | 0 (subgradient; full grad would use -floor(a/b)) |

Broadcasting: derivative w.r.t. a broadcasted operand is summed over the broadcast dimensions so the result has the same shape as the operand.

---

## 3. Matrix multiply (matmul)

C = A*B, with A (m x k), B (k x n), C (m x n).

- partial C / partial A: the linear map that, given a (m x n) matrix, returns (that matrix) * B^T  (result shape m x k).
- partial C / partial B: the linear map that, given a (m x n) matrix, returns A^T * (that matrix)  (result shape k x n).

So for a given (m x n) matrix G: derivative of (scalar function of C) w.r.t. A has contribution shape (m x k) given by G * B^T; w.r.t. B by A^T * G.

Batched: same formula per batch element.

---

## 4. Affine map (linear map + bias)

y = x*W^T + b. So y_ij = sum_k x_ik * W_jk + b_j.

- partial y / partial x = W  (so derivative w.r.t. x has shape of x; each row of derivative = row of (something) * W).
- partial y / partial W: derivative w.r.t. W has shape of W; given gradient of loss w.r.t. y (shape of y), the derivative w.r.t. W is (that gradient)^T * x.
- partial y / partial b: derivative w.r.t. b is sum of (gradient w.r.t. y) over batch and spatial indices so that result has shape of b.

---

## 5. 2D Convolution (conv2d)

y = conv(x, w, b) (standard stride/padding/dilation).

- partial y / partial x: the linear map adjoint to (x -> y). Applying it to a tensor of shape of y gives a tensor of shape of x. Implemented as transpose-convolution of that tensor with kernel w (same stride/padding/dilation, no bias).
- partial y / partial w: convolve input x with the (gradient) tensor of shape of y; result has shape of w.
- partial y / partial b: sum the (gradient) tensor of shape y over batch and spatial indices; result has shape of b.

---

## 6. Reductions

**sum:** y = sum_i x_i.  partial y / partial x_i = 1 for all i (broadcast scalar 1 to shape of x).

**sum over subset:** e.g. y[i] = sum_j x[i,j].  partial y[i] / partial x[i,j] = 1 (broadcast along j).

**max:** y = max_i x_i.  partial y / partial x_i = 1 where i = argmax(x), else 0 (subgradient; one-hot or normalized if multiple argmax).

**min:** Same idea with argmin. Implemented as select-at-argmin (d_body at argmin(primal)).

**prod:** y = prod_i x_i.  partial y / partial x_i = prod_{j != i} x_j. Implemented as (prod body) * sum_i (d_body_i / body_i); valid when body_i != 0.

---

## 7. Softmax and log_softmax

**softmax:** p_i = e^(x_i - m) / sum_j e^(x_j - m), m = max(x).

partial p_i / partial x_j = p_i * (delta_ij - p_j). So (Jacobian-vector product): given vector v, (partial p / partial x) * v = p * (v - sum_k p_k * v_k).

**log_softmax:** ell_i = x_i - logsumexp(x).

partial ell_i / partial x_j = delta_ij - p_j where p = softmax(x). So (Jacobian-vector product): (partial ell / partial x) * v = v - p * sum_k v_k.

---

## 8. IR mapping (what to differentiate)

Autodiff runs **before** Einstein lowering, so it sees **high-level `EinsteinIR`** (declarative sum-of-products, with optional where-clause). Matmul, conv (written as Einstein), and general einsum-style expressions are differentiated via the same machinery ([AUTODIFF_EINSTEIN.md](AUTODIFF_EINSTEIN.md)).

| IR / construct | Differentiable? | Formula section |
|----------------|-----------------|------------------|
| `BinaryOpIR` ADD, SUB, MUL, DIV, POW, MOD | Yes | 2 |
| `UnaryOpIR` NEG, POS | Yes | 1 |
| `UnaryOpIR` NOT, BOOL_NOT | No (boolean) | — |
| `BuiltinCallIR` (exp, ln, sqrt, sin, cos, tanh, …) | Yes | 1 |
| **`EinsteinIR`** (sum-of-products, optional where) — **matmul, conv, einsum** | Yes | 3, 5, 6 |
| `LoweredRecurrenceIR` | Yes | chain rule along recurrence |
| Comparisons, conditionals | Subgradient / branch | ReLU etc. in 1 |
| max/min over indices | Yes | 6 |

---

## 9. Edge cases

- **ReLU, abs at 0:** not differentiable; use a defined subgradient (e.g. 0).
- **POW:** partial y/partial b = a^b * ln(a) undefined for a <= 0.
- **Division by zero:** undefined; avoid or document (e.g. clamp denominator).
- **MOD, integer DIV:** not differentiable in the usual sense; exclude or define subgradient.

---

## 10. Multi-head attention (MHA)

**Stdlib:** `std::ml::attention_ops` provides `attention_dummy`, `multi_head_attention_simple`, and `multi_head_attention`. They compute **scores = Q @ K^T** (scaled), **attention_weights = softmax(scores)**, and **output = attention_weights @ V**.

- **Differentiable today:** The **matmul parts** (scores, output) are sum-of-products Einstein and are supported: you can differentiate **scores** or **output** w.r.t. Q, K, V when written inline (e.g. `let scores[i,j] = sum[k](Q[i,k]*K[j,k])*scale; let out[i,d] = sum[j](scores[i,j]*V[j,d]);` then `@out/@Q`, `@scores/@K`, etc.).
- **Softmax autodiff:** The pass differentiates **sum** and **max** reductions (∂sum/∂body = sum of ∂body; ∂max/∂body = ∂body at argmax), so softmax (max → subtract → exp → sum → div) is fully differentiable; use `@out/@x` when `out = softmax(x)` (inline or `std::ml::softmax`). **Through stdlib MHA:** Full autodiff **through** `std::ml::attention_dummy` (or MHA) from a single `@out/@query` in the caller: the attention body uses **softmax** (max, sum, exp, div). Full autodiff through MHA from @out/@query is now possible in principle.
- **How to get MHA gradients:** (1) Write a minimal single-head attention **inline** with a user `fn` + `@fn` for softmax (or a custom backward), or (2) add **@fn** rules for `attention_dummy` / `multi_head_attention` that define the standard backward (dL/dQ, dL/dK, dL/dV from dL/d_output). The formulas are standard (see e.g. PyTorch/TF attention backward).

High-level design: [AUTODIFF_DESIGN.md](AUTODIFF_DESIGN.md).
