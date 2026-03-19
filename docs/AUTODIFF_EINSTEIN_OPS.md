# ML operator derivatives in Einstein notation

This document expresses **∂y/∂x** (and, where relevant, ∂y/∂w, ∂y/∂b) for all ML operators using **Einstein/index notation**. It complements [AUTODIFF_OPS.md](AUTODIFF_OPS.md) and [AUTODIFF_EINSTEIN.md](AUTODIFF_EINSTEIN.md). All formulas are **output gradient w.r.t. inputs** (∂y/∂x), not loss gradients (∂L/∂x).

**Notation:**
- Indices: Latin letters (i, j, k, …); ranges implied by tensor shapes.
- **Kronecker delta:** δ_{ab} = 1 if a = b, 0 otherwise. For multi-index: δ_{I,J} = 1 if index tuple I equals J, else 0.
- **Sum:** repeated index implies sum over that index when not otherwise noted; we often write Σ_k explicitly for clarity.
- **Elementwise:** y_i = f(x_i) means same shape; ∂y_i/∂x_j = f'(x_i) δ_{ij}.

**Higher-rank tensors:** All formulas below apply to tensors of any rank. Extra indices (e.g. batch b, or spatial h,w) are **parallel indices** that participate in the same way: e.g. C_{bij} = Σ_k A_{bik} B_{bkj} is batched matmul; ∂C_{bij}/∂A_{brs} = δ_{ir} B_{bsj} (with batch b unchanged). Reductions use **parallel indices I** (e.g. b,i) and **reduction indices K** (e.g. j): y_I = Σ_K x_{I,K}, and ∂y_I/∂x_{I,K} has the same form as in the rank-1 case. No change to the derivative algebra; only the index sets I and K have more dimensions.

---

## 1. Elementwise unary

**Forward:** y_i = f(x_i) (same shape as x).

**Derivative:**
- ∂y_i / ∂x_j = f'(x_i) δ_{ij}.

So (∂y/∂x)_{ij} is diagonal in the flattened sense: only ∂y_i/∂x_i is nonzero, and equals f'(x_i).

| f | f'(x) |
|---|--------|
| neg | −1 |
| exp | exp(x) = y |
| ln | 1/x |
| sqrt | 1/(2√x) |
| tanh | 1 − y² |
| sigmoid | y(1−y) |
| relu | 1 if x>0 else 0 (subgradient at 0) |
| abs | sign(x) (subgradient at 0) |

**In Einlang:** One output index set I; ∂y_I/∂x_R = f'(x_I) δ_{I,R}.

---

## 2. Elementwise binary

**Forward:** y_i = g(a_i, b_i) (same shape; broadcasting can extend to a_i and b_i over repeated indices).

**Derivatives:**
- ∂y_i / ∂a_j = (∂g/∂a)(a_i, b_i) δ_{ij},
- ∂y_i / ∂b_j = (∂g/∂b)(a_i, b_i) δ_{ij}.

| g(a,b) | ∂y/∂a | ∂y/∂b |
|--------|--------|--------|
| a + b | 1 | 1 |
| a − b | 1 | −1 |
| a * b | b | a |
| a / b | 1/b | −a/b² |
| a^b | b·a^(b−1) | a^b·ln(a) |

**Broadcasting:** If a or b is broadcast, the derivative w.r.t. that operand is summed over the broadcast dimensions so the result has the operand’s shape.

---

## 3. Matrix multiply (matmul)

**Forward:**
- C_{ij} = Σ_k A_{ik} B_{kj}.
- A: (m×k), B: (k×n), C: (m×n).

**Derivatives (Jacobian structure):**
- ∂C_{ij} / ∂A_{rs} = δ_{ir} B_{sj}  ⇒  (∂C/∂A)·G = G B^T  (G same shape as C).
- ∂C_{ij} / ∂B_{rs} = δ_{js} A_{ir}  ⇒  (∂C/∂B)·G = A^T G.

**Einstein form for gradient of a scalar L(C):** If g_{ij} = ∂L/∂C_{ij}, then
- ∂L/∂A_{ik} = Σ_j g_{ij} B_{kj}  →  **dA = g B^T**  (in code: `dA[i,k] = sum[j](g[i,j]*B[k,j])`),
- ∂L/∂B_{kj} = Σ_i g_{ij} A_{ik}  →  **dB = A^T g**  (in code: `dB[k,j] = sum[i](A[i,k]*g[i,j])`).

**∂y/∂x (output gradient w.r.t. input):** For y = C, x = A or B, the derivative *tensor* ∂C/∂A has shape (m,n,m,k): (∂C/∂A)_{ijrs} = δ_{ir} B_{sj}. So (∂C/∂A)·G (contracting over C’s indices) gives Σ_j C_{ij} G_{ij} contribution to A: that is G B^T in matrix form.

---

## 4. Affine (linear map + bias)

**Forward:**
- y_{ij} = Σ_k x_{ik} W_{jk} + b_j.
- x: (batch, in_features), W: (out_features, in_features), b: (out_features), y: (batch, out_features).

**Derivatives:**
- ∂y_{ij} / ∂x_{pq} = δ_{ip} W_{jq}  ⇒  **∂y/∂x** applied to tensor G (shape of y): (∂y/∂x)·G has (∂y/∂x)·G)_{pq} = Σ_j G_{pj} W_{jq}  →  **d_x = G W** (same shape as x).
- ∂y_{ij} / ∂W_{pq} = δ_{jp} x_{iq}  ⇒  (∂y/∂W)·G: **d_W = G^T x**  (shape of W).
- ∂y_{ij} / ∂b_p = δ_{jp}  ⇒  (∂y/∂b)·G: **d_b = Σ_i G_{ij}**  (sum over batch/index i; shape of b).

**Einstein:**
- d_x[i,k] = Σ_j g[i,j] W[j,k],
- d_W[j,k] = Σ_i g[i,j] x[i,k],
- d_b[j] = Σ_i g[i,j].

---

## 5. 2D Convolution

**Forward (Einstein with where-clause):**
- y[b,oc,oh,ow] = Σ_{ic,kh,kw} x[b,ic,ih,iw] w[oc,ic,kh,kw]  where ih = oh+kh, iw = ow+kw (stride 1, no padding; adjust relation for stride/padding as needed).
- Plus bias: add b[oc] broadcast.

**Derivatives:**
- **∂y/∂x:** Same structure as forward; the linear map adjoint to (x ↦ y) is the **transpose convolution** of the gradient (shape of y) w.r.t. kernel w (same stride/padding). In index form: contributions to x[b,ic,ih,iw] come from y at [b,oc,oh,ow] where ih = oh+kh, iw = ow+kw, i.e. oh = ih−kh, ow = iw−kw:
  - d_x[b,ic,ih,iw] = Σ_{oc,kh,kw} g[b,oc,oh,ow] w[oc,ic,kh,kw]  where oh = ih−kh, ow = iw−kw (and indices in valid range).
- **∂y/∂w:** Convolution of input x with gradient g (shape of y):
  - d_w[oc,ic,kh,kw] = Σ_{b,oh,ow} x[b,ic,oh+kh,ow+kw] g[b,oc,oh,ow].
- **∂y/∂b:** d_b[oc] = Σ_{b,oh,ow} g[b,oc,oh,ow].

---

## 6. Reductions

### Sum
- **Forward:** y = Σ_i x_i  (or y_I = Σ_K x_{I,K} with parallel indices I and reduction indices K).
- **Derivative:** ∂y/∂x_j = 1 (all j); in Einstein with parallel indices: ∂y_I/∂x_{I,K} = 1 (broadcast over K). So gradient w.r.t. x has **same shape as x**; each entry gets the same upstream scalar (or broadcast from y).

### Max
- **Forward:** y = max_i x_i  (or y_I = max_K x_{I,K}).
- **Derivative:** ∂y/∂x_j = δ_{j, argmax(x)} (subgradient; 1 at argmax, 0 elsewhere). Gradient shape = **shape of x** (one nonzero per parallel slice). In Einstein: (∂y/∂x)_{I,K} is zero except at K = argmax_K(x_{I,K}), where it is 1.

### Min
- Same as max with argmin: (∂y/∂x) is 1 at argmin, 0 elsewhere; shape of x.

### Product
- **Forward:** y = Π_i x_i.
- **Derivative:** ∂y/∂x_j = Π_{i ≠ j} x_i = y/x_j (when x_j ≠ 0). In Einstein (parallel I, reduction K): (∂y/∂x)_{I,K} = (Π_{K'} x_{I,K'}) / x_{I,K} = y_I / x_{I,K}. Gradient has **shape of x**.

---

## 7. Softmax

**Forward:**
- m = max_i x_i;  p_i = exp(x_i − m) / Σ_j exp(x_j − m).

**Derivative (Jacobian):**
- ∂p_i / ∂x_j = p_i (δ_{ij} − p_j).
- So (∂p/∂x)·v = p ⊙ (v − (Σ_k p_k v_k) 1) = p ⊙ (v − ⟨p,v⟩).

**Einstein:** With indices i, j on the same dimension,
- (∂p/∂x)_{ij} = p_i (δ_{ij} − p_j).

---

## 8. Log-softmax

**Forward:** ℓ_i = x_i − log(Σ_j exp(x_j)) = x_i − logsumexp(x).

**Derivative:**
- ∂ℓ_i / ∂x_j = δ_{ij} − p_j,  where p = softmax(x).
- (∂ℓ/∂x)·v = v − p (Σ_k v_k).

**Einstein:** (∂ℓ/∂x)_{ij} = δ_{ij} − p_j.

---

## 9. Layer normalization

**Forward (over last axis, index k):**
- μ = (1/n) Σ_k x_k,  σ² = (1/n) Σ_k (x_k − μ)²,  x̂_k = (x_k − μ)/√(σ² + ε),  y_k = γ_k x̂_k + β_k.

**Derivatives (∂y/∂x, ∂y/∂γ, ∂y/∂β):**
- ∂y_k/∂x_j, ∂y_k/∂γ_j, ∂y_k/∂β_j are standard layer-norm backward formulas (see e.g. PyTorch). In index form: involve δ_{kj}, x̂_k, and sums over k; gradient w.r.t. x has same shape as x.

---

## 10. Attention (scaled dot-product)

**Forward:**
- Scores: s_{ij} = (1/√d) Σ_k Q_{ik} K_{jk}.
- m_i = max_j s_{ij};  e_{ij} = exp(s_{ij} − m_i);  a_{ij} = e_{ij} / Σ_{j'} e_{ij'}.
- Out: o_{id} = Σ_j a_{ij} V_{jd}.

**Derivatives (∂y/∂x for y = scores, attn, or output):**
- **∂s/∂Q,** **∂s/∂K:** Same as matmul: ∂s_{ij}/∂Q_{ik} = (1/√d) δ_{ii} K_{jk} ⇒ d_Q = (1/√d) (d_s) K^T; ∂s/∂K analogously.
- **∂a/∂s:** Softmax Jacobian per row i: ∂a_{ij}/∂s_{iq} = a_{ij}(δ_{jq} − a_{iq}).
- **∂o/∂a, ∂o/∂V:** o_{id} = Σ_j a_{ij} V_{jd} ⇒ ∂o/∂a and ∂o/∂V as matmul-like; d_V = a^T d_o, d_a = d_o V^T.
- **∂o/∂Q, ∂o/∂K, ∂o/∂V:** Chain rule through s → a → o; combine the above. Standard MHA backward: d_Q, d_K, d_V from d_o and primal Q, K, V, a.

**Einstein (key steps):**
- d_Q[i,k] = (1/√d) Σ_j (d_s)[i,j] K[j,k].
- d_K[j,k] = (1/√d) Σ_i (d_s)[i,j] Q[i,k].
- d_a[i,j] = Σ_d (d_o)[i,d] V[j,d].
- d_s[i,q] = Σ_j d_a[i,j] · a[i,j](δ_{jq} − a[i,q]) (softmax backward).
- d_V[j,d] = Σ_i a[i,j] (d_o)[i,d].

---

## Summary table (∂y/∂x in Einstein)

| Op | Forward (Einstein) | ∂y/∂x (Einstein) |
|----|--------------------|-------------------|
| Elementwise unary | y_i = f(x_i) | ∂y_i/∂x_j = f'(x_i) δ_{ij} |
| Elementwise binary | y_i = g(a_i,b_i) | ∂y_i/∂a_j = (∂g/∂a)_i δ_{ij}, same for b |
| Matmul | C_{ij} = Σ_k A_{ik}B_{kj} | (∂C/∂A)·G = G B^T; (∂C/∂B)·G = A^T G |
| Affine | y_{ij} = Σ_k x_{ik}W_{jk}+b_j | d_x = G W; d_W = G^T x; d_b = Σ_i G_{i·} |
| Conv2d | y = Σ x*w with where ih=oh+kh, iw=ow+kw | d_x = conv_transpose(g,w); d_w = conv(x,g); d_b = sum(g) |
| Sum | y = Σ_i x_i | ∂y/∂x_j = 1 |
| Max | y = max_i x_i | ∂y/∂x_j = δ_{j,argmax} |
| Min | y = min_i x_i | ∂y/∂x_j = δ_{j,argmin} |
| Prod | y = Π_i x_i | ∂y/∂x_j = y/x_j |
| Softmax | p_i = exp(x_i−m)/Σ_j exp(x_j−m) | ∂p_i/∂x_j = p_i(δ_{ij}−p_j) |
| Log-softmax | ℓ_i = x_i − logsumexp(x) | ∂ℓ_i/∂x_j = δ_{ij}−p_j |

See [AUTODIFF_OPS.md](AUTODIFF_OPS.md) for implementation notes and [AUTODIFF_EINSTEIN.md](AUTODIFF_EINSTEIN.md) for the general sum-of-products derivative rule with where-clauses.

---

## Test coverage (tests/unit/test_autodiff_pass.py)

Comparison of this document’s ops to the autodiff unit tests. Each row maps a § section or summary-table op to the test(s) that exercise it.

| § / Op | Doc section | Test(s) | Notes |
|--------|-------------|---------|--------|
| **1. Elementwise unary** | y_i = f(x_i), ∂y/∂x diagonal | `test_quotient_unary_neg`, `test_quotient_math_exp`, `test_quotient_math_ln`, `test_quotient_math_sqrt`, `test_quotient_math_sinh_cosh_tanh` (tanh), `test_pytorch_style_sigmoid`, `test_pytorch_style_relu`, `test_quotient_math_abs`, plus `test_quotient_math_*` for sin/cos/tan, asin/acos/atan, erf, log10/log2/log1p/expm1, neg, square, sign, rsqrt, min/max, clamp, saturate, deg/rad | Unary f and f' covered via scalar @y/@x and user/@fn. |
| **2. Elementwise binary** | g(a,b), ∂y/∂a and ∂y/∂b | `test_quotient_add`, `test_quotient_sub`, `test_quotient_mul`, `test_quotient_div`, `test_quotient_pow_*`, `test_quotient_math_pow_two_arg`, `test_quotient_mod` | Add, sub, mul, div, pow, mod. |
| **3. Matmul** | C = A@B, ∂C/∂A and ∂C/∂B | `test_einstein_quotient_compiles_and_runs`, `test_einstein_matmul_dC_dB`, `test_einstein_matmul_both_dC_dA_and_dC_dB`, `test_einstein_3x3_matmul_derivative` | Full Jacobian shape (m,n,m,k) and (m,n,k,n); refs match doc (δ_{ir}B_{sj}, A_{ir}δ_{js}). |
| **4. Affine** | y = xW + b, d_x/d_W/d_b | `test_einstein_affine_derivatives` | Gradients w.r.t. x, W, b; test asserts current backend (grad shape = input shape). Doc formula: d_x = G W, d_W = G^T x, d_b = Σ_i G_{i·} for upstream G. |
| **5. 2D Convolution** | conv2d, ∂y/∂x, ∂y/∂w, ∂y/∂b | — | **No test.** Only 1D conv with where-clause: `test_einstein_conv_1d_where_clause`. |
| **6. Reductions** | Sum, max, min, prod | `test_reduction_autodiff_sum`, `test_reduction_autodiff_max`, `test_reduction_autodiff_min`, `test_reduction_autodiff_prod`, `test_einstein_row_sum_derivative`, `test_einstein_column_sum_derivative`, `test_softmax_autodiff` (max + sum) | Sum ∂y/∂x = 1, max/min at argmax/argmin, prod = y/x_j; row/column sum grad shape = M. |
| **7. Softmax** | p_i = exp(x_i−m)/Σ…, ∂p/∂x | `test_softmax_autodiff` (builds max + sum; not full softmax Jacobian) | Full softmax Jacobian ∂p_i/∂x_j = p_i(δ_{ij}−p_j) not tested in one shot. |
| **8. Log-softmax** | ℓ_i = x_i − logsumexp, ∂ℓ/∂x | — | **No dedicated test.** |
| **9. Layer norm** | μ, σ², x̂, y = γ x̂ + β | — | **No test.** |
| **10. Attention** | Scores, softmax, out; ∂s/∂Q,K, ∂a/∂s, ∂o/∂a,V | `test_einstein_attention_matmul_chain_no_softmax` | Matmul chain only (scores, out); no softmax step, no ∂a/∂s or full MHA backward. |
| **Misc** | Two-factor (y = A b), scalar chain | `test_einstein_two_factor_product`, `test_chain_rule_through_lets`, `test_gradient_descent_autodiff_example` | Matrix–vector product ∂y/∂A, ∂y/∂b; chain rule; multi-let. |

**IR dump:** `test_autodiff_ir_dump_sexpr` compiles a program with matmul, row-sum, and affine quotients and writes `result.ir` (after autodiff) to `tests/unit/autodiff_ir_dump.sexpr` for inspection of the autodiff-generated IR.

**Per-op dumps and comparison with this doc:** Run `test_autodiff_ir_dump_all_ops` to write one IR file per op under `tests/unit/autodiff_ir_dumps/` (elementwise_unary, elementwise_binary, matmul, affine, conv1d, reduction_*, row_sum, column_sum, two_factor, attention_matmul_chain). See [AUTODIFF_EINSTEIN_OPS_IR_COMPARISON.md](AUTODIFF_EINSTEIN_OPS_IR_COMPARISON.md) for a side-by-side comparison of each dump with the Einstein notation in this document.
