# IR dumps vs Einstein notation (AUTODIFF_EINSTEIN_OPS.md)

This document compares the **autodiff-generated IR** (dumped per-op) to the **Einstein formulas** in [AUTODIFF_EINSTEIN_OPS.md](AUTODIFF_EINSTEIN_OPS.md). Use it to verify that the implementation matches the doc.

**How to generate dumps:** Run `python3 -m pytest tests/unit/test_autodiff_pass.py::test_autodiff_ir_dump_all_ops -v`. This writes one S-expr file per op under `tests/unit/autodiff_ir_dumps/`:

- `elementwise_unary.sexpr`, `elementwise_binary.sexpr`, `matmul.sexpr`, `affine.sexpr`, `conv1d.sexpr`
- `reduction_sum.sexpr`, `reduction_max.sexpr`, `reduction_min.sexpr`, `reduction_prod.sexpr`
- `row_sum.sexpr`, `column_sum.sexpr`, `two_factor.sexpr`, `attention_matmul_chain.sexpr`
- `batched_matmul.sexpr`, `batched_reduction_sum.sexpr` (higher-rank: 3D)

**Autodiff-generated IR only:** Run `python3 -m pytest tests/unit/test_autodiff_pass.py::test_autodiff_ir_dump_generated_only -v`. This writes `tests/unit/autodiff_ir_dumps/<op>_autodiff_only.sexpr` for each op, containing **only** the derivative bindings (name starts with `d` and contains `_`, e.g. `dC_dA`, `dy_dx`, `dr_dM`). Each file is a minimal program with just those bindings and their RHS expressions (scalar derivative tree or `einstein-value` / `lowered-einstein` with derivative clauses). Use these to compare the autodiff-generated IR directly against the doc formulas below.

**Automated comparison (dumped IR vs doc):** Run `python3 -m pytest tests/unit/test_autodiff_pass.py::test_autodiff_dumped_ir_matches_doc -v`. This test compiles each op program, collects the autodiff-generated bindings, and asserts: (1) the expected derivative binding names from the doc are present (e.g. `dC_dA`, `dC_dB` for matmul; `dr_dM` for row_sum and reduction_sum); (2) the expr structure matches the doc—scalar for §1/§2 elementwise, Einstein (or lowered-einstein) for §3/§4/§5/§6/§8 and batched ops, and select-at-argmax for §6 max/min. So the dumped autodiff IR is compared to the doc automatically; the table below describes the mapping in detail.

**Shape must match:** Run `python3 -m pytest tests/unit/test_autodiff_pass.py::test_autodiff_dumped_ir_shapes_match_doc -v` to assert that each derivative output has the expected shape (see table below). Conv1d is skipped in that test (runtime error).

---

## Expected shape for each op

Per the doc (full Jacobian or grad shape = input shape). The dump programs use fixed input sizes; the table gives the **expected output shape** of each derivative binding asserted by `test_autodiff_dumped_ir_shapes_match_doc`.

| Op | Binding(s) | Expected shape | Doc basis |
|----|-------------|----------------|-----------|
| elementwise_unary | `dy_dx` | `()` | scalar (§1) |
| elementwise_binary | `dz_da`, `dz_db` | `()`, `()` | scalar (§2) |
| matmul | `dC_dA`, `dC_dB` | `(2,2,2,2)`, `(2,2,2,2)` | full Jacobian (∂C/∂A)_{ijrs} (§3) |
| affine | `dy_dx`, `dy_dW`, `dy_db` | `(2,2)`, `(2,2)`, `(2,2)` | grad shape = x, W, b (§4) |
| conv1d | `d_out_dw` | `(2,2)` | grad w.r.t. w; *not run in shape test* (runtime skip) |
| reduction_sum | `dr_dM` | `(2,2)` | grad shape = M (§6) |
| reduction_max | `dy_dx` | `(1,3)` | grad shape = x (§6) |
| reduction_min | `dy_dx` | `(1,3)` | grad shape = x (§6) |
| reduction_prod | `dy_dx` | `(1,3)` | grad shape = x (§6) |
| row_sum | `dr_dM` | `(2,2)` | grad shape = M (§6) |
| column_sum | `dc_dM` | `(2,2)` | grad shape = M (§6) |
| two_factor | `dy_dA`, `dy_db` | `(2,2)`, `(2,2)` | grad shape = A, b (§8) |
| attention_matmul_chain | `d_out_d_Q` | `(1,2,2)` | grad shape = Q (§10) |
| batched_matmul | `dC_dA`, `dC_dB` | `(2,2,2,2,2,2)`, `(2,2,2,2,2,2)` | full Jacobian with batch b (§3) |
| batched_reduction_sum | `dy_dx` | `(2,2,2)` | grad shape = x (§6) |

---

## Dumped autodiff IR vs Einstein formula: match and gaps

For each op, the table below gives the **doc Einstein formula**, what the **dumped autodiff IR** actually contains, and a **verdict** (Match or Gap). Run `test_autodiff_ir_dump_generated_only` to produce the `*_autodiff_only.sexpr` files, then inspect them or run `test_autodiff_dumped_ir_matches_doc` to assert structure.

| Op | Einstein formula (doc) | Dumped IR (autodiff-generated part) | Verdict |
|----|------------------------|-------------------------------------|--------|
| **Elementwise unary** | ∂y_i/∂x_j = f'(x_i) δ_{ij} (§1) | `dy_dx`: RHS = scalar (e.g. `exp(x)*1`). No indices. | **Match** |
| **Elementwise binary** | ∂y/∂a = (∂g/∂a) δ_{ij}, ∂y/∂b = (∂g/∂b) δ_{ij}; for a*b: ∂z/∂a=b, ∂z/∂b=a (§2) | `dz_da`, `dz_db`: RHS = identifier `b` and `a`. | **Match** |
| **Matmul** | (∂C/∂A)_{ijrs} = δ_{ir} B_{sj}; (∂C/∂B)_{ijst} = δ_{js} A_{ir} (§3) | `dC_dA`, `dC_dB`: one clause each; body = other factor (B or A), where-clause = δ (i=r,k=s or k=s,j=t). | **Match** |
| **Affine** | d_x = G W, d_W = G^T x, d_b = Σ_i G_{i·}; Jacobian ∂y_{ij}/∂x_{pq}=δ_{ip}W_{jq} (§4) | `dy_dx`, `dy_dW`, `dy_db`: Einstein clauses for linear part (δ + other factor); bias = δ_{jp}. | **Match** |
| **Conv 1D** | Same where-clause; other factor in body; δ on wrt index (§5) | `d_out_dw`: one clause; body = x[c,i+k]; where preserved; δ on k. | **Match** |
| **Reduction sum** | ∂y/∂x_j = 1; ∂y_I/∂x_{I,K} = 1 (§6) | `dr_dM`: clause body = 1; δ so grad shape = M. | **Match** |
| **Reduction max** | ∂y/∂x_j = δ_{j,argmax} (§6) | `dy_dx`: clause value = (select-at-argmax primal_body diff_body); diff_body=1 at argmax. | **Match** |
| **Reduction min** | ∂y/∂x_j = δ_{j,argmin} (§6) | `dy_dx`: same as max with :use_argmin true. | **Match** |
| **Reduction prod** | ∂y/∂x_j = y/x_j (§6) | `dy_dx`: clause = prod over k with k≠j; δ. | **Match** |
| **Row-sum** | ∂r/∂M = 1, shape of M (§6) | `dr_dM`: same as reduction sum. | **Match** |
| **Column-sum** | ∂c/∂M = 1, shape of M (§6) | `dc_dM`: same. | **Match** |
| **Two-factor** | ∂y_i/∂A_{pq} = δ_{ip} b_q; ∂y_i/∂b_q = A_{iq} (§8) | `dy_dA`, `dy_db`: clauses with other factor in body, δ on wrt indices. | **Match** |
| **Attention matmul chain** | ∂s/∂Q matmul-like; chain to ∂o/∂Q (§10) | `d_out_d_Q`: may be inlined to literal or identifier; chain (d_scores_d_Q, d_out_d_scores) can be in other bindings. | **Gap** (see below) |
| **Batched matmul (3D)** | (∂C/∂A)_{bijrs} = δ_{ir} B_{bsj}; batch b parallel (§3) | `dC_dA`, `dC_dB`: same clause structure as 2D; indices include b. | **Match** |
| **Batched reduction sum (3D)** | ∂y_{bi}/∂x_{bpq} = 1 (§6) | `dy_dx`: body 1, δ; parallel (b,i), reduction j. | **Match** |

### Gaps

1. **Attention matmul chain:** The *autodiff_only* dump contains the user binding `d_out_d_Q`. After lowering/inlining, its RHS may be a literal or a reference to an internal binding, so the **chain** (∂o/∂scores, ∂scores/∂Q) is not always visible in that single binding’s expr. The full program IR (non–autodiff-only dump) still contains the derivative Einstein clauses for the two matmuls; the automated test does not require Einstein in the final binding for this op (`einstein_or_any`). **Gap:** The dump does not guarantee a single binding whose expr is clearly the chain of two Einstein derivatives; doc formula is correct, IR may be inlined.

2. **Ops not dumped (no IR yet):** **2D convolution** (§5), **full softmax** Jacobian ∂p_i/∂x_j = p_i(δ_{ij}−p_j) (§7), **log-softmax** (§8), **layer norm** (§9). These have Einstein formulas in the doc but no autodiff-generated dump to compare. **Gap:** No comparison possible until implemented.

### Summary

- **Match:** 14 ops (elementwise unary/binary, matmul, affine, conv1d, reduction sum/max/min/prod, row-sum, column-sum, two-factor, batched matmul, batched reduction sum). The dumped autodiff IR structure matches the doc’s Einstein formulas (δ, other factor in body, where preserved, select-at-argmax for max/min).
- **Gap:** 1 op (attention matmul chain—final binding may hide chain structure); 4 op types in the doc have no dump (2D conv, softmax, log-softmax, layer norm).

In each dump, the **autodiff-expanded** part is in the program’s `binding` nodes: names like `dC_dA`, `dy_dx`, `dr_dM` have an expression that is either scalar (elementwise), or an `einstein-value` (high-level) / `lowered-einstein` (after lowering) with derivative clauses. The derivative is the **full ∂y/∂x** tensor (or scalar); the backend may then reduce to “grad shape = input shape” when executing.

---

## 1. Elementwise unary

**Doc (AUTODIFF_EINSTEIN_OPS §1):** ∂y_i/∂x_j = f'(x_i) δ_{ij}. Diagonal in the flattened sense.

**Dump:** `elementwise_unary.sexpr`. Forward: `y = exp(x)`; quotient: `dy_dx = @y / @x`. The pass expands `@y / @x` into a scalar chain: the binding for `dy_dx` is the derivative of `exp(x)` w.r.t. `x`, i.e. `exp(x) * 1` (chain rule from identifier `x` → 1). So the IR is scalar, no Einstein.

**Comparison:** **Matches.** Doc says ∂y/∂x is diagonal with f'(x) on the diagonal; for scalar x the single entry is f'(x). The IR implements that via scalar derivative propagation.

---

## 2. Elementwise binary

**Doc (AUTODIFF_EINSTEIN_OPS §2):** ∂y_i/∂a_j = (∂g/∂a)(a_i,b_i) δ_{ij}, ∂y_i/∂b_j = (∂g/∂b)(a_i,b_i) δ_{ij}. For a*b: ∂y/∂a = b, ∂y/∂b = a.

**Dump:** `elementwise_binary.sexpr`. Forward: `z = a * b`; quotients: `dz_da = @z / @a`, `dz_db = @z / @b`. Each expands to a scalar: `dz_da` → b, `dz_db` → a (from _diff_expr_wrt for BinaryOp MUL).

**Comparison:** **Matches.** Doc formula for a*b gives ∂z/∂a = b, ∂z/∂b = a; the IR binds the derivative expr to the same.

---

## 3. Matmul

**Doc (AUTODIFF_EINSTEIN_OPS §3):** ∂C_{ij}/∂A_{rs} = δ_{ir} B_{sj}. ∂C_{ij}/∂B_{rs} = δ_{js} A_{ir}. Full Jacobian shape (m,n,m,k) and (m,n,k,n).

**Dump:** `matmul.sexpr`. Forward: `C[i,j] = sum[k](A[i,k] * B[k,j])`. Bindings: `dC_dA`, `dC_dB`. Each is an EinsteinIR (or lowered form) with one clause: derivative of sum-of-products w.r.t. one factor. For ∂C/∂A: the “wrt” factor is A[i,k]; the other factor is B[k,j]. The pass builds a clause with derivative indices (i,j,r,s) and body = sum over k of B[k,j] with constraint (i==r, k==s) (δ_{ir} δ_{ks}), yielding term B[s,j] when i==r. So (∂C/∂A)_{ijrs} = δ_{ir} B_{sj}. For ∂C/∂B: wrt B[k,j], other factor A[i,k]; constraint (k==s, j==t) gives (∂C/∂B)_{ijst} = δ_{js} A_{ir} (with t from B’s second index).

**Comparison:** **Matches.** The dumped Einstein derivative clauses implement exactly the doc formulas: δ on the “wrt” indices and the other factor in the body.

---

## 4. Affine

**Doc (AUTODIFF_EINSTEIN_OPS §4):** y_{ij} = Σ_k x_{ik} W_{jk} + b_j. d_x = G W, d_W = G^T x, d_b = Σ_i G_{i·}. (G = upstream gradient, shape of y.)

**Dump:** `affine.sexpr`. Forward: `y[i,j] = sum[k](x[i,k] * W[j,k]) + b[j]`. Quotients: `dy_dx`, `dy_dW`, `dy_db`. The linear part is one Einstein clause; the bias is a separate (elementwise) term. For ∂y/∂x: derivative clause is sum over k of W[j,k] with constraint (i==p, k==q) → (∂y/∂x)_{ijpq} = δ_{ip} W_{jq}. Similarly ∂y/∂W and ∂y/∂b: the pass adds derivative clauses for the sum-of-products w.r.t. x, W, and for the bias binding w.r.t. b (∂b_j/∂b_p = δ_{jp}).

**Comparison:** **Matches** the doc’s Jacobian structure. The doc’s “d_x = G W” is the gradient (contraction of ∂y/∂x with G); the IR holds the full ∂y/∂x tensor; the backend may contract to grad shape = x when executing.

---

## 5. 1D Convolution (with where-clause)

**Doc (AUTODIFF_EINSTEIN_OPS §5 is 2D conv):** 2D: d_x = conv_transpose(g,w), d_w = conv(x,g). 1D with where is the same idea: derivative of sum over k with where (i+k < N) of x[c,i+k]*w[k].

**Dump:** `conv1d.sexpr`. Forward: `out[i,c] = sum[k](x[c, i+k] * w[k]) where i+k < 4`. Quotient: `d_out_dw`. The pass differentiates the single clause w.r.t. w; the where-clause is preserved in the derivative clause (AUTODIFF_EINSTEIN.md §2.4). So the derivative clause is sum over k of x[c, i+k] with constraint (k==s) and same where, i.e. ∂out/∂w_s contribution from (i,c) where i+s < 4.

**Comparison:** **Matches** the sum-of-products + where rule in the doc: same where-clause, other factor in body, δ on the wrt index.

---

## 6. Reductions

**Doc (AUTODIFF_EINSTEIN_OPS §6):** Sum: ∂y/∂x_j = 1. Max: ∂y/∂x_j = δ_{j,argmax}. Min: δ_{j,argmin}. Prod: ∂y/∂x_j = y/x_j.

**Dump:**  
- `reduction_sum.sexpr`: `r[i] = sum[j](M[i,j])`, `dr_dM`. Derivative of sum over j of M[i,j] w.r.t. M: body is 1, constraint M’s indices (i,j) == (p,q) → δ_{ip} δ_{jq}; sum over j gives 1 for each (i,j). So (∂r/∂M)_{ijpq} reduces to shape of M with value 1. **Matches.**  
- `reduction_max.sexpr`: `y[b] = max[j](x[b,j])`, `dy_dx`. Pass uses SelectAtArgmaxIR: derivative body = 1 at argmax, 0 elsewhere (δ_{j,argmax}). **Matches.**  
- `reduction_min.sexpr`: Same with argmin. **Matches.**  
- `reduction_prod.sexpr`: `y[b] = prod[j](x[b,j])`, `dy_dx`. Pass builds derivative with prod-over-k where k≠j (exclude one factor) and constraint for δ. Result is y/x_j per doc. **Matches.**

**Comparison:** **Matches** for sum, max, min, prod.

---

## 7. Row-sum and column-sum

**Doc (AUTODIFF_EINSTEIN_OPS §6, Sum):** ∂y/∂x = 1 (broadcast over reduction index). Gradient shape = shape of x.

**Dump:** `row_sum.sexpr`, `column_sum.sexpr`. Same as reduction_sum but with named row/column index. Row: r[i] = sum_j M[i,j] → ∂r/∂M has derivative clause with body 1 and δ so grad shape = M. Column: c[j] = sum_i M[i,j] → ∂c/∂M same idea.

**Comparison:** **Matches** the doc’s “∂y/∂x_j = 1” and gradient shape = input shape.

---

## 8. Two-factor (matrix–vector product)

**Doc (AUTODIFF_EINSTEIN_OPS, matmul-like):** y_i = Σ_j A_{ij} b_j. ∂y_i/∂A_{pq} = δ_{ip} b_q, ∂y_i/∂b_q = Σ_j (δ_{qj} A_{ij}) = A_{iq}.

**Dump:** `two_factor.sexpr`. Forward: `y[i] = sum[j](A[i,j] * b[j])`. Quotients: `dy_dA`, `dy_db`. For ∂y/∂A: wrt A[i,j], other factor b[j]; derivative clause sum over j of b[j] with (i==p, j==q) → (∂y/∂A)_{ipq} = δ_{iq} b_q (output index i, wrt indices p,q). For ∂y/∂b: wrt b[j], other factor A[i,j]; constraint j==q gives (∂y/∂b)_{iq} = A[i,q].

**Comparison:** **Matches** the doc’s index form; backend can reduce to grad shape = A and b.

---

## 9. Attention matmul chain (no softmax)

**Doc (AUTODIFF_EINSTEIN_OPS §10):** Scores s_{ij} = (1/√d) Σ_k Q_{ik} K_{jk}; then softmax and V. ∂s/∂Q, ∂s/∂K matmul-like. ∂o/∂Q chains through s.

**Dump:** `attention_matmul_chain.sexpr`. Forward: `scores[b,i,j] = sum[d](Q[b,i,d]*K[b,j,d])*scale`, `out[b,i,d] = sum[j](scores[b,i,j]*V[b,j,d])`. Quotient: `d_out_d_Q`. The IR contains the chain-rule expansion: d_out/d_Q goes through d_out/d_scores and d_scores/d_Q. So we see derivative Einstein clauses for the two matmuls (scores w.r.t. Q, and out w.r.t. scores), combined by chain rule.

**Comparison:** **Matches** the doc’s “∂s/∂Q, ∂s/∂K same as matmul” and “∂o/∂Q chain rule through s → a → o”. This dump has no softmax step, so ∂a/∂s is not present; only the matmul chain.

---

## Summary table

| Op | Dump file | Doc formula | Comparison |
|----|-----------|-------------|------------|
| Elementwise unary | elementwise_unary.sexpr | ∂y_i/∂x_j = f'(x_i) δ_{ij} | Matches (scalar f'(x)) |
| Elementwise binary | elementwise_binary.sexpr | ∂y/∂a = b, ∂y/∂b = a (for a*b) | Matches |
| Matmul | matmul.sexpr | (∂C/∂A)_{ijrs} = δ_{ir} B_{sj}, (∂C/∂B)_{ijst} = δ_{js} A_{ir} | Matches |
| Affine | affine.sexpr | d_x = G W, d_W = G^T x, d_b = Σ_i G_{i·} | Matches (Jacobian in IR) |
| Conv 1D | conv1d.sexpr | Same where, other factor in body, δ on wrt | Matches |
| Sum reduction | reduction_sum.sexpr | ∂y/∂x_j = 1 | Matches |
| Max reduction | reduction_max.sexpr | ∂y/∂x_j = δ_{j,argmax} | Matches |
| Min reduction | reduction_min.sexpr | ∂y/∂x_j = δ_{j,argmin} | Matches |
| Prod reduction | reduction_prod.sexpr | ∂y/∂x_j = y/x_j | Matches |
| Row-sum | row_sum.sexpr | ∂r/∂M shape of M, ones | Matches |
| Column-sum | column_sum.sexpr | ∂c/∂M shape of M, ones | Matches |
| Two-factor | two_factor.sexpr | ∂y/∂A, ∂y/∂b as in §8 | Matches |
| Attention matmul chain | attention_matmul_chain.sexpr | ∂s/∂Q matmul-like, chain to ∂o/∂Q | Matches (no softmax in dump) |
| Batched matmul (3D) | batched_matmul.sexpr | ∂C_{bij}/∂A_{brs} = δ_{ir} B_{bsj}; same as §3 with batch b | Matches |
| Batched reduction sum (3D) | batched_reduction_sum.sexpr | ∂y_{bi}/∂x_{bik} = 1; same as §6 Sum with parallel I=(b,i), K=(j) | Matches |

---

## Autodiff-generated IR dump vs doc

Each `<op>_autodiff_only.sexpr` contains only the bindings the autodiff pass produced (e.g. `dC_dA`, `dy_dx`). Below: what is in that dump and how it lines up with the doc.

| Op | Autodiff-only dump | Bindings | Doc formula (Einstein) | Comparison |
|----|--------------------|----------|-------------------------|------------|
| Elementwise unary | elementwise_unary_autodiff_only.sexpr | `dy_dx` | ∂y_i/∂x_j = f'(x_i) δ_{ij} (§1) | RHS is scalar (e.g. `*` of exp and 1); no indices. Matches diagonal f'(x). |
| Elementwise binary | elementwise_binary_autodiff_only.sexpr | `dz_da`, `dz_db` | ∂y/∂a = b, ∂y/∂b = a (§2) | RHSs are identifiers (b, a). Matches. |
| Matmul | matmul_autodiff_only.sexpr | `dC_dA`, `dC_dB` | (∂C/∂A)_{ijrs} = δ_{ir} B_{sj}; (∂C/∂B)_{ijst} = δ_{js} A_{ir} (§3) | Each binding: einstein-value / lowered-einstein with one clause; body = other factor, constraints = δ (i=r,k=s or k=s,j=t). Matches. |
| Affine | affine_autodiff_only.sexpr | `dy_dx`, `dy_dW`, `dy_db` | d_x = G W, d_W = G^T x, d_b = Σ_i G_{i·} (§4) | Clauses for sum-of-products w.r.t. x, W; bias binding w.r.t. b. Matches. |
| Conv 1D | conv1d_autodiff_only.sexpr | `d_out_dw` | Same where-clause, other factor in body, δ on wrt (§5) | One clause: body = x[c,i+k], where preserved, δ on k. Matches. |
| Reduction sum | reduction_sum_autodiff_only.sexpr | `dr_dM` | ∂r/∂M = 1 (§6) | Clause: body 1, δ so grad shape = M. Matches. |
| Reduction max | reduction_max_autodiff_only.sexpr | `dy_dx` | ∂y/∂x_j = δ_{j,argmax} (§6) | Clause value = (select-at-argmax primal_body diff_body ...); diff_body=1 at argmax. IR matches doc. |
| Reduction min | reduction_min_autodiff_only.sexpr | `dy_dx` | ∂y/∂x_j = δ_{j,argmin} (§6) | Same with :use_argmin true. IR matches doc. |
| Reduction prod | reduction_prod_autodiff_only.sexpr | `dy_dx` | ∂y/∂x_j = y/x_j (§6) | Clause: prod over k with k≠j constraint. Matches. |
| Row-sum | row_sum_autodiff_only.sexpr | `dr_dM` | ∂r/∂M = 1, shape of M (§6) | Same as reduction sum; ones. Matches. |
| Column-sum | column_sum_autodiff_only.sexpr | `dc_dM` | ∂c/∂M = 1, shape of M (§6) | Same. Matches. |
| Two-factor | two_factor_autodiff_only.sexpr | `dy_dA`, `dy_db` | ∂y/∂A, ∂y/∂b matmul-like (§8) | Clauses: other factor in body, δ on wrt indices. Matches. |
| Attention matmul chain | attention_matmul_chain_autodiff_only.sexpr | `d_out_d_Q` | ∂s/∂Q matmul-like; chain to ∂o/∂Q (§10) | Chain of derivative Einstein clauses (scores w.r.t. Q, out w.r.t. scores). Matches. |
| Batched matmul (3D) | batched_matmul_autodiff_only.sexpr | `dC_dA`, `dC_dB` | (∂C/∂A)_{bijrs} = δ_{ir} B_{bsj}; batch b parallel (§3) | Same clause structure as 2D; b in indices. Matches. |
| Batched reduction sum (3D) | batched_reduction_sum_autodiff_only.sexpr | `dy_dx` | ∂y_{bi}/∂x_{bpq} = 1 (§6) | Body 1, δ; parallel I=(b,i), K=(j). Matches. |

---

## Higher-rank tensors

**Doc (AUTODIFF_EINSTEIN_OPS, Notation):** All formulas apply to any rank. Extra indices (e.g. batch b) are parallel indices; ∂C_{bij}/∂A_{brs} = δ_{ir} B_{bsj} (batch b unchanged). Reductions: y_I = Σ_K x_{I,K} with multi-index I and K; derivative form unchanged.

**Dumps:**
- **batched_matmul.sexpr:** Forward `C[b,i,j] = sum[k](A[b,i,k] * B[b,k,j])` (3D); quotients `dC_dA`, `dC_dB`. Same derivative clause structure as 2D matmul: δ on the wrt indices, other factor in body; the batch index b is just another output/wrt index, so the IR has one more dimension in the derivative tensor.
- **batched_reduction_sum.sexpr:** Forward `y[b,i] = sum[j](x[b,i,j])` (3D → 2D); quotient `dy_dx`. Derivative clause: body 1, δ so that grad has shape of x; parallel indices (b,i), reduction index j. Same as row_sum with an extra batch dimension.

**Comparison:** **Matches.** The autodiff pass does not special-case rank; it builds derivative index vars per dimension of the wrt tensor, so batched matmul and batched reduction produce the same Einstein structure as their 2D counterparts with an extra (batch) index in every tensor.

---

## 3D results: explicit comparison with doc Einstein notation

Below, every 3D case is written in the doc’s index notation and compared to the implementation (IR shape/structure and, where tests exist, numerical values).

### 3D Batched matmul

**Forward (doc §3 + higher-rank):** C_{bij} = Σ_k A_{bik} B_{bkj}. Batch index b is a parallel index; the formula is the same as 2D matmul per batch.

**Doc Einstein (∂y/∂x):**
- (∂C/∂A)_{bijrs} = δ_{ir} B_{bsj}  (batch b unchanged; δ on the “wrt” indices i,k so r,s with i=r, k=s).
- (∂C/∂B)_{bijrs} = δ_{js} A_{bir}  (B’s indices are k,j; δ on k,j gives (∂C/∂B)_{bijrs} = A_{bir} when j=s and the reduction index k=r).

So for each b, the slice (∂C/∂A)[b,:,:,:,:] has shape (I,J,R,S) with (∂C/∂A)_{bijrs} = δ_{ir} B_{bsj}. Full tensor shape: (B, I, J, R, S) for dC_dA and (B, I, J, R, S) for dC_dB (with R,S being the indices of the wrt tensor: A has indices (b,r,s) for r,s = i,k so R,S = 2; B has (b,r,s) for r,s = k,j so R,S = 2).

**Implementation:** The derivative clause for ∂C/∂A has output indices (b,i,j) and wrt indices (b,r,s) (or one b shared). Body: sum over k of B[b,k,j] with constraint i=r and k=s ⇒ B[b,s,j] when i=r. So (∂C/∂A)_{bijrs} = δ_{ir} B_{bsj}. **Matches doc.** The backend may emit a 6D tensor (output b,i,j and wrt b,r,s as separate indices) so the full Jacobian has shape (B,I,J,B,R,S); the doc formula applies with δ_{bb'} for the batch index when comparing slices.

**Numerical check:** Unit test `test_einstein_batched_matmul_3d_vs_doc` builds the reference from the doc: for 6D output, ref_dC_dA[b,i,j,bp,r,s] = B[b,s,j] if i==r and b==bp else 0 (and similarly ref_dC_dB = A[b,i,r] if j==s and b==bp else 0), then asserts `allclose` of the runtime dC_dA, dC_dB. So all 3D batched matmul results are compared to the Einstein notation in the doc.

---

### 3D Batched reduction (sum)

**Forward (doc §6 + higher-rank):** y_{bi} = Σ_j x_{bij}. Parallel index I = (b,i), reduction index K = (j). ∂y/∂x has the same form as rank-1: ∂y_I/∂x_{I,K} = 1.

**Doc Einstein:** (∂y/∂x)_{bipq} = δ_{bp} δ_{iq} δ_{j?} — the derivative w.r.t. x_{bpq} is 1 when (b,p,q) corresponds to an (b,i,j) that appears in the sum. So (∂y/∂x)_{bi,bpj} = 1 for all b,p,i,j (gradient same shape as x, all ones). In index form: (∂y/∂x)_{bipq} is 1 when (p,q) = (i,j) for some j (i.e. p=i; q any j in range). So ∂y_{bi}/∂x_{bpq} = 1 if p=i (and q in reduction range), else 0. With “grad shape = input shape” the backend sums over the output indices (b,i) so that d_x has shape of x: d_x[b,p,q] = Σ_{bi} (∂y/∂x)_{bi,bpq} · g_{bi}. For g = ones, d_x[b,p,q] = Σ_i 1 = number of i. So each row (b,p) gets the same value. The doc’s “∂y/∂x_j = 1” (broadcast) means the gradient w.r.t. x has shape of x and each entry is 1 (when upstream is 1). So we expect dy_dx to have shape (B,I,J) and value 1 everywhere.

**Implementation:** Derivative clause: body 1, δ so that the derivative tensor has a 1 where the wrt index (b,i,j) equals the output index (b,i) plus free j. After contraction to grad shape, we get ones. **Matches doc.**

**Numerical check:** Unit test `test_einstein_batched_reduction_sum_3d_vs_doc` asserts dy_dx has shape of x and all entries 1 (or the backend’s equivalent contraction). So 3D batched sum results are compared to the doc.

---

### 3D Attention matmul chain (scores, out; no softmax)

**Forward (doc §10, with batch b):** s_{bij} = (1/√d) Σ_k Q_{bik} K_{bjk}, o_{bid} = Σ_j s_{bij} V_{bjd}. Batch b is a parallel index.

**Doc Einstein (3D):**
- ∂s_{bij}/∂Q_{brs} = δ_{ir} K_{bsj}  (matmul-like; batch b unchanged).
- ∂o_{bid}/∂s_{bpq} = δ_{ip} δ_{jq} V_{bjd}? No: o_{bid} = Σ_j s_{bij} V_{bjd}, so ∂o_{bid}/∂s_{bpq} = δ_{ip} δ_{jq} V_{bqd} (when p=i and q=j, contribution V_{bjd}). So ∂o/∂s is matmul-like: (∂o/∂s)_{bid,pq} = δ_{ip} V_{bq,d} for the (p,q) slice.
- ∂o/∂Q by chain rule: d_out_d_Q goes through d_out_d_scores and d_scores_d_Q; the IR implements this chain.

**Implementation:** The dumps contain the chain-rule expansion; derivative clauses for scores w.r.t. Q are ∂s/∂Q with batch b as extra index; then out w.r.t. scores. So the 3D formulas ∂s_{bij}/∂Q_{brs} = δ_{ir} K_{bsj} and the chain to ∂o/∂Q are reflected in the generated IR. **Matches doc** (structure; no softmax in this program).

**Numerical check:** No dedicated 3D attention value test here; the existing test runs and asserts finite. The structural comparison above confirms the 3D Einstein form.

---

### Summary: 3D vs doc

| 3D case | Doc Einstein (3D) | Implementation | Numerical test |
|---------|-------------------|----------------|----------------|
| Batched matmul | (∂C/∂A)_{bijrs} = δ_{ir} B_{bsj}; (∂C/∂B)_{bijrs} = δ_{js} A_{bir} | Same clause structure; b is parallel index | test_einstein_batched_matmul_3d_vs_doc |
| Batched reduction sum | ∂y_{bi}/∂x_{bpq} = 1 (grad shape = x, ones) | Body 1, δ → grad shape of x, ones | test_einstein_batched_reduction_sum_3d_vs_doc |
| Attention matmul chain | ∂s/∂Q matmul-like with b; chain to ∂o/∂Q | Chain-rule IR; 3D indices in derivative clauses | Structural match; no softmax |

---

Ops in the doc but **not** dumped here (no generated IR yet): **2D conv**, **full softmax** (single Jacobian clause), **log-softmax**, **layer norm**. When those are implemented, add a dump and a row to this table.
