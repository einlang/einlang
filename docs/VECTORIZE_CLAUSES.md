# Vectorize Clauses in NumPy Einstein Backend

There are **four** execution paths for an Einstein clause. Three are vectorized (avoid per-cell Python loop); one is scalar fallback. The dispatcher tries them in order below.

---

## 1. Call-scalar hybrid

**When:** Body contains a **function call** that uses loop vars, and **only some** of the loop vars appear **inside that call’s arguments** (the rest appear only in outer indexing, e.g. `result[j]`).

**What:** Loop over the “call-arg” dims (scalar); for each combination, set the other loop vars to aranges and evaluate the body once (vectorized over those dims). Write each result into the corresponding slice of the output.

**Code:** `_try_call_scalar_vectorize_clause` in `numpy_einstein.py` (lines ~326–394). Triggered in `_execute_lowered_einstein_clause` when `has_call_using_loop` and `0 < len(scalar_loop_indices) < ndim`.

**Example (TopK):**
```rust
let values_work[i in 0..M_work, j in 0..k_val] = topk_2d_row_values(X_work, i, N_work, k_val)[j];
```
- `i` appears in the **call** (row index).
- `j` appears only in the **indexing** of the result.
- So: scalar over `i` (M_work iterations), vector over `j` (one eval per `i` with `j` as arange).
- Without this path, full vectorize would call `topk_2d_row_values(X_work, i_array, ...)` and get wrong semantics.

---

## 2. Full vectorize (fancy indexing)

**When:** No literal index in the clause indices, and either call-scalar did not apply or was not tried.

**What:** Set **all** loop vars to broadcast aranges (one dimension “active” per var), evaluate the body **once**, and use the result as the clause output (with shape/broadcast handling). No loop over indices.

**Code:** `_try_vectorize_clause` in `numpy_einstein.py` (lines ~183–251). Invoked from `_execute_lowered_einstein_clause` when `lowered.loops and not has_literal_idx`.

**Sub-cases:**

### 2a. Clause body is a reduction (parallel_shape passed; reduction runs scalar)

**When:** `clause.body` is a `LoweredReductionIR` (e.g. `sum[k](...)`).

**What:** Don’t call `body.accept(backend)` with aranges directly. Call `backend.evaluate_lowered_reduction(clause.body, tuple(output_shape))` so the reduction executor gets **parallel_shape**. The parallel loop vars are already set in the backend env by the clause; the reduction then runs over the reduction index (scalar path in `execute_reduction_with_loops`). (A vectorized reduction path exists in code but is disabled for correctness: it can produce wrong element order when reduction axis mapping does not match the body’s index order.)

**Example (encoder Q projection):**
```rust
let Q[s in 0..1500, d in 0..384] = sum[k in 0..384](ln1[s, k] * enc_blk_sa_q_w[L, k, d]) + enc_blk_sa_q_b[L, d];
```
- Loops: `s`, `d`. Body: one reduction over `k`.
- Full vectorize sets `s`, `d` to aranges and calls `evaluate_lowered_reduction(..., parallel_shape=(1500, 384))`.
- Reduction runs with parallel_shape → one vectorized body eval and one sum over the reduction axis (no 1500×384 scalar reduction loop).

### 2b. Clause body is not a reduction → single body eval with aranges

**When:** Body is not a `LoweredReductionIR` (e.g. element-wise op or call that supports arrays).

**What:** Set all loop vars to aranges; run `clause.body.accept(backend)` once; use the returned array (with shape/broadcast handling).

**Example (GELU):**
```rust
let act[s in 0..1500, k in 0..1536] = gelu(fc1[s, k]);
```
- `s`, `k` set to aranges → `fc1[s,k]` is a 2D array, `gelu(...)` is one vectorized call.
- One body eval for the whole clause.

**Example (element-wise arithmetic):**
```rust
let c1a[co in 0..384, t in 0..3000] = gelu(c1[co, t]);
```
- Same idea: one eval with `co`, `t` as aranges.

---

## 3. Recurrence hybrid

**When:** Full vectorize **succeeded** but the clause only fills a **partial** range (non-full slice), and the body **reads the same variable being written** at a different index (recurrence).

**What:** Treat the clause as having two kinds of dimensions: “recurrence” (must be scalar) and “parallel” (can be vectorized). Loop over recurrence dims; for each combination set the other loop vars to aranges and evaluate the body once; write the result into the correct slice.

**Code:** `_try_hybrid_vectorize_clause` in `numpy_einstein.py` (lines ~254–322). Triggered only inside the full-vectorize success path when `not range_is_full_partial` and `recurrence_dims` is non-empty (and full vectorize did not already fill the full output).

**Example (recurrence over one dim):**
```rust
let out[i in 0..N, j in 0..M] = x[i, j] + out[i - 1, j];  // if in bounds
```
- Body references `out` at `i - 1`, so dimension `i` is recurrence.
- Recurrence hybrid: scalar over `i`, vector over `j`; one body eval per `i` with `j` as arange.

---

## 4. Scalar (fallback)

**When:** No vectorized path was used (call-scalar not applicable, full vectorize not tried or returned `None`, recurrence hybrid not applicable or failed).

**What:** Classic loop: `execute_lowered_loops` over **all** loop vars; for each cell, set env and evaluate the body once; write the scalar result into `output[idx_tuple]`.

**Code:** The `else` branch in `_execute_lowered_einstein_clause` (lines ~811–882): `for loop_context in execute_lowered_loops(lowered.loops, ...)` then `value = lowered.body.accept(self)` and `output[idx_tuple] = value`.

**Example:** Any clause that uses a literal index, or a body that doesn’t support array indices (and doesn’t match call-scalar), or that fails during vectorize (e.g. exception or wrong shape).

---

## Summary table

| Path               | Condition (simplified)                          | Loop vars in body        | Example                          |
|--------------------|--------------------------------------------------|---------------------------|----------------------------------|
| Call-scalar hybrid | Call using loop vars; only some in call args    | Some scalar, rest vector  | `topk_2d_row_values(X, i, ...)[j]` |
| Full vectorize     | No literal index                                | All as aranges            | GELU, matmul (via 2a)            |
| 2a Vectorized red. | Body is reduction + parallel_shape              | Reduction vectorized      | `sum[k](A[s,k]*W[k,d])`         |
| 2b Fancy indexing  | Body not reduction                              | One body eval             | `gelu(fc1[s,k])`                |
| Recurrence hybrid  | Partial range + body reads LHS at other index   | Recurrence scalar, rest vector | `out[i,j] = f(out[i-1,j])`  |
| Scalar             | Otherwise                                       | All scalar                | Fallback                         |

---

## Debugging

- Set `EINLANG_DEBUG_VECTORIZE=1` to see which path runs: `[call-scalar]`, `[vectorized]`, `[hybrid]`, or `[scalar]` per clause (with line number).
- Set `EINLANG_PROFILE_REDUCTIONS=1` to see whether each reduction used the **vectorized** or **scalar** path inside `execute_reduction_with_loops`.
