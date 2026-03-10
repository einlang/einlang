
# Einstein clause vectorization (NumPy backend)

Design for how the NumPy backend chooses and executes vectorized, hybrid, or scalar paths for Einstein clauses, and how index types (loop variables, literals) are handled uniformly.

---

## 1. Goals

- **Prefer vectorized execution** so that clause bodies run over full slices (NumPy arrays) instead of per-element Python loops.
- **Support recurrence** (e.g. `state[t] = f(state[t-1])`) by iterating only over recurrence dimensions and vectorizing the rest (hybrid), or falling back to scalar when necessary.
- **Unify index handling** so that literal indices are treated as the half-open range `[lit, lit+1)` everywhere; no ad-hoc special cases for “literal” vs “loop” in slice/range logic.
- **Fast paths** for common cases: no recurrence, no literals → full vectorize; recurrence on one dim + literals (e.g. boundaries) → full vectorize when safe; partial writes (multi-clause same array) → correct slice assignment.

---

## 2. Index types and spaces

### 2.1 Clause indices (output space)

Each Einstein clause has **indices** on the LHS (and the body may read the same array at different indices). Indices are in **output (array) dimension order**.

- **LiteralIR**: constant integer (e.g. `0`, `127`). In lowering, this dimension gets **no loop**; the clause has fewer loops than output dimensions when literals are present.
- **IndexVarIR** (loop variable): e.g. `i`, `j`, `t`. Each corresponds to one **loop** with an iterable (e.g. `range(0, 128)`).

Example: `state[t in 1..500, c in 0..2, 0, j in 0..128]`  
→ indices `[t, c, 0, j]` (4 output dims), loops `[t, c, j]` (3 loops). Literal `0` is at output dimension 2.

### 2.2 Loop space vs output space

- **Loop index** `k` (0..`ndim-1`) refers to the k-th loop.
- **Output index** `d` (0..`output_ndim-1`) refers to the d-th dimension of the array.

When literals exist, loop indices and output indices do **not** coincide: e.g. loop 0 → output dim 0, loop 1 → output dim 1, loop 2 → output dim 3 (dim 2 is literal). The mapping is:

- **loop_dims**: for each loop index `k`, the output dimension it corresponds to.  
  `loop_dims[k] = d` means “loop k is the d-th output dimension.”  
  Computed by `_loop_dims_from_clause_indices(clause_indices, loops)` (list of output-dim indices for each loop; literals have no loop).

This mapping is required whenever we interpret **read indices** (e.g. in recurrence detection): read index lists are in **output** order, so we must index them with `out_d = loop_dims[k]`, not `k`, to avoid treating a literal slot as recurrence.

---

## 3. Unified range model: `[lit, lit+1)` and loop ranges

All “what range does this dimension use?” logic is derived from a single notion of **ranges** in output order.

### 3.1 Ranges per output dimension

- **Literal at output dim d** → range `(lit, lit+1)` (half-open, size 1). No loop is consumed.
- **Loop variable at output dim d** → range `(start, end)` from `_extract_loop_range(loop, expr_evaluator)` for the corresponding loop (consumed in clause index order).

Single source of truth: **`_ranges_from_clause_indices(clause_indices, lowered, expr_evaluator)`**  
Returns `Optional[List[Tuple[int,int]]]`: one `(start, end)` per output dimension, or `None` on mismatch/failure.

- Literals: `(int(lit.value), int(lit.value)+1)`.
- Loops: advance over `clause_indices`; for each non-literal, take next loop and call `_extract_loop_range(loops[loop_pos], expr_evaluator)`.
- Success only if every loop is consumed exactly once (e.g. 4 indices with 1 literal ⇒ 3 loops, 4 ranges).

### 3.2 Slice list and expand axes

- **Slice list** (for writing to the output): `[slice(s,e) for (s,e) in ranges]`. Used for both full-write and partial-write paths.
- **Expand axes** (for vectorized result): when the **result** has fewer dimensions than the output (because literal dims don’t appear in the result), we need to **expand** the result so its shape matches the slice.  
  `_slice_list_and_expand_axes_for_literal_as_range1(...)` returns `(slice_list, expand_axes)` where `expand_axes` are **result-axis** indices at which to insert size-1 dimensions (one per literal in output order, in the full-vectorize path). Expansion is applied in **ascending** axis order so dimensions line up with output positions.

No separate “literal handling” in the hot path: literals are just ranges of length 1.

---

## 4. Recurrence classification

The body may **read** the same array (LHS) at indices that differ from the write indices (e.g. `state[t-1, c, 0, j]`). Those dimensions where the read index is not the current loop variable (or a simple offset like `t-1`) are **recurrence dimensions**.

### 4.1 Why output-dim mapping matters

Recurrence is computed per **loop** dimension. Read index lists from the body are in **output** dimension order. So:

- For loop `k`, we must look at the read index at **output dimension** `loop_dims[k]`, not at position `k`.
- If we used `k` directly, a clause like `state[t, c, 0, j]` would have read indices `[t-1, c, 0, j]`; loop 2 is `j`, but at output index 2 we have the literal `0`. Comparing `0` to the loop variable for `j` would wrongly mark that dimension as recurrence.

So: **`_recurrence_dims(lowered, variable_defid, clause_indices)`** (and the hybrid/full helpers that use it) take **clause_indices** and use **loop_dims** to index into read index lists: `idx_list[loop_dims[k]]` vs `loop_defids[k]`.

### 4.2 Partial vs full recurrence

- **Partial recurrence**: `0 < len(recurrence_dims) < len(loops)`.  
  Example: `state[t, c, 0, j] = state[t-1, c, 0, j]` → recurrence only on `t`.  
  → Try **hybrid**: iterate recurrence dims (scalar), vectorize the rest.
- **Full recurrence**: `len(recurrence_dims) == len(loops)`.  
  Example: Fibonacci `fib[i] = fib[i-1]+fib[i-2]`, or 2D Laplacian `A[i,j] = A[i-1,j]+A[i,j-1]`.  
  → **Do not** use full vectorize (order matters). Set `recurrence_needs_scalar = True` and use the scalar path.
- **No recurrence**: `len(recurrence_dims) == 0`.  
  → Full vectorize is safe (and preferred).

When hybrid is **tried** but fails (e.g. iteration limit), we set `recurrence_needs_scalar = True` so the full-vectorize result is discarded and the scalar path is used.

When recurrence exists but the clause also has **literal** indices (e.g. boundary clause), we still allow using the **full vectorize** result (do not set `vec_result = None`) so boundary clauses can stay vectorized.

---

## 5. Execution path decision tree

For each clause, the backend chooses one of the following (in order):

1. **Call-scalar** (optional)  
   Condition: no literal indices, body has a call that uses loop variables in arguments (e.g. `topk(X, i)[j]`).  
   Action: iterate over those loop dimensions (scalar), vectorize the rest.  
   Ensures correct semantics when the call needs scalar indices.

2. **Hybrid**  
   Condition: body references LHS (recurrence), and `0 < len(recurrence_dims) < len(loops)`.  
   Action: iterate over `recurrence_dims` via `execute_lowered_loops`; for each iteration, set recurrence dims to scalar and non-recurrence dims to broadcasted arrays, evaluate body, then write the (possibly expanded) result to the correct output slice.  
   When the loop limit (config.DEFAULT_EINSTEIN_LOOP_MAX) is exceeded → **raise** (fail-fast). Other failures → set `recurrence_needs_scalar = True` and continue.

3. **Full recurrence**  
   Condition: body references LHS and `len(recurrence_dims) == len(loops)`.  
   Action: set `recurrence_needs_scalar = True` (no hybrid attempt).

4. **Full vectorize**  
   Condition: `lowered.loops` non-empty.  
   Sub-cases:
   - **Chunked** (optional): `EINLANG_CHUNK_ELEMENTS > 0`, no recurrence scalar requirement, no literals, same number of loops as output dims → vectorize in chunks along the first dimension to bound memory.
   - **Unified ranges**: `vec_shape` and `loop_ranges_override` from `_ranges_from_clause_indices` (when `clause_indices` and `output.ndim` match). For clauses with fewer loops than output dims, `vec_shape` and `loop_ranges_override` are built only from the loop dimensions (via `loop_dims`).
   - **Literal expand**: if the clause has literal indices, get `(slice_list, expand_axes)` from `_slice_list_and_expand_axes_for_literal_as_range1` and expand the vectorized result along `expand_axes` (ascending order) so its shape matches the slice.
   - If `recurrence_needs_scalar` is True and the clause has **no** literal indices, discard the vectorized result (`vec_result = None`) so the scalar path is used. If it has literal indices (e.g. boundary), keep the vectorized result.

5. **Scalar**  
   Fallback: iterate over all loop indices (nested loops), evaluate body once per cell, write to output. Subject to config.DEFAULT_EINSTEIN_LOOP_MAX; raises if exceeded.

---

## 6. Fast paths and special handling

### 6.1 Literal as `[lit, lit+1)` everywhere

- **Ranges**: literals contribute `(lit, lit+1)` from `_ranges_from_clause_indices`.
- **Hybrid**: when `clause_indices` is present, `loop_info` and per-iteration slice list are built from `_ranges_from_clause_indices` and `loop_dims`; literal output dims get `slice(lit, lit+1)` in the slice list; result is expanded only at **literal** output positions (in the squeezed-result axis order), not at recurrence positions (recurrence is already scalar in the slice).

### 6.2 Partial writes (multi-clause same array)

When the clause does not write to the full array (e.g. `concat[i in 3..7] = B[i-3]` with output shape `(7,)`), the vectorized result has shape matching the **clause** range (e.g. `(4,)`), not the full output. The slice into the output is in **output** space (e.g. `slice(3, 7)`).

- **Correct**: `output[tuple(slices_list)] = vec_result.astype(...)`  
  `vec_result` is already the clause-sized result; do **not** index it with `slices_list` (that would interpret 3:7 as indices into a length-4 array and produce a single element, then broadcast).

### 6.3 Chunked vectorize

When `EINLANG_CHUNK_ELEMENTS > 0` and the clause has no recurrence requirement and no literals, the backend may split the first dimension into chunks, run `_try_vectorize_clause` per chunk with `loop_ranges_override`, and write each chunk into the corresponding slice of the output. Reduces peak memory for very large first dimensions.

### 6.4 Hybrid with literals (boundary clauses)

Boundary clauses (e.g. `state[t, c, 0, j] = state[t-1, c, 0, j]`) have one recurrence dimension and one literal. Hybrid builds:

- `loop_info` from `_ranges_from_clause_indices` (so the literal dim gets range `(0, 1)` in the right place).
- Per-iteration slice list over **output** dims: recurrence dim → scalar from context; literal dim → `slice(lit, lit+1)`; other dims → `slice(start, end)`.
- After squeezing recurrence out of the result, expand only at **literal** positions (in squeezed-result axis indices), not at recurrence positions.

If hybrid fails (e.g. iteration limit), the clause can still use **full vectorize** because we do not discard `vec_result` when `has_literal_idx` is true (boundary case).

---

## 7. Helpers summary

| Helper | Purpose |
|--------|--------|
| `_loop_dims_from_clause_indices(clause_indices, loops)` | Map loop index → output dimension. When no literals or same count, returns `list(range(len(loops)))`. |
| `_ranges_from_clause_indices(clause_indices, lowered, expr_evaluator)` | One `(start, end)` per output dim; literal → `(lit, lit+1)`; loop → `_extract_loop_range`. |
| `_slice_list_from_clause_indices(...)` | Slice list from ranges: `[slice(s,e) for (s,e) in ranges]`. |
| `_slice_list_and_expand_axes_for_literal_as_range1(...)` | Slice list + result-axis indices for expanding result (literals only; recurrence_dims optional for hybrid). |
| `_recurrence_dims(lowered, variable_defid, clause_indices)` | Loop indices where body reads LHS at a different index; uses `loop_dims` to index read lists. |
| `_recurrence_dims_for_hybrid(lowered, variable_defid, clause_indices)` | Loop indices where every read is “backward” (e.g. `t-1`). Same output-dim mapping. |
| `_recurrence_dims_for_hybrid_or_full(...)` | Chooses hybrid-safe recurrence dims or falls back to full recurrence set. |
| `_extract_loop_range(loop, evaluator)` | `(start, end)` for a loop’s iterable (e.g. `LiteralIR(range)` or `RangeIR`). |

---

## 8. Environment and profiling

- **Loop limit**: Max iterations for hybrid, call-scalar, and scalar loops is `DEFAULT_EINSTEIN_LOOP_MAX` in `einlang.utils.config` (5000). No env override. When exceeded, the backend **raises** immediately (fail-fast). To allow longer recurrences, change the constant in config.
- **EINLANG_DEBUG_VECTORIZE**: When set (e.g. `1`), backend prints per-clause path (vectorized / hybrid / scalar / call-scalar) and a one-line summary of clause counts.
- **EINLANG_PROFILE_STATEMENTS**: Enables per-statement / per-clause timing in profile output.
- **EINLANG_CHUNK_ELEMENTS**: If > 0, full vectorize may run in chunks along the first dimension to limit memory.

---

## 9. Invariants

- Slice lists used for writing are always in **output** dimension order; their lengths equal `output.ndim`.
- When assigning a vectorized result into a **partial** slice, the result shape must equal the **slice** shape (number of elements in `output[tuple(slices_list)]`), and the assignment is `output[tuple(slices_list)] = vec_result` (never `vec_result[tuple(slices_list)]`).
- Recurrence detection always uses **output-dim** mapping when `clause_indices` is provided, so literal slots are never mistaken for recurrence.
- Literals are never given their own loop in lowering; they only appear in clause indices. So “literal” handling in the backend is entirely via the unified range `(lit, lit+1)` and expand axes.
