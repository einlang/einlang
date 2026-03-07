# Root cause from diff 9c48d54..c23921a

## Summary

The diff introduces several changes that cause the 8 test failures. Root causes below.

---

## 1. Loop iterations cap (deit_tiny, any >100-iteration Einstein)

**Where:** `src/einlang/backends/numpy_einstein.py`

- **Change:** `_MAX = 1_000_000` → `_MAX = int(os.environ.get("EINLANG_EINSTEIN_LOOP_MAX", "100"))` in:
  - `_try_hybrid_vectorize_clause` (recurrence loop cap)
  - Scalar clause path (the `for loop_context in execute_lowered_loops(...)` loop)
- **Effect:** Any Einstein clause with more than 100 iterations raises "Einstein clause loop iterations exceeded limit."
- **Failing case:** `let predictions = [infer(img) | img in images]` in deit_tiny has `len(images) > 100`.

**Fix:** Restore a high default, e.g. `"1000000"`, for `EINLANG_EINSTEIN_LOOP_MAX`.

---

## 2. Einstein output dtype (einstein_windowing weighted_avg)

**Where:** Allocation in `_execute_lowered_einstein` is unchanged in the diff; dtype still comes from `tensor_element_type` then fallback then `np.int32`.

- **Cause:** Lowering does not set `element_type` from the **reduction body** type. For `let weighted_avg[i in 0..3] = sum[k in 0..3](...)` the clause value is a reduction; if `element_type` is taken from the declaration or from the reduction’s own type_info (e.g. i32 from index), the output is allocated as int32 and the float reduction result is truncated to 100.
- **Relevant diff:** Einstein lowering in the diff only adds `location=clause_loc` to the clause; it does not change how `element_type` is inferred for the declaration. So the bug is pre-existing but surfaces when the reduction result is float and the backend has no type_info/body-based float hint.

**Fix:** In Einstein lowering, for a clause whose value is a reduction, set the declaration’s `element_type` from the reduction **body** type (e.g. body.type_info), not from the reduction’s type_info. In the backend, when the clause body is a reduction, use the reduction body’s type (or `_body_implies_float`) so the output is float32 when the result is float.

---

## 3. RectangularAccess indexing (scatter_ops, topk, quantize, linear_algebra)

**Where:** `src/einlang/backends/numpy_expressions.py` — `visit_rectangular_access`

- **Change:** For `len(indices) > 1`, indices are turned into arrays, broadcast, and **clipped** per dimension: `np.clip(idx, 0, high)` with `high = shape[d] - 1`.
- **Effect:** Out-of-bounds indices are silently clamped instead of raising or being left to NumPy. Any op that relies on exact indexing (e.g. scatter, topk indices, or conv-derived ops) can get wrong values.
- **Failing tests:** test_scatter_ops, test_topk_2d_axis0_k1, test_quantize_linear_*, test_linear_algebra_clustered_accuracy.

**Fix:** Either remove the clipping and keep original semantics, or make clipping opt-in and ensure scatter/topk/quantize paths do not use it (or use explicit bounds checks instead of silent clip).

---

## 4. conv_ops.ein padding (linear_algebra, quantize if they use conv)

**Where:** `stdlib/ml/conv_ops.ein`

- **Change:** Single-clause padding with an if/else is replaced by multiple segments (e.g. interior region + separate border clauses with `0.0`).
- **Effect:** Padding layout and write order change; if shape/rank or segment ordering is wrong, numerical results can differ (e.g. linear_algebra_clustered_accuracy, quantize tests).

**Fix:** Reconcile segment ranges and shapes with the previous single-clause semantics, or revert to the single if/else clause if the multi-segment version is not equivalent.

---

## 5. Vectorized reduction + parallel_shape (lowered_execution.py)

**Where:** `src/einlang/runtime/compute/lowered_execution.py`

- **Change:** `_try_vectorized_reduction` now takes `parallel_shape`; when set, reduction vars are broadcast to `parallel_shape + red_shape` and the result is expected to have shape `parallel_shape + expected_shape`. New fallback when shape does not match: try to reduce and reshape to `parallel_shape`.
- **Effect:** Different code paths and shapes for reductions inside Einstein (where `parallel_shape` is set). If the fallback is used or dtype is inferred differently, results can change (e.g. int vs float, or wrong shape).

**Fix:** Ensure that when `parallel_shape` is set, the result dtype and shape match what the backend expects (e.g. same as when `parallel_shape` was None), and that the fallback path does not change scalar vs vectorized behavior in a way that breaks weighted_avg or other tests.

---

## 6. Range / shape analysis (possible rank or range errors)

**Where:** `src/einlang/passes/range_analysis.py`, `src/einlang/passes/shape_analysis.py`

- **Change:** Rank consistency check across clauses; `variable_ranges` propagation; shape_analysis no longer returns `None` on first clause dimension failure but fills with `None` and continues.
- **Effect:** Stricter rank checks can reject previously accepted programs; different `variable_ranges` can change which loops/indices are generated and thus execution.

**Fix:** Confirm that all failing tests’ declarations have consistent rank and that variable_ranges are correct; relax or adjust checks only where the new semantics are intended.

---

## Recommended fix order

1. **Loop limit:** Set default `EINLANG_EINSTEIN_LOOP_MAX` back to `"1000000"` (fixes deit_tiny).
2. **Output dtype:** In lowering, set Einstein `element_type` from reduction body type; in backend, use reduction body type or `_body_implies_float` so float reductions get float32 output (fixes einstein_windowing).
3. **Indexing:** Revert or narrow the RectangularAccess clipping so scatter/topk/quantize/linear_algebra indexing semantics are unchanged (fixes ml_ops failures).
4. **conv_ops / reduction / range:** Revisit only if failures remain after 1–3.
