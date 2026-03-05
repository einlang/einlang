# Root cause of test_rnn_without_bias from git diff

## Summary

The failure (Y[1] = 0 instead of ~0.76) comes from the scalar path for the recurrence clause not writing to `output[1,0,0]`. The diff did not introduce a direct bug in the scalar write; it made recurrence clauses always take the scalar path, so any existing weakness in that path (or in iteration/index building) shows up for RNN.

## Relevant diff changes

1. **Recurrence no longer vectorized** (`numpy_einstein.py`):
   - Condition changed from `lowered.loops and not has_literal_idx` to also require `not _body_references_defid(lowered.body, variable_defid)`.
   - So recurrence clauses (e.g. RNN `hidden[t in 1..seq_length, ...] = ... hidden[t-1, ...]`) never call `_try_vectorize_clause` and always run in the scalar loop path.

2. **Scalar loop limit** (`numpy_einstein.py`):
   - `_MAX` reduced from `1_000_000` to `int(os.environ.get("EINLANG_EINSTEIN_LOOP_MAX", "100"))`.
   - For RNN without_bias there is only one recurrence iteration (t=1, b=0, h=0), so hitting the limit is not the cause of Y[1]=0.

3. **Hybrid path limit** (`_try_hybrid_vectorize_clause`):
   - Same `_MAX = 100` from env. Hybrid is only tried when `vec_result` is set; recurrence clauses do not set `vec_result`, so this path is not used for them.

4. **New `idx_tuple` fallback** (scalar path):
   - When `idx_tuple is None or len(idx_tuple) != output.ndim`, a new block builds `idx_tuple` by evaluating each `clause_indices` element with `idx.accept(self)`.
   - If that try block raises (e.g. wrong type or lookup), we `except` and `continue`, so we skip the write. That could explain missing output if `cell_index` or the loop-context fallback yields None or wrong length and the new fallback then fails.

5. **Slice assignment** (vectorized path only):
   - When `vec_result.shape != output.shape` we assign `output[out_slices] = vec_result` (no slicing of `vec_result`). This does not affect recurrence clauses because they never produce `vec_result`.

## Conclusion

- **Root cause from diff**: Recurrence is forced onto the scalar path by `_body_references_defid`. The scalar path must therefore correctly run the recurrence loop and build `idx_tuple` for the recurrence clause.
- **Why Y[1]=0**: Either (a) the recurrence loop runs 0 times (e.g. range `1..seq_length` evaluates to an empty range), or (b) we enter the loop but `idx_tuple` is None or has wrong length and the new fallback fails, so we `continue` and never write.
- **Next steps**: Add minimal logging (e.g. iteration count and `idx_tuple` for the recurrence clause) to see whether the loop runs and whether we skip the write due to `idx_tuple`/fallback, then fix that path (range evaluation, `cell_index`, or fallback) so the recurrence clause writes once to `output[1,0,0]`.
