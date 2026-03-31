# Design: Generic Conv Loop Optimization

This document describes how to speed up the **generic** convolution autodiff path in Einlang without introducing convolution-specific backward kernels as the primary solution.

The goal is to make the current lowered Einstein / reduction runtime handle rank-2 and rank-3 convolution pullbacks much more efficiently.

## Scope

In scope:

- Rank-2 and rank-3 convolution autodiff performance for `@y/@x`, `@y/@w`, `@y/@b`
- Generic runtime optimizations in lowered Einstein / lowered reduction execution
- Lowering rewrites that preserve generic execution but make it vectorizable
- Elimination of repeated Python loop re-entry and repeated replayed binding evaluation

Out of scope:

- Sparse tensor support
- GPU kernels
- Replacing the generic path entirely with hand-written `conv2d` / `conv3d` pullbacks

## Summary

The current rank-2/rank-3 generic conv autodiff path is slow because it lowers to:

1. padded tensor construction (`Xp`)
2. guarded convolution reduction (`if ... { ... } else { 0 }`)
3. nested pullback reductions on top of that guarded reduction

The runtime then sees:

- `if` in reduction bodies
- nested reductions inside reduction bodies

and intentionally falls back to **scalar Python loop execution**.

This is the main performance cliff.

## Rank-2 Loop Structure

For the rank-2 test case in [test_autodiff_pass.py](/Users/user/Documents/einlang/tests/unit/test_autodiff_pass.py#L806):

```ein
let x = [[[[1.0, 2.0, 3.0],
           [4.0, 5.0, 6.0],
           [7.0, 8.0, 9.0]]]];
let w = [[[[1.0, 1.0],
           [1.0, 1.0]]]];
let b = [0.0];
let y = std::ml::conv(x, w, b, [1, 1], [0, 0, 0, 0], [1, 1], 1);
```

Shapes:

- `X`: `(1, 1, 3, 3)`
- `W`: `(1, 1, 2, 2)`
- `B`: `(1,)`
- `Y`: `(1, 1, 2, 2)`
- `Xp`: `(1, 1, 3, 3)` for zero padding in this specific case

Forward conv loop structure:

- outer loops:
  - `batch.0 in 0..1`
  - `co in 0..1`
  - `i in 0..2`
  - `j in 0..2`
- reduction loops:
  - `cl in 0..1`
  - `m in 0..2`
  - `n in 0..2`

Logical contraction shape:

- parallel/output shape: `(1, 1, 2, 2)`
- reduction shape: `(1, 2, 2)`
- conceptual broadcasted working shape: `(1, 1, 2, 2, 1, 2, 2)`

The important point is that this shape is **not large**. The slowness is dominated by Python interpreter overhead, not tensor size.

## What The Runtime Is Doing Today

### 1. Guarded conv bodies are forced scalar

In [numpy_expressions_mixin.py](/Users/user/Documents/einlang/src/einlang/backends/numpy_expressions_mixin.py#L641), the backend computes:

- `force_scalar_reduction = _contains_nested_lowered_reduction(expr.body) or _contains_if_expression(expr.body)`

That disables windowed einsum / matmul / generic einsum for guarded conv bodies.

### 2. Scalar reduction executor takes over

In [lowered_execution.py](/Users/user/Documents/einlang/src/einlang/runtime/compute/lowered_execution.py#L431), speculative vectorization is gated off, and the executor falls back to:

- nested Python iteration over reduction loops
- per-iteration body evaluation via `body_evaluator`

### 3. Nested reduction makes `dy_dx` much worse than primal conv

The rank-2 `dy_dx` trace shows:

- primal conv reductions:
  - `body_kind=IfExpressionIR`
  - `force_scalar_reduction=true`
  - `n_loops=3`
  - `parallel_shape=[1,1,2,2]`
- reverse `dy_dx` outer reduction:
  - `body_kind=BinaryOpIR`
  - `has_nested_red=true`
  - `force_scalar_reduction=true`
  - `n_loops=7`

This comes from [/tmp/rank2-perf.ndjson](/tmp/rank2-perf.ndjson).

After the outer `7`-loop reduction starts, the same trace shows repeated evaluation of inner:

- `body_kind=LoweredEinsteinIR`
- `n_loops=4`

So the current hot path is effectively:

1. outer scalar reduction over 7 indices
2. inside each iteration, evaluate another 4-loop Einstein body

That is the core bottleneck.

## Why Sparse Tensors Are Not The Right Tool

Sparse tensors are not the right optimization here.

Reasons:

- The tensors are dense.
- The logical iteration space is small for the test case.
- The problem is repeated interpreter work, not density.
- Sparse support would add major complexity to lowering, shape tracking, broadcasting, guards, and autodiff.

Conclusion:

- optimize loop execution
- do not add sparse tensors for this problem

## Root Causes

The traced bottlenecks reduce to four generic issues.

### A. `if ... else 0` inside reduction bodies

This blocks all current fast paths, even when the body could be vectorized with a mask.

### B. Nested tensor-valued reductions

The current executor treats nested reductions conservatively and drops into scalar evaluation.

### C. Replayed bindings are reevaluated too often

Bindings like:

- `Xp`
- `conv_sum`
- `output`

are pure functions of the current environment, but in nested generic autodiff they can be re-entered many times from inside outer reductions.

### D. Reduction bodies are not normalized into vectorizable forms

The fast path matcher expects direct sum-of-products structure. Autodiff currently leaves behind:

- block wrappers
- zero-addends
- masked scalar branches
- nested lowered Einstein nodes

which obscures vectorizable structure.

## Design Goals

1. Keep the path generic.
2. Preserve exact semantics of guards and masking.
3. Reduce Python loop nesting.
4. Reuse existing NumPy array execution and einsum-based helpers where possible.
5. Avoid huge temporary materialization when a masked/fused vectorized formulation is available.

## Proposed Optimization Plan

### Phase 1: Normalize Reduction Bodies Before Execution

Add a runtime-side or lowering-side normalization step for `LoweredReductionIR.body`:

- unwrap trivial blocks
- strip zero-like `+ 0` / `- 0`
- canonicalize `if cond { expr } else { 0 }` into a masked body form
- flatten simple nested `BinaryOpIR` structure

Goal:

- make more bodies match “sum of products under a mask”
- improve hit rate for existing vectorized paths

### Phase 2: Vectorized Masked Reduction

Add a generic vectorized path for:

- `sum[red...](if cond { body } else { 0 })`

Implementation strategy:

1. Build broadcasted reduction index arrays.
2. Evaluate `cond` once as a boolean ndarray.
3. Evaluate `body` once as an ndarray.
4. Use `np.where(cond, body, 0)` or equivalent masked multiplication.
5. Reduce over reduction axes with NumPy.

Important:

- this should be allowed even when the original body contains `if`
- this is different from eager scalar branching

This directly targets primal conv and many pullback reductions.

### Phase 3: Nested Reduction Hoisting / Memoization

For a scalar reduction whose body contains a nested `LoweredEinsteinIR` or nested `LoweredReductionIR`:

- detect whether the nested node depends only on a subset of the outer loop variables
- cache by those outer loop variable values

Example:

- if an inner 4-loop Einstein depends on `(batch, co, i, j)` but not on `(cl, m, n)`,
  then the outer `7`-loop reduction should evaluate it once per `(batch, co, i, j)`,
  not once per `(batch, co, i, j, cl, m, n)`

This is the most important generic optimization for `dy_dx`.

### Phase 4: Tensor-Valued Reduction Vectorization

Generalize `_try_vectorized_reduction` in [lowered_execution.py](/Users/user/Documents/einlang/src/einlang/runtime/compute/lowered_execution.py) so it can safely reduce tensor-valued bodies over multiple reduction loops when:

- the body evaluates to an ndarray with stable shape
- guards are absent or already converted into a mask
- no analytical scalar shortcut is used

This should let tensor-valued inner pullbacks reduce directly with NumPy instead of Python loops.

### Phase 5: Binding Result Cache

Add an execution cache for pure lowered binding evaluation:

- key by binding defid plus fingerprints of dependency values
- cache only for `BindingIR` whose RHS is lowered Einstein / lowered reduction / pure tensor expressions

Expected wins:

- replayed `Xp`
- replayed `conv_sum`
- replayed `output`
- reused autodiff intermediates

Cache invalidation must be dependency-driven, not global.

### Phase 6: Hoist Outer Context Into Nested Bodies

Nested reductions already require explicit propagation of outer loop context.

Make this systematic:

- propagate current outer loop bindings into nested reduction bodies by semantic index name
- avoid repeated expensive name-collection per iteration
- precompute “outer name -> body defids” maps once per lowered node

This helps both correctness and speed.

## Non-Goals

This design does **not** try to:

- represent conv backward as sparse tensors
- introduce general symbolic loop fusion across all IR
- replace generic autodiff with op-specific kernels everywhere

Those may still be useful later, but they are separate decisions.

## Success Criteria

Minimum acceptable results:

1. Rank-2 `dy_dx` no longer spends most of its time in scalar Python nested reductions.
2. Rank-2 and rank-3 conv autodiff tests complete in reasonable time under pytest.
3. Existing correctness stays unchanged for:
   - rank-1 conv
   - rank-2 conv
   - rank-3 conv
   - non-conv reduction/autodiff tests

Suggested perf targets:

- rank-2 `dy_dx` standalone runtime: reduce by at least `5x`
- rank-2 full test (`dy_dx`, `dy_dw`, `dy_db`): reduce by at least `3x`
- rank-3 full test: reduce substantially enough that it no longer appears “hung”

## Implementation Order

Recommended order:

1. vectorized masked reduction for `if cond { expr } else { 0 }`
2. nested reduction hoisting/caching keyed by outer-loop subsets
3. binding result cache for replayed lowered bindings
4. broader tensor-valued reduction vectorization

Why this order:

- phase 1 unlocks primal conv immediately
- phase 2 attacks the `dy_dx` hot path directly
- phase 3 reduces duplicate replay cost
- phase 4 is broader and riskier, so it should come after the simpler wins

## Open Questions

1. Should masked vectorized reductions be represented explicitly in IR, or only as a runtime recognition?
2. How aggressively can nested lowered Einstein bodies be memoized without over-caching large tensors?
3. Should replayed binding caches live only inside one block execution scope, or persist across repeated nested calls in the same runtime execution?
4. Can some padding clauses be fused into a single masked read view so `Xp` never materializes at all on the generic path?

## Recommendation

Do **not** add sparse tensors for this issue.

Instead:

- optimize masked reductions
- hoist nested tensor-valued reductions
- cache replayed pure bindings
- keep the path generic, but stop paying Python-loop costs for patterns that are already dense and regular

This is the most direct way to speed up rank-2 and rank-3 generic conv autodiff without abandoning the generic execution model.
