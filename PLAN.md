# Julia-Aligned Tensor Quotient Rewrite

## Summary
Rewrite tensor-valued `@num / @den` handling around a formal seeded-pullback model, matching Julia/ChainRules semantics:

- scalar result: cotangent seed = `1`
- tensor result: cotangent seed = `ones(shape(result))`
- indexed tensor result: cotangent seed = `onehot(indices, shape(result))`
- reductions propagate seed structurally instead of scalarizing the callee output

The goal is to eliminate the current mixed symbolic/Jacobian shortcuts that still break `sum(max_pool(relu(...)))` and scalar aliases of tensor slices, while preserving the recurrence fixes already working.

## Design Changes
- Keep public syntax unchanged: `@x`, `@y / @x`, `print(@y)`.
- Treat tensor quotients as **VJP/pullback problems**, not full Jacobians or directional hacks.
- Add one internal implementation concept only:
  `build_seeded_pullback(expr, seed, wrt_defid, ctx) -> ExpressionIR`
  This is a compiler helper, not a new public syntax feature.
- Explicit seed rules:
  - `@tensor / @x`: seed = tensor of ones with tensor’s shape
  - `@(tensor[i...]) / @x`: seed = onehot at `i...`
  - `@(sum(body)) / @x`: seed = `1`, then push through reduction structurally
- No correctness logic may depend on callee names such as `max_pool`, `relu`, `softmax`, etc.
- Scalar aliases of tensor slices must be defined generically:
  if `e = T[idx...]`, then `@num / @e` is computed as `@num / @T` followed by projection to `idx...`.

## Implementation Changes
- Add a new helper module, recommended name:
  [src/einlang/passes/autodiff/_pullback.py](/Users/user/Documents/einlang/src/einlang/passes/autodiff/_pullback.py)
- Move all tensor-result quotient logic there. `JacobianVisitor` should remain the scalar/symbolic derivative engine; it should not keep growing tensor-call special cases.

### Pullback builder rules
Implement `build_seeded_pullback(expr, seed, wrt, bindings, resolver, ...)` with these exact rules:

- `IdentifierIR`
  - If `expr.defid == wrt`, return `seed`
  - Else if binding does not depend on `wrt`, return zero
  - Else recurse on the binding expression with the same seed
- `RectangularAccessIR(base, indices)`
  - Build a scatter seed for `base` with shape `shape(base)`
  - The scatter places `seed` at `indices`
  - Recurse on `base` with the scattered seed
- `ReductionExpressionIR(sum, loop_vars, body)`
  - Broadcast `seed` across the reduction loop vars
  - Recurse on `body` with that broadcasted seed
- `BinaryOpIR` / `UnaryOpIR`
  - Implement VJP rules, i.e. multiply the incoming `seed` by local partials before recursing
  - Example: for `a * b`, recurse with `seed * b` into `a` and `seed * a` into `b`
- `IfExpressionIR`
  - Preserve the branch condition, recurse into branches with the same seed
- `BlockExpressionIR`
  - Replay local bindings in order and recurse into the final expression with the same seed
- `EinsteinIR`
  - Treat clause values pointwise: multiply each differentiated clause body by the incoming output seed aligned to clause indices, then recurse through the clause value
- `FunctionCallIR`
  - No name checks
  - Resolve the callee binding
  - Build a primal substitution map `param -> arg`
  - Recurse into the callee body with the incoming seed applied to the callee result
  - For block-bodied callees, replay local lets and recurse on the final expression only
  - Do not scalarize tensor call outputs

### Quotient expansion
In [src/einlang/passes/autodiff/_expand.py](/Users/user/Documents/einlang/src/einlang/passes/autodiff/_expand.py):

- For scalar denominators with scalar numerators, keep the current scalar symbolic path.
- For any denominator that is:
  - a tensor binding, or
  - a scalar alias of a tensor slice,
  use `build_seeded_pullback(...)` instead of `JacobianVisitor` directly.
- Alias rule must be generic:
  - detect `den = T[idx...]`
  - compute root pullback wrt `T`
  - project back to `idx...`
- Remove the current experimental `_bound_tensor_call_output_element_jacobian` flow and related tensor-call shortcut branches once the new builder covers the existing passing cases.

### Backend cleanup
- Backend quotient execution remains allowed only as a compatibility bridge for true top-level quotient placeholders.
- Remove correctness reliance on call-name fast paths in [src/einlang/backends/numpy_core.py](/Users/user/Documents/einlang/src/einlang/backends/numpy_core.py). They may remain only as optional optimizations after the generic compiler-generated IR is correct and tested.
- Do not change the recurrence engine again as part of this rewrite, except to preserve the currently passing recurrence/RNN/value-iteration behavior.

## Test Plan
The implementation is complete only when all of these pass with no skips/xfails:

- [tests/unit/test_autodiff_pass.py](/Users/user/Documents/einlang/tests/unit/test_autodiff_pass.py)
  - `test_mnist_conv_pool_chain_quotients_regression`
  - `test_mnist_train_autodiff_ops_small`
  - `test_mnist_main_differentiable_ops_small`
- [tests/unit/test_quotient_golden.py](/Users/user/Documents/einlang/tests/unit/test_quotient_golden.py)
  - all reduction/loss/vector quotient cases, especially `reduce_sum`, `reduce_mean`, `reduce_l1`, `reduce_l2`, `reduce_sum_square`, `reduce_log_sum`, `reduce_log_sum_exp`, `mse_loss`, `mae_loss`, `huber_loss`, `binary_cross_entropy`, `cosine_similarity`
- [tests/unit/test_print_at_golden.py](/Users/user/Documents/einlang/tests/unit/test_print_at_golden.py)
  - `reduce_sum`, `reduce_mean`, `reduce_log_sum`, `reduce_log_sum_exp`, `mse_loss`, `mae_loss`, `huber_loss`, `binary_cross_entropy`, `cosine_similarity`
- [tests/unit/test_autodiff_stdlib_ml_catalog.py](/Users/user/Documents/einlang/tests/unit/test_autodiff_stdlib_ml_catalog.py)
  - unary generic batches
- Recurrence safety slice:
  - [tests/unit/test_statements.py](/Users/user/Documents/einlang/tests/unit/test_statements.py) tuple recurrence tests
  - [tests/stdlib/ml_ops/test_ml_rnn.py](/Users/user/Documents/einlang/tests/stdlib/ml_ops/test_ml_rnn.py) `test_rnn_basic`
  - [tests/stdlib/test_numerics.py](/Users/user/Documents/einlang/tests/stdlib/test_numerics.py) value iteration test

Add two new focused tests during implementation:
- scalar alias of tensor slice:
  `let e = W[i,j]; let g = @loss / @e;` must match `(@loss / @W)[i,j]`
- tensor call under reduction:
  `let y = f(x); let loss = sum(y); let dx = @loss / @x;`
  must equal `@y / @x` contracted with `ones(shape(y))`

## Assumptions
- No public syntax changes.
- No name-based specialization is acceptable for correctness.
- Julia alignment here means adopting the seeded-pullback/VJP model for tensor outputs, not reproducing ChainRules APIs verbatim.
- The recurrence backend should stay functionally as-is unless a change is required to preserve currently passing recurrence tests.
