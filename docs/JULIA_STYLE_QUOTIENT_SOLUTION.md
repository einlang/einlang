# Julia-Style Quotient Solution

This note captures the systematic fix strategy for the remaining quotient/autodiff edge cases by
following the same mental model used by Julia/Zygote/ChainRules:

- compute the **full cotangent on storage**
- then **project aliases with the same indices**
- avoid ad hoc masked sums that accidentally mix storage-wide and elementwise semantics

It is written as the reference for the current quotient/autodiff cleanup work.

## 1. Repo references

The local reference implementations already encode the desired behavior:

- [examples/julia_style_slice_alias_vjp_numpy.py](/Users/user/Documents/einlang/examples/julia_style_slice_alias_vjp_numpy.py)
- [examples/julia_style_conv_pullback_numpy.py](/Users/user/Documents/einlang/examples/julia_style_conv_pullback_numpy.py)
- [examples/julia_style_conv_vjp_numpy.py](/Users/user/Documents/einlang/examples/julia_style_conv_vjp_numpy.py)

The key takeaway from them is:

`getindex` / slice alias:
- `e = W[i0, j0]`
- `dL/de = (dL/dW)[i0, j0]`

Conv / pooling:
- represent the forward as sparse index triples
- run one reverse accumulation over those triples
- do not rebuild weight/input quotients via nested masked Einstein clauses when a direct pullback is available

## 2. Quotient semantics to preserve

For the current Einlang backend, tensor quotients are effectively using **pullback-style** semantics
rather than materializing the full Jacobian in every case.

That means:

- scalar alias quotients should be implemented as `pullback(root_storage)[alias_indices]`
- tensor reductions should use a seeded pullback on the tensor root and return a tensor of the input shape
- recurrence tests must not assume full per-output Jacobians if the runtime is intentionally computing the
  pullback for an all-ones cotangent

This already matches many passing tests after the recent fixes.

## 2.1 Backend boundary

Target architecture:

- the **autodiff pass** must expand every `@expr` and every `@num / @den` into plain IR
- the **NumPy backend** should execute only plain IR, lowered Einstein/reduction IR, and ordinary arrays
- the backend should **not** need to understand `DifferentialIR` or deferred quotient semantics

Current code still violates that boundary in a few places:

- [src/einlang/backends/numpy_core.py](/Users/user/Documents/einlang/src/einlang/backends/numpy_core.py)
  still has pending differential/quotient slot machinery
- [src/einlang/backends/numpy_expressions_mixin.py](/Users/user/Documents/einlang/src/einlang/backends/numpy_expressions_mixin.py)
  still has `visit_differential`
- [src/einlang/passes/autodiff/__init__.py](/Users/user/Documents/einlang/src/einlang/passes/autodiff/__init__.py)
  still records backend-facing autodiff analysis (`diff_block`, differential maps)

So the long-term fix is not “teach NumPy more autodiff,” but:

1. make quotient expansion in the pass produce the final plain IR
2. keep the Julia-style pullback logic in the pass layer
3. delete backend differential/quotient semantics once the IR generation is complete

## 3. What was fixed already

Recent fixes that align with the Julia model:

- Scalar alias quotient path now prefers a scalar Jacobian result when available, instead of always forcing
  a tensor pullback + projection.
- Nested lowered-Einstein rectangular accesses no longer eagerly scalarize in the wrong place when evaluating
  quotient subexpressions under an explicit `[...]` access.
- Several stdlib ML helpers were simplified from multi-binding blocks into direct expressions so the quotient
  path sees a cleaner pullback surface.
- `reduce_l2`, `reduce_log_sum`, `reduce_log_sum_exp`, `huber_loss`, `binary_cross_entropy`, and
  `cosine_similarity` now pass the quotient/golden checks under the current pullback semantics.

## 4. Remaining xfail buckets

At the time of writing, the remaining explicit `xfail`s are:

### 4.1 Softmax-family quotient execution

Files:

- [tests/unit/test_quotient_golden.py](/Users/user/Documents/einlang/tests/unit/test_quotient_golden.py)

Cases:

- `softmax`
- `log_softmax`
- `softmax_quotient`

Current failure shape:

- nested Einstein subexpressions inside the softmax quotient path collapse to lower-rank arrays
- later `RectangularAccessIR(...)[i, j]` re-applies two indices to a 1D temporary
- current runtime fast-path only handles the top-level `std::ml::softmax` / `log_softmax` call shape, not the
  lowered inlined quotient tree

Julia-style solution:

1. Treat softmax/log-softmax as **custom pullbacks**, not generic expanded quotient trees.
2. For softmax with cotangent `dy`, compute
   `dx = y * (dy - sum(dy * y))`.
3. For log-softmax with cotangent `dy`, compute
   `dx = dy - softmax(x) * sum(dy)`.
4. For `@y / @x` on a tensor `y`, define the seed explicitly:
   current quotient tests are effectively asking for the pullback under an all-ones cotangent on `y`.
5. Implement this either:
   - in the autodiff pass as a custom `@fn`/pullback rule for stdlib softmax/log-softmax, or
   - in the runtime deferred quotient path as a dedicated fast-path on the numerator binding.

Important semantic note:

- `softmax` under an all-ones cotangent gives zero.
- `log_softmax` under an all-ones cotangent gives `1 - n * softmax(x)`, not zero.

So if the tests keep expecting zero for `log_softmax`, the test is encoding an old shortcut rather than the
Julia-style pullback semantics.

### 4.2 MNIST SGD recurrence quotient

File:

- [tests/unit/test_autodiff_pass.py](/Users/user/Documents/einlang/tests/unit/test_autodiff_pass.py)

Case:

- `test_mnist_train_autodiff_ops_small`

Current mismatch:

- the recurrence body uses `g = @loss_b / @w_ij`
- the test expects a per-logit/per-weight Jacobian-style update
- the runtime is currently applying pullback-style quotient semantics, so the update trajectory differs

Julia-style solution:

1. Decide whether recurrence training tests should use:
   - a scalar pullback quotient, or
   - a full Jacobian object.
2. Keep the tests and runtime consistent with that choice.
3. If the desired behavior is true SGD on a scalar loss, the pullback semantics are the right one.
4. If the desired behavior is “differentiate each logit independently,” that should be a separate test and
   should not be expressed through the same scalar quotient.

### 4.3 Generic masked conv-weight quotient

File:

- [tests/unit/test_autodiff_pass.py](/Users/user/Documents/einlang/tests/unit/test_autodiff_pass.py)

Case:

- `test_conv_relu_pool_direct_tensor_weight_quotient_matches_numpy`

Current failure:

- the user-written conv body relies on `if in-bounds { x[...] * w0[...] } else { 0.0 }`
- the scalar quotient path eventually evaluates an out-of-bounds `x[...]` before the mask fully suppresses it
- this is exactly the class of problem the Julia-style sparse conv VJP avoids

Julia-style solution:

1. Lower masked conv quotients to a sparse pullback like the local NumPy Julia reference.
2. Reuse the same sparse index-triple accumulation for:
   - `dx`
   - `dw`
   - `db`
3. Do not rely on generic nested Einstein execution to preserve masked OOB safety for this path.

## 5. Recommended implementation order

1. Add a dedicated softmax/log-softmax pullback rule using the Julia formulas above.
2. Decide and document one tensor-quotient semantic contract:
   pullback-style all-ones cotangent vs full Jacobian.
3. Rewrite the MNIST recurrence test to match that contract.
4. Add a sparse conv-weight quotient/VJP path rather than pushing more masked OOB logic into the generic runtime.

## 6. Debugging checklist

When a quotient path disagrees with the Julia reference:

1. Check whether the quotient should be a scalar pullback or a full Jacobian.
2. If there is an alias `x = W[i, j]`, compute the pullback on `W` first, then project `[i, j]`.
3. If a nested `LoweredEinsteinIR` appears under `RectangularAccessIR`, verify that its result rank matches
   the number of indices being applied.
4. For conv/pool chains, compare against the sparse NumPy Julia references before changing generic Einstein code.

## 7. Current status

The repo already passes the broad quotient/golden/autodiff slices except for the explicit xfail cases above.
Those xfails are now narrow enough that the remaining work should be done by adding **dedicated Julia-style
pullback rules**, not by more generic quotient tree patching.
