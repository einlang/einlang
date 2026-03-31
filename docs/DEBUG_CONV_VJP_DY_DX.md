# Debug notes: 1D conv `@y/@x` (VJP) vs NumPy reference

This document records findings from debugging the failing Jacobian / reverse-mode test for rank-1 convolution autodiff. It is a **working log**, not a final postmortem of a fixed bug.

## Problem

- **Test:** `tests/unit/test_autodiff_pass.py::TestAutodiffPass::test_conv_autodiff_jacobian_rank1_matches_calculus`
- **Failure:** directional max error on **`dy_dx`**; Einlang output diverges from NumPy reference.

### Observed vs reference (`dy_dx`)

| | Value |
|---|--------|
| **Actual** | `[[[3.0, 1.5, 0.0, 0.0]]]` |
| **Reference** | `[[[1.0, 1.5, 1.5, 0.5]]]` |

Symptoms:

- First input position is **too large** (3 vs 1).
- Later positions are **too small or zero** (0, 0 vs 1.5, 0.5).

## Related outputs (runtime evidence)

Instrumented NumPy backend stored final gradient arrays (NDJSON log, session `fc3e74`, hypothesis tag `D`):

| Binding | Shape | Observed | Notes vs analytic ref |
|---------|-------|----------|------------------------|
| `dy_dx` | `(1, 1, 4)` | `[[[3.0, 1.5, 0.0, 0.0]]]` | Wrong pattern |
| `dy_dw` | `(1, 1, 2)` | `[[[18.0, 27.0]]]` | Ref ≈ `[[[6.0, 9.0]]]` → **~3× too large** |
| `dy_db` | `(1,)` | `[3.0]` | **Matches** reference |

**Implication:** Bias pullback is consistent with a correct global loss seed; the bug is likely in the chain that differentiates **conv / pad / masked access / Einstein sum**, not in a uniform mis-scaling of the entire VJP.

The factor **3** matches the **output spatial length** `L_out = 3` in the minimal failing setup (strong hint of **per-output-position over-counting** or **incorrect fold of output-loop dimensions**).

## Autodiff path

- **`@y/@x` expansion** (`src/einlang/passes/autodiff/_expand.py`): tensor numerator / tensor denominator uses **`build_seeded_pullback`** with **`build_default_seed`** (often `ones`-like tensor matching `y` rank).
- **`_QUOTIENT_PENDING` / NumPy deferred JVP path:** Logs showed many **“DIV binding not matched as quotient”** entries for unrelated bindings; the evidence **does not** support the hypothesis that **`dy_dx`** is produced by that JVP shortcut instead of **`PullbackBuilder`**.

**Conclusion from logs:** **`dy_dx` is driven by the VJP (`PullbackBuilder`) path**, not the quotient-pending JVP path.

## IR and lowering (qualitative)

- Dumps such as `/tmp/conv_dx_ir.sexpr` and `/tmp/conv_dx_lowered.sexpr` (local session artifacts) showed:
  - Padded-input and conv-sum intermediates involving **`IfExpressionIR`** for bounds.
  - Some annotations suggested **input-shaped** `(1, 1, 4)` where **output-shaped** `(1, 1, 3)` would be natural for conv output — but this is **not** the full story.

### Scatter / shape inference (instrumentation)

- **`_scatter_seed`** logs: **`binding_shape_info: None`** for key identifiers (`conv_sum`, `Xp`, `X`).
- **`_fresh_axes_for_expr`** logs: **dynamic** bounds via **`RectangularAccessIR`** on `.shape[...]`.

**Conclusion:** Static **`shape_info` stamps on lowered nodes** are an incomplete explanation; **dynamic** resolution is in play for loop bounds.

## Execution / lowering logs (H-series instrumentation)

From `numpy_expressions_mixin.py` / `lowered_execution.py` (hypotheses `H1`–`H5`):

- **`evaluate_lowered_reduction`:** mixes of **`parallel_shape=[1, 1, 3]`** and **`[1, 1, 4]`**, often **`force_scalar_reduction=True`**, **`has_if=True`**.
- **Vectorized path** often **disabled** (`can_try_vectorized: False`).

These align with **scalar nested loops** over masked / padded conv, not with a single smoking gun.

## Code areas that were inspected (hypotheses)

1. **`PullbackBuilder.visit_einstein`** (`src/einlang/passes/autodiff/_pullback.py`)
   - Uses **`seed_rank`** and **`_indexed_seed`** so the clause seed can be **`seed[batch, co, i]`** (full output indexing).
   - Outer clause index variables are folded via **`_pullback_fold_einstein_clause_loops`**, possibly merging with inner **`ReductionExpressionIR`** sums.
   - **Risk:** mismatch between **intended “sum over output positions once”** and **actual merged reduction** could yield **× `L_out`** on weights and wrong **scatter** pattern on `x`.

2. **`visit_if_expression`** (same file)
   - Pullback uses **`cond_f`** / **else mask** and **adds** branch pullbacks.
   - **`_prune_const_ifs_replayed`** (`src/einlang/passes/autodiff/_expr.py`) can **statically prune** dispatch like **`rank == 1`** when **`len(strides)`** is constant — changing whether **`visit_if`** runs for that guard. This affects **which IR** is differentiated, but **by itself** does not obviously explain **exactly ×3**.

3. **`visit_function_call` → callee body**
   - **`body_expr = _prune_const_ifs_replayed(fv.body, prune_bindings) or fv.body`** before **`PullbackBuilder`** visits the callee.

4. **NumPy `visit_if_expression`** (`src/einlang/backends/numpy_expressions_mixin.py`)
   - **`np.where`** for ndarray conditions was explored; **scalar `bool`** **`np.where`** was **reverted** after **OOB** from eager branch evaluation (padding access).

## What is not yet proven

- The **exact IR node / lowered reduction** where adjoint mass is **triple-counted** for **`dy_dw`** or **dropped** for **`dy_dx[k]`** for `k ≥ 2` has **not** been proven with a **line-precise** runtime trace.
- Recommended follow-up: **small, targeted logs** in **`visit_einstein`**, **`_pullback_fold_einstein_clause_loops`**, or **evaluation of specific pullback subexpressions** for one failing example.

## Reproduction and logs

- **Failing test:**  
  `python3 -m pytest tests/unit/test_autodiff_pass.py::TestAutodiffPass::test_conv_autodiff_jacobian_rank1_matches_calculus -x --tb=short`
- **Debug NDJSON path (this session):**  
  `/Users/user/Documents/einlang/.cursor/debug-fc3e74.log`  
  **Session ID:** `fc3e74`
- Before each instrumented run, **clear only this file** (do not delete other sessions’ logs).

## References in repo

- Quotient expansion: `src/einlang/passes/autodiff/_expand.py`
- VJP builder: `src/einlang/passes/autodiff/_pullback.py`
- Const-if pruning: `src/einlang/passes/autodiff/_expr.py` (`_prune_const_ifs_replayed`, `_eval_const_expr`)
- NumPy execution / `if`: `src/einlang/backends/numpy_expressions_mixin.py`, `src/einlang/runtime/compute/lowered_execution.py`
- Einstein VJP helpers: `src/einlang/passes/autodiff/_einstein_tensor_vjp.py`

## Transcript

Parent conversation transcript (full JSONL, includes tool history):  
[Cursor agent transcript](file:///Users/user/.cursor/projects/Users-user-Documents-einlang/agent-transcripts/fc3e740c-3525-4928-8f55-bc5a0628b86f/fc3e740c-3525-4928-8f55-bc5a0628b86f.jsonl)

---

*Last updated from debug session summaries and runtime log analysis; amend when root cause is fixed and verified.*
