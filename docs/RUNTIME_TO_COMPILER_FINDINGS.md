# Runtime-to-Compiler Move: Findings and Changes

Scope: `src/einlang/backends/` (numpy_einstein.py, numpy_expressions.py, numpy_core.py, numpy_helpers.py, numpy_arrow_pipeline.py).

## Changes Made (small, safe moves)

| File | Change |
|------|--------|
| **src/einlang/passes/ir_validation.py** | **Added** compile-time validation in `visit_lowered_einstein_clause`: if a lowered loop has `iterable is None`, report a compiler error ("Lowered loop has no iterable; cannot execute"). Backend can rely on this for lowered IR. |
| **src/einlang/backends/numpy_einstein.py** | **Documentation** in `_extract_loop_range`: noted that IR validation now fails compilation when iterable is missing; runtime check kept as safeguard. |

## Tests

- **Unit tests**: `python3 -m pytest tests/unit -n auto --tb=short -q` → **493 passed, 1 skipped.**

---

## Documented Findings (not moved; large/invasive or blocked)

1. **DefId required for IdentifierIR / IndexVarIR**
   - **Where**: `numpy_expressions.py` `visit_identifier` / `visit_index_var` raise at runtime if `expr.defid is None`.
   - **Attempted**: Require defid in IR validation and remove runtime check.
   - **Blocked**: Stdlib and cross-module identifiers (e.g. `pi`, `sqrt`, `log` in `stdlib/math/constants.ein`) are left with `defid is None` until specialization/execution. Requiring defid at compile time fails many tests (module linking, stdlib). **Recommendation**: Keep runtime check until name resolution guarantees defid for all evaluated identifiers (e.g. after specialization or when excluding Python/stdlib paths).

2. **Reduction loop variable defid and loop_var_ranges**
   - **Where**: `numpy_expressions.py` (einsum/matmul paths) checks `loop_var.defid is None` and returns `None` (fallback path). Similar for `loop_var_ranges` and reduction structure.
   - **Move**: A compiler pass could ensure reduction loop variables always have defid and that `loop_var_ranges` is complete; backend could then trust and avoid fallback checks. **Invasive**: Requires name resolution / range analysis to always set these for reduction expressions; may interact with generic/specialization.

3. **Loop iterable type (RangeIR vs LiteralIR)**
   - **Where**: `_extract_loop_range` and callers check `isinstance(iterable, RangeIR)` / `LiteralIR(range)` and raise if neither.
   - **Move**: Compiler could attach a discriminant or “iterable kind” on the loop so the backend does not need isinstance. **Small**: Could be a follow-up after ensuring all lowered loop iterables are well-typed.

4. **Per-execution lookups that could be compile-time**
   - **Examples**: `reduction_ranges.get(read_defid)`, `_reduction_axes_in_access(backend, indices, reduction_defids)`, `defid_of_var_in_expr` used inside backend hot paths.
   - **Move**: Attach precomputed maps (e.g. reduction_defid → axis index, or “reduction axes for this access”) on LoweredEinsteinIR or clause IR so the backend indexes into arrays instead of recomputing. **Larger**: Requires new IR fields and a pass to fill them.

5. **isinstance/getattr/hasattr in backends**
   - **Where**: Many backend branches use `isinstance(..., BinaryOpIR)`, `getattr(..., "defid", None)`, `hasattr(..., "accept")` for IR shape and resolution.
   - **Move**: Per .cursorrules, “Trust IR node structure” and avoid optional-attr checks for required IR attributes. **Incremental**: Replace defensive getattr/hasattr with direct attribute access where the IR type guarantees the attribute; leave isinstance where dispatch by node type is required (e.g. body is BinaryOpIR vs BlockExpressionIR).

6. **“Variable not found (defid=…)” vs “Variable not found (defid=None)”**
   - **Current**: Two runtime errors: missing defid vs missing env value.
   - **Move**: Once defid is guaranteed for evaluated identifiers (see #1), backend can drop the defid-is-None branch and keep only the env.get_value check.

---

## Summary

- **Done**: One compile-time check (lowered loop must have iterable) in `ir_validation.py`; backend comment in `numpy_einstein.py`.
- **Tests**: All unit tests pass.
- **Deferred**: Defid requirement (blocked by stdlib/module resolution), reduction-structure trust, and compile-time attachment of reduction axes / iterable kind are documented above for follow-up.
