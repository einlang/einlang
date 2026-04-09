# Autodiff Rewrite Design: JVP, VJP, and Lazy Jacobians

**Status:** Proposed replacement for the current autodiff implementation.

**Intent:** Delete the current autodiff implementation under `src/einlang/passes/autodiff/` and replace it with a much smaller system whose only AD primitives are:

- `jvp(expr, tangents)` for forward-mode linearization
- `vjp(expr, cotangent)` for reverse-mode pullback

Everything else is built from those two primitives.

In particular:

- `@expr` is no longer expanded into a compiler-generated differential program.
- `@y / @x` is no longer handled by a direct Jacobian builder or by backend quotient special-casing.
- `@y / @x` becomes a **lazy Jacobian tensor** value, backed by `jvp` or `vjp`.

This document is the design for the rewrite. It does **not** describe the current implementation in `docs/AUTODIFF_DESIGN.md` and related files.

---

## 1. Why rewrite

The current autodiff stack has three overlapping mechanisms:

1. Forward differential expansion into IR.
2. Direct symbolic Jacobian construction for quotient expressions.
3. Backend runtime special-casing for quotient slots.

That has a few bad consequences:

- operator rules are duplicated across forward, pullback, and Jacobian paths
- tensor Jacobians have a separate implementation from scalar differentiation
- quotient correctness depends on pass-time rewrites and backend seed orchestration
- the generated IR is large and hard to reason about
- it is difficult to add or fix one derivative rule without touching several files

The rewrite removes that split-brain design.

---

## 2. Goals

- Delete the current autodiff implementation files and replace them with a new, smaller core.
- Make `jvp` and `vjp` the only AD primitives.
- Represent `@y / @x` as a lazy Jacobian tensor with shape `shape(y) + shape(x)`.
- Use the same local derivative rule definitions for:
  - standalone tangents
  - gradients
  - Jacobian entries
  - full Jacobian materialization
- Remove compiler-generated diff blocks.
- Remove direct Jacobian-construction code paths.
- Remove backend quotient seeding special cases.
- Keep the surface syntax stable: users still write `@x` and `@y / @x`.

---

## 3. Non-goals for the first rewrite

- Preserving the current internal autodiff module layout.
- Preserving the current IR expansion strategy.
- Preserving symbolic `print(@y)` output exactly as it works today.
- Supporting every current autodiff feature in the first landing if that would force us to reintroduce the old architecture.

The rewrite should prefer a clean core plus staged feature reintroduction over carrying forward the current complexity.

---

## 4. Core design principles

### 4.1 One derivative rule source

Each differentiable operation gets one implementation contract:

- how to compute its `jvp`
- how to compute its `vjp`

There is no third “Jacobian rule” path.

### 4.2 Jacobians are views over linear maps

A Jacobian is not a separately-authored tensor program.

It is a lazy tensor view over the linear map exposed by `jvp` or `vjp`:

- columns from repeated `jvp` with basis tangents
- rows from repeated `vjp` with basis cotangents

### 4.3 Differentiate high-level IR, not lowered loops

The new AD engine should operate on high-level IR before Einstein lowering and recurrence lowering.

That keeps differentiation aligned with the source semantics:

- function calls are still function calls
- Einstein expressions are still Einstein expressions
- control flow is still structured control flow

Lowering remains the job of the existing lowering passes.

### 4.4 The pass should coordinate, not synthesize derivative programs

`AutodiffPass` should:

- collect autodiff requests
- snapshot the high-level binding graph needed to answer them
- rewrite surface `@...` syntax into runtime slot placeholders

It should not generate a separate derivative IR program.

---

## 5. Semantics after the rewrite

### 5.1 `@expr`

Internally, `@expr` means: “the tangent of `expr` under an explicit tangent environment.”

There is no ambient hidden differential program.

At a user-facing top-level binding, `@x` is lowered to a runtime autodiff request that asks for the tangent of `x` with the identity tangent seed for `x`.

That means:

- scalar `x`: `@x` materializes to `1`
- tensor `x`: `@x` materializes to `ones_like(x)`

This is intentionally simple and mechanical. If later we want richer symbolic differential printing, that should be layered on top of the new JVP engine, not by restoring the old diff-expansion path.

### 5.2 `@y / @x`

`@y / @x` becomes a runtime `LazyJacobianTensor(y, x)`.

Semantics:

- shape: output axes first, then input axes
- dtype: derivative dtype corresponding to `y` and `x`
- independence: if `y` does not depend on `x`, the lazy tensor behaves as zero

Examples:

- scalar `y`, scalar `x` -> scalar
- vector `y[i]`, scalar `x` -> vector
- scalar `y`, tensor `x[i,j]` -> tensor with shape `[i,j]`
- tensor `y[a,b]`, tensor `x[i,j]` -> rank-4 tensor with shape `[a,b,i,j]`

### 5.3 Lazy behavior

The lazy Jacobian must support:

- full materialization on demand
- single-entry access
- rectangular indexing and slicing when possible

The implementation should compute only the requested part when indexing is specific enough. Full materialization is a fallback, not the default.

---

## 6. Pipeline position

The rewritten `AutodiffPass` should stay **before** `EinsteinLoweringPass`.

Recommended order:

```text
... -> ShapeAnalysis -> TypeInference -> PreAutodiffPruning
    -> AutodiffPass
    -> PostAutodiffPruning
    -> AutodiffLeakCheck
    -> EinsteinLowering
    -> RecurrenceOrder
    -> validation passes
```

Why:

- the pass only needs typed, high-level IR
- the new runtime engine should work from a snapshot of the high-level binding graph
- later passes should see no `DifferentialIR` or derivative quotient syntax

`AutodiffPass` becomes a collector and rewriter, not a differentiating compiler.

---

## 7. New architecture

### 7.1 Compiler-side

`AutodiffPass` performs four jobs:

1. Collect autodiff requests from the typed high-level program.
2. Snapshot the relevant high-level binding graph into analysis.
3. Rewrite `@...` syntax into plain IR placeholders that survive the rest of the pipeline.
4. Remove all autodiff-only syntax from the post-pass IR.

### 7.2 Runtime-side

After the primal program executes, the backend resolves autodiff placeholders by consulting the stored analysis snapshot and the final primal environment.

That runtime resolver owns:

- `jvp`
- `vjp`
- `LazyJacobianTensor`

---

## 8. Analysis data model

The analysis stored on `TyCtxt` should look roughly like this:

```python
class AutodiffRequest:
    kind: Literal["tangent", "jacobian"]
    slot_defid: DefId
    target_defid: DefId | None
    numerator_defid: DefId | None
    denominator_defid: DefId | None


class AutodiffAnalysis:
    requests: dict[DefId, AutodiffRequest]
    graph_program: ProgramIR
    binding_map: dict[DefId, BindingIR]
    function_map: dict[DefId, FunctionValueIR]
```

Important points:

- `graph_program` is a cloned high-level program snapshot from before lowering.
- `binding_map` is the high-level binding graph used for JVP/VJP traversal.
- `requests` maps runtime slot bindings to the autodiff work the backend must answer.

This replaces the current `diff_block`, differential buffer maps, and pending quotient slot maps.

---

## 9. IR rewrite strategy

The pass should stop expanding derivatives into executable derivative IR.

Instead, it rewrites autodiff syntax into placeholder values that the backend recognizes after primal execution.

Two workable options:

### Option A: dedicated IR nodes

Add small runtime-only IR nodes such as:

- `AutodiffSlotIR(slot_defid)`
- `JacobianRequestIR(numerator_defid, denominator_defid)`

This is the cleanest contract but requires IR/visitor/backend additions.

### Option B: backend builtin placeholders

Rewrite to internal builtins such as:

- `__autodiff_tangent_slot(slot_id, target_id)`
- `__autodiff_jacobian_slot(slot_id, numerator_id, denominator_id)`

This avoids new general IR node kinds.

**Recommendation:** start with backend builtins because the rewrite already removes a lot of code, and builtins keep the compiler-side delta smaller.

---

## 10. JVP engine

`jvp(expr, tangent_env)` computes the tangent of `expr`.

`tangent_env` maps leaf `DefId`s to tangent values.

The JVP engine operates on the high-level binding graph snapshot and the already-computed primal environment.

### 10.1 API

```python
def jvp_expr(expr: ExpressionIR, tangents: dict[DefId, Any], ctx: AutodiffRuntimeContext) -> Any
def jvp_defid(target: DefId, tangents: dict[DefId, Any], ctx: AutodiffRuntimeContext) -> Any
```

### 10.2 Required behavior

- identifiers read tangents from the environment or recursively from their defining bindings
- literals have zero tangent
- blocks and local lets create scoped tangent environments
- `if` follows the executed primal branch
- function calls JVP through the callee body
- Einstein expressions differentiate clause values in place
- reductions preserve reduction structure

### 10.3 Einstein JVP rule

For an Einstein clause, JVP differentiates the clause value and keeps:

- the same output indices
- the same reduction indices
- the same where-clause

That gives a new Einstein value with the same structural shape as the primal clause.

This is much simpler than building a separate Jacobian-specific Einstein path.

---

## 11. VJP engine

`vjp(expr, cotangent)` computes the pullback from an output cotangent to leaf cotangents.

### 11.1 API

```python
def vjp_expr(expr: ExpressionIR, cotangent: Any, wrt: DefId, ctx: AutodiffRuntimeContext) -> Any
def vjp_defid(target: DefId, cotangent: Any, wrt: DefId, ctx: AutodiffRuntimeContext) -> Any
```

### 11.2 Required behavior

- reverse-accumulate through bindings in dependency order
- share the same primitive rule registry as JVP
- preserve executed control-flow choices from the primal run
- allow structured tensor pullbacks for Einstein expressions and reductions

### 11.3 Einstein VJP rule

For Einstein clauses, VJP distributes the cotangent into the clause value and preserves:

- primal output indexing
- reduction structure
- where-clause filtering

Conceptually:

- JVP gives us tangent pushforward
- VJP gives us cotangent pullback

Both are derived from the same clause semantics, not from separate Jacobian code.

---

## 12. Primitive rule registry

The rewrite should introduce a single primitive rule table, not separate forward and reverse files that drift apart.

Rough shape:

```python
class PrimitiveAdRule:
    def jvp(self, primal_inputs, tangent_inputs, meta) -> Any: ...
    def vjp(self, primal_inputs, cotangent_output, meta) -> list[Any]: ...
```

This registry covers:

- arithmetic operators
- unary math builtins
- reductions
- indexing and casts
- tensor/Einstein constructs
- selected stdlib functions when they should override body-based differentiation

That registry becomes the one place to add or fix derivative behavior.

---

## 13. Lazy Jacobian tensor

`LazyJacobianTensor` is the user-visible result of `@y / @x`.

### 13.1 Runtime shape contract

- `shape = shape(y) + shape(x)`
- output axes come first
- indexing semantics match that shape contract

### 13.2 Materialization strategy

The lazy tensor chooses between JVP and VJP:

- if `size(x) <= size(y)`, materialize by JVP over basis tangents of `x`
- otherwise materialize by VJP over basis cotangents of `y`

That gives a simple default heuristic:

- fewer seeds
- same core APIs
- no direct Jacobian builder

### 13.3 Partial access

When the user indexes into a Jacobian, the backend should compute only what is needed when practical:

- one Jacobian entry -> one basis JVP or one basis VJP
- one row -> one VJP with a row basis cotangent
- one column -> one JVP with a column basis tangent

If the surrounding operation demands a generic ndarray and partial evaluation is awkward, the backend can materialize eagerly as a fallback.

### 13.4 Backend interop

The NumPy backend should teach:

- rectangular access on `LazyJacobianTensor`
- conversion to `np.ndarray` via full materialization

Phase 1 does not need every backend path to understand laziness natively; automatic materialization is acceptable outside direct indexing/output/printing paths.

---

## 14. Function calls and custom derivative rules

The default function-call rule is structural:

- JVP a function by JVP-ing its body with parameter tangents
- VJP a function by VJP-ing its body with parameter cotangents

Custom derivative support, if kept, should be reintroduced as **explicit JVP/VJP overrides**, not as custom expanded derivative bodies interleaved with the main autodiff implementation.

That means the old `@fn` machinery should not be ported directly. If we keep a user-extensible derivative feature, its internal contract should be:

- “this function has a custom JVP”
- or “this function has a custom VJP”

not “splice this differential-expression body into the compiler.”

---

## 15. Control flow, reductions, and recurrence

### 15.1 Control flow

`if` and other control-flow nodes should use the executed primal branch.

This is standard traced AD behavior and avoids symbolic branch merging in the first rewrite.

### 15.2 Reductions

Reductions should be handled as primitive AD rules, not by quotient-specific logic.

That covers:

- `sum`
- `prod`
- `max`
- `min`

The same reduction rules then serve:

- JVP of reductions
- VJP of reductions
- Jacobians of reductions

### 15.3 Recurrence

Recurrence support should be staged.

Recommendation:

- do not block the rewrite on full recurrence AD
- keep recurrence explicitly listed as a phase-2 feature if needed

It is better to land a clean JVP/VJP core for expressions, functions, Einstein, and reductions first than to carry recurrence complexity into the foundation layer.

---

## 16. File plan

The current autodiff package should be removed and replaced, not incrementally patched.

### 16.1 Delete or fully replace

- `src/einlang/passes/autodiff/__init__.py`
- `src/einlang/passes/autodiff/_callee.py`
- `src/einlang/passes/autodiff/_cleanup.py`
- `src/einlang/passes/autodiff/_core.py`
- `src/einlang/passes/autodiff/_einstein_tensor_vjp.py`
- `src/einlang/passes/autodiff/_expand.py`
- `src/einlang/passes/autodiff/_expr.py`
- `src/einlang/passes/autodiff/_forward.py`
- `src/einlang/passes/autodiff/_graph.py`
- `src/einlang/passes/autodiff/_internal_iv_names.py`
- `src/einlang/passes/autodiff/_jacobian.py`
- `src/einlang/passes/autodiff/_print.py`
- `src/einlang/passes/autodiff/_pullback.py`
- `src/einlang/passes/autodiff/_tensor.py`

### 16.2 Replace with

- `src/einlang/passes/autodiff/__init__.py`
- `src/einlang/passes/autodiff/shared.py`
- `src/einlang/passes/autodiff/jvp.py`
- `src/einlang/passes/autodiff/vjp.py`
- `src/einlang/passes/autodiff/jacobian.py`
- `src/einlang/passes/autodiff/requests.py`

### 16.3 Other touch points

- `src/einlang/backends/numpy_core.py`
- `src/einlang/backends/numpy_expressions_mixin.py`
- `src/einlang/compiler/driver.py`
- `src/einlang/passes/autodiff_leak_check.py`
- autodiff unit tests

---

## 17. Migration plan

### Phase 0: land this design doc

- agree on the architecture
- explicitly stop adding logic to the old autodiff files

### Phase 1: skeleton replacement

- delete the current autodiff implementation files
- land a minimal `AutodiffPass` that:
  - collects requests
  - snapshots the high-level graph
  - rewrites `@...` to backend placeholders
- land the runtime request resolver and `LazyJacobianTensor`
- support scalar arithmetic first

### Phase 2: tensor expressions and reductions

- indexing
- casts
- array literals/comprehensions as needed
- reduction rules

### Phase 3: Einstein expressions

- JVP rule for Einstein clauses
- VJP rule for Einstein clauses
- quotient tests for tensor outputs and tensor inputs

### Phase 4: function calls and stdlib coverage

- body-based JVP/VJP for user functions
- builtin overrides where needed
- reintroduce custom derivative hooks only as JVP/VJP overrides

### Phase 5: recurrence and advanced cases

- recurrence, if still needed after the core is stable

---

## 18. Testing plan

The rewrite should add tests around the new contracts, not the old IR shape.

### 18.1 Engine consistency

- scalar JVP against finite differences
- scalar VJP against finite differences
- Jacobian-by-JVP matches Jacobian-by-VJP

### 18.2 Surface syntax

- `@x`
- `@y / @x`
- independence gives zero
- tensor quotient shapes are `shape(y) + shape(x)`

### 18.3 Expression coverage

- arithmetic
- unary builtins
- control flow
- reductions
- user functions
- Einstein expressions

### 18.4 Backend behavior

- lazy Jacobian indexing computes correct entries
- full materialization chooses JVP or VJP correctly
- no autodiff-only IR leaks past `AutodiffPass`

---

## 19. Acceptance criteria

The rewrite is successful when all of the following are true:

- there is no compiler-generated diff block
- there is no direct Jacobian visitor
- there is no backend quotient seed orchestration
- `@y / @x` is implemented as a lazy Jacobian backed only by JVP/VJP
- each differentiable primitive has one rule definition source
- adding a new derivative rule does not require touching three separate AD paths

---

## 20. Recommendation

Do the rewrite as a clean replacement, not as another layer on top of the current autodiff code.

The sequence should be:

1. freeze the old implementation
2. delete the old autodiff package
3. land a small request-collection pass
4. build `jvp`
5. build `vjp`
6. make `@y / @x` a lazy Jacobian tensor

That gives us one understandable AD story instead of three partially-overlapping ones.
