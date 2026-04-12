# Autodiff design

**Status:** Current implementation reference.

**Overview:** [AUTODIFF_HIGHLIGHTS.md](AUTODIFF_HIGHLIGHTS.md)

This document describes the autodiff architecture that matches the code in:

- `src/einlang/passes/autodiff/__init__.py`
- `src/einlang/passes/autodiff/compiler.py`
- `src/einlang/backends/numpy_ir_tensor_runtime.py`
- `src/einlang/backends/numpy_autodiff_core.py`

Older docs that describe compiler-generated diff blocks and dedicated Jacobian visitors are historical only.

## 1. Surface semantics

### 1.1 Numeric requests

Executable autodiff requests are currently centered on **named bindings**.

```rust
let x = 3.0;
let y = x * x;
let dy_dx = @y / @x;
print(dy_dx);   // 6.0
```

- `@x` materialized as a value means the identity tangent seed for `x`.
- `@y / @x` materialized as a value means the derivative or Jacobian of `y` with respect to `x`.

### 1.2 Symbolic print requests

Direct `print(@...)` is a symbolic debugging path:

```rust
print(@x);         // "@x"
print(@y / @x);    // "(@y / @x) · @x"
```

This is intentionally separate from numeric evaluation. If you want a numeric result, bind first:

```rust
let dy_dx = @y / @x;
print(dy_dx);
```

### 1.3 Current limitation

The runtime rewrite currently expects identifier-based executable autodiff requests. When in doubt, bind the expression first:

```rust
let y = x * x;
let dy_dx = @y / @x;
```

That is the reliable user-facing pattern today.

## 2. Compiler architecture

`AutodiffPass` no longer synthesizes a separate derivative IR program.

Instead it:

1. collects autodiff requests from typed high-level IR
2. clones the relevant high-level binding graph into `TyCtxt` analysis
3. rewrites autodiff syntax to runtime intrinsics
4. rewrites direct symbolic prints to symbolic autodiff intrinsics
5. clears autodiff-only IR before later passes run

The pass runs before Einstein lowering, so the stored graph still preserves high-level Einstein structure.

## 3. Runtime intrinsics

The compiler rewrites requests to internal builtins defined in `src/einlang/shared/autodiff_intrinsics.py`:

| Builtin | Meaning |
|--------|---------|
| `__autodiff_tangent` | materialize the identity tangent for a named target |
| `__autodiff_jacobian` | materialize a derivative/Jacobian for `(numerator, denominator)` |
| `__autodiff_symbolic_tangent` | symbolic string for `print(@x)` |
| `__autodiff_symbolic_jacobian` | symbolic string for `print(@y / @x)` |

These are compiler/runtime intrinsics, not user-facing library functions.

## 4. Analysis contract

`AutodiffPass` stores compiled graph facts on `TyCtxt`. The important current fields are:

- `compiled_graph`
- `graph_program`
- `graph_binding_by_defid`
- `graph_function_ir_map`
- `graph_leaf_defids`
- `graph_builtin_requests_by_expr_id`
- `pending_differential_slot_by_defid`
- `pending_quotient_slot_by_defid`

Legacy fields such as `diff_block` and `autodiff_differential_map` may still be present for compatibility, but they are not the primary path for the current runtime implementation.

## 5. Backend contract

The NumPy backend executes the primal program normally. When it encounters an autodiff slot or builtin request, it consults the compiled graph snapshot and resolves the request through `NativeIRAutodiffRuntime`.

That runtime:

- reconstructs a tensor-level view of the high-level IR graph
- evaluates function bodies and Einstein expressions against the current primal environment
- answers tangent requests directly
- answers Jacobian requests with `jacobian(output, wrt)`

## 6. JVP, VJP, and lazy Jacobians

The runtime AD core lives in `src/einlang/backends/numpy_autodiff_core.py`.

- `jvp` provides forward-mode linearization
- `vjp` provides reverse-mode pullback
- `LazyJacobianTensor` exposes `@y / @x` for non-scalar cases

`LazyJacobianTensor` chooses a materialization strategy dynamically:

- JVP when the input space is smaller
- VJP when the output space is smaller

It also supports direct indexing into rows, columns, or single entries without forcing full materialization when a cheaper path exists.

## 7. Einstein and function calls

Because autodiff snapshots the graph before lowering:

- Einstein expressions stay visible as `EinsteinIR`
- user function bodies stay visible as function bodies
- custom diff bodies can be rewritten into the same runtime intrinsics

That keeps the AD story centered on high-level program structure rather than on lowered loops or a separate derivative IR dialect.

## 8. Current user-visible behavior summary

| Source form | Current behavior |
|------------|------------------|
| `let dx = @x;` | numeric identity tangent seed |
| `let dC_dA = @C / @A;` | numeric derivative/Jacobian value |
| `print(@x);` | symbolic tangent string |
| `print(@y / @x);` | symbolic Jacobian relation string |

## 9. Pointers

- [AUTODIFF_VJP_JVP_REWRITE.md](AUTODIFF_VJP_JVP_REWRITE.md): deeper runtime architecture and current status
- [AUTODIFF_PIPELINE.md](AUTODIFF_PIPELINE.md): pass order and handoff details
- [AUTODIFF_OPS.md](AUTODIFF_OPS.md): derivative formulas
- [AUTODIFF_IMPLEMENTATION.md](AUTODIFF_IMPLEMENTATION.md), [AUTODIFF_ALGORITHM.md](AUTODIFF_ALGORITHM.md): archived notes for the retired diff-block implementation
