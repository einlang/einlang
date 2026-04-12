# Autodiff highlights

**Purpose:** Short overview of how autodiff works in Einlang today.

**Current implementation:** The compiler keeps autodiff in the language, but it no longer expands `@...` into a large derivative IR program. `AutodiffPass` snapshots the typed high-level binding graph, rewrites autodiff syntax to internal runtime builtins, and the NumPy backend answers those requests with a JVP/VJP-based runtime.

## Surface model

- Use `@y / @x` for derivatives and Jacobians.
- Use `@x` on a named binding when you want that binding's identity tangent seed.
- Use `print(@x)` or `print(@y / @x)` for symbolic debugging output.

Today the executable path is centered on **named bindings**. In practice, write the value first, then differentiate it:

```rust
let x = 3.0;
let y = x * x;
let dy_dx = @y / @x;
print(dy_dx);   // 6.0
```

If you want a symbolic display of the tangent relation instead of a numeric value:

```rust
let x = 3.0;
let y = x * x;
print(@x);         // "@x"
print(@y / @x);    // "(@y / @x) · @x"
```

## What the compiler does

1. `AutodiffPass` runs on typed, high-level IR before Einstein lowering.
2. It clones the relevant binding graph into analysis data on `TyCtxt`.
3. It rewrites executable autodiff requests to internal builtins such as `__autodiff_tangent` and `__autodiff_jacobian`.
4. It rewrites direct `print(@...)` calls to symbolic autodiff print builtins.
5. It strips `DifferentialIR` and other autodiff-only syntax before later passes continue.

The backend then resolves those builtins with:

- `src/einlang/backends/numpy_ir_tensor_runtime.py` for runtime graph evaluation
- `src/einlang/backends/numpy_autodiff_core.py` for JVP, VJP, and lazy Jacobians

## Runtime behavior

- `let dx = @x;` materializes the identity tangent of `x`.
  - scalar `x` -> `1.0`
  - tensor `x` -> `ones_like(x)`
- `let dy_dx = @y / @x;` materializes a numeric derivative.
  - scalar/scalar -> scalar
  - tensor cases -> `LazyJacobianTensor`, materialized on demand
- `print(@x)` and `print(@y / @x)` are symbolic display paths, not numeric evaluation.

## Why this is better than the retired diff-block design

- one runtime AD core instead of separate forward-diff, Jacobian-builder, and quotient-special-case paths
- high-level graph snapshot before lowering, so autodiff still sees Einstein structure and function bodies
- lazy Jacobians for tensor quotients instead of eagerly building full derivative IR
- smaller compiler-side rewrite and a clearer backend contract

## Practical guidance

- Bind expressions before differentiating them: `let y = ...; let dy_dx = @y / @x;`
- Prefer `print(@...)` for symbolic inspection and `let d = ...; print(d);` for numeric results
- The current runtime path is NumPy-backed

## Doc map

- [AUTODIFF_DESIGN.md](AUTODIFF_DESIGN.md): current compiler/runtime contract
- [AUTODIFF_VJP_JVP_REWRITE.md](AUTODIFF_VJP_JVP_REWRITE.md): runtime AD architecture and status
- [AUTODIFF_PIPELINE.md](AUTODIFF_PIPELINE.md): pass order and analysis/backend handoff
- [AUTODIFF_OPS.md](AUTODIFF_OPS.md): derivative formulas by op
- [PRINT_DIFFERENTIAL.md](PRINT_DIFFERENTIAL.md): current symbolic `print(@...)` behavior
- [AUTODIFF_IMPLEMENTATION.md](AUTODIFF_IMPLEMENTATION.md), [AUTODIFF_ALGORITHM.md](AUTODIFF_ALGORITHM.md): archived notes for the retired diff-block design
