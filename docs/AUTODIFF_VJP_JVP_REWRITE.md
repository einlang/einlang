# Autodiff runtime architecture: JVP, VJP, and lazy Jacobians

**Status:** Current direction, with the phase-1 runtime rewrite landed in the compiler and NumPy backend.

This document replaces the old "proposed rewrite" framing. The codebase now uses the runtime-builtins architecture described here:

- `AutodiffPass` snapshots high-level IR and rewrites requests to internal builtins
- the NumPy backend resolves those requests with a JVP/VJP runtime
- tensor quotients use `LazyJacobianTensor`

## 1. Landed pieces

### 1.1 Compiler-side

`src/einlang/passes/autodiff/__init__.py` now:

- collects autodiff requests
- clones the typed high-level binding graph
- rewrites executable requests to internal autodiff builtins
- rewrites direct `print(@...)` calls to symbolic autodiff builtins
- clears autodiff-only IR before later passes

### 1.2 Runtime-side

`src/einlang/backends/numpy_ir_tensor_runtime.py` now:

- rebuilds tensor views of bindings from the stored graph snapshot
- resolves tangent requests
- resolves Jacobian requests
- evaluates custom diff bodies through the same runtime intrinsics

`src/einlang/backends/numpy_autodiff_core.py` provides:

- `jvp`
- `vjp`
- `LazyJacobianTensor`

## 2. Core idea

There is no longer a compiler-generated diff block in the main implementation path.

Instead:

1. the compiler stores a high-level graph snapshot
2. the executable IR contains plain runtime placeholders
3. the backend answers those placeholders from the stored graph and current primal values

That keeps autodiff centered on one runtime AD core instead of several partially-overlapping compiler paths.

## 3. Current runtime model

### 3.1 Tangent requests

`@x` as an executable value request means "materialize the identity tangent seed for the named binding `x`."

Examples:

- scalar `x` -> `1.0`
- tensor `x` -> `ones_like(x)`

### 3.2 Jacobian requests

`@y / @x` means "materialize the derivative/Jacobian of `y` with respect to `x`."

- scalar/scalar -> scalar result
- non-scalar cases -> `LazyJacobianTensor`

Shape contract:

```text
shape(@y / @x) = shape(y) + shape(x)
```

### 3.3 Symbolic print requests

Direct print keeps a symbolic view:

- `print(@x)` -> symbolic tangent string
- `print(@y / @x)` -> symbolic Jacobian relation string

Numeric evaluation happens when the autodiff request is first bound and then used like an ordinary value.

## 4. Why JVP and VJP

The runtime core uses:

- JVP for forward pushforward
- VJP for reverse pullback

`LazyJacobianTensor` chooses the cheaper default basis direction:

- JVP when `size(x) <= size(y)`
- VJP otherwise

It can answer:

- full materialization
- a single entry
- a row
- a column

without always materializing the full Jacobian first.

## 5. High-level IR is still the differentiation boundary

Autodiff still runs before Einstein lowering. That matters because the stored graph preserves:

- `EinsteinIR`
- user function bodies
- high-level control flow
- custom diff bodies

So the runtime AD engine works from source-like structure instead of from lowered loops.

## 6. Internal builtins

The compiler/runtime handshake is expressed with these internal builtins:

| Builtin | Kind |
|--------|------|
| `__autodiff_tangent` | numeric tangent request |
| `__autodiff_jacobian` | numeric derivative/Jacobian request |
| `__autodiff_symbolic_tangent` | symbolic tangent print request |
| `__autodiff_symbolic_jacobian` | symbolic Jacobian print request |

Their `DefId`s and names live in `src/einlang/shared/autodiff_intrinsics.py`.

## 7. Current gaps and guidance

- The user-facing executable path currently works best for identifier-based requests.
  - Write `let y = ...; let dy_dx = @y / @x;`
- Direct `print(@...)` is symbolic by design.
- The current implementation is NumPy-backed.
- Legacy analysis fields from the diff-block era still exist in some places for compatibility, but they are no longer the main architecture.

## 8. What this doc replaces

The old rewrite plan talked about deleting the previous autodiff package and landing the runtime architecture in phases. That transition has happened.

The historical docs that still describe:

- compiler-generated diff blocks
- dedicated Jacobian visitors
- quotient seeding as the main mechanism

should now be read as archived notes, not as the current implementation contract.

## 9. Related docs

- [AUTODIFF_HIGHLIGHTS.md](AUTODIFF_HIGHLIGHTS.md)
- [AUTODIFF_DESIGN.md](AUTODIFF_DESIGN.md)
- [AUTODIFF_PIPELINE.md](AUTODIFF_PIPELINE.md)
- [AUTODIFF_OPS.md](AUTODIFF_OPS.md)
