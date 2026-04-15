# Automatic differentiation

**Status:** Consolidated current autodiff documentation. This file replaces the previous split autodiff docs such as `AUTODIFF_HIGHLIGHTS.md`, `AUTODIFF_DESIGN.md`, `AUTODIFF_VJP_JVP_REWRITE.md`, `AUTODIFF_PIPELINE.md`, `AUTODIFF_OPS.md`, `AUTODIFF_EINSTEIN.md`, and related legacy notes.

## 1. Overview

Einlang supports built-in automatic differentiation directly in the language. The key idea is:

- write the primal program normally
- bind values with `let`
- request tangents or derivatives with `@x` and `@y / @x`

This is not a separate `grad(f)` wrapper API, nor is it a tape-based user package. Autodiff is part of the compiler/runtime contract.

## 2. User-facing model

### 2.1 Numeric requests

- `let x = ...`
- `let y = ...`
- `let dy_dx = @y / @x`

This evaluates the derivative or Jacobian of `y` with respect to the named binding `x`.

- scalar/scalar → scalar
- tensor/non-scalar → lazy Jacobian-backed tensor value

### 2.2 Tangent requests

- `let tx = @x`

This materializes the identity tangent for binding `x`.

- scalar `x` → `1.0`
- tensor `x` → `ones_like(x)`

### 2.3 Symbolic display

Direct use of `print(@...)` is a symbolic debugging path:

- `print(@x)` prints a symbolic tangent string
- `print(@y / @x)` prints a symbolic Jacobian relation string

If you want numeric behavior, bind the request first and then print the bound value.

## 3. Current implementation

### 3.1 Compiler-side contract

The current autodiff implementation is centered on `AutodiffPass` in `src/einlang/passes/autodiff/__init__.py`.

`AutodiffPass` runs on typed, high-level IR before Einstein lowering. It does the following:

1. collects executable autodiff requests from the binding graph
2. clones the relevant high-level graph into analysis state
3. rewrites requests to internal runtime builtin intrinsics
4. rewrites direct `print(@...)` calls to symbolic autodiff intrinsics
5. clears autodiff-only IR before later passes continue

### 3.2 Runtime intrinsics

The compiler rewrites requests to internal builtin operations such as:

- `__autodiff_tangent` — tangent request for a named binding
- `__autodiff_jacobian` — numeric derivative/Jacobian request for `(numerator, denominator)`
- `__autodiff_symbolic_tangent` — symbolic `print(@x)`
- `__autodiff_symbolic_jacobian` — symbolic `print(@y / @x)`

These are not user-facing functions; they are the compiler/runtime handshake.

### 3.3 Backend behavior

The NumPy backend resolves these builtins using the high-level graph snapshot stored on `TyCtxt`.

The runtime path is implemented in:

- `src/einlang/backends/numpy_ir_tensor_runtime.py`
- `src/einlang/backends/numpy_autodiff_core.py`

The runtime evaluates primal values, answers tangent requests, and answers Jacobian requests using JVP/VJP machinery.

## 4. JVP, VJP, and lazy Jacobians

The backend uses forward- and reverse-mode components:

- `jvp` for forward-mode linearization
- `vjp` for reverse-mode pullback
- `LazyJacobianTensor` for tensor quotient values

`LazyJacobianTensor` chooses a cheaper path dynamically:

- JVP when the input space is smaller than the output space
- VJP when the output space is smaller than the input space

It can materialize:

- the full Jacobian
- a single row or column
- a single entry

without always building the entire dense matrix eagerly.

## 5. Pass pipeline

Autodiff runs before Einstein lowering. The relevant pass order is:

- RangeAnalysis
- UnifiedShapeAnalysis
- TypeInference
- ExtremumSelectionCanonicalization
- PreAutodiffPruning
- AutodiffPass
- PostAutodiffPruning
- AutodiffLeakCheck
- EinsteinLowering
- RecurrenceOrder
- validation passes

The important guarantee is that autodiff sees high-level Einstein structure and function bodies before lowering.

## 6. Supported operations

### 6.1 Elementwise unary ops

- `neg`, `pos`
- `exp`, `ln`, `sqrt`
- `sin`, `cos`, `tanh`
- `sigmoid`, `abs`, `relu`, `leaky_relu`

### 6.2 Elementwise binary ops

- `add`, `sub`, `mul`, `div`, `pow`, `mod`

Broadcasting is handled by summing back over broadcast dimensions to match the operand shape.

### 6.3 Einstein-style tensor ops

Einlang differentiates high-level Einstein expressions before lowering, so the following are supported when written as sum-of-products:

- matrix multiply
- convolution expressed as Einstein with `where` clauses
- general einsum-style reductions

Examples:

- `let C[i,j] = sum[k](A[i,k]*B[k,j]); let dC_dA = @C / @A;`
- convolution with `where ih = oh + kh`

### 6.4 Affine maps

For `y = x * W^T + b`:

- `@y / @x` computes the derivative w.r.t. `x`
- `@y / @W` computes the derivative w.r.t. `W`
- `@y / @b` computes the derivative w.r.t. `b`

### 6.5 Reductions

- `sum`: derivative is broadcast `1`
- `max` / `min`: derivative selects the argmax / argmin positions
- `prod`: derivative is `y / x_i` with the usual zero-handling caveats

### 6.6 Softmax and log_softmax

- `softmax`: `∂p_i/∂x_j = p_i (δ_ij - p_j)`
- `log_softmax`: `∂ℓ_i/∂x_j = δ_ij - p_j`

These are expressed through the same autodiff runtime path when the softmax body is visible in the high-level graph.

## 7. Practical guidance

- Bind values before differentiating them.
- Use `@x` for identity tangents and `@y / @x` for numeric derivatives.
- Use `print(@...)` only for symbolic debugging.
- For tensor quotients, expect lazy Jacobian behavior; the runtime may defer materialization.

Example:

```rust
let x = 3.0;
let y = x * x;
let dy_dx = @y / @x;
print(dy_dx);  // 6.0
```

## 8. Examples

Run practical examples with:

- `python3 -m einlang examples/autodiff_small.ein`
- `python3 -m einlang examples/autodiff_matmul.ein`
- `python3 -m einlang examples/autodiff_chain.ein`
- `python3 -m einlang examples/autodiff_loss.ein`

## 9. Tests and validation

Autodiff tests live in `tests/unit/test_autodiff_pass.py` and cover:

- scalar and tensor derivatives
- Einstein matmul and convolutions
- lazy Jacobian indexing and shapes
- symbolic `print(@...)` behavior

## 10. Historical notes

Older split autodiff docs have been consolidated into this file. The prior separation of design, runtime, pipeline, op formulas, and Einstein math is now unified here for a single source of truth.
