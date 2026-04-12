# Autodiff pipeline

**Status:** Current pass-order and handoff note for the runtime-builtins autodiff path.

**Overview:** [AUTODIFF_HIGHLIGHTS.md](AUTODIFF_HIGHLIGHTS.md)

## 1. Current compiler order

In `src/einlang/compiler/driver.py`, the relevant order is:

```text
RangeAnalysis
-> UnifiedShapeAnalysis
-> TypeInference
-> ExtremumSelectionCanonicalization
-> PreAutodiffPruning
-> AutodiffPass
-> PostAutodiffPruning
-> AutodiffLeakCheck
-> EinsteinLowering
-> RecurrenceOrder
-> validation passes
```

The important point is that `AutodiffPass` runs on **typed, high-level IR before Einstein lowering**.

## 2. What AutodiffPass assumes

- `type_info` and `shape_info` are available
- high-level Einstein structure is still present
- rank/shape pruning that should happen before autodiff has already happened
- a resolver exists for allocating temporary rewritten bindings

## 3. What AutodiffPass produces

- executable autodiff requests rewritten to runtime intrinsics
- symbolic `print(@...)` rewritten to symbolic autodiff intrinsics
- a cloned high-level graph snapshot stored in `TyCtxt` analysis
- no leaked `DifferentialIR` in the post-pass IR

## 4. Backend handoff

The NumPy backend reads the autodiff analysis snapshot and resolves:

- pending tangent slots
- pending quotient slots
- direct runtime autodiff builtins inside the stored graph

The main current entry points are:

- `src/einlang/backends/numpy_core.py`
- `src/einlang/backends/numpy_ir_tensor_runtime.py`

## 5. Historical note

Earlier docs described a different pipeline based on compiler-generated diff blocks, interleaved `d_*` bindings, and dedicated Jacobian construction. That is not the main implementation path anymore.

See:

- [AUTODIFF_DESIGN.md](AUTODIFF_DESIGN.md) for the current contract
- [AUTODIFF_VJP_JVP_REWRITE.md](AUTODIFF_VJP_JVP_REWRITE.md) for the runtime architecture
- [AUTODIFF_IMPLEMENTATION.md](AUTODIFF_IMPLEMENTATION.md) and [AUTODIFF_ALGORITHM.md](AUTODIFF_ALGORITHM.md) for archived notes on the retired design
