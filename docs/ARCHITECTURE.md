# Einlang architecture

This is the short contributor map: the usual homes for parser, compiler, runtime, documentation, and test changes.

## Flow

Einlang keeps a simple top-level split:

1. The CLI in [`src/einlang/__main__.py`](../src/einlang/__main__.py) or Python helper in [`src/einlang/run.py`](../src/einlang/run.py) receives source.
2. [`CompilerDriver`](../src/einlang/compiler/driver.py) parses, resolves names, lowers to IR, and runs the pass pipeline.
3. [`EinlangRuntime`](../src/einlang/runtime/runtime.py) picks a backend and delegates execution.
4. A backend in [`src/einlang/backends`](../src/einlang/backends) runs the lowered IR.

The runtime is intentionally thin. Most language behavior belongs in the compiler pipeline.

## Repo map

- [`src/einlang/frontend`](../src/einlang/frontend): parser, grammar, and frontend transforms
- [`src/einlang/analysis`](../src/einlang/analysis): analysis helpers, including module loading
- [`src/einlang/compiler`](../src/einlang/compiler): compiler orchestration
- [`src/einlang/passes`](../src/einlang/passes): lowering, analysis, autodiff, validation, and backend-facing transforms
- [`src/einlang/ir`](../src/einlang/ir): IR nodes, visitors, serialization
- [`src/einlang/runtime`](../src/einlang/runtime): runtime delegation layer
- [`src/einlang/backends`](../src/einlang/backends): execution backends
- [`stdlib`](../stdlib): standard library source
- [`examples`](../examples): public runnable examples
- [`tests`](../tests): unit, integration, and public example tests

## Compiler stages

The main flow inside [`CompilerDriver.compile`](../src/einlang/compiler/driver.py) is:

1. Parse source into AST via [`src/einlang/frontend/parser.py`](../src/einlang/frontend/parser.py)
2. Set up the module system in [`src/einlang/analysis/module_system`](../src/einlang/analysis/module_system)
3. Run name resolution in [`src/einlang/passes/name_resolution.py`](../src/einlang/passes/name_resolution.py)
4. Lower AST to IR with [`src/einlang/passes/ast_to_ir.py`](../src/einlang/passes/ast_to_ir.py)
5. Run IR passes from [`src/einlang/passes`](../src/einlang/passes), including coordinate analysis for calls such as `softmax[class](x)` and `argmax[class](x)`
6. Tree-shake and hand the result to the runtime/backend

## Where to edit what

- Parser or grammar change:
  [`src/einlang/frontend`](../src/einlang/frontend)
- Import or module-loading bug:
  [`src/einlang/analysis/module_system`](../src/einlang/analysis/module_system) and [`src/einlang/passes/name_resolution.py`](../src/einlang/passes/name_resolution.py)
- AST-to-IR bug:
  [`src/einlang/passes/ast_to_ir.py`](../src/einlang/passes/ast_to_ir.py)
- Shape, range, or type issue:
  [`range_analysis.py`](../src/einlang/passes/range_analysis.py), [`shape_analysis.py`](../src/einlang/passes/shape_analysis.py), [`type_inference.py`](../src/einlang/passes/type_inference.py)
- Coordinate-aware function or selection-reduction issue:
  [`coordinate_analysis.py`](../src/einlang/passes/coordinate_analysis.py), [`ast_to_ir.py`](../src/einlang/passes/ast_to_ir.py), [`range_analysis.py`](../src/einlang/passes/range_analysis.py)
- Autodiff issue:
  [`src/einlang/passes/autodiff`](../src/einlang/passes/autodiff)
- Einstein lowering or recurrence issue:
  [`einstein_lowering.py`](../src/einlang/passes/einstein_lowering.py), [`recurrence_order.py`](../src/einlang/passes/recurrence_order.py)
- Runtime or backend issue:
  [`src/einlang/runtime`](../src/einlang/runtime), [`src/einlang/backends`](../src/einlang/backends)
- Stdlib behavior:
  [`stdlib`](../stdlib)
- Public docs or examples:
  [`docs`](.), [`examples`](../examples)

## Testing map

- [`tests/unit`](../tests/unit): narrow compiler/runtime behavior
- [`tests/integration`](../tests/integration): end-to-end language behavior
- [`tests/examples`](../tests/examples): public docs/example promises

If you change public example paths or the onboarding flow, run the docs/example contract tests too.
