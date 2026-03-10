---
layout: default
title: Units
---

# Reference — Unit Tests

> **Previous**: [`whisper_tiny/`](../whisper_tiny/)

This directory contains 60+ standalone `.ein` files, each exercising a specific language feature. Use it as a lookup table when you need to check exact syntax or behavior.

## Categories

| Area | Files | What they cover |
|------|-------|-----------------|
| Primitives | `bootstrap_primitives.ein`, `basic_arithmetic.ein`, `boolean_logic.ein`, `unary_operators.ein` | Literal types, arithmetic, boolean ops, negation |
| Arrays | `array_literals.ein`, `array_comparisons.ein`, `simple_array_indexing.ein`, `advanced_indexing.ein` | Construction, equality, slicing, multi-dim indexing |
| Comprehensions | `comprehensions.ein`, `array_comprehension_syntax.ein`, `tuple_expressions.ein`, `tuple_syntax.ein` | `[expr \| var in range]`, nested comprehensions, tuple iteration |
| Reductions | `reduction_operations.ein`, `cumulative_operations.ein`, `scan_operations.ein`, `multidimensional_scans.ein` | `sum[i]`, `max[i]`, `min[i]`, scans, cumulative sums |
| Einstein notation | `chained_einstein.ein`, `einstein_windowing.ein`, `tensors.ein`, `forward_indexing_patterns.ein` | Index expressions, windowed ops, tensor contractions |
| Convolutions | `convolution_operations.ein`, `auto_convolution_inference.ein`, `windowed_operations_working.ein` | Manual and stdlib conv, automatic shape inference |
| Functions | `functions.ein`, `functions_advanced.ein`, `function_overloading.ein`, `bootstrap_functions.ein` | Definition, recursion, overloading, closures |
| Control flow | `conditionals.ein`, `expressions.ein`, `complex_expressions.ein`, `where_constraints.ein` | `if`/`else`, `where` clauses, expression nesting |
| Strings | `string_operations.ein`, `string_interpolation.ein`, `string_utility_functions.ein` | Concatenation, `"{x}"` interpolation, utility functions |
| Modules | `import_system_examples.ein`, `stdlib_module_imports.ein`, `native_modules.ein`, `use_statements.ein` | `use`, `mod`, stdlib access, Python interop |
| Ranges | `range_expressions.ein`, `range_behavior_examples.ein`, `automatic_range_inference.ein`, `bounds_inference_examples.ein` | `0..n`, inferred bounds, constraint-based ranges |
| Math | `mathematical_operations.ein`, `math_library_comprehensive.ein`, `math_utility_functions.ein`, `power_operator.ein` | `sqrt`, `exp`, `log`, `pow`, trig functions |
| Advanced | `variable_scope.ein`, `member_access.ein`, `method_calls.ein`, `pipeline_operators_demo.ein`, `pure_functional_programming.ein` | Scoping rules, member access, pipe operator, FP patterns |

## Running

Run any file directly:

```bash
python3 -m einlang examples/units/basic_arithmetic.ein
python3 -m einlang examples/units/chained_einstein.ein
```

Or run the full suite via pytest:

```bash
python3 -m pytest tests/examples/ -q
```
