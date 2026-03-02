# 1 — Basics

> **Previous**: (start here) · **Next**: [`demos/`](../demos/)

Your first look at Einlang. These four files cover the language fundamentals — run them in order.

## Files

| # | File | What it covers |
|---|------|----------------|
| 1 | `variables_demo.ein` | `let` bindings, type inference (`f32`, `str`, `bool`), arrays, indexing |
| 2 | `basic_math.ein` | Arithmetic (`+`, `-`, `*`, `/`, `%`), `assert` |
| 3 | `functions_demo.ein` | `fn` definitions, `sum[i]()` reductions, dot product, mean, variance |
| 4 | `data_processing.ein` | Comprehensions (`[x \| x in data, x > 10.0]`), `max[i]`, `min[i]`, conditional expressions |

## Running

```bash
python3 -m einlang examples/basics/variables_demo.ein
python3 -m einlang examples/basics/basic_math.ein
python3 -m einlang examples/basics/functions_demo.ein
python3 -m einlang examples/basics/data_processing.ein
```

## Concepts introduced

- **`let` bindings** — immutable variables with type inference: `let x = 42.0;`
- **Arrays** — `let numbers = [1, 2, 3, 4, 5];` with zero-based indexing
- **Functions** — `fn square(x) { x * x }` with implicit return (last expression)
- **Reductions** — `sum[i](arr[i])` sums over all valid indices of `arr`
- **Comprehensions** — `[expr | var in collection, predicate]` to filter and transform
- **Assertions** — `assert(cond, msg)` for runtime checks

These are the building blocks for everything that follows. Once you're comfortable, move on to [demos/](../demos/) for matrices, tensors, and imports.
