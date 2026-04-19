# Getting started with Einlang

This page is the shortest path from zero to a running program.

## What Einlang is

Einlang is a language for tensor programs with:

- explicit indices and reductions
- recurrences for sequential definitions
- compile-time checking of shapes and index structure
- built-in automatic differentiation

```rust
let C[i, j] = sum[k](A[i, k] * B[k, j]);
```

If that shape of code is what you want to write, you are in the right place.

## Install and try it

```bash
git clone https://github.com/einlang/einlang.git
cd einlang
python3 -m pip install -e .
python3 -m einlang -c "let x = 1 + 1; print(x);"
```

You should see `2`.

## Run your first file

```bash
python3 -m einlang examples/hello.ein
```

Then try a few representative examples:

```bash
python3 -m einlang examples/demos/matrix_operations.ein
python3 -m einlang examples/autodiff_small.ein
python3 -m einlang examples/recurrence/recurrence_suite.ein
```

## Use it from Python

```python
from einlang import run

out = run(file="examples/hello.ein")
# or: out = run(source="let x = 1 + 1; print(x);")
```

`out.outputs` contains the produced values, and `out.error` is set if compilation or execution failed.

## What to read next

| If you want to... | Read |
|-------------------|------|
| Learn the syntax | [reference](reference.md) |
| Look up functions and modules | [stdlib](stdlib.md) |
| Use autodiff | [AUTODIFF](AUTODIFF.md) |
| Find more runnable programs | [examples/README](../examples/README.md) |
| Understand the motivation | [WHY_EINLANG](WHY_EINLANG.md) |
| Translate from NumPy, Julia, or Rust habits | [SYNTAX_COMPARISON](SYNTAX_COMPARISON.md) |

That is the full path: run a small example, read the reference when you need precision, and use the examples guide when you want concrete programs.
