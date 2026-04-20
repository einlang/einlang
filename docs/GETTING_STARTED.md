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

## A 60-second taste

```rust
let x = 1.0;
let y = x * x + 3.0 * x;
let dy_dx = @y / @x;
print(dy_dx);
```

That is the core idea in four lines: the derivative request is part of the language, not a wrapper around it.

## Install and verify

```bash
git clone https://github.com/einlang/einlang.git
cd einlang
python3 -m pip install -e .
python3 -m einlang -c "let x = 1 + 1; print(x);"
```

You should see `2`.

## A better first visit

### 30 seconds

```bash
python3 -m einlang examples/hello.ein
```

### 5 minutes

Run the shortest example that shows the autodiff syntax directly:

```bash
python3 -m einlang examples/autodiff_small.ein
```

Then try two representative examples:

```bash
python3 -m einlang examples/demos/matrix_operations.ein
python3 -m einlang examples/applications/linear_regression_autodiff.ein
```

### 30 minutes

Once the syntax feels comfortable, branch into recurrences, numerics, or a full model example:

```bash
python3 -m einlang examples/recurrence/recurrence_suite.ein
python3 -m einlang examples/ode/ode_suite.ein
python3 -m einlang examples/mnist/main.ein
```

If you want training-oriented or directory-local showcases after that, use [examples/README](../examples/README.md) and the relevant example directory README. Those pages keep the extra setup notes out of the first-run path.

## Use it from Python

```python
from einlang import run

out = run(file="examples/hello.ein")
# or: out = run(source="let x = 1 + 1; print(x);")
```

`out.outputs` contains the produced values, and `out.error` is set if compilation or execution failed.

## Optional: try the IREE backend

The IREE path is still in progress, but you can enable it with the optional extra:

```bash
python3 -m pip install -e ".[iree]"
python3 -m einlang --backend iree examples/hello.ein
```

Supported functions compile through IREE; anything outside the current subset falls back to the NumPy backend.

## What to read next

| If you want to... | Read |
|-------------------|------|
| Learn the syntax | [reference](reference.md) |
| Look up functions and modules | [stdlib](stdlib.md) |
| Use autodiff | [AUTODIFF](AUTODIFF.md) |
| Find more runnable programs | [examples/README](../examples/README.md) |
| Understand the motivation | [WHY_EINLANG](WHY_EINLANG.md) |
| Translate from NumPy, Julia, or Rust habits | [SYNTAX_COMPARISON](SYNTAX_COMPARISON.md) |

That is the path: verify the install, run a tiny example, run the small autodiff example, then use the examples guide when you want broader workloads.
