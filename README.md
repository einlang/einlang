# Einlang

[![Tests](https://github.com/einlang/einlang/actions/workflows/tests.yml/badge.svg)](https://github.com/einlang/einlang/actions/workflows/tests.yml)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python 3.7-3.14](https://img.shields.io/badge/Python-3.7--3.14-3776AB?logo=python&logoColor=white)](https://github.com/einlang/einlang/blob/main/README.md#quick-start)

Einlang is a language and compiler for tensor programs with explicit indices, reductions, recurrences, and built-in automatic differentiation.

```rust
let C[i, j] = sum[k](A[i, k] * B[k, j]);
let dC_dA = @C / @A;
```

The goal is simple: let tensor code look like the math while still getting compile-time checks on shapes and index structure.

## Quick start

Clone the repo if you want the examples:

```bash
git clone https://github.com/einlang/einlang.git
cd einlang
python3 -m pip install -e .
python3 -m einlang examples/hello.ein
```

Or install directly and run a one-liner:

```bash
python3 -m pip install "git+https://github.com/einlang/einlang.git"
python3 -m einlang -c "let x = 1 + 1; print(x);"
```

You should see `2`.

From Python:

```python
from einlang import run

out = run(file="examples/hello.ein")
# or: out = run(source="let x = 1 + 1; print(x);")
```

## Start here

- [Getting started](docs/GETTING_STARTED.md)
- [Docs index](docs/README.md)
- [Language reference](docs/reference.md)
- [Standard library](docs/stdlib.md)
- [Autodiff guide](docs/AUTODIFF.md)
- [Examples guide](examples/README.md)

## What Einlang is good at

- Einstein-style tensor expressions without string-based `einsum`
- Gradients written directly in the language with `@loss / @weights`
- Recurrences for sequences, dynamic programs, and time stepping
- NumPy-backed execution from the CLI or from Python
- Examples that cover basics, optimization, simulation, and full models

## Example entry points

| Goal | Run |
|------|-----|
| Learn the basics | `python3 -m einlang examples/hello.ein` |
| See tensor notation | `python3 -m einlang examples/demos/matrix_operations.ein` |
| Try autodiff on a real fit | `python3 -m einlang examples/applications/linear_regression_autodiff.ein` |
| Run a recurrence | `python3 -m einlang examples/recurrence/recurrence_suite.ein` |
| Run numerics examples | `python3 -m einlang examples/ode/ode_suite.ein` |
| Run a model | `python3 -m einlang examples/mnist/main.ein` |

More paths live in [examples/README](examples/README.md).

## Status

Einlang currently executes through a NumPy backend. The repository includes the language implementation, standard library, tests, and a large example set.

If you want the motivation first, read [Why Einlang](docs/WHY_EINLANG.md). If you want to contribute, start with [CONTRIBUTING](CONTRIBUTING.md).

## License

Apache 2.0. See [LICENSE](LICENSE).
