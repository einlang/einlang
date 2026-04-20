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

## Why it looks different

| Instead of... | In Einlang... |
|---------------|---------------|
| `y = np.einsum("bi,ci->bc", x, W) + bias` | `let y[b, c] = sum[i](x[b, i] * W[c, i]) + bias[c];` |
| `dloss_dW = jax.grad(loss_fn)(W)` | `let dloss_dW = @loss / @W;` |

The point is not only that Einlang can run tensor code. The structure you care about stays visible.

## A one-minute taste

```rust
let x = 1.0;
let y = x * x + 3.0 * x;
let dy_dx = @y / @x;
print(dy_dx);
```

When `x = 1`, this prints `5.0`.

The derivative request lives in the language right next to the primal expression; you do not switch to a separate autodiff API.

## Visitor path

| Time | Run | What you learn |
|------|-----|----------------|
| 30 seconds | `python3 -m einlang examples/hello.ein` | The toolchain is working and the language reads cleanly. |
| 5 minutes | `python3 -m einlang examples/autodiff_small.ein`<br>`python3 -m einlang examples/demos/matrix_operations.ein` | Autodiff syntax and Einstein-style tensor code without helper APIs. |
| 30 minutes | `python3 -m einlang examples/applications/linear_regression_autodiff.ein`<br>`python3 -m einlang examples/recurrence/recurrence_suite.ein`<br>`python3 -m einlang examples/ode/ode_suite.ein`<br>`python3 -m einlang examples/mnist/main.ein` | The same notation on fitting, recurrences, numerics, and a full model example. |

The commands above all run from the repository root after install. If a larger showcase needs extra setup or a different working directory, its directory README calls that out explicitly instead of putting friction in the first run.

## Quick start

Clone the repo if you want the examples:

```bash
git clone https://github.com/einlang/einlang.git
cd einlang
python3 -m pip install -e .
python3 -m einlang examples/hello.ein
```

Then run the shortest example that shows why people stick around:

```bash
python3 -m einlang examples/autodiff_small.ein
```

Or install directly and run a one-liner:

```bash
python3 -m pip install "git+https://github.com/einlang/einlang.git"
python3 -m einlang -c "let x = 1 + 1; print(x);"
```

You should see `2`.

If you want the latest formal release artifact instead of the current `main` branch:

```bash
python3 -m pip install "https://github.com/einlang/einlang/releases/latest/download/einlang-latest-py3-none-any.whl"
```

From Python:

```python
from einlang import run

out = run(file="examples/hello.ein")
# or: out = run(source="let x = 1 + 1; print(x);")
```

To try the in-progress IREE backend, install the optional extra and select it explicitly:

```bash
python3 -m pip install -e ".[iree]"
python3 -m einlang --backend iree examples/hello.ein
```

## Why visitors stick around

- The same notation keeps working from tiny autodiff snippets to recurrences, numerics, MNIST, DeiT, and Whisper-style examples.
- The example set is curated through [examples/README](examples/README.md) instead of expecting visitors to browse the whole tree cold.
- The front-door examples are checked in [README example tests](tests/examples/test_readme_examples.py), and larger showcases are covered by [demo/example suites](tests/examples/test_demo.py) and [autodiff/model checks](tests/examples/test_mnist_autodiff.py).

## Start here

- [Getting started](docs/GETTING_STARTED.md)
- [Examples guide](examples/README.md)
- [Docs index](docs/README.md)
- [Architecture guide](docs/ARCHITECTURE.md)
- [Language reference](docs/reference.md)
- [Standard library](docs/stdlib.md)
- [Autodiff guide](docs/AUTODIFF.md)
- [Why Einlang](docs/WHY_EINLANG.md)

## What Einlang is good at

- Gradients written directly in the language with `@loss / @weights`
- Einstein-style tensor expressions without string-based `einsum`
- Recurrences for sequences, dynamic programs, and time stepping
- NumPy-backed execution from the CLI or from Python
- Examples that cover basics, optimization, simulation, and full models

## Example entry points

| Goal | Run |
|------|-----|
| Learn the basics | `python3 -m einlang examples/hello.ein` |
| See autodiff immediately | `python3 -m einlang examples/autodiff_small.ein` |
| See tensor notation | `python3 -m einlang examples/demos/matrix_operations.ein` |
| Try autodiff on a real fit | `python3 -m einlang examples/applications/linear_regression_autodiff.ein` |
| Run a recurrence | `python3 -m einlang examples/recurrence/recurrence_suite.ein` |
| Run numerics examples | `python3 -m einlang examples/ode/ode_suite.ein` |
| Run a model | `python3 -m einlang examples/mnist/main.ein` |

More paths live in [examples/README](examples/README.md).

## Status

Einlang currently executes through a NumPy backend. The repository includes the language implementation, standard library, tests, and a large example set.
An IREE execution path is also being implemented behind the optional `iree` extra; supported functions compile through IREE and unsupported ones fall back to NumPy.
If you want to contribute, start with [CONTRIBUTING](CONTRIBUTING.md).

## License

Apache 2.0. See [LICENSE](LICENSE).
