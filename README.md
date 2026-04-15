# Einlang

[![Tests](https://github.com/einlang/einlang/actions/workflows/tests.yml/badge.svg)](https://github.com/einlang/einlang/actions/workflows/tests.yml)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python 3.7-3.14](https://img.shields.io/badge/Python-3.7--3.14-3776AB?logo=python&logoColor=white)](https://github.com/einlang/einlang/blob/main/README.md#install--run)

Einlang is a language and compiler for tensor programs with explicit indices, reductions, recurrences, and derivative expressions.

Main pieces:

- Einstein-style tensor code as language syntax
- built-in automatic differentiation with forms such as `@name`, `@loss / @w`, and `@C / @A`
- compile-time checking of shape and index structure
- a NumPy backend for execution today
- a repository of examples spanning model code, optimization, recurrence, and numerical programs

```rust
let C[i, j] = sum[k](A[i, k] * B[k, j]);   // matrix multiply
let dC_dA = @C / @A;                       // derivative tensor
```

Start with [Getting started](https://github.com/einlang/einlang/blob/main/docs/GETTING_STARTED.md). For the full doc map, open [docs/README](https://github.com/einlang/einlang/blob/main/docs/README.md).

---

## Try it

```bash
python3 -m pip install "git+https://github.com/einlang/einlang.git"
python3 -m einlang -c "let x = 1+1; print(x);"
```

You should see `2`.

Run a real file:

```bash
python3 -m einlang examples/hello.ein
```

---

## Install & run

**Install:** Python 3.7+ (tested 3.7-3.14). Use `python3 -m pip install "git+https://github.com/einlang/einlang.git"` for the simplest install. For local development from a clone, use `pip install -e .`.

**Run a file:**

```bash
python3 -m einlang examples/hello.ein
python3 -m einlang path/to/file.ein
```

**Inline source:**

```bash
python3 -m einlang -c "let x = 1+1; print(x);"
echo 'let x = 2; print(x);' | python3 -m einlang -
```

**From Python:**

```python
from einlang import run

out = run(file="examples/hello.ein")
# or: out = run(source="let x = 1+1; print(x);")
```

---

## Repository contents

The repository has four main parts:

| Area | What to inspect |
|------|-----------------|
| **Language** | [reference](https://github.com/einlang/einlang/blob/main/docs/reference.md), [stdlib](https://github.com/einlang/einlang/blob/main/docs/stdlib.md), [MATH](https://github.com/einlang/einlang/blob/main/docs/MATH.md) |
| **Autodiff** | [AUTODIFF](https://github.com/einlang/einlang/blob/main/docs/AUTODIFF.md), [examples/autodiff_small.ein](https://github.com/einlang/einlang/blob/main/examples/autodiff_small.ein), [examples/autodiff_matmul.ein](https://github.com/einlang/einlang/blob/main/examples/autodiff_matmul.ein) |
| **Examples** | [examples/README](https://github.com/einlang/einlang/blob/main/examples/README.md), [mnist](https://github.com/einlang/einlang/tree/main/examples/mnist), [optimization](https://github.com/einlang/einlang/tree/main/examples/optimization), [ode](https://github.com/einlang/einlang/tree/main/examples/ode), [recurrence](https://github.com/einlang/einlang/tree/main/examples/recurrence) |
| **Implementation** | [src/einlang/passes](https://github.com/einlang/einlang/tree/main/src/einlang/passes), [src/einlang/backends](https://github.com/einlang/einlang/tree/main/src/einlang/backends), [DEVELOPMENT](https://github.com/einlang/einlang/blob/main/docs/DEVELOPMENT.md) |

---

## Project shape

- a readable tensor language, not a Python wrapper around `einsum`
- autodiff in the language and compiler, not a separate gradient API
- a NumPy execution path now, with future backend work documented separately
- examples that cover both ML-shaped programs and non-ML numerical workloads
- documentation that points directly to reference material and concrete artifacts

If you want the motivation and comparison page, read [WHY_EINLANG](https://github.com/einlang/einlang/blob/main/docs/WHY_EINLANG.md).

---

## Examples

Start with [examples/README](https://github.com/einlang/einlang/blob/main/examples/README.md). A few entry points:

| Area | Run |
|------|-----|
| **Basics** | `python3 -m einlang examples/hello.ein` |
| **Autodiff** | `python3 -m einlang examples/autodiff_small.ein` |
| **Recurrence** | `python3 -m einlang examples/recurrence/recurrence_suite.ein` |
| **Optimization** | `python3 -m einlang examples/optimization/optimization_suite.ein` |
| **ODEs** | `python3 -m einlang examples/ode/ode_suite.ein` |
| **MNIST** | `python3 -m einlang examples/mnist/main.ein` |

Examples span:

- language basics and tensor expressions
- autodiff and gradient-based examples
- recurrence and dynamic programs
- optimization and value iteration
- ODE, PDE, and wave-style simulation
- MNIST, quantized MNIST, DeiT-tiny, and Whisper-tiny artifacts

---

## Docs

Use these as the main entry points:

- [Getting started](https://github.com/einlang/einlang/blob/main/docs/GETTING_STARTED.md)
- [Docs index](https://github.com/einlang/einlang/blob/main/docs/README.md)
- [Language reference](https://github.com/einlang/einlang/blob/main/docs/reference.md)
- [Standard library](https://github.com/einlang/einlang/blob/main/docs/stdlib.md)
- [Why Einlang](https://github.com/einlang/einlang/blob/main/docs/WHY_EINLANG.md)
- [Syntax comparison](https://github.com/einlang/einlang/blob/main/docs/SYNTAX_COMPARISON.md)
- [Einlang for Julia programmers](https://github.com/einlang/einlang/blob/main/docs/EINLANG_FOR_JULIA_PROGRAMMERS.md)

Autodiff docs:

- [AUTODIFF](https://github.com/einlang/einlang/blob/main/docs/AUTODIFF.md)

---

## Development

For contributors:

- [CONTRIBUTING](https://github.com/einlang/einlang/blob/main/CONTRIBUTING.md)
- [DEVELOPMENT](https://github.com/einlang/einlang/blob/main/docs/DEVELOPMENT.md)

The current runtime path uses the NumPy backend. Longer-term direction is tracked in [ROADMAP](https://github.com/einlang/einlang/blob/main/docs/ROADMAP.md).

---

## License

Apache 2.0 — see [LICENSE](LICENSE).
