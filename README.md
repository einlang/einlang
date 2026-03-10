---
layout: default
title: Einlang
---

# Einlang

[![Tests](https://github.com/einlang/einlang/actions/workflows/tests.yml/badge.svg)](https://github.com/einlang/einlang/actions/workflows/tests.yml)

**Tensor code is either readable or safe—usually neither.** Einlang is both: write math in Einstein notation, get shape errors at compile time instead of at 3am.

```rust
let A = [[1, 2], [3, 4]];
let B = [[5, 6], [7, 8]];
let C[i, j] = sum[k](A[i, k] * B[k, j]);   // matrix multiply — shapes checked by the compiler
```

**New here?** [Getting started](docs/GETTING_STARTED.md) tells the full story in one page. Or try it below.

---

## Try it

Run the commands below (no account required). You'll see `2` in a few seconds.

```bash
git clone https://github.com/einlang/einlang.git
cd einlang
pip install -e .
python3 -m einlang -c "let x = 1+1; print(x);"
```

**Prefer a file?** Run `python3 -m einlang examples/hello.ein` for the matrix multiply above, or [use the Python API](#install--run) with a source string. Once that works, pick what you want to do next.

---

## What's next?

One step is enough — no account or long read required.

| You want to… | Do this |
|--------------|--------|
| **Run another example** | `python3 -m einlang examples/basics/basic_math.ein` or [examples/demos/matrix_operations.ein](examples/demos/matrix_operations.ein) |
| **Use it in your code** | [Install & run](#install--run) — `run(source="...")` or `run(file="path.ein")`; one call and you're a user. |
| **Learn the language** | [Language Reference](docs/reference.md) · [Standard Library](docs/stdlib.md) |
| **See what's possible** | [What you get](#what-you-get) · [Examples](#examples) |
| **Stay in the loop** | [★ Star the repo](https://github.com/einlang/einlang) · Watch → Releases |
| **Contribute** | [CONTRIBUTING](CONTRIBUTING.md) — doc fixes and small bugs are a great start |

---

## Install & run

**Install:** Python 3.7+ (tested 3.9–3.12). From repo: `pip install -e .` (deps: numpy, lark, sexpdata).

**Run a file:**
```bash
python3 -m einlang examples/hello.ein
python3 -m einlang path/to/file.ein
```

**Inline (like Python -c / stdin):**
```bash
python3 -m einlang -c "let A = [[1,2],[3,4]]; let B = [[5,6],[7,8]]; let C[i,j] = sum[k](A[i,k]*B[k,j]); print(C);"
echo 'let x = 2; print(x);' | python3 -m einlang -
```

**From Python (use in your project):**
```python
from einlang import run

out = run(source="let A = [[1,2],[3,4]]; let B = [[5,6],[7,8]]; let C[i,j] = sum[k](A[i,k]*B[k,j]); print(C);")
# or: out = run(file="examples/hello.ein")
# out.outputs["C"] → numpy array; out.error if failed
```

Compile-only or custom backend: `from einlang import CompilerDriver, EinlangRuntime` then `compile()` and `execute()`.

---

## What you get

Einlang gives you readable tensor math with compile-time shape checking. In practice that means:

| Feature | What you get |
|--------|---------------|
| **Einstein notation** | `let C[i, j] = sum[k](A[i, k] * B[k, j]);` — indices and shapes checked at compile time |
| **Where clauses** | Index algebra (`where ih = oh + kh`) and guards (`where data[i] > 0`) next to the math |
| **Recurrences** | `let fib[0]=0; let fib[1]=1; let fib[n in 2..20]=fib[n-1]+fib[n-2]` — range in bracket; compiler handles order |
| **Reductions** | `sum[i](x[i])`, `max[j](M[i,j])`, `sum[i,j](A[i,j]*A[i,j])` with inferred ranges |
| **Stdlib** | `use std::math::{sin, sqrt};` · 300+ functions · [Reference](docs/reference.md) · [Stdlib](docs/stdlib.md) |
| **Real models** | [MNIST CNN](examples/mnist/main.ein), [quantized (int8)](examples/mnist_quantized/main.ein), [ViT](examples/deit_tiny/), [Whisper](examples/whisper_tiny/) — same language, same checks |

**Real-world use cases** (same space as [Julia’s demos](docs/JULIA_DEMOS.md): one language for simulation and ML.

| Domain | Use case | Example |
|--------|----------|---------|
| **Scientific simulation** | ODE + PDEs: diffusion, wave, reaction–diffusion (recurrence + stencil) | [ode](examples/ode/), [pde_1d](examples/pde_1d/) (heat, advection), [wave_2d](examples/wave_2d/), [brusselator](examples/brusselator/) |
| **Parameter estimation & scenarios** | Calibrate then forecast; scenario/sensitivity runs (one model, many parameters) | [applications](examples/applications/) |
| **Computer vision** | Digit recognition, int8 quantization, ImageNet ViT | [mnist](examples/mnist/), [mnist_quantized](examples/mnist_quantized/), [deit_tiny](examples/deit_tiny/) |
| **Speech & sequence** | Speech-to-text (encoder–decoder, autoregressive) | [whisper_tiny](examples/whisper_tiny/) |

---

## Why it's different

- **Einstein notation as syntax** — Indices like `i, k, j` are part of the language. The compiler infers ranges from array shapes. Wrong dimensions → compile error, not a runtime crash.  
  `let C[i, j] = sum[k](A[i, k] * B[k, j]);`
- **Where clauses** — Index algebra, guards, and bindings live next to the computation (e.g. conv2d with `ih = oh + kh, iw = ow + kw`).  
  `let out[oh, ow] = sum[kh, kw](input[ih, iw] * kernel[kh, kw]) where ih = oh + kh, iw = ow + kw;`
- **Recurrences as declarations** — Base cases + recursive rule; index range in the bracket (not in `where`); compiler handles evaluation order (RNNs, dynamic programming).  
  `let fib[0]=0; let fib[1]=1; let fib[n in 2..N]=fib[n-1]+fib[n-2];`
- **No stringly-typed einsum** — No `einsum('ik,kj->ij', A, B)`. The compiler sees every index and checks shapes and ranks.

**If it type-checks, the shapes are correct.** That’s the deal: you write the math, the compiler checks the shapes.

---

## Why not NumPy / einsum?

With NumPy you get manual shapes and loops, or `einsum` with string indices — no static checking, no first-class recurrences or index algebra. Einlang keeps the “write the math” feel and adds compile-time shape and index checking.

```rust
let C[i, j] = sum[k](A[i, k] * B[k, j]);
```

---

## Examples

Grouped **by domain**; full list: [examples/README](examples/README.md).

| Domain | Examples | Run |
|--------|----------|-----|
| **Scientific simulation** | ODEs, 1D PDE (heat, advection), 2D wave, Brusselator | [ode/](examples/ode/), [pde_1d/](examples/pde_1d/), [wave_2d/](examples/wave_2d/), [brusselator/](examples/brusselator/) |
| **Discrete dynamics** | Recurrence, Markov, logistic | [recurrence/](examples/recurrence/) |
| **Finance** | Savings / compound interest | [finance/](examples/finance/) |
| **Economics / optimization** | Bellman value iteration; gradient descent, power iteration, projected gradient | [value_iteration/](examples/value_iteration/), [optimization/](examples/optimization/) |
| **Computer vision** | MNIST CNN, quantized CNN, ViT (ImageNet) | [mnist/](examples/mnist/), [mnist_quantized/](examples/mnist_quantized/), [deit_tiny/](examples/deit_tiny/) |
| **Speech & sequence** | Speech-to-text (Whisper) | [whisper_tiny/](examples/whisper_tiny/) |
| **Language & basics** | Variables, matrices, Einstein notation, units | [basics/](examples/basics/), [demos/](examples/demos/), [units/](examples/units/) |

**Quick run:** `python3 -m einlang examples/hello.ein` · `examples/ode/ode_suite.ein` · `examples/optimization/optimization_suite.ein` · `examples/finance/savings.ein` · `examples/job_search/mccall.ein` · `examples/time_series/exponential_smoothing.ein`

Every simulation example has a Julia equivalent in the `.ein` file and is [accuracy-tested](tests/examples/test_simulation_accuracy.py). See [Julia demos → Einlang](docs/JULIA_DEMOS.md).

---

## Docs and roadmap

**[Doc index](docs/README.md)** — by audience (starter, student, ML, engineer, Python/Julia/Rust, contributor, paper).  
**[Getting started](docs/GETTING_STARTED.md)** — one-page story to first example and Python API.  
Canonical: [reference](docs/reference.md) · [stdlib](docs/stdlib.md) · Install & run above. Design: [docs/DOCUMENTATION_DESIGN.md](docs/DOCUMENTATION_DESIGN.md).

**Roadmap:** NumPy backend (now) → MLIR via Python (next) → native/GPU. Einstein notation, where-clauses, recurrences, 300+ stdlib functions, and type and shape inference are in place.

---

## Community

[**★ Star us**](https://github.com/einlang/einlang) — it helps others find Einlang. **Issues and PRs welcome** — [CONTRIBUTING](CONTRIBUTING.md).

## License

Apache 2.0 — see [LICENSE](LICENSE).
