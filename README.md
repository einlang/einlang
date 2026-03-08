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
| **Contribute** | [CONTRIBUTING.md](CONTRIBUTING.md) — doc fixes and small bugs are a great start |

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

From one-liners to full models: run by **feature** (one capability at a time) or follow the **learning path** from basics to CNN, quantized CNN, ViT, and Whisper. Full path: [examples/README.md](examples/README.md).

| Feature | Run this |
|--------|----------|
| Einstein matmul + print | `python3 -m einlang examples/hello.ein` |
| Matrix ops, norms, stats | [matrix_operations.ein](examples/demos/matrix_operations.ein) |
| Reductions, contractions | [reduction_operations.ein](examples/units/reduction_operations.ein) |
| Where constraints | [where_constraints.ein](examples/units/where_constraints.ein) |
| Convolution-style indexing | [convolution_operations.ein](examples/units/convolution_operations.ein) |
| Functions + overloading | [functions_demo.ein](examples/basics/functions_demo.ein), [function_overloading_complete.ein](examples/demos/function_overloading_complete.ein) |
| Full CNN (MNIST) | [mnist/main.ein](examples/mnist/main.ein) |
| Quantized CNN (int8) | [mnist_quantized/main.ein](examples/mnist_quantized/main.ein) |
| ViT / Whisper | [deit_tiny/](examples/deit_tiny/), [whisper_tiny/](examples/whisper_tiny/) |
| PDE simulations | [heat_animation.py](examples/heat_animation.py) (diffusion), [wave_2d/](examples/wave_2d/) (acoustic wave) |

| Step | Run | What it is |
|------|-----|------------|
| **0** | [hello.ein](examples/hello.ein) | Intro: matmul + print |
| 1 | [basics/](examples/basics/), [demos/](examples/demos/) | Variables, functions, matrices, Einstein notation |
| 2 | [mnist/main.ein](examples/mnist/main.ein) | CNN digit recognition |
| 2b | [mnist_quantized/main.ein](examples/mnist_quantized/main.ein) | Same CNN with int8 weights (`qconv`, `qlinear`, `quantize_linear`) |
| 3 | [deit_tiny/](examples/deit_tiny/), [whisper_tiny/](examples/whisper_tiny/) | Vision Transformer, speech-to-text |
| 3b | [heat_animation.py](examples/heat_animation.py), [wave_2d/](examples/wave_2d/) | 2D heat and wave equation (recurrence + stencil) |

More in the [examples/](examples/) tree.

---

## Docs and roadmap

**[Doc index](docs/README.md)** — by audience (starter, student, ML, engineer, Python/Julia/Rust, contributor, paper).  
**[Getting started](docs/GETTING_STARTED.md)** — one-page story to first example and Python API.  
Canonical: [reference](docs/reference.md) · [stdlib](docs/stdlib.md) · Install & run above. Design: [docs/DOCUMENTATION_DESIGN.md](docs/DOCUMENTATION_DESIGN.md).

**Roadmap:** NumPy backend (now) → MLIR via Python (next) → native/GPU. Einstein notation, where-clauses, recurrences, 300+ stdlib functions, and type and shape inference are in place.

---

## Community

[**★ Star us**](https://github.com/einlang/einlang) — it helps others find Einlang. **Issues and PRs welcome** — [CONTRIBUTING.md](CONTRIBUTING.md).

## License

Apache 2.0 — see [LICENSE](LICENSE).
