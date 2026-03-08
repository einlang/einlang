# Getting started with Einlang

One page: what Einlang is, try it, run your first real example, use it from Python, then where to go next.

---

## What is Einlang?

Einlang is a language for tensor math that looks like the notation you’d write on a whiteboard — **Einstein notation** — and checks shapes at **compile time**. No stringly-typed `einsum`, no shape bugs at 3am. If it type-checks, the shapes are correct.

```rust
let C[i, j] = sum[k](A[i, k] * B[k, j]);   // matrix multiply — compiler checks A, B, C shapes
```

You get **where-clauses** (index algebra, guards), **recurrences** (RNNs, dynamic programming), and a **stdlib** of 300+ functions. Real models — CNN, quantized CNN, ViT, Whisper — are written in the same language.

---

## Try it (30 seconds)

From a terminal:

```bash
git clone https://github.com/einlang/einlang.git
cd einlang
pip install -e .
python3 -m einlang -c "let x = 1+1; print(x);"
```

You should see `2`. No account, no config.

---

## First real example

Run the matrix multiply from the code block above:

```bash
python3 -m einlang examples/hello.ein
```

That’s real Einlang: indices `i`, `j`, `k`, shape checking, and output. From here you can [run more examples by feature](../README.md#examples) or follow the [learning path](../examples/README.md) from basics to MNIST to ViT and Whisper.

---

## Use it in your project

Install with `pip install -e .` from the repo, then:

```python
from einlang import run

out = run(file="examples/hello.ein")
# or: out = run(source="let C[i,j] = sum[k](A[i,k]*B[k,j]); ...")
# out.outputs["C"]  → numpy array; out.error if something failed
```

One call and you’re a user. See [Install & run](../README.md#install--run) in the main README for `-c`, stdin, and the compiler API.

---

## Where to go next

| You want to… | Go here |
|--------------|--------|
| **Learn the language** | [Language Reference](reference.md) — syntax, types, Einstein notation, where-clauses, recurrences |
| **Look up functions** | [Standard Library](stdlib.md) — math, arrays, ML ops |
| **Run examples by feature or step** | [README — Examples](../README.md#examples) · [Examples guide](../examples/README.md) |
| **See the full doc map** | [Docs index](README.md) |
| **Contribute** | [CONTRIBUTING.md](../CONTRIBUTING.md) — doc fixes and small bugs are a great start |

You’re set. The rest is depth: reference, stdlib, and examples showcase everything Einlang can do.
