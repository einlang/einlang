---
title: Einlang
---

# Einlang

**Tensor code is either readable or safe—usually neither.** Einlang is both: write math in Einstein notation, get shape errors at compile time.

```rust
let C[i, j] = sum[k](A[i, k] * B[k, j]);   // matrix multiply — shapes checked by the compiler
```

**New here?** [Getting started](docs/GETTING_STARTED.md) tells the full story in one page.

---

## Try it

```bash
git clone https://github.com/einlang/einlang.git
cd einlang
pip install -e .
python3 -m einlang -c "let x = 1+1; print(x);"
```

Run a real example: `python3 -m einlang examples/hello.ein`

---

## Docs

| You want to… | Go here |
|--------------|--------|
| **Get going** | [Getting started](docs/GETTING_STARTED.md) |
| **Doc index (by audience)** | [Documentation](docs/README.md) |
| **Language & stdlib** | [Reference](docs/reference.md) · [Stdlib](docs/stdlib.md) |
| **Examples (learning path & by domain)** | [examples/README.md](examples/README.md) · [GitHub: examples](https://github.com/einlang/einlang/tree/main/examples) |

**Repo:** [github.com/einlang/einlang](https://github.com/einlang/einlang) · **Contribute:** [CONTRIBUTING.md](CONTRIBUTING.md)
