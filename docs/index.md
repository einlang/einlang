
# Einlang

**Tensor code is either readable or safe—usually neither.** Einlang is both: write math in Einstein notation, get shape errors at compile time.

```rust
let C[i, j] = sum[k](A[i, k] * B[k, j]);   // matrix multiply — shapes checked by the compiler
```

**New here?** [Getting started](https://github.com/einlang/einlang/blob/main/docs/GETTING_STARTED.md) tells the full story in one page.

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
| **Get going** | [Getting started](https://github.com/einlang/einlang/blob/main/docs/GETTING_STARTED.md) |
| **Doc index (by audience)** | [Documentation](https://github.com/einlang/einlang/blob/main/docs/README.md) |
| **Language & stdlib** | [Reference](https://github.com/einlang/einlang/blob/main/docs/reference.md) · [Stdlib](https://github.com/einlang/einlang/blob/main/docs/stdlib.md) |
| **Autodiff (gradients / derivatives)** | [AUTODIFF_DESIGN](https://github.com/einlang/einlang/blob/main/docs/AUTODIFF_DESIGN.md) · [examples/autodiff_small.ein](https://github.com/einlang/einlang/blob/main/examples/autodiff_small.ein), [autodiff_matmul.ein](https://github.com/einlang/einlang/blob/main/examples/autodiff_matmul.ein) |
| **Examples (learning path & by domain)** | [examples/README](https://github.com/einlang/einlang/blob/main/examples/README.md) · [examples](https://github.com/einlang/einlang/tree/main/examples) |

**Repo:** [github.com/einlang/einlang](https://github.com/einlang/einlang) · **Contribute:** [CONTRIBUTING](https://github.com/einlang/einlang/blob/main/CONTRIBUTING.md)
