
# Einlang

Einlang is a language and compiler for tensor programs written with explicit indices, reductions, recurrences, and derivatives.

Automatic differentiation is part of the language: `@expr`, `@y / @x`, and `@C / @A` compile directly into derivative computations.

```rust
let C[i, j] = sum[k](A[i, k] * B[k, j]);   // matrix multiply — shapes checked by the compiler
```

Start with [Getting started](https://github.com/einlang/einlang/blob/main/docs/GETTING_STARTED.md).

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
| **Why Einlang? (features & comparison)** | [Why Einlang](https://github.com/einlang/einlang/blob/main/docs/WHY_EINLANG.md) — notation, static checking, recurrences, and autodiff |
| **Doc index** | [Documentation](https://github.com/einlang/einlang/blob/main/docs/README.md) |
| **Language & stdlib** | [Reference](https://github.com/einlang/einlang/blob/main/docs/reference.md) · [Stdlib](https://github.com/einlang/einlang/blob/main/docs/stdlib.md) |
| **Autodiff (gradients / derivatives)** | **Built-in automatic differentiation** — compiler derives gradients from `@expr` and `@a / @b` directly on the expressions you wrote; no hand-written gradient code and no separate `grad(f)` layer. For supported ops, this is the replacement for finite-difference gradient estimates. [AUTODIFF_HIGHLIGHTS](https://github.com/einlang/einlang/blob/main/docs/AUTODIFF_HIGHLIGHTS.md) · [AUTODIFF_DESIGN](https://github.com/einlang/einlang/blob/main/docs/AUTODIFF_DESIGN.md) · [examples/autodiff_small.ein](https://github.com/einlang/einlang/blob/main/examples/autodiff_small.ein), [autodiff_matmul.ein](https://github.com/einlang/einlang/blob/main/examples/autodiff_matmul.ein) |
| **Examples (learning path & by domain)** | [examples/README](https://github.com/einlang/einlang/blob/main/examples/README.md) · [examples](https://github.com/einlang/einlang/tree/main/examples) |

**Repo:** [github.com/einlang/einlang](https://github.com/einlang/einlang) · **Contribute:** [CONTRIBUTING](https://github.com/einlang/einlang/blob/main/CONTRIBUTING.md)
