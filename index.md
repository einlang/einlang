---
layout: default
title: Einlang
---

# Einlang

**Math on the page. Indices, sums, shapes—the compiler reads them too.**

Wrong shape? It tells you before you run. No string indices. No guessing. The notation you’d use on a whiteboard, with a type system that’s actually watching.

---

## One line: matrix multiply

```rust
let C[i, j] = sum[k](A[i, k] * B[k, j]);
```

Shapes checked. That’s it.

---

## Explore

**Sums and norms** — same bracket notation you write on paper:

```rust
let n = sqrt(sum[i](x[i] * x[i]));           // L2 norm
let n = sqrt(sum[i, j](A[i, j] * A[i, j])); // Frobenius
```

**Recurrences** — declare base + step; the compiler figures the order:

```rust
let fib[0] = 0;
let fib[1] = 1;
let fib[n in 2..25] = fib[n - 1] + fib[n - 2];
```

**Convolution** — index algebra in the open:

```rust
let convolved[i] = sum[k](signal[i + k] * kernel[k]);
```

Ranges inferred: `k` from `kernel`, then `i` from `signal` (so `i + k` stays in bounds).

**ODEs** — e.g. Lorenz, step by step:

```rust
let u[0, 0] = x0;
let u[0, 1] = y0;
let u[0, 2] = z0;
let u[t in 1..150, 0] = {
  let x = u[t-1, 0];
  let y = u[t-1, 1];
  let z = u[t-1, 2];
  x + dt * (sigma * (y - x))
};
// ... dy, dz the same way
```

- **Where clauses** (in Einlang) — index constraints next to the math (e.g. `where ih = oh + kh`).
- **Rest index variables** (in Einlang) — e.g. `..batch` in the bracket for trailing dimensions; the compiler infers the rest.
- **One guarantee** (in Einlang) — ODEs, PDEs, linalg, RNNs, MNIST, ViT, Whisper-style: if it type-checks, shapes are correct.

---

## Try it

```bash
pip install -e .
python3 -m einlang -c "let C[i,j] = sum[k](A[i,k]*B[k,j]);"
```

[Open the repo →](https://github.com/einlang/einlang) · [Getting started](https://github.com/einlang/einlang/blob/main/docs/GETTING_STARTED.md)
