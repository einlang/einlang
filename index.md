---
layout: default
title: Einlang
---

# Einlang

**Math on the page. Indices, sums, shapes—the compiler reads them too.**

[Install](https://github.com/einlang/einlang#readme) · [Docs](https://github.com/einlang/einlang/blob/main/docs/GETTING_STARTED.md) · [Repo](https://github.com/einlang/einlang)

---

## Einlang in a Nutshell

### Readable

You write the notation you’d use on a whiteboard. Indices and sums in the open—no string subscripts, no hidden loops.

### Checked

Wrong shape? The compiler tells you before you run. If it type-checks, shapes are correct. Same guarantee for ODEs, PDEs, linalg, RNNs, and models like MNIST or Whisper-style.

### Inferred

Index ranges come from array shapes. Sum over `k`? Inferred from the tensors. Convolution over `i`? Inferred so `i + k` stays in bounds. You can write ranges explicitly when you want.

### One notation

Einstein declarations, recurrences, where clauses, and rest index variables (e.g. `..batch`) live in one language. One toolchain, one story.

---

## Explore

### Matrix multiply

```rust
let C[i, j] = sum[k](A[i, k] * B[k, j]);
```

Shapes checked. That’s it.

### Sums and norms

Same bracket notation you write on paper:

```rust
let n = sqrt(sum[i](x[i] * x[i]));
let n = sqrt(sum[i, j](A[i, j] * A[i, j]));
```

### Recurrence

Declare base cases and step; the compiler figures the order:

```rust
let fib[0] = 0;
let fib[1] = 1;
let fib[n in 2..25] = fib[n - 1] + fib[n - 2];
```

### Convolution

Index algebra in the open. Ranges inferred: `k` from `kernel`, then `i` from `signal` (so `i + k` stays in bounds).

```rust
let convolved[i] = sum[k](signal[i + k] * kernel[k]);
```

### ODEs (e.g. Lorenz)

Step by step, same recurrence style:

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
```

### More

- **Where clauses** — index constraints next to the math (e.g. `where ih = oh + kh`).
- **Rest index variables** — e.g. `..batch` in the bracket for trailing dimensions; the compiler infers the rest.

---

## Try it

```bash
pip install -e .
python3 -m einlang -c "let C[i,j] = sum[k](A[i,k]*B[k,j]);"
```

[Open the repo →](https://github.com/einlang/einlang) · [Getting started](https://github.com/einlang/einlang/blob/main/docs/GETTING_STARTED.md)
