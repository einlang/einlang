---
layout: book
title: "Appendices"
---

# Appendices

## Appendix A: Syntax Cheat Sheet

```rust
// scalar binding
let x = 5.0;

// indexed tensors
let y[i] = x[i] + 1.0;
let M[i, j] = A[i, j] + B[i, j];

// explicit ranges
let v[i in 0..10] = i;    // end-exclusive
let u[i in 0..=10] = i;   // end-inclusive

// reductions
let total = sum[i](x[i]);
let row_max[i] = max[j](A[i, j]);
let product = prod[i](x[i]);

// index arithmetic and guards
let conv[oh, ow] = sum[kh, kw](I[oh + kh, ow + kw] * K[kh, kw]);
let pos = sum[i](x[i]) where x[i] > 0.0;

// named rest patterns
let sums[..batch] = sum[k](x[..batch, k]);

// recurrence
let fib[0] = 0;
let fib[1] = 1;
let fib[n in 2..20] = fib[n - 1] + fib[n - 2];

// autodiff
let dy_dx = @y / @x;

// Python interop
let arr = python::numpy::array([1, 2, 3]);
```

## Appendix B: Formula Mapping Table

| Formula | Einlang |
| --- | --- |
| `y = x^2` | `let y = x * x;` |
| `y_i = x_i + b` | `let y[i] = x[i] + b;` |
| `C_{i,j} = A_{i,j} + B_{i,j}` | `let C[i, j] = A[i, j] + B[i, j];` |
| `dot = sum_i x_i y_i` | `let dot = sum[i](x[i] * y[i]);` |
| `C_{i,j} = sum_k A_{i,k} B_{k,j}` | `let C[i, j] = sum[k](A[i, k] * B[k, j]);` |
| `A^T_{j,i} = A_{i,j}` | `let AT[j, i] = A[i, j];` |
| `row_sum_i = sum_j A_{i,j}` | `let row_sum[i] = sum[j](A[i, j]);` |
| `F_n = F_{n-1} + F_{n-2}` | `let fib[n in 2..N] = fib[n - 1] + fib[n - 2];` |
| `d y / d x` | `let dy_dx = @y / @x;` |

## Appendix C: Common Static Errors

| Category | Example | Meaning |
| --- | --- | --- |
| Shape mismatch | `A[i, k] * B[k, j]` with different `k` sizes | Shared index ranges disagree. |
| Invalid index | `A[4, j]` for a 3-row matrix | Constant subscript is out of bounds. |
| Future recurrence read | `h[t] = h[t + 1] + 1` | A point reads a value not computed yet. |
| Undetermined rest | `let y[..batch] = x[i];` | `..batch` is not determined by an input access. |
| Jagged Einstein input | `jagged[i, j]` | Einstein notation requires rectangular arrays. |
| Unsupported derivative path | `@y / @x` through unsupported operations | Einlang should report an error rather than silently approximate. |

## Appendix D: Python Interop Practices

Use Python for data loading, filesystem work, plotting, tokenizers, and existing
numerical kernels that are not the point of the experiment. Use Einlang for the
part where named tensor structure matters.

Good boundary code documents shape:

```rust
use std::io::load_npy;

let W = load_npy("weights/W.npy") as [f32; 128, 10];
let labels = python::mnist_data::load_train_labels() as [f32; 1437, 10];
```

Prefer casts at the boundary. A value returned from Python may have dynamic
shape or dynamic rank. Casting it to `[f32; 128, 10]` documents the contract and
gives the compiler shape information for later indexed code.
