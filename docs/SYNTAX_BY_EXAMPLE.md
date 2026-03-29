# Syntax by example

This page is for readers who want to see real Einlang syntax before reading the full [reference](https://github.com/einlang/einlang/blob/main/docs/reference.md).

If you want runnable examples after this page, go to [examples/README](https://github.com/einlang/einlang/blob/main/examples/README.md).

---

## 1. Einstein notation

Einlang uses explicit indices in the source. Reductions such as matrix multiply are written directly:

```rust
let C[i, j] = sum[k](A[i, k] * B[k, j]);
```

This is the core style of the language: tensor structure is visible, and the compiler checks the index usage and shapes before execution.

---

## 2. Range-bound declarations

You can bind index ranges directly in the declaration bracket:

```rust
let eye[i in 0..n, j in 0..n] = if i == j { 1.0 } else { 0.0 };
let diag[i in 0..n] = eye[i, i];
```

This is common when you want a tensor with an explicit rectangular shape.

---

## 3. Where-clauses

Use `where` to attach guards or index relations to a declaration or reduction.

Simple guard:

```rust
let upper[i, j] = matrix[i, j] where i <= j;
```

Index relation next to the math:

```rust
let conv[b, oc, oh, ow] = sum[ic, kh, kw](
    input[b, ic, ih, iw] * weight[oc, ic, kh, kw]
) where ih = oh + kh, iw = ow + kw;
```

The point is that the index logic stays close to the computation instead of being hidden in loops.

---

## 4. Named rest patterns

Named rest patterns such as `..batch` stand for zero or more leading dimensions. They let tensor code stay rank-generic:

```rust
let row_sum[..batch] = sum[j](X[..batch, j]);
let centered[..batch, j] = X[..batch, j] - row_sum[..batch];
```

Pooling is a typical example:

```rust
let pooled[..batch, c, i, j] = max[m, n](
    X[..batch, c, i * 2 + m, j * 2 + n]
);
```

The same `..batch` name means “the same pack of dimensions” everywhere it appears in the declaration.

---

## 5. Built-in autodiff

Derivatives are language syntax, not a separate API:

```rust
let z = x + y;
let dz_dx = @z / @x;
let dz_dy = @z / @y;
```

Tensor-valued derivatives work the same way:

```rust
let C[i, j] = sum[k](A[i, k] * B[k, j]);
let dC_dA = @C / @A;
```

Loss/parameter code reads directly as math:

```rust
let loss = sum[i]((pred[i] - target[i]) ** 2.0);
let dloss_dpred = @loss / @pred;
```

---

## 6. Custom autodiff rules

When the primal is foreign or intentionally opaque, define a rule with `@fn`:

```rust
fn exp(x) { python::numpy::exp(x) }
@fn exp(x) { exp(x) * @x }
```

This keeps the primal definition and derivative rule close together.

---

## 7. Recurrences

Recurrences are written as declarations with base cases plus later indices that read earlier ones.

Fibonacci:

```rust
let fib[0] = 0;
let fib[1] = 1;
let fib[n in 2..25] = fib[n - 1] + fib[n - 2];
```

A simple time-step recurrence:

```rust
let steps = [1, -1, 1, 1, -1];
let position[0] = 0;
let position[t in 1..6] = position[t - 1] + steps[t - 1];
```

A recurrence can also keep extra dimensions:

```rust
let psi[0, s in 0..3] = 1.0 / 3.0;
let psi[k in 1..50, j in 0..3] = {
    let row = psi[k - 1, 0] * P[0, j]
            + psi[k - 1, 1] * P[1, j]
            + psi[k - 1, 2] * P[2, j];
    row
};
```

For more recurrence examples, see [examples/recurrence/README](https://github.com/einlang/einlang/blob/main/examples/recurrence/README.md).

---

## 8. A small mixed example

Here is the language shape in one place:

```rust
fn exp(x) { python::numpy::exp(x) }
@fn exp(x) { exp(x) * @x }

let hidden[..batch, j] = sum[k](X[..batch, k] * W[k, j]) + b[j];
let activated[..batch, j] = exp(hidden[..batch, j]);
let row_sum[..batch] = sum[j](activated[..batch, j]);
let probs[..batch, j] = activated[..batch, j] / row_sum[..batch];
let loss = sum[..batch, j]((targets[..batch, j] - probs[..batch, j]) ** 2.0);
let dloss_dW = @loss / @W;
```

That combines explicit indices, a reduction, a named rest pattern, a custom autodiff rule, and a derivative expression in ordinary program syntax.

---

## Next

- [Getting started](https://github.com/einlang/einlang/blob/main/docs/GETTING_STARTED.md)
- [Language reference](https://github.com/einlang/einlang/blob/main/docs/reference.md)
- [Why Einlang](https://github.com/einlang/einlang/blob/main/docs/WHY_EINLANG.md)
- [examples/README](https://github.com/einlang/einlang/blob/main/examples/README.md)
