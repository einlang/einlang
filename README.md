# Einlang

[![Tests](https://github.com/einlang/einlang/actions/workflows/tests.yml/badge.svg)](https://github.com/einlang/einlang/actions/workflows/tests.yml)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)

---

A tensor has shape `(32, 64, 256)`. The data loader author knows these are `batch`, `channel`, and `spatial`. Then:

```python
x = x.mean(dim=1)
```

`dim=1` erases a dimension. At the time of writing, position 1 holds `channel`. The intent is "average over channels."

Three months later, the data pipeline is refactored. Channel moves to position 2. The shape is now `(32, 256, 64)`. `mean(dim=1)` now silently erases `spatial`. No errors. No warnings. The loss descends. The model deploys. The customer complaint arrives on Thursday.

**The notation had no slot for the fact that would have caught it.**

Now imagine a different notation:

```rust
let y[b, s] = mean[channel](x[b, channel, s]);
```

The bracket after `mean` names the coordinate being consumed. That `channel` exists on `x` is statically checked. The fact that was in a comment is now in the syntax, where the compiler can enforce it.

---

Einlang is an experimental language where tensor coordinates have names. The names are checked. That is the whole idea.

**[The Name in the Bracket](https://einlang.github.io/einlang/book/)** is a free book that makes the argument in full. Every example runs.

## At a glance

```rust
let C[i, j] = sum[k](A[i, k] * B[k, j]);            // matrix multiply
let p[b, class] = softmax[class](logits[b, class]);  // normalize over the named coordinate
let dC_dA = @C / @A;                                 // built-in autodiff
let fib[n in 2..25] = fib[n - 1] + fib[n - 2];       // recurrence
```

| Instead of... | In Einlang... |
|---------------|---------------|
| `x.mean(dim=1)` — axis 1 is channel today, spatial tomorrow | `mean[channel](x)` — the bracket says what the number cannot |
| `y = np.einsum("bi,ci->bc", x, W) + bias` | `let y[b, c] = sum[i](x[b, i] * W[c, i]) + bias[c];` |
| `dloss_dW = jax.grad(loss_fn)(W)` | `let dloss_dW = @loss / @W;` |

When `batch == time == feature == 128`, normalizing over each produces the same output shape. The positional calls differ by a single integer. The bracket calls differ by a name:

```python
# PyTorch — all produce shape [128, 128, 128]. Which is which?
F.layer_norm(x, (128,))               # dim=-1
(x - x.mean(1, True)) / x.std(1, True) # dim=1
(x - x.mean(0, True)) / x.std(0, True) # dim=0
```

```rust
// Einlang — the bracket carries the intent
normalize_over[feature](x[batch, time, feature]);
normalize_over[time](x[batch, time, feature]);
normalize_over[batch](x[batch, time, feature]);
```

## How is this different from einops?

Einops gives you string-based named rearrangement:

```python
rearrange(x, "batch channel spatial -> batch spatial channel")
```

The names are local to the string. They are not checked against any declaration. If the tensor actually contains `time` rather than `channel`, the string won't catch it.

In Einlang, coordinates are part of the type. `softmax[class](logits)` is checked: the compiler verifies that `logits` actually has a `class` coordinate. Rename `class` to `category` upstream, and every `softmax[class]` call becomes a compile error naming the missing coordinate. The einops string stays a string.

Einops is a rearrangement tool. Einlang is a language where coordinate identity survives composition — through function calls, through differentiation, through recurrence. The same coordinate name threads from the forward pass through the gradient.

## Quick start

```bash
git clone https://github.com/einlang/einlang.git && cd einlang
python3 -m pip install -e .
python3 -m einlang -c "let x = 1 + 1; print(x);"   # prints 2
python3 -m einlang examples/hello.ein                # matrix multiplication
python3 -m einlang examples/autodiff_small.ein       # @y / @x in action
```

`examples/hello.ein` multiplies two small matrices:

```rust
let A = [[1, 2], [3, 4]];
let B = [[5, 6], [7, 8]];
let C[i, j] = sum[k](A[i, k] * B[k, j]);
```

Read it as: for each output position `(i, j)`, multiply matching entries from row `i` of `A` and column `j` of `B`, then add over `k`.

## What it is good at

- Tensor expressions without string-based `einsum`
- Gradients written directly in the language with `@loss / @weights`
- Coordinate-aware function calls: `softmax[class]`, `argmax[class]`
- Recurrences, dynamic programs, and time evolution
- Running from the CLI or from Python on a NumPy backend today

## What it is not

- Not a general-purpose replacement for Python
- Not limited to machine learning — the examples span ODEs, optimization, simulation
- Not a finished GPU stack — NumPy is the main path today, IREE is experimental

## Python integration

```python
from einlang import run

out = run(file="examples/hello.ein")
# or: out = run(source="let x = 1 + 1; print(x);")
```

## Status

Einlang runs on a NumPy backend. An IREE path is in progress behind the optional `iree` extra. The language, standard library, tests, and example suite are in active development.

## Going deeper

- **[The Name in the Bracket](https://einlang.github.io/einlang/book/)** — a free book, 15 chapters + epilogue. The argument in full.
- [examples/README](examples/README.md) — browse runnable programs
- [docs/reference.md](docs/reference.md) — syntax and semantics
- [docs/AUTODIFF.md](docs/AUTODIFF.md) — autodiff in detail
- [docs/COORDINATE_FUNCTIONS.md](docs/COORDINATE_FUNCTIONS.md) — coordinate-aware calls
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — implementation
- [CONTRIBUTING](CONTRIBUTING.md) — how to get involved

## License

Apache 2.0. See [LICENSE](LICENSE).
