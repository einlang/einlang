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

**The notation had no slot for the fact that would have caught it.** The fact—"erasing `channel`, not `spatial`"—was in a comment, in a variable name, in the author's head. It was absent from the one place the compiler could see.

Now imagine a different notation:

```rust
let y[b, s] = mean[channel](x[b, channel, s]);
```

The bracket after `mean` names the coordinate being consumed. That `channel` exists on `x` is statically checked. The reader sees the consumption without reconstructing it. The fact that was in a comment is now in the syntax, where the compiler can enforce it.

You park your car in Row D, Slot 7. Your ticket says "D-7." You return to find the lot repainted—rows reoriented, Row D now somewhere else. The ticket sends you to the wrong car. This is what happens when you write `x.mean(dim=1)` and upstream changes the dimension order. A named-coordinate notation is a ticket that says "the blue Honda Civic." The car may move. The description finds it.

---

Einlang is an experimental language and compiler for tensor programs. Coordinates have names. The names are checked. This is the whole idea.

**[The Name in the Bracket](https://einlang.github.io/einlang/book/)** is a free book that makes the argument in full. Every example runs.

> Every time you write `dim=-1`, you know what it means. The compiler doesn't. This is about what happens when it does.

## Quick start

```bash
git clone https://github.com/einlang/einlang.git && cd einlang
python3 -m pip install -e .
python3 -m einlang -c "let x = 1 + 1; print(x);"   # prints 2
python3 -m einlang examples/hello.ein                # matrix multiplication
python3 -m einlang examples/autodiff_small.ein       # @y / @x in action
```

## Syntax at a glance

```rust
let C[i, j] = sum[k](A[i, k] * B[k, j]);            // matrix multiply
let p[b, class] = softmax[class](logits[b, class]);  // coordinate-aware call
let dC_dA = @C / @A;                                 // built-in autodiff
let fib[n in 2..25] = fib[n - 1] + fib[n - 2];       // recurrence
```

| Notation | Meaning |
|----------|---------|
| `let x = 3.0;` | Bind an immutable value |
| `let C[i, j] = ...;` | Build `C` by output position |
| `sum[k](expr)` | Add up `expr` over all values of `k` |
| `softmax[class](logits)` | Name the coordinate this function operates on |
| `@y / @x` | Derivative of `y` with respect to `x` |
| `let fib[n in 2..25] = ...` | Fill in values across a range using earlier values |

## Why it feels different

| Instead of... | In Einlang... |
|---------------|---------------|
| `y = np.einsum("bi,ci->bc", x, W) + bias` | `let y[b, c] = sum[i](x[b, i] * W[c, i]) + bias[c];` |
| `dloss_dW = jax.grad(loss_fn)(W)` | `let dloss_dW = @loss / @W;` |
| `x.mean(dim=1)` — axis 1 is time today, feature tomorrow | `mean[time](x)` — the bracket says what the number cannot |

When `batch == time == feature == 128`, normalizing over each produces the same output shape. The positional calls differ by a single integer. The bracket calls differ by a name:

```python
# PyTorch — all produce shape [128, 128, 128]. Which is which?
normed_feature = F.layer_norm(x, (128,))               # dim=-1
normed_time    = (x - x.mean(1, True)) / x.std(1, True) # dim=1
normed_batch   = (x - x.mean(0, True)) / x.std(0, True) # dim=0
```

```rust
// Einlang — the bracket carries the intent
let normed_feature = normalize_over[feature](x[batch, time, feature]);
let normed_time    = normalize_over[time](x[batch, time, feature]);
let normed_batch   = normalize_over[batch](x[batch, time, feature]);
```

## Syntax at a Glance

Most new readers do not need the full language reference first. These are the five patterns that matter most:

| Syntax | Read it as |
|--------|------------|
| `let x = 3.0;` | Bind an immutable value named `x`. |
| `let C[i, j] = ...;` | Build `C` one output position at a time, using `i` and `j` as the row/column-style positions. |
| `sum[k](expr)` | Add up `expr` over all values of `k`. |
| `softmax[class](logits)` | Call a coordinate-aware function and name the coordinate it normalizes, selects, or otherwise uses. |
| `@y / @x` | Ask for the derivative of `y` with respect to `x`. |
| `let fib[n in 2..25] = fib[n - 1] + fib[n - 2];` | Fill in `fib` across a range using earlier values of `fib`. |

For example, this line:

```rust
let C[i, j] = sum[k](A[i, k] * B[k, j]);
```

is just matrix multiplication written in indexed form:

- `i` and `j` are the output positions
- `k` is the dimension being added over
- if the sizes do not line up, you get an error before execution instead of a surprising runtime failure

That same pattern shows up again and again as the programs get larger.

## What it is good at

- Gradients written directly in the language with `@loss / @weights`
- Matrix and tensor expressions without string-based `einsum`
- Coordinate-aware function calls for axis-sensitive operations such as `softmax[class]` and `argmax[class]`
- Step-by-step definitions for sequences, dynamic programs, and time evolution
- Running from the CLI or from Python on a NumPy backend today
- Staying in one notation as you move from tiny examples to fitting, numerics, and optimization

## What it is not

- Not a general-purpose replacement for Python for every part of your project
- Not limited to machine learning; the same language is used for recurrences, simulations, and optimization examples
- Not a finished GPU-first stack yet; today the main execution path is NumPy and IREE is still experimental

## A one-minute taste

```rust
let x = 1.0;
let y = x * x + 3.0 * x;
let dy_dx = @y / @x;
print(dy_dx);
```

When `x = 1`, this prints `5.0`.

That small example captures the central idea: the derivative request lives in the same code as the quantity you care about, not in a separate wrapper API.

## Start here

If you want the shortest path from a fresh checkout to a real program, run this from the repository root. You do not need any other doc first:

```bash
git clone https://github.com/einlang/einlang.git
cd einlang
python3 -m pip install -e .
python3 -m einlang examples/hello.ein
```

`examples/hello.ein` is intentionally tiny. It proves the toolchain works, shows the CLI on a complete program, and lets you start from a real file instead of setup notes.

It multiplies two small matrices and prints:

```text
A * B =
[[19, 22], [43, 50]]
```

It is also a good first piece of syntax to read:

```rust
let A = [[1, 2], [3, 4]];
let B = [[5, 6], [7, 8]];
let C[i, j] = sum[k](A[i, k] * B[k, j]);
```

Read it as: "make `C`; for each output position `(i, j)`, multiply matching entries from row `i` of `A` and column `j` of `B`, then add over `k`."

For an even quicker sanity check after install, this prints `2`:

```bash
python3 -m einlang -c "let x = 1 + 1; print(x);"
```

## The first five minutes

The next program to run is `examples/autodiff_small.ein`:

```bash
python3 -m einlang examples/autodiff_small.ein
```

This is usually where the language clicks. The derivative request sits right next to the original expression instead of sending you to a separate autodiff API.

You do not need to understand every token on first read. The important thing to notice is that the math stays visible and the gradient request stays local.

That file uses `x = 1.0` and `y = 2.0`, then asks for derivatives of `x + y`, `x * y`, `x - y`, and `x / y`. A few representative results are:

- for `w = x * y`, `dw_dx = 2` and `dw_dy = 1`
- for `v = x / y`, `dv_dx = 0.5` and `dv_dy = -0.25`

So the point is more than syntax. The example shows that derivative requests are normal executable parts of the program.

Once that feels comfortable, widen the scope a little:

```bash
python3 -m einlang examples/demos/matrix_operations.ein
python3 -m einlang examples/applications/linear_regression_autodiff.ein
```

`examples/demos/matrix_operations.ein` keeps the indexed matrix notation front and center. `examples/applications/linear_regression_autodiff.ein` carries the same syntax into a small fitting loop.

If you only run one more example after `autodiff_small.ein`, make it `linear_regression_autodiff.ein`. It is usually the point where the language stops feeling like a neat syntax trick and starts feeling like a usable tool.

By this point, without leaving the README, you have already seen the core promise of Einlang in sequence:

- a complete runnable program
- direct autodiff syntax
- explicit array notation with visible indices
- the same language applied to a small real optimization task

## The first half hour

From there, the best way to learn Einlang is to see the same notation survive across very different workloads:

```bash
python3 -m einlang examples/recurrence/recurrence_suite.ein
python3 -m einlang examples/ode/ode_suite.ein
python3 -m einlang examples/optimization/optimization_suite.ein
```

The three examples are independent; the closest one is enough for a first pass.

- sequences, evolving state, or dynamic programs: `examples/recurrence/recurrence_suite.ein`
- time stepping and simulation: `examples/ode/ode_suite.ein`
- fitting loops and iterative methods: `examples/optimization/optimization_suite.ein`

Those three examples answer the practical question most people have by this point: does the language still feel coherent once the introductory examples are gone? That is the real promise of the project, and these are the examples that make it concrete.

By the end of that path, you have seen Einlang handle:

- tensor algebra
- built-in autodiff
- recurrence-style programs
- simulation-style workflows
- optimization-oriented examples

For heavier showcases or directory-specific setup, [examples/README](examples/README.md) maps the larger repo.

## Common first questions

**Do I need to already know Einstein notation?**

No. You can read `let C[i, j] = ...` as "build `C` by output position." The notation becomes natural quickly once you see a few examples.

**Is this only for ML?**

No. Autodiff is built in, but the repo also leans heavily on recurrences, ODEs, PDEs, optimization, and other numerical workflows.

**Do I have to use the CLI?**

No. The CLI is the easiest first run, but you can also call Einlang from Python.

**Is the project experimental?**

Yes, in the honest sense that it is still growing. The NumPy path is the main one today, the example suite is large, and the IREE path is still in progress.

## Python Integration

If you would rather stay in Python while trying things out, that works too:

```python
from einlang import run

out = run(file="examples/hello.ein")
# or: out = run(source="let x = 1 + 1; print(x);")
```

`out.success` tells you whether the run worked, `out.outputs` contains the produced values, and `out.error` is set if something failed.

## Install without cloning

If you want the tool first and the repository later, this is the shortest route:

```bash
python3 -m pip install "git+https://github.com/einlang/einlang.git"
python3 -m einlang -c "let x = 1 + 1; print(x);"
```

If you want the latest formal release artifact instead of the current `main` branch:

```bash
python3 -m pip install "https://github.com/einlang/einlang/releases/latest/download/einlang-latest-py3-none-any.whl"
```

## Optional: try the IREE backend

The IREE path is still in progress, but you can enable it with the optional extra if you want to experiment:

```bash
python3 -m pip install -e ".[iree]"
python3 -m einlang --backend iree examples/hello.ein
```

Supported functions compile through IREE; anything outside the current subset falls back to NumPy.

## After the first session

If you want to keep going after the path above, here is the practical split. None of these are required for the first run:

- [examples/README](examples/README.md) is the place to browse more runnable programs, especially the heavier showcases with their own setup details.
- [The Name in the Bracket](https://einlang.github.io/einlang/book/) is a book about what notation hides, and what happens when you refuse to let it.
- [docs/reference.md](docs/reference.md) is for syntax and semantics once you want the language spelled out precisely.
- [docs/stdlib.md](docs/stdlib.md) is the lookup page for built-ins, modules, and library surface.
- [docs/COORDINATE_FUNCTIONS.md](docs/COORDINATE_FUNCTIONS.md) explains coordinate-aware calls such as `softmax[class]` and `argmax[class]`.
- [docs/AUTODIFF.md](docs/AUTODIFF.md) goes deeper on differential expressions and autodiff-specific behavior.
- [docs/GETTING_STARTED.md](docs/GETTING_STARTED.md) is the short companion page if you want the same onboarding path in a docs section.
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) and [docs/WHY_EINLANG.md](docs/WHY_EINLANG.md) are for readers deciding whether to contribute or understand the implementation and motivation in more depth.

## Status

Einlang currently runs mainly through a NumPy backend. The repository includes the language implementation, standard library, tests, and a large example set.
An IREE execution path is also being built behind the optional `iree` extra; supported functions compile through IREE and unsupported ones fall back to NumPy.
If you want to contribute, start with [CONTRIBUTING](CONTRIBUTING.md).

## License

Apache 2.0. See [LICENSE](LICENSE).
