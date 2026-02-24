# Einlang

A compiled language for tensor computation. Write math, run math.

```einlang
let A = [[1, 2], [3, 4]];
let B = [[5, 6], [7, 8]];

let C[i, j] = sum[k](A[i, k] * B[k, j]);   // matrix multiply
```

Einlang is a statically-checked language where tensor operations are first-class syntax. The compiler infers index ranges from array shapes, verifies dimensions at compile time, and generates code. The current backend interprets via NumPy; the target is MLIR for native compiled execution.

## What makes Einlang different

**Einstein notation as language primitive.** Index variables are part of the syntax, not string arguments to a library function. The compiler sees `A[i, k] * B[k, j]`, knows `k` is a contraction index, and infers its range from both arrays — if `A` is 3x4 and `B` is 5x3, you get a compile-time error, not a runtime crash.

**Where clauses for index algebra.** Derived indices, guards, and intermediate bindings attach directly to the computation. A 2D convolution that would be dozens of lines of index bookkeeping in Python:

```einlang
let out[b, oc, oh, ow] = sum[ic, kh, kw](
    input[b, ic, ih, iw] * kernel[oc, ic, kh, kw]
) where ih = oh + kh, iw = ow + kw;
```

The compiler resolves `ih` and `iw` as functions of the output and kernel indices and ensures all accesses stay in bounds.

**Recurrence relations as declarations.** Define sequences the way you'd write them on paper — base cases, then a recursive rule. The compiler determines evaluation order automatically:

```einlang
let fib[0] = 0;
let fib[1] = 1;
let fib[n in 2..20] = fib[n-1] + fib[n-2];
```

This extends to multi-dimensional recurrences (RNN hidden states, dynamic programming tables).

**Compile-time shape analysis.** Dimension mismatches, rank errors, and index range conflicts are caught before any code runs. No "shapes don't align" at runtime halfway through a training loop.

## Why not NumPy / einsum?

```python
# NumPy: shape logic is manual and implicit
C = np.zeros((A.shape[0], B.shape[1]))
for i in range(A.shape[0]):
    for j in range(B.shape[1]):
        for k in range(A.shape[1]):
            C[i, j] += A[i, k] * B[k, j]

# np.einsum('ik,kj->ij', A, B) works for simple cases,
# but try expressing a conv2d with index remapping, or a
# recurrence with base cases, or a filtered reduction.
```

```einlang
let C[i, j] = sum[k](A[i, k] * B[k, j]);
```

`einsum` is a string-based mini-language embedded in Python — no type checking, no shape errors at compile time, no support for conditionals, recurrences, or index algebra. Einlang makes all of that part of the language itself.

## Examples

A neural network forward pass — linear layer, ReLU, softmax:

```einlang
use std::math::exp::exp;

fn softmax(logits) {
    let m = max[i](logits[i]);
    let e[i] = exp(logits[i] - m);
    let s = sum[i](e[i]);
    let out[i] = e[i] / s;
    out
}

let input = [[1.0, 0.5], [0.3, 0.8], [0.7, 0.2]];
let weight = [[0.4, 0.6, 0.1], [0.2, 0.3, 0.9]];
let bias = [0.1, -0.1, 0.0];

let z[i, j] = sum[k](input[i, k] * weight[k, j]) + bias[j];
let activated[i, j] = if z[i, j] > 0.0 { z[i, j] } else { 0.0 };
let probs = softmax(activated[0]);
print(probs);
```

Comprehensions and pattern matching:

```einlang
let even_squares = [x * x | x in 1..50, x % 2 == 0];

let label = match category {
    0 => "cat",
    1 => "dog",
    _ => "unknown",
};
```

## Install

```bash
git clone https://github.com/einlang/einlang.git
cd einlang
pip install -e .
```

Python 3.7+. Dependencies: `numpy`, `lark`, `sexpdata`.

## Run

```bash
python3 -m einlang program.ein
```

Or from Python:

```python
from einlang.compiler.driver import CompilerDriver
from einlang.runtime.runtime import EinlangRuntime

compiler = CompilerDriver()
runtime = EinlangRuntime()

result = compiler.compile(source, "<input>")
output = runtime.execute(result)
# output.outputs["C"] → numpy array
```

## Docs

- [Language Reference](docs/reference.md) — full syntax and semantics
- [Standard Library](docs/stdlib.md) — `std::math`, `std::array`, `std::ml`, `std::io` (300+ functions)
- [Development](docs/DEVELOPMENT.md) — project structure, how to contribute

## Roadmap

**Working now**
- Einstein notation with automatic range and shape inference
- Where clauses: guards, variable bindings, index remapping
- Recurrence relations, array comprehensions, pattern matching
- Functions with monomorphization, module system, 300+ stdlib functions
- Type inference, compile-time shape checking
- NumPy interpreter backend (prototype)

**Next**
- Lambda execution as first-class values
- Pipeline operators: `data |> normalize |> transform`

**Target**
- Arrow combinators for ML graphs: `>>>` (sequential), `***` (parallel), `&&&` (fanout), `|||` (choice)
- MLIR backend for compiled native execution
- GPU acceleration

## License

Apache 2.0
