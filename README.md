# Einlang

A programming language for tensor computations with Einstein notation.

Write tensor operations as mathematical expressions — the compiler handles shape inference, index range analysis, and code generation. Currently backed by a NumPy interpreter.

```einlang
let a = [[1, 2], [3, 4]];
let b = [[5, 6], [7, 8]];

let c[i, j] = sum[k](a[i, k] * b[k, j]);
```

## Install

```bash
git clone https://github.com/einlang/einlang.git
cd einlang
pip install -e .
```

Requires Python 3.7+. Dependencies: `numpy`, `lark`.

## Usage

Write a `.ein` file:

```einlang
let data = [1, -2, 3, -4, 5];
let pos_sum = sum[i in 0..5](data[i] where data[i] > 0);
assert(pos_sum == 9);
print("positive sum:", pos_sum);
```

Run it:

```bash
python3 -m einlang program.ein
```

### Python API

```python
from einlang.compiler.driver import CompilerDriver
from einlang.runtime.runtime import EinlangRuntime

compiler = CompilerDriver()
runtime = EinlangRuntime()

result = compiler.compile(source_code, "<input>")
output = runtime.execute(result)
```

`CompilerDriver.compile(source, source_file, root_path=None)` returns a `CompilationResult`:
- `.success` — whether compilation succeeded
- `.ir` — the compiled IR program
- `.get_errors()` — list of formatted error strings

`EinlangRuntime.execute(compilation_result, inputs=None)` returns an `ExecutionResult`:
- `.success` — whether execution succeeded
- `.value` — result of the last expression
- `.outputs` — dict of all named outputs

Inputs and outputs are NumPy arrays:

```python
import numpy as np

code = "let result[i, j] = matrix[i, j] * 2;"
result = compiler.compile(code, "scale.ein")
output = runtime.execute(result, inputs={"matrix": np.array([[1, 2], [3, 4]])})
print(output.outputs["result"])  # [[2, 4], [6, 8]]
```

## Architecture

```
Source (.ein) → Frontend (Lark → AST) → Passes → IR → Backend (NumPy)
```

- **`frontend/`** — Lark grammar, parser, AST transformers
- **`passes/`** — Name resolution, type inference, Einstein lowering, range/shape analysis, AST-to-IR, IR validation
- **`ir/`** — IR node definitions, S-expression serialization
- **`backends/`** — NumPy evaluator, Einstein executor
- **`runtime/`** — Scope stack, reduction engine
- **`analysis/`** — Module system, monomorphization
- **`compiler/`** — Driver orchestrating the pipeline
- **`shared/`** — DefId system, types, AST/IR nodes, error codes

## Tests

```bash
python3 -m pytest tests/ --tb=short -q
```

80 test files across `tests/unit/`, `tests/integration/`, `tests/examples/`, and `tests/stdlib/`. 62 `.ein` programs under `examples/units/` are run end-to-end through `tests/examples/test_units.py`.

## Documentation

- [Language Reference](docs/reference.md) — Syntax, types, Einstein notation, error codes
- [Standard Library](docs/stdlib.md) — `std::math`, `std::array`, `std::ml`, `std::io`
- [Development](docs/DEVELOPMENT.md) — Project structure, contributing

## Roadmap

**Near-term**
- Lambda execution
- Pipeline operators: `data |> normalize |> transform`
- Try expressions

**Medium-term**
- Arrow combinators: `linear(784, 256) >>> relu >>> linear(256, 10)`
- Generic types
- Automatic differentiation
- GPU dispatch via CuPy

**Long-term**
- MLIR backend
- Native compilation
- Distributed tensor operations

## License

Apache 2.0
