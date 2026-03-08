# 2 — Demos

> **Previous**: [`basics/`](../basics/) · **Next**: [`mnist/`](../mnist/)

Intermediate examples that build on the fundamentals. These introduce Einstein notation for tensor operations, the module/import system, and more expressive patterns.

## Files

| File | What it covers |
|------|----------------|
| `matrix_operations.ein` | Matrix multiply via `sum[k](A[i,k] * B[k,j])`, Frobenius norm, trace, determinant, statistics |
| `computer_vision_tensors.ein` | 4D NCHW tensors, per-channel normalization, manual convolution, pooling, batch norm, channel attention |
| `function_overloading_complete.ein` | Functions dispatched by argument count and type |
| `mathematical_overloading_demo.ein` | Overloaded math operators on custom shapes |
| `improved_math_accuracy_demo.ein` | Math function accuracy (sqrt, trig, constants) |
| `array_structure_comparison.ein` | Nested arrays vs flat arrays, structural equality |
| `in_operator_demo.ein` | `x in collection` membership tests |
| `in_operator_simple_comprehensions.ein` | `[f(x) \| x in items]` comprehension patterns |
| `tuple_comprehensions_with_in.ein` | Multi-variable iteration with tuples |
| `cli_demo.ein` | Reading command-line arguments |
| `rust_syntax_demo.ein` | Rust-inspired syntax features |
| `simple_import_test.ein` | `use` and `mod` for importing modules |
| `test_import_system.ein` | Cross-module calls and re-exports |
| `math_utils.ein`, `string_utils.ein`, `tensor_functions.ein` | Utility modules imported by other demos |

## Running

```bash
python3 -m einlang examples/demos/matrix_operations.ein
python3 -m einlang examples/demos/computer_vision_tensors.ein
python3 -m einlang examples/demos/function_overloading_complete.ein
```

## Profiling

To see per-clause timings and whether each Einstein clause took the vectorized, hybrid, or scalar path, set:

```bash
export EINLANG_PROFILE_STATEMENTS=1
export EINLANG_DEBUG_VECTORIZE=1
python3 -m einlang examples/demos/<file>.ein
```

For wave and RD, run the `.ein` file with the same env vars (no runner script needed for profiling):

```bash
EINLANG_PROFILE_STATEMENTS=1 EINLANG_DEBUG_VECTORIZE=1 python3 -m einlang examples/wave_2d/main.ein
EINLANG_PROFILE_STATEMENTS=1 EINLANG_DEBUG_VECTORIZE=1 python3 -m einlang examples/reaction_diffusion/main.ein
```

For heat (PDE is inline in the script), use the runner with `--profile-einlang`:

```bash
python3 examples/heat_animation.py --profile-einlang
```

For whisper or deit_tiny, run from the example directory so paths resolve, then set the env vars:

```bash
cd examples/whisper_tiny && EINLANG_PROFILE_STATEMENTS=1 EINLANG_DEBUG_VECTORIZE=1 python3 -m einlang main.ein
```

## Key concepts introduced

- **Einstein notation** — `let result[i,j] = sum[k](A[i,k] * B[k,j])` defines a matrix multiply by declaring the output shape and a reduction over `k`. This notation is used extensively in the ML examples.
- **4D tensors (NCHW)** — `image_batch[batch, channel, height, width]` is the standard tensor layout for images. The `computer_vision_tensors.ein` file walks through normalization, convolution, and pooling on this format.
- **Module system** — `use std::math::sqrt` imports from stdlib; `mod my_module` imports a local file. Cross-file organization becomes important in larger programs.
- **Function overloading** — same function name with different signatures, dispatched at call time.
- **`in` operator** — membership testing and iteration in comprehensions.

After these demos, you have all the language tools needed for real neural networks. Continue to [mnist/](../mnist/) to put them to work.
