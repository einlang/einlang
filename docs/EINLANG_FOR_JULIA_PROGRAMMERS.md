# Einlang for Julia programmers

You write **Julia** for numerical simulation (DifferentialEquations.jl, QuantEcon.jl, SciML) or for tensor-heavy ML. This page maps that mindset to Einlang: same “write the math” feel, **compile-time shape checking**, and no stringly-typed einsum. We focus on the **numerical** side — discrete time-stepping, recurrences, stencils — not the symbolic layer (ModelingToolkit, Symbolics.jl).

**Try first:** `python3 -m einlang examples/ode/decay.ein` — same idea as [DiffEqDocs ODE example](https://docs.sciml.ai/DiffEqDocs/stable/getting_started/#ode_example): define the equation, step in time. Then see [Julia demos → Einlang](JULIA_DEMOS.md) for the full mapping.

---

## Mental model in one sentence

**Same class of problems** (ODEs, PDEs, recurrence, Markov, value iteration, tensors); you write the **discrete update** in Einstein notation and recurrences instead of a solver API; the compiler checks shapes and indices.

---

## Julia → Einlang: syntax in detail

Below is a direct mapping of common Julia syntax to Einlang. Einlang uses **0-based indexing** and **end-exclusive** ranges; Julia uses 1-based and inclusive. Everywhere you see `1:N` in Julia, think `0..N` in Einlang (values 0,1,…,N-1). For “1 through N inclusive” in Einlang use `0..=N-1` or `1..=N` (if the type supports it; for indices we use `0..N`).

### Comments

| Julia | Einlang |
|-------|---------|
| `# single line` | `// single line` |
| `#= multi =#` | `//` on each line (no block comment) |

### Variables and constants

| Julia | Einlang |
|-------|---------|
| `x = 42` | `let x = 42;` |
| `const K = 3` | `let K = 3;` (all bindings are immutable by default) |
| `local y = 1` | Inner `let y = 1;` in a block |

Semicolons end statements. The last expression in a block (or file) can be the value; no semicolon after it if it’s the result.

### Types and literals

| Julia | Einlang |
|-------|---------|
| `42`, `3.14`, `true`, `"hi"` | Same: `42`, `3.14`, `true`, `"hi"` |
| `Int32`, `Float64` | `i32`, `f64` (and `i64`, `f32`, etc.) |
| `x::Float64 = 1.0` | `let x: f64 = 1.0;` |
| `[1, 2, 3]` | `[1, 2, 3]` |
| `[1.0 2.0; 3.0 4.0]` (matrix) | `[[1.0, 2.0], [3.0, 4.0]]` (nested rows; row-major) |
| `(1, 2)` tuple | `(1, 2)`; access with `p.0`, `p.1` only |

### Ranges

| Julia | Einlang | Notes |
|-------|---------|--------|
| `1:N` (1 through N inclusive) | `0..N` | Einlang: 0,1,…,N-1 (N values). So “Julia 1:N” → “Einlang 0..N” for same length. |
| `1:10` | `0..10` (exclusive) = 0..9 | Or `1..11` if you want 1..10. |
| `a:b` inclusive | `a..=b` | Inclusive range. |
| `a:b` exclusive | `a..b` | End-exclusive (b not included). |

Used in comprehensions, Einstein index domains, and recurrence bounds: `i in 0..n`, `t in 1..T`.

### Arrays: creation and indexing

| Julia | Einlang |
|-------|---------|
| `A[i]` (1-based) | `A[i]` (0-based): first element is `A[0]` |
| `A[i, j]` | `A[i, j]` |
| `A[:, j]` (column) | No slicing. Use Einstein: `let col[i] = A[i, j];` for a fixed `j`. |
| `A[i, :]` (row) | `let row[j] = A[i, j];` for a fixed `i`. |
| `zeros(5)`, `ones(2,3)` | No built-in; use literals or Einstein, e.g. `let z[i in 0..5] = 0.0;` |
| `length(A)` | `len(A)` (stdlib) or shape from context |
| `size(A, 1)` | No direct equivalent; shapes come from type or inference |

### Loops → comprehensions, Einstein, or recurrence

Julia uses `for`/`while`; Einlang has **no** `for`/`while`. Use one of:

| Julia | Einlang |
|-------|---------|
| `[x^2 for x in 1:10]` | `[x * x \| x in 0..10]` or `[i * i \| i in 1..11]` for 1..10 |
| `[f(i,j) for i in 1:I, j in 1:J]` | `let out[i, j] = f(i, j);` (with `i in 0..I`, `j in 0..J` as inferred or explicit) |
| `for t in 2:N; u[t] = g(u[t-1]); end` | Recurrence: `let u[0] = u0; let u[t in 1..N] = g(u[t - 1]);` |

Comprehensions support filters: `[i \| i in 0..100, i % 2 == 0]`. Einstein notation infers index ranges from array shapes.

### Matrix and element-wise operations

| Julia | Einlang |
|-------|---------|
| `A * B` (matrix multiply) | `let C[i, j] = sum[k](A[i, k] * B[k, j]);` |
| `A .* B` (element-wise) | `let out[i, j] = A[i, j] * B[i, j];` or `A * B` when shapes match or are compatible |
| `A .+ 1`, `2 .* A` | `A + 1`, `A * 2` (scalar broadcasts) |
| `A + B` (same shape) | `A + B` (element-wise; shapes must match or be compatible) |
| `A'` (adjoint) | No postfix `'`; transpose by indexing the other way or a small `let`. |
| `sum(A)` | `sum[i,j](A[i, j])` or `sum[i](sum[j](A[i, j]))` |

### Reductions

| Julia | Einlang |
|-------|---------|
| `sum(A)` | `sum[i](A[i])` (1D) or `sum[i,j](A[i,j])` (2D) |
| `sum(A, dims=1)` | `let row_sum[j] = sum[i](A[i, j]);` |
| `maximum(A)` | `max[i](A[i])` |
| `minimum(A)` | `min[i](A[i])` |
| `prod(A)` | `prod[i](A[i])` |

Reduction index is in brackets: `sum[k](...)`, `max[i](...)`. Ranges are inferred from how the index is used (e.g. `A[i, k]` and `B[k, j]` fix `k`’s range).

### Functions

| Julia | Einlang |
|-------|---------|
| `function f(x) ... end` | `fn f(x) { ... }` |
| `function f(x, y) x + y end` | `fn f(x, y) { x + y }` (last expression is the return value) |
| `f(x) = x^2` (one-liner) | `fn f(x) { x * x }` or `let f = \|x\| x * x;` (lambda) |
| `function f(x::Float64)::Float64` | `fn f(x: f64) -> f64 { ... }` |
| Return value | Last expression in block; no `return` needed (optional for early exit) |
| Anonymous function `x -> x^2` | Lambda: `\|x\| x * x` |

### Conditionals

| Julia | Einlang |
|-------|---------|
| `if x > 0; a else b end` | `if x > 0 { a } else { b }` |
| `if x > 0; a elseif x < 0; b else c end` | `if x > 0 { a } else if x < 0 { b } else { c }` |
| Ternary-style | Same: `if cond { then_expr } else { else_expr }` is an expression. |

Use braces `{ }` and no `end`. The whole `if`-expression has a value.

### Pattern matching

| Julia | Einlang |
|-------|---------|
| `if x == 0 ... elseif x == 1 ...` | `match x { 0 => ..., 1 => ..., _ => ... }` |
| No built-in `match` in base Julia | `match expr { literal => value, _ => default }` |

### Blocks and local scope

| Julia | Einlang |
|-------|---------|
| `let x = 1; y = 2; x + y end` | `{ let x = 1; let y = 2; x + y }` (block: last expression is value) |
| `begin ... end` | `{ stmt; stmt; expr }` |

### Index algebra (e.g. convolution)

| Julia | Einlang |
|-------|---------|
| Manual index math in loops: `ih = oh + kh` | **Where clause:** `where ih = oh + kh, iw = ow + kw` |
| Example: 2D conv | `let out[oh, ow] = sum[kh, kw](in[ih, iw] * w[kh, kw]) where ih = oh + kh, iw = ow + kw;` |

Where clauses also support **guards**: `where data[i] > 0` to filter.

### Imports / modules

| Julia | Einlang |
|-------|---------|
| `using LinearAlgebra` | `use std::math::*;` or `use std::math::{sin, cos};` |
| `import LinearAlgebra: norm` | `use std::math::{norm};` |
| `using A: f, g` | `use my_module::{f, g};` |
| Alias | `use std::math as m;` then `m::sqrt(4.0)` |

File = module. Paths like `std::math::basic` map to stdlib files.

### Recurrence (time-stepping, DP, sequences)

| Julia | Einlang |
|-------|---------|
| `u[1] = u0; for t in 2:N u[t] = F(u[t-1]); end` | `let u[0] = u0; let u[t in 1..N] = F(u[t - 1]);` |
| Base case + loop | **Multiple consecutive `let` clauses** for the same name (base then step). No other `let` in between. |
| Reading “previous” step | Use `u[t - 1]`, `h[t-1, i, j]` etc. **Backward references only** (no `t+1` in the body). |
| Recurrence index | Must be in the bracket: `let u[t in 1..N] = ...`, not in a `where` clause. |

See [Recurrence relations](reference.md#recurrence-relations) for rules (no future indices; index only identifier or literal in the bracket).

### I/O and files

| Julia | Einlang |
|-------|---------|
| `read("file.txt", String)` | `use std::io;` then `read_file("file.txt")` (returns string) |
| `write("file.txt", s)` | `write_file("file.txt", s)` (text only) |
| `using FileIO; save("x.npy", A)` | **std::io:** `load_npy(path)` and `save_npy(path, arr)` (matches NumPy load/save). Example: `use std::io::{load_npy, save_npy}; let x = load_npy("x.npy") as [f32; 10, 20]; save_npy("out.npy", x);` See [Stdlib: std::io](stdlib.md#stdio). |
| Directory/path helpers | `std::io`: `list_dir`, `create_dir`, `join_path`, `dirname`, `basename`, `file_exists`, `is_file`, `is_dir`, etc. See [Stdlib: std::io](stdlib.md#stdio). |

So: **text I/O** is in `std::io`; **binary/tensor I/O** (.npy) is read via `python::numpy::load`, and write is typically done from Python after execution or via `python::numpy::save` if available.

### What Einlang does not have

| Julia | Einlang |
|-------|---------|
| `for` / `while` | Use comprehensions, Einstein, or recurrence instead. |
| `1-based` indexing | 0-based only. |
| `end` keyword | Braces `{ }` for blocks and function bodies. |
| Multiple dispatch | Single function per name (overloading by type exists but is not full multiple dispatch). |
| `A * B` for matmul | Write `sum[k](A[i,k]*B[k,j])` in a `let`. |
| Slicing `A[:, j]` | Use Einstein to define a new array from a row/column. |
| Solver APIs (e.g. `solve(prob, Tsit5())`) | You write the discrete update yourself (recurrence). |

---

## Indexing: 0-based

| | Julia | Einlang |
|---|--------|---------|
| **Indexing** | 1-based `A[1]`, `u[1,:]` | 0-based `A[0]`, `u[0, i]` |
| **Ranges** | `1:N` inclusive | `0..N` end-**exclusive** (so `0..N` = 0,1,…,N-1) |

Every simulation `.ein` file has a **Julia equivalent (1-based)** in comments so you can compare line-by-line. Our tests use 0-based; the comment block at the top of each example shows the same logic in Julia.

---

## Arrays and matrices

- **Literals:** `[[1.0, 2.0], [3.0, 4.0]]` — nested arrays (row-major).
- **Matrix multiply:** No `A * B`; you write the contraction:  
  `let C[i, j] = sum[k](A[i, k] * B[k, j]);`  
  The compiler infers ranges from shapes and checks them.
- **No `for`/`while`:** Use Einstein notation, comprehensions `[x | i in 0..n]`, or **recurrence** declarations (see below).

More: [Syntax comparison: Julia](SYNTAX_COMPARISON.md#julia) · [Language reference: Einstein notation](reference.md#einstein-notation).

---

## ODEs and time-stepping

There is **no** DifferentialEquations.jl-style solver API. You write the **discrete step** yourself (e.g. explicit Euler) as a recurrence.

**Julia (conceptually):**
```julia
u[1] = u0
for t in 2:N
  u[t] = u[t-1] + dt * f(u[t-1])
end
```

**Einlang:** Base case + recurrence clause(s); the compiler handles the order.

```rust
let u[0] = u0;
let u[t in 1..N] = u[t - 1] + dt * f(u[t - 1]);
```

Examples: [ode/](../../examples/ode/) (decay, linear, Lorenz, Lotka–Volterra, pendulum, van der Pol, SIR, harmonic, fitzhugh_nagumo, lorenz96). Same equations as in DiffEqDocs; each file has the Julia equivalent in comments and is accuracy-tested against a reference.

---

## PDEs (heat, wave, reaction–diffusion)

Same idea: you write the **discrete update** (stencil) in Einstein notation. No MethodOfLines API — just the spatial operator and the time step.

- **1D heat / advection:** [pde_1d/](../../examples/pde_1d/) (heat_1d.ein, advection_1d.ein).
- **2D wave:** [wave_2d/](../../examples/wave_2d/main.ein) — two-level recurrence (h[t-1], h[t-2]).
- **Brusselator:** [brusselator/](../../examples/brusselator/) — reaction–diffusion, 4D state.

---

## Recurrence and dynamic programming

**Base case(s) + inductive step.** Like a `for` loop over time or state, but declared as equations.

- **Fibonacci, random walk, logistic map:** [recurrence/](../../examples/recurrence/).
- **Markov stationary distribution:** [markov_stationary.ein](../../examples/recurrence/markov_stationary.ein) — ψ = ψ P (QuantEcon-style).
- **Optimization (gradient descent, power iteration, projected gradient, Rosenbrock):** [optimization/](../../examples/optimization/) — [gradient_descent.ein](../../examples/optimization/gradient_descent.ein), [power_iteration.ein](../../examples/optimization/power_iteration.ein), [projected_gradient.ein](../../examples/optimization/projected_gradient.ein), [rosenbrock.ein](../../examples/optimization/rosenbrock.ein) (Optim.jl/SciML-style).
- **Time series (exponential smoothing):** [time_series/exponential_smoothing.ein](../../examples/time_series/exponential_smoothing.ein) — StateSpaceModels.jl/TimeSeries.jl-style forecasting.
- **Finance (savings / compound interest):** [finance/savings.ein](../../examples/finance/savings.ein).
- **Value iteration (Bellman):** [value_iteration/](../../examples/value_iteration/) — same idea as QuantEcon.jl.
- **McCall job search (reservation wage):** [job_search/mccall.ein](../../examples/job_search/mccall.ein) — value function iteration, QuantEcon-style.

Syntax: `let x[0] = ...; let x[k in 1..K] = ...;` Multiple clauses for the same array must be **consecutive** (no other `let` in between). [Reference: Recurrence relations](reference.md#recurrence-relations).

---

## What we don’t do

- **Symbolic / solver layer** — No ModelingToolkit, Symbolics.jl, or “define ODE symbolically and solve”. We focus on **numerical**: you write the discrete update, we check shapes and run it.
- **Plotting / IDE** — We run from CLI or Python; you can pipe outputs to your own plotting (e.g. run scripts that write HTML or call matplotlib).

---

## Where to go next

| You want… | Go to |
|-----------|--------|
| **Full Julia ↔ Einlang example mapping** | [Julia demos → Einlang](JULIA_DEMOS.md) |
| **Syntax side-by-side (Julia, Python, Rust)** | [Syntax comparison](SYNTAX_COMPARISON.md) |
| **All examples by domain** | [Examples README](../../examples/README.md) |
| **Language and semantics** | [Language reference](reference.md) |
| **Stdlib (math, ML, etc.)** | [Standard library](stdlib.md) |
| **Install and run** | [README: Install & run](../../README.md#install--run) |
