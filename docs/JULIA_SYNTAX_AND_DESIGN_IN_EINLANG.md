# Julia Syntax and Design — Examples in Einlang

Key Julia concepts and how they map to Einlang, with side-by-side or equivalent examples.

---

## 1. Arrays and literals

**Julia:** 1-based indexing by default; `[1, 2, 3]` for vectors, `[1 2; 3 4]` or `[1 2 3; 4 5 6]` for matrices.

```julia
v = [1, 2, 3]
M = [1 2 3; 4 5 6]
```

**Einlang:** 0-based indexing; nested brackets for 2D; same logical idea.

```rust
let v = [1, 2, 3];
let M = [[1, 2, 3], [4, 5, 6]];
```

---

## 2. Ranges

**Julia:** `a:b` is inclusive on both ends; `a:s:b` is step. Often used for loops and comprehensions.

```julia
1:10        # 1, 2, ..., 10
1:2:10      # 1, 3, 5, 7, 9
```

**Einlang:** `a..b` is start inclusive, end exclusive (like Rust). Used in comprehensions and recurrence bounds.

```rust
1..11       // 1, 2, ..., 10 (end exclusive)
// Step: use comprehension, e.g. [i | i in 1..11, i % 2 == 1] for odds
let odds = [i | i in 1..11, i % 2 == 1];
```

---

## 3. Array comprehensions

**Julia:** `[expr for x in range]` or `[expr for x in r1, y in r2]` for multidimensional; optional filter with `if`.

```julia
[x^2 for x in 1:10]
[x for x in 1:20 if x % 2 == 0]
[i + j for i in 1:3, j in 1:3]
```

**Einlang:** `[expr | x in range]` with optional filters (comma-separated). Multiple generators nest.

```rust
let squares = [x * x | x in 1..11];
let evens = [x | x in 1..21, x % 2 == 0];
let grid = [(i + j) | i in 1..4, j in 1..4];
```

---

## 4. Element-wise operations (Julia’s “broadcasting”)

**Julia:** Dot syntax applies a function or operator element-wise; chained dots fuse into one loop.

```julia
sin.(x)
a .+ b
2 .* x .^ 2 .+ 1
```

**Einlang:** Operators support **broadcasting by default** (no dot) only for **same rank** (same-shape tensor with tensor) or **tensor vs scalar**. In those cases explicit indexing is not required (e.g. `A + 5`, `A + B`). For different-rank combinations, use rectangular `let` with indices.

```rust
use std::math::sin;
let doubled[i, j] = matrix[i, j] * 2.0;
let summed[i, j] = A[i, j] + B[i, j];
let applied[i] = sin(x[i]);
let formula[i, j] = 2.0 * matrix[i, j] * matrix[i, j] + 1.0;
```

Scalar + tensor: index the tensor and add the scalar in the body (scalar is “broadcast” by repetition over indices):

```rust
let result[i, j] = tensor[i, j] + 5;
```

---

## 5. Matrix multiplication vs element-wise

**Julia:** `*` is matrix multiply; `.*` is element-wise.

```julia
C = A * B           # matrix multiply
C = A .* B          # element-wise
```

**Einlang:** `*` on two tensors is element-wise. Matrix multiply is **Einstein sum over a contraction index**.

```rust
let C[i, j] = A[i, j] * B[i, j];           // element-wise
let C[i, j] = sum[k](A[i, k] * B[k, j]);  // matrix multiply
```

---

## 6. Reductions (sum, max, min)

**Julia:** `sum(A)`, `sum(A, dims=1)`, `maximum(A)`, etc. Dimensions are often specified by keyword.

```julia
sum(A)
sum(A, dims=1)
maximum(A, dims=2)
```

**Einlang:** Reductions are **first-class syntax** with index variables. Which indices appear in the body and on the left define the reduction axes; the compiler infers ranges from shapes.

```rust
let total = sum[i, j](A[i, j]);
let row_sums[i] = sum[j](A[i, j]);
let col_max[j] = max[i](A[i, j]);
```

---

## 7. Where clauses / index algebra (Einlang strength)

**Julia:** Index arithmetic is usually manual (loop bounds, `CartesianIndex`, or comprehensions with computed indices).

**Einlang:** **Where clauses** bind derived indices and guards; the compiler checks bounds and infers iteration.

**1D convolution (Julia-style loop):**

```julia
out = [sum(signal[i+k] * kernel[k+1] for k in 0:length(kernel)-1)
       for i in 1:length(signal)-length(kernel)+1]
```

**Einlang (explicit ranges):**

```rust
let signal = [1, 2, 3, 4, 5, 6] as [f32];
let kernel = [0.5, 0.5];
let convolved[i in 0..5] = sum[k in 0..2](signal[i + k] * kernel[k]);
```

**2D convolution with index remapping (Einlang where-clause):**

```rust
let out[b, oc, oh, ow] = sum[ic, kh, kw](
    input[b, ic, ih, iw] * kernel[oc, ic, kh, kw]
) where ih = oh + kh, iw = ow + kw;
```

The compiler resolves `ih`, `iw` from `oh`, `ow`, `kh`, `kw` and enforces in-bounds access. In Julia you’d typically write the index arithmetic and bounds yourself.

---

## 8. Recurrence relations (Einlang strength)

**Julia:** Recurrences are written as loops or recursive functions; you manage base case and order yourself.

```julia
function fib(n)
    n <= 1 && return n
    a, b = 0, 1
    for _ in 2:n
        a, b = b, a + b
    end
    return b
end
```

**Einlang:** **Recurrence declarations**: base cases and recursive case in one place; the compiler determines evaluation order.

```rust
let fib[0] = 0;
let fib[1] = 1;
let fib[n] = fib[n - 1] + fib[n - 2] where n in 2..20;
```

**RNN-style hidden state (Julia loop):**

```julia
h = zeros(batch, hidden)
for t in 1:T
    h = tanh.(W * x[t,:,:] + R * h)
end
```

**Einlang recurrence:**

```rust
let hidden[0, b in 0..batch_size, h in 0..hidden_size] = initial_h[b, h];
let hidden[t in 1..seq_length, b in 0..batch_size, h in 0..hidden_size] =
    tanh(sum[i](W[h, i] * X[t, b, i]) + sum[h_prev](R[h, h_prev] * hidden[t-1, b, h_prev]));
```

---

## 9. Functions and “dispatch” (Julia vs Einlang)

**Julia:** Multiple dispatch — one function name, many methods by argument types; dispatch on all arguments.

```julia
f(x::Int) = x + 1
f(x::Float64) = x + 1.0
f(x::String) = x * "!"
```

**Einlang:** Single function name per scope. **Monomorphization**: untyped parameters get a specialized implementation per call-site type. No runtime dispatch; no separate “methods” syntax.

```rust
fn double(x) { x * 2 }
let a = double(3);    // i32
let b = double(3.14); // f32
```

Einlang does **not** support multiple `fn` with the same name (overloading by type). A second `fn describe(...)` in the same scope is a redefinition error. For type-specific behavior use a single function and **pattern match on the type** (e.g. `match typeof(x) { ... }`) or a single generic function that handles all cases.

---

## 10. Structs / composite types

**Julia:** `struct` with optional field types; dot access; works with multiple dispatch.

```julia
struct Point
    x::Float64
    y::Float64
end
p = Point(1.0, 2.0)
p.x, p.y
```

**Einlang:** As of the reference and examples, custom struct/record syntax is **proposed or demo-only** (e.g. `struct Point { x: f32, y: f32 }` in adt demos). For now, use **tuples** or **arrays** for fixed layouts. Tuple access uses **only** dot and numeric index: `p.0`, `p.1` (not `p[0]`).

```rust
let p = (1.0, 2.0);
let x = p.0;
let y = p.1;
```

Or a small array: `let p = [1.0, 2.0];` and index by position. Once struct/record types are stable, they will map directly to Julia-style composite types.

---

## 11. Mathematical notation (dense formulas)

**Julia:** Looks like math; `*` for matrix multiply, `'` for transpose; good for linear algebra.

```julia
α * A * B + β * C
norm(x)
x' * y   # dot product (vector)
```

**Einlang:** Dense math is expressed with **Einstein notation** and **where**; shapes and indices are part of the type/expression, so the compiler checks consistency.

```rust
let C[i, j] = alpha * sum[k](A[i, k] * B[k, j]) + beta * C_in[i, j];
let n = sqrt(sum[i](x[i] * x[i]));
let dot = sum[i](x[i] * y[i]);
```

---

## 12. Quick reference table

| Concept            | Julia                    | Einlang                                      |
|--------------------|--------------------------|----------------------------------------------|
| Vector/matrix      | `[1,2,3]`, `[1 2; 3 4]`  | `[1,2,3]`, `[[1,2],[3,4]]`                    |
| Range              | `1:10` (inclusive)       | `1..11` (end exclusive)                      |
| Comprehension      | `[x^2 for x in 1:10]`   | `[x*x \| x in 1..11]`                        |
| Element-wise       | `f.(x)`, `a .+ b`       | `let out[i] = f(x[i]);` / `let out[i,j] = A[i,j]+B[i,j];` |
| Matrix multiply    | `A * B`                 | `sum[k](A[i,k]*B[k,j])`                      |
| Reduction          | `sum(A)`, `sum(A,dims=1)` | `sum[i,j](A[i,j])`, `sum[j](A[i,j])`       |
| Index algebra      | Manual loops/indices     | `where ih = oh+kh, iw = ow+kw`               |
| Recurrence         | Loops / recursion       | `let fib[0]=0; let fib[1]=1; let fib[n]=...`  |
| Many implementations | Multiple dispatch     | Monomorphization; use match/typeof for type-specific behavior |
| Struct             | `struct T ... end`      | Tuples/arrays today; struct in proposal      |

---

## 13. Summary

- **Arrays, ranges, comprehensions:** Einlang uses 0-based exclusive ranges and `[expr \| gen, filter]` comprehensions; the ideas align with Julia.
- **“Broadcasting”:** In Einlang, element-wise and scalar-tensor behavior is expressed with rectangular `let` and index variables; no separate dot operator.
- **Linear algebra:** Matrix multiply and reductions are first-class via Einstein notation; shapes and contraction indices are checked at compile time.
- **Where clauses and recurrences:** Einlang’s index algebra and recurrence declarations give you Julia-like expressiveness with compile-time shape and bounds checking.
- **Dispatch and types:** Einlang uses monomorphization for untyped params; no overloading by type (same name = redefinition). Use match/typeof or one generic fn for type-specific behavior.
- **Structs:** Use tuples/arrays until Einlang’s struct/record types are final; then the mapping is straightforward.

Using this mapping, you can translate Julia-style scientific and tensor code into Einlang and leverage its guarantees (shape and index checking) while keeping the code close to mathematical notation.
