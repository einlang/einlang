# Julia Syntax and Design — Examples in Einlang

Key Julia concepts and how they map to Einlang, with side-by-side or equivalent examples. This document aims to cover Julia syntax so you can translate code and know what exists in both languages.

---

## 1. Variables and assignment

**Julia:** Assignment with `=`. No keyword; variables are created on first assignment. Type is inferred.

```julia
x = 5
name = "Julia"
```

**Einlang:** Explicit **`let`** binding; semicolon terminates the statement. Type inferred or annotated.

```rust
let x = 5;
let name: str = "Einlang";
```

---

## 2. Primitive types and literals

**Julia:** Common types: `Int64`, `Float64`, `Bool`, `String`. Literals: `42`, `3.14`, `true`, `"hi"`. Rationals: `2//3`. Complex: `1 + 2im`. Type annotation: `x::Int`. Julia does **not** support literal suffixes (e.g. no `1i32` or `1.0f32`); integer literals default to `Int` (platform Int32/Int64), floats to `Float64`. Use `convert(Int32, 1)` or a typed variable: `x::Float32 = 1.0`.

```julia
n = 42
x = 3.14
y::Float32 = 1.0
z = convert(Int32, 42)
```

**Einlang:** Primitives: `i32`, `i64`, `f32`, `f64`, `bool`, `str`. Literals: `42`, `3.14`, `true`, `"hi"`. Annotation: `x: i32`; literals can be coerced at the binding (e.g. `let x: i64 = 42`). No rational or complex in the current reference.

```rust
let n = 42;
let x: f64 = 3.14;
let b: bool = true;
```

---

## 3. Arrays and literals

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

**Julia rectangular access** (matrix `Matrix{T}` — comma-separated indices, 1-based). The colon `:` means “all indices along this dimension”:

```julia
M = [1 2 3; 4 5 6]   # 2×3 matrix
M[1, 2]              # 2  (row 1, col 2) — scalar
M[2, :]              # [4, 5, 6]  (row 2: fix row, all cols)
M[:, 3]              # [3, 6]     (col 3: all rows, fix col)
M[1:2, 2:3]          # submatrix (range slice)
```

So `M[i,:]` is used to get **row i** (a vector); `M[:,j]` to get **column j** (a vector). Without the colon, `M[i,j]` is a single element. These slices are normal expressions and can be used inside other expressions (e.g. `M[i,:] .+ 1`, `sum(M[:,j])`, or as function arguments). You can chain indexing: `M[1,:][2]` is the second element of row 1 (same as `M[1,2]`).

**Einlang rectangular access** (0-based, comma in one bracket; no colon-slices):

```rust
let M = [[1, 2, 3], [4, 5, 6]];
let elem = M[0, 1];   // 2
let row = M[1];       // [4, 5, 6]
// Col or slice: use comprehensions or explicit loops
```

In Einlang, `M[i]` (row `i`) is also a normal expression and can be used in other expressions (e.g. `M[0] * 2.0`, or as an argument to a function). Chained indexing is supported: `M[i][j]` is the same as `M[i,j]` (row `i`, then element `j`).

**Single index on a matrix:** Julia uses **linear indexing** (column-major): `M[1]` on a 2×3 matrix is the first element (scalar), `M[2]` the second element down the first column, etc. To get a row or column in Julia you use slices: `M[i,:]`, `M[:,j]`. Einlang uses **one index = one dimension**: `M[i]` is row `i` (a vector), not a linear index; there is no linear index for rectangular arrays.

---

## 4. Ranges

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

## 5. Array comprehensions

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

## 6. Element-wise operations (Julia’s “broadcasting”)

**Julia:** Dot syntax applies a function or operator element-wise; chained dots fuse into one loop.

```julia
sin.(x)
a .+ b
2 .* x .^ 2 .+ 1
```

**Einlang:** Operators support **broadcasting by default** (no dot) only for **same rank** (same-shape tensor with tensor) or **tensor vs scalar**. In those cases explicit indexing is not required (e.g. `A + 5`, `A + B`). For different-rank combinations, use rectangular `let` with indices.

```rust
let A = [[1.0, 2.0], [3.0, 4.0]];
let scaled = A * 2.0;           // scalar * tensor (no indexing)
let B = [[1.0, 1.0], [1.0, 1.0]];
let summed = A + B;             // same rank, same shape (no indexing)
let with_bias[i, j] = A[i, j] + bias[j];  // different rank: explicit indices
```


## 7. Matrix multiplication vs element-wise

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

## 8. Reductions (sum, max, min)

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

## 8b. Jagged (ragged) arrays

**Julia:** Rectangular vs jagged are **different types**. Rectangular: `Matrix{T}` (or `Array{T,2}`) — fixed shape, indexing `A[i, j]`. Jagged: `Vector{Vector{T}}` — each row can have a different length, indexing `A[i][j]`. You choose the type; the compiler does not infer “this literal is jagged” from row lengths.

```julia
# Vector of vectors — rows have different lengths
jagged = [[1, 2], [3, 4, 5], [6]]
jagged[1]          # [1, 2]
jagged[2][3]       # 5
length(jagged)     # 3
length(jagged[i])  # length of row i
```

**Einlang:** `jagged[T]` type; literal with rows of different lengths. Access with chained brackets `A[i][j]`. Cannot use Einstein notation on jagged arrays.

```rust
let ragged: jagged[i32] = [[1, 2], [3, 4, 5], [6]];
let row0 = ragged[0];       // [1, 2]
let elem = ragged[1][2];    // 5
let n = len(ragged);        // 3
```

Both languages: jagged data is “array of arrays”; use loops or comprehensions to iterate. Einstein notation and shape inference apply only to rectangular arrays.

---

## 9. Where clauses / index algebra (Einlang strength)

**Julia:** Index arithmetic is usually manual (loop bounds, `CartesianIndex`, or comprehensions with computed indices).

**Einlang:** **Where clauses** bind derived indices and guards; the compiler checks bounds and infers iteration.

**1D convolution (Julia-style loop):**

```julia
out = [sum(signal[i+k] * kernel[k+1] for k in 0:length(kernel)-1)
       for i in 1:length(signal)-length(kernel)+1]
```

**Einlang:** The compiler can infer ranges from array shapes. You can use implicit ranges when inference succeeds:

```rust
let signal = [1, 2, 3, 4, 5, 6] as [f32];
let kernel = [0.5, 0.5];
let convolved[i] = sum[k](signal[i + k] * kernel[k]);
```

If needed (e.g. when inference is not possible), use explicit ranges: `let convolved[i in 0..5] = sum[k in 0..2](signal[i + k] * kernel[k]);`

**2D convolution with index remapping (Einlang where-clause):**

```rust
let out[b, oc, oh, ow] = sum[ic, kh, kw](
    input[b, ic, ih, iw] * kernel[oc, ic, kh, kw]
) where ih = oh + kh, iw = ow + kw;
```

The compiler resolves `ih`, `iw` from `oh`, `ow`, `kh`, `kw` and enforces in-bounds access. In Julia you’d typically write the index arithmetic and bounds yourself.

---

## 10. Recurrence relations (Einlang strength)

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

**Einlang:** **Recurrence declarations**: base cases and recursive case in one place; the compiler determines evaluation order. Put the recurrence index range in the bracket, not in a where clause: `[n in 2..20]`. Using `where n in 2..20` for the recurrence range is invalid and issues an error (E0303).

```rust
let fib[0] = 0;
let fib[1] = 1;
let fib[n in 2..20] = fib[n - 1] + fib[n - 2];
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

## 11. Functions and “dispatch” (Julia vs Einlang)

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

## 12. Structs / composite types

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

## 13. Mathematical notation (dense formulas)

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

## 14. Control flow: if, for, while

**Julia:** `if` / `elseif` / `else` / `end`; `for x in iter ... end`; `while cond ... end`. No expression form for if (use ternary for expressions).

```julia
if x > 0
    y = sqrt(x)
elseif x == 0
    y = 0.0
else
    y = -1.0
end
for i in 1:10
    println(i)
end
while n > 0
    n -= 1
end
```

**Einlang:** `if` / `else if` / `else` with braces `{}`; **if is an expression** (returns a value). No `for` or `while`; use comprehensions, recurrence, or Einstein notation for iteration.

```rust
let y = if x > 0.0 { sqrt(x) } else if x == 0.0 { 0.0 } else { -1.0 };
let squared = [i * i | i in 1..11];
```

---

## 15. Blocks and statements

**Julia:** Blocks use `end`; statements separated by newlines or `;`. No block expression returning a value (use `begin ... end` for a sequence that returns the last expression).

```julia
begin
    a = 1
    b = 2
    a + b
end
```

**Einlang:** Block with `{ }`; semicolons terminate statements. Block **returns** its final expression.

```rust
let result = {
    let a = 1;
    let b = 2;
    a + b
};
```

---

## 16. Function definition forms

**Julia:** Long form `function f(x) ... end`; short form `f(x) = x + 1`. Optional `return`; without it, the last expression is returned. Multiple methods (same name, different signatures).

```julia
function add(x, y)
    x + y
end
double(x) = 2 * x
```

**Einlang:** Single form `fn name(params) { body }`. Body is an expression or block; no `return` keyword for final expression. One function name per scope (no overloading).

```rust
fn add(x, y) { x + y }
fn double(x) { x * 2 }
```

---

## 17. Anonymous functions (lambdas)

**Julia:** `x -> x^2` or multi-line `function (x) x^2 end`. Used for callbacks and higher-order functions.

```julia
map(x -> x^2, [1, 2, 3])
```

**Einlang:** Lambda `|params| body` or `|params| { statements; expr }`. Immediately invoked: `(|x| x + 1)(5)`.

```rust
let sq = |x| x * x;
let y = (|x| x + 1)(5);
```

---

## 18. Modules and imports

**Julia:** `module M ... end`; `using M` (bring names into scope) or `import M: f, g`. Namespaced: `M.f`.

```julia
module MyMod
export foo
foo(x) = x + 1
end
using .MyMod
foo(2)
```

**Einlang:** File = module. `use path::name` or `use path::*`; `pub` for visibility. Namespaced: `std::math::sin`.

```rust
use std::math::{ sin, cos };
use std::array::*;
let x = sin(0.0);
```

---

## 19. Operators (summary)

**Julia:** Arithmetic `+ - * / % ^`; comparison `== != < > <= >=`; logical `&& || !`; string `*` (concat). Dot for broadcasting: `.+`, `.*`, etc.

**Einlang:** Same arithmetic and comparison; `**` for power (not `^`). Logical `&&`, `||`, `!`. No string concat operator in reference (use interpolation or library). Broadcasting is default for same-rank/scalar; no dot.

| Operator | Julia | Einlang |
|----------|--------|---------|
| Power | `^` | `**` |
| Integer div | `÷` or `div` | `/` (truncates) |
| And/Or/Not | `&&` `\|\|` `!` | same |
| Equality | `==`, `!=` | same |

---

## 20. Comments

**Julia:** `#` to end of line. No block comments in base syntax.

**Einlang:** `//` to end of line only (Rust-aligned).

---

## 21. Pattern matching

**Julia:** No built-in pattern matching on values. Use `if`/`elseif`, or multiple dispatch on types.

**Einlang:** **`match`** expression with arms: literals, wildcard `_`, identifiers. Exhaustiveness checked.

```rust
let label = match n { 0 => "zero", 1 => "one", _ => "other" };
```

---

## 22. Exceptions and try

**Julia:** `try` / `catch` / `finally`; `throw(ErrorException("msg"))`.

**Einlang:** **Try expressions** are planned (`try expr` wrapping in Result). Not yet in the core reference.

---

## 23. Tuples

**Julia:** `(a, b)` or `(1, "two", 3.0)`; immutable; index with `t[1]` (1-based). Named tuples: `(x=1, y=2)`.

**Einlang:** `(a, b)`; immutable; access **only** with **dot**: `t.0`, `t.1` (0-based). No `t[0]`. No named tuples in reference.

```rust
let p = (1.0, 2.0);
let x = p.0;
let y = p.1;
```

---

## 24. Other Julia syntax

**Ternary**

Julia: `cond ? a : b`. Einlang: use `if` as an expression: `if cond { a } else { b }`.

```julia
# Julia
y = x > 0 ? sqrt(x) : 0.0
```

```rust
// Einlang
let y = if x > 0.0 { sqrt(x) } else { 0.0 };
```

**String interpolation**

Julia: `"x = $x"` or `"sum = $(a + b)"`. Einlang: `"x = {x}"` or `"sum = {a + b}"`; format with `{expr:.4f}`.

```rust
let name = "world";
let msg = "hello {name}";
let fmt = "pi = {pi:.4f}";
```

**Do block (passing a block as first argument)**

Julia: `open("file") do io ... end` passes a lambda as the first argument. Einlang: pass a lambda explicitly: `open("file", |io| { ... })`. No dedicated do-block syntax.

**Pipe**

Julia: `x |> f` or `x |> f |> g`. Einlang: pipeline operator `|>` is planned (e.g. `data |> normalize |> transform`). Not yet implemented.

**Short-circuit**

Julia: `a && b` (b evaluated only if a is true), `a || b`. Einlang: same; `&&` and `||` short-circuit.

**Splat / varargs**

Julia: `f(a, b...)` to accept or pass variable arguments. Rust: fixed arity only; no variadic/splat; use tuples, slices, or macros. Einlang: same as Rust; no splat in the current reference; use fixed-arity or arrays.

---

## 25. What Julia has that Einlang doesn’t

- **Multiple dispatch**: many methods per function name; Einlang has one name per scope (monomorphization for untyped params).
- **for / while loops**: Julia has both; Einlang uses comprehensions, recurrence, and Einstein notation instead.
- **Linear indexing**: `M[1]` on a matrix is scalar in Julia; Einlang uses `M[i]` for row.
- **Colon slices**: `M[:, j]`, `M[i, :]`; Einlang has `M[i]` for row, column via comprehension.
- **Rational/complex literals**: `2//3`, `1+2im`; not in Einlang reference.
- **Macros**: `@macro`; Einlang has no macro system.
- **do-block**: `open(f) do io ... end`; Einlang uses explicit lambda.
- **Named tuples**: `(x=1, y=2)`; Einlang has positional tuples only.
- **1-based indexing**: Julia default; Einlang is 0-based.

---

## 26. What Einlang has that Julia doesn’t

- **Where clauses**: index binding and guards in tensor expressions; Julia uses manual index math.
- **Recurrence declarations**: `let fib[n in 2..N] = ...` (range in bracket; `where n in ...` is invalid); Julia uses loops or recursion.
- **Einstein notation**: first-class `let C[i,j] = sum[k](A[i,k]*B[k,j])` with shape inference; Julia uses `*` and manual dims.
- **Pattern matching**: `match x { ... }`; Julia uses dispatch or if.
- **Tuple access**: only `.0`, `.1` (no `t[0]`); enforces distinction from arrays.
- **Explicit let**: every binding is `let`; no implicit global assignment.

---

## 27. Quick reference table

| Concept            | Julia                    | Einlang                                      |
|--------------------|--------------------------|----------------------------------------------|
| Variable           | `x = 5`                  | `let x = 5;`                                 |
| Type annotation    | `x::Int`                 | `x: i32`                                     |
| Vector/matrix      | `[1,2,3]`, `[1 2; 3 4]`  | `[1,2,3]`, `[[1,2],[3,4]]`                    |
| Range              | `1:10` (inclusive)       | `1..11` (end exclusive)                      |
| Comprehension      | `[x^2 for x in 1:10]`   | `[x*x \| x in 1..11]`                        |
| Element-wise       | `f.(x)`, `a .+ b`       | Same rank or scalar: `A + B`, `A * 2.0`; different rank: `let out[i,j] = A[i,j] + bias[j];` |
| Matrix multiply    | `A * B`                 | `sum[k](A[i,k]*B[k,j])`                      |
| Reduction          | `sum(A)`, `sum(A,dims=1)` | `sum[i,j](A[i,j])`, `sum[j](A[i,j])`       |
| Index algebra      | Manual loops/indices     | `where ih = oh+kh, iw = ow+kw`               |
| Recurrence         | Loops / recursion       | `let fib[0]=0; let fib[1]=1; let fib[n]=...`  |
| If                 | `if ... elseif ... else ... end` | `if ... { } else if ... { } else { }` (expression) |
| For/while          | `for ... end`, `while ... end` | Comprehensions, recurrence; no bare for/while |
| Block              | `begin ... end`         | `{ stmt; stmt; expr }` (returns last expr)   |
| Function           | `function f() end`, `f() =` | `fn f() { }`                               |
| Lambda             | `x -> x^2`              | pipe form: `\|x\| x * x`                     |
| Module/import      | `module`, `using`, `import` | `use path::name`, `use path::*`            |
| Tuple access       | `t[1]` (1-based)        | `t.0`, `t.1` only (0-based)                 |
| Pattern match      | No built-in             | `match x { pat => expr, _ => default }`      |
| Many implementations | Multiple dispatch     | Monomorphization; match/typeof for type-specific |
| Struct             | `struct T ... end`      | Tuples/arrays today; struct in proposal      |

---

## 28. Summary

- **Arrays, ranges, comprehensions:** Einlang uses 0-based exclusive ranges and `[expr \| gen, filter]` comprehensions; the ideas align with Julia.
- **“Broadcasting”:** In Einlang, element-wise and scalar-tensor behavior is expressed with rectangular `let` and index variables; no separate dot operator.
- **Linear algebra:** Matrix multiply and reductions are first-class via Einstein notation; shapes and contraction indices are checked at compile time.
- **Where clauses and recurrences:** Einlang’s index algebra and recurrence declarations give you Julia-like expressiveness with compile-time shape and bounds checking.
- **Dispatch and types:** Einlang uses monomorphization for untyped params; no overloading by type (same name = redefinition). Use match/typeof or one generic fn for type-specific behavior.
- **Structs:** Use tuples/arrays until Einlang’s struct/record types are final; then the mapping is straightforward.

Using this mapping, you can translate Julia-style scientific and tensor code into Einlang and leverage its guarantees (shape and index checking) while keeping the code close to mathematical notation.
