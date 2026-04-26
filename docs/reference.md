
# Einlang Language Reference

Full syntax and semantics. If you are new here, start with [GETTING_STARTED](GETTING_STARTED.md) or the [examples guide](../examples/README.md) first. For modules and built-in library functions, see [stdlib](stdlib.md). For derivatives and tangents, see [AUTODIFF](AUTODIFF.md).

---

## Statements

A program is a sequence of statements. Every statement ends with `;`.

### Comments

Einlang uses `//` for line comments. There is no `--` comment syntax.

```rust
// compute one value
let x = 1 + 1;   // inline comment
```

### `let` declarations

All bindings are immutable. The type annotation is optional; when omitted the type is inferred from the right-hand side.

```rust
let x = 42;                   // inferred i32
let pi: f64 = 3.141592653589793;   // explicit f64
let matrix: [f32; 2, 3] = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
```

If both annotation and value are present, the value must be assignment-compatible with the annotation (see Type Compatibility).

Tuple-valued expressions can be destructured directly in a `let` binding:

```rust
let pair = (10, 20);
let (x, y) = pair;
let (a: f32, b: f32) = (1.0, 2.0);
```

Tuple destructuring supports typed elements. It is intended for fixed-arity
results such as library calls returning `(values, indices)`. Array
destructuring is not part of `let` declarations; use `match` array patterns
when you need to inspect array shape.

Rectangular declarations bind a new tensor by iterating over index variables:

```rust
let scaled[i, j] = data[i, j] * 2.0;
```

This produces a new tensor whose shape matches `data`. Each element is computed independently. Index variables introduced on the left-hand side (`i`, `j` here) are in scope for the body expression and the where clause, but not outside this statement.

### `const` declarations

`const` binds a named constant expression:

```rust
const HIDDEN: i32 = 256;
const EPS: f32 = 1e-6;
```

Constants participate in name resolution like other module items and can be
used from later declarations. They are immutable and are intended for values
that should not be recomputed at each use site.

### `fn` declarations

```rust
fn add(a, b) { a + b }

fn clamp(x: f32, lo: f32, hi: f32) -> f32 {
    if x < lo { lo }
    else if x > hi { hi }
    else { x }
}

pub fn exported(x) { x * 2 }
```

The return value is the final expression in the block. There is no `return` keyword. If the block ends with a statement (trailing `;`), the return value is unit.

Parameters without type annotations accept any type; the compiler monomorphizes the function at each call site (see Monomorphization below). Parameters with annotations are checked at the call site.

`pub` makes the function visible to other modules.

Functions are hoisted: a function can be called before its textual definition in the same block.

### `@fn` custom differentiation rules

A function can provide a custom autodiff rule with `@fn`:

```rust
fn ratio(x, y) { x / y }

@fn ratio(x, y) {
    (y * @x - x * @y) / y ** 2.0
}
```

The `@fn` declaration has the same function name and parameter list as the
primal function. Inside the rule body, `@x` means the tangent flowing through
parameter `x`, `@y` means the tangent flowing through parameter `y`, and so on.
The body describes how the output tangent is assembled from parameter
tangents.

Custom rules are useful when the primal function calls an external
implementation or intentionally chooses a surrogate derivative. For example,
the standard library defines `exp` through NumPy and gives it a rule:

```rust
pub fn exp(x) {
    python::numpy::exp(x)
}

@fn exp(x) { exp(x) * @x }
```

The rule is used only when an autodiff request reaches a call to the matching
function. If no derivative request is present, the `@fn` body is dormant.

### `use` declarations

```rust
use std::math::{sin, cos, pi};    // import specific names
use std::array::*;                 // import all exports
use std::math as m;                // import with alias; use as m::sqrt(4.0)
```

Imports bring names into the current scope. Module paths are resolved relative to the project root and the `stdlib/` directory.

`pub use` re-exports imported names, making them visible to importers of the current module:

```rust
pub use std::math::sin;
```

---

## Types

### Scalar types

| Type | Description | Default literal |
|------|-------------|-----------------|
| `i8` | 8-bit signed integer | explicit cast |
| `i32` | 32-bit signed integer | `42` |
| `i64` | 64-bit signed integer | requires annotation |
| `f8e4m3` | 8-bit floating point, E4M3 format | explicit cast |
| `f16` | 16-bit float | explicit cast |
| `bf16` | bfloat16 | explicit cast |
| `f32` | 32-bit float | `3.14` |
| `f64` | 64-bit float | requires annotation |
| `bool` | Boolean | `true`, `false` |
| `str` | String | `"hello"` |

Integer literals default to `i32`, float literals to `f32`. To get `i64` or `f64`, use a type annotation:

```rust
let x: i64 = 42;       // literal coerced to i64
let y: f64 = 3.14;     // literal coerced to f64
```

For narrower or reduced-precision numeric types, use an explicit cast:

```rust
let a = 1 as i8;
let b = 1.0 as f16;
let c = 1.0 as bf16;
let d = 1.0 as f8e4m3;
```

### Rectangular types

A rectangular array has a fixed element type and a fixed number of dimensions. All sub-arrays at the same depth have the same length. This is the array kind used by Einstein notation.

```rust
let v: [f32] = [1.0, 2.0, 3.0];           // 1D, size unknown at compile time
let m: [f32; 3, 4] = load_matrix();        // 2D, compile-time known 3×4
let t: [f32; ?, ?] = load_matrix();        // 2D, both dimensions unknown
let d: [f32; *] = load_from_file("w.npz"); // dynamic rank (number of dims unknown)
```

`?` is a wildcard dimension: it matches any concrete size during assignment checks, but rank (number of dimensions) must still match. So `[i32; ?, ?]` accepts any 2D integer array but rejects a 1D or 3D array.

Shape inference from literals: `[[1,2],[3,4]]` has shape `(2, 2)`. Inconsistent row lengths are a compile-time error.

### Jagged types

Jagged arrays allow variable-length sub-arrays. They cannot be used with Einstein notation.

```rust
let ragged: jagged[i32] = [[1, 2], [3, 4, 5]];   // rows have different lengths
```

### Function types

```rust
let f: (f32, f32) -> f32 = add;
```

### Type compatibility

1. **Same type**: always compatible.
2. **`unknown`**: compatible with anything (gradual typing during inference).
3. **Literal coercion**: numeric literals can be coerced at the binding site.

```rust
let a: i64 = 42;       // OK: 42 is a literal, coerced to i64
let b: i32 = 42;
let c: i64 = b;        // ERROR: b is not a literal, no implicit widening
let d: i64 = b as i64; // OK: explicit cast
```

4. **Rectangular types**: element types must match exactly. Rank must match. Each dimension in the expected type must either equal the actual dimension or be `?`.

```rust
let m: [i32; ?, ?] = [[1, 2, 3], [4, 5, 6]];   // OK: ? matches 2 and 3
let n: [i32; 2, ?] = [[1, 2, 3], [4, 5, 6]];   // OK: 2 matches, ? matches 3
let p: [i32; 3, ?] = [[1, 2, 3], [4, 5, 6]];   // ERROR: first dim is 2, not 3
let q: [i32; ?] = [[1, 2], [3, 4]];             // ERROR: rank 1 vs rank 2
```

5. **Jagged and rectangular** are not interchangeable.

### Cast expressions

Explicit conversion between numeric types. No implicit widening or narrowing.

```rust
let x: i32 = 42;
let y = x as f64;       // 42.0
let z = 3.14 as i32;    // 3 (truncates toward zero)
```

---

## Expressions

Everything in Einlang is an expression (except declarations). Blocks, `if`, and `match` all produce values.

### Literals

```rust
42          // i32
3.14        // f32
true        // bool
"hello"     // str
[1, 2, 3]  // array literal
(1, 2)      // tuple literal; access with .0, .1 only
```

### String interpolation

Strings support `{expr}` interpolation. Use `{{` and `}}` for literal braces. Format specifiers follow the expression after `:`.

```rust
let name = "world";
let msg = "hello {name}";          // "hello world"
let fmt = "pi = {pi:.4f}";        // "pi = 3.1416"
let escaped = "literal {{braces}}"; // "literal {braces}"
```

### Operators

Precedence from lowest (loosest) to highest (tightest):

| Precedence | Operators | Associativity |
|------------|-----------|---------------|
| 1 | `\|\|` | left |
| 2 | `&&` | left |
| 3 | `==`, `!=` | left |
| 4 | `<`, `>`, `<=`, `>=` | left |
| 5 | `+`, `-` | left |
| 6 | `*`, `/`, `%` | left |
| 7 | `**` | right |
| 8 | `!`, unary `-` | prefix |

Subtleties:
- Integer division truncates toward zero: `7 / 2` is `3`, `-7 / 2` is `-3`.
- `%` returns the remainder with the sign of the dividend: `-7 % 3` is `-1`.
- `**` is right-associative: `2 ** 3 ** 2` is `2 ** 9 = 512`, not `8 ** 2 = 64`.
- All arithmetic operators require operands of the same type. `1 + 1.0` is an error; write `1.0 + 1.0` or `(1 as f32) + 1.0`.
- **Broadcasting:** Operators support broadcasting by default only for **same rank** (tensor with same-shape tensor) or **tensor vs scalar**. In those cases explicit indexing is not required.

**Broadcasting examples (no explicit indexing):**

```rust
let A = [[1.0, 2.0], [3.0, 4.0]];

let scaled = A * 2.0;                       // scalar * tensor: every element doubled
let shifted = A + 10.0;                     // scalar + tensor
let normalized = (A - 2.5) / 1.5;           // scalar mean/std broadcast

let B = [[1.0, 1.0], [1.0, 1.0]];
let sum_AB = A + B;                         // same rank, same shape: element-wise add
```

For different-rank combinations (e.g. vector with matrix), use rectangular `let` with explicit indices: `let out[i, j] = A[i, j] + bias[j];`.

- `**` is the exception — it allows mixed base/exponent types, following Rust's `pow`/`powi`/`powf` pattern:

```rust
let a = 2 ** 10;        // i32 ** i32 → i32 (integer pow)
let b = 2.0 ** 3;       // f32 ** i32 → f32 (like Rust's powi)
let c = 2.0 ** 0.5;     // f32 ** f32 → f32 (like Rust's powf)
```

`sqrt` in the stdlib is `x ** 0.5`, so it requires a float argument — same as Rust where `sqrt` is only defined on `f32`/`f64`. Pass an integer and you get a type error; use `sqrt(x as f32)` to convert first.

```rust
use std::math::basic::sqrt;

let r = sqrt(16.0);         // 4.0 — OK
let s = sqrt(16 as f32);    // 4.0 — OK, explicit cast
let t = sqrt(16);            // ERROR: i32 has no sqrt
```

The same applies to other `std::math` functions (`sin`, `cos`, `exp`, `ln`, etc.) — they operate on floats only.

### `if` expressions

`if` is an expression that returns a value. Both branches must produce the same type.

```rust
let abs_x = if x >= 0 { x } else { -x };

let category = if x > 100 { "large" }
    else if x > 10 { "medium" }
    else { "small" };
```

When `if` is used as a statement (result discarded), the `else` branch can be omitted.

### `match` expressions

Arms are evaluated top-to-bottom; the first matching pattern wins. All arms must produce the same type.

```rust
let label = match n {
    0 => "zero",
    1 => "one",
    _ => "other",    // wildcard: matches anything, binds nothing
};
```

Pattern kinds:
- **Literal**: `0`, `true`, `"hello"` — matches by value.
- **Wildcard**: `_` — matches anything, does not bind.
- **Identifier**: `x` — matches anything and binds the matched value to `x` in the arm body.
- **Tuple**: `(x, y)` — matches fixed-arity tuple values and binds elements.
- **Array**: `[head, ..tail]`, `[..prefix, last]`, `[first, ..middle, last]` — matches array length/shape and optionally binds a rest array.
- **Range**: `0..10`, `0..=10` — matches numeric literals in an exclusive or inclusive range.
- **Binding**: `whole @ pattern` — binds the whole matched value while also matching the nested pattern.
- **Or pattern**: `0 | 1 => ...` — tries several patterns for the same arm.
- **Guard**: `pattern where condition => ...` — accepts the arm only when the condition is true.

The compiler checks exhaustiveness: a `match` without `_` or an identifier catch-all must cover all possible values.

### Block expressions

A block evaluates its statements in order, then returns its final expression. Variables declared inside are scoped to the block.

```rust
let result = {
    let a = compute_a();
    let b = compute_b();
    a + b       // this is the block's value
};
```

If the last item in the block is a statement (ends with `;`), the block returns unit.

### Array access

For rectangular arrays, comma-separated indices in a single bracket operation. Each index reduces rank by one:

```rust
let matrix = [[1, 2, 3], [4, 5, 6]];
let row = matrix[0];       // [1, 2, 3] — shape goes from (2,3) to (3,)
let elem = matrix[0, 1];   // 2 — scalar
```

For jagged arrays, use chained brackets: `A[i][j]`.

### Tuple expressions and tuple access

Tuple literals: `(a, b)` or `(1, 2, 3)`. Tuple elements are accessed **only** by dot and a zero-based field index: `t.0`, `t.1`, `t.2`, and so on. Do not use bracket notation for tuples; `t[0]` is array access.

```rust
let p = (1.0, 2.0);
let x = p.0;
let y = p.1;
```

### Ranges

Two forms are supported:

- **`a..b`** — exclusive: integers from `a` up to but not including `b`. Example: `0..3` is `[0, 1, 2]`.
- **`a..=b`** — inclusive: integers from `a` through `b`. Example: `0..=3` is `[0, 1, 2, 3]`.

Both are used in comprehension generators, explicit Einstein index domains, and recurrence bounds.

### Pipeline expressions

The standard pipeline operator passes the value on the left to the callable on
the right:

```rust
let result = 42
    |> |x| x * 2
    |> |x| x + 10;
```

Pipeline expressions are primarily useful with lambdas or named functions when
you want to write a sequence of transformations in data-flow order.

---

## Einstein Notation

The core feature for tensor computation. Named index variables declare how to iterate over tensor dimensions; the compiler infers ranges from array shapes.

### Rectangular declarations with indices

Index variables on the left-hand side define the output tensor's dimensions. The compiler determines each index range by examining how the variable is used to index arrays in the body.

```rust
let C[i, j] = sum[k](A[i, k] * B[k, j]);
```

Here `i` ranges over `0..A.shape[0]`, `j` over `0..B.shape[1]`, and `k` over `0..A.shape[1]` (which must equal `B.shape[0]`). If they don't match, the compiler reports E004 (shape mismatch).

Element-wise operations don't need a reduction:

```rust
let doubled[i, j] = matrix[i, j] * 2.0;
let sum_AB[i, j] = A[i, j] + B[i, j];
```

Index slots may be:

- A name: `i`
- A name with an explicit domain: `i in 0..n`
- A literal base-case index: `0`
- A named rest index: `..batch`

Named rest indices stand for zero or more adjacent axes and are useful for
batch-polymorphic code:

```rust
let result[..batch, j] = x[..batch, j] + bias[j];
let row_sum[..batch] = sum[j](x[..batch, j]);
let total = sum[..batch](row_sum[..batch]);
```

The same rest name must describe the same axis span within the expression.
If `..batch` is inferred from `x[..batch, j]`, it can be reused on the output
and in reductions over those axes.

### Reductions

A reduction iterates over its index variables and combines values. Available operations: `sum`, `max`, `min`, `prod`.

```rust
let total = sum[i](data[i]);                         // scalar
let row_sums[i] = sum[j](matrix[i, j]);              // 1D
let explicit = sum[i in 0..10](data[i]);              // explicit range
let max_per_row[i] = max[j](matrix[i, j]);            // max reduction
```

Identity elements: `sum` starts from 0, `prod` from 1, `max` from negative infinity, `min` from positive infinity. The body is evaluated once per combination of index values.

### Range inference

When a reduction index `k` appears as `A[..., k, ...]` in the body, the compiler infers `k in 0..A.shape[axis]` where `axis` is the position of `k` in the indexing expression. If `k` indexes multiple arrays at different positions, the inferred ranges must agree; a mismatch is E004.

```rust
// k indexes A at axis 1 (shape[1]) and B at axis 0 (shape[0])
// so A.shape[1] must equal B.shape[0]
let C[i, j] = sum[k](A[i, k] * B[k, j]);
```

### Scoping rules

Index variables introduced in `sum[k]` or on the left-hand side `let C[i, j]` are in scope for:
- The reduction/declaration body.
- The `where` clause attached to that expression.

They are **not** in scope outside the statement.

```rust
let row_sums[i] = sum[j](matrix[i, j]);
// i and j are NOT available here
let x = row_sums[0];   // access the result by concrete index
```

### Restriction to rectangular types

Einstein notation requires all indexed arrays to be rectangular. Jagged arrays cannot be used. Attempting `jagged_arr[i, j]` in an Einstein expression is a compile-time error.

---

## Where Clauses

A where clause attaches to a rectangular declaration or a reduction. Constraints are evaluated for each combination of the enclosing index variables.

### Variable binding

Binds a name to a computed value. Useful for avoiding repeated computation. Bindings are evaluated in order; later bindings can reference earlier ones.

```rust
let output[i, j] = activated
    where z = sum[k](input[i, k] * weight[k, j]) + bias[j],
          activated = if z > 0.0 { z } else { 0.0 };
```

Without the where clause, you'd have to write the `sum[k](...)` expression twice (once for the comparison, once for the value).

### Index arithmetic

Write derived coordinates directly in index positions. The compiler uses these
expressions to determine the valid iteration space.

```rust
let conv[b, oc, oh, ow] = sum[ic, kh, kw](
    input[b, ic, oh + kh, ow + kw] * weight[oc, ic, kh, kw]
);
```

Here `oh + kh` and `ow + kw` are coordinate expressions, not new axes. The
compiler ensures the resulting indices stay within `input`'s bounds.

### Boolean guards

A bare expression in the where clause acts as a filter.

```rust
let pos_sum = sum[i](data[i]) where data[i] > 0;
let upper[i, j] = matrix[i, j] where i <= j;
```

For reductions, elements where the guard is false are skipped (the identity element is used instead). For rectangular declarations, the default value (zero for numeric types) is used for filtered-out positions.

---

## Array Comprehensions

Produces a new array by iterating over generators left-to-right (nested). Filters discard elements where the condition is false.

```rust
let squares = [i * i | i in 1..5];               // [1, 4, 9, 16, 25]
let evens = [i | i in 1..100, i % 2 == 0];       // [2, 4, 6, ..., 100]
let pairs = [(i, j) | i in 0..3, j in 0..3, i != j];
```

Unlike Einstein notation, comprehensions do not require rectangular inputs and can produce variable-length output. The result length depends on the filters, so when filters are present, the output size is not known at compile time.

---

## Recurrence Relations

Self-referential rectangular declarations that define sequences. Base cases are evaluated first; the recursive case is evaluated in index order so earlier elements are available when computing later ones.

```rust
let fib[0] = 0;
let fib[1] = 1;
let fib[n in 2..8] = fib[n-1] + fib[n-2];
// fib = [0, 1, 1, 2, 3, 5, 8, 13]
```

**Alignment with math:** You cannot read **future** values: when defining the element at index `(t, i, j)`, you must not read the same array at an index that is not yet computed (e.g. `h[t+1, i, j]` or `h[t, i+1, j]`). Use **backward references only** along every dimension: e.g. `h[t-1, i, j]` in time, `h[t, i-1, j]` and `h[t, i, j-1]` in space. So for time index `t` use `h[t-1, ...]`, not `h[t+1, ...]`; for space index `i` use `h[t, i-1, j]`, not `h[t, i+1, j]`.

The recurrence index range goes **in the bracket** (`n in 2..8`), not in a `where` clause.

**Declaration bracket:** Each index slot in `let x[...] = ...` may only be an **identifier** (e.g. `n`, `i`, `t`) or a **literal** (e.g. `0`, `1`) or a named rest (`..name`). Expressions like `n-1` or `t+1` are **not** allowed in the declaration bracket. In the **body**, use only **backward** references when reading the same array (no future indices in any dimension).

**Example (time + space, backward only):** `let h[t in 1..T, i in 1..N-1, j in 1..N-1] = f(h[t-1, i, j], h[t, i-1, j], h[t, i, j-1]);` — valid. Do not read `h[t, i+1, j]` or `h[t, i, j+1]` when defining `h[t, i, j]`; that would be a future value.

---

## Lambda Expressions

Creates an anonymous function. The body is a single expression. Lambdas capture variables from the enclosing scope.

```rust
let double = |x| x * 2;
let add = |a, b| a + b;
let result = (|x| x + 1)(5);   // immediately invoked: 6

let factor = 3;
let scale = |x| x * factor;    // captures 'factor' from enclosing scope
let y = scale(10);              // 30
```

Lambdas can be stored in variables, passed to functions, and returned from functions.

---

## Monomorphization

Untyped function parameters cause the compiler to generate a specialized copy for each distinct set of argument types at the call site:

```rust
fn double(x) { x * 2 }
let a = double(3);      // specializes for i32
let b = double(3.14);   // specializes for f32
```

Both calls succeed. If the body doesn't make sense for a given type (e.g., calling a numeric operation on a string), the error appears at the specialized call site.

---

## Redeclaration

A `let` binding may not redefine a name that is already bound in the same
scope. Write a fresh name instead.

```rust
let x = 10;
let x = x + 1;       // ERROR: redefinition of `x` in the same scope
let x_next = x + 1;  // OK
```

Inner scopes may still introduce local names without mutating outer bindings.
Same-scope duplicate declarations are rejected so each binding has a stable
meaning.

---

## Module System

### File layout

A file is a module. The name comes from the file path relative to the project root.

```
project/
├── main.ein              → (entry point)
├── utils.ein             → module utils
└── stdlib/
    ├── math/
    │   ├── basic.ein     → module std::math::basic
    │   └── trig.ein      → module std::math::trig
    └── array.ein         → module std::array
```

### Visibility

All declarations are private by default. `pub` makes them visible to importers.

```rust
pub fn exported(x) { x * 2 }   // visible to importers
fn internal(x) { x + 1 }       // only visible in this file
```

### Module declarations and re-exports

Use `mod` to declare a submodule file, and `pub mod` to make it visible through
the current module:

```rust
mod basic;
pub mod math;
```

`pub use` re-exports imported names:

```rust
pub use math::*;
pub use math::sqrt as root;
```

### Name resolution order

When resolving a name, the compiler searches:
1. Local scope (let bindings, function parameters, index variables)
2. Current module scope (fn declarations, use imports)
3. Builtins (`print`, `assert`, `len`, `shape`, `typeof`)

---

## Built-in vs stdlib

- **Built-ins** are language primitives: a small, fixed set known to the compiler and runtime (e.g. by DefId in a builtin crate). They are available without any `use` and cannot be removed. Examples: `print`, `assert`, `len`, `shape`, `typeof`, `array_append`, `sum`, `max`, `min`.
- **Stdlib** is the standard library: modules and functions implemented in Einlang (`.ein` in `stdlib/`). You bring them in with `use std::math::{...};` etc. They are normal code; the compiler does not treat them as primitives. Many call out to Python/NumPy or (in future) C under the hood.

So: built-in = part of the language; stdlib = library that ships with the language.

---

## Built-in Functions

Available without any import:

| Function | Description |
|----------|-------------|
| `print(args...)` | Print values to stdout |
| `assert(cond)` | Abort if false |
| `assert(cond, msg)` | Abort with message if false |
| `len(arr)` | Length of first dimension |
| `shape(arr)` | Full shape as array of dimension sizes |
| `typeof(val)` | Type name as string |

Reduction operations (`sum`, `max`, `min`, `prod`) use Einstein notation syntax, not function-call syntax.

See [Standard Library](https://github.com/einlang/einlang/blob/main/docs/stdlib.md) for `std::math`, `std::array`, `std::ml`, and `std::io`.

---

## Automatic differentiation

The compiler supports **built-in automatic differentiation**: you can request tangents, derivatives, and Jacobians directly in the language without a separate `grad(f)` wrapper API. The current executable path is centered on **named bindings**: bind the value first, then differentiate it. For an overview of the current compiler/runtime design, see [AUTODIFF.md](AUTODIFF.md).

- **`@x`** — for a named binding `x`, the executable value form materializes the identity tangent seed of `x` (`1.0` for scalars, ones-like for tensors). Direct `print(@x)` is instead a symbolic display path.
- **`@a / @b`** — the numeric derivative/Jacobian of named binding `a` with respect to named binding `b`. Scalar/scalar cases evaluate to a scalar; tensor cases use a lazy Jacobian-backed runtime value.
- **`@fn f(...) { ... }`** — a custom tangent rule for calls to function `f`. See [`@fn` custom differentiation rules](#fn-custom-differentiation-rules).
- **Mental model** — write the program value first, then differentiate that value in place: `@loss / @w`, `@state / @dt`, and `@C / @A` are ordinary Einlang expressions, not calls to a separate gradient API.

**Matmul, conv, einsum:** Derivatives of Einstein expressions are supported: e.g. `let C[i,j] = sum[k](A[i,k]*B[k,j]); let dC_dA = @C / @A;` (matmul), or convolution written as `sum[kh,kw](in[oh+kh,ow+kw]*w[kh,kw])`. Any sum-of-products declaration can be differentiated w.r.t. any input array. See [AUTODIFF.md](AUTODIFF.md).

Example:

```rust
let x = 1.0;
let y = 2.0;
let z = x + y;
let dz_dx = @z / @x;   // 1.0
let dz_dy = @z / @y;   // 1.0
print(dz_dx);
print(dz_dy);
```

Custom rules compose with ordinary derivative requests:

```rust
fn square_plus_one(x) { x * x + 1.0 }
@fn square_plus_one(x) { 2.0 * x * @x }

let x = 3.0;
let y = square_plus_one(x);
let dy_dx = @y / @x;   // 6.0
```

The compiler derives gradients via the chain rule, but the current implementation answers autodiff requests through runtime JVP/VJP machinery instead of emitting a standalone derivative IR program. Supported operations and rules are documented in [AUTODIFF.md](AUTODIFF.md). Examples: run `python3 examples/run_autodiff_examples.py` or see [examples/](https://github.com/einlang/einlang/tree/main/examples) for scalar checks (`autodiff_small.ein`, `autodiff_matmul.ein`), fitting workflows (`applications/linear_regression_autodiff.ein`, `applications/decay_calibration_autodiff.ein`), and training-style workloads (`gradient_descent_autodiff.ein`, `mnist/train_recurrence.ein`).

---

## Error Codes

| Code | Name | Trigger |
|------|------|---------|
| E001 | Syntax Error | Missing semicolon, mismatched brackets |
| E002 | Type Mismatch | Incompatible types in assignment or operation |
| E003 | Undefined Variable | Reference to undeclared name |
| E004 | Shape Mismatch | Incompatible tensor dimensions in Einstein notation |
| E005 | Invalid Index | Array index out of bounds |
| E006 | Invalid Value | Domain error (e.g. `sqrt(-1)`) |
| E007 | Runtime Error | Division by zero, overflow |
| E008 | Memory Error | Allocation too large |
| E009 | I/O Error | File not found, permission denied |
| E010 | Import Error | Module not found |
| E011 | Not Implemented | Feature exists in grammar but not backend |

---

## Unsupported by design

Einlang intentionally does not support some familiar constructs:

- no `for` or `while`; use comprehensions, Einstein notation, or recurrences
- no `return`; the last expression in a block is the value
- no string-based `einsum`; use named indices directly
- no implicit numeric widening; use explicit casts
- no slice `:` syntax; build the view you want with indices

These are part of the language shape, not temporary omissions.

---

## Planned Features

The following are parsed by the grammar but not yet executed by the backend.

**Pipeline operators**: `|>` (deterministic), `?>` (optional), `!>` (fallible), with `else` and `catch` clauses.

```rust
let result = data |> normalize |> transform;
let safe = data !> parse !> validate catch |e| default;
```

**Try expressions**: `try expr` wraps a failable expression into a Result type.
