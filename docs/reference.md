# Einlang Language Reference

---

## Statements

A program is a sequence of statements. Every statement ends with `;`.

### `let` declarations

All bindings are immutable. The type annotation is optional; when omitted the type is inferred from the right-hand side.

```einlang
let x = 42;                   // inferred i32
let pi: f64 = 3.141592653589793;   // explicit f64
let matrix: [f32; 2, 3] = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
```

If both annotation and value are present, the value must be assignment-compatible with the annotation (see Type Compatibility).

Rectangular declarations bind a new tensor by iterating over index variables:

```einlang
let scaled[i, j] = data[i, j] * 2.0;
```

This produces a new tensor whose shape matches `data`. Each element is computed independently. Index variables introduced on the left-hand side (`i`, `j` here) are in scope for the body expression and the where clause, but not outside this statement.

### `fn` declarations

```einlang
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

### `use` declarations

```einlang
use std::math::{sin, cos, pi};    // import specific names
use std::array::*;                 // import all exports
use std::math as m;                // import with alias; use as m::sqrt(4.0)
```

Imports bring names into the current scope. Module paths are resolved relative to the project root and the `stdlib/` directory.

`pub use` re-exports imported names, making them visible to importers of the current module:

```einlang
pub use std::math::sin;
```

---

## Types

### Scalar types

| Type | Description | Default literal |
|------|-------------|-----------------|
| `i32` | 32-bit signed integer | `42` |
| `i64` | 64-bit signed integer | requires annotation |
| `f32` | 32-bit float | `3.14` |
| `f64` | 64-bit float | requires annotation |
| `bool` | Boolean | `true`, `false` |
| `str` | String | `"hello"` |

Integer literals default to `i32`, float literals to `f32`. To get `i64` or `f64`, use a type annotation:

```einlang
let x: i64 = 42;       // literal coerced to i64
let y: f64 = 3.14;     // literal coerced to f64
```

### Rectangular types

A rectangular array has a fixed element type and a fixed number of dimensions. All sub-arrays at the same depth have the same length. This is the array kind used by Einstein notation.

```einlang
let v: [f32] = [1.0, 2.0, 3.0];           // 1D, size unknown at compile time
let m: [f32; 3, 4] = load_matrix();        // 2D, compile-time known 3×4
let t: [f32; ?, ?] = load_matrix();        // 2D, both dimensions unknown
let d: [f32; *] = load_from_file("w.npz"); // dynamic rank (number of dims unknown)
```

`?` is a wildcard dimension: it matches any concrete size during assignment checks, but rank (number of dimensions) must still match. So `[i32; ?, ?]` accepts any 2D integer array but rejects a 1D or 3D array.

Shape inference from literals: `[[1,2],[3,4]]` has shape `(2, 2)`. Inconsistent row lengths are a compile-time error.

### Jagged types

Jagged arrays allow variable-length sub-arrays. They cannot be used with Einstein notation.

```einlang
let ragged: jagged[i32] = [[1, 2], [3, 4, 5]];   // rows have different lengths
```

### Function types

```einlang
let f: (f32, f32) -> f32 = add;
```

### Type compatibility

1. **Same type**: always compatible.
2. **`unknown`**: compatible with anything (gradual typing during inference).
3. **Literal coercion**: numeric literals can be coerced at the binding site.

```einlang
let a: i64 = 42;       // OK: 42 is a literal, coerced to i64
let b: i32 = 42;
let c: i64 = b;        // ERROR: b is not a literal, no implicit widening
let d: i64 = b as i64; // OK: explicit cast
```

4. **Rectangular types**: element types must match exactly. Rank must match. Each dimension in the expected type must either equal the actual dimension or be `?`.

```einlang
let m: [i32; ?, ?] = [[1, 2, 3], [4, 5, 6]];   // OK: ? matches 2 and 3
let n: [i32; 2, ?] = [[1, 2, 3], [4, 5, 6]];   // OK: 2 matches, ? matches 3
let p: [i32; 3, ?] = [[1, 2, 3], [4, 5, 6]];   // ERROR: first dim is 2, not 3
let q: [i32; ?] = [[1, 2], [3, 4]];             // ERROR: rank 1 vs rank 2
```

5. **Jagged and rectangular** are not interchangeable.

### Cast expressions

Explicit conversion between numeric types. No implicit widening or narrowing.

```einlang
let x: i32 = 42;
let y = x as f64;       // 42.0
let z = 3.14 as i32;    // 3 (truncates toward zero)
```

---

## Expressions

Everything in Einlang is an expression (except declarations). Blocks, `if`, and `match` all produce values.

### Literals

```einlang
42          // i32
3.14        // f32
true        // bool
"hello"     // str
[1, 2, 3]  // array literal
```

### String interpolation

Strings support `{expr}` interpolation. Use `{{` and `}}` for literal braces. Format specifiers follow the expression after `:`.

```einlang
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
- `**` is the exception — it allows mixed base/exponent types, following Rust's `pow`/`powi`/`powf` pattern:

```einlang
let a = 2 ** 10;        // i32 ** i32 → i32 (integer pow)
let b = 2.0 ** 3;       // f32 ** i32 → f32 (like Rust's powi)
let c = 2.0 ** 0.5;     // f32 ** f32 → f32 (like Rust's powf)
```

`sqrt` in the stdlib is `x ** 0.5`, so it requires a float argument — same as Rust where `sqrt` is only defined on `f32`/`f64`. Pass an integer and you get a type error; use `sqrt(x as f32)` to convert first.

```einlang
use std::math::basic::sqrt;

let r = sqrt(16.0);         // 4.0 — OK
let s = sqrt(16 as f32);    // 4.0 — OK, explicit cast
let t = sqrt(16);            // ERROR: i32 has no sqrt
```

The same applies to other `std::math` functions (`sin`, `cos`, `exp`, `ln`, etc.) — they operate on floats only.

### `if` expressions

`if` is an expression that returns a value. Both branches must produce the same type.

```einlang
let abs_x = if x >= 0 { x } else { -x };

let category = if x > 100 { "large" }
    else if x > 10 { "medium" }
    else { "small" };
```

When `if` is used as a statement (result discarded), the `else` branch can be omitted.

### `match` expressions

Arms are evaluated top-to-bottom; the first matching pattern wins. All arms must produce the same type.

```einlang
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

The compiler checks exhaustiveness: a `match` without `_` or an identifier catch-all must cover all possible values.

### Block expressions

A block evaluates its statements in order, then returns its final expression. Variables declared inside are scoped to the block.

```einlang
let result = {
    let a = compute_a();
    let b = compute_b();
    a + b       // this is the block's value
};
```

If the last item in the block is a statement (ends with `;`), the block returns unit.

### Array access

For rectangular arrays, comma-separated indices in a single bracket operation. Each index reduces rank by one:

```einlang
let matrix = [[1, 2, 3], [4, 5, 6]];
let row = matrix[0];       // [1, 2, 3] — shape goes from (2,3) to (3,)
let elem = matrix[0, 1];   // 2 — scalar
```

For jagged arrays, use chained brackets: `A[i][j]`.

### Ranges

`a..b` produces the integer sequence from `a` to `b` exclusive: `0..3` is `[0, 1, 2]`.

Used in comprehension generators, explicit Einstein index domains, and recurrence bounds.

---

## Einstein Notation

The core feature for tensor computation. Named index variables declare how to iterate over tensor dimensions; the compiler infers ranges from array shapes.

### Rectangular declarations with indices

Index variables on the left-hand side define the output tensor's dimensions. The compiler determines each index range by examining how the variable is used to index arrays in the body.

```einlang
let C[i, j] = sum[k](A[i, k] * B[k, j]);
```

Here `i` ranges over `0..A.shape[0]`, `j` over `0..B.shape[1]`, and `k` over `0..A.shape[1]` (which must equal `B.shape[0]`). If they don't match, the compiler reports E004 (shape mismatch).

Element-wise operations don't need a reduction:

```einlang
let doubled[i, j] = matrix[i, j] * 2.0;
let sum_AB[i, j] = A[i, j] + B[i, j];
```

### Reductions

A reduction iterates over its index variables and combines values. Available operations: `sum`, `max`, `min`, `prod`.

```einlang
let total = sum[i](data[i]);                         // scalar
let row_sums[i] = sum[j](matrix[i, j]);              // 1D
let explicit = sum[i in 0..10](data[i]);              // explicit range
let max_per_row[i] = max[j](matrix[i, j]);            // max reduction
```

Identity elements: `sum` starts from 0, `prod` from 1, `max` from negative infinity, `min` from positive infinity. The body is evaluated once per combination of index values.

### Range inference

When a reduction index `k` appears as `A[..., k, ...]` in the body, the compiler infers `k in 0..A.shape[axis]` where `axis` is the position of `k` in the indexing expression. If `k` indexes multiple arrays at different positions, the inferred ranges must agree; a mismatch is E004.

```einlang
// k indexes A at axis 1 (shape[1]) and B at axis 0 (shape[0])
// so A.shape[1] must equal B.shape[0]
let C[i, j] = sum[k](A[i, k] * B[k, j]);
```

### Scoping rules

Index variables introduced in `sum[k]` or on the left-hand side `let C[i, j]` are in scope for:
- The reduction/declaration body.
- The `where` clause attached to that expression.

They are **not** in scope outside the statement.

```einlang
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

```einlang
let output[i, j] = activated
    where z = sum[k](input[i, k] * weight[k, j]) + bias[j],
          activated = if z > 0.0 { z } else { 0.0 };
```

Without the where clause, you'd have to write the `sum[k](...)` expression twice (once for the comparison, once for the value).

### Index remapping

Bind derived indices to expressions of the output indices. The compiler uses these equalities to determine the valid iteration space.

```einlang
let conv[b, oc, oh, ow] = sum[ic, kh, kw](
    input[b, ic, ih, iw] * weight[oc, ic, kh, kw]
) where ih = oh + kh, iw = ow + kw;
```

Here `ih` and `iw` are not free variables — they are computed from `oh + kh` and `ow + kw`. The compiler ensures the resulting indices stay within `input`'s bounds.

### Boolean guards

A bare expression in the where clause acts as a filter.

```einlang
let pos_sum = sum[i](data[i]) where data[i] > 0;
let upper[i, j] = matrix[i, j] where i <= j;
```

For reductions, elements where the guard is false are skipped (the identity element is used instead). For rectangular declarations, the default value (zero for numeric types) is used for filtered-out positions.

---

## Array Comprehensions

Produces a new array by iterating over generators left-to-right (nested). Filters discard elements where the condition is false.

```einlang
let squares = [i * i | i in 1..5];               // [1, 4, 9, 16, 25]
let evens = [i | i in 1..100, i % 2 == 0];       // [2, 4, 6, ..., 100]
let pairs = [(i, j) | i in 0..3, j in 0..3, i != j];
```

Unlike Einstein notation, comprehensions do not require rectangular inputs and can produce variable-length output. The result length depends on the filters, so when filters are present, the output size is not known at compile time.

---

## Recurrence Relations

Self-referential rectangular declarations that define sequences. Base cases are evaluated first; the recursive case is evaluated in index order so earlier elements are available when computing later ones.

```einlang
let fib[0] = 0;
let fib[1] = 1;
let fib[n] = fib[n-1] + fib[n-2] where n in 2..8;
// fib = [0, 1, 1, 2, 3, 5, 8, 13]
```

Backward references (`n-1`, `n-2`) and forward definitions (`t+1`) are both supported and semantically equivalent:

```einlang
let seq[0] = 1;
let seq[t+1] = seq[t] * 2 where t in 0..5;
// seq = [1, 2, 4, 8, 16, 32]
```

Multi-dimensional recurrences work the same way — the time axis advances while other axes iterate freely:

```einlang
let hidden[0, i] = initial[i] where i in 0..H;
let hidden[t+1, i] = tanh(hidden[t, i] + input[t, i])
    where t in 0..T, i in 0..H;
```

---

## Lambda Expressions

Creates an anonymous function. The body is a single expression. Lambdas capture variables from the enclosing scope.

```einlang
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

```einlang
fn double(x) { x * 2 }
let a = double(3);      // specializes for i32
let b = double(3.14);   // specializes for f32
```

Both calls succeed. If the body doesn't make sense for a given type (e.g., calling a numeric operation on a string), the error appears at the specialized call site.

---

## Shadowing

A `let` binding or inner `fn` can shadow an outer name. The inner binding takes precedence within its scope.

```einlang
let x = 10;
let x = x + 1;     // shadows previous x; x is now 11

fn min(a, b) { a * 2 }   // shadows the builtin min
let r = min(3, 4);        // 6, not 3
```

Builtins can be shadowed. The shadowing is lexical — it applies within the block where the new binding is introduced.

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

```einlang
pub fn exported(x) { x * 2 }   // visible to importers
fn internal(x) { x + 1 }       // only visible in this file
```

### Name resolution order

When resolving a name, the compiler searches:
1. Local scope (let bindings, function parameters, index variables)
2. Current module scope (fn declarations, use imports)
3. Builtins (`print`, `assert`, `len`, `shape`, `typeof`)

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

See [Standard Library](stdlib.md) for `std::math`, `std::array`, `std::ml`, and `std::io`.

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

## Planned Features

The following are parsed by the grammar but not yet executed by the backend.

**Pipeline operators**: `|>` (deterministic), `?>` (optional), `!>` (fallible), with `else` and `catch` clauses.

```einlang
let result = data |> normalize |> transform;
let safe = data !> parse !> validate catch |e| default;
```

**Arrow combinators**: `>>>` (sequential), `***` (parallel), `&&&` (fanout), `|||` (choice) for ML graph construction.

```einlang
let model = input >>> linear(784, 128) >>> relu >>> linear(128, 10);
```

**Try expressions**: `try expr` wraps a failable expression into a Result type.
