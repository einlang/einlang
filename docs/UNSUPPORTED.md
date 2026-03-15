
# Unsupported by design

This page lists **syntax and features that Einlang intentionally does not support**, and explains why. For what *is* supported, see the [Language reference](https://github.com/einlang/einlang/blob/main/docs/reference.md). For planned-but-not-yet-implemented features, see [Planned features](https://github.com/einlang/einlang/blob/main/docs/reference.md#planned-features) in the reference.

---

## 1. No `for` or `while` loops

**Not in the language:** C-style or Python-style `for` and `while` loops.

**Why:** Einlang wants the compiler to see *what* is being computed over (indices, ranges, recurrences) so it can check shapes and evaluation order. Loops hide that structure. So iteration is expressed in three ways the compiler understands:

- **Einstein notation** — `let out[i, j] = sum[k](A[i, k] * B[k, j]);` — the compiler infers ranges from array shapes and checks consistency.
- **Comprehensions** — `[x * 2 | x in data]` — explicit generator and optional filter; the compiler sees the iteration.
- **Recurrence declarations** — `let fib[n in 2..N] = fib[n-1] + fib[n-2]` — base cases plus recursive case; range in bracket; the compiler enforces evaluation order.

**Use instead:** [Einstein notation](https://github.com/einlang/einlang/blob/main/docs/reference.md#einstein-notation), [Array comprehensions](https://github.com/einlang/einlang/blob/main/docs/reference.md#array-comprehensions), [Recurrence relations](https://github.com/einlang/einlang/blob/main/docs/reference.md#recurrence-relations).

---

## 2. No `return` keyword

**Not in the language:** A `return` statement inside a function.

**Why:** Einlang follows Rust-style expression-oriented syntax. The value of a block or function is its last expression; no need for `return`. This keeps control flow simple and avoids “early return” patterns that can obscure the actual result.

**Use instead:** Make the last expression in the function body the value you want. If the block ends with a statement (trailing `;`), the function returns unit. See [fn declarations](reference.md#fn-declarations).

---

## 3. No string-based tensor contractions (e.g. `einsum` strings)

**Not in the language:** A function or syntax that takes a string of indices to define a contraction, e.g. `einsum('ik,kj->ij', A, B)`.

**Why:** String indices are opaque to the compiler — it cannot check shapes or infer types from them. Einlang makes indices **first-class syntax**: you write `let C[i, j] = sum[k](A[i, k] * B[k, j]);` and the compiler sees every index, infers ranges from array shapes, and reports shape mismatches at compile time. If it type-checks, the shapes are correct.

**Use instead:** [Einstein notation](reference.md#einstein-notation) with named indices. The stdlib does provide an `einsum`-style helper in `std::ml::special_ops` for compatibility, but idiomatic Einlang is direct index syntax.

---

## 4. No implicit type widening

**Not in the language:** Implicit conversion from `i32` to `i64`, or from `f32` to `f64`, when you use a narrow type where a wider type is expected.

**Why:** Implicit widening can hide precision and performance choices. Einlang allows coercion only at the binding site for **literals** (e.g. `let x: i64 = 42`). For non-literal values, you must cast explicitly: `let d: i64 = b as i64;`.

**Use instead:** Explicit [cast expressions](reference.md#cast-expressions): `x as f64`, `n as i32`, etc. See [Type compatibility](reference.md#type-compatibility).

---

## 5. No tuple indexing with `t[0]` or `t[1]`

**Not in the language:** Bracket indexing on tuples, e.g. `pair[0]` or `pair[1]`.

**Why:** Tuples and arrays are different: tuples have a fixed number of elements with possibly different types; arrays have one element type and a shape. Using `t[0]` would blur that distinction and conflict with array indexing. So tuple elements are accessed only by **field index**: `t.0`, `t.1`, `t.2`, …

**Use instead:** [Tuple access](reference.md#tuple-expressions-and-tuple-access) with `.0`, `.1`, etc. Do not use brackets on tuples.

---

## 6. No jagged arrays in Einstein notation

**Not in the language:** Using jagged (variable-length) arrays inside Einstein notation, e.g. `let out[i] = sum[j](ragged[i][j]);`.

**Why:** Einstein notation infers index ranges from array shapes. A jagged array has no single shape per dimension, so the compiler cannot define a well-defined iteration space or check consistency. Rectangular arrays have a fixed shape, so every index range is well-defined.

**Use instead:** Use [rectangular arrays](reference.md#rectangular-types) in Einstein notation. Jagged arrays are supported elsewhere (e.g. [comprehensions](reference.md#array-comprehensions), plain indexing); they just cannot appear inside `let x[i, j, ...] = ...` or `sum[k](...)` with that array. See [Restriction to rectangular types](reference.md#restriction-to-rectangular-types).

---

## 7. No colon/slice syntax (e.g. `M[:, j]` or `M[i, :]`)

**Not in the language:** Python/NumPy-style slice notation like `M[:, j]` (all rows, column `j`) or `M[i, :]` (row `i`, all columns).

**Why:** Einlang does not use a single `:` to mean “all indices along this dimension.” The language prefers explicit indices or comprehensions so shapes are visible. A “slice” is either a single index (reducing rank) or an explicit comprehension.

**Use instead:** For “row `i`”: `matrix[i]` (one index reduces rank). For “column `j`” or “all rows, column `j`”: use a comprehension `[matrix[i, j] | i in 0..n]` or an Einstein expression that explicitly iterates over the range. See [Array access](reference.md#array-access) and [Array comprehensions](reference.md#array-comprehensions).

---

## 8. No implicit broadcasting across different ranks

**Not in the language:** Automatically broadcasting a vector to a matrix (e.g. “add this 1D array to every row of a 2D array”) without writing indices.

**Why:** Implicit broadcasting across ranks can make shape behavior hard to reason about. Einlang allows broadcasting only for **same rank** (same-shape tensors) or **tensor vs scalar**. For different-rank combinations you must write the indices so the shape is explicit.

**Use instead:** Use a rectangular declaration with explicit indices, e.g. `let out[i, j] = A[i, j] + bias[j];`. See [Operators — Broadcasting](reference.md#operators).

---

## 9. Index range in `where` (all cases invalid)

**Not in the language:** Putting **any** index iteration range in the `where` clause. All of the following are invalid:

- `let fib[n] = fib[n-1] + fib[n-2] where n in 2..20` — recurrence index range in `where`
- `sum[k](x[k]) where k in 0..n` — reduction index range in `where`
- `let out[i, j] = A[i, j] where i in 0..n` — rectangular declaration index range in `where`

**Why:** The `where` clause is only for *constraints*: index algebra (e.g. `ih = oh + kh`), boolean guards (e.g. `data[i] > 0`), and variable bindings. The **iteration domain** (which indices run over which range) must always be given **in the bracket**, not in `where`. That keeps a single, clear place for “what is being iterated” and avoids ambiguous or inconsistent semantics.

**Use instead:** Put every index range in the bracket: `let fib[n in 2..20] = fib[n-1] + fib[n-2];`, `sum[k in 0..n](x[k])`, `let out[i in 0..n, j in 0..m] = A[i, j];`. Use `where` only for index equations and guards. See [Recurrence relations](reference.md#recurrence-relations), [Where clauses](reference.md#where-clauses), and [Einstein notation](reference.md#einstein-notation). The compiler reports **E0303** when an index range appears in `where`.

---

## 10. LHS index expression (e.g. `t+1`) in Einstein / recurrence

**Not in the language:** Using an expression like `t+1` as the index in the bracket of an Einstein or recurrence declaration, e.g. `let seq[t+1] = seq[t] * 2`.

**Why:** In the **declaration** bracket (`let x[...] = ...`), each index slot may only be an **identifier** (one variable name like `n`, `t`), a **literal** (e.g. `0`, `1`), or a named rest (`..name`). Expressions like `n-1` or `t+1` are not allowed there. In the **body**, any expression is allowed when indexing into an array (e.g. `fib[n-1]` is fine). So “forward” recurrence with `seq[t+1]` in the bracket is not supported.

**Use instead:** Use a **named index and range in the bracket**, and refer to the previous step in the body (e.g. `let seq[t in 0..N] = ...` with body reading `seq[t-1]` when `t > 0`, or use base cases for `0` and then `t in 1..N`). See [Recurrence relations](reference.md#recurrence-relations).

**Forward reference (reading future values):** When defining the element at index `(t, i, j)` you must not read the same array at a **future** index — one not yet computed. That rules out `h[t+1, i, j]` (future in time) and also `h[t, i+1, j]` or `h[t, i, j+1]` (future in space at the same time step). Use **backward references only** in every dimension (e.g. `h[t-1, i, j]`, `h[t, i-1, j]`, `h[t, i, j-1]`). See [Recurrence relations](reference.md#recurrence-relations).

---

## 11. No mutable bindings (no `var` or reassignment)

**Not in the language:** Reassigning a variable (e.g. `x = x + 1`) or declaring a mutable binding.

**Why:** All bindings are immutable. This simplifies reasoning about tensor expressions and avoids order-dependent bugs. Updates are expressed by producing new values (e.g. new arrays, or recurrence declarations that define a sequence).

**Use instead:** Use new `let` bindings for new values. For sequences, use [recurrence relations](reference.md#recurrence-relations). See [let declarations](reference.md#let-declarations).

---

## 12. ~~No automatic differentiation~~ — Autodiff is supported

**Supported:** The compiler supports automatic differentiation. Use `@expr` to refer to the differential of an expression and `@a / @b` to compute the derivative (numeric quotient). The compiler derives gradients via the chain rule; no hand-written gradient code required for supported ops.

**Docs and examples:** [AUTODIFF_DESIGN.md](AUTODIFF_DESIGN.md), [AUTODIFF_IMPLEMENTATION.md](AUTODIFF_IMPLEMENTATION.md), [AUTODIFF_PIPELINE.md](AUTODIFF_PIPELINE.md), [AUTODIFF_OPS.md](AUTODIFF_OPS.md). Examples: [autodiff_small.ein](https://github.com/einlang/einlang/blob/main/examples/autodiff_small.ein), [autodiff_matmul.ein](https://github.com/einlang/einlang/blob/main/examples/autodiff_matmul.ein). Language reference: [Automatic differentiation](reference.md#automatic-differentiation).

**You can still** use explicit gradients (e.g. [numerics::optim](https://github.com/einlang/einlang/blob/main/docs/stdlib.md), [optimization_suite.ein](https://github.com/einlang/einlang/blob/main/examples/optimization/optimization_suite.ein)) when you prefer or when autodiff does not yet cover an op.

---

## Summary table

| Not supported | Use instead | Reference |
|---------------|-------------|-----------|
| `for` / `while` | Einstein notation, comprehensions, recurrences | [Einstein](reference.md#einstein-notation), [Comprehensions](reference.md#array-comprehensions), [Recurrences](reference.md#recurrence-relations) |
| `return` | Last expression in block | [fn declarations](reference.md#fn-declarations) |
| String einsum | Named indices in Einstein notation | [Einstein notation](reference.md#einstein-notation) |
| Implicit widening | Explicit `as` cast | [Cast expressions](reference.md#cast-expressions) |
| `t[0]` on tuple | `t.0`, `t.1` | [Tuple access](reference.md#tuple-expressions-and-tuple-access) |
| Jagged in Einstein | Rectangular arrays only in Einstein | [Restriction to rectangular types](reference.md#restriction-to-rectangular-types) |
| `M[:, j]` slice | `matrix[i]` or comprehension with explicit range | [Array access](reference.md#array-access), [Comprehensions](reference.md#array-comprehensions) |
| Cross-rank broadcasting | Explicit indices in rectangular `let` | [Operators](reference.md#operators) |
| Index range in `where` (any) | Range in bracket: `[n in 2..20]`, `sum[k in 0..n](...)`, `[i in 0..n, j in 0..m]` | [Recurrences](reference.md#recurrence-relations), [Where clauses](reference.md#where-clauses), [Einstein](reference.md#einstein-notation) |
| LHS index expression (e.g. `t+1`) | Name or literal in bracket; refer to prior step in body (e.g. `seq[t-1]`) | [Recurrence relations](reference.md#recurrence-relations) |
| Forward ref / future value (e.g. `h[t+1,i,j]`, `h[t,i+1,j]` when defining `h[t,i,j]`) | Backward ref only in every dim (e.g. `h[t-1,i,j]`, `h[t,i-1,j]`, `h[t,i,j-1]`) | [Recurrence relations](reference.md#recurrence-relations) |
| Mutable bindings | New `let`; recurrences for sequences | [let](reference.md#let-declarations), [Recurrences](reference.md#recurrence-relations) |
| ~~Automatic differentiation~~ | **Supported:** `@expr`, `@a / @b` — see [AUTODIFF_DESIGN](AUTODIFF_DESIGN.md), [reference](reference.md#automatic-differentiation), [autodiff examples](https://github.com/einlang/einlang/tree/main/examples) |

---

**Planned features** (parsed but not yet executed) are listed in the reference under [Planned features](https://github.com/einlang/einlang/blob/main/docs/reference.md#planned-features). Those are not “unsupported by design” — they are intended for a future release.
