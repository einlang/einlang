# Syntax comparison

**If you already think in NumPy, Julia, or Rust,** this page maps your mental model to Einlang. Same ideas, different syntax — so you can guess the rest. Full details: [Language reference](reference.md).

---

## Python / NumPy

**Intuition:** You’re used to `einsum` or `@` for tensor ops. In Einlang you write the same indices as real names (no strings), and the compiler checks shapes. You still run everything from Python with one call: `run(file="...")` or `run(source="...")`.

| You usually… | Python / NumPy | In Einlang | More in reference |
|--------------|-----------------|------------|-------------------|
| Matrix multiply | `np.einsum('ik,kj->ij', A, B)` or `A @ B` | `let C[i, j] = sum[k](A[i, k] * B[k, j]);` | [Einstein notation](reference.md#einstein-notation) |
| Sum over an axis | `np.sum(x, axis=1)` | `let s[i] = sum[j](x[i, j]);` | [Reductions](reference.md#einstein-notation) |
| Element-wise ops | `A * B`, `A + 1` | `let out[i, j] = A[i, j] * B[i, j];` or same-shape `A + B` | [Rectangular declarations](reference.md#einstein-notation), [Operators](reference.md#operators) |
| Run your code | — | `from einlang import run; run(file="...")` or `run(source="...")` | [Install & run](../../README.md#install--run) |

Einlang runs **inside** your Python process; you pass a file path or a source string. The language looks Rust-like, not Python — but you only need to write the `.ein` side; calling it is one line.

---

## Julia

**Intuition:** Same array-first, “write the math” feel. Einlang uses Rust-style syntax (`let`, `fn`, `;`) and has **no** `for`/`while` — you use comprehensions, Einstein notation, or recurrences. Ranges: `1..10` is end-**exclusive** (like Rust); use `0..=10` for inclusive.

| You usually… | Julia | In Einlang | More in reference |
|--------------|--------|------------|-------------------|
| Matrix multiply | `A * B` or loops | `let C[i, j] = sum[k](A[i, k] * B[k, j]);` | [Einstein notation](reference.md#einstein-notation) |
| Comprehensions | `[x^2 for x in 1:10]` | `[i * i \| i in 1..10]` | [Array comprehensions](reference.md#array-comprehensions) |
| Ranges | `1:10` (inclusive) | `1..10` (exclusive), `0..=10` (inclusive) | [Ranges](reference.md#ranges) |
| Functions | `function f(x) ... end` | `fn f(x) { ... }` (last expression is the return value) | [fn declarations](reference.md#fn-declarations) |
| Index algebra (e.g. conv) | Manual `ih = oh + kh` in loops | `where ih = oh + kh, iw = ow + kw` on the expression | [Where clauses](reference.md#where-clauses) |
| Recurrence (e.g. RNN) | Loops or recursion | `let h[t in 0..T, i in 0..H] = ...` with body reading prior step (e.g. `h[t-1, i]`). LHS index must be a name or literal, not `t+1`. | [Recurrence relations](reference.md#recurrence-relations) |

---

## Rust

**Intuition:** You’ll feel at home: `let`, `fn`, blocks, `match`, semicolons. The main difference: **no `for` or `while`**. Loops become comprehensions `[x \| i in 0..n]`, Einstein `let out[i] = ...`, or recurrence declarations. Types are inferred by default.

| You usually… | Rust | In Einlang | More in reference |
|--------------|------|------------|-------------------|
| Bindings | `let x = 42;` | `let x = 42;` | [let declarations](reference.md#let-declarations) |
| Functions | `fn f(x: i32) -> i32 { ... }` | `fn f(x) { ... }` or add types: `fn f(x: i32) -> i32 { ... }` | [fn declarations](reference.md#fn-declarations) |
| Blocks | `{ stmt; expr }` | `{ stmt; expr }` (last expr is the value) | [Block expressions](reference.md#block-expressions) |
| Match | `match x { 0 => ..., _ => ... }` | `match x { 0 => ..., _ => ... }` | [match expressions](reference.md#match-expressions) |
| Arrays | `let a: [i32; 3] = [1, 2, 3];` | `let a = [1, 2, 3];` (inferred) or add type if you want | [Types](reference.md#types), [Literals](reference.md#literals) |
| Loops | `for i in 0..n { ... }` | `[expr \| i in 0..n]` or `let out[i] = ...` or recurrence | [Comprehensions](reference.md#array-comprehensions), [Einstein](reference.md#einstein-notation), [Recurrences](reference.md#recurrence-relations) |

---

## Where to go next

- **Full syntax and semantics:** [Language reference](reference.md)
- **Built-in functions:** [Standard library](stdlib.md)
- **Run from Python:** [Install & run](../../README.md#install--run) in the main README
