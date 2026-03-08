# Design

Non-normative notes on Einlang’s design goals and influences. For the actual language rules, see the [Language reference](reference.md).

---

## Goals

- **Readable tensor math** — Code should look like the notation you’d write on a whiteboard (Einstein notation), not stringly-typed APIs or opaque DSLs.
- **Compile-time shape safety** — Shape and index errors are caught by the compiler, not at runtime. If it type-checks, the shapes are correct.
- **First-class index algebra** — Where-clauses express index relationships (e.g. conv2d `ih = oh + kh`) and guards; the compiler checks bounds.
- **First-class recurrences** — RNNs and dynamic programming are expressed as recurrence declarations; the compiler handles evaluation order.
- **One source, multiple backends** — Same Einlang source can target NumPy (today), and in the future MLIR / native / GPU, without rewriting the math.

---

## Influences

- **Syntax** — Rust-inspired (blocks, `let`, `fn`, `use`, `match`). Semicolons, no `return`, last expression is value.
- **Tensor notation** — Einstein summation and index notation as language syntax; no string indices like `einsum('ik,kj->ij', ...)`.
- **Array programming** — Similar spirit to Julia and NumPy for “write the math”; Einlang adds static shape and index checking and first-class where-clauses and recurrences.

---

## Where the rules live

All normative syntax and semantics are in the [Language reference](reference.md). This page is for context and rationale only.
