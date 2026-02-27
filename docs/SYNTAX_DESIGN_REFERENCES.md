# Syntax design references: Rust and Julia

Notes on well-designed syntax from Rust and Julia to inform Einlang. Not a spec—use as a reference when adding or refining language features.

---

## What Einlang already supports

| Area | Supported today |
|------|------------------|
| **Expression-oriented** | `if`/`match`/blocks are expressions; pipelines with `else`/`catch`. |
| **Match exhaustiveness** | `ExhaustivenessPass` checks match coverage (integer/boolean). |
| **Patterns in match** | Tuple, array (incl. `..rest`), range (`..`/`..=`), literal, binding (`@`), wildcard (`_`), or-patterns (`\|`), guards (`where`). |
| **Destructuring in let** | `let (a, b) = pair;` (tuple destructure). |
| **try / catch** | `try expr` with optional `catch` handler; expression-oriented. |
| **Modules** | `mod`/`pub`/`use`, path namespacing (`std::math::sqrt`). |
| **Comprehensions** | `[expr \| i in 0..N, guard]`; multiple variables = Cartesian product. |
| **Ranges** | `0..N` (exclusive), `0..=N` (inclusive) in expressions and patterns. |
| **Generics / specialization** | Monomorphization on argument types; stdlib generics (e.g. `sqrt`). |
| **Arrows / pipelines** | `x |> f |> g`, pipeline `else`/`catch`, arrow choice/fanout/parallel/sequential. |

---

## Top gaps (prioritized)

**Focus: inference first.** Backend will be MLIR, so **fusion is handled by the backend** (linalg, tiling, loop fusion). No need to express fusion in language syntax.

**For inference + MLIR backend, no new language features are required.** The language already has what's needed for the forward pass (Einstein, stdlib, layers, activations). The work is lowering/export to MLIR, not new syntax. The list below is **optional improvements** (ergonomics, safety), not blockers. **Inference score:** 1 = low value for inference, 5 = high.

| Item | Inference score | Why |
|------|------------------|-----|
| **Arrows / pipelines** (already supported) | **5** | High. `input \|> preprocess \|> model \|> postprocess` is the natural inference composition pattern; pipeline `else`/`catch` fit error handling. Use them; no new feature needed. |
| **Result type and `?`** | 4 | Load failures, shape errors, backend errors—typed propagation helps inference pipelines and tooling. |
| **Patterns in function parameters** | 3 | Cleaner APIs for `fn forward((batch, dim), x)`-style; not required. |
| **`if let`** | 2 | Occasional optional/config handling; full `match` is fine. |
| **Refutability** | 2 | Catches bad destructuring at compile time; inference often fixed shapes, so lower impact. |
| **Automatic differentiation** | 1 | Not used in inference; forward only. |
| **Mutation convention** | 1 | Inference is mostly pure; in-place matters more for training / buffer reuse. |

---

## Rust

### Expression-oriented design
- Almost everything is an expression; `if`/`match`/blocks produce values. Reduces boilerplate and keeps control flow and data flow aligned.
- **Relevance:** Einlang already has expression-oriented `if` and `match`. Keep new constructs (e.g. `try`, pipelines) as expressions.

### Pattern matching
- **Exhaustiveness:** Compiler checks that all cases are covered. Prevents runtime "no pattern matched" and documents intent.
- **Refutability:** Irrefutable patterns (always match) vs refutable (can fail). `let`/parameters/`for` use only irrefutable; `if let`/`match` allow refutable. Compiler enforces this.
- **Destructuring:** Same syntax for constructing and deconstructing (structs, tuples, enums). `..` for "rest of fields"; `_` for one ignored.
- **Use sites:** `let`, function/closure parameters, `match`, `if let`, `while let`, `for`.
- **Relevance:** Einlang has match exhaustiveness (pass), tuple/array/range/binding patterns. Consider: `if let pat = expr { ... }`, pattern in `for`/parameters, and consistent `..`/`_` semantics.

### Error handling: Result and `?`
- `Result<T, E>` with `Ok`/`Err`. Recoverable errors in the type system.
- `?` on expressions that return `Result`: unwraps `Ok`, early-returns `Err` (with conversion). Keeps "happy path" readable.
- **Relevance:** Einlang has `try expr` and catch. Could align with a Result-like type and a `?`-style operator for propagation.

### Module and visibility
- `mod`/`pub`/`use` with path-based namespacing (`std::math::sqrt`). Clear public API and discovery.
- **Relevance:** Einlang's `std::math::sqrt` and `pub fn`/`use` already follow this idea; keep path rules and visibility consistent.

---

## Julia

### Multiple dispatch
- Function behavior chosen from the types of **all** arguments, not just the first. Enables small, composable implementations and generic array code.
- **Relevance:** Einlang's generics and monomorphization can be guided by "specialize on all argument shapes/types" where it helps tensor code.

### Broadcasting and loop fusion
- **Dot syntax:** `f.(x)`, `x .+ y`, `x .^ 2`. Elementwise by syntax, not only by types.
- **Fusion:** Chained dot calls fuse into one loop; no temporary arrays. Guaranteed by syntax, not best-effort optimization.
- **Relevance:** With an MLIR backend, fusion is delegated to the backend (linalg, tiling, loop fusion). Einstein and comprehensions give a clear IR; no need for language-level fusion syntax.

### Comprehensions
- `[f(x) for x in range]`; multidimensional: `[f(x,y) for x in r1, y in r2]` (Cartesian product). Optional filter.
- **Relevance:** Einlang's `[expr | i in 0..N, guard]` is close; keep Cartesian product and filter semantics clear and consistent with documentation.

### Naming and mutation
- Functions that mutate arguments often end with `!` (e.g. `sort!`). Makes mutation obvious at the call site.
- **Relevance:** If Einlang adds in-place or mutating APIs, a similar convention could improve clarity.

### Arrays as a library
- Array types and operations are largely implemented in Julia; compiler and multiple dispatch make them fast. Custom array types build on `AbstractArray`.
- **Relevance:** Stdlib and backends can mirror this: core semantics in the language, rich behavior in stdlib and well-defined extension points.

---

## Summary table

| Area           | Rust                         | Julia                        | Einlang: supported → gap              |
|----------------|------------------------------|------------------------------|--------------------------------------|
| Control flow   | Expressions everywhere       | —                            | Supported: `if`/`match` as expressions |
| Match          | Exhaustive, refutability      | —                            | Exhaustive ✓; gap: refutability, `if let` |
| Errors         | `Result` + `?`               | —                            | Gap: `try`/catch only; no Result/`?`  |
| Modules        | `mod`/`pub`/paths             | —                            | Supported: `std::`, `pub fn`, `use`  |
| Arrays/tensors | —                             | Multiple dispatch, broadcast | Supported: generics, Einstein; fusion via MLIR backend |
| Comprehensions| —                             | `[f(x) for x in r]`, product  | Supported: `[expr \| i in range, guard]` |
| Mutation       | Explicit (e.g. `mut`)        | `!` suffix                   | Gap: no mut yet; no convention        |

Use this when designing new syntax or refining existing ones so Einlang stays consistent and learns from both languages.
