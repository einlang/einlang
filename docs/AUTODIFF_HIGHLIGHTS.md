# Autodiff highlights — why Einlang’s AD is an expression-native language feature

**Purpose:** A single narrative overview of Einlang’s automatic differentiation: what makes it distinctive, how it fits the compiler, and where to read the formal specs. For algorithms and IR details, start with [AUTODIFF_DESIGN.md](AUTODIFF_DESIGN.md); for op rules see [AUTODIFF_OPS.md](AUTODIFF_OPS.md) and [AUTODIFF_EINSTEIN_OPS.md](AUTODIFF_EINSTEIN_OPS.md).

**Autodiff scope:** This is full compiler-owned, expression-first AD: Einstein-aware tensor math, quotient derivatives as syntax, readable symbolic debug output, and runtime execution on the same backend as primals.

---

## Compiler pass, not a bolt-on library

Einlang does not require a tape, tracing context, or a separate AD package. You write `@y` for the **differential** (tangent in the same space as `y`) and `@y / @x` for a **derivative** (the linear coefficient relating `d(y)` to `d(x)`) directly on program expressions and bindings. The autodiff pass in `src/einlang/passes/autodiff.py` rewrites that into ordinary IR: internal `_@name` tangent bindings run through the same dataflow as primals, and the NumPy backend executes forward values and tangents like any other Einlang program. The design in [AUTODIFF_DESIGN.md](AUTODIFF_DESIGN.md) maps one-to-one to that implementation.

## Math-first semantics

**`@y` is a differential, not a “gradient object.”** The model is \(d(y) = f'(x)\,d(x)\); the derivative is the **coefficient** of \(d(x)\) inside \(d(y)\). In practice that means Einlang differentiates the expression you already wrote, not a separately packaged `f`. For `@y / @x`, the compiler seeds \(d(x)=1\) and other leaf tangents to \(0\) and extracts that coefficient via `JacobianVisitor`, separate from forward propagation with `DiffVisitor`. That keeps symbolic tangents and numeric partials explicit instead of overloading one notion.

## Four-phase pipeline

1. **Analysis** — Collect `@` and `@y/@x` targets, build a dependency graph, topo-sort reachable bindings.  
2. **Forward differentiation** — `DiffVisitor` with `DiffContext` (DefId → \(d(\text{def})\)); insert `_@*` bindings after their primals.  
3. **Expand and emit** — Replace differential IR; `JacobianVisitor` for quotients; `DiffPrinter` for `print(@y)`.  
4. **Cleanup** — Strip `DifferentialType`, `DiffRuleIR`, and custom-rule metadata so later passes see normal IR.

Only bindings that matter for your `@` targets are touched; the pass has explicit artifacts and a clear handoff to lowering and backends. Pass placement in the **current** driver: `AutodiffPass` runs after type inference and **before** `EinsteinLoweringPass` (high-level `EinsteinIR` only). See `src/einlang/compiler/driver.py`. ([AUTODIFF_PIPELINE.md](AUTODIFF_PIPELINE.md) also discusses design alternatives.)

## What `DiffVisitor` actually covers

Forward-mode chain rule across atoms, binary ops (product, quotient, power rules), unary ops, rectangular indexing (tangents follow the same index pattern), casts, `if` (condition not differentiated), reductions (**SUM** linearity, **MAX/MIN** via argmax-style selection, **PROD** via the standard factor-wise rule), **Einstein clauses** (product rule inside sums; forward differentials stay in the **same index space** as the primal), **block bodies** with simplification and **zero-inlining** (when \(d(\text{binding})=0\), no useless `_@binding` is emitted), and **function calls as just another expression form** either by **inlining and differentiating the callee body** with \(d(\text{param}_i)=d(\text{arg}_i)\) or via **custom `@fn` rules**.

The [AUTODIFF_DESIGN.md](AUTODIFF_DESIGN.md) `reduce_mean`-style walkthrough shows how quotient cancellation and zero tangents on non-differentiable paths simplify to the expected tangent expression.

## `@fn` — custom derivative syntax, not an escape hatch

Pair a NumPy (or other) primal with an explicit rule:

```rust
fn exp(x) { python::numpy::exp(x) }
@fn exp(x) { exp(x) * @x }
```

This `@fn` syntax is a key Einlang feature. It lets you keep the primal wherever it belongs while still making the derivative part of the language and compiler pipeline. The stdlib uses the same pattern for `exp`, `sin`, `cos`, `log`, `atan2`, and other functions whose primals delegate to foreign code.

For multiple arguments, the implementation decomposes into partials with unit tangents and combines them as \(\sum_i (\partial f/\partial x_i)\,d(x_i)\), so multi-arg rules compose correctly:

```rust
fn atan2(y, x) { python::numpy::arctan2(y, x) }
@fn atan2(y, x) {
    (x / (x * x + y * y)) * @y +
    (-y / (x * x + y * y)) * @x
}
```

## Einstein and `@y / @x`

Forward differentials need no index explosion: they live in the same Einstein structure as the primal. **Jacobian** extraction for `@y/@x` uses **index expansion** where tensor-shaped partials are required — so matmul-style `sum[k](...)`, convolutions as sum-of-products with `where`, and general einsum-like forms are first-class, not “scalars only.” See [AUTODIFF_EINSTEIN.md](AUTODIFF_EINSTEIN.md) and [AUTODIFF_OPS.md](AUTODIFF_OPS.md).

## Guardrails

Autodiff is **float-only** (`f32` / `f64` and tensors thereof): differentiating integer-typed expressions is rejected as undefined, in the same spirit as systems that require smooth inputs (e.g. Julia’s ForwardDiff-style expectations).

## Doc map

| Doc | Role |
|-----|------|
| [AUTODIFF_DESIGN.md](AUTODIFF_DESIGN.md) | Canonical pass design; visitors; invariants |
| [AUTODIFF_OPS.md](AUTODIFF_OPS.md) | Per-op derivative formulas |
| [AUTODIFF_EINSTEIN_OPS.md](AUTODIFF_EINSTEIN_OPS.md) | Same in Einstein/index notation |
| [AUTODIFF_EINSTEIN.md](AUTODIFF_EINSTEIN.md) | Math spec for Einstein differentiation |
| [AUTODIFF_ALGORITHM.md](AUTODIFF_ALGORITHM.md) | Formal algorithm |
| [AUTODIFF_IMPLEMENTATION.md](AUTODIFF_IMPLEMENTATION.md) | Implementation blueprint |
| [AUTODIFF_PIPELINE.md](AUTODIFF_PIPELINE.md) | Pass interactions and alternatives |
| [AUTODIFF_EINSTEIN_OPS_IR_COMPARISON.md](AUTODIFF_EINSTEIN_OPS_IR_COMPARISON.md) | Dumped IR vs doc formulas |
| [PRINT_DIFFERENTIAL.md](PRINT_DIFFERENTIAL.md) | `print(@y)` stringification |

**Bottom line:** Einlang’s autodiff is a **documented, multi-visitor compiler pass**: forward tangents, a dedicated Jacobian path for quotients, Einstein-aware rules, optional `@fn` rules, simplification and zero-inlining, and a float-only contract. It is not a thin wrapper around an external `grad` API — the project owns the path from expression-level `@` syntax through tangent bindings to NumPy execution.
