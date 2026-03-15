# Autodiff implementation design (@y / @x)

**Status:** Implementation blueprint. Builds on [AUTODIFF_DESIGN.md](AUTODIFF_DESIGN.md) and [AUTODIFF_OPS.md](AUTODIFF_OPS.md). For a formal step-by-step algorithm, see [AUTODIFF_ALGORITHM.md](AUTODIFF_ALGORITHM.md).

This document specifies how to implement the `@expr` (differential) and `df / dx` (derivative quotient) semantics in the compiler and runtime. The presentation is **math-oriented**: differentials (d·), chain rule, and derivative quotients—not ML terms like “gradient” or “VJP.”

---

## 1. Overview

- **Syntax:** `let d_f = @f;` (differential of `f`), `let g = d_f / d_x;` (derivative ∂f/∂x when both operands are @-sourced).
- **Semantics (math-first):** **@expr** denotes the **differential** of `expr`. Differentials are **symbolic** — `@x`, `@y` do not have a numeric value by themselves. **Only the quotient @y / @x has a numeric value** (the derivative). The compiler propagates differentials (e.g. dz = dx + dy) so that quotient(s) can be computed. Each `@`-target is seeded with 1.0 in the diff block; contributions are accumulated via the chain rule. At runtime, only the numeric value(s) of the derivative quotient(s) need to be produced; differentials themselves remain symbolic.
- **Compiler:** New IR nodes → autodiff pass (builds a **diff block** from forward IR using the chain rule) → backend runs forward then diff block and produces the **numeric value(s)** of the quotient(s) (e.g. `df / dx`). Internal differential buffers may be used only to compute those quotients.

---

## 2. Grammar and tokens

**Goal:** Add prefix `@ expr` and keep `df / dx` as division with special typing when both sides are differentials.

- **Terminal:** Add `AT: "@"` in the operators section. Use `AT` in the existing binding rule so `name_pattern` stays `NAME AT pattern` (no grammar change for binding).
- **Unary expression:** Extend `unary_op` so prefix `@` is allowed:
  - `unary_op: MINUS | NOT | PLUS | AT`
  - Then `@ loss` parses as `unary_expr` with one unary operator `AT` and `primary_expr` = `identifier` (e.g. `loss`).
- **Quotient:** No new token. Use existing `DIVIDE` for `df / dx`. Semantics: when both operands have differential type, `/` denotes the partial of the numerator w.r.t. the denominator. Optionally later add a dedicated operator (e.g. `//` or a named builtin) to avoid overloading `/`.

**Files:**

- `src/einlang/frontend/grammar.lark`: Add `AT: "@"` (if not already a terminal), add `AT` to `unary_op`.

---

## 3. AST and shared types

- **UnaryOp:** Add `UnaryOp.DIFF = "@"` in `src/einlang/shared/types.py` (math: differential, not “grad”).
- **AST:** No new node. `@ expr` is a `UnaryExpression` with `operator = UnaryOp.DIFF` and `operand = expr`. Existing transformer builds `UnaryExpression(operator, operand)` from `unary_expr`; ensure `unary_op` for `AT` maps to `UnaryOp.DIFF`.
- **BinaryOp:** Keep `DIV`; no new operator. “Derivative quotient” is a type/semantic distinction when both sides are differential-typed.

**Files:**

- `src/einlang/shared/types.py`: Add `DIFF = "@"` to `UnaryOp`.
- `src/einlang/frontend/transformers/base.py`: In `unary_op`, map `AT` token to `UnaryOp.DIFF`.

---

## 4. IR nodes

- **Differential:** Represent `@expr` in IR as a dedicated node so passes can recognize and type it:
  - **`DifferentialIR(operand: ExpressionIR, location, type_info, shape_info)`**
  - Meaning: “differential of the value of `operand`” (differentials combine, e.g. dz = dx + dy). Same shape and dtype as `operand` (enforced in type/shape passes).
- **Quotient (derivative) — compile-time sugar:** Treat **@z / @x** as sugar and expand at compile time so we never materialize @z and @x at runtime. Two ways to represent the expansion:
  - **Option A (expand to dedicated node):** Replace `DIV(DifferentialIR(z), DifferentialIR(x))` (or DIV of two differential-typed operands) with **`DerivativeQuotientIR(numerator_operand, denominator_operand)`**. The backend then sees a single “compute derivative of num w.r.t. den” node: it runs the diff block once and returns the quotient (e.g. buffer[num] / buffer[den]), without evaluating two differentials.
  - **Option B (keep DIV, special-case in backend):** Keep `BinaryOpIR(DIV, left, right)` with both sides differential-typed. Backend detects this and, instead of “evaluate left, evaluate right, divide”, runs the diff block (if needed) and computes the quotient from the buffers. Same effect; expansion is implicit in backend.

Recommendation: **Option A** for a clear contract (one IR node = one derivative value; no ordering issues). Add `DerivativeQuotientIR` and a pass that rewrites DIV-of-differentials to it after type inference.

**Files:**

- `src/einlang/ir/nodes.py`: Add `DifferentialIR`. Extend `IRVisitor` (and serialization) with `visit_differential(self, node: DifferentialIR)`.
- `src/einlang/ir/serialization.py`: Serialize/deserialize `DifferentialIR`; update visitor dispatch.

---

## 5. Type and shape rules

- **Differential type:** For any type `T` (scalar or tensor), introduce `Differential(T)` (or a flag on the existing type) meaning “differential of a value of type T”. Same shape and dtype as `T`; only the semantic tag differs (“this is a differential”).
- **@expr:** Type of `@expr` is `Differential(T)` where `T` is the type of `expr`. Shape of `@expr` = shape of `expr`.
- **df / dx:** When type of `left` is `Differential(Tf)` and type of `right` is `Differential(Tx)`:
  - Semantics: derivative of (the quantity that df is the differential of) w.r.t. (the quantity that dx is the differential of). Result type can be “derivative” (e.g. scalar ratio when both scalar, or a shape agreed by convention). For same-shape differentials, result is often scalar (∂f/∂x) or same shape; document in design.
- **Non-differential `/`:** Unchanged; numeric division.

**Files:**

- `src/einlang/passes/type_inference.py`: When inferring `DifferentialIR(operand)`, set type to `Differential(inferred_type(operand))` and shape = shape(operand). For `BinaryOpIR(DIV, left, right)` when both sides are `Differential(...)`, infer quotient type (e.g. scalar or per-element ratio).
- `src/einlang/passes/shape_analysis.py`: Differential expressions: same shape as operand. Quotient of two differentials: define and implement (e.g. elementwise or scalar).

---

## 6. Name resolution and scope

- **@expr:** `expr` must resolve to a value in scope (identifier, array access, etc.). No special resolution for `@` itself; only the operand is resolved. So `@loss` and `@w` are valid when `loss` and `w` are in scope.
- **Math-first (differential semantics):** There is **no** single “ambient output” or “loss” binding. Each `@expr` where `expr` is an identifier (or otherwise resolves to a definition) is a **differential target**. The diff block seeds each such target with 1.0 and propagates differentials backward through the dataflow via the chain rule. So no pass is needed to pick “the” scalar output; the set of differential targets is exactly the set of requested `@`-operands that resolve to definitions.

**Files:**

- `src/einlang/passes/name_resolution.py`: No change for `@expr`; operand is resolved as usual.
- Autodiff pass: Collect **differential targets** from `DifferentialIR` nodes (operands that resolve to a definition); no ambient-output pass.

---

## 7. Autodiff pass (core)

**Role:** Given forward IR that contains `DifferentialIR` and possibly derivative-quotient divisions, (1) identify all **differential targets** (values w.r.t. which differentials are requested); (2) build a **diff block** that propagates differentials using the **chain rule** (so e.g. dz = dx + dy holds); (3) tie differential slots to uses of `DifferentialIR` and quotient.

**Inputs:**

- Forward IR (with `DifferentialIR(operand)` and `BinaryOpIR(DIV, df, dx)` where df, dx are differential-typed).

**Outputs:**

- Either:
  - **A:** Extended forward IR + a separate **diff block** (a block that, given forward values, fills differential buffers), or
  - **B:** A single combined program that runs forward then the diff block and exposes differential buffers as outputs.

**Steps (high level):**

1. **Collect differential targets:** Only **@ variables** are involved. For each `DifferentialIR(operand)`, the **differential target** is the definition that `operand` refers to (e.g. a binding). Record the set of targets. No ambient output; each target is seeded with 1.0 in the diff block. Variables that never appear in an `@` (e.g. `y` in `let z = x + y;` when only `@z` and `@x` are used) are not targets; they may get `d_*` buffers only as intermediates for the chain rule and are not seeded or exposed.
2. **Build diff block:** From the set of differential targets, determine the reachable subset of the forward dataflow (all bindings that contribute to any target). In **reverse** order of that dataflow:
   - Allocate a **differential buffer** for each relevant binding (e.g. `d_w` for binding `w`), same shape as the binding.
   - Initialize: `d_target = 1.0` for each differential target, and `d_* = 0` for all other buffers.
   - For each binding in reverse order: given the **differential at the output** of that binding (e.g. `d_y`), apply the **chain rule** to compute contributions to differentials at its inputs (e.g. `d_a`, `d_b`), and **accumulate** (add) those contributions into the corresponding buffers.
3. **Chain-rule emission per op:** The **chain rule (math)** is e.g. \(\mathrm{d}z = \mathrm{d}x + \mathrm{d}y\) for \(z = x + y\); the differential at the output is expressed in terms of the differentials at the inputs (\(\mathrm{d}x \to \mathrm{d}z\)). For each op we **implement** this by solving for the input contributions when we know the output differential: given `d_y` and forward values, emit IR that adds the correct contribution into `d_x`, … (e.g. for \(z = x + y\): \(\mathrm{d}z = \mathrm{d}x + \mathrm{d}y\) ⇒ we add \(\mathrm{d}z\) into both \(\mathrm{d}x\) and \(\mathrm{d}y\)). Implement for:
   - `BinaryOpIR` (ADD, SUB, MUL, DIV, POW) — see AUTODIFF_OPS §2.
   - `UnaryOpIR` (NEG, POS) and math builtins (exp, ln, sqrt, sin, cos, tanh, relu, etc.) — see AUTODIFF_OPS §1.
   - `LoweredEinsteinIR`: elementwise → elementwise; sum reduction → broadcast; matmul → matmul transpose; conv → conv_transpose / weight diff — see AUTODIFF_OPS §§3–6.
   - `LoweredRecurrenceIR`: reverse-time recurrence; at each step apply chain rule for the recurrence body — see AUTODIFF_OPS §9.
4. **Differential slots:** Only the **@ variables** (differential targets) get exposed differential buffers. Each target (e.g. a binding `w` when the user writes `@w`) gets a buffer **`d_w`** (same shape as `w`). When the user writes `let d_w = @w;`, the value of `d_w` at runtime is read from that buffer after the diff block runs. Other bindings used only in the chain rule are not exposed.
5. **Quotient df/dx:** At runtime, `df` and `dx` are already filled by the diff block. So `df / dx` is literal division of two tensors (or scalars). If shapes differ, define convention (e.g. reduce to scalar, or elementwise where broadcast applies). Document in AUTODIFF_DESIGN.

**Placement in pipeline:** After type inference and shape analysis (so differential types and shapes are known), and after Einstein lowering (so we have `LoweredEinsteinIR` / `LoweredRecurrenceIR` to attach chain-rule logic). So: after `EinsteinLoweringPass` and `RecurrenceOrderPass`; before or after `CastValidationPass` (before is fine). New pass: `AutodiffPass`; depends on `TypeInferencePass`, `UnifiedShapeAnalysisPass`, `EinsteinLoweringPass`, `RecurrenceOrderPass`.

**Files:**

- `src/einlang/passes/autodiff.py`:
  - Collect **differential targets** from `DifferentialIR` nodes (no ambient output).
  - Build **diff block** (expressions that implement the chain rule and accumulate into `d_*` buffers).
  - Attach differential buffers to the program (e.g. extra outputs or a dedicated diff block).
- `src/einlang/compiler/driver.py`: Register `AutodiffPass` after recurrence order, before validation.

---

## 8. Chain-rule (diff) emission (reference)

The single source of truth for the math is [AUTODIFF_OPS.md](AUTODIFF_OPS.md). The autodiff pass should:

- The **chain rule (math)** is e.g. \(\mathrm{d}y = \frac{\partial f}{\partial x}\,\mathrm{d}x\) (direction: \(\mathrm{d}x \to \mathrm{d}y\)). For each op, have a function that, given the **output differential** (`d_y`) and forward values, returns IR that **adds** the correct contribution into each input differential (`d_x`, …), so that the identity holds. E.g. \(\mathrm{d}(a+b) = \mathrm{d}a + \mathrm{d}b\) ⇒ we add \(\mathrm{d}y\) into both \(\mathrm{d}a\) and \(\mathrm{d}b\).
- Forward values needed in the diff block (e.g. `y` for exp, `p` for softmax) must be either recomputed in the diff block or stored during forward. Prefer storing in a “forward cache” (e.g. a struct or tuple of intermediate values) passed into the diff block, to avoid recomputation and to handle non-invertible ops.

**Stored forward values (examples):**

- Elementwise unary (exp, tanh, etc.): output `y` (and input `x` where needed).
- Binary (mul, div): both operands.
- Sum reduction: no extra storage (differential is broadcast).
- Matmul: inputs A, B and output C (or recompute from A, B).
- Conv: input, weight, bias, output (for weight and bias differentials).
- Recurrence: full history of states if needed, or recompute from initial + recurrence (expensive).

---

## 9. Backend contract

- **Forward:** Unchanged; backend runs the forward IR and produces outputs (including any tensors used later in the diff block).
- **Symbolic vs numeric:** `@x` and `@y` are **symbolic** (no numeric value by themselves). **Only the quotient `@y / @x` has a numeric value** (the derivative). The backend must produce that numeric value; it may do so by running the diff block to fill internal differential buffers and then computing the quotient(s), or by fusing so that only the quotient(s) are materialized.
- **Diff block:** Backend must run the generated **diff block** (or equivalent) so that derivative quotients can be computed. Inputs: (1) forward outputs (or the subset needed for the chain rule), (2) **seed** for each differential target (1.0). The diff block produces contributions to differentials; the backend uses these to evaluate **quotient(s)** (e.g. `df / dx`), which are the only values that need a numeric result.
- **DifferentialIR:** A use of `@w` (e.g. `let d_w = @w;`) is symbolic; the “value” of `d_w` is only meaningful when used in a quotient. When the backend evaluates a **quotient** `df / dx`, it needs the numeric numerator and denominator (or an equivalent fused computation); those come from the diff block’s outputs for the corresponding differential targets.
- **Quotient:** The expression `df / dx` (when both are differential-typed) has a **numeric value** (the derivative). The backend computes it from the diff block’s result (e.g. literal division of the two differential buffers, or a single fused derivative value), with shape/broadcast rules as defined.

**Files:**

- `src/einlang/backends/numpy_core.py` (and any other backends): Add handling for “differential slot” reads and for running the diff block (or a combined forward+diff function). No change to existing forward-only execution until autodiff is enabled.

---

## 10. Validation

- **Differentiability:** Where `@expr` or a derivative quotient is used, the program must be differentiable w.r.t. the relevant variables at the point of use (no kinks/jumps unless a subgradient is defined, e.g. ReLU at 0). Optionally: a pass that checks that only differentiable ops appear on the diff path.
- **Shape:** Every differential has the same shape as its target; check in type/shape passes.
- **Differentials:** Every `DifferentialIR` refers to a value in scope; differentials combine per the math (e.g. dz = dx + dy).

---

## 11. Implementation checklist (summary)

| Step | Item | Files |
|------|------|--------|
| 1 | Add `AT` terminal and `unary_op` including `AT` | `grammar.lark` |
| 2 | Add `UnaryOp.DIFF` | `shared/types.py` |
| 3 | Map `AT` to `UnaryOp.DIFF` in transformer | `frontend/transformers/base.py` |
| 4 | Add `DifferentialIR` and visitor support | `ir/nodes.py`, `ir/serialization.py` |
| 5 | Lower `UnaryExpression(DIFF, operand)` to `DifferentialIR` | `passes/ast_to_ir.py` |
| 6 | Type/shape for `DifferentialIR`: same as operand, type = Differential(T) | `passes/type_inference.py`, `passes/shape_analysis.py` |
| 7 | Derivative quotient: type DIV when both sides Differential(T) | `passes/type_inference.py` |
| 8 | (Optional) If supporting “gradient of L” variant: ambient output pass | Otherwise not needed for differential semantics |
| 9 | Autodiff pass: collect differential targets, build diff block, chain-rule per op | `passes/autodiff.py` |
| 10 | Register AutodiffPass in driver after EinsteinLowering + RecurrenceOrder | `compiler/driver.py` |
| 11 | Backend: run diff block, expose differential slots for DifferentialIR bindings | `backends/numpy_core.py` (or dedicated backend) |

---

## 12. Open design choices (to resolve in implementation)

- **Quotient operator:** Keep overloaded `/` for derivative quotient vs. introduce a dedicated operator or builtin.
- **(Not used for math-first differentials)** Ambient output rule only if we add a “gradient of L” variant.
- **Recurrence diff:** Store full state history vs. recompute (memory vs. compute).
- **Non-differentiable ops:** Reject in autodiff scope vs. allow with subgradient (e.g. ReLU at 0) and document.

This implementation design is intended to be followed step-by-step; each step can be tested (grammar/parse, then IR, then types, then autodiff pass, then backend) for incremental delivery.

---

## 13. File touch points (quick reference)

| Area | Path |
|------|------|
| Grammar / tokens | `src/einlang/frontend/grammar.lark` |
| UnaryOp enum | `src/einlang/shared/types.py` |
| AST transformer (unary) | `src/einlang/frontend/transformers/base.py` |
| IR nodes + visitor | `src/einlang/ir/nodes.py` |
| IR serialization | `src/einlang/ir/serialization.py` |
| AST → IR lowering | `src/einlang/passes/ast_to_ir.py` |
| Type inference | `src/einlang/passes/type_inference.py` |
| Shape analysis | `src/einlang/passes/shape_analysis.py` |
| **Autodiff pass** | `src/einlang/passes/autodiff.py` |
| Pass registration | `src/einlang/compiler/driver.py` |
| NumPy backend | `src/einlang/backends/numpy_core.py` |
| Design / ops reference | `docs/AUTODIFF_DESIGN.md`, `docs/AUTODIFF_OPS.md` |
