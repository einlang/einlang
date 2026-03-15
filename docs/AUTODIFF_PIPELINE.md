# Autodiff pass: pipeline placement and pass interactions

**Status:** Design. Defines where the autodiff pass sits, what it assumes, what it guarantees, and how it interacts with other passes. No ad-hoc workarounds; every contract is explicit.

**References:** [AUTODIFF_DESIGN.md](AUTODIFF_DESIGN.md), [AUTODIFF_IMPLEMENTATION.md](AUTODIFF_IMPLEMENTATION.md), [AUTODIFF_OPS.md](AUTODIFF_OPS.md), [AUTODIFF_ALGORITHM.md](AUTODIFF_ALGORITHM.md).

---

## 1. Pipeline position

**Order (relevant segment):**

```
… → TypeInferencePass → DerivativeQuotientRewritePass → EinsteinLoweringPass → RecurrenceOrderPass → AutodiffPass → CastValidationPass → … → IRValidationPass → (backend)
```

**Placement rationale:**

| After | Why |
|-------|-----|
| **TypeInferencePass** | Every expression has `type_info`. Autodiff uses it for the types of new d_* bindings and for well-typed IR. Differential types and quotient types are already inferred. |
| **UnifiedShapeAnalysisPass** | Shapes are known; needed if we ever emit shape-dependent d_* (e.g. reductions). Keeps autodiff shape-agnostic where possible. |
| **DerivativeQuotientRewritePass** | `@num / @den` is already rewritten to `DerivativeQuotientIR(numerator, denominator)`. Autodiff only sees this node and does not special-case DIV of differentials. |
| **EinsteinLoweringPass** | Einstein notation is lowered to `LoweredEinsteinIR` (and related). Autodiff implements the chain rule for this IR; it does not handle high-level EinsteinIR. |
| **RecurrenceOrderPass** | Recurrences are ordered and lowered. Autodiff can assume a fixed execution order for recurrence bodies. |

**Before:**

| Before | Why |
|--------|-----|
| **CastValidationPass** | Validation passes assume IR is structurally and typely valid. Autodiff must output well-typed IR so these checks pass. |
| **IRValidationPass** | Same: no `DifferentialIR` or `DerivativeQuotientIR` left; all new nodes have `type_info`. |

So autodiff runs **after** lowering and analysis and **before** all validation.

---

## 2. Contracts

### 2.1 Preconditions (what AutodiffPass assumes)

The following are **guarantees** that upstream passes must provide. Autodiff does **not** patch or infer missing data; if something is missing, that is a bug in an earlier pass (or in autodiff’s own output).

| Guarantee | Provided by | Used by Autodiff |
|-----------|-------------|------------------|
| Every **ExpressionIR** in the program has `type_info` set (non-None). | TypeInferencePass (and any pass that creates new expressions must set it). | Not used to “fill” existing nodes; used as the **type for every new node** autodiff creates (d_* RHS and d_* bindings). |
| Every binding has `defid` and (for non-function bindings) `expr`. | ASTToIRLoweringPass, NameResolutionPass. | Building `binding_by_defid`, dependency graph, and new bindings. |
| `DifferentialIR` and `DerivativeQuotientIR` appear only as designed. | AST lowering (DifferentialIR), DerivativeQuotientRewritePass (DerivativeQuotientIR). | Collecting targets and quotient pairs; expanding to d_* refs. |
| Einstein code is lowered to `LoweredEinsteinIR` / `LoweredRecurrenceIR`. | EinsteinLoweringPass, RecurrenceOrderPass. | Chain rule for einsum-style and recurrence. |
| `tcx.resolver` is available for allocating new DefIds for d_* bindings. | Driver (name resolution / lowering). | Allocating `d_*` names and DefIds. |

If any precondition is violated, autodiff may fail or produce invalid IR; validation (e.g. IRValidationPass) will then report errors. Autodiff does **not** “fix” missing `type_info` on nodes it did not create.

### 2.2 Postconditions (what AutodiffPass guarantees)

| Guarantee | For whom |
|-----------|----------|
| **No `DifferentialIR` or `DerivativeQuotientIR`** remain in the program. | All later passes and the backend; they see only plain IR. |
| **`program.bindings` and `program.statements`** are the same list and include both primal and d_* bindings in execution order (each primal followed by its d_* if any). | Backend (execution order); any pass that iterates `program.statements`. |
| Every **new** node created by autodiff (LiteralIR, BinaryOpIR, IdentifierIR, BindingIR for d_*) has **`type_info`** (and `shape_info` where the node type has it) set from the corresponding primal binding’s type. | IRValidationPass (no “missing type_info” on autodiff-introduced nodes). |
| **Analysis dict** is set: `diff_block`, `differential_targets`, `differential_buffer_by_defid`. | Backend or other passes that need to know what was differentiated. |

Autodiff does **not** modify or “fill” `type_info` on existing nodes (e.g. literals from the original program). Those must already be typed by TypeInferencePass.

---

## 3. Interaction with each pass

### 3.1 Upstream

| Pass | Interaction |
|------|-------------|
| **TypeInferencePass** | Ensures all expressions are typed. Autodiff uses binding `type_info` as the type for the d_* binding and for the entire d_* RHS tree it builds. Autodiff does not infer types; it copies from the primal. |
| **UnifiedShapeAnalysisPass** | Provides shape information. Autodiff can rely on it for any shape-using logic (e.g. future recurrence or reduction d_*). |
| **DerivativeQuotientRewritePass** | Optional. Replaces `DIV(@num, @den)` with `DerivativeQuotientIR(num, den)`. Autodiff accepts **either** DerivativeQuotientIR **or** `BinaryOpIR(DIV, @num, @den)`: it collects (num_defid, den_defid) from both and replaces the quotient by a reference to the d_num binding (when seed d_den=1). |
| **EinsteinLoweringPass** | Produces `LoweredEinsteinIR` (and related). Autodiff implements forward diff for these (e.g. sum-of-products → contributions to d_inputs). |
| **RecurrenceOrderPass** | Orders recurrence execution. Autodiff assumes this order when handling recurrences (when implemented). |

### 3.2 Downstream

| Pass | Interaction |
|------|-------------|
| **CastValidationPass** | Sees only plain IR; no special handling for @ or quotient. |
| **PipelineTypeValidationPass** | Same. |
| **ExhaustivenessPass** | Same. |
| **IRValidationPass** | Expects every ExpressionIR to have `type_info`. Autodiff guarantees this only for **nodes it creates**; existing nodes must already satisfy this (TypeInferencePass). |
| **Backend** | Runs `program.statements` in order. Autodiff ensures d_* bindings are in that list so that e.g. `db_da = d_b` reads the correct d_b. |

### 3.3 Dataflow summary

```
Source
  → AST → Name resolution → ASTToIR (DifferentialIR, BinaryOpIR(DIV,…))
  → TypeInference (type_info on all expressions)
  → DerivativeQuotientRewrite (DIV(diff,diff) → DerivativeQuotientIR)
  → EinsteinLowering (LoweredEinsteinIR, …)
  → RecurrenceOrder
  → AutodiffPass
       • Reads: bindings, binding.expr (DifferentialIR, DerivativeQuotientIR, LoweredEinsteinIR, …), type_info
       • Writes: new d_* bindings (with typed RHS), program.bindings, program.statements
       • Removes: DifferentialIR, DerivativeQuotientIR (replaced by IdentifierIR to d_*)
  → Validation passes
  → Backend (execute statements)
```

---

## 4. Pass dependencies (requires)

AutodiffPass should declare:

```python
requires = [
    TypeInferencePass,
    UnifiedShapeAnalysisPass,
    EinsteinLoweringPass,
    RecurrenceOrderPass,
]
```

- **DerivativeQuotientRewritePass** is **not** required: autodiff collects quotient pairs from both `DerivativeQuotientIR` and `BinaryOpIR(DIV, @num, @den)` and expands either to a reference to d_num (when d_den=1).
- **TypeInferencePass** and **UnifiedShapeAnalysisPass** so that types and shapes are available.
- **EinsteinLoweringPass** and **RecurrenceOrderPass** so that the IR is in the form for which the chain rule is implemented.

The driver registers passes in a fixed order; the pass manager runs them in topological order consistent with `requires`, so AutodiffPass always runs after these four.

---

## 5. What Autodiff does not do (no workarounds)

- **Does not** set `type_info` (or `shape_info`) on nodes it did not create. If an existing node is missing `type_info`, that is a bug in an earlier pass; validation will fail and the fix belongs there.
- **Does not** run or emulate type inference. It uses the primal binding’s `type_info` (and `shape_info`) for every new node it creates.
- **Does** recognize “DIV of two differentials” in the IR: it collects quotient pairs from both DerivativeQuotientIR and BinaryOpIR(DIV, @num, @den) and expands them to a reference to d_num (so DerivativeQuotientRewritePass is optional).
- **Does not** depend on a separate “diff block” execution in the backend; it embeds d_* bindings in the main program and updates `program.statements` so a single forward execution computes both primal and differentials.

---

## 6. Implementation checklist (pipeline and contracts)

| Item | Action |
|------|--------|
| AutodiffPass.requires | TypeInference, ShapeAnalysis, EinsteinLowering, RecurrenceOrder (no DerivativeQuotientRewritePass; autodiff accepts quotient from DIV or DerivativeQuotientIR). |
| New nodes only | Set `type_info` (and `shape_info` where applicable) on every **new** expression and binding created by autodiff (d_* RHS and d_* bindings), using the primal binding’s type. |
| No program-wide type fill | Do not iterate over all bindings to “fill” missing `type_info` on existing nodes. |
| program.statements | Keep in sync with `program.bindings` whenever bindings are replaced (same list, same order). |
| TypeInferencePass | (Upstream.) Ensure every ExpressionIR reached during type inference gets `type_info` set (including literals in all bindings). |

This keeps the design clear and avoids ad-hoc fixes inside the autodiff pass.

---

## 7. Alternative: Autodiff right after TypeInference, remove DerivativeQuotientRewritePass

**Idea:** Run AutodiffPass immediately after TypeInferencePass and delete DerivativeQuotientRewritePass. One fewer pass; all @ and quotient handling in one place.

**Pipeline would be:**

```
… → TypeInferencePass → AutodiffPass → EinsteinLoweringPass → RecurrenceOrderPass → …
```

### 7.1 What changes

| Aspect | Current | Alternative |
|--------|--------|-------------|
| Quotient | Separate pass rewrites `DIV(@num, @den)` → `DerivativeQuotientIR`. Autodiff consumes DerivativeQuotientIR. | No DerivativeQuotientIR. Autodiff must **recognize quotient** itself: e.g. `BinaryOpIR(DIV, left, right)` where both operands are differential-typed (or `DifferentialIR` or identifier bound to a differential). Autodiff collects (num, den) from such DIV nodes and expands them the same way (seed d_den=1, quotient = d_num). Straightforward. |
| Einstein | Autodiff runs **after** lowering; it sees `LoweredEinsteinIR` (concrete loops, sum-of-products). Chain rule is implemented for this form. | Autodiff runs **before** lowering; it sees **high-level `EinsteinIR`** (declarative: indices, value expr, clauses). No `LoweredEinsteinIR` yet. |
| Recurrence | Autodiff sees `LoweredRecurrenceIR` (if any). | Recurrence not yet lowered; autodiff would see high-level recurrence or none. |

### 7.2 Pros

- **Simpler pipeline:** One pass fewer; quotient logic lives only in autodiff.
- **Single place for @ semantics:** Everything about differentials and derivatives is in AutodiffPass.
- **Quotient is easy:** In autodiff, detect `BinaryOpIR(DIV, left, right)` with both sides differential-typed (or `DifferentialIR` / diff-bound identifier), collect (num_defid, den_defid), and expand as today (reference to d_num when d_den=1).

### 7.3 High-level EinsteinIR may be easier to differentiate

Differentiating **EinsteinIR** (high-level) can be simpler than differentiating **LoweredEinsteinIR** (low-level):

| Aspect | High-level EinsteinIR | LoweredEinsteinIR |
|--------|------------------------|-------------------|
| **Structure** | Declarative: each clause has `indices` and a **value expression** (e.g. `A[i,k]*B[k,j]`). | Concrete: loops, `LoweredReductionIR`, product factors as `RectangularAccessIR` lists, loop structures. |
| **Differentiation** | Value is an expression tree (BinaryOp, RectangularAccess, …). We already have symbolic diff for expression trees (`_diff_expr_wrt`). Differentiate the value w.r.t. each array that appears (by DefId); the result is again an expression in the same index space. Then: derivative of a sum is a sum of derivatives; contribution to `d_array` is “sensitivity × partial” summed over the right indices—again expressible as Einstein (indices + expression). | Must manually identify factors, build adjoint loop structures, match index lists to loops, and construct new `LoweredEinsteinClauseIR` / `LoweredReductionIR`. Easy to get indices or reduction axes wrong. |
| **Output** | Emit new **EinsteinIR** (or clauses) for each `d_array`: e.g. `dA[i,k] += sum_j dC[i,j] * B[k,j]`. Lowering then turns that into loops as usual. | Emit new **LoweredEinsteinIR** by hand (same complexity as the lowering pass itself). |

So at the high level we only need:

1. **Symbolic diff of the clause value** w.r.t. each array that appears (extend `_diff_expr_wrt` to treat `RectangularAccessIR(array, indices)` as “variable” keyed by array DefId; partial is the coefficient of that access in the value expression).
2. **Rule for reductions:** e.g. “output = sum_k value(k)” ⇒ contribution to d_input is sum over the same reduction indices of (d_output × partial). That contribution is itself an indexed expression, so we can emit it as one or more high-level Einstein clauses.
3. **No loop construction:** we never build `LoopStructure`, `LoweredReductionIR`, or index-to-loop maps; we emit Einstein and let the existing lowering pass do that.

**Conclusion:** High-level EinsteinIR is likely **easier** to diff than LoweredEinsteinIR: reuse expression-level symbolic diff and emit high-level Einstein for gradients; lowering stays the single place that knows about loops and reduction structure.

### 7.4 Remaining considerations (shape, recurrence)

- **Shape / range:** For high-level Einstein, we may still need shape/range for type-checking the new d_* bindings or for where-clauses. TypeInference and ShapeAnalysis already run before the proposed autodiff position, so basic types and shapes are available; any extra needs can be addressed there.
- **Recurrence:** High-level recurrence (if present before lowering) would need a separate rule. That could be “differentiate the recurrence body symbolically and emit a high-level recurrence for d_*” or “don’t support recurrence diff when autodiff runs early”; the former is consistent with the “differentiate at high level” idea.

### 7.5 Recommendation (revised)

- **Removing DerivativeQuotientRewritePass** and doing quotient recognition inside AutodiffPass is a clear win: one fewer pass, same behaviour.
- **Moving Autodiff to right after TypeInference** is **reasonable and possibly simpler** for Einstein:
  - Quotient: handle `DIV(diff, diff)` in autodiff (see §7.1).
  - Einstein: implement the chain rule on **EinsteinIR** (symbolic diff of clause value + reduction rule, emit Einstein for d_*); no need to touch LoweredEinsteinIR. Lowering then runs as usual on the expanded program.
- So the recommended pipeline is:

  ```
  … → TypeInferencePass → AutodiffPass → EinsteinLoweringPass → RecurrenceOrderPass → …
  ```

  with **DerivativeQuotientRewritePass removed**, and autodiff implementing:
  - quotient detection and expansion (as today),
  - forward diff for BinaryOp, UnaryOp, FunctionCall, **EinsteinIR** (and optionally high-level recurrence),
  and **not** implementing the chain rule for LoweredEinsteinIR (that path can be removed or kept for a “late” autodiff mode if ever needed).

---

## 8. print(@x) — can it work?

**Semantics:** `@x` is **symbolic**: it denotes the **differential of x** (dx), i.e. the tangent/increment. It is **not** dx/dx (that would be 1). The *value* of dx at runtime is whatever is stored in the d_x binding (0, 1, or propagated by the chain rule).

**Yes, `print(@x)` can work.** After the autodiff pass, every `@x` is rewritten to a reference to the **d_x** binding. So `print(@x)` becomes "print the value of d_x" (i.e. print dx). For that to be a meaningful number, **d_x** must be computed: e.g. **x** is a quotient denominator (we seed d_x = 1) or **x** is a differential target and we seed leaves with 1. The backend runs `program.statements` in order; after that, `print(d_x)` uses the current value of d_x. So **`print(@x)` works** whenever the pipeline computes d_x. If there is no quotient and no seed for x, d_x may be 0.

**Examples (semantics):**

| Source | Meaning |
|--------|--------|
| `print(@x);` | Print the value of **dx** (the d_x binding). |
| `@x` | The differential of x (dx). |
| `print(@(2*x));` | Print the value of **d(2*x)** = 2·dx. |
| `2*@x` | 2 times the differential of x (2·dx). So **@(2*x)** and **2*@x** are the same (differential of a linear expression). |
| `print(@(2*x + y*y));` | Print **d(2*x + y²)** = 2·dx + 2·y·dy. |
| `2*@x + 2*y*@y` | Same as above. The **y** in the coefficient 2*y is the **primal** (runtime) value of y; only the differentials @x, @y are symbolic. So the expanded form substitutes primal values where needed. |
| `let y = @(x**2)/@x;` | The **derivative** d(x²)/dx = **2·x**. The **x** in 2*x is the **primal** (runtime) value of x. So after expansion and execution, `y` holds 2*x (evaluated at the current x). |
