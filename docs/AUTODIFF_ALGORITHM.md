# Autodiff algorithm (formal specification)

**Status:** Formal specification. Defines the autodiff algorithm in math and pseudocode. Builds on [AUTODIFF_DESIGN.md](AUTODIFF_DESIGN.md), [AUTODIFF_IMPLEMENTATION.md](AUTODIFF_IMPLEMENTATION.md), [AUTODIFF_OPS.md](AUTODIFF_OPS.md), and [AUTODIFF_PIPELINE.md](AUTODIFF_PIPELINE.md).

---

## 1. Notation and semantics

### 1.1 Source language

- **Differential:** `@expr` denotes the **differential** of `expr` (same shape as `expr`). It is **symbolic**; it has no numeric value by itself.
- **Derivative quotient:** The only numeric derivative form is `@num / @den`, meaning the **partial derivative** of (the quantity that `num` refers to) with respect to (the quantity that `den` refers to). We write this as \(\frac{\partial\,\text{num}}{\partial\,\text{den}}\) or, when `num`/`den` are bindings \(z,x\), as \(\partial z / \partial x\).

### 1.2 Forward program

A **forward program** is a sequence of bindings (in dependency order):

\[
v_1 = e_1,\quad v_2 = e_2,\quad \ldots,\quad v_n = e_n
\]

where each \(e_i\) is an expression over previously defined names and inputs. We assume a **dependency graph**: \(\text{deps}(v_i)\) is the set of variables (or DefIds) that appear in \(e_i\). **Forward order** is any topological order of this graph (dependencies before dependents).

### 1.3 Differential propagation (forward mode)

For each binding \(y = f(x_1,\ldots,x_k)\), the **chain rule** gives:

\[
\mathrm{d}y = \frac{\partial f}{\partial x_1}\,\mathrm{d}x_1 + \cdots + \frac{\partial f}{\partial x_k}\,\mathrm{d}x_k
\]

We implement **forward-mode** autodiff: we introduce a **differential binding** \(\mathrm{d}y\) whose RHS is the right-hand side above, expressed in terms of \(\mathrm{d}x_1,\ldots,\mathrm{d}x_k\) and the **primal** (forward) values needed for the partials (e.g. for \(y = a \cdot b\), \(\mathrm{d}y = b\,\mathrm{d}a + a\,\mathrm{d}b\) uses primal \(a,b\)).

**Seeding:** To compute a **single partial** \(\partial z / \partial x\), we set:
- \(\mathrm{d}x = 1\),
- \(\mathrm{d}w = 0\) for every other **leaf** (variable with no dependencies in the reachable set) that is not \(x\).

Then after evaluating all \(\mathrm{d}\_\) bindings in forward order, \(\mathrm{d}z\) holds the value \(\partial z / \partial x\). So:

\[
\text{quotient}\ \frac{\partial z}{\partial x} = \mathrm{d}z\ \text{when}\ \mathrm{d}x=1\ \text{and other leaves}\ \mathrm{d}w=0.
\]

**Multiple quotients:** If the program has several quotient requests (e.g. \(\partial z/\partial x\) and \(\partial z/\partial y\)), then **one run with a single seed vector does not suffice**: seeding both \(\mathrm{d}x=1\) and \(\mathrm{d}y=1\) yields \(\mathrm{d}z = \partial z/\partial x + \partial z/\partial y\) (total differential), not the two partials. So we need either:
- **Per-quotient run:** For each quotient \(\partial\,\text{num}/\partial\,\text{den}\), run the differential block once with \(\mathrm{d}(\text{den})=1\) and all other leaves \(0\); then the value of \(\mathrm{d}(\text{num})\) is that partial. **This is the correct semantics.**
- Or: expose \(\mathrm{d}(\text{num})\) and \(\mathrm{d}(\text{den})\) as buffers and compute the quotient as \(\mathrm{d}(\text{num}) / \mathrm{d}(\text{den})\) **only when** the run was done with a single denominator seed (otherwise the ratio is not a partial).

---

## 2. Algorithm: AutodiffPass (compiler pass)

**Input:** Program IR with:
- A list of bindings (primal) in dependency order,
- Some expressions containing `DifferentialIR(@expr)` and/or derivative quotients `@num / @den` (or `DerivativeQuotientIR(num, den)`).

**Output:** Program IR where:
- Every `DifferentialIR` and every quotient is replaced by references to new **differential bindings** \(\mathrm{d}v\) (or by a single quotient value when per-quotient run is used).
- New bindings \(\mathrm{d}v\) are inserted so that, when executed with appropriate seeds, they implement the chain rule.

**Steps:**

1. **Collect differential targets and quotient pairs**
   - **Differential targets** \(T\): every DefId that appears as the operand of a `DifferentialIR` (e.g. `@z` → \(z\)) or as numerator/denominator of a quotient (e.g. `@z/@x` → \(z,x\)).
   - **Quotient pairs** \(Q\): list of \((\text{num},\text{den})\) DefIds for each quotient \(\partial\,\text{num}/\partial\,\text{den}\).

2. **Reachable set**
   - From \(T\), close under dependencies: \(\text{reachable} = \{ v \mid v \leadsto t \text{ for some } t \in T \}\) (all bindings that contribute to any target).

3. **Forward order**
   - Topological sort of reachable bindings (dependencies before dependents). Call this list \(\text{forward\_order}\).

4. **Leaves**
   - \(\text{leaves} = \{ v \in \text{reachable} \mid \text{deps}(v) = \emptyset \}\).

5. **Allocate \(\mathrm{d}\_\) bindings**
   - For each \(v \in \text{forward\_order}\): allocate a new binding \(\mathrm{d}v\) with a fresh DefId and an RHS to be filled in step 7.

6. **Seeding (current implementation — single run)**
   - For each \(v \in \text{reachable}\):
     - If \(v \in \text{quotient\_denominators}\) (i.e. \(v\) is the denominator of some quotient): \(\text{seed}[v] = 1\).
     - Else if \(v \in \text{leaves} \cap T\): \(\text{seed}[v] = 1\).
     - Else: \(\text{seed}[v] = 0\).
   - **Correct semantics (per-quotient):** For each \((\text{num},\text{den}) \in Q\), run the diff block with \(\text{seed}[\text{den}]=1\) and \(\text{seed}[w]=0\) for all other leaves \(w \ne \text{den}\); then \(\mathrm{d}(\text{num})\) gives \(\partial\,\text{num}/\partial\,\text{den}}\). So either:
     - the backend runs the diff block once per quotient with that seed, and assigns the quotient binding the value of \(\mathrm{d}(\text{num})\); or
     - the pass expands the quotient to \(\mathrm{d}(\text{num})/\mathrm{d}(\text{den})\) and the backend ensures a run with only \(\text{den}\) seeded so \(\mathrm{d}(\text{den})=1\) (then the ratio equals \(\mathrm{d}(\text{num})\)).

7. **RHS of each \(\mathrm{d}v\) (chain rule)**
   - For each \(v \in \text{forward\_order}\):
     - If \(\text{seed}[v] = 1\): \(\mathrm{d}v.\text{RHS} := 1\) (literal).
     - Else: \(\mathrm{d}v.\text{RHS} := \text{ChainRule}(e_v, v, \mathrm{d}\_)\) where \(e_v\) is the RHS of the primal binding \(v = e_v\), and \(\mathrm{d}\_\) maps each dependency \(u\) to the reference \(\mathrm{d}u\). Implement \(\text{ChainRule}\) per expression form (see §3).

8. **Expand program**
   - Replace every `DifferentialIR(operand)` by a reference to \(\mathrm{d}(\text{operand})\) (if operand is a variable) or by \(\text{ChainRule}(\text{operand}, \ldots)\) (if compound).
   - Replace every quotient \(\partial\,\text{num}/\partial\,\text{den}\) by either:
     - **Option A:** Reference to \(\mathrm{d}(\text{num})\) **only when** the runtime will run the diff block once per quotient with \(\text{den}\) seeded (then \(\mathrm{d}(\text{num})\) is the partial).  
     - **Option B:** IR node that evaluates to \(\mathrm{d}(\text{num}) / \mathrm{d}(\text{den})\) at runtime, with the runtime guaranteeing a run where \(\mathrm{d}(\text{den})=1\) for that quotient (so the ratio equals the partial).  
     - **Option C (incorrect if multiple denominators):** Reference to \(\mathrm{d}(\text{num})\) and a single run with all denominators seeded to 1 — then \(\mathrm{d}(\text{num})\) is the **total** differential, not the partial; **do not use** for correct \(\partial\,\text{num}/\partial\,\text{den}\).

9. **Program output**
   - Set \(\text{program.bindings} = \text{primal bindings interleaved with } \mathrm{d}\_ \text{ bindings}\) (each primal followed by its \(\mathrm{d}\_\) if any), and \(\text{program.statements} = \text{program.bindings}\).

10. **Analysis output**
    - Store in TyCtxt: \(\text{diff\_block}\), \(\text{differential\_targets}\), \(\text{autodiff\_differential\_map}\) (map from primal DefId to \(\mathrm{d}\_\) DefId) so the backend can run the diff block and/or fill buffers per quotient.

---

## 3. Chain rule by expression form

For a binding \(y = e\), \(\mathrm{d}y = \text{ChainRule}(e, y, \mathrm{d}\_)\). Below, \(a,b\) are expressions; \(\mathrm{d}a, \mathrm{d}b\) denote the IR for the differentials of the variables that \(a,b\) refer to (or 0 if not a single variable).

### 3.1 Binary ops

| Forward \(y\) | \(\mathrm{d}y\) |
|---------------|------------------|
| \(a + b\) | \(\mathrm{d}a + \mathrm{d}b\) |
| \(a - b\) | \(\mathrm{d}a - \mathrm{d}b\) |
| \(a \cdot b\) | \(b\,\mathrm{d}a + a\,\mathrm{d}b\) |
| \(a / b\) | \(\frac{\mathrm{d}a}{b} - \frac{a\,\mathrm{d}b}{b^2}\) |
| \(a^b\) | \(b\,a^{b-1}\,\mathrm{d}a + a^b \ln(a)\,\mathrm{d}b\) (or simplified when \(b\) is constant) |

### 3.2 Unary ops

| Forward \(y\) | \(\mathrm{d}y\) |
|---------------|------------------|
| \(-a\) | \(-\mathrm{d}a\) |
| \(+a\) | \(\mathrm{d}a\) |

### 3.3 Elementwise unary (builtins)

| Forward \(y\) | \(\mathrm{d}y\) |
|---------------|------------------|
| \(\exp(a)\) | \(\exp(a)\,\mathrm{d}a = y\,\mathrm{d}a\) |
| \(\ln(a)\) | \(\mathrm{d}a / a\) |
| \(\sqrt{a}\) | \(\mathrm{d}a / (2\sqrt{a})\) |
| \(\sin(a)\) | \(\cos(a)\,\mathrm{d}a\) |
| \(\cos(a)\) | \(-\sin(a)\,\mathrm{d}a\) |
| \(\tanh(a)\) | \((1-y^2)\,\mathrm{d}a\) |
| \(\mathrm{relu}(a)\) | \(\mathrm{d}a\) where \(a>0\), else 0 (subgradient at 0) |

(Full table: [AUTODIFF_OPS.md](AUTODIFF_OPS.md).)

### 3.4 Reductions (e.g. sum)

- \(y = \sum_i x_i\) ⇒ \(\mathrm{d}y = \sum_i \mathrm{d}x_i\) (and \(\partial y / \partial x_i = 1\), so contribution to \(\mathrm{d}x\) is broadcast of \(\mathrm{d}y\) over reduction indices).

### 3.5 Matrix multiply

- \(C = A B\) ⇒ \(\mathrm{d}C = (\mathrm{d}A)\,B + A\,(\mathrm{d}B)\). So:
  - Contribution to \(\mathrm{d}A\): \(\mathrm{d}C\,B^\top\)
  - Contribution to \(\mathrm{d}B\): \(A^\top\,\mathrm{d}C\)

(Used when differentiating Einstein lowering output; see [AUTODIFF_OPS.md](AUTODIFF_OPS.md) §3.)

### 3.6 Function calls

- \(y = f(a_1,\ldots,a_k)\) with \(f\) defined by body \(B(\mathit{params})\). Then
  \[
  \mathrm{d}y = \sum_{i=1}^{k} \frac{\partial B}{\partial p_i}\Big|_{\mathit{args}}\,\mathrm{d}a_i.
  \]
  Implement by symbolic \(\partial B/\partial p_i\) (e.g. \(\mathit{diff\_expr\_wrt}(B, p_i)\)), substitute arguments, then multiply by \(\mathrm{d}a_i\) and sum.

---

## 4. Runtime (backend) contract

### 4.1 Single run (current, incorrect for multiple quotients)

1. Run all primal bindings in order; then run all \(\mathrm{d}\_\) bindings in order with the **single** seed computed in step 6 (all quotient denominators and leaf targets set to 1).
2. For a quotient binding \(\partial\,\text{num}/\partial\,\text{den}\): if the pass expanded it to “reference \(\mathrm{d}(\text{num})\)”, then the binding’s value is \(\mathrm{d}(\text{num})\) from this run — which is the **total** differential of num when multiple denominators are seeded, **not** the partial. So this is correct only when there is at most one quotient or one denominator.

### 4.2 Correct semantics (per-quotient run)

1. Run all primal bindings once.
2. For **each** quotient \((\text{slot}, \text{num}, \text{den})\):
   - Set \(\mathrm{d}w = 0\) for all leaves \(w\); then set \(\mathrm{d}(\text{den}) = 1\).
   - Run all \(\mathrm{d}\_\) bindings in forward order (using the same RHS expressions; only seeds differ).
   - Assign \(\text{slot} := \mathrm{d}(\text{num})\) (this is \(\partial\,\text{num}/\partial\,\text{den}}\)).
3. Optionally: if the pass leaves a quotient as \(\mathrm{d}(\text{num})/\mathrm{d}(\text{den})\), the backend can instead run once per quotient with \(\mathrm{d}(\text{den})=1\), then evaluate the ratio (which equals \(\mathrm{d}(\text{num})\)).

### 4.3 Differential buffer map

Backend needs a map **primal DefId → \(\mathrm{d}\_\) DefId** so it can:
- Run the \(\mathrm{d}\_\) bindings and store results in a “differential buffer” keyed by primal DefId, and/or
- Evaluate quotient as \(\text{buffer}[\text{num}] / \text{buffer}[\text{den}]\) when both are from a run with only \(\text{den}\) seeded (so \(\text{buffer}[\text{den}]=1\) and the ratio is the partial).

The pass must set **autodiff_differential_map** in analysis to this map.

---

## 5. Summary (checklist)

| Step | Description |
|------|--------------|
| 1 | Collect differential targets \(T\) and quotient pairs \(Q\) from IR. |
| 2 | Compute reachable set from \(T\) under dependencies. |
| 3 | Topological sort → forward order. |
| 4 | Identify leaves in reachable set. |
| 5 | Allocate \(\mathrm{d}v\) for each reachable \(v\). |
| 6 | **Correct:** For each quotient, seed only that denominator; **or** run diff block once per quotient in backend. |
| 7 | Fill \(\mathrm{d}v.\text{RHS}\) with chain-rule expression (per §3). |
| 8 | Expand all `DifferentialIR` and quotients to references/IR using \(\mathrm{d}\_\). |
| 9 | Emit program with interleaved primal and \(\mathrm{d}\_\) bindings. |
| 10 | Set analysis: diff_block (optional), differential_targets, autodiff_differential_map. |

This document is the single formal reference for the autodiff algorithm; implementation in `passes/autodiff.py` and the backend should match it (including the per-quotient seeding fix for correct partials).
