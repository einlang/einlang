
# Recurrence order: systematic design

**Status:** Design note. When multiple dimensions have recurrence, the **order** in which we iterate them affects correctness. This doc records the problem and design options from Einlang’s perspective.

---

## 1. Problem

Some algorithms have **multi-dimensional recurrence**: the body reads the same array at indices that differ from the write on **more than one** dimension. Example: Cholesky

- We write `L[i, j]` and read `L[i, s]` and `L[j, s]` with `s < j`.
- So both dimension 0 (row) and dimension 1 (column) are recurrence dimensions (read index ≠ write index).
- **Correctness** requires a specific iteration order: column `j` must be completed before column `j+1` (column-major). Equivalently: **j** must be the **outer** loop, **i** the inner (i moves faster, j moves slower).

Today:

- **Detection** is automatic: `_recurrence_dims` in the backend marks any dimension where the body reads the LHS at a different index.
- **Order** is **implicit**: recurrence dimensions are iterated in **clause index order**. So `L[i in 0..n, j in 0..n]` → loop 0 = i, loop 1 = j → we run **i** outer, **j** inner (row-major). That is wrong for Cholesky.
- The only workaround is to **transpose the declaration** (e.g. use `L[j, i]` so j is first and becomes outermost) and transpose back at the end. That is brittle and easy to get wrong.

So we need a **systematic** way to control or infer recurrence order.

---

## 2. Current behaviour (reference)

- **Recurrence dims:** `_recurrence_dims(lowered, variable_defid, clause_indices)` returns the list of **loop indices** k where some read index differs from the write index at that dimension. See [VECTORIZATION_DESIGN.md §4](VECTORIZATION_DESIGN.md#4-recurrence-classification).
- **Execution order:** The backend builds `recurrence_loops_for_outer = [loops[d] for d in rec_dims]` and iterates in that order. So **the first recurrence dimension (first index in the bracket) is outermost**.
- **Override:** A pass can set `recurrence_dims_override` on a clause (e.g. for same-timestep deps). The backend uses that instead of inferring. There is no way for the **user** to set this in source today.

---

## 3. Design options

### 3.1 Document convention only (no language change)

- **Idea:** Document that “first index in the bracket is the outermost recurrence dimension; put the dependency/time dimension first.”
- **Pros:** No implementation; Cholesky written as `L[j, i]` + final transpose.
- **Cons:** Easy to forget; wrong order gives subtle numerical bugs; transpose is boilerplate.

### 3.2 Explicit recurrence-order annotation (source-level)

- **Idea:** Let the programmer specify which dimension(s) are recurrence and in what order, e.g.:
  - `let L[i, j] = ... recur(j, i)` or
  - `let L[i, j] = ... order(j, i)` or
  - A separate attribute on the binding.
- **Pros:** Clear intent; no need to transpose indices in the math.
- **Cons:** New syntax/semantics; must be checked (e.g. only recurrence dims, consistent with bracket).

### 3.3 “Recurrence” vs “parallel” in the bracket

- **Idea:** Mark which indices are recurrence (outer, ordered) vs parallel (inner, can be vectorized), e.g. `let L[j in 0..n recur, i in 0..n] = ...`.
- **Pros:** Order is explicit; semantics: recur dimensions are iterated in order, then parallel.
- **Cons:** New syntax; might not cover “both recurrence but j before i” when both are recurrence.

### 3.4 Infer “true” recurrence dimension from read pattern

- **Idea:** Analyze reads: e.g. for Cholesky we read `L[i,s]` (same row i, earlier cols s) and `L[j,s]` (row j, earlier cols). The **sequential** dependency is “column j depends on columns 0..j-1”, so only dimension 1 (j) is the “time” dimension; dimension 0 (i) is just “we read another row already filled in earlier columns”. So infer recurrence dim = 1 only, run j outer, vectorize over i.
- **Pros:** No syntax change; could fix Cholesky automatically.
- **Cons:** Hard to define and implement in general (e.g. 2D stencils, triangular solves); may be wrong for other algorithms.

### 3.5 Override in IR / compiler only (no new syntax)

- **Idea:** A compiler pass or IR annotation (e.g. `recurrence_dims_override = [1, 0]`) that reorders which loop is outer. Set by a pass that heuristically detects “column-major” patterns (e.g. reads at (i,s), (j,s) with s < j) and sets override so j is outer.
- **Pros:** No user-facing syntax; could fix some cases.
- **Cons:** Heuristic can be wrong; no way for user to fix when it’s wrong; magic.

---

## 4. Recommendation (open)

- **Short term:** Document current behaviour (first index = outermost recurrence) and the workaround (transpose bracket + transpose result) in the language reference or a “patterns” doc, and mention Cholesky as an example.
- **Medium term:** Consider an explicit **recurrence order** annotation (e.g. `recur(j, i)` or `order(j, i)`) so that:
  - The math can stay in natural (i, j) order.
  - Order is first-class and checked (only recurrence dims, no typos).
- **Alternative:** Invest in a **dependency analysis** that infers the “true” recurrence dimension(s) and order when possible (e.g. “all reads use s < j” ⇒ j is the recurrence dim), and fall back to bracket order or require an override when ambiguous.

---

## 5. Related

- [VECTORIZATION_DESIGN.md §4](VECTORIZATION_DESIGN.md#4-recurrence-classification) — how recurrence dims are detected.
- [VECTORIZATION_DESIGN.md §5](VECTORIZATION_DESIGN.md#5-execution-path-decision-tree) — hybrid / scalar path and `recurrence_loops_for_outer`.
- `stdlib/ml/linalg_ops.ein` — Cholesky (currently xfail; needs j-outer order).
- `src/einlang/backends/numpy_einstein.py` — `_recurrence_dims`, `recurrence_dims_override`, `recurrence_loops_for_outer`.
