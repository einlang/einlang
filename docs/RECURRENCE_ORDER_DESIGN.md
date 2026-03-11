
# Recurrence order: systematic design

**Status:** Design note. When multiple dimensions have recurrence, the **order** in which we iterate them can affect correctness (for some algorithms) or consistency/performance (for others). This doc records the problem and design options from Einlang’s perspective.

---

## 1. Problem

Some algorithms have **multi-dimensional recurrence**: the body reads the same array at indices that differ from the write on **more than one** dimension. Example: Cholesky

- We write `L[i, j]` and read `L[i, s]` and `L[j, s]` with `s < j`.
- So both dimension 0 (row) and dimension 1 (column) are recurrence dimensions (read index ≠ write index).
- For **Cholesky specifically**, both row-major (i outer, j inner) and column-major (j outer, i inner) are **correct**: the dependency DAG allows either order. For **other** algorithms (e.g. some triangular solves or time-like recurrences), only one order may be valid.

Today:

- **Detection** is automatic: `_recurrence_dims` in the backend marks any dimension where the body reads the LHS at a different index.
- **Order** is **implicit**: recurrence dimensions are iterated in **clause index order**. So `L[i in 0..n, j in 0..n]` → loop 0 = i, loop 1 = j → we run **i** outer, **j** inner (row-major). For Cholesky that is valid; we may still want a **predictable** or **preferred** order (e.g. column-major for consistency or cache behaviour) instead of “whatever the bracket says.”
- Without an override, the only way to get a different order is to **transpose the declaration** (e.g. use `L[j, i]`) and transpose back at the end. That is brittle and easy to get wrong.

So we need a **systematic** way to control or infer recurrence order (for correctness when only one order works, and for consistency/performance when several orders work).

---

## 2. Current behaviour (reference)

- **Recurrence dims:** `_recurrence_dims(lowered, variable_defid, clause_indices)` returns the list of **loop indices** k where some read index differs from the write index at that dimension. See [VECTORIZATION_DESIGN.md §4](VECTORIZATION_DESIGN.md#4-recurrence-classification).
- **Execution order:** The backend builds `recurrence_loops_for_outer = [loops[d] for d in rec_dims]` and iterates in that order. So **the first recurrence dimension (first index in the bracket) is outermost**.
- **Override:** A pass can set `recurrence_dims_override` on a clause (e.g. for same-timestep deps). The backend uses that instead of inferring. There is no way for the **user** to set this in source today.

---

## 3. Design options

### 3.1 Document convention only (no language change)

- **Idea:** Document that “first index in the bracket is the outermost recurrence dimension; put the dependency/time dimension first.”
- **Pros:** No implementation; Cholesky can be written as `L[j, i]` + final transpose if a specific order is desired.
- **Cons:** Easy to forget; for algorithms where only one order is correct, wrong order gives subtle bugs; transpose is boilerplate.

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

- **Idea:** Analyze reads: e.g. for Cholesky we read `L[i,s]` (same row i, earlier cols s) and `L[j,s]` (row j, earlier cols). One can treat dimension 1 (j) as the “time” dimension and run j outer, vectorize over i. (For Cholesky both j-outer and i-outer are correct; inference would pick one.)
- **Pros:** No syntax change; gives a consistent order for Cholesky-style clauses.
- **Cons:** Hard to define and implement in general (e.g. 2D stencils, triangular solves); may be wrong for other algorithms.

### 3.5 Override in IR / compiler only (no new syntax)

- **Idea:** A compiler pass or IR annotation (e.g. `recurrence_dims_override = [1, 0]`) that reorders which loop is outer. Set by a pass that heuristically detects “column-major” patterns (e.g. reads at (i,s), (j,s) with s < j) and sets override so j is outer.
- **Pros:** No user-facing syntax; could fix some cases.
- **Cons:** Heuristic can be wrong; no way for user to fix when it’s wrong; magic.

---

## 4. Recommendation (open)

- **Short term:** Document current behaviour (first index = outermost recurrence). When a specific order is desired, the workaround is transpose bracket + transpose result; mention Cholesky as an example (both orders are correct for Cholesky; override or convention picks one).
- **Medium term:** Consider an explicit **recurrence order** annotation (e.g. `recur(j, i)` or `order(j, i)`) so that:
  - The math can stay in natural (i, j) order.
  - Order is first-class and checked (only recurrence dims, no typos).
- **Alternative:** Invest in a **dependency analysis** that infers the “true” recurrence dimension(s) and order when possible (e.g. “all reads use s < j” ⇒ j is the recurrence dim), and fall back to bracket order or require an override when ambiguous.

---

## 5. Automatic inference design: “two index vars on same dim”

**Goal:** Detect Cholesky-style clauses automatically and pick a **consistent** recurrence order (e.g. **j** outer, **i** inner) with no new syntax and no transpose workaround. For Cholesky both orders are correct; we choose one so behaviour is predictable and (if desired) tuned for e.g. cache.

### 5.1 Pattern we are targeting

- **Write:** `L[i, j]` (or more generally, any 2+ recurrence dims).
- **Reads:** Same array at indices that use **two different loop variables on the same dimension** across the body. Example: `L[i, s]` and `L[j, s]` → dimension 0 is read with both `i` and `j`; dimension 1 is read with `s` (reduction variable).
- **Preferred order (convention):** The dimension where “all reads are provably before the write index” (e.g. column index `s` in `0..(j-1)`) is the **sequential** dimension; we put it **outer**. The dimension where reads use “mixed” indices (both `i` and `j`) is **inner**. For Cholesky this yields column-major; row-major would also be correct, but we fix one convention.

So we classify each recurrence dimension as:

- **Strictly backward:** At that dimension, every read index is either (a) the loop variable (or ± constant), or (b) a **reduction variable whose range upper bound is that loop variable** (e.g. `s` in `sum[s in 0..(j-1)]` ⇒ `s < j`). We treat that dimension as the “time” dimension and put it **first (outer)** by convention.
- **Mixed:** At least one read at that dimension uses a **different** loop variable (e.g. row dim sees both `i` and `j`). That dimension is **inner** (or vectorized when we support it).

Inference rule: **Put all strictly-backward dims first (in existing clause order), then all mixed dims.** That yields e.g. `[1, 0]` for Cholesky (j outer, i inner). Both that order and bracket order `[0, 1]` are correct for Cholesky; we pick one for consistency.

### 5.2 Reduction-bound detection

To treat a read index as “bounded by loop var j” we need:

- The read index is a **reduction variable** (e.g. `s` in `sum[s in 0..(j-1)](...)`).
- The clause’s `reduction_ranges` (on lowered IR) map that variable’s DefId to a `LoopStructure` whose `iterable` is a `RangeIR`.
- The range’s **end** is exactly `j` or `j - 1` (same DefId as the loop variable for the dimension we’re classifying). Then we consider that read “strictly backward” for dimension j.

No symbolic range analysis: we only need to recognize “range end is loop_var or loop_var - 1” on the IR. That suffices for Cholesky and similar patterns.

### 5.3 Scope and limits

- **When we infer:** Only when a clause has **at least two** recurrence dimensions and does **not** already have `recurrence_dims_override` set (e.g. by same-timestep pass). We run this inference in the same pass that handles same-timestep, **before** same-timestep override so that Cholesky gets `[1, 0]` and we don’t overwrite it.
- **When we don’t infer:** If no dimension is classified strictly backward (e.g. both dims are mixed), we do **not** set an override; bracket order stays. If the user later adds an explicit annotation (e.g. `recur(j, i)`), that would take precedence over inference when we have syntax.
- **Backend:** Unchanged. It already uses `recurrence_dims_override` when present; we only feed it the inferred order.
- **Edge cases:** Triangular solves, 2D stencils with more complex deps may need explicit annotation later; this design fixes the “two index vars on same dim” pattern without claiming to solve all multi-dim recurrences.

### 5.4 Pass placement

- **Where:** Inside the existing **RecurrenceOrderPass** (or equivalent).
- **Order of steps:**  
  1. For each Einstein clause with ≥2 recurrence dims and no existing override, run the classifier and set `recurrence_dims_override` to the inferred order when we get exactly “some strictly backward, rest mixed”.  
  2. Then run the existing same-timestep logic (override when body reads same recurrence index as write). Do **not** overwrite an override that was just set by step 1.

This keeps one pass, one place for recurrence order, and makes Cholesky work without source or backend changes.

---

## 6. Related

- [VECTORIZATION_DESIGN.md §4](VECTORIZATION_DESIGN.md#4-recurrence-classification) — how recurrence dims are detected.
- [VECTORIZATION_DESIGN.md §5](VECTORIZATION_DESIGN.md#5-execution-path-decision-tree) — hybrid / scalar path and `recurrence_loops_for_outer`.
- `stdlib/ml/linalg_ops.ein` — Cholesky (currently xfail; inference would pick j-outer for consistency).
- `src/einlang/backends/numpy_einstein.py` — `_recurrence_dims`, `recurrence_dims_override`, `recurrence_loops_for_outer`.
