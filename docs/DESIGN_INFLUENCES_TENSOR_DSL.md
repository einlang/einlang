# Design Influences for Einlang — Tensor/Array DSLs

Einlang’s scope is a **tensor/array DSL with shape and index guarantees**, not a general-purpose scientific language like Julia. This document lists systems that align with that scope and what Einlang can learn from them: key concepts, code or API examples, and concrete takeaways.

See also: [JULIA_SYNTAX_AND_DESIGN_IN_EINLANG.md](JULIA_SYNTAX_AND_DESIGN_IN_EINLANG.md) for Julia comparison; that doc covers syntax and migration gaps. Here we focus on **alternative influences** (Halide, Futhark, JAX, MLIR Linalg, TVM, Dex, Rust) for language and compiler design.

---

## 1. Halide

**Scope:** DSL for image processing and stencil/conv-style code. **Algorithm vs schedule** are separate.

### Key concepts

- **Algorithm:** What is computed (pure functions over infinite integer domain; images as functions).
- **Schedule:** How and in what order (tiling, vectorization, parallelism, storage); can change without touching the algorithm.
- **Scheduling primitives:** `reorder`, `split`, `fuse`, `tile`, `vectorize`, `parallel`, `unroll`.

### Example (algorithm + schedule)

```cpp
// Algorithm: one line
Func gradient("gradient");
gradient(x, y) = x + y;

// Schedule: separate from algorithm
Var x_outer, y_outer, x_inner, y_inner, tile_index;
gradient.tile(x, y, x_outer, y_outer, x_inner, y_inner, 4, 4)
        .fuse(x_outer, y_outer, tile_index)
        .parallel(tile_index);

Buffer output = gradient.realize({8, 8});
```

Default schedule is row-major; changing to `reorder(y, x)` gives column-major. Same algorithm, different codegen.

### What Einlang can learn

- **Keep “what” separate from “how”.** Einstein + where clauses already express “what”; a future backend (or schedule layer) can own “how” (tiling, vectorize, parallel).
- **Stencil/conv patterns** and boundary handling: Halide’s patterns validate that where-clause index algebra is the right level of abstraction.
- Even without a full schedule API today, design IR and lowering so that **schedule hints or a separate schedule pass** could be added later without rewriting the algorithm.

---

## 2. Futhark

**Scope:** Purely functional **parallel array language**; compiles to GPU (CUDA/OpenCL) and multicore.

### Key concepts

- **Size parameters** in types: `[n]` constrains array sizes and is inferred at call sites.
- **Second-order array combinators:** map, reduce, scan, etc., as the main abstraction (no raw loops).
- **Shape polymorphism:** Functions generic over dimensions; compiler enforces shape consistency.

### Example (size parameters)

```futhark
def dotprod [n] (xs: [n]i32) (ys: [n]i32): i32 =
  reduce (+) 0 (map2 (*) xs ys)

def res = dotprod [1,2,3] [4,5,6]
```

`[n]` is a size parameter: both inputs must have the same length; `n` is inferred, not passed. Result type can be constrained too, e.g. `[1]i32` for singleton.

### What Einlang can learn

- **Shape polymorphism:** Expose size/rank in the type system (e.g. `[n]` or named dimensions) so generic tensor code can be typed; your shape inference is a step toward that.
- **Small, predictable primitives** that map cleanly to parallel/GPU (map, reduce, scan, scatter/gather); your Einstein + reductions are in the same family.
- **No raw for/while:** Futhark’s choice matches Einlang’s (comprehensions, recurrence, Einstein); their type/shape rules and backend story are a good reference for future evolution.

---

## 3. JAX / XLA

**Scope:** NumPy-like front end; **tracing + compilation** to XLA; AD, JIT, multiple backends.

### Key concepts

- **Pure functions:** Traced and compiled; side-effect-free code is easy to optimize and transform.
- **Composable transforms:** `grad`, `jit`, `vmap`, `pmap`; can be stacked (e.g. `jit(vmap(grad(f)))`).
- **Single abstraction:** Array + primitives; many backends implement the same ops.

### Example (transforms)

```python
import jax.numpy as jnp
from jax import jit, grad, vmap

def loss(params, x, y):
    pred = predict(params, x)
    return jnp.sum((pred - y) ** 2)

# Composed: vectorized gradients, then JIT-compiled
grad_loss = jit(vmap(grad(loss), in_axes=(None, 0, 0)))
```

Tracing turns the Python function into a Jaxpr; each transform operates on that representation. Same program can run on CPU, GPU, or TPU.

### What Einlang can learn

- **Purity and tracing:** Explicit `let` and value-oriented style already fit “traceable”; avoid hidden state so future AD or JIT can transform the program.
- **Transform pipeline:** Think in “transform over a tensor program” (e.g. vmap = add batch dim, grad = backward pass); even without AD yet, this guides IR design.
- **One IR, many backends:** Your backend interface is in the same spirit; JAX’s mental model (one high-level description, many targets) is a good fit for Einlang’s goals.

---

## 4. MLIR (Linalg, Tensor dialects)

**Scope:** Compiler IR and dialects for **tensor/linear algebra**; targets CPU, GPU, TPU, etc.

### Key concepts

- **Indexing maps:** Each operand has an affine map from loop indices to tensor dimensions; they define the “structure” of the op.
- **Structured ops:** Matmul, conv, reduction as first-class; iteration space derived from operands.
- **Contraction detection:** Ops are identified as contractions via indexing maps, not op names; enables generic fusion and codegen.

### Example (concept)

`linalg.generic` expresses a custom contraction: you specify indexing maps for inputs/output; the iteration space is derived. Matrix multiply: one map uses `(i, k)`, another `(k, j)`, output `(i, j)`; the missing `k` in the output = reduction. Same idea as Einstein notation: indices that appear only on the RHS are reduction axes.

### What Einlang can learn

- **Einstein → Linalg:** Your `sum[k](A[i,k]*B[k,j])` is a natural source for Linalg-style generic ops; a future native/GPU lowering could target MLIR.
- **Named dimensions and indexing maps** instead of ad-hoc loop nests; your shape analysis already produces structure that can map to affine maps.
- **Structured op vocabulary:** Matmul, conv, reduction, broadcast as first-class in IR so backends can recognize and optimize them.

---

## 5. TVM / Apache TVM

**Scope:** Compiler stack for **tensor expressions** and ML; many backends (CPU, GPU, accelerators).

### Key concepts

- **Tensor expression IR (TeLang):** Describe compute; separate **schedule** (tile, split, bind to thread/block, etc.).
- **Auto-tuning:** Search over schedules for a given target; shows how a high-level DSL can stay portable and still get performance.
- **Reduction and broadcast** as first-class; same semantic level as Einlang’s reductions and same-rank broadcast.

### What Einlang can learn

- **“What” vs “how” again:** Same as Halide; your Einstein form is the “what”; a schedule layer (current or future) is the “how.”
- **Op vocabulary:** TVM’s set of primitives (compute, reduce, broadcast) aligns with what you already have; their scheduling and codegen are a reference for a future native backend.

---

## 6. Dex

**Scope:** Typed **array language** with **index and shape in the type system**; research language.

### Key concepts

- **Indices as types:** `Fin n` = indices 0..n-1; you cannot index with an arbitrary integer.
- **Shapes in types:** Array type reflects dimensions and their index types, e.g. `(Fin 3) => (Fin 8) => Float32`.
- **Inferred loop bounds:** When iterating over arrays with known shape, ranges come from types; no explicit size passing.

### Example (concept)

```text
for i j.  -- i, j inferred from context (e.g. array shape)
  ...
```

Array type carries the “size” so that loops and indexing are checked at compile time. Flattened or reshaped arrays can be typed with product indices `(Fin 3 & Fin 8) => Float32`.

### What Einlang can learn

- **Index and shape in types:** For “what could Einlang’s type system grow into?”, Dex is a reference: indices and bounds as first-class, leading to safe indexing and inferred ranges.
- **Where clauses:** Your index algebra (e.g. `where ih = oh+kh`) is in the same spirit as Dex’s index sets; both avoid hand-written bounds and improve safety.

---

## 7. Rust (ndarray, burn)

**Scope:** General language with **explicit tensor/array** libraries; no GC, zero-cost.

### Key concepts

- **Explicit dimensions:** `Array2`, `Dim<[I, J]>`; shape mismatches are compile-time errors.
- **No hidden allocation:** APIs make when and where memory is used clear; fusion in Einlang has the same goal.
- **Fallible APIs, no implicit broadcast by default:** Aligns with Einlang’s “explicit when ranks differ.”

### What Einlang can learn

- **API design:** How to expose shape and layout without sacrificing safety; how to avoid implicit behavior that breaks tracing or optimization.
- **If you add a native backend:** Rust’s array libraries show how to keep zero-cost and predictable performance while staying shape-safe.

---

## 8. Summary table

| System       | Main lesson for Einlang                                      |
|-------------|---------------------------------------------------------------|
| **Halide**  | Separate algorithm (what) from schedule (how); tiling/vectorize/parallel as separate layer. |
| **Futhark** | Size parameters and shape polymorphism; second-order combinators; no raw loops. |
| **JAX**     | Purity and tracing; composable transforms (grad, jit, vmap); one abstraction, many backends. |
| **MLIR Linalg** | Indexing maps and contractions; Einstein notation maps naturally; target for future codegen. |
| **TVM**     | Tensor expr + schedule; “what” vs “how”; op vocabulary and auto-tuning. |
| **Dex**     | Indices and shapes in types; inferred bounds; index sets. |
| **Rust ndarray** | Explicit shapes and no implicit behavior; API design for safety and zero-cost. |

---

## 9. Suggested directions

- **Short term:** Keep algorithm/schedule separation in mind in IR and lowering; avoid baking “how” into the language so a future backend or schedule pass can own it.
- **Type system:** Consider size/shape parameters (Futhark) or index types (Dex) for shape-polymorphic and bounds-safe APIs.
- **Transforms:** Design so that “transform over tensor program” (e.g. vmap, grad) is possible later without reworking the core language.
- **Native/GPU:** Use MLIR Linalg (and Halide/TVM schedule ideas) as the target and vocabulary when you add a non-NumPy backend.

This document can be updated as Einlang’s backend or type system evolves; treat it as a living reference for tensor/array DSL design.
