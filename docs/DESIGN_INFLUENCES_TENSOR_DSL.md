# Design Influences for Einlang — Tensor/Array DSLs

Einlang's scope is a **tensor/array DSL with shape and index guarantees**, not a general-purpose scientific language like Julia. This document has two focuses: **focus on what** (the language describes what is computed, not how) and **learn from what** (what to take from each influence to inform Einlang's *new* syntax and language design—concepts, structure, and design—not to adopt the existing syntax of the systems below). Sections 1–6 and the summary table are the main "learn from what" reference.

See also: [JULIA_SYNTAX_AND_DESIGN_IN_EINLANG.md](JULIA_SYNTAX_AND_DESIGN_IN_EINLANG.md) for Julia comparison; that doc covers syntax and migration gaps. Here we focus on **alternative influences** (Taichi, Triton, Accelerate, Tensor Comprehensions, Weld, ONNX) for language and compiler design.

---

## Design principle: what, not how

**Einlang focuses on describing *what* is computed, not *how* it is executed.**

- **What:** Declarative tensor expressions (Einstein notation, reductions, comprehensions, where-clauses), shapes, and types. The language specifies the mathematical meaning and shape of the result.
- **How:** Execution strategy (tiling, vectorization, parallelism, memory layout, backend) is **not** part of the language. It belongs to the compiler, the backend, or optional schedule metadata that backends may interpret. The same "what" can be executed in different ways without changing the source.

This keeps the language small, portable, and backend-agnostic. New backends can choose their own "how" without new syntax.

---

## 1. Taichi

**Scope:** Embedded DSL for **GPU and sparse computation**; Python front end; **kernel (what) vs schedule (how)** are separate.

### Key concepts

- **Kernel:** What is computed (element-wise, reduction, or structured access); written in a restricted, traceable subset.
- **Schedule:** Placement (CPU/GPU), block/thread layout, and memory hierarchy are chosen by the compiler or user separately from the kernel.
- **Sparse and dense:** Same "what" can target dense arrays or hierarchical sparse structures; backend decides layout.

### What Einlang can learn

- **Kernel = "what".** Keeping the language to a declarative kernel (Einstein, reductions, where-clauses) and leaving placement/tiling to the backend matches Taichi's split.
- **Stencil and reduction patterns** at the kernel level validate that index algebra and shape are the right abstraction; no schedule syntax in the language.

---

## 2. Triton

**Scope:** **GPU kernel DSL**; tile-based programming; used in PyTorch for fused kernels.

### Key concepts

- **Block-level "what":** Programs describe computation at the granularity of tiles (blocks); the compiler handles mapping to warps and threads.
- **Declarative loads/stores:** Operations on blocks (e.g. matrix multiply over tiles) are expressed; layout and synchronization are inferred or constrained, not hand-written at the thread level.
- **One description, many GPU targets:** Triton IR is lowered to different GPU backends; the same tile-level program can target different architectures.

### What Einlang can learn

- **Tile-level or tensor-level "what"** instead of thread-level "how"; Einlang's Einstein and reductions are at a similar semantic level—good for portability.
- **Structured ops (matmul, reduction, element-wise)** as the vocabulary; backends can map them to tiles; no need for user-facing schedule syntax.

---

## 3. Accelerate (Haskell)

**Scope:** **Embedded array language** in Haskell; compiles to CPU/GPU via LLVM; purely functional.

### Key concepts

- **Array combinators:** `map`, `zipWith`, `fold`, `scan`, stencil; no raw loops; shape is part of the type (e.g. `Array DIM2 Float`).
- **Shape in types:** Dimensions and extent are tracked; mismatches are type errors; fusion and rewriting are done on the combinator tree.
- **Embedded "what":** The Haskell host expresses control flow; the embedded array language expresses only data-parallel "what"; codegen is separate.

### What Einlang can learn

- **Combinator-style "what"** (map, reduce, scan) aligns with Einstein and comprehensions; shape in the type system supports Einlang's shape guarantees.
- **Fusion and one IR:** Accelerate fuses combinators into a single kernel; Einlang's single declarative form and lowering are in the same spirit—one "what," backend chooses "how."

---

## 4. Tensor Comprehensions (Meta)

**Scope:** **Declarative tensor expressions** with reduction formulas; C++/Python front end; compiles to optimized CUDA.

### Key concepts

- **Reduction formula:** You write a mathematical expression over index variables; reductions (sum, max, etc.) are explicit in the formula; the compiler infers iteration space and memory access.
- **No explicit loops:** Indices and bounds come from the tensor shapes and the formula; same idea as Einstein notation—indices define structure.
- **Codegen as backend:** The "what" is the formula; tiling, parallelization, and memory layout are chosen by the compiler, not the user.

### What Einlang can learn

- **Formula as "what":** Tensor Comprehensions validate that index-based, reduction-style expressions (like Einlang's Einstein + where) are a good portable "what"; no need to expose loop or schedule syntax.
- **Shape and index consistency:** Bounds are derived from shapes and index use; similar to Einlang's shape/range inference for safety and lowering.

---

## 5. Weld

**Scope:** **Parallel data IR**; libraries expose operations that build a Weld IR; **one IR, many backends** (CPU, GPU); fusion across library boundaries.

### Key concepts

- **Lazy IR construction:** Operations (map, filter, merge, etc.) build a Weld IR tree; execution is deferred; optimizer fuses and schedules the whole tree.
- **Single abstraction:** Many libraries (NumPy-like, pandas-like, etc.) can target the same IR; the backend compiles the fused tree to the target.
- **"What" in the IR:** The IR describes the computation (reductions, scans, builders); the compiler decides parallelism, vectorization, and memory.

### What Einlang can learn

- **One IR for "what":** Einlang's IR can be the single representation of the computation; multiple front ends or backends can target it; Weld shows the benefit of a small, fusion-friendly "what" IR.
- **Fusion and backend:** Keeping the language declarative allows the backend to fuse and schedule without language-level schedule primitives.

---

## 6. ONNX

**Scope:** **Portable graph format** for tensor ops; used for inference and cross-framework export; **ops as declarative "what".**

### Key concepts

- **Op-centric:** Nodes are named ops (MatMul, ReduceSum, Conv, etc.) with typed inputs/outputs and attributes; the graph is the "what"—no execution order or schedule in the spec.
- **Shape and type:** Tensors have element type and shape; shape inference is defined per op; runtimes implement the op semantics and choose "how."
- **Many runtimes:** Same graph runs on CPU, GPU, NPU, etc.; the graph describes structure and semantics; backends handle codegen and scheduling.

### What Einlang can learn

- **Structured op vocabulary:** A small set of well-defined ops (contraction, reduction, broadcast, element-wise) as the semantic building blocks; Einlang's Einstein and reductions map naturally to such a vocabulary for a future backend or export.
- **Shape and type in the "what":** ONNX's shape inference and type rules reinforce that the graph (or IR) should carry shape and type so that backends and tools can reason about the computation without running it.

---

## 7. Summary table

| System       | Main lesson for Einlang                                      |
|-------------|---------------------------------------------------------------|
| **Taichi**  | Kernel (what) vs schedule (how); stencil/reduction at kernel level; no schedule in the language. |
| **Triton**  | Tile-level "what"; structured ops as vocabulary; one description, many GPU targets. |
| **Accelerate** | Array combinators and shape in types; fusion and one IR; embedded "what" vs host control. |
| **Tensor Comprehensions** | Formula as "what"; index-based reductions; bounds from shapes; codegen as backend. |
| **Weld**    | One IR for "what"; lazy IR, fusion across boundaries; backend chooses parallelism and memory. |
| **ONNX**    | Op-centric graph as "what"; shape/type in the spec; many runtimes, portable semantics. |

---

## 8. No new syntax proposed

The influences above are used for **learn from what** (concepts, IR design, backend targets). None of the corresponding feature ideas (size parameters, schedule hints, vmap, index/dimension in type, named dimensions, pure/traceable annotation, grad) are proposed or needed for Einlang's current scope; existing shape inference, comprehensions, Einstein, and where-clauses already cover the "what." Execution strategy stays out of the language.

---

## 9. Suggested directions

- **Focus on what:** Declarative tensor expressions, shapes, and types. Leave **how** (tiling, parallelism, codegen) to the backend.
- **Learn from what:** Use the summary table and sections 1–6 to inform Einlang's *new* syntax and language—take concepts, structure, IR, safety from each system; do not copy their existing syntax or add "how" into the language.
- **Native/GPU:** If you add a non-NumPy backend, MLIR Linalg or a similar structured-op vocabulary is a natural target; schedule and codegen stay in the backend.

This document can be updated as Einlang's backend or type system evolves; treat it as a living reference for tensor/array DSL design.
