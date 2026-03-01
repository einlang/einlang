# Performance

## Profiling baseline

Profiled with `cProfile` on `examples/demos/mnist_onnx_arch.ein` (digit 3, fresh run).

```
9,144,559 function calls (8,471,083 primitive calls) in 6.34 s
```

| Phase | Time |
|-------|------|
| Runtime execution | ~4.3 s (68%) |
| Compiler (type inference, lowering) | ~1.7 s (27%) |
| Import / startup | ~0.7 s (11%) |

Times overlap; the compiler is invoked inside the total wall time.

---

## Optimization directions

### 1. Cache `_defid_of_var_in_expr` results (0.58 s, 266K calls)

**File:** `passes/einstein_lowering.py:28`

`_defid_of_var_in_expr` does a recursive tree walk to find the `DefId` of a named
variable inside an expression subtree. It is called at three call sites during
lowering — once per loop variable per binding. Because the same expression tree
is walked repeatedly with different `name` arguments, it pays for every node in
the tree on every call.

Fix: after name resolution is complete, store a `{name -> DefId}` map directly on
each binding or on the `LoweredEinsteinClauseIR`. The function becomes a single
dict lookup.

---

### 2. Eliminate interpreter dispatch overhead (1.35 M `isinstance`, 1.6 M `hasattr`)

**Files:** `ir/nodes.py`, `backends/numpy_expressions.py`

The visitor pattern calls `accept()` on every IR node, which calls the matching
`visit_*` method on the visitor. The profiler records:

- 1,614,106 calls to `builtins.hasattr` (0.35 s)
- 1,351,454 calls to `builtins.isinstance` (0.25 s)
- 945,748 calls to `builtins.getattr` (0.16 s)

These arise from defensive attribute access in expression visitors
(`visit_binary_op`, `visit_rectangular_access`, etc.) and from the `_defid_of_var_in_expr`
child-discovery logic. Together they account for ~0.75 s of pure dispatch with no
computation.

Fix options:
- Replace `hasattr`/`getattr` guards with typed `__slots__` (already partially
  present) and direct attribute access where the node type is guaranteed by the
  visitor dispatch.
- Replace the `isinstance`-based child-discovery in `_defid_of_var_in_expr` with
  a declared `children()` method or `__slots__`-derived tuple on each IR node class.

---

### 3. Intern the scope-stack lookup in the hot loop (0.30 s, 180K calls)

**File:** `runtime/environment.py:68 get_value`

`ExecutionEnvironment.get_value` walks `_scope_stack` in reverse on every call.
During Einstein loop execution the innermost scope is only one dict deep for loop
variables, but the full stack is still scanned.

Fix: keep a flat `_flat: Dict[DefId, Any]` mirror that is updated on `set_value`
and invalidated on `exit_scope`. The common case then becomes a single dict
lookup. Alternatively, for the common case where a reduction body only references
loop variables and closed-over arrays, pass an explicit `context: dict` argument
directly to the body evaluator and skip the scope stack entirely (this pattern
already exists partially in `_execute_lowered_einstein_clause`'s `full_context`
dict).

---

### 4. Avoid re-evaluating `visit_identifier` inside loops (0.32 s, 115K calls)

**File:** `backends/numpy_expressions.py:135`

`visit_identifier` is called 115K times. Each call retrieves the `defid` from the
`IdentifierIR` node and calls `get_value`. Inside a reduction or comprehension
loop, the closed-over array identifiers resolve to the same value on every
iteration.

Fix: hoist closed-over identifier loads out of the loop body. In
`_execute_lowered_einstein_clause` the body expression can be pre-scanned for
identifiers whose `defid` is not a loop variable; their values can be captured
once into a closure dict before the loop starts.

---

### 5. Replace `cell_index` per-element tuple construction (0.09 s, 12.9K calls)

**File:** `backends/numpy_einstein.py:216`

`cell_index` is a closure defined inside `_execute_lowered_einstein_clause` that
builds a Python tuple from loop variable values on every inner loop iteration.
With 8 Einstein clauses × ~1,600 cells per clause this runs ~12,900 times.

Fix: precompute the output index array (a NumPy meshgrid over loop variable
ranges) before the loop, or vectorise the entire clause body using NumPy
broadcasting (the `_try_vectorized_reduction` path already does this for
reductions; extend the same strategy to Einstein comprehensions).

---

### 6. Replace `deepcopy` in monomorphization (0.09 s, 72K copies)

**File:** `analysis/monomorphization_service.py` (via `copy.deepcopy`)

72K `deepcopy` calls during monomorphization take 0.09 s. Each instantiation of
a generic function deep-copies the IR subtree before rewriting type variables.

Fix: implement a purpose-built `clone()` method on IR node classes that allocates
fresh nodes without the generality overhead of `deepcopy`, or cache already-
instantiated monomorphic copies keyed by (function DefId × concrete type
signature) and return the cached copy on subsequent identical instantiations.

---

### 7. Reduce compiler startup time (0.73 s type inference + 0.67 s import)

**File:** `passes/type_inference.py`

Type inference takes 0.73 s for 24 bindings (the MNIST program). The profiler
shows 301 calls to `visit_function_value` (0.57 s cumulative), implying that
each unique function body is type-checked multiple times — once per call site —
rather than once per definition.

Fix: cache the inferred type signature of each function (keyed by DefId) after
the first inference and skip re-inference for subsequent call sites with the same
argument types. This is a standard monomorphic caching step before full
polymorphic inference.

Module import time (0.67 s) is dominated by loading `compiler/driver.py` and its
transitive imports. This is a one-time cost but could be reduced with lazy imports
inside the less-common code paths (e.g. backend-specific modules imported only
when needed).

---

## Expected impact

Addressing items 1–4 targets the interpreter loop, which is the dominant cost
(~4 s). A conservative estimate is 2–3× speedup on programs with large Einstein
expressions. Items 5–6 are smaller wins. Item 7 reduces compiler latency for
short programs where compilation dominates.
