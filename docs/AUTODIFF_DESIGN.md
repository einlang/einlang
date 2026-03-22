# Autodiff Pass — Design Document

> Canonical reference for the forward-mode automatic differentiation pass in
> Einlang. Every algorithm described here maps 1-to-1 to the implementation
> in `src/einlang/passes/autodiff.py`.

---

## 1. Terminology

| Symbol | Meaning |
|--------|---------|
| `d(x)` or `_@x` | **Differential** (tangent) of `x` — an infinitesimal change in the same space as `x`. |
| `@x` | User-facing syntax for the differential of `x`. Printed as `@x`. |
| `@y / @x` | **Derivative** (Jacobian entry) — the coefficient relating `d(y)` to `d(x)`. Computed by extracting the linear coefficient from the differential. |
| `_@` | Internal binding prefix for differential bindings (e.g. `_@sum_val`). |
| `DIFF_PREFIX` | The string `"_@"`. |

**Key invariant**: `@y` is a *differential*, not a derivative.
`d(y) = f'(x) * d(x)` — the derivative `f'(x)` is the *coefficient* of `d(x)` inside `d(y)`.

---

## 2. Architecture Overview

```
Phase 1: Analysis
  ├─ Collect @y and @y/@x targets from program IR
  ├─ Build dependency graph (DefId → Set[DefId])
  └─ Topo-sort reachable bindings

Phase 2: Forward Differentiation
  ├─ Assign symbolic differentials: d(leaf) = _@leaf (an opaque tangent)
  │   Only @y/@x quotients override this: d(denominator) = 1, d(other_leaves) = 0
  ├─ For each binding in topo order:
  │     d(y) = DiffVisitor.visit(y.expr)
  │     Create _@y binding
  └─ Insert _@* bindings after their primal bindings

Phase 3: Expand & Emit
  ├─ Replace DifferentialIR(@y) with _@y reference
  ├─ Compute @y/@x via JacobianVisitor
  └─ Format print(@y) via DiffPrinter

Phase 4: Cleanup
  ├─ Strip DifferentialType from all type_info
  ├─ Remove DiffRuleIR statements
  └─ Clear custom_diff_body on FunctionValueIR
```

---

## 3. Visitors

### 3.1 DiffVisitor — Forward Differential

The core visitor. Computes `d(expr)` given a `DiffContext` that maps each
known `DefId` to its differential expression.

**Responsibilities:**
- Forward-mode chain rule through all expression types
- Function call inlining (differentiate callee body with caller tangents)
- Custom `@fn` rule application
- Einstein clause differentiation (product rule + reduction linearity)

**Replaces (old code):** `_ForwardDiffVisitor`, `_forward_d_y_expr`,
forward-mode parts of `_SymbolicDiffVisitor`,
`_symbolic_diff_function_body_block`.

### 3.2 JacobianVisitor — Coefficient Extraction

Computes `∂(expr)/∂(wrt)` symbolically. Used only for `@y/@x` quotients.

**Responsibilities:**
- Symbolic partial derivative w.r.t. a single `DefId`
- Einstein index expansion for tensor Jacobians (`_build_jacobian_indices`)
- Chain rule through binding references (resolves let-bindings)

**Replaces (old code):** `_SymbolicDiffVisitor` (symbolic derivative path),
`_diff_einstein_wrt`.

### 3.3 DiffPrinter — Formatting

Converts differential IR to human-readable string for `print(@y)`.

**Replaces (old code):** `_expr_to_diff_source`, `_format_print_differential_message`.

### 3.4 CleanupVisitor — Post-Pass

Strips autodiff-only artifacts from the IR.

**Replaces (old code):** `_ClearAutodiffArtifactsVisitor`,
`_StripDiffTypesWalker`.

---

## 4. DiffContext

```python
class DiffContext:
    """Maps DefId → d(DefId) expression for the current differentiation scope."""

    def __init__(self, diffs: Dict[DefId, ExpressionIR]):
        self._diffs = dict(diffs)

    def get(self, defid: DefId) -> ExpressionIR:
        """Return d(defid), defaulting to LiteralIR(0) for unknown variables."""
        return self._diffs.get(defid, LiteralIR(0))

    def set(self, defid: DefId, expr: ExpressionIR) -> None:
        self._diffs[defid] = expr

    def child(self, overrides: Dict[DefId, ExpressionIR]) -> "DiffContext":
        """Create child scope (for function body). Parent entries visible unless overridden."""
        merged = dict(self._diffs)
        merged.update(overrides)
        return DiffContext(merged)
```

The context carries the differential of every variable known at a given
program point. When entering a callee body, a child context is created with
`d(param_i) = d(arg_i)`.

**Leaf treatment**: At the top level, leaf variables (inputs with no
defining expression in the program) get *symbolic* differentials —
`d(x)` = `IdentifierIR("_@x")`, an opaque tangent that propagates
through the chain rule unchanged. Only for `@y/@x` quotient computation
are leaves overridden: `d(denominator) = 1`, `d(other_leaves) = 0`,
which extracts the Jacobian coefficient.

---

## 5. DiffVisitor — Complete Algorithm

### 5.1 Atoms

```
visit_identifier(node):
    return ctx.get(node.defid)       # lookup from context; 0 if unknown

visit_literal(node):
    return LiteralIR(0)

visit_array_literal(node):
    return LiteralIR(0)
```

### 5.2 Binary Operations

```
visit_binary_op(node):
    da = visit(node.left)
    db = visit(node.right)
    a, b = node.left, node.right

    match node.operator:
        ADD:  return da + db
        SUB:  return da - db
        MUL:  return a * db + b * da          # product rule
        DIV:  return (b * da - a * db) / b²   # quotient rule
        POW:  return pow_chain_rule(a, b, da, db)
        MOD:  return da                        # subgradient
```

**Power chain rule** `d(a^b)`:
- If `b` is a constant literal `n`: `n * a^(n-1) * da`
- If `a` is a constant literal `c`: `c^b * ln(c) * db`
- General: `a^b * (b/a * da + ln(a) * db)`

### 5.3 Unary Operations

```
visit_unary_op(node):
    d_operand = visit(node.operand)
    match node.operator:
        NEG:  return -d_operand
        POS:  return d_operand
```

### 5.4 Reductions

```
visit_reduction_expression(node):
    d_body = visit(node.body)

    match node.operation:
        SUM:   return sum[loop_vars](d_body)             # linearity
        MAX:   return SelectAtArgmaxIR(body, d_body, …)  # subgradient
        MIN:   return SelectAtArgmaxIR(body, d_body, …, use_argmin=True)
        PROD:  return prod_diff_rule(node, d_body)
```

**PROD differentiation** `d(prod[j](f_j))`:

`d(prod) = sum[j]( (prod / f_j) * d(f_j) )`

Implemented as: `prod[j](f) / f * d_body` (element-wise in the reduction
variable, using the full product divided by each factor).

### 5.5 Indexing and Casting

```
visit_rectangular_access(node):
    d_array = visit(node.array)
    return RectangularAccessIR(d_array, node.indices)

visit_cast_expression(node):
    d_inner = visit(node.expr)
    return CastExpressionIR(d_inner, node.target_type)
```

**Shape invariant**: `d(A[i,j])` has the same indices as `A[i,j]`.

### 5.6 Control Flow

```
visit_if_expression(node):
    d_then = visit(node.then_expr)
    d_else = visit(node.else_expr) if node.else_expr else LiteralIR(0)
    return IfExpressionIR(node.condition, d_then, d_else)
```

The condition is not differentiated (it is boolean).

### 5.7 Einstein Expressions

```
visit_einstein(node):
    new_clauses = []
    for clause in node.clauses:
        d_value = visit(clause.value)    # product rule / reduction compose naturally
        new_clauses.append(EinsteinClauseIR(
            indices=clause.indices,      # same output indices
            value=d_value,
            where_clause=clause.where_clause,
            variable_ranges=clause.variable_ranges,
        ))
    return EinsteinIR(clauses=new_clauses, shape=node.shape, …)
```

**No index expansion needed** for forward-mode differentials. The
differential stays in the same index space as the primal.

Example: `C[i] = sum[j](A[i,j] * B[j])`

`d(C[i]) = sum[j]( A[i,j] * d(B[j]) + d(A[i,j]) * B[j] )`

The product rule inside the sum body is handled by `visit_binary_op(MUL)`
and the sum linearity by `visit_reduction_expression(SUM)`.

### 5.8 Block Expressions

```
visit_block_expression(block):
    out_stmts = []
    for stmt in block.statements:
        if stmt is BindingIR:
            d_expr = visit(stmt.expr)
            d_expr = simplify(d_expr)

            if is_literal_zero(d_expr):
                ctx.set(stmt.defid, LiteralIR(0))    # zero-inline, no binding
            else:
                d_binding = BindingIR("_@" + stmt.name, d_expr, …)
                out_stmts.append(d_binding)
                ctx.set(stmt.defid, IdentifierIR(d_binding))

    d_final = visit(block.final_expr)
    d_final = simplify(d_final)

    if out_stmts:
        return BlockExpressionIR(out_stmts, d_final)
    return d_final
```

**Zero-inlining rule**: When `d(binding) = 0`, no `_@binding` is created.
The literal `0` is stored directly in the context, so subsequent
expressions that reference `d(binding)` get `LiteralIR(0)` and simplify
correctly.

### 5.9 Function Calls

The most complex visit method. Two paths:

#### Path A: Custom @fn Rule

```
if callee.custom_diff_body is not None:
    return apply_custom_diff_rule(callee, args)
```

See Section 6 for the custom rule algorithm.

#### Path B: Differentiate Callee Body

```
params = callee.parameters
args = node.arguments

# d(param_i) = d(arg_i) in caller context
param_diffs = {p.defid: visit(args[i]) for i, p in enumerate(params)}
param_primals = {p.defid: args[i] for i, p in enumerate(params)}

return visit_callee_body(callee.body, param_diffs, param_primals)
```

### 5.10 Callee Body Algorithm

This is the function-inlining core. It processes the callee's block body
with the caller's tangents threaded through.

```
visit_callee_body(block, param_diffs, param_primals):
    child_ctx = ctx.child(param_diffs)
    sub_visitor = DiffVisitor(child_ctx, bindings, resolver)

    out_stmts = []
    for stmt in block.statements:
        # 1. Differentiate in callee coordinates
        d_expr = sub_visitor.visit(stmt.expr)

        # 2. Replace callee-local identifiers with caller-site primals
        d_expr = substitute(d_expr, param_primals)

        # 3. Simplify (0*x → 0, x+0 → x, (b*u)/b² → u/b, etc.)
        d_expr = simplify(d_expr)

        # 4. Zero-inlining
        if is_literal_zero(d_expr):
            child_ctx.set(stmt.defid, LiteralIR(0))
        else:
            d_binding = BindingIR("_@" + stmt.name, d_expr, …)
            out_stmts.append(d_binding)
            child_ctx.set(stmt.defid, IdentifierIR(d_binding))

    d_final = sub_visitor.visit(block.final_expr)
    d_final = substitute(d_final, param_primals)
    d_final = simplify(d_final)

    if out_stmts:
        return BlockExpressionIR(out_stmts, d_final)
    return d_final
```

**Why this fixes reduce_mean correctly:**

Given `reduce_mean(x)` with body:
```
let sum_val[..batch] = sum[j](x[..batch, j])
let count = len(x[0]) as f32
let mean[..batch] = sum_val[..batch] / count
mean
```

Step-by-step:

1. `d(sum_val)` = `sum[j](d(x[.., j]))` = `sum[j](_@x[.., j])`
   — Non-trivial → creates `_@sum_val` binding, stored in context.

2. `d(count)` = `d(cast(len(x[0]), f32))` = `cast(d(len(x[0])), f32)` = `cast(0, f32)` → simplifies to `0`.
   — **Zero-inlined**: `LiteralIR(0)` in context, no `_@count` binding.

3. `d(mean)` = quotient rule on `sum_val / count`:
   `(count * d(sum_val) - sum_val * d(count)) / count²`

   - `d(count)` is fetched from context = `LiteralIR(0)` directly
   - `sum_val * 0` → `0` by `_simplify`
   - `count * _@sum_val - 0` → `count * _@sum_val` by `_simplify`
   - `(count * _@sum_val) / count²` → `_@sum_val / count` by quotient cancellation rule

4. Substitute callee primals: `count` → `cast(len(x[0]), f32)`, `sum_val` → Einstein expression.

Result: `_@sum_val / cast(len(x[0]), f32)` — mathematically correct.

---

## 6. Custom @fn Rule Algorithm

Einlang supports user-defined differentiation rules via `@fn`:

```einlang
fn exp(x) { python::numpy::exp(x) }
@fn exp(x) { exp(x) * @x }
```

The `@fn` body uses `@param` to refer to the differential of the parameter.

### 6.1 Single-Parameter Functions (Common Case)

```
apply_custom_diff_rule(callee, args):
    rule_body = callee.custom_diff_body

    primal_map  = {param.defid: args[0]}           # param → arg
    diff_map    = {param.defid: visit(args[0])}     # @param → d(arg)

    result = substitute_with_diffs(rule_body, primal_map, diff_map)
    result = substitute_callee_locals(result, callee.body, primal_map)
    return result
```

`substitute_with_diffs` replaces:
- `IdentifierIR(param)` → `args[0]` (primal substitution)
- `DifferentialIR(@param)` → `d(args[0])` (tangent substitution)

### 6.2 Multi-Parameter Functions

For `@fn f(x, y) { rule_body }`:

```
terms = []
for i, param in enumerate(params):
    # Extract coefficient for param_i: set @param_i = 1, @param_j = 0 (j ≠ i)
    unit_diffs = {p.defid: (lit(1) if j == i else lit(0))
                  for j, p in enumerate(params)}

    coef = substitute_with_diffs(rule_body, primal_map, unit_diffs)
    coef = substitute_callee_locals(coef, callee.body, primal_map)
    coef = simplify(coef)

    # Multiply coefficient by actual tangent
    terms.append(coef * visit(args[i]))

return sum(terms)    # d(f) = ∂f/∂x * d(x) + ∂f/∂y * d(y)
```

Example: `@fn atan2(y, x) { x / (x*x + y*y) * @y + (-y / (x*x + y*y)) * @x }`

With `unit_diffs = {@y: 1, @x: 0}`:
coef_y = `x / (x*x + y*y) * 1 + (-y / (x*x + y*y)) * 0` = `x / (x*x + y*y)`

With `unit_diffs = {@y: 0, @x: 1}`:
coef_x = `x / (x*x + y*y) * 0 + (-y / (x*x + y*y)) * 1` = `-y / (x*x + y*y)`

Result: `coef_y * d(y) + coef_x * d(x)`.

---

## 7. JacobianVisitor — @y/@x Quotients

Used for `@y/@x` syntax. Computes `∂(y_expr)/∂(wrt_defid)`.

### 7.1 Core Difference from DiffVisitor

| Aspect | DiffVisitor | JacobianVisitor |
|--------|------------|-----------------|
| Computes | `d(expr)` (differential) — symbolic | `∂(expr)/∂(wrt)` (partial derivative) — numeric coefficient |
| `visit_identifier` | `ctx.get(defid)` — returns symbolic `_@x` for leaves | `1` if `defid == wrt`, `0` if independent, chain through bindings otherwise |
| Einstein | Differentiate clause value (no index expansion) | Index expansion for Jacobian shape |
| Used by | Phase 2 (main pass), `print(@y)` | Phase 3 (`@y/@x` expansion only) |

### 7.2 visit_identifier

```
visit_identifier(node):
    if node.defid == wrt_defid:
        return LiteralIR(1)     # ∂x/∂x = 1

    if node.defid in stmt_partial_by_defid:
        return stmt_partial_by_defid[node.defid]  # chain rule through block lets

    if node.defid in bindings:
        b = bindings[node.defid]
        if b.name starts with DIFF_PREFIX:
            return node  # opaque tangent, don't re-differentiate
        return b.expr.accept(self)   # chain through binding RHS

    return LiteralIR(0)   # independent variable
```

### 7.3 Einstein Index Expansion

For `@C/@A` where `C[i] = sum[j](A[i,j] * B[j])`:

The Jacobian `∂C[i]/∂A[i',j']` has shape `[i, i', j']` — the output
indices of C plus the indices of A.

Algorithm:
1. Find positions where `A` (the `wrt` variable) appears as a factor
2. Build derivative index variables `[i', j']` with ranges from A's shape
3. Add delta constraints: `where i == i' and j == j'`
4. Apply product rule with remaining factors

Result: `dC[i, i', j'] = sum[j](delta(j,j') * delta(i,i') * B[j])` which
simplifies to `dC[i, i', j'] = B[j'] where i == i'`.

This logic is encapsulated in `_build_jacobian_indices()`.

### 7.4 Quotient Dispatch

When expanding `@y/@x`:

```
if x_defid is a direct dependency of y_expr:
    # Use JacobianVisitor for symbolic derivative
    result = JacobianVisitor(wrt=x_defid).visit(y_expr)

elif x_defid is transitively reachable from y_defid:
    # Use forward-mode: d(y) / d(x)  (both already computed)
    result = JacobianVisitor(wrt=x_defid).visit(y_expr)

else:
    result = LiteralIR(0)   # independent
```

---

## 8. Simplification Rules

`_simplify(expr)` applies algebraic identities recursively. Applied eagerly
at every binding creation point.

### 8.1 Complete Rule Set

```
Additive:
    0 + x   →  x
    x + 0   →  x
    lit + lit  →  lit     (constant fold)
    x + x   →  2 * x     (like-term collection)

Subtractive:
    x - 0   →  x
    lit - lit  →  lit

Multiplicative:
    0 * x   →  0
    x * 0   →  0
    1 * x   →  x
    x * 1   →  x
    lit * lit  →  lit

Division:
    0 / x     →  0
    (b * u) / b²  →  u / b    (quotient cancellation, structural equality on b)
    (u * b) / b²  →  u / b    (commutativity variant)

Power:
    x ** 1  →  x

Unary:
    -(-x)   →  x
```

### 8.2 Structural Equality (`_ir_structurally_equal`)

Used by the quotient cancellation rule. Compares IR trees by structure:

- `IdentifierIR`: by `defid`
- `LiteralIR`: by `value`
- `BinaryOpIR`: operator + recursive left/right
- `UnaryOpIR`: operator + recursive operand
- `RectangularAccessIR`: array + indices (recursive)
- `IndexVarIR`: by `defid`
- `FunctionCallIR`: by `function_defid` + arguments
- `CastExpressionIR`: by inner expression + target type

---

## 9. Substitution

### 9.1 `substitute(expr, map: Dict[DefId, ExpressionIR])`

Replaces every `IdentifierIR` whose `defid` is in `map` with the
corresponding expression. Recurses into all IR node types including:

- `BinaryOpIR`, `UnaryOpIR` operands
- `BlockExpressionIR` statement exprs and final_expr
- `EinsteinClauseIR` values, indices, where_clause, variable_ranges
- `ReductionExpressionIR` body, where_clause, loop_var_ranges
- `RectangularAccessIR` array and indices
- `IfExpressionIR` condition, then_expr, else_expr
- `FunctionCallIR` arguments
- `CastExpressionIR` inner expression
- `SelectAtArgmaxIR` primal_body, diff_body

### 9.2 `substitute_with_diffs(expr, primal_map, diff_map)`

Like `substitute` but additionally replaces `DifferentialIR(@param)` nodes:

- `DifferentialIR(IdentifierIR(defid))` where `defid in diff_map` → `diff_map[defid]`
- `IdentifierIR(defid)` where `defid in primal_map` → `primal_map[defid]`

Used by custom `@fn` rule application.

---

## 10. Shape Handling

### 10.1 Shape Invariant

**`d(x)` has the same shape as `x`.**

- `d(scalar)` is scalar (`f32`)
- `d(tensor[i,j])` is `_@tensor[i,j]` — same shape, same element type
- For Einstein clauses: output indices unchanged
- `_@y` inherits `type_info` and `shape_info` from primal `y`

### 10.2 Jacobian Shapes

For `@y/@x` where `y` has shape `[a, b]` and `x` has shape `[c, d]`:

The Jacobian has shape `[a, b, c, d]` — output dimensions concatenated
with input dimensions.

This is handled by `_build_jacobian_indices` which creates new `IndexVarIR`
nodes with ranges derived from `x.shape[dim]`.

When the backend materializes a **cotangent w.r.t. a single array argument** (the usual `@y/@x` use case), that value has **`size` matching the argument** — the same shape convention as Julia Zygote and ChainRules pullbacks. Full Jacobians keep the concatenated index layout above; reductions that contract with the primal indices still yield an array-shaped result for each input, not a ragged or squeezed layout unless the primal rule is inherently lower-rank (documented separately for max/min).

---

## 11. print(@y) Formatting

### 11.1 Algorithm

1. Look up primal binding `y` and its `_@y` differential binding
2. Collect all transitive `_@*` dependencies of `_@y`
3. Sort by program order
4. Format preamble: for each dependency, `@name = format(d_binding.expr)`
5. Format main: `@y = format(_@y.expr)`

### 11.2 Display Conventions

- Internal prefix `_@` is displayed as `@` in user output
- Identifier `_@sum_val` prints as `@sum_val`
- Full expression trees are formatted with standard math precedence
- Einstein indices printed inline: `@e[i] = exp(x[i]) * @x[i]`

### 11.3 Preamble Example

For `softmax(x)`:
```
@e = exp(x[i]) * @x[i]
@s = sum[k](@e[k])
@y[i] = (s * @e[i] - e[i] * @s) / s ** 2
```

---

## 12. Pass Entry Point

### 12.1 AutodiffPass.run

```python
class AutodiffPass(BasePass):
    requires = [TypeInferencePass, UnifiedShapeAnalysisPass]

    def run(self, ir: ProgramIR, tcx: TyCtxt) -> ProgramIR:
        result = self._run_autodiff_core(ir, tcx)
        CleanupVisitor().visit(result)
        return result
```

### 12.2 _run_autodiff_core

```
1. Collect targets
   differential_targets = collect_differential_targets(program)
   quotient_pairs = collect_quotient_pairs(program)

2. Build binding map
   binding_by_defid: Dict[DefId, BindingIR]
   Include program bindings + function_ir_map from tcx

3. Build dependency graph
   binding_to_deps: Dict[DefId, Set[DefId]]
   Via DefIdCollector on each binding's expr

4. Compute reachable set
   BFS from target defids through dependency graph

5. Topo-sort
   DFS through reachable set, respecting dependencies

6. Create _@* identifiers
   For each reachable binding, allocate DefId for _@name.
   Leaves get symbolic differentials (_@leaf is an opaque IdentifierIR).
   Exception: for @y/@x quotients, seed d(denominator) = 1, d(other_leaves) = 0.

7. Build _@* RHS via DiffVisitor
   ctx = DiffContext({defid: _@ref for each reachable defid})
   For each binding in forward order:
     if leaf (no deps): _@y = symbolic _@y identifier (or 0/1 for quotient seeds)
     else: _@y = DiffVisitor(ctx).visit(binding.expr)

8. Create _@* BindingIR nodes
   Propagate type_info, shape_info from primal

9. Insert into program
   After each primal binding, insert its _@* binding

10. Expand DifferentialIR and @y/@x
    _expand_derivative_nodes_in_program(…)

11. Store analysis results in tcx
    diff_block, differential_targets, autodiff_differential_map, etc.
```

---

## 13. File Organization

```
src/einlang/passes/autodiff.py

    DIFF_PREFIX = "_@"

    # ─── Helpers ───
    _float_lit()
    _is_zero_const()
    _ir_structurally_equal()
    _simplify()

    # ─── Substitution ───
    _substitute()
    _substitute_with_diffs()
    _substitute_where_clause()
    _substitute_loop_var_ranges()

    # ─── DiffContext ───
    class DiffContext

    # ─── Utility Visitors ───
    class _DefIdCollector(IRVisitor[None])
    class CleanupVisitor              # strips DifferentialType, DiffRuleIR, custom_diff_body

    # ─── Visitor 1: DiffVisitor ───
    class DiffVisitor(IRVisitor[ExpressionIR])
        visit_identifier()
        visit_literal()
        visit_binary_op()
        visit_unary_op()
        visit_function_call()
        visit_callee_body()           # function body inlining
        apply_custom_diff_rule()      # @fn rule handling
        visit_reduction_expression()
        visit_einstein()
        visit_rectangular_access()
        visit_block_expression()
        visit_if_expression()
        visit_cast_expression()
        … (stubs return 0 for non-differentiable nodes)

    # ─── Visitor 2: JacobianVisitor ───
    class JacobianVisitor(IRVisitor[ExpressionIR])
        visit_identifier()            # returns 1 for wrt, chains through bindings
        visit_einstein()              # index expansion for tensor Jacobians
        _build_jacobian_indices()     # creates derivative index vars + delta constraints
        … (other methods delegate to standard diff rules)

    # ─── Formatting ───
    _expr_to_diff_source()
    _format_print_differential_message()
    _binary_op_precedence()
    _needs_parens_left(), _needs_parens_right()

    # ─── Target Collection ───
    _collect_differential_targets()
    _collect_quotient_pairs()
    _collect_defids()

    # ─── Expansion ───
    _expand_derivative_in_expr()
    _expand_derivative_nodes_in_program()
    _ensure_block_has_d_bindings()

    # ─── Pass Entry Point ───
    class AutodiffPass(BasePass)
        run()
        _run_autodiff_core()
```

---

## 14. Migration Table

| Old (current) | New |
|---------------|-----|
| `_SymbolicDiffVisitor` | `JacobianVisitor` (for @y/@x) + `DiffVisitor.visit_callee_body` (for function inlining) |
| `_ForwardDiffVisitor` | `DiffVisitor` |
| `_forward_d_y_expr` | Eliminated — `DiffVisitor` handles all expression types via visitor dispatch |
| `_diff_einstein_wrt` | `JacobianVisitor.visit_einstein` (index expansion) + `DiffVisitor.visit_einstein` (simple clause diff) |
| `_symbolic_diff_function_body_block` | `DiffVisitor.visit_callee_body` |
| `_substitute_prior_literal_partial_bindings` | Eliminated — zero-inlining in `DiffContext` handles this |
| `DIFF_PREFIX = "∂"` | `DIFF_PREFIX = "_@"` |
| `_DefIdCollector` | Kept unchanged |
| `_StripDiffTypesWalker` | Folded into `CleanupVisitor` |
| `_ClearAutodiffArtifactsVisitor` | Folded into `CleanupVisitor` |
| `_wrap_forward_call_tangent_binding` | Folded into `DiffVisitor.visit_function_call` |
| `_flatten_add_block_terms` | Folded into `DiffVisitor` result assembly |
| `_lift_block_for_binary_op` | Folded into `DiffVisitor.visit_binary_op` |

---

## 15. Phase 3 Expansion Algorithm

Phase 3 walks the IR after `_@*` bindings have been inserted (Phase 2) and
replaces all `DifferentialIR` nodes and `@y/@x` quotients with plain IR.

### 15.1 `_expand_derivative_in_expr` — Core Dispatch

Recursive structural walk returning a new `ExpressionIR`.

**DifferentialIR(@y)**:
```
if operand is IdentifierIR with defid in defid_to_d_ident:
    return IdentifierIR(ref.name, ref.defid)    # resolved _@y reference
else:
    return DiffVisitor(defid_to_d_ident, scope_bindings, resolver).visit(operand)
```

**BinaryOpIR(DIV, DifferentialIR, DifferentialIR)** — quotient `@num/@den`:

Simplified dispatch (old code had redundant direct-dep vs transitive-dep
branches that did the same thing):
```
num_expr = resolve_defining_expr(num_operand)
den_defid = resolve_defid(den_operand)

if den_defid is reachable from num_expr:
    return simplify(JacobianVisitor(wrt=den_defid).visit(num_expr))
else:
    return LiteralIR(0)
```

When `den_operand` is an arbitrary expression (not a simple IdentifierIR):
extract the single DefId it depends on, then compute `d_num / d_den`
via JacobianVisitor on both sides.

**BuiltinCallIR("print", [DifferentialIR(@y)])** — `print(@y)`:

Delegates to `_format_print_at(y_defid, y_name, ...)` (extracted helper):
```
1. Look up y_expr = scope_defid_to_expr[y_defid]
2. diff_rhs = DiffVisitor(pretty_call_tangents=True).visit(y_expr)
3. Build preamble: for each transitive dep with a _@* binding, format it
4. Format main line and emit BuiltinCallIR("print", [LiteralIR(message)])
```

**BlockExpressionIR** — lazy block-local differentials:

**REDESIGNED**: The old `_ensure_block_has_d_bindings` duplicated the entire
autodiff core (dep graph, topo-sort, seeding) for each block. Eliminated.

Instead, expansion handles blocks lazily:
```
expand_block(block):
    extended_scope = dict(outer_scope)
    extended_defid_to_d = dict(defid_to_d_ident)
    new_stmts = []
    for stmt in block.statements:
        new_stmts.append(expanded_stmt)
        extended_scope[stmt.defid] = stmt
    # DifferentialIR / @y/@x inside block.final_expr or stmt exprs
    # are resolved using extended_scope which includes block-locals.
    # JacobianVisitor chains through bindings naturally.
    # DiffVisitor for @local uses a DiffContext built from extended_scope.
```

When the expansion encounters `@local` where `local` is block-scoped and
not in `defid_to_d_ident`: create the `_@local` binding on the fly by
running `DiffVisitor` on its defining expr with a block-scoped context,
insert it, and register in extended map. This avoids the old mini-autodiff.

**All other node types**: recurse into children, rebuilding the node.
Leaf types (`IdentifierIR`, `LiteralIR`, `IndexVarIR`, etc.) pass through.

### 15.2 `_expand_derivative_nodes_in_program`

In-place walk over `program.bindings` then `program.statements`, expanding
each expr. Builds scope maps incrementally as it walks.

### 15.3 `_primal_to_following_diff_binding_map(bindings)`

Scan for consecutive `(primal, _@primal)` pairs in binding list. Returns
`Dict[DefId, BindingIR]`. Used by `print(@y)` preamble.

### 15.4 `_transitive_primal_dep_defids_from_expr(expr, binding_by_defid)`

Walk expression collecting all reachable primal DefIds by following
IdentifierIR → binding RHS recursively. Stops at FunctionValueIR.
Returns `Set[DefId]`.

---

## 16. Einstein Jacobian — Full Algorithm

`_diff_einstein_wrt(expr, wrt_defid, loc, binding_by_defid, resolver, wrt_tangent)`

Computes `d(EinsteinIR)/d(wrt)` producing a new `EinsteinIR`. Processes each
clause independently:

### 16.1 Non-Reduction Clause

If clause value is not `ReductionExpressionIR`: delegate entire value to
`JacobianVisitor`, keep same clause indices.

### 16.2 SUM — Single Factor (Linear Shortcut)

When `val.operation == SUM` and the body is a single `RectangularAccessIR`
of the wrt variable (only 1 factor, and that factor is wrt):

```
d_inner = JacobianVisitor(wrt).visit(val.body)
→ sum[j](d(x[.., j]))  — no index expansion needed
```

### 16.3 SUM — Product Rule (General Case)

```
1. Flatten inner product into factors: [(arr_id, indices), ...]
2. Find wrt_positions: indices where arr_id.defid == wrt_defid
3. If no wrt_positions:
   - Chain rule: d(sum_r f)/d(wrt) = sum_r JacobianVisitor(wrt).visit(f)
4. Build derivative index vars via _build_derivative_index_vars()
5. For each wrt_position:
   - other_factors = all factors except this position
   - delta_constraints = [wrt_idx[p] == deriv_idx[p] for new indices]
   - body = product of other_factors (or 1 if none)
   - Create ReductionExpressionIR(SUM, loop_vars, body, where=constraints)
6. Sum all reduction terms with ADD
7. New clause indices = derivative_index_vars (allow_reuse) or output + deriv indices
```

### 16.4 MAX/MIN

```
1. Build derivative index vars
2. Build diff_body = nested IfExpressionIR:
   for each new index p: if wrt_idx[p] == deriv_idx[p] then 1 else 0
3. Return SelectAtArgmaxIR(primal_body=val.body, diff_body, loop_vars,
                           use_argmin=(operation==MIN))
```

### 16.5 PROD

When `allow_reuse` (output and wrt share index structure):
```
1. Build derivative index vars
2. prod_exclude_constraints = [wrt_idx[p] != deriv_idx[p] for new indices]
3. Create ReductionExpressionIR(PROD, loop_vars, val.body,
                                where=original_where + exclude_constraints)
```
This produces `prod[k != j](x[k])` — product of all elements except the one
being differentiated.

### 16.6 Helper: `_build_derivative_index_vars(clause_indices, wrt_indices, wrt_id, resolver, loc, allow_reuse)`

For each dimension `p` of the wrt variable:
- If `allow_reuse` and wrt_indices[p] has the same defid as a clause index:
  reuse that clause index variable.
- Otherwise: allocate new `IndexVarIR("_ad_p", new_defid)` with
  `range = 0..wrt_id.shape[p]`.

Returns `(index_vars, new_defids, new_var_ranges)`.

### 16.7 Helper: `_flatten_product(expr)`

Recursively decomposes `BinaryOpIR(MUL, ...)` into
`List[(array_ident, [indices])]` where each element is a
`RectangularAccessIR(IdentifierIR, indices)`. Returns `None` if the
expression is not a pure product of indexed arrays.

### 16.8 Helper: `_merged_reduction_loop_var_ranges(val, clause)`

Returns `dict(val.loop_var_ranges) | matching clause.variable_ranges` so
that derivative reductions have explicit ranges even when the primal relied
on inference.

---

## 17. CleanupVisitor — Full Algorithm

In-place IR mutation. Extends `IRVisitor` to walk every node:

### 17.1 Type Stripping

For every node that has `type_info`: if it is a `DifferentialType`, replace
with `inner_type` via `strip_differential_types_deep()` (from `shared/types.py`).

Additional type-bearing fields:
- `CastExpressionIR.target_type` — strip if `DifferentialType`
- `FunctionValueIR.return_type` — strip
- `ParameterIR.param_type` — strip for each parameter
- `EinsteinIR.element_type` — strip
- `BindingIR.type_info` — strip

### 17.2 Artifact Removal

- `FunctionValueIR.custom_diff_body`: walk it (to strip types inside), then
  set to `None`.
- `ProgramIR.statements`: filter out `DiffRuleIR` nodes, rebuild
  `program.bindings` from the filtered list.
- Modules: recurse into `functions`, `constants`, `submodules`.

### 17.3 Traversal

Must visit all expression types exhaustively. Key types and their children:
- `BinaryOpIR`: left, right
- `UnaryOpIR`: operand
- `BlockExpressionIR`: statements, final_expr
- `IfExpressionIR`: condition, then_expr, else_expr
- `FunctionCallIR`: callee_expr, arguments
- `BuiltinCallIR`: args
- `RectangularAccessIR`: array, indices
- `JaggedAccessIR`: base, index_chain
- `ReductionExpressionIR`: loop_vars, body, where_clause constraints
- `EinsteinIR/EinsteinClauseIR`: clauses → indices, value, where_clause
- `SelectAtArgmaxIR`: primal_body, diff_body
- `CastExpressionIR`: expr
- `DifferentialIR`: operand
- `LambdaIR`: parameters, body
- `RangeIR`: start, end
- `ArrayLiteralIR/TupleExpressionIR`: elements
- `TupleAccessIR`: tuple_expr
- `MemberAccessIR`: object
- `MatchExpressionIR`: scrutinee, arm bodies
- `InterpolatedStringIR`: parts (if ExpressionIR)
- `IndexVarIR`: range_ir
- `IndexRestIR`: (collect defid)
- `WhereExpressionIR`: expr, constraints
- `PipelineExpressionIR`: left, right
- `ArrayComprehensionIR`: loop_vars, ranges, constraints, body
- `TryExpressionIR`: operand

---

## 18. Utility Algorithms

### 18.1 `_DefIdCollector(IRVisitor[None])`

Walks all IR nodes and collects `defid` from `IdentifierIR` and `IndexRestIR`
into `self.defids: Set[DefId]`. Must visit every expression type's children
(same traversal list as CleanupVisitor Section 17.3).

### 18.2 `_collect_defids(expr) -> Set[DefId]`

Convenience: create `_DefIdCollector`, accept expr, return `collector.defids`.

### 18.3 `_collect_autodiff_targets(program)` — unified collector

**REDESIGNED**: Old code had two separate walks (`_collect_differential_targets`
and `_collect_quotient_pairs`) with duplicate recursion logic. Merged into a
single walk that collects both in one pass.

```
_collect_autodiff_targets(program) -> (List[(DefId, name)], List[(num_defid, den_defid)])

Single recursive walk over the IR. At each node:
  - DifferentialIR(IdentifierIR): add to differential_targets
  - BinaryOpIR(DIV, DifferentialIR, DifferentialIR): add to quotient_pairs,
    and also add both operands to differential_targets
  - Recurse into children

Returns (differential_targets, quotient_pairs).
```

In-expression variant `_collect_autodiff_targets_in_expr(expr)` walks a
single subtree (used by block-local expansion).

### 18.4 `_is_reachable(source_defid, target_defid, binding_by_defid) -> bool`

BFS from `source_defid`. At each node, collect defids from the binding's
RHS via `_collect_defids`. Return `True` if `target_defid` is reached.

### 18.5 `_set_type_info(expr, type_info, shape_info)`

In-place recursive propagation: for each node, if `type_info is None` and
caller provides non-None type_info, fill it. Recurse into all children.
Only fills `None` slots — never overwrites existing type information.

### 18.6 Block Lifting

**`_lift_block_for_binary_op(op, left, right, loc)`**:
When either operand is `BlockExpressionIR`, extract statements and final_expr:
```
{s1}e1 OP {s2}e2  →  {s1; s2}(e1 OP e2)
```
If neither is a block, return plain `BinaryOpIR(op, left, right)`.

**`_flatten_add_block_terms(terms, loc)`**:
Combine a list of ADD terms. For each `BlockExpressionIR` term, extract its
statements into a merged list and keep only its `final_expr` as the
arithmetic operand. Fold all `final_expr`s with `ADD`. Wrap in
`BlockExpressionIR` if there are any merged statements.

### 18.7 `_bindings_in_block(block, program)`

For `ProgramIR`: return `program.bindings`. For `BlockExpressionIR`: return
binding statements. Else empty list.

---

## 19. Formatting

### 19.1 `_expr_to_diff_source(expr, d_defid_to_at_name, scope_binding_by_defid, parent_op=None)`

Recursive IR-to-string converter for `print(@y)` output.

**Dispatch by node type:**

| Node type | Format |
|-----------|--------|
| `IdentifierIR` | `d_defid_to_at_name[defid]` if in map, else binding name |
| `LiteralIR` | Integer-format floats when exact (e.g. `2` not `2.0`) |
| `BinaryOpIR` | `left op right` with precedence-based parenthesization |
| `UnaryOpIR` | `op operand` |
| `IfExpressionIR` | `if cond { then } else { else }` |
| `FunctionCallIR` | `name(args)` |
| `RectangularAccessIR` | `array[indices]` |
| `ReductionExpressionIR` | `op[loop_vars](body)` |
| `EinsteinIR` | Single clause: inline value; multi: `{ [idx] = val; ... }` |
| `CastExpressionIR` | Format inner expression (cast is invisible) |
| `BlockExpressionIR` | `stmt1;\nstmt2;\nfinal` or inline if single let |
| `SelectAtArgmaxIR` | `select_at_argmax(primal, diff)` |
| `DifferentialIR` | `@operand` |
| `IndexVarIR` | name |
| Other | `?` |

### 19.2 Precedence and Parenthesization

```
Precedence levels:
  ADD, SUB = 1
  MUL, DIV, MOD = 2
  POW = 3
  Unary = 4 (implicit, highest)

_needs_parens_left(child, parent_op):
    child is BinaryOpIR and precedence(child.op) < precedence(parent_op)

_needs_parens_right(child, parent_op):
    Same as left, plus: if same precedence AND parent is SUB or DIV or POW,
    parenthesize (right-associativity / non-commutative cases).

_needs_parens(expr, parent_op):
    If parent_op is None, no parens. Else check if expr is BinaryOpIR with
    lower precedence than parent.
```

### 19.3 `_format_print_differential_message(lhs, rhs_str)`

If `rhs_str` is single-line: return `"lhs = rhs"`.
If multi-line: split into `preamble_lines` and `last_line`. Strip trailing
semicolons from preamble. Return `preamble\nlhs = last_line`.

### 19.4 Index Formatting

`_idx_str(idx)`:
- `IndexVarIR` → `name`
- `IndexRestIR` → `..name` (or `..` if no name)
- `IdentifierIR` → `name`
- Other → `?`
