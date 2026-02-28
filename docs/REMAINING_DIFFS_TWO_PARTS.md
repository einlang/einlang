# Plan: Split Remaining Branch Diffs into 2 Parts

Branch: `wip/ir-defid-patterns`. Goal: apply the remaining refactor in two reviewable chunks.

---

## Part 1 – IR model and producers

**Theme:** Single binding node (BindingIR), ProgramIR from statements only, function/constant as binding with expr. No consumer pass changes yet; keep existing visitor method names where possible so Part 2 can switch gradually.

### 1.1 `src/einlang/ir/nodes.py`

- **FunctionValueIR**: New node (expression). Holds `parameters`, `return_type`, `body`, `_is_partially_specialized`, `_generic_defid`. No name/defid (those live on the binding).
- **FunctionDefIR**: Subclass of BindingIR; `expr` is FunctionValueIR. Add properties `parameters`, `body` from `self.expr`.
- **ConstantDefIR**: Subclass of BindingIR (binding that is not a function).
- **VariableDeclarationIR**: Remove. All variable declarations are BindingIR (name, expr, defid, location).
- **EinsteinDeclarationIR**: Remove. Einstein declarations are BindingIR with `expr = EinsteinIR`; clauses live on `expr.clauses`.
- **ProgramIR**: Constructor takes `statements` (and optional `source_files`, `modules`). Derive `bindings = [s for s in statements if isinstance(s, BindingIR)]`, `_functions` / `_constants` as filtered views; expose `functions` and `constants` as properties so existing `node.functions` / `node.constants` still work.
- **ModuleIR**: `functions` / `constants` types become `List[BindingIR]`.
- **Helpers**: `is_einstein_binding(binding)`, `is_function_binding(binding)`, `is_constant_binding(binding)` (branch logic).
- **IRVisitor**: Add `visit_function_value_expr` for FunctionValueIR. Keep `visit_function_def` / `visit_constant_def` / `visit_variable_declaration` / `visit_einstein_declaration` as thin delegates to `visit_binding` (or same signature calling `visit_binding`) so existing pass code still compiles until Part 2.

### 1.2 `src/einlang/ir/__init__.py`

- Export FunctionValueIR, updated BindingIR/FunctionDefIR/ConstantDefIR, and `is_einstein_binding`, `is_function_binding`, `is_constant_binding`.
- Stop exporting VariableDeclarationIR, EinsteinDeclarationIR if removed (or keep as deprecated aliases to BindingIR for one commit if needed).

### 1.3 `src/einlang/ir/serialization.py`

- Serialize/deserialize BindingIR, FunctionValueIR, and ProgramIR with statements-only constructor; backward compatibility or one-time migration for existing blobs if required.

### 1.4 `src/einlang/passes/ast_to_ir.py`

- Build BindingIR for all top-level and local bindings (variables, constants, functions, Einstein).
- For functions: build FunctionValueIR as `expr`, wrap in BindingIR(name, defid, expr=FunctionValueIR(...)).
- Build ProgramIR(statements=[...], source_files=..., modules=...) so that `program.functions` / `program.constants` (properties) match current behavior.

### 1.5 Tests / docs (minimal for Part 1)

- Adjust tests that construct IR directly (e.g. ProgramIR, VariableDeclarationIR, EinsteinDeclarationIR) to use BindingIR and new ProgramIR constructor.
- No need to change every pass test in Part 1; they still see the same logical shape via `program.functions`, `program.constants`, `program.statements`.

**Part 1 outcome:** IR tree is built in the new shape; existing code that uses `node.functions`, `node.constants`, `node.statements`, and existing visitor method names still runs. Passes that reach into `.value`, `.clauses`, `.body` on the old node types will need Part 2.

---

## Part 2 – Passes, backends, runtime, and tests

**Theme:** Every consumer uses BindingIR, `expr`, `getattr(decl, 'expr', None)`, and `is_einstein_binding` / `is_function_binding` / `is_constant_binding`. Remove or replace use of VariableDeclarationIR, EinsteinDeclarationIR, and direct FunctionDefIR/ConstantDefIR attributes where they duplicated the binding shape.

### 2.1 Passes

| File | Changes |
|------|--------|
| **range_analysis.py** | Use BindingIR and `is_einstein_binding`; `decl.expr` for Einstein (clauses on `expr.clauses`); program scope from BindingIR; function body from `getattr(expr, 'body', None)` and params from `getattr(expr, 'parameters', [])`. |
| **shape_analysis.py** | `_resolve_dependent_ranges_on_decl` / `infer_einstein_shape` / `check_perfect_partition` / `_get_index_combinations` take BindingIR; clauses via `getattr(getattr(decl, 'expr', None), 'clauses', [])`. |
| **einstein_lowering.py** | visit_einstein_declaration(node: BindingIR); clauses from `getattr(node.expr, 'clauses', [])`; replace VariableDeclarationIR with BindingIR when assigning lowered result; use `is_einstein_binding(stmt)`. |
| **rest_pattern_preprocessing.py** | Use BindingIR and `is_einstein_binding`; clauses from `getattr(node.expr, 'clauses', [])`; variable lookups via BindingIR. |
| **const_folding.py** | visit_binding instead of visit_function_def / visit_constant_def / visit_einstein_declaration; drop those methods or keep as no-ops; LambdaIR/function value from expr. |
| **type_inference.py** | visit_program: iterate bindings/functions/constants from program; visit_binding; function parameters/body from binding.expr (FunctionValueIR). |
| **ir_validation.py** | visit_program: functions/constants/statements; visit_binding; recurse into binding.expr; no separate visit_variable_declaration / visit_einstein_declaration if removed from base. |
| **tree_shaking.py** | Treat bindings (BindingIR) for refs; function/constant/Einstein via expr. |
| **name_resolution.py** | Resolve names for BindingIR; function/constant from binding.expr. |
| **einstein_grouping.py** | Use is_einstein_binding; clauses from binding.expr. |
| **implicit_range_detector.py** | Program statements as bindings; Einstein from is_einstein_binding and expr.clauses. |
| **arrow_optimization.py** | Any declaration/binding handling updated to BindingIR. |

### 2.2 Analysis

| File | Changes |
|------|--------|
| **dce_visitor.py** | Block statements as bindings; visit_binding; use expr for value/body. |
| **monomorphization_service.py** | Rewrite calls in statements; function definitions as BindingIR with expr = FunctionValueIR. |

### 2.3 Backends and driver

| File | Changes |
|------|--------|
| **numpy_arrow_pipeline.py** | Use BindingIR / expr where declarations are read. |
| **numpy_core.py** | Same. |
| **numpy_einstein.py** | Same. |
| **numpy_expressions.py** | Same (remaining call sites). |
| **numpy_helpers.py** | Same. |
| **compiler/driver.py** | Build or pass ProgramIR with statements-only; use program.functions / program.constants as properties. |

### 2.4 Runtime

| File | Changes |
|------|--------|
| **runtime/environment.py** | Lookup/register bindings by defid; value from binding.expr for constants/variables. |
| **runtime/runtime.py** | Execute program.statements; treat BindingIR for execution. |

### 2.5 Tests and docs

- **tests/unit/test_passes_einstein_lowering.py** (and integration): Update any direct use of VariableDeclarationIR / EinsteinDeclarationIR to BindingIR; expectations on structure (e.g. binding.expr).
- **tests/unit/test_arrows_pipeline.py**: Add or align arrow/pipeline tests if present on branch.
- **tests/test_utils.py**: Helpers that build IR updated to BindingIR and new ProgramIR.
- **docs**: DEVELOPMENT.md, SYNTAX_DESIGN_REFERENCES.md, UNCOMMITTED_CHANGES.md as needed; no formatting-only churn.

---

## Dependency order

- Part 1 must land first so that the IR and ast_to_ir produce the new shape.
- Part 2 can be one large MR or split further (e.g. 2a: passes that only read program.functions/constants/statements and visit_binding; 2b: passes that touch Einstein/variable declarations and backends/runtime).

---

## Optional: Part 1 compatibility shim

If desired, in Part 1 only:

- Keep `VariableDeclarationIR` and `EinsteinDeclarationIR` as thin wrappers that hold a `BindingIR` and delegate to it (or construct BindingIR from existing .value / .clauses), and have ast_to_ir produce BindingIR only. Then remove the wrappers in Part 2. This allows a gentler transition but adds code; the plan above assumes a direct cutover in Part 1 (remove old node types, ast_to_ir builds BindingIR only).
