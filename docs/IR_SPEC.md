# IR Specification

## Binding Model

All top-level definitions (functions, constants, Einstein declarations, variable declarations) use a single node type: **`BindingIR`**.

There are no subclasses like `FunctionDefIR` or `ConstantDefIR`. The binding kind is determined entirely by `expr` type.

### BindingIR

```
BindingIR(IRNode)
  __slots__ = ('name', 'expr', 'type_info', 'defid')
```

- `name: str` — binding name
- `expr: Any` — the value expression (determines binding kind)
- `type_info: Optional[Any]` — type annotation / inferred type
- `defid: Optional[DefId]` — definition id

Convenience properties (delegate to `self.expr`):
- `value` → `self.expr`
- `pattern` → `self.name`
- `parameters` → `self.expr.parameters` (when `expr` is `FunctionValueIR`)
- `body` → `self.expr.body` (when `expr` is `FunctionValueIR`)
- `return_type` → `self.expr.return_type` (when `expr` is `FunctionValueIR`)
- `clauses` → `self.expr.clauses` (when `expr` is `EinsteinExprIR`)

### Binding Kind Predicates

```python
is_function_binding(b)   # expr is FunctionValueIR
is_einstein_binding(b)   # expr is EinsteinExprIR
is_constant_binding(b)   # has defid, not function, not einstein
```

### Backward Compatibility Aliases

```python
FunctionDefIR = BindingIR
ConstantDefIR = BindingIR
VariableDeclarationIR = BindingIR
EinsteinDeclarationIR = BindingIR
```

**Warning**: `isinstance(x, FunctionDefIR)` matches ALL `BindingIR`. Use predicates instead.

## Visitor Dispatch

### BindingIR.accept(visitor)

Always calls `visitor.visit_binding(self)`.

### IRVisitor.visit_binding (default)

```python
def visit_binding(self, node: BindingIR) -> T:
    expr = getattr(node, 'expr', None)
    return expr.accept(self) if expr is not None else None
```

The default delegates to `node.expr.accept(self)`, which dispatches to the appropriate expression visitor:

| `node.expr` type | Dispatches to |
|---|---|
| `FunctionValueIR` | `visit_function_value_expr` |
| `EinsteinExprIR` | `visit_einstein_declaration_expr` |
| `LiteralIR` | `visit_literal` |
| other `ExpressionIR` | the corresponding `visit_*` |

### Consumer Pattern

Consumers that need binding-level info override `visit_binding`:

```python
def visit_binding(self, node: BindingIR) -> T:
    if is_function_binding(node):
        # function-specific logic using node.name, node.defid, node.expr (FunctionValueIR)
        ...
    elif is_einstein_binding(node):
        # einstein-specific logic using node.expr (EinsteinExprIR)
        ...
    else:
        # constant/variable logic
        ...
```

Consumers that only care about the value expression don't override `visit_binding` at all — the default recurses into `node.expr.accept(self)`.

## ProgramIR

```
ProgramIR
  __slots__ = ('modules', 'statements', 'source_files', 'defid_to_name', '_bindings')
```

- `statements: List[Any]` — all top-level statements (BindingIR + bare expressions)
- `_bindings: List[BindingIR]` — derived from statements
- `functions` (property) — `[b for b in _bindings if is_function_binding(b)]`
- `constants` (property) — `[b for b in _bindings if is_constant_binding(b)]`

## Expression Nodes (unchanged)

All expression nodes inherit from `ExpressionIR(IRNode)` with `__slots__ = ('type_info', 'shape_info')`.

Key value types for bindings:
- `FunctionValueIR(ExpressionIR)` — `__slots__ = ('parameters', 'return_type', 'body', '_is_partially_specialized', '_generic_defid')`
- `EinsteinExprIR(IRNode)` — `__slots__ = ('clauses', 'shape', 'element_type')`
- Any other expression for constants/variables

## Mutation Rules

- Slot attributes: direct assignment (`node.name = "x"`)
- Properties that delegate to `node.expr`: use `object.__setattr__(node.expr, 'attr', value)`
  - Example: `object.__setattr__(node.expr, 'body', new_body)` for function body
