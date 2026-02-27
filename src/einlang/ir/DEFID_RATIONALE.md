# IR nodes with DefId — rationale

DefId is the canonical identity for definitions (variables, functions, parameters, etc.). Only nodes that represent or reference a definition carry defid. Rust alignment: similar to `rustc_hir::Node` and DefId-based resolution.

| Node | Field | Rationale |
|------|--------|-----------|
| **IdentifierIR** | `defid` | Reference to a single definition (variable or function). Lookup key for env/get_value. |
| **IndexVarIR** | `defid` | Index loop variable identity; same role as IdentifierIR for index slots. |
| **IndexRestIR** | `defid` | Rest index slot identity. |
| **FunctionCallIR** | `function_defid` | Callee when call is name-based (e.g. `f(x)`). None when callee is expression (e.g. `callee_expr`). |
| **ArrayComprehensionIR** | `variable_defids` | One DefId per bound loop variable in the comprehension. |
| **BuiltinCallIR** | `defid` | Which builtin is called; used for dispatch and tree shaking. |
| **ParameterIR** | `defid` | Parameter binding identity (single DefId per parameter). |
| **FunctionValueIR** | `_generic_defid` | Monomorphization: link to generic template. Name/defid of the binding live on BindingIR. |
| **ModuleIR** | `defid` | Module *definition* identity (the module node in ProgramIR); allocated by ModuleResolver. |
| **IdentifierPatternIR** | `defid` | Pattern binding identity (binds matched value to variable); only pattern type with defid. |
| **BindingIR** | `defid` | The LHS definition identity (name + defid); expr is the rvalue. |

No defid on: LiteralIR, BinaryOpIR, BlockExpressionIR, LambdaIR, RangeIR, ReductionExpressionIR, WhereExpressionIR, ArrowExpressionIR, PipelineExpressionIR, EinsteinClauseIR, EinsteinIR (value), MatchExpressionIR, TuplePatternIR, LiteralPatternIR, WildcardPatternIR, ArrayPatternIR, RestPatternIR, GuardPatternIR, OrPatternIR, ConstructorPatternIR, BindingPatternIR, RangePatternIR, etc. — these are pure expressions or structural nodes; reduction identity comes from the binding when the reduction is the RHS; loop vars use IndexVarIR/IdentifierIR.defid.

**Function references:** First-class function values (e.g. `let f = mean`) are represented as **IdentifierIR** (name + defid); there is no separate FunctionRefIR.

**Module reference in calls:** The *reference* to a module (e.g. in `np.sum(x)`) is currently **FunctionCallIR.module_path** (`Optional[Tuple[str, ...]]`), i.e. a name path like `("np",)` or `("python", "numpy")`, not an IdentifierIR. So: definition = ModuleIR.defid; reference = module_path (tuple). A possible future alignment would be an optional **module_ref: IdentifierIR** on FunctionCallIR (IdentifierIR.defid → ModuleIR when resolved; defid None for Python/external modules).
