# `print(@y)` symbolic tangent output

This document catalogs how the autodiff pass builds the **string** passed to `print` when the argument is a differential on a binding (e.g. `print(@y)`). It is **display-only**: lowered programs for `@y / @x` and other uses of forward diff are unchanged.

## Where it lives (code)

| Piece | Role |
|--------|------|
| `_expand_derivative_in_expr` → `BuiltinCallIR("print", [DifferentialIR])` | Only here: builds `ForwardDiffVisitor(..., pretty_call_tangents=True)`, stringifies, replaces `print` with `print(literal)`. |
| `_ForwardDiffVisitor` elsewhere | `pretty_call_tangents=False` (default). |
| `_format_print_differential_message` | If the RHS string has **newlines**, intermediate lines are plain equalities; **`@y =` appears only on the last line** (“math style”). |
| `_wrap_forward_call_tangent_binding` | Wraps the **full** callee forward tangent in `let @name = …; @name` **only** on the body–symbolic-diff path (see below). |
| `_lift_block_for_binary_op` | For **ADD/SUB** only, when pretty: hoists `BlockExpressionIR` on either operand so bindings precede the sum/difference. |
| `_expr_to_diff_source` | IR → string for the message (identifiers, blocks, `@` leaves, etc.). |

## Case matrix

### 1. Single expression, no user calls

Examples: `y = x**2`, `y = x + 1`, literals.

- **Output:** one line, `@y = …`.
- **Pretty:** no callee wrap; no ADD lifting needed for a single additive term.

### 2. `y = … + …` / `… - …` with **user** callees on the body–symbolic path

Example: `y = x + f(x)` with `fn f(t) { t + 1.0 }` and **no** `@fn f` rule.

- **Callee tangent** is wrapped: `@fx = (symbolic) * @x` (see naming below).
- **ADD** is lifted: preamble lines are `@fx = …`, final line is `@x + @fx`.
- **Message:** `_format_print_differential_message` →  
  `@fx = …` then `@y = @x + @fx`.

Same idea for **SUB** (e.g. `y = f(x) - x` → `@fx = …` then `@y = @fx - @x`).

### 3. Several calls combined with `+`

Example: `y = f(x) + g(x)`.

- Each call gets its own wrapped name (`@fx`, `@gx`, …).
- ADD lifting merges blocks → multiple preamble lines, then `@y = @fx + @gx`.

### 4. Callee name patterns (`_wrap_forward_call_tangent_binding`)

Uses call-site `IdentifierIR` for the callee when present; else the function binding’s name.

| Situation | Pattern | Example |
|-----------|---------|---------|
| Unary, callee name length **1** | `@{c}{a}` | `f(x)` → `@fx` |
| Unary, callee name **longer** | `@{c}_{a}` | `foo(x)` → `@foo_x` |
| **Multi-argument** | `@{c}_call` | `f(a, b)` → `@f_call` |

### 5. Custom `@fn` derivative (`custom_diff_body`)

If the callee has an `@fn` rule, `visit_function_call` returns on that path and **does not** apply `_wrap_forward_call_tangent_binding`.

Example: `fn f(t){ t+1 }` + `@fn f(t){ @t }`, `y = x + f(x)`:

- Printed tangent can collapse (e.g. `@y = 2 * @x`) with **no** intermediate `@fx`.

**Stdlib / `std::ml::…`** calls usually carry such rules → names like `@relu_x` still appear when the **expanded** IR introduces a block/binding from elsewhere, but the call itself is not wrapped the same way as a plain user `fn` without `@fn`.

### 6. Operations that are **not** ADD/SUB (pretty)

**MUL, DIV, POW, etc.** do not use `_lift_block_for_binary_op`.

If a **wrapped call** (a `BlockExpressionIR`) sits **inside** a product or other binary op, `_expr_to_diff_source` concatenates text linearly. The result can look **misleading** (mixed `*` and `=` on one line) or split oddly with `_format_print_differential_message`.

**Example to avoid trusting blindly:** `y = x * f(x)` with user `f` — preamble/last-line split may not match readable “math steps”.

### 7. Nested calls `f(g(x))`

Forward tangents compose correctly for **execution**; **pretty** stringification can produce **confusing multi-line** output (nested blocks + wrapping). Treat as **known rough edge** for display until hoisting rules are extended beyond ADD/SUB at the top of `d(expr)`.

### 8. Targets that are not simple scalars

Example: `y` a small tensor literal — `print(@y)` may simplify to something like `@y = 0` depending on IR. Non-scalar Einstein bindings follow the same pipeline but may use indexed LHS (`@y[i,…] = …`) when applicable.

### 9. Failure / skip cases (tests)

In `tests/unit/test_autodiff_pass.py`, `_PRINT_DIFF_ML_OPS` includes **skipped** programs (pytest `marks=`): e.g. softmax/log_softmax without `@fn`, many `reduce_*` / composite ML ops (“multi-step inlined function: intermediate var out of scope”), matmul shape issues, etc. Reasons are in the test file.

Local diagnosis: `python3 scripts/test_print_at.py --study` (see header in `scripts/test_print_at.py`).

**Full dump of each skipped program + pytest reason + math reference:** [TEST_PRINT_AT_STUDY_SKIP_DUMP.md](TEST_PRINT_AT_STUDY_SKIP_DUMP.md) (regenerate with `python3 scripts/test_print_at.py --dump-study-only > docs/TEST_PRINT_AT_STUDY_SKIP_DUMP.md`).

**Table: expected math vs actual compile/exec:** [TEST_PRINT_AT_STUDY_SKIP_COMPARE.md](TEST_PRINT_AT_STUDY_SKIP_COMPARE.md) (regenerate with `python3 scripts/gen_study_skip_compare.py`).

**IR dumps (S-expr + meta):** [study_skip_ir/README.md](study_skip_ir/README.md) (generate `.sexpr` with `python3 scripts/dump_study_skip_ir.py`). **Analysis of those dumps:** [STUDY_SKIP_IR_ANALYSIS.md](STUDY_SKIP_IR_ANALYSIS.md).

## Reference outputs (illustrative)

Repro: `PYTHONPATH=src python3` with `CompilerDriver` + `EinlangRuntime(backend="numpy")`, `print` captured from stdout.

| Program sketch | Printed shape (typical) |
|----------------|-------------------------|
| `y = x**2` | `@y = 2 * x * @x` |
| `y = x + f(x)` (user `f`, no `@fn`) | `@fx = 1 * @x` then `@y = @x + @fx` |
| `y = f(x) + g(x)` | `@fx = …`, `@gx = …`, then `@y = @fx + @gx` |
| `y = x + f(x)` with `@fn f` | `@y = 2 * @x` (no `@fx` line) |
| `y = std::ml::relu(x)` | Preamble `@relu_x = …` then `@y = @relu_x` (rule-dependent); **parenthesization** in `_expr_to_diff_source` may omit parens around `if` in products — readability quirk. |

## Tests

- **Unit:** `tests/unit/test_autodiff_pass.py` — `test_print_differential`, `test_print_differential_call_plus_x_shows_at_fx_then_sum`, parametrized `_PRINT_DIFF_ML_OPS`.
- **Script:** `scripts/test_print_at.py` — golden substrings / `--study` / `--dump-study-only` (see [TEST_PRINT_AT_STUDY_SKIP_DUMP.md](TEST_PRINT_AT_STUDY_SKIP_DUMP.md)).

## Related docs

- [AUTODIFF_DESIGN.md](AUTODIFF_DESIGN.md) — language meaning of `@` and quotients.
- [AUTODIFF_PIPELINE.md](AUTODIFF_PIPELINE.md) — pass order (this rewrite runs during derivative expansion).
