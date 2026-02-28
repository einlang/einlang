# Development

## Todo

- Remove arrow support

## Setup

```bash
git clone https://github.com/einlang/einlang.git
cd einlang
pip install -e ".[dev]"
python3 -m pytest tests/ --tb=short -q
```

Requires Python 3.7+.

## Project Layout

```
src/einlang/
├── frontend/       Lark grammar, parser, AST transformers
├── passes/         Name resolution, type inference, Einstein lowering,
│                   range/shape analysis, AST-to-IR, IR validation
├── ir/             IR nodes, S-expression serialization
├── backends/       NumPy evaluator, Einstein executor
├── runtime/        Scope stack, reduction engine
├── analysis/       Module system, monomorphization
├── compiler/       Pipeline driver
└── shared/         DefId, types, AST/IR nodes, error codes

stdlib/             Standard library (.ein files)
examples/units/     62 end-to-end .ein test programs
tests/
├── unit/           Component-level tests
├── integration/    Cross-component tests
├── examples/       Runs .ein programs from examples/
└── stdlib/         Stdlib function tests
```

## Adding a Language Feature

1. Grammar — `frontend/grammar.lark`
2. AST nodes — `shared/nodes.py`
3. Transformer — `frontend/transformers/`
4. Pass — `passes/` (type rules, lowering, analysis)
5. Backend — `backends/`
6. Tests — `tests/unit/` + `tests/examples/`

## Adding a Stdlib Function

1. Implement in Einlang — `stdlib/` (mark `pub fn`)
2. Backend support if needed — `backends/`
3. Tests — `tests/stdlib/`
4. Docs — `docs/stdlib.md`

## Error System

```python
from einlang.shared.errors import EinlangSourceError

raise EinlangSourceError(
    message="Undefined variable",
    error_code="E0003",
    category="name_resolution",
    location=location,
)
```

E001–E011 are user-facing errors. E9999 is for internal bugs.

## Style

- Precise type hints — avoid `Any`
- Dataclasses for data, not classes
- Fast-fail — no silent fallbacks
- No speculative abstractions
- `black` + `isort` for formatting, `ruff` for linting
