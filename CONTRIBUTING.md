
# Contributing to Einlang

Thanks for your interest. You do not need to be a compiler expert to help. Doc fixes, tests, and small bug fixes are all valuable.

If you are new to the project, start with [Getting started](https://github.com/einlang/einlang/blob/main/docs/GETTING_STARTED.md), [docs/README](https://github.com/einlang/einlang/blob/main/docs/README.md), and [examples/README](https://github.com/einlang/einlang/blob/main/examples/README.md).
If you want the codebase map before touching implementation, read [docs/ARCHITECTURE](https://github.com/einlang/einlang/blob/main/docs/ARCHITECTURE.md).

## You don't need to ask permission

- **Typos and docs** — Edit and open a PR. No issue needed.
- **Small fixes** — Same. If it's clearly a bug, fix it and reference the behavior in the PR.
- **Features or design changes** — Open an issue first so we can align; then PR when ready.

## Get going in a few minutes

```bash
git clone https://github.com/einlang/einlang.git
cd einlang
pip install -e ".[test]"
python3 -m einlang examples/hello.ein
python3 -m pytest -n auto tests/ --tb=short -q
```

If those pass, you are ready to work in the repo.

## Good first contributions

- Fix or clarify something in [docs/reference](https://github.com/einlang/einlang/blob/main/docs/reference.md) or [docs/stdlib](https://github.com/einlang/einlang/blob/main/docs/stdlib.md)
- Add a test for a corner case (see `tests/unit/` and `tests/examples/`)
- Improve an error message or add a `help:` suggestion in the compiler
- Try an example from [examples/](https://github.com/einlang/einlang/tree/main/examples) and report what was confusing or broken

## Where should this change go?

- Parser or syntax work: `src/einlang/frontend/`
- Name resolution or module loading: `src/einlang/passes/name_resolution.py` and `src/einlang/analysis/module_system/`
- AST to IR lowering: `src/einlang/passes/ast_to_ir.py`
- Shape, range, or type analysis: `src/einlang/passes/range_analysis.py`, `shape_analysis.py`, `type_inference.py`
- Autodiff behavior: `src/einlang/passes/autodiff/`
- Einstein lowering or recurrence behavior: `src/einlang/passes/einstein_lowering.py`, `recurrence_order.py`
- Runtime or backend execution: `src/einlang/runtime/` and `src/einlang/backends/`
- Public docs or examples: `docs/` and `examples/`

The longer version lives in [docs/ARCHITECTURE](https://github.com/einlang/einlang/blob/main/docs/ARCHITECTURE.md).

## How to run tests

```bash
pip install -e ".[test]"
python3 -m pytest -n auto tests/ --tb=short -q
```

For a single test file: `python3 -m pytest tests/unit/test_errors.py -v`

## Where to ask

- **Bugs and ideas** — [GitHub Issues](https://github.com/einlang/einlang/issues)
- **Usage questions** — Open a Discussion or an issue with the "question" label if available

## Code style

- Type hints (avoid `Any` where you can)
- `black` + `isort` for formatting, `ruff` for linting
- No mocks in tests — use real objects or fixtures (see [.cursorrules](.cursorrules))
