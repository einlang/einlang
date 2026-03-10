---
layout: default
title: Contributing
---

# Contributing to Einlang

Thanks for your interest. You don't need to be a compiler expert — **doc fixes and small bugs are a great way to start.** Every fix, improvement, and idea helps. New to Einlang? [Getting started](docs/GETTING_STARTED.md) or [doc index by audience](docs/README). Experts: [DEVELOPMENT](docs/DEVELOPMENT.md) for project layout and adding features.

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
python3 -m pytest tests/ --tb=short -q
```

If those pass, you're set. See [DEVELOPMENT](docs/DEVELOPMENT.md) for project layout, adding language features, and the error system.

## Good first contributions

- Fix or clarify something in [docs/reference](docs/reference.md) or [docs/stdlib](docs/stdlib.md)
- Add a test for a corner case (see `tests/unit/` and `tests/examples/`)
- Improve an error message or add a `help:` suggestion in the compiler
- Try an example from [examples/](examples/) and report what was confusing or broken

## How to run tests

```bash
pip install -e ".[test]"
python3 -m pytest tests/ --tb=short -q
```

For a single test file: `python3 -m pytest tests/unit/test_errors.py -v`

## Where to ask

- **Bugs and ideas** — [GitHub Issues](https://github.com/einlang/einlang/issues)
- **Usage questions** — Open a Discussion or an issue with the "question" label if available

## Code style

- Type hints (avoid `Any` where you can)
- `black` + `isort` for formatting, `ruff` for linting
- No mocks in tests — use real objects or fixtures (see [.cursorrules](.cursorrules))
