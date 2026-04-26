# Einlang docs

This doc set is intentionally small. If you are using Einlang, start with the user docs below. If you are changing Einlang, jump to the contributor docs at the end.

## First visit

| Time | Run | Then read |
|------|-----|-----------|
| 30 seconds | `python3 -m einlang examples/hello.ein` | [GETTING_STARTED](GETTING_STARTED.md) |
| 5 minutes | `python3 -m einlang examples/autodiff_small.ein`<br>`python3 -m einlang examples/demos/matrix_operations.ein` | [AUTODIFF](AUTODIFF.md) |
| 30 minutes | `python3 -m einlang examples/applications/linear_regression_autodiff.ein`<br>`python3 -m einlang examples/recurrence/recurrence_suite.ein`<br>`python3 -m einlang examples/ode/ode_suite.ein`<br>`python3 -m einlang examples/optimization/optimization_suite.ein` | [examples/README](../examples/README.md) |

## Einlang in Practice

- [Getting started](GETTING_STARTED.md): install, first run, and the shortest learning path.
- [Einlang: Formulas as Code](https://einlang.github.io/einlang/book/): a book-style introduction to Einlang's core abstractions.
- [Examples guide](../examples/README.md): curated runnable programs by goal.
- [Autodiff guide](AUTODIFF.md): how `@x`, `@y / @x`, and related forms work.
- [Why Einlang](WHY_EINLANG.md): project motivation and positioning.
- [Syntax comparison](SYNTAX_COMPARISON.md): bridge from NumPy, Julia, or Rust habits.

## Reference

- [Language reference](reference.md): syntax and semantics.
- [Standard library](stdlib.md): library reference.

## Research

- [Paper](https://einlang.github.io/einlang/paper/einlang_paper.pdf): the compact language-design argument.
- [Thesis-form report](https://einlang.github.io/einlang/thesis/einlang_thesis.pdf): the fuller implementation writeup with compiler algorithms.

## Contribute

- [Architecture guide](ARCHITECTURE.md): repo map, compiler flow, and where to edit what.
- [Contributing](../CONTRIBUTING.md): project workflow and contribution expectations.
