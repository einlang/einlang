# Einlang documentation

Start with [Getting started](https://github.com/einlang/einlang/blob/main/docs/GETTING_STARTED.md), [Language reference](https://github.com/einlang/einlang/blob/main/docs/reference.md), [Standard library](https://github.com/einlang/einlang/blob/main/docs/stdlib.md), and [examples/README](https://github.com/einlang/einlang/blob/main/examples/README.md).

---

## Main docs

| You want to… | Go here |
|--------------|--------|
| **Get running quickly** | [Getting started](https://github.com/einlang/einlang/blob/main/docs/GETTING_STARTED.md) |
| **Install or use from Python** | [README — Install & run](https://github.com/einlang/einlang/blob/main/README.md#install--run) |
| **Learn by examples** | [examples/README](https://github.com/einlang/einlang/blob/main/examples/README.md) |
| **Understand the language** | [reference.md](https://github.com/einlang/einlang/blob/main/docs/reference.md) |
| **Look up functions** | [stdlib.md](https://github.com/einlang/einlang/blob/main/docs/stdlib.md) |
| **Understand autodiff** | [AUTODIFF_HIGHLIGHTS](https://github.com/einlang/einlang/blob/main/docs/AUTODIFF_HIGHLIGHTS.md) · [AUTODIFF_DESIGN](https://github.com/einlang/einlang/blob/main/docs/AUTODIFF_DESIGN.md) |
| **Find the right doc by background** | [SYNTAX_COMPARISON](https://github.com/einlang/einlang/blob/main/docs/SYNTAX_COMPARISON.md) · [EINLANG_FOR_JULIA_PROGRAMMERS](https://github.com/einlang/einlang/blob/main/docs/EINLANG_FOR_JULIA_PROGRAMMERS.md) |
| **Get brief answers** | [FAQ](https://github.com/einlang/einlang/blob/main/docs/FAQ.md) |

---

## Core docs

| Doc | What it is |
|-----|------------|
| [GETTING_STARTED](https://github.com/einlang/einlang/blob/main/docs/GETTING_STARTED.md) | Short introduction, first commands, first example, and next steps |
| [reference](https://github.com/einlang/einlang/blob/main/docs/reference.md) | Full language reference: syntax, types, Einstein notation, where-clauses, recurrences, autodiff |
| [stdlib](https://github.com/einlang/einlang/blob/main/docs/stdlib.md) | Built-in modules and functions |
| [WHY_EINLANG](https://github.com/einlang/einlang/blob/main/docs/WHY_EINLANG.md) | Motivation and high-level comparison |
| [MATH](https://github.com/einlang/einlang/blob/main/docs/MATH.md) | Math notation mapped to Einlang code |
| [AUTODIFF_HIGHLIGHTS](https://github.com/einlang/einlang/blob/main/docs/AUTODIFF_HIGHLIGHTS.md) | Short overview of autodiff in the language |
| [AUTODIFF_DESIGN](https://github.com/einlang/einlang/blob/main/docs/AUTODIFF_DESIGN.md) | Main autodiff design and implementation doc |
| [examples/README](https://github.com/einlang/einlang/blob/main/examples/README.md) | Learning path and examples by domain |

---

## Background guides

| If you are coming from… | Start here |
|-------------------------|-----------|
| **Python / NumPy** | [SYNTAX_COMPARISON](https://github.com/einlang/einlang/blob/main/docs/SYNTAX_COMPARISON.md) · [README — Install & run](https://github.com/einlang/einlang/blob/main/README.md#install--run) |
| **Julia** | [EINLANG_FOR_JULIA_PROGRAMMERS](https://github.com/einlang/einlang/blob/main/docs/EINLANG_FOR_JULIA_PROGRAMMERS.md) · [JULIA_DEMOS](https://github.com/einlang/einlang/blob/main/docs/JULIA_DEMOS.md) |
| **Rust** | [SYNTAX_COMPARISON](https://github.com/einlang/einlang/blob/main/docs/SYNTAX_COMPARISON.md) |

---

## Quick answers

| Question | Answer |
|----------|--------|
| **How do I run Einlang?** | `python3 -m einlang -c "let x = 1+1; print(x);"` or `python3 -m einlang examples/hello.ein` |
| **How do I use it from Python?** | `from einlang import run` — see [README — Install & run](https://github.com/einlang/einlang/blob/main/README.md#install--run) |
| **Where are the examples?** | [examples/README](https://github.com/einlang/einlang/blob/main/examples/README.md) |
| **How do I get derivatives?** | Use `@expr`, `@a / @b`, `@loss / @w`, or `@C / @A`; see [AUTODIFF_HIGHLIGHTS](https://github.com/einlang/einlang/blob/main/docs/AUTODIFF_HIGHLIGHTS.md) |
| **How do I report a bug?** | Open an [issue](https://github.com/einlang/einlang/issues) or see [CONTRIBUTING](https://github.com/einlang/einlang/blob/main/CONTRIBUTING.md) |

---

## Design summary

Short version:

- keep tensor structure explicit in the source
- catch shape and index errors before execution
- keep differentiation in the language
- use one language for model code, optimization, and numerical programs

For the user-facing motivation page, see [WHY_EINLANG](https://github.com/einlang/einlang/blob/main/docs/WHY_EINLANG.md).

---

## Contributor docs

These are useful if you are changing the compiler or maintaining the docs. They are not the best starting point for learning Einlang.

| Doc | What it is |
|-----|------------|
| [DEVELOPMENT](https://github.com/einlang/einlang/blob/main/docs/DEVELOPMENT.md) | Project layout, setup, and how to add features |
| [DOCUMENTATION_DESIGN](https://github.com/einlang/einlang/blob/main/docs/DOCUMENTATION_DESIGN.md) | How the doc set is organized |
| [RELEASE_READINESS](https://github.com/einlang/einlang/blob/main/docs/RELEASE_READINESS.md) | Pre-release checklist |
| [LEARNING_FROM_JULIA](https://github.com/einlang/einlang/blob/main/docs/LEARNING_FROM_JULIA.md) | Maintainer notes on docs/showcase strategy |
| [PAPER](https://github.com/einlang/einlang/blob/main/docs/PAPER.md) | Citation text and repository citation |

---

## Deep technical notes

These docs are useful when you are working on a specific compiler or backend area.

| Area | Docs |
|------|------|
| **Autodiff internals** | [AUTODIFF_PIPELINE](https://github.com/einlang/einlang/blob/main/docs/AUTODIFF_PIPELINE.md) · [AUTODIFF_IMPLEMENTATION](https://github.com/einlang/einlang/blob/main/docs/AUTODIFF_IMPLEMENTATION.md) · [AUTODIFF_ALGORITHM](https://github.com/einlang/einlang/blob/main/docs/AUTODIFF_ALGORITHM.md) · [AUTODIFF_OPS](https://github.com/einlang/einlang/blob/main/docs/AUTODIFF_OPS.md) |
| **Einstein / lowering details** | [AUTODIFF_EINSTEIN](https://github.com/einlang/einlang/blob/main/docs/AUTODIFF_EINSTEIN.md) · [AUTODIFF_EINSTEIN_OPS](https://github.com/einlang/einlang/blob/main/docs/AUTODIFF_EINSTEIN_OPS.md) · [VECTORIZATION_DESIGN](https://github.com/einlang/einlang/blob/main/docs/VECTORIZATION_DESIGN.md) · [RECURRENCE_ORDER_DESIGN](https://github.com/einlang/einlang/blob/main/docs/RECURRENCE_ORDER_DESIGN.md) |
| **Targeted investigations** | `RUNTIME_TO_COMPILER_FINDINGS.md`, `STUDY_SKIP_IR_ANALYSIS.md`, `TEST_PRINT_AT_STUDY_SKIP_*.md`, `TRT_MNIST_EXAMPLE.md` |
