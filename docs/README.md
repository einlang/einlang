# Einlang docs

This directory now keeps a small user-focused doc set. If you are learning or using Einlang, these are the pages that matter.

## Start here

- [Getting started](GETTING_STARTED.md)
- [Examples guide](../examples/README.md)
- [Language reference](reference.md)
- [Standard library](stdlib.md)
- [Autodiff guide](AUTODIFF.md)

## Choose by goal

| You want to... | Read |
|----------------|------|
| Understand what Einlang is for | [WHY_EINLANG](WHY_EINLANG.md) |
| Run your first program | [GETTING_STARTED](GETTING_STARTED.md) |
| Learn the syntax precisely | [reference](reference.md) |
| Look up modules and functions | [stdlib](stdlib.md) |
| Use gradients and tangents | [AUTODIFF](AUTODIFF.md) |
| Translate from Python, Julia, or Rust habits | [SYNTAX_COMPARISON](SYNTAX_COMPARISON.md) |
| Find runnable programs | [examples/README](../examples/README.md) |

## Quick answers

| Question | Answer |
|----------|--------|
| How do I run Einlang? | `python3 -m einlang examples/hello.ein` or `python3 -m einlang -c "let x = 1 + 1; print(x);"` |
| How do I use it from Python? | `from einlang import run` |
| Where are the examples? | [examples/README](../examples/README.md) |
| How do I take derivatives? | Bind the value, then use `@x`, `@y / @x`, or `@loss / @weights`; see [AUTODIFF](AUTODIFF.md) |
| Where do I start if I know NumPy or Julia? | [SYNTAX_COMPARISON](SYNTAX_COMPARISON.md) |

## Core docs

| Doc | Purpose |
|-----|---------|
| [README](../README.md) | Project landing page, install, first commands |
| [GETTING_STARTED](GETTING_STARTED.md) | First run and first next steps |
| [reference](reference.md) | Language syntax and semantics |
| [stdlib](stdlib.md) | Standard library reference |
| [AUTODIFF](AUTODIFF.md) | User guide for derivatives and tangents |
| [WHY_EINLANG](WHY_EINLANG.md) | Motivation and positioning |
| [SYNTAX_COMPARISON](SYNTAX_COMPARISON.md) | Mental-model bridge from other languages |
| [examples/README](../examples/README.md) | Curated example map |

## Contributing

For docs fixes, bugs, or small improvements, start with [CONTRIBUTING](../CONTRIBUTING.md).
