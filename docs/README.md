
# Einlang documentation

One place to find your path. **Single source of truth:** [Language reference](https://github.com/einlang/einlang/blob/main/docs/reference.md) · [Standard library](https://github.com/einlang/einlang/blob/main/docs/stdlib.md) · [Install & run](https://github.com/einlang/einlang/blob/main/README.md#install--run) (in main README). Everything below links there; nothing here duplicates those.

---

## New to Einlang? / Need help?

| You want to… | Go here |
|--------------|--------|
| **Get going in one page** | [Getting started](https://github.com/einlang/einlang/blob/main/docs/GETTING_STARTED.md) |
| **Learn by doing** | [Try it](https://github.com/einlang/einlang/blob/main/README.md#try-it) → [Learning path](https://github.com/einlang/einlang/blob/main/examples/README.md) |
| **Learn by background** | [Syntax comparison](https://github.com/einlang/einlang/blob/main/docs/SYNTAX_COMPARISON.md) (Python/Julia/Rust) · [Einlang for Julia programmers](https://github.com/einlang/einlang/blob/main/docs/EINLANG_FOR_JULIA_PROGRAMMERS.md) |
| **Common questions** | [FAQ](https://github.com/einlang/einlang/blob/main/docs/FAQ.md) |
| **Ask or contribute** | [CONTRIBUTING](https://github.com/einlang/einlang/blob/main/CONTRIBUTING.md) · [GitHub issues](https://github.com/einlang/einlang/issues) |

---

## By audience

| You are… | Start here | Then |
|----------|------------|------|
| **Starter** | [Getting started](https://github.com/einlang/einlang/blob/main/docs/GETTING_STARTED.md) | [Try it](https://github.com/einlang/einlang/blob/main/README.md#try-it) → [Learning path](https://github.com/einlang/einlang/blob/main/examples/README.md) |
| **Student** | [Getting started](https://github.com/einlang/einlang/blob/main/docs/GETTING_STARTED.md) | [Learning path](https://github.com/einlang/einlang/blob/main/examples/README.md) → [Reference](https://github.com/einlang/einlang/blob/main/docs/reference.md) |
| **User (any)** | [Try it](https://github.com/einlang/einlang/blob/main/README.md#try-it) · [Install & run](https://github.com/einlang/einlang/blob/main/README.md#install--run) | [Reference](https://github.com/einlang/einlang/blob/main/docs/reference.md) · [Stdlib](https://github.com/einlang/einlang/blob/main/docs/stdlib.md) |
| **ML practitioner** | [Install & run](https://github.com/einlang/einlang/blob/main/README.md#install--run) · [Examples: MNIST, ViT, Whisper](https://github.com/einlang/einlang/blob/main/README.md#examples) | [Stdlib: ML](https://github.com/einlang/einlang/blob/main/docs/stdlib.md) · [Reference](https://github.com/einlang/einlang/blob/main/docs/reference.md) |
| **Researcher** | Same as ML | [Paper & citation](https://github.com/einlang/einlang/blob/main/docs/PAPER.md) |
| **Engineer** | [Install & run](https://github.com/einlang/einlang/blob/main/README.md#install--run) · [Python API](https://github.com/einlang/einlang/blob/main/README.md#install--run) | [Reference](https://github.com/einlang/einlang/blob/main/docs/reference.md) · [Examples](https://github.com/einlang/einlang/blob/main/examples/README.md) |
| **Math-focused** | [Math intuition](https://github.com/einlang/einlang/blob/main/docs/MATH.md) — **math-intuitive**: equations and indices map directly to code (equations → Einlang) | [Reference](https://github.com/einlang/einlang/blob/main/docs/reference.md) · [Stdlib: math](https://github.com/einlang/einlang/blob/main/docs/stdlib.md) |
| **Autodiff / gradients** | **Built-in autodiff** — compiler derives derivatives and gradients from `@expr` and `@a / @b`; no hand-written gradient code. [Autodiff design](https://github.com/einlang/einlang/blob/main/docs/AUTODIFF_DESIGN.md) · [Pipeline](https://github.com/einlang/einlang/blob/main/docs/AUTODIFF_PIPELINE.md) · [Ops](https://github.com/einlang/einlang/blob/main/docs/AUTODIFF_OPS.md) | [examples/autodiff_small.ein](https://github.com/einlang/einlang/blob/main/examples/autodiff_small.ein), [autodiff_matmul.ein](https://github.com/einlang/einlang/blob/main/examples/autodiff_matmul.ein) · [Reference: Automatic differentiation](https://github.com/einlang/einlang/blob/main/docs/reference.md) |
| **Feature / language study** | [Reference](https://github.com/einlang/einlang/blob/main/docs/reference.md) (TOC) | [Design](https://github.com/einlang/einlang/blob/main/docs/DESIGN.md) |
| **Python user** | [Install & run](https://github.com/einlang/einlang/blob/main/README.md#install--run) — `run(source=...)` | [Syntax comparison](https://github.com/einlang/einlang/blob/main/docs/SYNTAX_COMPARISON.md) · [Reference](https://github.com/einlang/einlang/blob/main/docs/reference.md) · [Getting started](https://github.com/einlang/einlang/blob/main/docs/GETTING_STARTED.md) |
| **Julia user** | [Einlang for Julia programmers](https://github.com/einlang/einlang/blob/main/docs/EINLANG_FOR_JULIA_PROGRAMMERS.md) | [Julia demos → Einlang](https://github.com/einlang/einlang/blob/main/docs/JULIA_DEMOS.md) · [Syntax comparison](https://github.com/einlang/einlang/blob/main/docs/SYNTAX_COMPARISON.md) · [Examples](https://github.com/einlang/einlang/blob/main/examples/README.md) |
| **Rust user** | [Syntax comparison](https://github.com/einlang/einlang/blob/main/docs/SYNTAX_COMPARISON.md) · [Reference](https://github.com/einlang/einlang/blob/main/docs/reference.md) | [Getting started](https://github.com/einlang/einlang/blob/main/docs/GETTING_STARTED.md) · [Examples](https://github.com/einlang/einlang/blob/main/examples/README.md) |
| **Contributor** | [CONTRIBUTING](https://github.com/einlang/einlang/blob/main/CONTRIBUTING.md) | [DEVELOPMENT](https://github.com/einlang/einlang/blob/main/docs/DEVELOPMENT.md) |
| **Paper / citation** | [Paper & citation](https://github.com/einlang/einlang/blob/main/docs/PAPER.md) | — |

---

## Canonical docs

| Doc | What it is |
|-----|------------|
| [reference](https://github.com/einlang/einlang/blob/main/docs/reference.md) | Full language: syntax, types, Einstein notation, where-clauses, recurrences |
| [stdlib.md](https://github.com/einlang/einlang/blob/main/docs/stdlib.md) | All built-in modules and functions |
| [Install & run](https://github.com/einlang/einlang/blob/main/README.md#install--run) | Install, CLI, Python API (only place they're defined) |
| [Examples README](https://github.com/einlang/einlang/blob/main/examples/README.md) | Learning path and how to run examples |
| [GETTING_STARTED](https://github.com/einlang/einlang/blob/main/docs/GETTING_STARTED.md) | One-page story: what is Einlang → try → first example → use from Python → next |
| [SYNTAX_COMPARISON](https://github.com/einlang/einlang/blob/main/docs/SYNTAX_COMPARISON.md) | Python/NumPy, Julia, Rust → Einlang snippet mapping (links to reference) |
| [EINLANG_FOR_JULIA_PROGRAMMERS](https://github.com/einlang/einlang/blob/main/docs/EINLANG_FOR_JULIA_PROGRAMMERS.md) | Einlang for Julia programmers: indexing, ODEs/PDEs, recurrence, where to start |
| [JULIA_DEMOS](https://github.com/einlang/einlang/blob/main/docs/JULIA_DEMOS.md) | Julia demos and case studies with links; maps each to Einlang examples (simulation/ML overlap) |
| [MATH](https://github.com/einlang/einlang/blob/main/docs/MATH.md) | Math notation → Einlang (sums, index relations, guards, recurrences; links to reference) |
| [Autodiff](https://github.com/einlang/einlang/blob/main/docs/AUTODIFF_DESIGN.md) | **Built-in automatic differentiation**: `@expr` (differential), `@a / @b` (derivative); compiler derives gradients; design, pipeline, ops, examples |
| [UNSUPPORTED](https://github.com/einlang/einlang/blob/main/docs/UNSUPPORTED.md) | Syntax and features not supported by design, with rationale and alternatives |
| [RECURRENCE_ORDER_DESIGN](RECURRENCE_ORDER_DESIGN.md) | Recurrence iteration order (multiple recurrence dims); design options |
| [DEVELOPMENT](https://github.com/einlang/einlang/blob/main/docs/DEVELOPMENT.md) | For contributors: project layout, adding features |
| [FAQ](https://github.com/einlang/einlang/blob/main/docs/FAQ.md) | Common questions: run, learn, by background, where to ask |
| [DOCUMENTATION_DESIGN](https://github.com/einlang/einlang/blob/main/docs/DOCUMENTATION_DESIGN.md) | How this doc set is designed (for maintainers) |
| [LEARNING_FROM_JULIA](https://github.com/einlang/einlang/blob/main/docs/LEARNING_FROM_JULIA.md) | Lessons from Julia’s docs for showcase and retention (for maintainers) |
| [RELEASE_READINESS](https://github.com/einlang/einlang/blob/main/docs/RELEASE_READINESS.md) | Pre-release checklist (try-it, docs, examples, discoverability) |
