# Frequently asked questions

Short answers; details live in the canonical docs (reference, stdlib, [Install & run](../../README.md#install--run)).

---

## How do I run Einlang?

**One-liner:** `python3 -m einlang -c "let x = 1+1; print(x);"`  
**File:** `python3 -m einlang examples/hello.ein`  
**From Python:** `from einlang import run; run(file="examples/hello.ein")` or `run(source="...")`  

Full install and CLI options: [Install & run](../../README.md#install--run).

---

## Where do I learn the language?

- **By doing:** [Try it](../../README.md#try-it) → [Getting started](GETTING_STARTED.md) → [Learning path](../../examples/README.md) (examples from basics to ML).
- **By reading:** [Language reference](reference.md) · [Standard library](stdlib.md).
- **By background:** [Syntax comparison](SYNTAX_COMPARISON.md) (Python/NumPy, Julia, Rust). Julia users: [Einlang for Julia programmers](EINLANG_FOR_JULIA_PROGRAMMERS.md) and [Julia demos → Einlang](JULIA_DEMOS.md).

---

## I come from Julia / Python — where do I start?

- **Julia:** [Einlang for Julia programmers](EINLANG_FOR_JULIA_PROGRAMMERS.md) → [JULIA_DEMOS](JULIA_DEMOS.md) → [examples](../../examples/README.md).
- **Python:** [Install & run](../../README.md#install--run) (`run(source=...)`), then [Syntax comparison](SYNTAX_COMPARISON.md) and [reference](reference.md).

---

## Where are the examples?

[examples/README](../../examples/README.md) — learning path and list by domain (simulation, recurrence, finance, value_iteration, job_search, optimization, time_series, ML, etc.). Run from repo root: `python3 -m einlang examples/hello.ein`.

---

## How do I report a bug or ask something?

Open an [issue](https://github.com/einlang/einlang/issues). For contributing (docs, small fixes): [CONTRIBUTING](../../CONTRIBUTING.md).
