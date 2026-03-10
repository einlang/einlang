
# Frequently asked questions

Short answers; details live in the canonical docs (reference, stdlib, [Install & run](https://github.com/einlang/einlang/blob/main/README.md#install--run)).

---

## How do I run Einlang?

**One-liner:** `python3 -m einlang -c "let x = 1+1; print(x);"`  
**File:** `python3 -m einlang examples/hello.ein`  
**From Python:** `from einlang import run; run(file="examples/hello.ein")` or `run(source="...")`  

Full install and CLI options: [Install & run](https://github.com/einlang/einlang/blob/main/README.md#install--run).

---

## Where do I learn the language?

- **By doing:** [Try it](https://github.com/einlang/einlang/blob/main/README.md#try-it) → [Getting started](https://github.com/einlang/einlang/blob/main/docs/GETTING_STARTED.md) → [Learning path](https://github.com/einlang/einlang/blob/main/examples/README.md) (examples from basics to ML).
- **By reading:** [Language reference](https://github.com/einlang/einlang/blob/main/docs/reference.md) · [Standard library](https://github.com/einlang/einlang/blob/main/docs/stdlib.md).
- **By background:** [Syntax comparison](https://github.com/einlang/einlang/blob/main/docs/SYNTAX_COMPARISON.md) (Python/NumPy, Julia, Rust). Julia users: [Einlang for Julia programmers](https://github.com/einlang/einlang/blob/main/docs/EINLANG_FOR_JULIA_PROGRAMMERS.md) and [Julia demos → Einlang](https://github.com/einlang/einlang/blob/main/docs/JULIA_DEMOS.md).

---

## I come from Julia / Python — where do I start?

- **Julia:** [Einlang for Julia programmers](https://github.com/einlang/einlang/blob/main/docs/EINLANG_FOR_JULIA_PROGRAMMERS.md) → [JULIA_DEMOS](https://github.com/einlang/einlang/blob/main/docs/JULIA_DEMOS.md) → [examples](https://github.com/einlang/einlang/blob/main/examples/README.md).
- **Python:** [Install & run](https://github.com/einlang/einlang/blob/main/README.md#install--run) (`run(source=...)`), then [Syntax comparison](https://github.com/einlang/einlang/blob/main/docs/SYNTAX_COMPARISON.md) and [reference](https://github.com/einlang/einlang/blob/main/docs/reference.md).

---

## Where are the examples?

[examples/README](https://github.com/einlang/einlang/blob/main/examples/README.md) — learning path and list by domain (simulation, recurrence, finance, value_iteration, job_search, optimization, time_series, ML, etc.). Run from repo root: `python3 -m einlang examples/hello.ein`.

---

## How do I report a bug or ask something?

Open an [issue](https://github.com/einlang/einlang/issues). For contributing (docs, small fixes): [CONTRIBUTING](https://github.com/einlang/einlang/blob/main/CONTRIBUTING.md).
