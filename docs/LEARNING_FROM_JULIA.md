# Learning from Julia: showcase and retention

What makes Julia’s docs and community successful at attracting and retaining users, and how Einlang can apply the same ideas without duplicating content.

**Important:** We should learn from Julia's **real applications** and production use — [JuliaHub case studies](https://juliahub.com/case-studies) (Aviva, Betterment, AOT, etc.), full QuantEcon workflows, SciML in practice, and ecosystem adoption — not only from short tutorials or minimal samples. Our examples are entry points; the goal is to align with the same class of problems and patterns that real Julia applications use.

---

## 1. What Julia does well

- **One “help” entry point** — [julialang.org/about/help/](https://julialang.org/about/help/) (“New to Julia?”): Getting Started, learning resources (video, Exercism, manual, books, Pluto), FAQ, “have a question?” (Discourse, Stack Overflow, chat), “want to contribute?”. One URL for “I need help” or “I’m new.”
- **Multiple learning pathways** — By style: “try it” (REPL), “by example,” “by manual,” “by book,” “by course.” By background: “Coming from MATLAB / R / Python / C.” Einlang already has “by background” (Python/Julia/Rust) and “by example” (learning path); we can make pathways explicit.
- **Problem-first showcases** — SciML, QuantEcon, JuMP: state the problem, then show code. Users see “this is my problem → this is how you do it.” Einlang examples (ODE, finance, optimization) already state the problem in comments; we surface that in the examples README and JULIA_DEMOS.
- **Clear “where next” after first success** — Getting Started flows into “Performance tips,” “Workflow,” package ecosystem. Einlang’s GETTING_STARTED “Where to go next” table does this; we can add one more step: “After your first example → pick a domain (simulation / ML / finance) → then reference.”
- **FAQ in one place** — Julia’s FAQ covers “What is Julia?”, installation, learning, community. A short FAQ reduces “where do I ask?” and “how do I X?” friction.
- **Community and contribution in the same breath as learning** — Help page links to Discourse, Stack Overflow, chat, and “contribute.” Einlang: CONTRIBUTING and “where to ask” (e.g. issues) should sit next to “getting started” in the help story.

---

## 2. Concrete recommendations for Einlang

| Julia tactic | Einlang action |
|--------------|----------------|
| One “help” entry point | **Docs index** is that entry: add a short “New to Einlang? / Need help?” block at the top of [docs/README.md](README.md) with: Getting started, Learning path, By background (Python/Julia/Rust), [FAQ](FAQ.md), Where to ask (CONTRIBUTING / issues). No new URL; one place in the doc index. |
| Multiple learning pathways | Make them explicit in the doc index and GETTING_STARTED: **by doing** (Try it → examples learning path), **by reading** (reference, stdlib), **by background** (Syntax comparison, EINLANG_FOR_JULIA_PROGRAMMERS). |
| Problem-first showcases | Keep and extend: each simulation/domain example states the problem first (in .ein comments and examples/README). JULIA_DEMOS and “What you get” tables stay problem/use-case oriented. |
| “Where next” after first run | GETTING_STARTED has a table and “After your first example” line: pick a domain ([ode](../../examples/ode/), [optimization](../../examples/optimization/), [finance](../../examples/finance/), [job_search](../../examples/job_search/), [time_series](../../examples/time_series/)) then [reference](reference.md) for depth. |
| FAQ in one place | Add **[docs/FAQ.md](FAQ.md)** with a few questions: How do I run a one-liner? Use from Python? Where are examples? I come from Julia/Python — where do I start? How do I report a bug? Link from “Need help?” and CONTRIBUTING. |
| Community + contribute | In “Need help?”: link to CONTRIBUTING and GitHub issues. README already has Community; doc index “Need help?” should point there too. |

---

## 3. What we don’t duplicate

- **No second “Install & run”** — Canonical remains README; GETTING_STARTED and FAQ link to it.
- **No second reference** — FAQ and help block only link to reference, stdlib, examples.
- **No long “Why Einlang”** — That stays in README and GETTING_STARTED; FAQ can have one sentence and a link.

---

## 4. Summary

- **Attract:** One clear “New to Einlang? / Need help?” in the doc index; learning pathways (by doing, by reading, by background); problem-first examples and JULIA_DEMOS.
- **Retain:** Clear “where next” after try-it and first example (domain examples → reference); FAQ for common questions; CONTRIBUTING and issues visible from the same help entry.

This doc is for maintainers. The user-facing result is: a better doc index (help + pathways + FAQ) and a short FAQ, with no new canonical content—only links and a single, small FAQ file.
