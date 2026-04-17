# Learning from Julia: showcase and retention

What makes Julia’s docs and community successful at attracting and retaining users, and how Einlang can apply the same ideas without duplicating content.

**Important:** We should learn from Julia's **real applications** and production use — [JuliaHub case studies](https://juliahub.com/case-studies) (Aviva, Betterment, AOT, etc.), full QuantEcon workflows, SciML in practice, and ecosystem adoption — not only from short tutorials or minimal samples. Our examples are entry points; the goal is to align with the same class of problems and patterns that real Julia applications use.

---

## 1. What Julia does well

- **One “help” entry point** — [julialang.org/about/help/](https://julialang.org/about/help/) (“New to Julia?”): Getting Started, learning resources (video, Exercism, manual, books, Pluto), FAQ, “have a question?” (Discourse, Stack Overflow, chat), “want to contribute?”. One URL for “I need help” or “I’m new.”
- **Multiple learning pathways** — By style: “try it” (REPL), “by example,” “by manual,” “by book,” “by course.” By background: “Coming from MATLAB / R / Python / C.” Einlang already has “by background” (Python/Julia/Rust) and “by example” (learning path); we can make pathways explicit.
- **Problem-first showcases** — SciML, QuantEcon, JuMP: state the problem, then show code. Users see “this is my problem → this is how you do it.” Einlang examples (ODE, finance, optimization) already state the problem in comments; we surface that in the examples README and JULIA_DEMOS.
- **Clear “where next” after first success** — Getting Started flows into “Performance tips,” “Workflow,” package ecosystem. Einlang’s GETTING_STARTED “Where to go next” table does this; we can add one more step: “After your first example → pick a domain (simulation / ML / finance) → then reference.”
- **FAQ in one place** — Julia’s FAQ covers “What is Julia?”, installation, learning, community. A short quick-answers section reduces “where do I ask?” and “how do I X?” friction.
- **Community and contribution in the same breath as learning** — Help page links to Discourse, Stack Overflow, chat, and “contribute.” Einlang: CONTRIBUTING and “where to ask” (e.g. issues) should sit next to “getting started” in the help story.

---

## 2. Concrete recommendations for Einlang

| Julia tactic | Einlang action |
|--------------|----------------|
| One “help” entry point | **Docs index** is that entry: keep a short “Need help?” block at the top of [docs/README.md](https://github.com/einlang/einlang/blob/main/docs/README.md) with: Getting started, Try it, Learning path, By background (Python/Julia/Rust), quick answers, and where to ask (CONTRIBUTING / issues). No new URL; one place in the doc index. |
| Multiple learning pathways | Make them explicit in the doc index and GETTING_STARTED: **by doing** (Try it → examples learning path), **by reading** (reference, stdlib), **by background** (Syntax comparison, EINLANG_FOR_JULIA_PROGRAMMERS). |
| Problem-first showcases | Keep and extend: each simulation/domain example states the problem first (in .ein comments and examples/README). JULIA_DEMOS and “What you get” tables stay problem/use-case oriented. |
| “Where next” after first run | GETTING_STARTED has a table and “After your first example” line: pick a domain ([ode](https://github.com/einlang/einlang/tree/main/examples/ode), [optimization](https://github.com/einlang/einlang/tree/main/examples/optimization), [finance](https://github.com/einlang/einlang/tree/main/examples/finance), [job_search](https://github.com/einlang/einlang/tree/main/examples/job_search), [time_series](https://github.com/einlang/einlang/tree/main/examples/time_series)) then [reference](https://github.com/einlang/einlang/blob/main/docs/reference.md) for depth. |
| FAQ in one place | Keep the short Q&A in **[docs/README.md#quick-answers](https://github.com/einlang/einlang/blob/main/docs/README.md#quick-answers)** and leave **[docs/FAQ.md](https://github.com/einlang/einlang/blob/main/docs/FAQ.md)** as a redirect page to that section. Link to quick answers from “Need help?” and CONTRIBUTING-adjacent docs. |
| Community + contribute | In “Need help?”: link to CONTRIBUTING and GitHub issues. README already has Community; doc index “Need help?” should point there too. |

---

## 3. What we don’t duplicate

- **No second “Install & run”** — Canonical remains README; GETTING_STARTED and FAQ link to it.
- **No second reference** — FAQ and help block only link to reference, stdlib, examples.
- **No long “Why Einlang”** — That stays in README and GETTING_STARTED; FAQ can have one sentence and a link.

---

## 4. Summary

- **Attract:** One clear “Need help?” block in the doc index; learning pathways (by doing, by reading, by background); problem-first examples and JULIA_DEMOS.
- **Retain:** Clear “where next” after try-it and first example (domain examples → reference); quick answers for common questions; CONTRIBUTING and issues visible from the same help entry.

This doc is for maintainers. The user-facing result is: a better doc index (help + pathways + quick answers) and a short FAQ redirect page, with no new canonical content beyond routing and concise answers.
