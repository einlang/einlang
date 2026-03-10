
# Einlang documentation design

High-level design: one source of truth per topic, no large duplication, clear entry points per audience.

---

## 1. Principles

- **Single source of truth** — Each piece of content lives in exactly one place. Other docs link to it; they do not copy it.
- **Audience-first entry** — The doc index (and README) route people by *who they are* (starter, ML, Python user, contributor, …) to the right starting link. Each audience gets a short path, not one giant doc.
- **Canonical docs are reference-only** — The language reference has all syntax/semantics; the stdlib doc has all APIs; the README has the only “Install & run” section. Narrative and “what to do next” live in thin docs that only link.
- **No duplicate blobs** — “What is Einlang” appears in short form in README and in GETTING_STARTED only. Try-it commands: canonical in README; GETTING_STARTED may link or repeat once. No third copy of reference content.

---

## 2. Canonical docs (single source of truth)

| Doc | Owns | Must not duplicate |
|-----|------|--------------------|
| **README.md** | Project landing, tagline, one code block, **Try it** (commands), **What's next** (table), **Install & run** (install, CLI, Python API, compiler API), short “What you get” / “Why different” / “Why not NumPy”, **Examples** (tables + link to examples/README), **Docs** (link to doc index), Community, License. | No syntax/semantics; no full API lists. |
| **docs/reference.md** | All language semantics: statements, types, expressions, Einstein notation, where-clauses, recurrences, comprehensions, modules, etc. | No “what is Einlang” beyond one line; no install; no example lists. |
| **docs/stdlib.md** | All standard library modules and functions. | No language syntax; no install. |
| **examples/README.md** | Learning path (ordered list of example dirs + one-line description), by-feature index (optional), “how to run” from repo root. | No syntax; no API; links to reference/stdlib for concepts. |
| **CONTRIBUTING.md** | How to contribute (permission, get going, good first contributions, tests, where to ask, code style). | Links to DEVELOPMENT, GETTING_STARTED; no project layout. |
| **docs/DEVELOPMENT.md** | Project layout, adding a language feature, adding a stdlib function, error system. | For contributors only; no language reference. |

**Install & run** is only in README. GETTING_STARTED and any audience page link to `README.md#install--run` (or show one minimal snippet and “full details: Install & run”).

---

## 3. Narrative / journey docs (thin, link-heavy)

| Doc | Purpose | May contain | Must not contain |
|-----|---------|------------|------------------|
| **docs/GETTING_STARTED.md** | One-page story: what is Einlang (2–3 sentences + one code block), try it (link to README#try-it or 4 commands once), first example (hello.ein), use from Python (link to Install & run or one snippet), “where next” table (links only). | Minimal “what is”, try commands (once), one Python snippet, links. | No full reference; no full stdlib; no second copy of Install & run section. |
| **docs/README.md** | **Doc index by audience.** Table/sections: Starter, Student, User, ML, Researcher, Engineer, Math, Python/Julia/Rust, Feature/language study, Contributor, Paper. Each row: audience → “Start here” link(s) → “Then” link(s). All links; no copied reference/stdlib content. | Optional 1–2 sentence intro per audience. | No syntax; no API; no install instructions. |

---

## 4. Audience matrix (who uses what)

Entry points and paths. **Start** = first link(s); **Then** = next link(s). All point into canonical docs.

| Audience | Start here | Then |
|----------|------------|------|
| **Starter** | [Getting started](https://github.com/einlang/einlang/blob/main/docs/GETTING_STARTED.md) | [Try it](https://github.com/einlang/einlang/blob/main/README.md#try-it) → [Learning path](https://github.com/einlang/einlang/blob/main/examples/README.md) |
| **Student** | [Getting started](https://github.com/einlang/einlang/blob/main/docs/GETTING_STARTED.md) | [Learning path](https://github.com/einlang/einlang/blob/main/examples/README.md) → [Reference](https://github.com/einlang/einlang/blob/main/docs/reference.md) |
| **User (any)** | [Try it](https://github.com/einlang/einlang/blob/main/README.md#try-it) · [Install & run](https://github.com/einlang/einlang/blob/main/README.md#install--run) | [Reference](https://github.com/einlang/einlang/blob/main/docs/reference.md) · [Stdlib](https://github.com/einlang/einlang/blob/main/docs/stdlib.md) |
| **ML practitioner** | [Install & run](https://github.com/einlang/einlang/blob/main/README.md#install--run) · [Examples: MNIST, ViT, Whisper](https://github.com/einlang/einlang/blob/main/README.md#examples) | [Stdlib: ML](https://github.com/einlang/einlang/blob/main/docs/stdlib.md) · [Reference](https://github.com/einlang/einlang/blob/main/docs/reference.md) |
| **Researcher** | Same as ML | + [Paper & citation](https://github.com/einlang/einlang/blob/main/docs/PAPER.md) if applicable |
| **Engineer** | [Install & run](https://github.com/einlang/einlang/blob/main/README.md#install--run) · [Python API](https://github.com/einlang/einlang/blob/main/README.md#install--run) | [Reference](https://github.com/einlang/einlang/blob/main/docs/reference.md) · [Examples](https://github.com/einlang/einlang/blob/main/examples/README.md) |
| **Math-focused** | [MATH](https://github.com/einlang/einlang/blob/main/docs/MATH.md) (equations → Einlang) | [Reference](https://github.com/einlang/einlang/blob/main/docs/reference.md) · [Stdlib: math](https://github.com/einlang/einlang/blob/main/docs/stdlib.md) |
| **Feature / language study** | [Reference](https://github.com/einlang/einlang/blob/main/docs/reference.md) (TOC) | [Design](https://github.com/einlang/einlang/blob/main/docs/DESIGN.md) if present |
| **Python user** | [Install & run](https://github.com/einlang/einlang/blob/main/README.md#install--run) — `run(source=...)` | [Syntax comparison](https://github.com/einlang/einlang/blob/main/docs/SYNTAX_COMPARISON.md) · [Reference](https://github.com/einlang/einlang/blob/main/docs/reference.md) · [Getting started](https://github.com/einlang/einlang/blob/main/docs/GETTING_STARTED.md) |
| **Julia user** | [Syntax comparison](https://github.com/einlang/einlang/blob/main/docs/SYNTAX_COMPARISON.md) · [Reference](https://github.com/einlang/einlang/blob/main/docs/reference.md) | [Getting started](https://github.com/einlang/einlang/blob/main/docs/GETTING_STARTED.md) · [Examples](https://github.com/einlang/einlang/blob/main/examples/README.md) |
| **Rust user** | [Syntax comparison](https://github.com/einlang/einlang/blob/main/docs/SYNTAX_COMPARISON.md) · [Reference](https://github.com/einlang/einlang/blob/main/docs/reference.md) | [Getting started](https://github.com/einlang/einlang/blob/main/docs/GETTING_STARTED.md) · [Examples](https://github.com/einlang/einlang/blob/main/examples/README.md) |
| **Contributor** | [CONTRIBUTING](https://github.com/einlang/einlang/blob/main/CONTRIBUTING.md) | [DEVELOPMENT](https://github.com/einlang/einlang/blob/main/docs/DEVELOPMENT.md) |
| **Paper / citation** | [PAPER](https://github.com/einlang/einlang/blob/main/docs/PAPER.md) | — |

---

## 5. Optional docs (no duplication)

| Doc | Purpose | Content rule |
|-----|---------|--------------|
| **docs/PAPER.md** | Citation, “how to describe Einlang” in a paper, repo URL, license. | One short page; no syntax copy. |
| **docs/DESIGN.md** | Design goals, influences, tradeoffs (non-normative). | Links to reference for semantics; no syntax duplication. |
| **docs/SYNTAX_COMPARISON.md** | Map "in Python/NumPy / Julia / Rust you write X → in Einlang you write Y". | Snippet mappings only; every Einlang construct links to reference. No duplication of semantics. |
| **docs/MATH.md** | Map "math notation (Σ, indices, recurrences) → Einlang". | Equation ↔ code only; links to reference. No duplication of semantics. |
| **docs/UNSUPPORTED.md** | List syntax/features not supported by design; rationale; what to use instead. | Links to reference for alternatives. Single place for "we don't support X". |
| **docs/audience/*.md** | Optional per-audience page (e.g. `ml.md`, `from-python.md`). | 2–3 sentences + bullet list of links only. No reference/stdlib copy. |

If we do *not* add audience/*.md files, the audience matrix in docs/README.md is enough: one table, one place to maintain.

---

## 6. File layout (target)

```
README.md                     # Canonical: landing, try, what's next, install & run, features, examples, docs link
CONTRIBUTING.md               # Canonical: how to contribute
docs/
  README.md                   # Doc index by audience (table)
  DOCUMENTATION_DESIGN.md      # This file — design only
  GETTING_STARTED.md           # Narrative: what is, try, first example, use from Python, where next
  SYNTAX_COMPARISON.md         # Optional: Python/Julia/Rust → Einlang snippet mapping (link-only to reference)
  MATH.md                     # Optional: math notation → Einlang (link-only to reference)
  UNSUPPORTED.md              # Optional: not supported by design + rationale + alternatives
  reference.md                # Canonical: full language reference
  stdlib.md                   # Canonical: full standard library
  DEVELOPMENT.md              # Canonical: contributor project layout, add feature, errors
  PAPER.md                    # Optional: citation, description for papers
  DESIGN.md                   # Optional: design goals, influences (link-only to reference)
  audience/                   # Optional: one short file per audience (links only)
    (e.g. ml.md, from-python.md, …)
examples/
  README.md                   # Canonical: learning path, by-feature, run instructions
  hello.ein, basics/, demos/, mnist/, …
```

---

## 7. What each audience page may contain (if we add audience/)

- **starter.md** — “You’re new. 1) [Getting started]. 2) [Try it]. 3) [Learning path].” (links only)
- **ml.md** — “You do ML. [Install & run], [Examples: MNIST, ViT, Whisper], [Stdlib: ML], [Reference].” (links only)
- **from-python.md** — “You use Python. Einlang runs via `run(source=...)`; [Install & run]. Syntax: [Reference]. [Examples].” (links only)
- **from-julia.md** — “You use Julia. Similarities and differences: [Reference]. [Getting started], [Examples].” (links only)
- **from-rust.md** — “You use Rust. Syntax is Rust-inspired; [Reference]. [Getting started], [Examples].” (links only)
- **paper.md** → moved to docs/PAPER.md (single file for citation).

---

## 8. Summary

- **Canonical:** README (install, try, examples index), reference (language), stdlib (API), examples/README (learning path), CONTRIBUTING, DEVELOPMENT.
- **Thin narrative:** GETTING_STARTED (one-page story), docs/README (audience index). They only link; they don’t duplicate.
- **Audience:** Served by one table in docs/README (and optionally by short audience/*.md files that only link).
- **Optional:** PAPER.md (citation), DESIGN.md (design notes), audience/*.md (per-audience links).

This keeps a single source of truth, avoids large duplicates, and gives starters, experts, users, contributors, ML, math, researchers, students, engineers, paper writers, feature/language study, and Python/Julia/Rust users a clear entry point and path.
