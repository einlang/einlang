---
layout: book
title: "For Instructors"
description: "A teaching guide for using Einlang: Formulas as Code in courses, reading groups, or independent study."
---

# For Instructors

This book can support several teaching contexts: a short unit inside a
programming languages course, a tensor-programming seminar, an independent
reading group, or a methods companion in a machine learning class. Its central
promise is not coverage of every tensor API. Its value is conceptual
compression: students learn one reading discipline that explains names, axes,
reductions, derivatives, and recurrences with the same vocabulary.

## Where It Fits

The strongest course pairings are:

- programming languages, especially courses on notation, semantics, or DSLs;
- scientific computing, where students already know arrays but not
  source-visible tensor structure;
- machine learning systems, where autodiff and tensor kernels are important but
  often treated as opaque infrastructure;
- independent projects on compilers, tensor runtimes, or differentiable
  programming.

## Suggested Sequence

For a compact 4-week module:

1. Chapter 1 and Chapter 2: names, indices, and reduction.
2. Chapter 3: broadcasting, transposition, and convolution patterns.
3. Chapter 4 and Chapter 5: autodiff plus recurrence as dependency analysis.
4. Chapter 6 and Chapter 7: Python boundaries, language scope, and design
   trade-offs.

For a full semester reading path, assign one chapter per week and use the
appendices as a running reference sheet rather than required prose.

## What Students Should Produce

Good evidence of mastery includes:

- a line-by-line reading of an Einlang fragment in plain English;
- shape and dependency analysis of a new indexed program;
- translation of a mathematical formula into Einlang and back;
- a short memo defending one language-design choice in the small core;
- a final project that implements, lowers, or critiques a tensor pattern.

## Assignment Types

Use a mix of short and long forms:

- reading checks: ask which indices survive, which are consumed, and which
  shapes are forced by a fragment;
- rewrite exercises: convert loop-oriented pseudocode into indexed equations;
- trace exercises: follow a derivative request or recurrence dependency through
  named intermediate values;
- design prompts: ask students where Python should remain the host and where
  Einlang should carry the semantics.

## Assessment Lens

The book works best when assessment rewards explanation, not only final code.
Students should be able to say why a definition has a given rank, why a
reduction is legal, why a recurrence can be scheduled, and why a boundary cast
belongs in the source. In other words, the text is successful when students can
read programs as structured mathematical arguments.
