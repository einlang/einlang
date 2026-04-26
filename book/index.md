---
layout: book
title: "Einlang: Formulas as Code"
description: "A book-length introduction to Einlang as formulas-as-code tensor programming."
---

# Einlang: Formulas as Code

A book-length treatment of Einlang as a tensor language where formulas become
executable source.

Each section starts from a mathematical formula, writes the corresponding
Einlang code, and then develops the abstraction behind it. The goal is not only
to show syntax. The goal is to build a model in the reader's head: what a name
means, how an index binds a family of values, when an axis disappears, why a
derivative can be an expression, and how recurrence gives the compiler enough
information to choose an evaluation order and a storage plan.

The book is constructive. A vector is not introduced as a library object; it is
introduced as a family of scalar expressions. Matrix multiplication is not
introduced as a primitive; it is introduced as two surviving indices and one
consumed index. Gradient descent is not introduced as mutation; it is introduced
as a recurrence over versions of a parameter. The language grows by composing
small ideas until familiar tensor programs fall out naturally.

It is written for a wider readership than a formal textbook alone. A student
can read it in sequence. A working engineer can drop into the chapter on
autodiff, recurrence, or Python boundaries. A language designer can read it as
an argument about notation and compiler-visible structure. A researcher can use
it as a compact statement of the design space around explicit tensor syntax.

This book tracks the current project reality:

- statements end with semicolons;
- ranges use `a..b` for an exclusive end and `a..=b` for an inclusive end;
- named rest patterns use `..batch`;
- reduction syntax currently includes `sum`, `max`, `min`, and `prod`;
- whole-array mean and many ML operations live in the standard library;
- Python interop uses paths such as `python::numpy::load(...)`;
- `@y / @x` works on named bindings and may be represented lazily for tensor
  derivatives;
- recurrence storage optimization is a compiler/runtime consequence of visible
  dependency offsets, with conservative full materialization when needed.

## Contents

### Front Matter

{% for entry in site.data.book %}
{% if entry.kind == "frontmatter" and entry.url != "/book/" %}
- [{{ entry.title }}]({{ entry.url | relative_url }})
{% endif %}
{% endfor %}

### Chapters

{% for entry in site.data.book %}
{% if entry.kind == "chapter" %}
- [{{ entry.title }}]({{ entry.url | relative_url }})
{% endif %}
{% endfor %}

### Back Matter

{% for entry in site.data.book %}
{% if entry.kind == "backmatter" %}
- [{{ entry.title }}]({{ entry.url | relative_url }})
{% endif %}
{% endfor %}

## Reading Paths

Read the book in order the first time if you want the full conceptual arc. The
first three chapters build the tensor notation. The fourth chapter adds
derivative requests. The fifth chapter adds recurrence and storage analysis.
The final two chapters explain how Einlang stays small by cooperating with
Python and refusing unnecessary generality.

Readers arriving with a specific agenda can also enter sideways:

- start with Chapter 4 if you care most about differentiable programming;
- start with Chapter 5 if you care most about recurrence, scheduling, or
  dynamic programs;
- start with Chapter 6 if you want the practical Einlang/Python boundary first;
- read Chapter 7 as the design-philosophy capstone after any earlier chapter.

Most examples are fragments: names such as `A`, `x`, `logits`, or `W` may be
provided by earlier bindings, a host program, a Python loader, or a test fixture.
When a fragment needs a specific boundary behavior, the text says so directly.

## Abstraction Path

The book follows one thread: make structure visible.

- Scalar binding gives stable names.
- Indexed binding turns one expression into a whole family of values.
- Reduction introduces local binders that consume axes.
- Missing indices explain broadcasting without an additional broadcasting API.
- Rest patterns let the same equation survive rank changes.
- Derivative requests turn sensitivity questions into source expressions.
- Recurrence turns state evolution into a dependency graph.
- Python interop keeps the language small by moving non-tensor work out of the
  kernel.

The intended reading style is slow, but not narrow. Each section gives a
formula, a program, and a way to read the program operationally and
denotationally. The book uses worked readings of small programs to show how the
same mental model scales from introductory examples to design questions about
languages, runtimes, and boundaries.
