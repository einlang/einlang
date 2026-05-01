---
layout: book
title: "Preface"
---

# Preface

Einlang started with a small annoyance I kept running into while reading tensor
programs. The code would run, the shapes would line up, and still the most
important part of the program was somewhere else: in a comment, in a variable
name, in a convention about axis order, or simply in the author's memory.

Here is the sort of question I mean. A tensor is reshaped, transposed,
broadcast, reduced, and passed to an automatic differentiator. Which dimension
is batch? Which one is time? Which one was split into groups? Which value is
being reused because it does not depend on the current coordinate? These are
not exotic concerns. They are ordinary facts that decide whether the program
means what the author thinks it means.

Most tensor libraries are good at checking extents. They can tell whether two
arrays have compatible shapes. They are less able to check which role a
dimension was supposed to play. A dimension of size `64` might be a feature axis, a time
axis, a hidden-state axis, or a batch axis. The number alone is not the role.

This book asks what changes if some of those roles are written into the source.
Not as comments beside the program, but as indices and coordinate relationships
the compiler can inspect.

Einlang is the small language used for that question. It is not presented here
as a replacement for Python, NumPy, PyTorch, or existing compiler stacks. It is
a deliberately narrow object: enough syntax to write tensor formulas with named
coordinates, reductions, derivative requests, and recurrences, and enough
compiler machinery to see what those formulas preserve.

The book does not ask every useful operation to stay expanded forever. A common
operation can be a coordinate function: `softmax[class]`, `argmax[class]`,
and similar calls are meant to hide stable mechanics while keeping the
coordinate contract at the boundary. That compromise matters throughout the
argument. Without it, visible coordinates would become a verbose teaching
notation instead of a usable programming model.

The intended reader is someone who builds things below the level of an API call:
a DSL, a compiler pass, a numerical library, an autodiff system, a model
compiler, or even just a careful notation shared by a team. If you mainly want
another framework function to call, this manuscript will probably spend too much
time looking under the floorboards. That is by design. The interesting question
here is not how to spell one operation, but what facts a language lets later
tools rely on.

## Reader Map

There are several reasonable ways through the book. If you are here for tensor
notation and shape bugs, read chapters 1 through 6 first. If automatic
differentiation is your main concern, you can skim the first six chapters for
the vocabulary and then slow down in chapters 7 through 9. If recurrence,
storage, or model compilers are the draw, chapters 10 through 12 are the
shortest path. If you are designing framework boundaries, diagnostics, or
library APIs, chapters 13 through 15 are the payoff. If you want to see the
principle stressed by low-rank attention and dynamic expert routing, chapter 16
is the late test case. The chapters are ordered as an argument, but they are
also a toolbox; enter where the next hidden axis is already hurting you.

One inconvenience should be stated early. The book uses extra source text where
meaning is fragile. That is the bargain: more local facts, fewer silent
conventions for a future reader or compiler to reconstruct.

The chapters follow a recurring pattern. They begin with a piece of tensor code
that feels familiar. Then they ask which fact has been left implicit. Sometimes
the missing fact is an axis role. Sometimes it is the coordinate a term does
not depend on. Sometimes it is the direction of a recurrence, the address of a
gradient, or the contract at a boundary with host-language data.

The examples are small because small examples leave fewer places to hide. A
single cell of a matrix product can show the same issue that later appears in a
batched layer. One recurrent edge can show the distinction between a loop and a
dependency graph. One attention score can show why query and key positions
should not be allowed to collapse into "the sequence axis."

There is one rule I return to more than any other:

```text
Do not hide a fact that later reasoning must recover.
```

That rule is not a demand to make every program verbose. Good notation hides
plenty. It should hide details that are accidental, not details that decide
correctness. Layout, scheduling, storage policy, and kernel choice may be
lowering decisions. Axis roles, consumed indices, omitted coordinates,
derivative addresses, and time dependencies often have to be known before those
decisions are safe.

A small vocabulary point matters throughout the book. Axis names are not scalar
types. Einlang has ordinary scalar and tensor types such as `i32`, `f32`,
`[f32; 3, 4]`, `[f32; ?, ?]`, and `[f32; *]`. Index variables are coordinates,
introduced by indexed bindings, reductions, or ranges such as `i in 0..n`.
The notation does not write `i: batch`. When I say that an axis behaves like a
contract, I mean that its name, range, and tensor shape give later compiler
passes a fact they can check.

The implementation is included to keep the prose honest. The compiler parses
source, resolves names, lowers formulas to IR, groups Einstein-style
declarations, classifies constraints, infers ranges, checks shapes, infers
types, rewrites autodiff requests, lowers recurrences, records execution facts,
and validates the result. The book does not document every corner of that
implementation, but it tries not to claim a property the implementation has no
way to represent.

The larger claim is modest: tensor code often contains more structure than its
surface syntax admits. Naming coordinates is one way to keep some of that
structure available long enough for a reader, a checker, or a compiler pass to
use it.

This book will not make you write less code. It asks for something stricter:
write lines that can survive being questioned.
