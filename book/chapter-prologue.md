---
layout: book
title: "Prologue: The Error That Didn't Error"
---

# Prologue: The Error That Didn't Error

> "We shall not cease from exploration. And the end of all our exploring will be to arrive where we started and know the place for the first time."
>
> — T. S. Eliot

Here is a story about a bug.

It is not a dramatic bug. It produces no stack trace, no NaN cascade, no `CUDA error: device-side assert triggered`. It does not crash the training run. It does not even make the loss go up.

It is worse than all of those things. It makes the loss go down, smoothly and convincingly, while the program learns the wrong thing.

The tensor has shape `(32, 64, 256)`. A human being—the one who wrote the data loader—knows that these three dimensions are `batch`, `channel`, and `spatial`. The human wrote a comment. The human chose a variable name: `spatial_features`. The human did everything a responsible programmer does.

Then the human wrote:

```python
x = x.mean(dim=1)
```

`dim=1` erases a dimension. Which dimension? At the time of writing, position 1 held `channel`. The operation's *intent* was "average over channels." But the operation's *text* says nothing about channels. It says `dim=1`. A position. A number.

Three months pass. Another human—or the same human, after enough context has drained from memory—refactors the data pipeline. Channel moves to position 2. The new shape is `(32, 256, 64)`. `mean(dim=1)` now silently erases `spatial`.

Shape check: pass. Type check: pass. Unit tests: pass. Integration tests: pass. The loss descends. The eval metrics look normal. The model deployed on Tuesday. The customer complaint arrived on Thursday.

This bug lived for three weeks because **the notation had no slot for the fact that would have caught it.** The fact—"I am erasing `channel`, not `spatial`"—was present in a comment, in a variable name, in the author's mental model. It was absent from the one place the compiler could see: the source text of the operation itself.

Positional notation is not *wrong*. It is *insufficient*. It records the arithmetic of shapes. It does not record the identity of coordinates. When those two things diverge—when a shape is correct but a coordinate is wrong—positional notation gives you no place to notice.

This book is about closing that gap.

---

The book you are holding is organized around a single pedagogical debt, borrowed from the opening pages of *Structure and Interpretation of Computer Programs*. A powerful language, SICP argues, rests on three things:

- **Primitive expressions**—the simplest things the language can say.
- **Means of combination**—how simple things are composed into complex ones.
- **Means of abstraction**—how complex things are named and reused as if they were primitive.

Every chapter is anchored to one of these three layers. The primitives occupy Chapters 1 through 5. Combination occupies Chapters 6 through 9. Abstraction occupies Chapters 10 and 11. The remaining chapters apply, formalize, and stress-test what you've built.

A second structure runs beneath these layers. Starting in Chapter 3, we introduce the syntax of a small language called **einlang**—a tensor notation where coordinate names live in brackets, where every reduction states which coordinate it consumes, and where the compiler can check the facts that positional notation leaves to comments. The syntax appears piece by piece, exactly when the concept it serves has earned its introduction. By Chapter 6 the language has a name. By Chapter 10 you can build your own abstractions in it. By Chapter 14 you can see the whole grammar in one place.

But einlang is not the argument. It is the microscope.

The argument is a set of four questions you can ask about any tensor expression, in any framework, that will tell you whether the notation preserved the facts that correctness depends on:

1. **Which coordinate does this operation eliminate?** Is the name still in the code?
2. **Which coordinate does this operation copy along?** Is the copy explicit, or hidden by a broadcasting rule?
3. **Where did this coordinate come from, and where is it going?** Can you trace the permutation without reconstructing it from position numbers?
4. **The forward pass eliminated a coordinate—how does the backward gradient handle it?** Are the two directions symmetric?

These four questions—**eliminate with a name, copy with a signature, permute with a source, forward and backward symmetric**—are the coordinate habit. They work in any framework because they are questions about *meaning*, not about *syntax*. They cost nothing to ask. They catch the class of bugs that shape checks cannot.

Chapters 12, 15, and 16 exist to make these questions automatic.

---

A word about what this book is not.

It is not a language reference. The appendices and the documentation serve that purpose. It is not a production migration guide. Einlang is a young language. It is not a neutral survey of tensor notations. Every chapter takes a side: when a coordinate's identity determines correctness, the source code should be able to state it.

The book is an argument in the form of sixteen chapters, two appendices, and one idea. The idea was stated two pages ago. Everything that follows is evidence.

Let's begin.
