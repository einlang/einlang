---
layout: book
title: "Epilogue · A Friend Named einlang"
---

# Epilogue · A Friend Named einlang

> "Programs must be written for people to read, and only incidentally for machines to execute."
>
> — Harold Abelson and Gerald Jay Sussman, *Structure and Interpretation of Computer Programs*

---

In Chapter 5, einlang was a name. A label for the notation we'd been building since Chapter 3. Now, at the end of fifteen chapters, you know what the name actually refers to.

Einlang is not a language in the sense that Python is a language, or C++, or Rust. It does not aspire to run your web server or render your UI. It is a language built on three ideas: primitive expressions, means of combination, and means of abstraction—organized around a single purpose.

The idea is that **coordinates have identities, and those identities belong in the source code.**

Everything else in the language serves that idea. Reductions name the coordinate they consume. Broadcasts name the coordinate they replicate along. Permutations state the coordinate correspondence explicitly. Functions declare which coordinates they use by identity, and the compiler checks those declarations at every call site. Gradients preserve coordinate structure through differentiation. Recurrences make the direction of time a syntactic constraint.

---

The SICP quotation that opens this epilogue is famous for a reason. It states a truth that is obvious once you hear it and difficult to practice consistently: code is communication between humans before it is instruction to machines.

The coordinate habit is an application of that truth to tensor programming. When you write `x.mean(dim=1)`, you are communicating to the machine ("reduce axis 1") but not to the human ("eliminate channel"). When you write `mean[channel](x)`, you are communicating to both. The machine still knows what to do. The human now also knows what you intended. And the compiler can check that your intent is consistent with the tensor's actual coordinate structure.

This is the bet the book has asked you to make: that the extra keystrokes of naming coordinates are repaid, with interest, in debugging sessions avoided, in refactoring confidence gained, in the quiet satisfaction of reading code that says what it means.

---

SICP ends with a meta-circular evaluator—a Scheme interpreter written in Scheme—as if to say: you now understand the language deeply enough to implement it yourself.

I am not going to ask you to implement einlang. I am going to ask you something smaller and harder.

Take the four habits into your next tensor program. Not an einlang program—the language is young, the tooling is sparse, and you have deadlines. Take them into PyTorch. Into JAX. Into whatever framework you use to get work done.

When you write a reduction, pause. Ask: which coordinate am I eliminating? Is the name in the code?

When you write a broadcast, pause. Ask: which coordinate am I copying along? Is independence genuinely justified?

When you write a permutation, pause. Ask: can I trace one coordinate from source to destination without reconstructing the position map?

When you inspect a gradient, pause. Ask: does the backward reduction match the forward broadcast?

These questions cost seconds. The bugs they catch cost hours. The ratio is favorable.

---

## How to Start

You don't need einlang to practice the coordinate habit. You need a place to put a name and a discipline to keep it honest.

Start small. Name one coordinate at a time. The data-entry boundary is the most important one—if coordinates are named when tensors enter the program, the names flow downstream. Name the reductions next—they are where coordinates are consumed, and the consumption is the hardest fact to reconstruct later. Name the broadcasts last—they are often implicit, and making them explicit is the most verbose change.

In PyTorch, a comment is your first bridge: `x.mean(dim=1)  # dim 1 = channel`. In JAX, einops patterns are your bridge: `rearrange(x, "batch channel spatial -> batch spatial channel")`. The bridge doesn't have to be perfect. It has to be there.

The goal is not to convert your entire codebase to named dimensions overnight. The goal is to develop the reflex: when you write an operation that depends on a coordinate's identity, put that identity in the source. Not in your head. Not in a Slack message. In the source.

---

There is a sentence I have been saving for the end. It is the one-sentence version of this book:

**Let your coordinate names be as visible as your intent—because six months from now, when you return to this code to fix a bug, the intent will be gone, and only the names will remain.**

Thank you for reading.
