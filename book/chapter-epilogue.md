---
layout: book
title: "Epilogue · A Friend Named Einlang"
---

# Epilogue · A Friend Named Einlang

> "Programs must be written for people to read, and only incidentally for machines to execute."
>
> — Harold Abelson and Gerald Jay Sussman, *Structure and Interpretation of Computer Programs*

---

In Chapter 5, Einlang was a name. A label for the notation we'd been building since Chapter 3. Now, at the end of sixteen chapters, you know what the name actually refers to.

Einlang is not a language in the sense that Python is a language, or C++, or Rust. It does not aspire to run your web server or render your UI. It is a language built on three ideas: primitive expressions, means of combination, and means of abstraction—organized around a single purpose.

The idea is that **coordinates have identities, and those identities belong in the source code.**

Everything else in the language serves that idea. Reductions name the coordinate they consume. Broadcasts name the coordinate they replicate along. Permutations state the coordinate correspondence explicitly. Functions declare which coordinates they use by identity, and the compiler checks those declarations at every call site. Gradients preserve coordinate structure through differentiation. Recurrences make the direction of time a syntactic constraint.

---

The SICP quotation that opens this epilogue is famous for a reason. It states a truth that is obvious once you hear it and difficult to practice consistently: code is communication between humans before it is instruction to machines.

The coordinate habit is an application of that truth to tensor programming. When you write `x.mean(dim=1)`, you are communicating to the machine ("reduce axis 1") but not to the human ("eliminate channel"). When you write `mean[channel](x)`, you are communicating to both. The machine still knows what to do. The human now also knows what you intended. And the compiler can check that your intent is consistent with the tensor's actual coordinate structure.

This is the bet the book has asked you to make: that the extra keystrokes of naming coordinates are repaid, with interest, in debugging sessions avoided, in refactoring confidence gained, in the quiet satisfaction of reading code that says what it means.

---

SICP ends with a meta-circular evaluator—a Scheme interpreter written in Scheme—as if to say: you now understand the language deeply enough to implement it yourself.

I am not going to ask you to implement Einlang. I am going to ask you something smaller and harder.

Take the four habits into your next tensor program. Not an Einlang program—the language is young, the tooling is sparse, and you have deadlines. Take them into PyTorch. Into JAX. Into whatever framework you use to get work done.

When you write a reduction, pause. Ask: which coordinate am I eliminating? Is the name in the code?

When you write a broadcast, pause. Ask: which coordinate am I copying along? Is independence genuinely justified?

When you write a permutation, pause. Ask: can I trace one coordinate from source to destination without reconstructing the position map?

When you inspect a gradient, pause. Ask: does the backward reduction match the forward broadcast?

These questions cost seconds. The bugs they catch cost hours. The ratio is favorable.

---

You are about to close this book.

Maybe you're at a desk. Maybe on a train. Maybe it's late and you're reading by a screen's glow. Wherever you are, there is a moment—right now, or in five minutes, or tomorrow morning—when you will look up from this page and return to the code you were writing before you opened the book.

What will be different?

Not the framework. You're still using PyTorch, or JAX, or NumPy. Not the deadline. It hasn't moved. Not the model architecture. The layers are the same. The loss function is the same. The optimizer is the same.

But something in how you read has changed.

You will type `x.mean(dim=1)` and pause. Not because the line is wrong—because you now notice that `dim=1` is a number, and you know which coordinate it refers to, and you wonder whether the next person to read this line will know too.

You will write a broadcast and think: *which coordinate am I silent on?* Not because the framework requires you to answer—because you now know that the silence is a claim, and claims should be checkable.

You will trace a bug through a reshape-permute-reshape chain and think: *if these dimensions had names, this chain would be three lines instead of fifteen, and the bug would have been caught before runtime.*

You will read attention code and notice whether `seq_q` and `seq_k` are the same coordinate or different coordinates, because you now know that when they happen to have the same length at development time, the positional code for self-attention and cross-attention is identical.

You will not convert your entire codebase to Einlang. The language is young. The tooling is sparse. You have a deadline. But you will start putting names where they cost nothing and prevent everything: in comments, in variable names, in the structure of your tensor shapes. `# dim=1 is channel`. `x: Tensor["batch", "channel", "spatial"]`. `rearrange(x, "batch channel spatial -> batch spatial channel")`.

A name in a comment can rot. But it rots slower than a name that was never written.

The book is almost over. The habit is just beginning. Close the cover. Open your editor. Read the first tensor line you see. Ask the question that the preceding pages have taught you to ask:

**Where is the name?**

If the answer is *nowhere*—you have found your starting point.

---

## The Invariant

Fifteen chapters. One invariant. Say it once more before you go:

**Every tensor operation that depends on a coordinate's identity must record that identity in the source code.**

This is not a language rule. It is a practice rule. It applies in Einlang, in PyTorch, in JAX, in NumPy, in any framework where tensors carry coordinates that mean different things. The notation you use determines *how* you record the identity—brackets, comments, variable names, einops strings—but the invariant is the same.

The invariant does not prevent all bugs. It prevents the class of bugs where the coordinate identity was lost before the operation was performed. A reduction over `dim=1` does not know it's reducing over `channel`. A reduction over `channel` knows. When the channel moves to `dim=2`, the first reduction silently becomes wrong. The second reduction becomes a compile error. The difference is whether the identity was recorded.

You now know how to record it. The rest is practice.

---

## How to Start

You don't need Einlang to practice the coordinate habit. You need a place to put a name and a discipline to keep it honest.

Start small. Name one coordinate at a time. The data-entry boundary is the most important one—if coordinates are named when tensors enter the program, the names flow downstream. Name the reductions next—they are where coordinates are consumed, and the consumption is the hardest fact to reconstruct later. Name the broadcasts last—they are often implicit, and making them explicit is the most verbose change.

In PyTorch, a comment is your first bridge: `x.mean(dim=1)  # dim 1 = channel`. In JAX, einops patterns are your bridge: `rearrange(x, "batch channel spatial -> batch spatial channel")`. The bridge doesn't have to be perfect. It has to be there.

The goal is not to convert your entire codebase to named dimensions overnight. The goal is to develop the reflex: when you write an operation that depends on a coordinate's identity, put that identity in the source. Not in your head. Not in a Slack message. In the source.

If you completed Chapters 12–14, you have something else: a miniature compiler. It checks five rules. It is not industrial-grade, but it is yours. When you write `x.mean(dim=1)` in PyTorch tomorrow, that compiler runs in your head. It notices that the reduction consumes a coordinate with no name. It asks: *which coordinate?* The compiler cannot halt your Python program—but you can. That silent compiler, running on attention rather than electricity, is what the construction chapters built. Not a tool you install. A reflex you keep.

---

## Three Scenarios

You are not going to use Einlang tomorrow. But you are going to encounter these three scenarios. Here is what the coordinate habit looks like in each.

**Scenario 1: The legacy codebase.** You inherit a PyTorch model with 200 occurrences of `dim=-1`. Nobody remembers which dimension is which. The README doesn't say. The original author left the company.

You have two choices. Choice A: print shapes at every layer, trace dimensions manually, build a mental map that lives in your head and dies when you context-switch. Choice B: spend one afternoon adding a comment at every `dim` argument. `# dim=-1 = feature`. `# dim=1 = channel`. `# dim=(2,3) = spatial`.

Choice A costs one afternoon now and ten afternoons over the next year. Choice B costs two afternoons now and zero later. The coordinate habit is Choice B. The names don't need to be checked by a compiler to be useful. They need to be visible. A comment is a name that the compiler can't read but the next programmer can. That's already 80% of the value.

**Scenario 2: The new project.** You are designing a data pipeline from scratch. You can name your dimensions however you want. You have the rare luxury of a greenfield.

Here is the coordinate habit for a greenfield: name your dimensions in the data loader, not in the model. The moment a tensor enters your program—from a file, from a database, from a random generator—attach coordinate names. If your framework doesn't support named dimensions, use a convention: batch is always first, spatial is always last, feature is always second. Document the convention in the project README. Make every `dim` argument consistent with the convention.

The goal is not compiler-checkable contracts. The goal is that six months from now, when you've forgotten the details, the convention tells you what `dim=1` means. A convention is a name that lives in the project rather than in the code. It's less reliable than a compiler check, but infinitely more reliable than nothing.

**Scenario 3: The bug investigation.** It's 3 AM. The model's loss is NaN. You're printing tensor shapes, looking for a mismatch. You find one: a tensor has shape `(32, 64)` where you expected `(64, 32)`. The transpose is missing. Or is it? Maybe the shapes are correct and the transpose happened upstream. You can't tell from the shapes alone.

The coordinate habit for debugging: before you print another shape, write down which coordinate you *think* each dimension is. `dim 0 = batch? dim 1 = feature?` Then check whether the operations make sense for those identities. If `x.mean(dim=0)` is normalizing over `batch`, something is wrong—regardless of whether the shapes match.

This is the coordinate audit from the Appendix, applied to a live bug. Four questions. Which coordinate is consumed? Which coordinate is copied along? Can you trace a coordinate from source to destination? Does the backward reduction match the forward broadcast? Ask them of the operation that produced the unexpected shape. The answers will tell you whether the bug is in the shapes or in the semantics—whether the transpose is missing or the reduction is over the wrong axis.

---

## A Week with the Habit

What does the coordinate habit look like in practice, day by day? Not a conversion project. A week of small changes.

**Monday.** Open a file. Find a `dim=` argument. Write a comment next to it saying which coordinate it refers to. `x.mean(dim=1)  # dim 1 = channel`. Do this for five `dim=` arguments. Time: ten minutes.

**Tuesday.** Find a broadcast. `A + b`. Ask whether the broadcast is semantically justified. Write down which coordinate `b` is silent on. If the answer is not obvious from the variable names, rename `b` so that it is.

**Wednesday.** Find a permutation. `x.permute(0, 2, 1)`. Rewrite it as an einops `rearrange` string, naming the dimensions. `rearrange(x, "batch height width -> batch width height")`. Compare the two lines. Which one tells you what moved where?

**Thursday.** Find a reduction used in a loss function. `loss = x.sum()`. Which coordinate did it sum over? All of them. Is that correct? If `x` has coordinates `(batch, class)`, `sum()` produces a scalar. But `sum[class]` followed by `mean[batch]` produces a per-batch average loss—which may be what you intended but is not what you wrote. Name the reduction. Check the intent.

**Friday.** Audit one function end to end. Pick a function with at least two tensor operations. Write its coordinate signature in a comment above the `def` line. `# fn(batch, feature) -> (batch, class)`. Walk through the body. Does every operation respect the declared coordinate flow? Does every reduction consume the right coordinate? Does every broadcast copy along the right coordinate? Time: twenty minutes.

**Saturday.** You are not working. But if you think about tensor shapes anyway—and you will, because the habit is settling in—notice which coordinate you are uncertain about. The uncertainty is the gap. Write down the name you are unsure of. On Monday, put it in the code.

A week. Five small actions. No new tools. No framework migration. Just a shift in what you notice when you read a tensor operation. The coordinate habit is not a flag you plant. It is a lens you wear. Once you put it on, you see the gaps. The gaps were always there. You just didn't have a name for what was missing.

---

## What the Coordinate Habit Does Not Solve

The book has made a case for named coordinates. But the case has limits. It is worth naming them before you leave, so you do not carry false expectations into your next project.

**Names do not guarantee correctness.** You can name the wrong coordinate. `mean[channel](x)` where you should have written `mean[spatial](x)` compiles without error. The name `channel` exists on `x`. The reduction is well-formed. The gradient will be correct—for the wrong reduction. Names catch inconsistency. They do not catch wrongness. A coordinate named `channel` that is actually `spatial` in the data is a semantic error, and semantic errors survive any notation.

**Names do not replace testing.** The compiler checks that the coordinate structure is internally consistent. It does not check that the computation achieves what you intended. A softmax normalized over `batch` instead of `class` is internally consistent—every reduction, broadcast, and gradient aligns perfectly. The program compiles. It is still wrong. Only a test that checks the output's shape and statistical properties would catch it.

**Names do not eliminate runtime shape errors.** Dynamic dimensions—sequence lengths, batch sizes that vary per call—cannot be checked at compile time. The compiler can verify that `seq` is a declared coordinate and that functions consuming it have consistent contracts. It cannot verify that `seq` has length 64 rather than 128. That check lives at runtime, in an assertion or a shape guard.

**Names cost keystrokes.** This is the honest objection. `mean[channel](x)` is longer than `x.mean(dim=-1)`. The book has argued that the keystrokes are repaid in debugging time. But the cost is real, and in a codebase where dimension order is stable and well-documented, the positional shorthand may be the right tradeoff. The coordinate habit is not a moral imperative. It is a tool. Use it where the cost of a silent axis swap exceeds the cost of typing a bracket.

**Einlang itself is young.** The language this book uses to make its arguments is a research prototype. Its tooling is sparse. Its error messages are the ones shown in these pages—no more. It does not compile to CUDA. It does not have a package manager. It does not integrate with PyTorch or JAX. The compiler described in Chapters 12–14 is a frontend—it produces lowered NumPy, not optimized GPU kernels. A production-grade named-coordinate compiler would need an autodiff engine, a scheduler, and a backend that generates efficient code for the lowering patterns this book described. None of that exists today. The book's argument does not depend on Einlang's maturity—the coordinate habit works through comments, einops strings, and naming conventions in any framework. But if you are tempted to build the rest: the IR, the check rules, and the lowering pass in these pages are a starting point. The distance from here to a production compiler is measured in engineering years, not ideas.

These five limitations do not weaken the case for named coordinates. They clarify it. Named coordinates prevent one class of error: the error where the coordinate identity exists in the programmer's head but not in the source text, and the notation provides no place to record it. For that class of error—the silent axis swap, the broadcast that drifts with the layout, the reduction that changes meaning without changing syntax—names are the only defense. For errors outside that class, other defenses apply.

**A note on existing named-tensor systems.** PyTorch has named tensors. xarray has labeled dimensions. Einops has named patterns. Why not just use those? Each catches a subset of the errors this book describes. PyTorch's named tensors check broadcast alignment by name but are not part of the type system and do not survive through autograd. xarray labels dimensions for data analysis but does not compile to GPU kernels. Einops patterns are local to each call—they do not propagate across function boundaries. None of these systems provide the five-check wall of Chapter 13, where every coordinate contract is verified at every call site before a single value is computed. The book built a complete system not because the existing tools are useless—they are useful, and the coordinate habit works through them—but because only a complete system can show the full distance between what positional notation checks and what named notation can check. The distance is the argument.

---

## Three Books, One Idea

When I started writing this book, I thought it would be about a language. About syntax, about compiler passes, about the engineering of a notation that treats coordinates as first-class citizens.

A third of the way through, I realized it was about a bug. One bug, from one line of code, that lived for three weeks because the notation had no place to record the fact that would have caught it. The bug was not unusual. It was not the programmer's fault. It was a gap in the notation, and the gap was invisible to anyone who had not learned to see it.

Two thirds of the way through, I realized it was about a habit. Not the notation, not the compiler, not the language. The habit of pausing before a reduction and asking: which coordinate am I consuming? The habit of looking at `dim=-1` and seeing an absence where a name should be. The habit of tracing a coordinate from data entry to loss and noticing where its identity is lost.

The language is a vehicle for the habit. The compiler is a proof that the habit can be mechanized. The bug is a demonstration of what happens without the habit. But the habit is the thing. It is portable. It works in any framework, in any language, with any notation—as long as there is a place to put a name.

This book is three books. The first is about a language called Einlang. The second is about a bug that a positional notation could not catch. The third is about a habit that you can practice starting today, in the codebase you already have, with the tools you already use.

The language may not survive. Languages rarely do. The bug will survive—it is recreated every time a programmer writes `dim=-1` and trusts that the last axis will always be the right one. The habit can survive, if you choose to practice it.

This book rests on a simple conviction: information that correctness depends on should live in the source code, not in the programmer's head. A comment can rot. A variable name can drift. A convention can be forgotten. A bracket with a name in it checks itself. This is not an engineering preference. It is an ethic of notation—that the notation should carry the facts its correctness depends on, so that those facts survive the programmer who wrote them. The coordinate habit is the practice of that ethic. The language is one implementation of it. The compiler is the proof that it can be mechanized.

---

---

If you take only one thing from this book, take this:

**Names are type information.**

A coordinate name is not documentation. It is not a comment. It is not a convention. It is a type-level fact—as checkable as `int` vs `float`, as structural as `(batch, feature)` vs `(feature, batch)`. When a name is in the bracket, the compiler can ask questions about it. When a name is in a comment, only a human can. The distance between those two kinds of name is the distance between a check and a hope.

That is the book. The rest is practice.

1. Name the coordinate you consume.
2. Name the coordinate you broadcast along.
3. Put the name where the next reader can see it.

Thank you for reading.

---

## If You Want More

This book did not begin in a vacuum. Three works shaped its thinking more than any others.

**Structure and Interpretation of Computer Programs** (Abelson, Sussman, and Sussman) taught generations of programmers that a language is built from primitive expressions, means of combination, and means of abstraction—and that building a metacircular evaluator is the final proof of understanding. The compiler chapters in this book are a shorter, humbler descendant of that tradition.

**Learn You a Haskell for Great Good** (Miran Lipovača) showed that a book about a programming language can be warm, direct, and relentlessly focused on the reader's understanding rather than the author's expertise. Its tone echoes through every chapter of this book.

**"Tensor Considered Harmful"** (Aleksander Mądry, 2018) and the named-tensor work at Harvard and Stanford asked the question this book tries to answer in full: what happens when tensor dimensions have names that the compiler can check?

These are not tutorials. They are not documentation. They are the road signs that pointed toward the coordinate habit. If this book left you wanting to go deeper—into compilers, into notation design, into the philosophy of what source code makes visible—start here.
