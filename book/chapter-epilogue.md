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

---

## What the Coordinate Habit Does Not Solve

The book has made a case for named coordinates. But the case has limits. It is worth naming them before you leave, so you do not carry false expectations into your next project.

**Names do not guarantee correctness.** You can name the wrong coordinate. `mean[channel](x)` where you should have written `mean[spatial](x)` compiles without error. The name `channel` exists on `x`. The reduction is well-formed. The gradient will be correct—for the wrong reduction. Names catch inconsistency. They do not catch wrongness. A coordinate named `channel` that is actually `spatial` in the data is a semantic error, and semantic errors survive any notation.

**Names do not replace testing.** The compiler checks that the coordinate structure is internally consistent. It does not check that the computation achieves what you intended. A softmax normalized over `batch` instead of `class` is internally consistent—every reduction, broadcast, and gradient aligns perfectly. The program compiles. It is still wrong. Only a test that checks the output's shape and statistical properties would catch it.

**Names do not eliminate runtime shape errors.** Dynamic dimensions—sequence lengths, batch sizes that vary per call—cannot be checked at compile time. The compiler can verify that `seq` is a declared coordinate and that functions consuming it have consistent contracts. It cannot verify that `seq` has length 64 rather than 128. That check lives at runtime, in an assertion or a shape guard.

**Names cost keystrokes.** This is the honest objection. `mean[channel](x)` is longer than `x.mean(dim=-1)`. The book has argued that the keystrokes are repaid in debugging time. But the cost is real, and in a codebase where dimension order is stable and well-documented, the positional shorthand may be the right tradeoff. The coordinate habit is not a moral imperative. It is a tool. Use it where the cost of a silent axis swap exceeds the cost of typing a bracket.

These four limitations do not weaken the case for named coordinates. They clarify it. Named coordinates prevent one class of error: the error where the coordinate identity exists in the programmer's head but not in the source text, and the notation provides no place to record it. For that class of error—the silent axis swap, the broadcast that drifts with the layout, the reduction that changes meaning without changing syntax—names are the only defense. For errors outside that class, other defenses apply.

---

## Three Books, One Idea

When I started writing this book, I thought it would be about a language. About syntax, about compiler passes, about the engineering of a notation that treats coordinates as first-class citizens.

A third of the way through, I realized it was about a bug. One bug, from one line of code, that lived for three weeks because the notation had no place to record the fact that would have caught it. The bug was not unusual. It was not the programmer's fault. It was a gap in the notation, and the gap was invisible to anyone who had not learned to see it.

Two thirds of the way through, I realized it was about a habit. Not the notation, not the compiler, not the language. The habit of pausing before a reduction and asking: which coordinate am I consuming? The habit of looking at `dim=-1` and seeing an absence where a name should be. The habit of tracing a coordinate from data entry to loss and noticing where its identity is lost.

The language is a vehicle for the habit. The compiler is a proof that the habit can be mechanized. The bug is a demonstration of what happens without the habit. But the habit is the thing. It is portable. It works in any framework, in any language, with any notation—as long as there is a place to put a name.

This book is three books. The first is about a language called Einlang. The second is about a bug that a positional notation could not catch. The third is about a habit that you can practice starting today, in the codebase you already have, with the tools you already use.

The language may not survive. Languages rarely do. The bug will survive—it is recreated every time a programmer writes `dim=-1` and trusts that the last axis will always be the right one. The habit can survive, if you choose to practice it.

---

That is the book. The rest is practice.

1. When you write a reduction, can you name the coordinate you are consuming—from the code alone, without checking shapes?
2. When you write a broadcast, can you name the coordinate you are copying along—and is the independence justified?
3. When you read `dim=`, can you say which coordinate that position refers to—and would it survive a layout change six months from now?

If the answer to any of these is no, you have found a gap. The gap is not your fault. The notation failed to provide a slot for the fact. But you now know the slot exists. Put the name in a comment. Put it in an einops string. Put it in a bracket, if the language supports it. Put it somewhere the next reader—who may be you, at 3 AM, six months from now—can see it.

That is the book. The rest is practice.

Thank you for reading.
