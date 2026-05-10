---
layout: book
title: "Chapter 15 · The Outline of the Name"
---

# Chapter 15 · The Outline of the Name

> "We shall not cease from exploration. And the end of all our exploring will be to arrive where we started and know the place for the first time."
>
> — T. S. Eliot

*Reflection · What the names did for us*

---

This book began with a bug.

A tensor had shape `(32, 64, 256)`. A programmer wrote `x.mean(dim=1)`. The intent was "average over channels." The text said `dim=1`. Three months later, a refactoring moved `channel` to position 2. `dim=1` now silently erased `spatial`. The shape was still `(32, 256, 64)`. The loss still went down. The model deployed on Tuesday. The customer complaint arrived on Thursday.

The bug lived for three weeks because the notation had no slot for the fact that would have caught it.

Thirteen chapters later, the slot exists. It is a bracket with a name in it. And the bracket changes everything—not because it is syntactically novel, but because it makes a fact checkable that was previously only commentable.

This chapter is not a summary. It is a question: after everything we have built, what did the names actually do for us?

---

## The Life of a Name

A coordinate name in einlang lives through four stages. It is **written** in source, as a letter between brackets: `[i, j]`, `[class]`, `[batch]`. It is **preserved** in the intermediate representation, stripped of syntax but keeping every name intact: `(let-decl (output C (i j)) ...)`. It is **verified** by analysis, where the compiler derives its range, checks its consistency across call sites, and records which operations consume it. And it is **burned** in lowering, translated into the integers that machines require: `class → axis=1`, `i → loop 0..batch`.

Four stages. Written, preserved, verified, burned. At no point is the name decoration. At every point, it is load-bearing.

A decoration can be omitted without consequence. You can remove the comment `# dim=1 is channel` and the code still runs. You can rename the variable `spatial_features` to `x` and the compiler says nothing. Decorations are for humans. The machine does not read them.

A name in einlang is not a decoration. The compiler reads it. The checker verifies it. The lowering pass translates it. If you write `softmax[class](logits)` and `logits` has no coordinate called `class`, the compiler stops. Not at runtime—at compile time. The name is part of the contract.

This is the difference between a comment and a coordinate. A comment records intent. A coordinate enforces it.

---

## What Names Caught

Walk back through the book and ask, at each stage: what would a positional notation have let through?

**Chapter 1**: `x.mean(dim=1)`. The positional notation let it through for three weeks. The named version `mean[channel](x)` would have broken at compile time the moment `channel` moved to position 2, because the tensor would no longer have a coordinate called `channel` at the position the compiler expected. Or, more precisely: the compiler would have required the refactoring programmer to update the coordinate declaration, and that update would have surfaced the fact that `mean[channel]` was still referencing the old layout.

**Chapter 2**: `A + bias`. The positional notation broadcasts `bias` along whichever dimensions happen to be missing. If `A` changes from `(batch, feature)` to `(feature, batch)`, the broadcast silently flips. The named version `out[i, j] = A[i, j] + bias[j]` makes the omission visible: `bias` has no `i`, so it broadcasts over `i`. If `i` and `j` swap meanings upstream, the indexing pattern breaks visibly.

**Chapter 3**: `softmax(logits, dim=-1)`. When `batch_size == num_classes`, the square matrix test applies: `softmax(logits, dim=0)` and `softmax(logits, dim=-1)` both produce valid probability distributions. The named version `softmax[class](logits)` does not let you silently normalize over `batch`. The name `class` is either present on `logits` or it isn't.

**Chapter 5**: `u[t, i] = u[t+1, i] + f(...)`. In a positional recurrence, writing `t+1` instead of `t-1` produces a forward reference—a read from the future. The positional loop runs. The values are whatever was in memory. The named version rejects it: the causality check sees `t+1 > t` and halts.

**Chapter 7**: The gradient of a broadcast. Forward: `bias` omits `batch`, broadcasting over it. Backward: the gradient must sum over `batch` to recover `bias`'s shape. In a positional framework, this sum is implicit in the autodiff engine. If the broadcast changes because the shape changed, the gradient sum changes with it—silently. In the named version, the coordinate sets tell you exactly what the gradient must sum over: `C` has `{i, j}`, `A` has `{i, k}`, sum over `{j}`. The set subtraction is checkable.

**Chapter 8**: GroupNorm's reshape chain: `x.reshape(N, G, C//G, H, W).mean(dim=(2,3,4))`. The positions `(2,3,4)` are only correct after the reshape. If the reshape changes, the positions change. The named version `mean[c_in_group, ..spatial]` names the coordinates directly. The reshape is unnecessary because the coordinates are separate from the start.

**Chapter 9**: Self-attention and cross-attention in PyTorch have identical code. The difference is only in the shapes of the tensors passed at runtime. The named version distinguishes `self_attention[seq, ...]` from `cross_attention[seq_q, seq_k, ...]` in the type signatures. A reader can see which is which without checking runtime shapes.

Every one of these bugs was shape-correct. Every one survived the checks that positional frameworks perform. Every one was caught by a name.

---

## What Names Couldn't Catch

The boundary in Chapter 6 was not an accident. Named coordinates verify that a coordinate *exists* on a tensor. They do not verify that index arithmetic stays within bounds. `oh + kh` is syntactically valid as long as both `oh` and `kh` are declared coordinates. Whether `oh + kh` exceeds the input's spatial extent is a runtime question.

This boundary is a design choice, not a limitation. The compiler checks what can be proven from names and domains alone. Bounds checking is the runtime's job. Semantic correctness—whether the formula means what you think it means—is yours.

The names reduce the surface area of uncheckable facts. They do not eliminate it. A comment that says "channel" can be wrong. A coordinate named `channel` can also be wrong—you might have named the wrong axis `channel`. But the coordinate, once named, is checked for consistency everywhere it appears. The comment is checked nowhere. The coordinate is a fact. The comment is a hope.

---

## The Shape of the Book

A name in einlang is not a string. It is a structural element with a defined life cycle. The book's four parts trace that life cycle:

**Part I (Chapters 1–2)** established that a coordinate has a name, and that the name can be used to declare what survives, what is consumed, and what is broadcast. The megaphone model gave intuition: a tensor speaks on some coordinates and stays silent on others.

**Part II (Chapters 3–7)** showed how names compose. Coordinate-aware functions make the operated-on coordinate part of the type-level contract. Recurrence makes the direction of time a syntactic constraint. Differentiation reads the forward pass backward, with the same names organizing both directions.

**Part III (Chapters 8–10)** put einlang next to PyTorch and NumPy. The same computations, two notations. The question was not "which is better" but "what does each notation make visible."

**Part IV (Chapters 11–13)** opened the compiler. The IR preserves names. Analysis derives range, shape, and type from them. Five check rules verify their consistency. Lowering burns them into integers. The names that survived three parts of the book finally become numbers—but only after they have done all their work.

---

This is the outline the names leave behind. Not a list of features. Not a syntax reference. An outline of *what changes* when coordinates have names that the compiler can read.

When you write `x.mean(dim=1)`, the `1` is a position. It records where. It does not record what.

When you write `mean[channel](x)`, the `channel` is a name. It records both.

The distance between `where` and `what` is the distance between a program that runs correctly by accident and a program that runs correctly by construction. The names do not guarantee correctness. They guarantee that incorrectness, when it occurs, leaves a trace that the compiler can find and the reader can see.

---

You are now holding the book that began with a bug and became a compiler. The compiler is not the point. The language is not the point. The point is the question you can now ask about any tensor operation in any framework:

**Where is the name?**

If the name is in a comment, it can rot. If the name is in the code, it can be checked. If the name is nowhere, the intent exists only in your head—and your head will not be there at 3 AM when the model deployed on Tuesday produces a customer complaint on Thursday.

---

## What the Names Did for You

Before this book, you read tensor operations as arithmetic. `x.mean(dim=1)` was "reduce axis 1." The operation was complete. The text contained everything the compiler needed.

After this book, you read `x.mean(dim=1)` and see an absence. The axis has no name. The coordinate it erases has no identity. The code records *where* but not *what*. The operation is incomplete. The text tells the compiler what to execute. It does not tell the reader what was intended.

This shift—from seeing operations as complete to seeing them as underspecified—is the coordinate habit. It is not a skill. It is not a technique. It is a change in what you notice when you look at a tensor operation. Every person who reads this book starts at a different point and ends at a different point. But everyone ends by seeing the gaps.

The gaps were always there. You just didn't have a name for what was missing.

## The Reader's Review

This chapter has traced the outline the names leave behind. But an outline is not a review. A review is not something you read—it is something you do.

So before you close this book, do the review. Not by rereading. By answering. The following questions walk through every chapter, but they don't summarize. They ask you to notice what has changed in how you read.

Take your time. There is no rush. The book will still be here when you finish.

---

**Chapter 1.** You read this line on page one:

```python
x = x.mean(dim=1)
```

At the time, you saw a tensor operation. `dim=1` was a position. Now read it again. What do you see?

Do you see the absent name? The coordinate that `1` refers to—is it `channel`, `feature`, `class`? The code doesn't say. Three months ago, before this book, you might not have noticed the absence. Now the absence is the loudest thing on the line.

---

**Chapter 2.** You wrote `out[i, j] = A[i, j] + bias[j]`. The omission was the point: `bias` has no `i`, so it broadcasts.

Now read `A + bias` in NumPy. What do you see? Do you see the broadcast that the code doesn't name? Do you see the coordinate that `bias` silently copies along? Do you know, from the code alone, whether the broadcast is intentional or accidental?

If you can answer the third question, you have learned the broadcast self-audit. If you cannot—if the code gives you no information to answer with—you have learned why the self-audit is necessary.

---

**Chapter 3.** The Square Matrix Test. Read this:

```python
softmax(logits, dim=-1)
```

When `batch_size == num_classes`, `dim=0` and `dim=-1` both produce valid probability distributions. The code is correct either way. The *program* is correct only one way.

Now read `softmax[class](logits)`. Does the name `class` appear on `logits`? If not, it's an error. If yes, it's checked at every call site. The Square Matrix Test is not a trick. It is a fact: when extents coincide, only names differ. If you don't write the names, you cannot tell the difference.

---

**Chapter 4.** GroupNorm. Read this PyTorch line:

```python
x = x.reshape(N, G, C//G, H, W).mean(dim=(2,3,4))
```

`dim=(2,3,4)` means "reduce over c_in_group, H, W." But only because the reshape put them at those positions. If the reshape changes, the tuple changes.

Now read `mean[c_in_group, H, W](x[..batch, group, c_in_group, H, W])`. The bracket names the reduced coordinates. If the layout changes, the names don't. Which line would you rather be responsible for maintaining six months from now, at 11 PM, during an incident?

---

**Chapter 5.** Causality. `u[t+1, i]` on the right-hand side of a recurrence with declaration `u[t in 1..T, i]`. Before this book, would you have caught it? Would the compiler? In Python, the loop runs. In einlang, the compiler halts. The difference is whether time has a direction in your notation.

---

**Chapter 6.** The boundary. `oh + kh` is valid. `oh + 1000` is valid too—even if `oh + 1000` overflows the input. The compiler checks that `oh` and `kh` are coordinates. It does not check that their sum is in bounds.

This boundary is a design choice. Why is it here? Because the coordinate system guarantees that `oh` and `kh` are the right *kind* of thing—spatial indices. It does not guarantee that they are the right *value*. Bounds checking is the runtime's job. Semantic correctness is yours. The names reduce the surface area of uncheckable facts. They do not eliminate it.

---

**Chapter 7.** The gradient. Read `dA[i, k] = sum[j](dC[i, j] * B[k, j])`. You derived this yourself in Chapter 7, from the coordinate sets alone. Now read `dA = dC @ B.T`. Do you know which axes are being contracted? Do you know why the transpose is there? If you had to re-derive it from the positional code, could you?

The coordinate accounting gave you set subtraction: `C` has `{i, j}`, `A` has `{i, k}`, sum over `{j}`. The positional code gives you `dC @ B.T`. Both produce the same result. Only one explains itself.

---

**Chapters 8–10.** The comparisons. You saw LayerNorm, RMSNorm, GroupNorm, attention, and the heat equation in two notations. Each time, the positional code was correct. Each time, the named code made different facts visible.

After reading all three comparison chapters, answer this: which bug would you rather debug at 3 AM—a `dim=-1` that should have been `dim=-2`, or a `softmax[class]` where `class` doesn't exist? The first is silent. The second is a compiler error. The compiler cannot prevent all bugs. But it can prevent the ones where the notation records enough information for the compiler to check.

---

**Chapters 11–13.** The compiler. You watched a name travel from source through IR through analysis through lowering to generated code. Five forms, one name. `class` survived every stage. It was verified at the stage where it was still a name—where the compiler could ask "does this name exist on this tensor?" and get a yes/no answer. Then it was burned into `axis=1`—after all checks had passed.

The positional compiler could not have asked that question, because the positional compiler never knew the name. The name was in the programmer's head. The compiler burned a number that was already a number. No verification occurred.

---

Now close your eyes. Or look away from the page. Ask yourself one question:

**If you had to explain to a colleague why named coordinates matter—not with arguments, but by showing them a single page from this book—which page would you choose?**

Your answer to that question is your review. It is not the same as mine. It doesn't need to be. The book is not a doctrine. It is a lens. What you see through it depends on what you brought to it.

---

Now read this line:

```python
x = x.mean(dim=1)
```

What appears in your mind?

Does a coordinate name surface—`channel`, `feature`, `class`? Does a question arise: which coordinate is position 1, and how do I know? Does the gap between the number and the identity feel wider than it did seventeen chapters ago?

If the answer to any of these is yes, the book has done its work. Not by converting you to einlang. Not by persuading you to rename every dimension in your codebase. By changing what you notice when you read a tensor operation. The coordinate habit is that change.

---

Before you turn the final page, one last exercise. Open a terminal. Navigate to your most recent project. Pick the first `.py` file that contains a tensor operation. Find the first `dim=`, `axis=`, or `permute` call. Read it. Ask: if the dimension order changed tomorrow, would this line still be correct?

If you can answer with confidence—because a comment records the coordinate name, because an einops string names the dimensions, because the variable naming convention is consistent—you have applied the habit. If you cannot answer with confidence—because the number records only position, not identity—you have found a ghost. You now know its name. And you know that naming it is the first step toward making it visible.

The ghost has been there since before you opened this book. The difference is that now you can see it. And what you can see, you can name. And what you can name, you can check.

Turn the page.

