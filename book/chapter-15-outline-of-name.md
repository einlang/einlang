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

A coordinate name in Einlang lives through four stages. It is **written** in source, as a letter between brackets: `[i, j]`, `[class]`, `[batch]`. It is **preserved** in the intermediate representation, stripped of syntax but keeping every name intact: `(let-decl (output C (i j)) ...)`. It is **verified** by analysis, where the compiler derives its range, checks its consistency across call sites, and records which operations consume it. And it is **burned** in lowering, translated into the integers that machines require: `class → axis=1`, `i → loop 0..batch`.

Four stages. Written, preserved, verified, burned. At no point is the name decoration. At every point, it is load-bearing.

A decoration can be omitted without consequence. You can remove the comment `# dim=1 is channel` and the code still runs. You can rename the variable `spatial_features` to `x` and the compiler says nothing. Decorations are for humans. The machine does not read them.

A name in Einlang is not a decoration. The compiler reads it. The checker verifies it. The lowering pass translates it. If you write `softmax[class](logits)` and `logits` has no coordinate called `class`, the compiler stops. Not at runtime—at compile time. The name is part of the contract.

This is the difference between a comment and a coordinate. A comment records intent. A coordinate enforces it.

---

## What Names Caught

Walk back through the book and ask, at each stage: what would a positional notation have let through?

**Chapter 1**: `x.mean(dim=1)`. The positional notation let it through for three weeks. The named version `mean[channel](x)` would have broken at compile time the moment `channel` moved to position 2, because the tensor would no longer have a coordinate called `channel` at the position the compiler expected. Or, more precisely: the compiler would have required the refactoring programmer to update the coordinate declaration, and that update would have surfaced the fact that `mean[channel]` was still referencing the old layout.

**Chapter 2**: `A + bias`. The positional notation broadcasts `bias` along whichever dimensions happen to be missing. If `A` changes from `(batch, feature)` to `(feature, batch)`, the broadcast silently flips. The named version `out[i, j] = A[i, j] + bias[j]` makes the omission visible: `bias` has no `i`, so it broadcasts over `i`. If `i` and `j` swap meanings upstream, the indexing pattern breaks visibly.

**Chapter 3**: `softmax(logits, dim=-1)`. When `batch_size == num_classes`, the square matrix test applies: `softmax(logits, dim=0)` and `softmax(logits, dim=-1)` both produce valid probability distributions. The named version `softmax[class](logits)` does not let you silently normalize over `batch`. The name `class` is either present on `logits` or it isn't.

**Chapter 6**: `u[t, i] = u[t+1, i] + f(...)`. In a positional recurrence, writing `t+1` instead of `t-1` produces a forward reference—a read from the future. The positional loop runs. The values are whatever was in memory. The named version rejects it: the causality check sees `t+1 > t` and halts.

**Chapter 8**: The gradient of a broadcast. Forward: `bias` omits `batch`, broadcasting over it. Backward: the gradient must sum over `batch` to recover `bias`'s shape. In a positional framework, this sum is implicit in the autodiff engine. If the broadcast changes because the shape changed, the gradient sum changes with it—silently. In the named version, the coordinate sets tell you exactly what the gradient must sum over: `C` has `{i, j}`, `A` has `{i, k}`, sum over `{j}`. The set subtraction is checkable.

**Chapter 9**: GroupNorm's reshape chain: `x.reshape(N, G, C//G, H, W).mean(dim=(2,3,4))`. The positions `(2,3,4)` are only correct after the reshape. If the reshape changes, the positions change. The named version `mean[c_in_group, ..spatial]` names the coordinates directly. The reshape is unnecessary because the coordinates are separate from the start.

**Chapter 10**: Self-attention and cross-attention in PyTorch have identical code. The difference is only in the shapes of the tensors passed at runtime. The named version distinguishes `self_attention[seq, ...]` from `cross_attention[seq_q, seq_k, ...]` in the type signatures. A reader can see which is which without checking runtime shapes.

Every one of these bugs was shape-correct. Every one survived the checks that positional frameworks perform. Every one was caught by a name.

---

## What Names Couldn't Catch

The boundary in Chapter 7 was not an accident. Named coordinates verify that a coordinate *exists* on a tensor. They do not verify that index arithmetic stays within bounds. `oh + kh` is syntactically valid as long as both `oh` and `kh` are declared coordinates. Whether `oh + kh` exceeds the input's spatial extent is a runtime question.

This boundary is a design choice, not a limitation. The compiler checks what can be proven from names and domains alone. Bounds checking is the runtime's job. Semantic correctness—whether the formula means what you think it means—is yours.

The names reduce the surface area of uncheckable facts. They do not eliminate it. A comment that says "channel" can be wrong. A coordinate named `channel` can also be wrong—you might have named the wrong axis `channel`. But the coordinate, once named, is checked for consistency everywhere it appears. The comment is checked nowhere. The coordinate is a fact. The comment is a hope.

---

## When the Name Is Wrong

The Panorama traced `class` through a journey where every check passed. But what happens when the name is wrong?

Suppose the programmer writes `softmax[batch](logits[batch, class])`. `batch` is the coordinate argument. The parser accepts it. The resolver finds it on `logits`. Analysis may or may not catch it, depending on how packs are bound. The code could compile. It would produce a valid probability distribution—over the batch dimension, not the class dimension.

The name was wrong. The check passed. The program is incorrect.

This is the boundary from Chapter 7, restated for the final time: **names check consistency, not correctness.** `softmax[batch]` is internally consistent—every reduction, broadcast, and gradient aligns over `batch`. The error is that the programmer wanted `class`. The compiler cannot read the programmer's mind. It can only verify the contract the programmer wrote.

But the name `batch` is visible. When the next programmer reads `softmax[batch](logits)`, they see the error immediately. The positional equivalent `softmax(logits, dim=0)` hides the error behind a number. The reader sees `dim=0` and must reconstruct whether axis 0 is batch or class. The reconstruction may be wrong.

A wrong name is a visible error. A missing name is an invisible one.


---

## If the Names Had Been There

The book has shown what names caught. But there is another way to understand their value: replay the book's scenarios with names absent, and watch the bugs survive.

**If Chapter 1's bug had names...** The programmer who refactored `channel` from position 1 to position 2 would have updated the coordinate declaration `[batch, channel, spatial]` to `[batch, spatial, channel]`. The line `mean[channel](x)` would have compiled—because `channel` still exists on `x`. But wait. Would the bug have been caught? The answer depends on what else changed. If the mean was always `mean[channel]`, and the declaration still has a coordinate named `channel`, the reduction is still over the right coordinate. The name protects the reduction from the layout change. The positional version `dim=1` silently changes meaning; the named version `mean[channel]` doesn't. This is the simplest case, and the name handles it completely.

**If Chapter 2's broadcast had names...** `A + bias` becomes `A[i, j] + bias[j]`. The broadcast over `i` is visible. Six months later, a programmer adds a time dimension to `A`—it becomes `A[t, i, j]`. The broadcast `bias[j]` now omits both `t` and `i`. The backward pass sums over `{t, i}`. In positional NumPy, the broadcast would silently extend to the new leading dimension. The gradient would sum over both. The shapes would match. But would the semantics be correct? A bias that is independent of `i` (sample) might not be independent of `t` (time). The name `t` is present in `A` but absent from `bias`—the omission is a claim: bias doesn't depend on time. The claim is visible. The programmer reviewing the diff sees `bias[j]` where `A` now has `A[t, i, j]` and asks: *should bias really be constant across time?* The name doesn't catch the semantic error—it makes the assumption visible so the programmer can catch it.

**If the Square Matrix Test had names...** A classifier with `batch_size = num_classes = 64`. Train loss goes down. Test loss goes down. Model deploys. Everything works. Then a new dataset arrives with `num_classes = 100`. The positional code `softmax(logits, dim=-1)` had been normalizing over the last axis—which was `class` by coincidence, because `batch` and `class` were both 64 and the rows and columns of the softmax matrix both summed to 1. The new dataset breaks the square coincidence. The bug surfaces. With names: `softmax[class](logits[batch, class])` normalizes over `class` regardless of whether `batch_size` and `num_classes` are equal. The Square Matrix Test is not a bug that names catch at compile time—it is a test that names make you pass at development time, because the name `class` in the bracket records your intent, and your intent doesn't change when the data changes.

**If the GroupNorm reshape chain had names...** The positional code:

```python
x = x.reshape(N, G, C//G, H, W).mean(dim=(2,3,4))
```

With names: `mean[c_in_group, H, W](x[batch, group, c_in_group, H, W])`. The reshape is unnecessary because the coordinates are already separate. The reduction names `c_in_group`, `H`, `W` directly. A refactoring that changes group size computes `c_in_group = C // G` from the named coordinate `C`. The positions don't need to be recalculated because they are never written. The names absorb the layout change.

**If the attention code had names...** Self-attention and cross-attention are identical in PyTorch. The difference is only in the shapes of the query and key tensors at runtime. A programmer debugging a cross-attention bug prints shapes: `q: (batch, seq, d)`, `k: (batch, seq, d)`. The shapes match. The code is self-attention. The programmer expected cross-attention because the encoder and decoder should have different sequence lengths. With names: `self_attention[seq, ...]` and `cross_attention[seq_q, seq_k, ...]` are distinct function signatures. `seq_q` and `seq_k` are different coordinate names. The function call `cross_attention[seq_q, seq_k](q, k, v)` where `q` only has `seq_q`—not `seq_k`—and `k` only has `seq_k` would have a coordinate contract that reflects the asymmetry. The names distinguish the two attention patterns at the call site.

**If the recurrence had names...** A programmer writes `h[t] = h[t+1] + f(...)`. Python runs it. The result is either garbage (if `h` is pre-allocated zeros) or an IndexError (if `t+1 >= T`). With names: the causality check rejects `t+1` because it is a forward reference. The error is caught at compile time. This is the cleanest case: one character changes the program from correct to wrong, and the name `t`—combined with the recurrence domain `t in 1..T`—gives the compiler enough information to reject the wrong one.

**If the gradient had names...** Forward: `factor[j]` broadcasts over `i`. Backward: `d_factor[j] = sum[i](d_scaled[i, j] * x[i, j])`. The coordinate `i` is the broadcast coordinate, recovered by the backward sum. In a positional framework, if `factor` changes from 1D `(j,)` to 2D `(i, j)`, the backward pass changes its sum. With names, the broadcast record says `factor omits {i}`. The backward sum is over `{i}`—unchanged by the shape change, because the name `i` is still the omitted coordinate. The name stabilizes the backward pass across shape changes.

---

Six scenarios. In each, the positional code could be correct—it is correct, in the hands of a careful programmer who tracks dimension order and documents every axis. The question is not whether positional code can be correct. It is whether the notation makes the correctness checkable. In every scenario, the named code records the coordinate identity that correctness depends on. The positional code does not. The identity lives in the programmer's head, in a comment, or nowhere. The names move it into the source—where the compiler and the next reader can both check it.

The "if" is not hypothetical. Every one of these bugs has occurred in production, in codebases you have used, in frameworks you have imported. The bugs were caught—eventually—by tests, by code review, by the programmer staring at shapes at 3 AM. The question the book has asked is not *can these bugs be caught?* It is *can these bugs be caught at compile time, by the compiler, from information already present in the code?* The answer, for every scenario above, is yes—if the information is in the code. And the notation determines whether it is.

---

## The Shape of the Book

A name in Einlang is not a string. It is a structural element with a defined life cycle. The book's four parts trace that life cycle:

**Part I (Chapters 1–2)** established that a coordinate has a name, and that the name can be used to declare what survives, what is consumed, and what is broadcast. The megaphone model gave intuition: a tensor speaks on some coordinates and stays silent on others.

**Part II (Chapters 3–8)** showed how names compose. Coordinate-aware functions make the operated-on coordinate part of the type-level contract. Recurrence makes the direction of time a syntactic constraint. Differentiation reads the forward pass backward, with the same names organizing both directions.

**Part III (Chapters 9–11)** put Einlang next to PyTorch and NumPy. The same computations, two notations. The question was not "which is better" but "what does each notation make visible."

**Part IV (Chapters 12–14)** opened the compiler. The IR preserves names. Analysis derives range, shape, and type from them. Five check rules verify their consistency. Lowering burns them into integers. The names that survived three parts of the book finally become numbers—but only after they have done all their work.

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

**Chapter 4.** The Inversion Rule. Forward broadcast means backward reduction. `bias[j]` omits `i` in the forward pass—its gradient must sum over `i`. Now look at a broadcast in your own code. Which coordinate does the bias omit? Which coordinate will the gradient sum over? If you can't answer from the code alone, the broadcast's silence is unrecorded, and the gradient's correctness depends on shapes that can change.

---

**Chapter 5.** The normalization skeleton. Read this PyTorch line:

```python
x = x.reshape(N, G, C//G, H, W).mean(dim=(2,3,4))
```

`dim=(2,3,4)` means "reduce over c_in_group, H, W." But only because the reshape put them at those positions. If the reshape changes, the tuple changes.

Now read `mean[c_in_group, H, W](x[..batch, group, c_in_group, H, W])`. The bracket names the reduced coordinates. If the layout changes, the names don't. Which line would you rather be responsible for maintaining six months from now, at 11 PM, during an incident?

---

**Chapter 6.** Causality. `u[t+1, i]` on the right-hand side of a recurrence with declaration `u[t in 1..T, i]`. Before this book, would you have caught it? Would the compiler? In Python, the loop runs. In Einlang, the compiler halts. The difference is whether time has a direction in your notation.

---

**Chapter 7.** The boundary. `oh + kh` is valid. `oh + 1000` is valid too—even if `oh + 1000` overflows the input. The compiler checks that `oh` and `kh` are coordinates. It does not check that their sum is in bounds.

This boundary is a design choice. Why is it here? Because the coordinate system guarantees that `oh` and `kh` are the right *kind* of thing—spatial indices. It does not guarantee that they are the right *value*. Bounds checking is the runtime's job. Semantic correctness is yours. The names reduce the surface area of uncheckable facts. They do not eliminate it.

---

**Chapter 8.** The gradient. Read `dA[i, k] = sum[j](dC[i, j] * B[k, j])`. You derived this yourself in Chapter 8, from the coordinate sets alone. Now read `dA = dC @ B.T`. Do you know which axes are being contracted? Do you know why the transpose is there? If you had to re-derive it from the positional code, could you?

The coordinate accounting gave you set subtraction: `C` has `{i, j}`, `A` has `{i, k}`, sum over `{j}`. The positional code gives you `dC @ B.T`. Both produce the same result. Only one explains itself.

---

**Chapters 9–11.** The comparisons. You saw LayerNorm, RMSNorm, GroupNorm, attention, and the heat equation in two notations. Each time, the positional code was correct. Each time, the named code made different facts visible.

After reading all three comparison chapters, answer this: which bug would you rather debug at 3 AM—a `dim=-1` that should have been `dim=-2`, or a `softmax[class]` where `class` doesn't exist? The first is silent. The second is a compiler error. The compiler cannot prevent all bugs. But it can prevent the ones where the notation records enough information for the compiler to check.

---

**Chapters 12–14.** The compiler. You watched a name travel from source through IR through analysis through lowering to generated code. Five forms, one name. `class` survived every stage. It was verified at the stage where it was still a name—where the compiler could ask "does this name exist on this tensor?" and get a yes/no answer. Then it was burned into `axis=1`—after all checks had passed.

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

Does a coordinate name surface—`channel`, `feature`, `class`? Does a question arise: which coordinate is position 1, and how do I know? Does the gap between the number and the identity feel wider than it did fourteen chapters ago?

If the answer to any of these is yes, the book has done its work. Not by converting you to Einlang. Not by persuading you to rename every dimension in your codebase. By changing what you notice when you read a tensor operation. The coordinate habit is that change.

---

Before you turn the final page, one last exercise. Open a terminal. Navigate to your most recent project. Pick the first `.py` file that contains a tensor operation. Find the first `dim=`, `axis=`, or `permute` call. Read it. Ask: if the dimension order changed tomorrow, would this line still be correct?

If you can answer with confidence—because a comment records the coordinate name, because an einops string names the dimensions, because the variable naming convention is consistent—you have applied the habit. If you cannot answer with confidence—because the number records only position, not identity—you have found a ghost. You now know its name. And you know that naming it is the first step toward making it visible.

The ghost has been there since before you opened this book. The difference is that now you can see it. And what you can see, you can name. And what you can name, you can check.

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

This is the coordinate audit from Chapter 16, applied to a live bug. Four questions. Which coordinate is consumed? Which coordinate is copied along? Can you trace a coordinate from source to destination? Does the backward reduction match the forward broadcast? Ask them of the operation that produced the unexpected shape. The answers will tell you whether the bug is in the shapes or in the semantics—whether the transpose is missing or the reduction is over the wrong axis.

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

There is a sentence I have been saving for the end. It is the one-sentence version of this book:

**Let your coordinate names be as visible as your intent—because six months from now, when you return to this code to fix a bug, the intent will be gone, and only the names will remain.**

---

The book opened with a bug. A tensor of shape `(32, 64, 256)`. A programmer wrote `x.mean(dim=1)`. The intent was "average over channels." The text said `dim=1`. Three months later, `channel` moved to position 2. The bug lived for three weeks.

The bug's name was never written. Its identity was in a comment, in a variable name, in the author's mental model. It was absent from the one place the compiler could see: the source text of the operation itself.

Notation determines what you can notice. When a notation has no place for a fact, that fact becomes invisible. The fact—"I am erasing `channel`, not `spatial`"—was absent from `dim=1` because `dim=1` is a number, and a number has no slot for a name.

You now know where the slot is. It is a bracket. It holds a name. And the name, once written, can be checked.

Turn the page.

