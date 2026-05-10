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

**Day 0.** A tensor has shape `(32, 64, 256)`. A programmer writes `x.mean(dim=1)`. The intent is "average over channels." The text says `dim=1`. If names existed: `mean[channel](x[batch, channel, spatial])`. The compiler would check that `channel` exists on `x`. It does. The check passes. The name records the intent. The bug has not yet occurred—but the slot for catching it already exists.

**Day 90.** Channel moves to position 2. The shape is now `(32, 256, 64)`. `x.mean(dim=1)` silently erases `spatial`. If names existed: `mean[channel](x[...])` — `channel` is still `channel`. The coordinate declaration updated from `[batch, channel, spatial]` to `[batch, spatial, channel]`. The reduction still names the right coordinate. The positional code is now silently wrong. The named code never noticed the change, because the name absorbed it.

**Day 100.** The model deploys. The customer complaint arrives. The bug lived for ten days in production, and for ninety days in the code—ever since the line was written, when the programmer knew `dim=1` meant `channel` but had no way to record it. If names had been there from Day 0: the refactoring at Day 90 would have updated the coordinate declaration, the name `channel` would have stayed attached to the right axis, and `mean[channel]` would have remained correct. There would have been no bug to deploy on Day 100.

The ninety-day gap—between when the intent was clear and when the bug was discovered—is the distance between a comment and a contract. Day 0 to Day 100 is the life of a bug that names prevent. Every chapter between Chapter 1 and this one filled a different piece of that prevention. Now let's trace what each piece caught.

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

But names do not catch everything. The boundary from Chapter 7 still holds: named coordinates verify that a coordinate *exists* on a tensor. They do not verify that index arithmetic stays within bounds. `oh + kh` is syntactically valid as long as both `oh` and `kh` are declared coordinates. Whether `oh + kh` exceeds the input's spatial extent is a runtime question. The names reduce the surface area of uncheckable facts. They do not eliminate it. A comment that says "channel" can be wrong. A coordinate named `channel` can also be wrong—you might have named the wrong axis `channel`. But the coordinate, once named, is checked for consistency everywhere it appears. The comment is checked nowhere. The coordinate is a fact. The comment is a hope.

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

Replay the book's key moments with names present from the start—not as a thought experiment, but as a counterfactual: what would have been different?

The bug that opened Chapter 1 would not have occurred. When the programmer refactored `channel` from position 1 to position 2, the coordinate declaration would have changed from `[batch, channel, spatial]` to `[batch, spatial, channel]`. The line `mean[channel](x)` would have compiled without error—because `channel` still exists on `x`. The name absorbs the layout change. The positional version `dim=1` silently changes meaning; the named version `mean[channel]` doesn't. This is the simplest case, and the name handles it completely.

A more interesting scenario: `A + bias` becomes `A[i, j] + bias[j]`. Six months later, a programmer adds a time dimension to `A`—it becomes `A[t, i, j]`. The broadcast `bias[j]` now omits both `t` and `i`. In positional NumPy, the broadcast silently extends to the new leading dimension and the gradient sums over both. The shapes match. But a bias independent of `i` (sample) might not be independent of `t` (time). The name `t` is present in `A` but absent from `bias`—the omission is a visible claim: bias doesn't depend on time. The programmer reviewing the diff sees `bias[j]` where `A` now has `A[t, i, j]` and asks: *should bias be constant across time?* The name doesn't catch the semantic error—it makes the assumption visible so the programmer can catch it.

The Square Matrix Test becomes a development-time check rather than a dataset-dependent time bomb. A classifier with `batch_size = num_classes = 64` trains perfectly. Then a new dataset arrives with `num_classes = 100`. The positional code `softmax(logits, dim=-1)` had been normalizing over the last axis—which was `class` by coincidence, because `batch` and `class` were both 64. The named code `softmax[class](logits[batch, class])` normalizes over `class` regardless of whether the extents are equal. The name records the intent, and the intent doesn't change when the data changes.

The GroupNorm reshape chain becomes a single line with no reshape. `x.reshape(N, G, C//G, H, W).mean(dim=(2,3,4))` becomes `mean[c_in_group, H, W](x[batch, group, c_in_group, H, W])`. The positions `(2,3,4)` are only correct after the reshape. If the reshape changes, the tuple changes. The names `c_in_group`, `H`, `W` name the reduced coordinates directly, and the coordinates are separate from the start—no reshape needed.

Self-attention and cross-attention become visibly distinct at the call site. In PyTorch they are identical code; the difference is only in runtime shapes. A programmer debugging a cross-attention bug prints shapes: `q: (batch, seq, d)`, `k: (batch, seq, d)`. The shapes match—it's self-attention. With names, `self_attention[seq, ...]` and `cross_attention[seq_q, seq_k, ...]` are different signatures. `seq_q` and `seq_k` are different coordinate names. The reader sees which is which without checking runtime shapes.

The recurrence bug becomes a compile-time error. `h[t] = h[t+1] + f(...)` runs in Python. The result is garbage or an IndexError. With names, the causality check rejects `t+1` because it is a forward reference. One character changes the program from correct to wrong, and the name `t`—combined with the recurrence domain—gives the compiler enough information to reject the wrong one.

The gradient's backward sum stabilizes across shape changes. Forward: `factor[j]` broadcasts over `i`. Backward: `d_factor[j] = sum[i](d_scaled[i, j] * x[i, j])`. If `factor` changes from 1D `(j,)` to 2D `(i, j)`, a positional backward pass changes its sum silently. With names, the broadcast record says `factor omits {i}`. The backward sum is over `{i}`—unchanged by the shape change, because the name `i` is still the omitted coordinate.

Six scenarios. In each, the positional code could be correct—it is correct, in the hands of a careful programmer. The question is not whether positional code can be correct. It is whether the notation makes the correctness checkable. In every scenario, the named code records the coordinate identity that correctness depends on. The positional code does not. The identity lives in the programmer's head, in a comment, or nowhere. The names move it into the source—where the compiler and the next reader can both check it.

The "if" is not hypothetical. Every one of these bugs has occurred in production, in codebases you have used, in frameworks you have imported. They were caught—eventually—by tests, by code review, by the programmer staring at shapes at 3 AM. The question the book has asked is not *can these bugs be caught?* It is *can they be caught at compile time, by the compiler, from information already present in the code?* The answer, for every scenario above, is yes—if the information is in the code. And the notation determines whether it is.

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

An outline is not a review. A review is not something you read—it is something you do.

Open a terminal. Open your most recent project. Find a file with at least ten tensor operations. You are going to perform the coordinate audit on your own code, without a chapter-by-chapter guide, without prompts. The questions are the same ones that recurred through every chapter. You already know them.

For each `dim=`, `axis=`, or `permute` in the file:

1. Which coordinate is being consumed, copied, or moved?
2. Is its name recorded anywhere the next reader can see?
3. If the dimension order changed tomorrow, would the operation still be correct?

For each broadcast: which coordinate is the value silent on? Is the silence justified, or is it a convenience that will become a bug when the shape changes?

For each function: does its signature declare which coordinates it consumes and which it preserves? If a caller passed the wrong coordinate, would anything catch it?

If you answered "I don't know" to any of these—you found a gap. Not your fault. The notation had no slot. But you now know the slot exists. Put the name in a comment. Put it in an einops string. Put it wherever the next reader—who may be you, at 3 AM—will see it.

Now read this line one more time:

```python
x = x.mean(dim=1)
```

What appears in your mind? Does a coordinate name surface? Does the gap between the number and the identity feel wider than it did fifteen chapters ago?

If yes, the book has done its work. Not by converting you to Einlang. By changing what you notice when you read a tensor operation. The coordinate habit is that change.

---

Before you turn the final page, one last exercise. Open a terminal. Navigate to your most recent project. Pick the first `.py` file that contains a tensor operation. Find the first `dim=`, `axis=`, or `permute` call. Read it. Ask: if the dimension order changed tomorrow, would this line still be correct?

If you can answer with confidence—because a comment records the coordinate name, because an einops string names the dimensions, because the variable naming convention is consistent—you have applied the habit. If you cannot answer with confidence—because the number records only position, not identity—you have found a ghost. You now know its name. And you know that naming it is the first step toward making it visible.

The ghost has been there since before you opened this book. The difference is that now you can see it. And what you can see, you can name. And what you can name, you can check.

---

There is a sentence I have been saving for the end. It is the one-sentence version of this book:

**Let your coordinate names be as visible as your intent—because six months from now, when you return to this code to fix a bug, the intent will be gone, and only the names will remain.**

---

The book opened with a bug. A tensor of shape `(32, 64, 256)`. A programmer wrote `x.mean(dim=1)`. The intent was "average over channels." The text said `dim=1`. Three months later, `channel` moved to position 2. The bug lived for three weeks.

The bug's name was never written. Its identity was in a comment, in a variable name, in the author's mental model. It was absent from the one place the compiler could see: the source text of the operation itself.

Notation determines what you can notice. When a notation has no place for a fact, that fact becomes invisible. The fact—"I am erasing `channel`, not `spatial`"—was absent from `dim=1` because `dim=1` is a number, and a number has no slot for a name.

You now know where the slot is. It is a bracket. It holds a name. And the name, once written, can be checked.

Turn the page.

