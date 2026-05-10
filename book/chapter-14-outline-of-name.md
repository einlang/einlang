---
layout: book
title: "Chapter 14 · The Outline of the Name"
---

# Chapter 14 · The Outline of the Name

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

Now read this line:

```python
x = x.mean(dim=1)
```

What appears in your mind?

