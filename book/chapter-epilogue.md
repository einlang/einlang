---
layout: book
title: "Epilogue · A Friend Named Einlang"
---

# Epilogue · A Friend Named Einlang

> "Programs must be written for people to read, and only incidentally for machines to execute."
>
> — Harold Abelson and Gerald Jay Sussman, *Structure and Interpretation of Computer Programs*

---

You just reviewed the complete grammar. Now step back and ask what it adds up to.

In Chapter 5, Einlang was a name. A label for a notation under construction. Now, at the end, the name has a referent.

Einlang is not a language in the sense that Python is a language, or C++, or Rust. It does not aspire to run your web server or render your UI. It is a language built on three ideas: primitive expressions, means of combination, and means of abstraction—organized around a single purpose.

The idea is that **coordinates have identities, and those identities belong in the source code.**

Everything else in the language serves that idea. Reductions name the coordinate they consume. Broadcasts name the coordinate they replicate along. Permutations state the coordinate correspondence explicitly. Functions declare which coordinates they use by identity, and the compiler checks those declarations at every call site. Gradients preserve coordinate structure through differentiation. Recurrences make the direction of time a syntactic constraint.

---

The SICP quotation that opens this epilogue is famous for a reason. It states a truth that is obvious once you hear it and difficult to practice consistently: code is communication between humans before it is instruction to machines.

The coordinate habit is an application of that truth to tensor programming. When you write `x.mean(dim=1)`, you are communicating to the machine ("reduce axis 1") but not to the human ("eliminate channel"). When you write `mean[channel](x)`, you are communicating to both. The machine still knows what to do. The human now also knows what you intended. And the compiler can check that your intent is consistent with the tensor's actual coordinate structure.

The bet: that the extra keystrokes of naming coordinates are repaid, with interest, in debugging sessions avoided, in refactoring confidence gained, in the quiet satisfaction of reading code that says what it means.

---

SICP ends with a meta-circular evaluator—a Scheme interpreter written in Scheme—as if to say: you now understand the language deeply enough to implement it yourself.

The corresponding question here is smaller and harder.

Take the four habits into your next tensor program. Not an Einlang program—the language is young, the tooling is sparse, and deadlines don't move. Take them into PyTorch. Into JAX. Into whatever framework gets the work done.

When you write a reduction, pause. Ask: which coordinate am I eliminating? Is the name in the code?

When you write a broadcast, pause. Ask: which coordinate am I copying along? Is independence genuinely justified?

When you write a permutation, pause. Ask: can I trace one coordinate from source to destination without reconstructing the position map?

When you inspect a gradient, pause. Ask: does the backward reduction match the forward broadcast?

These questions cost seconds. The bugs they catch cost hours. The ratio is favorable.

## The Life of a Name

A coordinate name in Einlang lives through four stages. It is **written** in source, as a letter between brackets: `[i, j]`, `[class]`, `[batch]`. It is **preserved** in the intermediate representation, stripped of syntax but keeping every name intact: `(let-decl (output C (i j)) ...)`. It is **verified** by analysis, where the compiler derives its range, checks its consistency across call sites, and records which operations consume it. And it is **burned** in lowering, translated into the integers that machines require: `class → axis=1`, `i → loop 0..b`.

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

**Chapter 11**: GroupNorm's reshape chain: `x.reshape(N, G, C//G, H, W).mean(dim=(2,3,4))`. The positions `(2,3,4)` are only correct after the reshape. If the reshape changes, the positions change. The named version `mean[c_in_group, ..s]` names the coordinates directly. The reshape is unnecessary because the coordinates are separate from the start.

**Chapter 12**: Self-attention and cross-attention in PyTorch have identical code. The difference is only in the shapes of the tensors passed at runtime. The named version distinguishes `self_attention[seq, ...]` from `cross_attention[seq_q, seq_k, ...]` in the type signatures. A reader can see which is which without checking runtime shapes.

Every one of these bugs was shape-correct. Every one survived the checks that positional frameworks perform. Every one was caught by a name.

---

## The Hermeneutics of Naming

Naming is a discipline—a habit to practice, an audit to perform. But there is a prior question: *what if you don't know what to call it?*

You are designing a new attention mechanism. An intermediate tensor has shape `(batch, heads, seq1, seq2, features)`. You stare at it. Is `seq1` the query sequence and `seq2` the key sequence? Are they symmetric? Should you call the last dimension `features` or `d_model` or `embedding`? You are not sure. The mechanism is still taking shape in your mind. Committing to a name feels premature—like naming a child before it is born.

So you write `dim=-1`. Not because you think positional notation is better. Not because you are lazy. But because the number `-1` does not ask you to decide what the dimension *means*. It asks only what the dimension *is*—the last one. And that you can answer.

This is **delayed commitment**. A name is a claim. A number is a placeholder. When you are in the early stages of designing a computation, you may not be ready to make the claim. The number lets you defer it. The number says: "I know where this dimension is. I do not yet know what it is."

The coordinate habit does not require you to name everything immediately. It requires you to name things before the code is read by someone else—including your future self. A number in a draft is a scaffold. A number in a merged pull request is a landmine. The difference is not whether the name exists at the moment of writing. The difference is whether the name exists at the moment of reading.

Practical advice: if you cannot name a dimension, write `dim=-1`—but write a comment next to it recording your uncertainty. `# dim=-1: last dim, currently feature-like but may change`. The comment is not a name. The comment is a flag. It tells the next reader: "the author was uncertain here. Check whether this dimension still means what you think it means." A number with a confession is better than a number with silent confidence.

And sometimes, after reflection, you realize the dimension genuinely does not have a stable identity. It is a transient intermediate that exists only inside this function, consumed by the next operation, never exposed to a caller. In that case, `dim=-1` may be the right choice permanently. Not every coordinate deserves a name. The coordinate habit is not a moral obligation. It is a judgment: *does the correctness of this operation depend on which coordinate this is?* If the answer is no, a number is fine. If the answer is yes, the name earns its keystrokes.

When you do commit to a name, remember the boundary from Chapter 14: the compiler cannot catch `softmax[batch]` where `softmax[class]` was intended. But a reader sees the wrong name and asks the question. The positional equivalent `softmax(logits, dim=0)` hides the error behind a number; the reader must reconstruct which coordinate axis 0 refers to, and the reconstruction may be wrong. The name earns its keystrokes twice: once by recording intent, once by making errors visible when intent is misrecorded.

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

The "if" is not hypothetical. Every one of these bugs has occurred in production, in codebases you have used, in frameworks you have imported. They were caught—eventually—by tests, by code review, by the programmer staring at shapes at 3 AM. The question is not *can these bugs be caught?* It is *can they be caught at compile time, by the compiler, from information already present in the code?* The answer, for every scenario above, is yes—if the information is in the code. And the notation determines whether it is.

---

## The Invariant

Chapter 14 surveyed the landscape — assertions, einops, named tensors, compiler. The coordinate habit works at every step. It only asks: *is the name in the code?* The habit does not prescribe the tool. It prescribes the information.

Fifteen chapters. One invariant. Say it once more before you go:

**Every tensor operation that depends on a coordinate's identity must record that identity in the source code.**

This is not a language rule. It is a practice rule. It applies in Einlang, in PyTorch, in JAX, in NumPy, in any framework where tensors carry coordinates that mean different things. The notation you use determines *how* you record the identity—brackets, comments, variable names, einops strings—but the invariant is the same.

The invariant does not prevent all bugs. It prevents the class of bugs where the coordinate identity was lost before the operation was performed. A reduction over `dim=1` does not know it's reducing over `channel`. A reduction over `channel` knows. When the channel moves to `dim=2`, the first reduction silently becomes wrong. The second reduction becomes a compile error. The difference is whether the identity was recorded.

You now know how to record it. The rest is practice.

---


## Day 100, Replayed

Two files. One refactoring. Three months.

`data.ein`, updated 90 days ago:

```rust
fn load_samples(path: &str) -> [f32; batch, spatial, feature] {
    // column 1 = batch, column 2 = spatial, column 3 = feature
    read_csv(path)
}
```

`model.ein`, untouched for 90 days:

```rust
fn predict(x: [f32; batch, channel, spatial]) -> [f32; batch, spatial] {
    mean[channel](x[batch, channel, spatial])
}
```

`main.ein`, today:

```rust
let x = load_samples("train.csv");
let y = predict(x);
```

Save.

```
error[E0061]: coordinate contract mismatch
  --> main.ein:2:16
   |
 2 | let y = predict(x);
   |                 ^
   |                 in call to `predict`
   |
   = argument `x`:
   =   provided:  batch, spatial, feature
   =   expected:  batch, channel, spatial
   |
   = missing coordinate:  `channel`
   = unexpected:           `feature`
   |
help: `channel` was renamed to `feature` in data.ein:1
help: update parameter declaration in model.ein:1:
      fn predict(x: [f32; batch, spatial, feature]) -> [f32; batch, spatial]
```

The positional equivalent:

```python
x = x.mean(dim=1)
```

`dim=1` was `channel` before the refactoring and `spatial` after. It compiled. It ran. It passed integration tests. It deployed to staging. It failed silently in production for three weeks. Found at 3 AM on Day 100, by a human tracing one number backward through twelve layers.

The fix: rename `channel` to `feature` in model.ein:1. Ten seconds.

---

## What the Coordinate Habit Does Not Solve

Chapter 14 catalogued the limits: names check consistency, not correctness; they do not replace testing; they cost keystrokes. Two limits belong here, at the end, because they define the boundary between this book's argument and its honest modesty.

**Names don't write the program.** The coordinate habit tells you to record which coordinate a reduction consumes. It does not tell you whether the reduction should be a mean or a sum, whether the normalization should be over `feature` or `batch`, whether the attention should be self or cross. Those decisions are modeling decisions. The names record them; they don't make them. A well-named wrong model is still a wrong model. The difference is that the names make the model's structure visible, so the next reader—the colleague, the reviewer, the future you—can see what the model assumed and judge whether the assumption still holds.

**Einlang itself is young.** The language in these pages is a research prototype—no CUDA backend, no package manager, no PyTorch integration. The coordinate habit works through comments, einops strings, and naming conventions in any framework today. But if you want to build the rest: the IR, the check rules, and the lowering pass described in Chapters 9 and 10 are a starting point. The distance from here to a production compiler is measured in engineering years, not ideas. The ideas are in this book. They are ready. The compiler will catch up.

---

## If You Want More

Three works shaped the thinking behind these pages.

**Structure and Interpretation of Computer Programs** (Abelson, Sussman, and Sussman) taught generations of programmers that a language is built from primitive expressions, means of combination, and means of abstraction—and that building a metacircular evaluator is the final proof of understanding.

**Learn You a Haskell for Great Good** (Miran Lipovača) showed that a book about a programming language can be warm, direct, and relentlessly focused on the reader's understanding rather than the author's expertise.

**"Tensor Considered Harmful"** (Aleksander Mądry, 2018) and the named-tensor work at Harvard and Stanford asked: what happens when tensor dimensions have names that the compiler can check?

They point to a destination—code that says what it means—and leave the walking to you.

---

## Close the Cover

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

Close the cover. Open your editor. Read the first tensor line you see. Ask the question that the preceding pages have taught you to ask:

**Where is the name?**

If the answer is *nowhere*—you have found your starting point.
