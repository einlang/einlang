---
layout: book
title: "Epilogue · A Friend Named Einlang"
---

# Epilogue · A Friend Named Einlang

> "Programs must be written for people to read, and only incidentally for machines to execute."
>
> — Harold Abelson and Gerald Jay Sussman, *Structure and Interpretation of Computer Programs*

---

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

Close the cover. Open your editor. Read the first tensor line you see. Ask the question that the preceding pages have taught you to ask:

**Where is the name?**

If the answer is *nowhere*—you have found your starting point.

---


---

---

## The Life of a Name

A coordinate name in Einlang lives through four stages. It is **written** in source, as a letter between brackets: `[i, j]`, `[class]`, `[batch]`. It is **preserved** in the intermediate representation, stripped of syntax but keeping every name intact: `(let-decl (output C (i j)) ...)`. It is **verified** by analysis, where the compiler derives its range, checks its consistency across call sites, and records which operations consume it. And it is **burned** in lowering, translated into the integers that machines require: `class → axis=1`, `i → loop 0..batch`.

Four stages. Written, preserved, verified, burned. At no point is the name decoration. At every point, it is load-bearing.

A decoration can be omitted without consequence. You can remove the comment `# dim=1 is channel` and the code still runs. You can rename the variable `spatial_features` to `x` and the compiler says nothing. Decorations are for humans. The machine does not read them.

A name in Einlang is not a decoration. The compiler reads it. The checker verifies it. The lowering pass translates it. If you write `softmax[class](logits)` and `logits` has no coordinate called `class`, the compiler stops. Not at runtime—at compile time. The name is part of the contract.

This is the difference between a comment and a coordinate. A comment records intent. A coordinate enforces it.

---

---

## What Names Caught

Walk back through the book and ask, at each stage: what would a positional notation have let through?

**Chapter 1**: `x.mean(dim=1)`. The positional notation let it through for three weeks. The named version `mean[channel](x)` would have broken at compile time the moment `channel` moved to position 2, because the tensor would no longer have a coordinate called `channel` at the position the compiler expected. Or, more precisely: the compiler would have required the refactoring programmer to update the coordinate declaration, and that update would have surfaced the fact that `mean[channel]` was still referencing the old layout.

**Chapter 2**: `A + bias`. The positional notation broadcasts `bias` along whichever dimensions happen to be missing. If `A` changes from `(batch, feature)` to `(feature, batch)`, the broadcast silently flips. The named version `out[i, j] = A[i, j] + bias[j]` makes the omission visible: `bias` has no `i`, so it broadcasts over `i`. If `i` and `j` swap meanings upstream, the indexing pattern breaks visibly.

**Chapter 3**: `softmax(logits, dim=-1)`. When `batch_size == num_classes`, the square matrix test applies: `softmax(logits, dim=0)` and `softmax(logits, dim=-1)` both produce valid probability distributions. The named version `softmax[class](logits)` does not let you silently normalize over `batch`. The name `class` is either present on `logits` or it isn't.

**Chapter 6**: `u[t, i] = u[t+1, i] + f(...)`. In a positional recurrence, writing `t+1` instead of `t-1` produces a forward reference—a read from the future. The positional loop runs. The values are whatever was in memory. The named version rejects it: the causality check sees `t+1 > t` and halts.

**Chapter 8**: The gradient of a broadcast. Forward: `bias` omits `batch`, broadcasting over it. Backward: the gradient must sum over `batch` to recover `bias`'s shape. In a positional framework, this sum is implicit in the autodiff engine. If the broadcast changes because the shape changed, the gradient sum changes with it—silently. In the named version, the coordinate sets tell you exactly what the gradient must sum over: `C` has `{i, j}`, `A` has `{i, k}`, sum over `{j}`. The set subtraction is checkable.

**Chapter 11**: GroupNorm's reshape chain: `x.reshape(N, G, C//G, H, W).mean(dim=(2,3,4))`. The positions `(2,3,4)` are only correct after the reshape. If the reshape changes, the positions change. The named version `mean[c_in_group, ..spatial]` names the coordinates directly. The reshape is unnecessary because the coordinates are separate from the start.

**Chapter 12**: Self-attention and cross-attention in PyTorch have identical code. The difference is only in the shapes of the tensors passed at runtime. The named version distinguishes `self_attention[seq, ...]` from `cross_attention[seq_q, seq_k, ...]` in the type signatures. A reader can see which is which without checking runtime shapes.

Every one of these bugs was shape-correct. Every one survived the checks that positional frameworks perform. Every one was caught by a name.

---

---

## The Hermeneutics of Naming

Naming is a discipline—a habit to practice, an audit to perform. But there is a prior question: *what if you don't know what to call it?*

You are designing a new attention mechanism. An intermediate tensor has shape `(batch, heads, seq1, seq2, features)`. You stare at it. Is `seq1` the query sequence and `seq2` the key sequence? Are they symmetric? Should you call the last dimension `features` or `d_model` or `embedding`? You are not sure. The mechanism is still taking shape in your mind. Committing to a name feels premature—like naming a child before it is born.

So you write `dim=-1`. Not because you think positional notation is better. Not because you are lazy. But because the number `-1` does not ask you to decide what the dimension *means*. It asks only what the dimension *is*—the last one. And that you can answer.

This is **delayed commitment**. A name is a claim. A number is a placeholder. When you are in the early stages of designing a computation, you may not be ready to make the claim. The number lets you defer it. The number says: "I know where this dimension is. I do not yet know what it is."

The coordinate habit does not require you to name everything immediately. It requires you to name things before the code is read by someone else—including your future self. A number in a draft is a scaffold. A number in a merged pull request is a landmine. The difference is not whether the name exists at the moment of writing. The difference is whether the name exists at the moment of reading.

Practical advice: if you cannot name a dimension, write `dim=-1`—but write a comment next to it recording your uncertainty. `# dim=-1: last dim, currently feature-like but may change`. The comment is not a name. The comment is a flag. It tells the next reader: "the author was uncertain here. Check whether this dimension still means what you think it means." A number with a confession is better than a number with silent confidence.

And sometimes, after reflection, you realize the dimension genuinely does not have a stable identity. It is a transient intermediate that exists only inside this function, consumed by the next operation, never exposed to a caller. In that case, `dim=-1` may be the right choice permanently. Not every coordinate deserves a name. The coordinate habit is not a moral obligation. It is a judgment: *does the correctness of this operation depend on which coordinate this is?* If the answer is no, a number is fine. If the answer is yes, the name earns its keystrokes.

When you do commit to a name, remember this: a wrong name is a visible error. Chapter 14 explored the boundary—`softmax[batch]` where `softmax[class]` was intended compiles without error. But a reader sees the wrong name and catches the mistake. The positional equivalent `softmax(logits, dim=0)` hides the error behind a number; the reader must reconstruct which coordinate axis 0 refers to, and the reconstruction may be wrong. A wrong name is visible. A missing name is invisible. The name earns its keystrokes twice: once by recording intent, once by making errors visible when intent is misrecorded.

---

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

---

## Audit in the Wild: The Middle Grounds

Chapter 14 surveyed the landscape between pure positional notation and a complete named-coordinate compiler: defensive assertions, einops, PyTorch Named Tensors, and Einlang's compiler. The distance between *no checking* and *complete checking* is measurable. Einops catches local errors. Named tensors catch errors that survive through supported operations. Einlang catches all of them—at the cost of a compiler.

The coordinate habit works at every step. It only asks: *is the name in the code?* In an einops string, that's a name. In a PyTorch named tensor, that's a name. In a bracket that a compiler checks, that's a name with a guarantee. The habit does not prescribe the tool. It prescribes the information.
## The Invariant

Fifteen chapters. One invariant. Say it once more before you go:

**Every tensor operation that depends on a coordinate's identity must record that identity in the source code.**

This is not a language rule. It is a practice rule. It applies in Einlang, in PyTorch, in JAX, in NumPy, in any framework where tensors carry coordinates that mean different things. The notation you use determines *how* you record the identity—brackets, comments, variable names, einops strings—but the invariant is the same.

The invariant does not prevent all bugs. It prevents the class of bugs where the coordinate identity was lost before the operation was performed. A reduction over `dim=1` does not know it's reducing over `channel`. A reduction over `channel` knows. When the channel moves to `dim=2`, the first reduction silently becomes wrong. The second reduction becomes a compile error. The difference is whether the identity was recorded.

You now know how to record it. The rest is practice.

---

## How to Start

You don't need Einlang to practice the coordinate habit. Start at the data-entry boundary: when tensors enter your program, put the coordinate names where they can be seen—a docstring, a comment, a naming convention. Then name the reductions (`# dim=1 = channel`), because consumption is the hardest fact to reconstruct later. Use einops strings where they fit, variable name conventions (`x_batch_channel_spatial`) where they don't, and a code review checklist—which coordinate does this integer refer to? would it change if the dimension order changed?—where neither is available. The bridge doesn't have to be perfect. It has to be there. Put the name where the next reader can see it. Not in a design doc. Not in a Slack message. In the code, at the operation whose correctness depends on it.

---

## Three Scenarios

**The legacy codebase.** You inherit a PyTorch model with 200 occurrences of `dim=-1`. Spend one afternoon adding a comment at every `dim` argument: `# dim=-1 = feature`, `# dim=1 = channel`. One afternoon now beats ten afternoons of shape-tracing over the next year. The names need to be visible—not checked, not guaranteed, just visible.

**The new project.** Name your dimensions at the data loader, not in the model. The moment a tensor enters your program, attach coordinate names—in a docstring, in a convention (`batch` always first, `spatial` always last), in a project README. Six months from now, the convention tells you what `dim=1` means.

**The bug investigation.** Before you print another shape, write down which coordinate you think each dimension is. If `x.mean(dim=0)` is normalizing over `batch`, something is wrong—regardless of whether the shapes match. Which coordinate is consumed? Is the answer visible in the code? The question is the audit.

---

## A Practical Guide for Non-Migrators

The coordinate habit does not require Einlang. It requires only that you put the name where the next reader can see it. Here are the patterns that work in PyTorch, JAX, and NumPy today.

**At the data loader.** The moment a tensor enters your program, its coordinates have identities. Record them before they are lost.

```python
# x: (batch, channel, spatial) — order guaranteed by DataLoader
x = next(iter(dataloader))
```

This is a single line. It costs nothing to maintain—when the DataLoader changes, the comment is right next to the code that needs updating. Six months later, a reader tracing `x.mean(dim=1)` through the model sees the comment and knows: `dim=1` is `channel`.

**At every reduction.** A `dim=` argument consumes a coordinate. Which one? Write it.

```python
h = x.mean(dim=1)          # dim=1 = channel
h = logits.softmax(dim=-1)  # dim=-1 = class
h = scores.sum(dim=(2, 3))  # dims=(2,3) = (height, width)
```

The comment records intent. When a refactor changes the shape, the comment is a flag: *this integer should match the coordinate named here*. If they no longer match, the reader knows to investigate.

**At every broadcast.** An operation between tensors of different ranks is a broadcast. Which coordinates are being replicated? Write the pattern.

```python
# broadcasting: bias[channel] over (batch, channel)
out = x + bias

# broadcasting: scale[1, channel, 1, 1] over (batch, channel, height, width)
out = x * scale
```

The comment makes the silence audible. It records which coordinates the smaller tensor is silent on—the same information the compiler's broadcast ledger would record.

**At every reshape.** A reshape changes the coordinate layout. What was the layout before? What is it after? The names answer both questions.

```python
# (batch, group, c_per_group, height, width) -> (batch, group, -1)
x = x.reshape(batch, group, -1)
```

The comment is the map from the old layout to the new one. Without it, the reader must reconstruct the layout from context—or run the code and print shapes.

**At every permutation.** A `permute`, `transpose`, or `swapaxes` reorders coordinates. Which coordinates moved? Write the correspondence.

```python
# (batch, seq, heads, d_head) -> (batch, heads, seq, d_head)
x = x.permute(0, 2, 1, 3)
```

Or use einops, which records the correspondence as part of the expression:

```python
x = rearrange(x, "batch seq heads d -> batch heads seq d")
```

The einops string is checked at runtime. The comment is not. Both record the intent. Choose based on whether you need the runtime check.

**At function boundaries.** A function that takes a tensor and returns a tensor has a coordinate contract. What does it consume? What does it produce? Write the contract in the docstring.

```python
def layer_norm(x: Tensor) -> Tensor:
    """
    x: (batch, ..., feature)
    Returns: (batch, ..., feature)
    Normalizes over: feature
    """
    mean = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, keepdim=True)
    return (x - mean) / (var + eps).sqrt() * gamma + beta
```

A reader of the call site does not need to read the implementation. The docstring tells them which coordinate is consumed and which survive. The contract is not checked by the compiler, but it is checked by the next programmer—and that is enough to catch the mistake where the author intended `layer_norm` but the reader expects `instance_norm`.

**In code review.** Add one question to the checklist: *for each `dim=` argument in this diff, is the coordinate identity documented?* If the answer is no, ask the author to add a comment. The habit compounds.

**When you can't name it.** Write the number. But write the uncertainty too.

```python
x = x.mean(dim=-1)  # dim=-1: last dim, currently feature-like; may change
```

The confession is better than silence.

---

Each of these patterns is a bridge. It does not check. It does not enforce. It does not survive a refactoring automatically. But it records. The name moves from the programmer's head into the source file, where the next reader—the colleague, the reviewer, the future you—can find it. The bridge is imperfect. But a bridge that exists is better than a bridge that was never built.

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

These are not tutorials. They are not documentation. They are road signs. They point to a destination—code that says what it means—and leave the walking to you.

You know the question now. It's the one you ask at every tensor line. It doesn't require a new language. It doesn't require a compiler. It requires only that you pause, look at the integer after `dim=`, and refuse to let the coordinate's identity stay silent.

Where is the name?

Close the cover.
