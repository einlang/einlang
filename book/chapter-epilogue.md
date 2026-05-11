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

**Chapter 12**: GroupNorm's reshape chain: `x.reshape(N, G, C//G, H, W).mean(dim=(2,3,4))`. The positions `(2,3,4)` are only correct after the reshape. If the reshape changes, the positions change. The named version `mean[c_in_group, ..spatial]` names the coordinates directly. The reshape is unnecessary because the coordinates are separate from the start.

**Chapter 13**: Self-attention and cross-attention in PyTorch have identical code. The difference is only in the shapes of the tensors passed at runtime. The named version distinguishes `self_attention[seq, ...]` from `cross_attention[seq_q, seq_k, ...]` in the type signatures. A reader can see which is which without checking runtime shapes.

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

---

---

## When the Name Is Wrong

The Panorama traced `class` through a journey where every check passed. But what happens when the name is wrong?

Suppose the programmer writes `softmax[batch](logits[batch, class])`. `batch` is the coordinate argument. The parser accepts it. The resolver finds it on `logits`. Analysis may or may not catch it, depending on how packs are bound. The code could compile. It would produce a valid probability distribution—over the batch dimension, not the class dimension.

The name was wrong. The check passed. The program is incorrect.

This is the boundary from Chapter 7, restated for the final time: **names check consistency, not correctness.** `softmax[batch]` is internally consistent—every reduction, broadcast, and gradient aligns over `batch`. The error is that the programmer wanted `class`. The compiler cannot read the programmer's mind. It can only verify the contract the programmer wrote.

But the name `batch` is visible. When the next programmer reads `softmax[batch](logits)`, they see the error immediately. The positional equivalent `softmax(logits, dim=0)` hides the error behind a number. The reader sees `dim=0` and must reconstruct whether axis 0 is batch or class. The reconstruction may be wrong.

A wrong name is a visible error. A missing name is an invisible one.

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

The comparison has been between two endpoints: pure positional notation (`dim=-1`) and a complete named-coordinate compiler. But the real world is not a binary choice. Between those two endpoints lie several intermediate solutions. Each catches *some* of the bugs that pure positional notation misses. None catches all of them. Here is how each one fares against the channel-drift bug from Chapter 1.

The bug: a tensor declared as `(batch, channel, spatial)` is refactored to `(batch, spatial, channel)`. `x.mean(dim=1)` silently changes from averaging over `channel` to averaging over `spatial`.

**Defensive assertions.** A `assert x.shape[1] == old_channel_size` at the data-loader boundary catches the refactoring when the sizes differ. But if `channel` and `spatial` happen to have the same extent, the assertion passes silently. Assertions check shapes, not identities. They protect against size mismatches, not semantic drift. And they must be maintained separately from the operations they protect—when a new dimension is added, every assertion downstream must be updated.

**Einops.** `rearrange(x, "batch channel spatial -> batch spatial channel")` makes the permutation visible and self-documenting. `reduce(x, "batch channel spatial -> batch spatial", "mean")` names the reduced coordinate at the call site. Einops is locally excellent—within a single expression, it records coordinate identities with clarity comparable to Einlang. But einops strings are not part of the type system. They do not propagate across function boundaries. A function that receives an einops-rearranged tensor has no way to know what the dimensions are called. The contract is local. The check is local. The name dies at the edge of the expression.

**PyTorch Named Tensors.** `x.rename("batch", "channel", "spatial").mean("channel")` checks that `channel` exists on `x` and eliminates it. Named tensors propagate names through operations, catching broadcasts that align the wrong dimensions. But the named tensor implementation is incomplete—many operations strip names silently (`torch.matmul`, `torch.cat`, most `torch.nn` layers). When a name is stripped, the protection vanishes without warning. The system catches some bugs but cannot guarantee that a name survives through an entire model. The contract is partial. The check is partial. The name can fall off without the programmer knowing.

**Einlang's compiler.** The coordinate contract is part of the function type. Every call site is checked. Every operation preserves names or explicitly consumes them. The name `channel` survives from data entry through the entire computation. If an operation strips it, the compiler reports a contract violation at the call site—not at runtime, not when the shapes happen to diverge, but at the moment the code is written. The contract is global. The check is complete.

Einops requires no new tooling—it is a Python library. PyTorch named tensors require no new compiler—they are built into the framework. Defensive assertions require nothing at all. Einlang requires a compiler that does not exist in production. The intermediate solutions are available today. They catch a meaningful subset of bugs. They are strictly better than pure positional notation.

The argument is not that Einlang is the only solution. It is that the distance between *no checking* and *complete checking* is measurable, and that every step along that distance catches more bugs. Einops catches the local ones. Named tensors catch the ones that survive through supported operations. Einlang catches all of them—but at the cost of a compiler that must be built. Which step you take depends on how much correctness you need and how much infrastructure you can afford. The coordinate habit works at every step. It only asks: *is the name in the code?* If it is in an einops string, that's a name. If it is in a PyTorch named tensor, that's a name. If it is in a bracket that a compiler checks, that's a name with a guarantee. The habit does not prescribe the tool. It prescribes the information.
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

Here are five specific techniques. None requires a new language. Each works in PyTorch, JAX, or NumPy today.

### 1. The comment beside the integer

Every `dim=` or `axis=` argument is a place where a coordinate name was lost. Put it back:

```python
x.mean(dim=1)            # dim 1 = channel
logits.softmax(dim=-1)    # dim -1 = class
x.permute(0, 2, 1)        # batch height width -> batch width height
```

The comment format is always the same: `# dim <number> = <name>`. One name per integer. The comment lives on the same line as the operation, not in a docstring three functions away.

When the dimension order changes, the comment drifts. But it drifts *visibly* — `# dim 1 = channel` on a line where `dim=1` is now spatial jumps out during code review. A comment that drifted is a flag. A missing comment is invisible.

### 2. Einops: what it catches, what it doesn't

```python
from einops import rearrange, reduce, einsum

# Instead of x.permute(0, 2, 1)
rearrange(x, "batch height width -> batch width height")

# Instead of x.mean(dim=1)
reduce(x, "batch channel spatial -> batch spatial", "mean")

# Instead of torch.matmul(A, B)
einsum(A, B, "batch in, out in -> batch out")
```

What einops catches: the coordinate name is in the expression string. A reader sees `"batch channel spatial -> batch spatial"` and knows `channel` was consumed. This is local — within one expression, the names are present.

What einops does not catch: names do not propagate across function boundaries. A function that receives an einops-rearranged tensor has no way to read the coordinate names — they are in the `rearrange` string of the caller, not in the tensor's type. The contract is local. The check is local. The name dies at the edge of the expression.

Einops is a complement to the coordinate habit, not a replacement. Use it to make coordinate identities visible within expressions. Use comments and conventions to carry those identities across function boundaries.

### 3. The code review checklist

When reviewing a pull request, for every `dim=`, `axis=`, `permute`, `transpose`, or `reshape`, ask four questions:

1. **Which coordinate does this integer refer to?** Is the answer visible in the code, or only in the author's head?
2. **Would the integer change if the dimension order changed upstream?** If yes, the code is fragile to layout refactoring.
3. **Is there a `keepdim` or equivalent?** If yes, a broadcast is happening. Which coordinate is being broadcast along?
4. **Are two integers with different meanings adjacent?** `dim=(2, 3, 4)` often means spatial dimensions — but only by convention. Should there be a comment?

If any question cannot be answered from the code alone, request a comment. The comment costs one line. The bug it prevents costs hours.

### 4. Variable naming as poor man's named coordinates

When a framework doesn't support named dimensions, the variable name carries the coordinate identity:

```python
# Instead of x, y, z:
x_batch_channel_spatial = load_data()
logits_batch_class = model(x_batch_channel_spatial)
probs_batch_class = logits_batch_class.softmax(dim=-1)  # dim -1 = class
```

The convention: `name_coord1_coord2_...`. The coordinates are listed in order. A tensor named `x_batch_channel_spatial` declares: batch is dim 0, channel is dim 1, spatial is dim 2. The name is the documentation. It is also the audit trail — when `dim=1` appears, the reader glances at the variable name and confirms `channel` is indeed dim 1.

This is the poorest form of named coordinates. The compiler cannot check the names. They can drift. But they are *in the code*, and that already makes them more useful than names in the programmer's head. A naming convention costs nothing to adopt and survives until the next refactoring — which is longer than most comments.

### 5. The data loading boundary

There is exactly one place in every program where coordinate identities are known with certainty: where tensors enter the program.

```python
def load_data(path: str) -> tuple[Tensor, Tensor]:
    """Returns (features, labels).

    features: (batch, channel, spatial) — float32
    labels:   (batch, class) — int64
    """
    ...
    return features, labels
```

The data loader is the only function that knows what each dimension actually *is*. The CSV column `class` is the class dimension. The image channel count is the channel dimension. After the data loader, every tensor is a derived quantity — its dimensions inherit meaning from the data loader's output, but that meaning is encoded nowhere.

Put the coordinate declaration at this boundary. A docstring is one form. A named tensor (PyTorch `refine_names`, xarray `DataArray`) is stronger. A comment at the return statement is the minimum.

If the data loader declares coordinate identities, and every downstream operation that depends on those identities carries a comment stating the dependency, the coordinate information flows from entry to loss without gaps. The gaps are where the bugs hide.

These five techniques share a structure: put the name where the next reader can see it. Not in a design doc. Not in a meeting note. Not in the variable names of a different file. In the code, at the operation whose correctness depends on it.

If you completed Chapters 12–14, you have something else: a miniature compiler. It checks five rules. It is not industrial-grade, but it is yours. When you write `x.mean(dim=1)` in PyTorch tomorrow, that compiler runs in your head. It notices that the reduction consumes a coordinate with no name. It asks: *which coordinate?* The compiler cannot halt your Python program—but you can.

Think about what that means. The compiler you built is not primarily a tool for generating NumPy code. It is an **exoskeleton for the mind**—a way to externalize the coordinate-tracking that expert programmers already do in their heads, silently, without being taught. When you manually audit a broadcast by comparing coordinate sets, you are running the compiler by hand. The S-expression IR is a formalism for what your working memory does when you trace a coordinate from data entry to loss. The check rules are a written-down version of the questions you ask yourself when something feels wrong about a shape.

The compiler's main value may not be as industrial software. It is as an internalized sense—a form of hearing. After these chapters, you cannot un-hear the silence where a coordinate name should be. You see `dim=-1` and a part of your mind automatically annotates it: *(reduction mean (axis -1 unknown-coordinate))*. You see a broadcast and your working memory performs coordinate set subtraction whether you ask it to or not. The compiler is not installed on your machine. It is installed in your attention. That is what the construction chapters built. Not a tool you install. A reflex you keep.

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

Named coordinates have limits. It is worth naming them before you leave, so you do not carry false expectations into your next project.

**Names do not guarantee correctness.** You can name the wrong coordinate. `mean[channel](x)` where you should have written `mean[spatial](x)` compiles without error. The name `channel` exists on `x`. The reduction is well-formed. The gradient will be correct—for the wrong reduction. Names catch inconsistency. They do not catch wrongness. A coordinate named `channel` that is actually `spatial` in the data is a semantic error, and semantic errors survive any notation.

**Names do not replace testing.** The compiler checks that the coordinate structure is internally consistent. It does not check that the computation achieves what you intended. A softmax normalized over `batch` instead of `class` is internally consistent—every reduction, broadcast, and gradient aligns perfectly. The program compiles. It is still wrong. Only a test that checks the output's shape and statistical properties would catch it.

**Names do not eliminate runtime shape errors.** Dynamic dimensions—sequence lengths, batch sizes that vary per call—cannot be checked at compile time. The compiler can verify that `seq` is a declared coordinate and that functions consuming it have consistent contracts. It cannot verify that `seq` has length 64 rather than 128. That check lives at runtime, in an assertion or a shape guard.

**Names cost keystrokes.** `mean[channel](x)` is longer than `x.mean(dim=-1)`. The cost is real, and in a codebase where dimension order is stable and well-documented, the positional shorthand may be the right tradeoff. The coordinate habit is not a moral imperative. It is a tool. Use it where the cost of a silent axis swap exceeds the cost of typing a bracket.

**Einlang itself is young.** The language used to make these arguments is a research prototype. Its tooling is sparse. Its error messages are the ones shown in these pages—no more. It does not compile to CUDA. It does not have a package manager. It does not integrate with PyTorch or JAX. The compiler described in the construction chapters is a frontend—it produces lowered NumPy, not optimized GPU kernels. A production-grade named-coordinate compiler would need an autodiff engine, a scheduler, and a backend that generates efficient code for the lowering patterns described here. None of that exists today. The coordinate habit works through comments, einops strings, and naming conventions in any framework, regardless of Einlang's maturity. But if you are tempted to build the rest: the IR, the check rules, and the lowering pass in these pages are a starting point. The distance from here to a production compiler is measured in engineering years, not ideas.

These five limitations do not weaken the case for named coordinates. They clarify it. Named coordinates prevent one class of error: the error where the coordinate identity exists in the programmer's head but not in the source text, and the notation provides no place to record it. For that class of error—the silent axis swap, the broadcast that drifts with the layout, the reduction that changes meaning without changing syntax—names are the only defense. For errors outside that class, other defenses apply.

**A note on existing named-tensor systems.** PyTorch has named tensors. xarray has labeled dimensions. Einops has named patterns. Why not just use those? Each catches a subset of the errors described in these pages. PyTorch's named tensors check broadcast alignment by name but are not part of the type system and do not survive through autograd. xarray labels dimensions for data analysis but does not compile to GPU kernels. Einops patterns are local to each call—they do not propagate across function boundaries. None of these systems provide the five-check wall, where every coordinate contract is verified at every call site before a single value is computed. A complete system was built not because the existing tools are useless—they are useful, and the coordinate habit works through them—but because only a complete system can show the full distance between what positional notation checks and what named notation can check. The distance is the argument.

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
Error[E006]: coordinate contract mismatch
  --> main.ein:2:16
   |
 2 | let y = predict(x);
   |                ^
   |                in call to `predict`
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

## If You Want More

Three works shaped the thinking behind these pages.

**Structure and Interpretation of Computer Programs** (Abelson, Sussman, and Sussman) taught generations of programmers that a language is built from primitive expressions, means of combination, and means of abstraction—and that building a metacircular evaluator is the final proof of understanding.

**Learn You a Haskell for Great Good** (Miran Lipovača) showed that a book about a programming language can be warm, direct, and relentlessly focused on the reader's understanding rather than the author's expertise.

**"Tensor Considered Harmful"** (Aleksander Mądry, 2018) and the named-tensor work at Harvard and Stanford asked: what happens when tensor dimensions have names that the compiler can check?

These are not tutorials. They are not documentation. They are road signs.
