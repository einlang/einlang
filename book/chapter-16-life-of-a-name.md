---
layout: book
title: "Chapter 16 · The Life of a Name"
---

# Chapter 16 · The Life of a Name

> "It is written. It is preserved. It is verified. It is burned. At no point is it decoration."
>
> — From Chapter 14

*Reflection · One coordinate name tracked through five forms, from source to execution*

---

This chapter is a single exercise, performed slowly.

Take the name `class`. You first wrote it in Chapter 3, inside a softmax bracket: `softmax[class](logits)`. It was one letter among many—five characters, lowercase, easy to type, easy to forget.

But that name did not stay in Chapter 3. It traveled. Through the IR, through the analysis passes, through the lowering engine, through the code generator. At each stage, it was read. At each stage, it was used. At each stage, the compiler asked it questions—what is your range? where do you appear? are you consumed or preserved?—and the answers determined whether the program compiled or failed.

This chapter traces that journey. Not as a technical description. As a walk. You and I, following a single name from the moment it was typed to the moment it became a number, stopping at each form to see what it became and what it enabled.

---

## Form 0: Before the Name

Before `class` was a coordinate, it was a position. In a PyTorch program, the softmax looked like this:

```python
probs = torch.softmax(logits, dim=-1)
```

`-1` is the last axis. If `logits` has shape `(batch, class)`, `-1` is `class`. If `logits` has shape `(batch, feature)`, `-1` is `feature`. If a refactoring changes the layout, `-1` silently changes meaning.

The positional form records *where*. It does not record *what*. The information that would distinguish `class` from `feature` is not in the code. It is in the programmer's head, in the variable name `logits`, in the documentation string that may or may not exist.

Now the programmer writes a different line:

```
softmax[class](logits[batch, class])
```

The name `class` appears twice: once in the function bracket, once in the index pattern. It is a string of five characters. It is also a coordinate identity—a claim that the tensor `logits` has a dimension called `class`, and that `softmax` will normalize over it.

This is Form 0. The name as written. What happens next is what this chapter is about.

---

## Form 1: The Parse Tree

The compiler reads the source. The parser turns characters into tokens, tokens into a tree. The name `class` survives this transformation unchanged:

```lisp
(call softmax
  (index logits (batch class))
  (coord-args class))
```

The parser does not know what `class` means. It knows that `class` appears in two places: as a coordinate argument in `coord-args`, and as an element in the index list of `logits`. It records both appearances faithfully.

At this stage, `class` is a symbol in a tree. It has no range, no type, no positional mapping. It is a name and nothing more. But it is in the right place: the `coord-args` slot says "this name will be bound to the function's coordinate parameter," and the index list says "this tensor is indexed by this name."

The parser's job is to recognize structure. It recognized that `softmax[class](logits[batch, class])` is a function call with a coordinate argument and a value argument. It placed `class` in the correct slots. The structure is now queryable by every subsequent pass.

---

## Form 2: After Name Resolution

Name resolution connects every occurrence of `class` to its definition.

```lisp
(call softmax
  (index logits (batch class))   ;; class → defined on logits, 2nd axis
  (coord-args class))             ;; class → bound to softmax's j parameter
```

The resolver asks: where is `class` defined? It finds the definition on `logits`—the tensor was declared with coordinates `(batch, class)`. It finds the binding in `softmax`'s signature—`fn softmax[j](x: [f32; ..left, j, ..right])`, and `class` is bound to `j`.

Two occurrences. Two resolutions. The first says: `class` is the second coordinate of `logits`. The second says: `class` is the coordinate argument to `softmax`, bound to parameter `j`. If either resolution failed—if `logits` had no `class`, or if `softmax` expected no coordinate parameter—the compiler would stop here.

This is the first moment the name becomes load-bearing. Before resolution, `class` was just characters. After resolution, `class` is a reference—a pointer to a definition. If the definition doesn't exist, the reference is dangling. The compiler reports it. The programmer fixes it. The name has done its first job.

---

## Form 3: After Analysis

Chapter 13's analysis passes walk the resolved tree. They derive range, shape, and type. They apply the five check rules. And they annotate every node with what they find.

After analysis, `class` carries annotations:

```lisp
(call softmax
  (index logits (batch class))
  (coord-args class))
;; ── analysis ──
;; class: (range 0 n_class), reduction axis
;; Rule 1 ✓ — class exists on logits
;; Rule 2 ✓ — class appears in all operands of max[class] and sum[class]
;; Rule 5 ✓ — class bound to j, contract satisfied
```

The name `class` now has a range: `0..n_class`. The compiler derived this from `logits`'s shape declaration. It knows that `class` spans `n_class` elements. It knows that `class` is consumed by reductions inside `softmax`. It knows that the coordinate contract at the call site is satisfied.

Three check marks. Three verifications that the name is used consistently. If any had failed—if `class` didn't exist, or wasn't consumed by the right reductions, or didn't match the function's contract—the compiler would have reported an error. The name enabled the check. Without the name, there is nothing to check.

Compare to `dim=-1`. The positional compiler sees `-1` and says: "valid integer." It does not ask whether `-1` is the right integer. It does not know that `-1` is supposed to mean `class`. The name-based compiler asks and answers. The positional compiler cannot ask.

---

## Form 4: After Lowering

Chapter 14's lowering pass maps names to numbers. `class`—the word you typed—becomes `axis=1`.

```lisp
(call softmax
  (index logits (batch class))
  (coord-args class))
;; ── lowering ──
;; batch → axis=0, loop over 0..batch_size
;; class → axis=1, reduction
;; keepdims=True (inferred from broadcast requirement)
;; strategy: vectorized
```

Why `axis=1`? Because the declaration order is `(batch, class)`. `batch` is first → axis 0. `class` is second → axis 1. The mapping is deterministic. You can verify it by hand.

Why `reduction`? Because `softmax` contains `max[class]` and `sum[class]`. Both consume `class`. The lowering pass reads the reduction annotations from analysis and emits the reduction axis.

Why `keepdims=True`? Because the subtraction `logits - max` requires `max` to broadcast back over `class`. The analyzed shapes showed the broadcast requirement. The lowering pass inferred the `keepdims` from it. The programmer didn't write `keepdims=True`. The compiler derived it.

Why `vectorized`? Because all ranges are known at compile time. The shapes are concrete. NumPy's C kernels can handle the entire operation. If arrays were jagged, the strategy would be `scalar`. The compiler chooses.

Every annotation on `class` from Form 3 has been translated into an execution decision. The range became a loop bound. The reduction annotation became `axis=1`. The broadcast requirement became `keepdims=True`. The name `class` is still in the tree—it hasn't been erased—but it is now accompanied by the numbers that will replace it.

---

## Form 5: The Generated Code

The code generator reads the lowered tree and emits Python:

```python
def softmax(logits):
    m = np.max(logits, axis=1, keepdims=True)
    e = np.exp(logits - m)
    return e / np.sum(e, axis=1, keepdims=True)
```

`class` is gone. In its place is `axis=1`. The name has been burned. The number remains.

But—and this is the point of the entire journey—the number is **correct** because the name was **verified**. `axis=1` is not a guess. It is not a convention. It is not "the last axis, whatever that happens to be." It is the axis that was named `class` in the source, preserved through the IR, verified by analysis, and mapped to 1 by lowering.

The number inherits the correctness of the name. If the name was right—if the programmer correctly identified `class` as the normalization coordinate—then `axis=1` is right. If the name was wrong—if the programmer wrote `class` but should have written `feature`—then `axis=1` is wrong. But the error is a name error, caught by the checks in Form 3 (does `feature` exist on this tensor? no → error). The name-based pipeline catches name errors. The positional pipeline catches nothing.

---

## The Five Forms, Side by Side

```
Form 0 (Source):       softmax[class](logits[batch, class])
Form 1 (Parse):        (call softmax (index logits (batch class)) (coord-args class))
Form 2 (Resolved):     class → defined on logits, bound to softmax's j
Form 3 (Analyzed):     class: range 0..n_class, reduction, Rules 1/2/5 ✓
Form 4 (Lowered):      class → axis=1, reduction, keepdims=True, vectorized
Form 5 (Code):         np.max(logits, axis=1, keepdims=True)
```

Five forms. One name. Five stages of increasing concreteness. At each stage, the name did work that a number could not have done.

A number can be an axis. A number cannot be checked for existence. A number cannot be verified against a function's coordinate contract. A number cannot tell the lowering pass whether it's a reduction axis or a loop axis—both are just integers. The name distinguishes. The name enables. The name is burned only after all its work is done.

---

## What the Name Enabled

Let's be precise about what `class` made possible at each stage.

**Form 1 (Parse):** The name `class` in the `coord-args` position told the parser: this is a coordinate argument, not a value argument. The parser placed it in the correct AST slot. A positional `dim=-1` would be a value argument—indistinguishable from any other integer argument until semantic analysis. The name disambiguates at the syntax level.

**Form 2 (Resolution):** The resolver connected `class` in the call to `j` in the function signature. It verified that `logits` carries a coordinate called `class`. These are identity checks—do the names match? Positional resolution connects `-1` to the last axis. That connection is always valid, regardless of whether the last axis is the intended one.

**Form 3 (Analysis):** The analysis passes derived `class`'s range from `logits`'s declaration. They checked that `class` appears in every operand of the reductions inside `softmax`. They verified that the coordinate contract at the call site is satisfied. These are semantic checks, enabled by the name's presence in the tree.

**Form 4 (Lowering):** The lowering pass mapped `class` to `axis=1` using the declaration order. It inferred `keepdims=True` from the broadcast requirement in the subtraction. These are engineering decisions, made deterministically from the analyzed annotations.

**Form 5 (Code):** The generated code is correct because every decision that produced it was verified. The integer `1` is not arbitrary. It is the integer that corresponds to the coordinate named `class`, which was checked for existence, verified for consistency, and mapped deterministically to its position.

The name enabled every check. Remove the name, and every check collapses. The positional pipeline has only one check: "is `axis=1` a valid integer?" It is. Always. The check is vacuous.

---

## The Name's Life and the Programmer's

Now step back. Not the compiler's perspective—yours.

You wrote `class`. Five characters. You pressed the keys. And because you wrote them, the compiler asked questions that it could not have asked otherwise. Because the compiler asked, bugs that would have survived to runtime were caught at compile time. Because they were caught, you didn't spend three hours at 3 AM tracing a shape mismatch backward through twelve layers.

The name's life is the compiler's work. But the name's value is your time.

Every `dim=-1` in your codebase is a name that was never written. Every one of them is a question the compiler couldn't ask. Every one of them is a potential 3 AM.

The name `class` took five keystrokes to write. It enabled five stages of verification. The ratio is not merely favorable. It is the difference between a program that is correct by construction and a program that is correct by coincidence.

---

## When the Name Is Wrong

The chapter has traced `class` through a journey where every check passed. But what happens when the name is wrong?

Suppose the programmer writes `softmax[batch](logits[batch, class])`. `batch` is the coordinate argument. `softmax` expects `j`—a coordinate it will consume in reductions and reconstruct in the output. `batch` is bound to `j`.

Form 1 (Parse): passes. `batch` is a valid symbol in the `coord-args` position.

Form 2 (Resolution): passes. `batch` exists on `logits` (the declaration says `[batch, class]`). `batch` is bound to `j` in `softmax`'s signature.

Form 3 (Analysis): Rule 5—Coordinate Contract. `softmax`'s signature says `j` appears in `..left, j, ..right`. The packs are bound: `..left` is empty. `..right` is `[class]`. The return type is `[f32; batch, class]`. But wait—`j` is `batch`, and `softmax` consumes `j` in a reduction. The reduction `sum[j]` consumes `batch`. The function contracts `j` and then reconstructs it. But `batch` was never supposed to be contracted—it was supposed to pass through in `..left`. The error is that `batch` was used as the coordinate parameter when it should have been a pack.

What error does the compiler emit? It depends on how the contract is checked. If the compiler checks that the coordinate parameter `j` is distinct from the pack coordinates, it would catch this: `batch` is both the coordinate parameter AND a pack coordinate? No—packs are bound at the call site. `batch` is a concrete coordinate from the argument `logits[batch, class]`. When `softmax[batch]` is called, `batch` is bound to `j`. Then `j` is consumed inside `softmax`. The function returns `[f32; batch, class]`—but `batch` was consumed and reconstructed, which means the function's semantics are: "normalize over batch." The code compiles. It produces a valid probability distribution—over the batch dimension, not the class dimension.

The name was wrong. The check passed. The program is incorrect.

This is the boundary from Chapter 7 and Chapter 15, restated for the final time: **names check consistency, not correctness.** `softmax[batch]` is internally consistent—every reduction, broadcast, and gradient aligns over `batch`. The error is that the programmer wanted `class`. The compiler cannot read the programmer's mind. It can only verify the contract the programmer wrote.

But the name `batch` is visible. When the next programmer reads `softmax[batch](logits)`, they see the error immediately—"why are we normalizing over batch?" The name makes the error readable. The positional equivalent `softmax(logits, dim=0)` hides the error behind a position number. The reader sees `dim=0` and must reconstruct whether axis 0 is batch or class. The reconstruction may be wrong. The name `batch` in `softmax[batch]` carries its meaning with it.

A wrong name is a visible error. A missing name is an invisible one. This has been the book's refrain since Chapter 1. The refrain survives the final chapter because it is the one claim that every chapter has tested and none has falsified.

---

## Burned, Not Lost

The name is burned. `class` became `axis=1`. The word is not in the generated code. The CPU will never read it.

But the name is not lost. It is in the source code, where you can read it. It is in the compiler's analysis log, where the checks are recorded. It is in the error message, if a future refactoring breaks the contract. It is in the lowering annotations, documenting why `axis=1` was chosen.

The name burned. The heat remains. And the heat is a program whose axes are correct—not because the programmer memorized the dimension order, but because the compiler verified the coordinate names.

This is what the seventeen chapters of this book have been building toward. Not a language. Not a compiler. A guarantee: **if you write the name, the compiler will check it. If the compiler checks it, the execution will match the intent. If the intent changes, the name changes—and the compiler will tell you everywhere the old name was used.**

The guarantee is not perfect. Names can be wrong—you can name the wrong coordinate `class`. But a wrong name is a visible error. A missing name is an invisible one. The book has argued that the second kind is more dangerous, because it compounds silently. The name `class`—burned, verified, gone—is the answer to that argument. It was here. It did its work. The code is correct because it was.

---


*You first wrote `class` in Chapter 3. It has now traveled through every chapter of this book—through primitives, combinations, comparisons, construction, and reflection. It was written, preserved, verified, and burned. Close your eyes. Trace its journey in your mind. The path you trace is the coordinate habit.*

---

This chapter traced one name through five forms. But the name was not the point. The point was what each form made possible.

The parse tree made the name queryable by compiler passes. The resolved tree made it verifiable against its definition. The analyzed tree made its consistency checkable across the whole program. The lowered tree made it mappable to a position number. The generated code made it executable.

At each stage, the name enabled a question that could not have been asked without it. "Does this tensor have a coordinate called `class`?" → Form 1. "Is `class` bound to the right parameter?" → Form 2. "Is `class` consumed consistently by every reduction?" → Form 3. "What position does `class` map to?" → Form 4. "Is that position the correct one?" → Form 5.

Five questions. Five answers. One name. The name was the key that unlocked each question. Without it, the compiler asks only: "is this a valid integer?" That question has one answer: yes, always. The positional pipeline is a pipeline of one question. The named pipeline is a pipeline of five. The difference is not the number of questions. It is what the questions verify.

The next time you write `dim=-1`, remember: the positional compiler asked one question of that argument ("is -1 a valid integer?"). The named compiler would have asked five—existence, binding, consistency, position, and correctness of that position. The four unasked questions are the gaps where bugs live.
