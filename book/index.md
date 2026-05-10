---
layout: book
title: "The Name in the Bracket"
description: "The Name in the Bracket — a book about what notation hides, and what happens when you refuse to let it."
---

# The Name in the Bracket

Notation determines what you can notice. When a notation has no place for a fact, that fact becomes invisible—not just to the compiler, but to the act of reading and reasoning itself.

This book traces one idea—the coordinate audit—through seventeen chapters: from a Tuesday bug through naming, reduction, broadcasting, differentiation, comparison with PyTorch, and compiler construction. Einlang is the microscope. The habit is the payload.

The bracket is where the name enters, where it is checked, and where it is finally burned. The bracket is the beginning and the end. Everything between is the life of a name in source code—the only life that a compiler can see.

[Download PDF](../the-name-in-the-bracket.pdf)

## Preface

This is a book of ideas, not a tool manual.

Every programmer who works with tensors has debugged a silent shape bug. The code runs. The shapes line up. The loss goes down. And yet the program is wrong—systematically, invisibly wrong—because an axis means `class` but the operation treated it as `batch`. The notation never recorded the difference. The compiler never checked it. The bug survived integration tests, code review, and the training dashboard. It was found at 3 AM by a human being who traced one number backward through twelve layers of a deployed model and realized that the axis had been silently swapped three weeks earlier.

This book is about the gap between what tensor notation records and what tensor programs mean. The central claim: notation determines what you can notice. When a notation has no place for a fact, that fact becomes invisible.

The book uses a small language called Einlang to make the demonstrations precise. You can type the examples. You can run them. The compiler will catch the shape-compatible wrong versions. But Einlang is not the argument. It is the microscope.

**Who this book is for.** If you build things below the level of an API call—a compiler pass, a numerical library, an autodiff engine, a tensor DSL, or a notation shared by a team—this book is written for you. If you have ever stared at `RuntimeError: mat1 and mat2 shapes cannot be multiplied` at 3 AM and wished the traceback told you which dimension was supposed to be `head` and which one was supposed to be `feature`, this book is written for you. If you mainly want another framework function to call, it will probably spend too much time under the floorboards. That is by design.

**How to read this book.** The book is designed for three kinds of readers. Pick the path that fits where you are.

**The full journey (all chapters, in order).** For readers who want to understand what happens when tensor coordinates carry names—from the primitives through gradient derivation through compiler construction. This path traces one idea through seventeen chapters and builds a miniature compiler by the end. Start at Chapter 1. Do the Follow Along exercises. Derive the gradients. The chapters build on each other, and each one assumes the previous one's vocabulary.

**The comparison path (Chapters 1–2, then 9–11).** For readers who are already comfortable with broadcasting and reduction, and want to see the two notations side by side on real code. Read Chapters 1–2 to learn the megaphone model and coordinate set subtraction—these are the vocabulary the comparison chapters use. Then jump to Chapters 9–11, where LayerNorm, GroupNorm, multi-head attention, Flash Attention, and physical simulation are written twice: once in PyTorch/NumPy, once in Einlang. The comparisons will show you what positional notation hides and what named notation reveals. If the comparisons convince you, return to Chapters 3–8 for the deeper machinery—coordinate-aware function signatures, the broadcast self-audit, and differentiation by coordinate accounting.

**The compiler path (Chapters 1–4, then 12–14).** For readers who want to see how a name-based type system for tensors actually works. Read Chapters 1–4 to learn the primitives and the Inversion Rule. Then jump to Chapters 12–14, which build a compiler frontend: IR as S-expressions, five check rules, and lowering from names to integers. The compiler is small enough to hold in your head—the core loop is ten lines. If you have ever wondered what a type checker for tensor dimensions would look like, these chapters are your answer.

**If you only have an afternoon.** Read Chapter 1 (the ghost in the name), Chapter 4 (the broadcast self-audit), and the Epilogue. That ninety-minute path gives you the core idea, the core diagnostic tool, and the core habit. Everything else is depth.

Part I (Chapters 1–2) introduces the primitives—naming, permuting, reducing, broadcasting—with the megaphone model that unifies them. Part II (Chapters 3–8) teaches combination: coordinate-aware functions, the broadcast self-audit, normalization skeletons, recurrence, complex terrain, and differentiation. Part III (Chapters 9–11) puts Einlang side by side with PyTorch and NumPy on normalization, attention, and physical simulation. Part IV (Chapters 12–14) opens the compiler: intermediate representation, analysis and check rules, and lowering from names to integers. Chapters 15–16 and the Epilogue reflect on what was built and what it means.

A note on language. Throughout this book, "we" means the author and the reader together—we are tracing the same cells, debugging the same bugs, asking the same questions. The coordinate habit is learned collaboratively or not at all.

The book is an argument in the form of seventeen chapters and one idea. Turn the page.

## Contents

### Part I: Primitives
- [1. The Ghost in the Name](chapter-01-ghost-in-the-name.html)
- [2. The Megaphone's Promise](chapter-02-megaphone-promise.html)

### Part II: Combinations
- [3. Names as Contracts](chapter-03-names-as-contracts.html)
- [4. The Broadcast Self-Audit](chapter-04-broadcast-self-audit.html)
- [5. Blocks and Skeletons](chapter-05-blocks-skeletons.html)
- [6. Names in Time](chapter-06-names-in-time.html)
- [7. Complex Terrain](chapter-07-complex-terrain.html)
- [8. Names Through Differentiation](chapter-08-gradients.html)

### Part III: Comparisons
- [9. Comparison: Normalization](chapter-09-comparison-normalization.html)
- [10. Comparison: Attention](chapter-10-comparison-attention.html)
- [11. Comparison: Physics](chapter-11-comparison-physics.html)

### Part IV: Construction
- [12. The Shape of Thought](chapter-12-shape-of-thought.html)
- [13. The Name in the Mirror](chapter-13-name-in-mirror.html)
- [14. Firewood](chapter-14-firewood.html)
- [15. The Outline of the Name](chapter-15-outline-of-name.html)
- [16. The Life of a Name](chapter-16-life-of-a-name.html)
- [17. The Complete Picture](chapter-17-complete-picture.html)

- [Epilogue: A Friend Named Einlang](chapter-epilogue.html)
