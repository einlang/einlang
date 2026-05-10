---
layout: book
title: "The Name in the Bracket"
description: "The Name in the Bracket — a book about what notation hides, and what happens when you refuse to let it."
---

# The Name in the Bracket

Notation determines what you can notice. When a notation has no place for a fact, that fact becomes invisible—not just to the compiler, but to the act of reading and reasoning itself.

This book traces one idea—the coordinate audit—through fifteen chapters: from a Tuesday bug through naming, reduction, broadcasting, differentiation, comparison with PyTorch, and compiler construction. Einlang is the microscope. The habit is the payload.

[Download PDF](../the-name-in-the-bracket.pdf)

## Preface

This is a book of ideas, not a tool manual.

Every programmer who works with tensors has debugged a silent shape bug. The code runs. The shapes line up. The loss goes down. And yet the program is wrong—systematically, invisibly wrong—because an axis means `class` but the operation treated it as `batch`. The notation never recorded the difference. The compiler never checked it. The bug survived integration tests, code review, and the training dashboard. It was found at 3 AM by a human being who traced one number backward through twelve layers of a deployed model and realized that the axis had been silently swapped three weeks earlier.

This book is about the gap between what tensor notation records and what tensor programs mean. The central claim: notation determines what you can notice. When a notation has no place for a fact, that fact becomes invisible.

The book uses a small language called Einlang to make the demonstrations precise. You can type the examples. You can run them. The compiler will catch the shape-compatible wrong versions. But Einlang is not the argument. It is the microscope.

**Who this book is for.** If you build things below the level of an API call—a compiler pass, a numerical library, an autodiff engine, a tensor DSL, or a notation shared by a team—this book is written for you. If you have ever stared at `RuntimeError: mat1 and mat2 shapes cannot be multiplied` at 3 AM and wished the traceback told you which dimension was supposed to be `head` and which one was supposed to be `feature`, this book is written for you. If you mainly want another framework function to call, it will probably spend too much time under the floorboards. That is by design.

**How to read this book.** The Prologue is a ten-minute encounter with the notation and the habit. Do not skip it. Part I (Chapters 1–2) introduces the primitives—naming, permuting, reducing, broadcasting—with the megaphone model that unifies them. Part II (Chapters 3–7) teaches combination: coordinate-aware functions, normalization skeletons, recurrence, complex terrain, and differentiation. Part III (Chapters 8–10) puts einlang side by side with PyTorch and NumPy on normalization, attention, and physical simulation. Part IV (Chapters 11–15) opens the compiler: intermediate representation, analysis and check rules, lowering, reflection, and a complete syntax reference.

A note on language. Throughout this book, "we" means the author and the reader together—we are tracing the same cells, debugging the same bugs, asking the same questions. The coordinate habit is learned collaboratively or not at all.

The book is an argument in the form of fifteen chapters and one idea. Turn the page.

## Contents

### Part I: Primitives
- [1. The Ghost in the Name](chapter-01-ghost-in-the-name.html)
- [2. The Megaphone's Promise](chapter-02-megaphone-promise.html)

### Part II: Combinations
- [3. Names as Contracts](chapter-03-names-as-contracts.html)
- [4. Blocks and Skeletons](chapter-04-blocks-skeletons.html)
- [5. Names in Time](chapter-05-names-in-time.html)
- [6. Complex Terrain](chapter-06-complex-terrain.html)
- [7. Names Through Differentiation](chapter-07-gradients.html)

### Part III: Comparisons
- [8. Comparison: Normalization](chapter-08-comparison-normalization.html)
- [9. Comparison: Attention](chapter-09-comparison-attention.html)
- [10. Comparison: Physics](chapter-10-comparison-physics.html)

### Part IV: Construction
- [11. The Shape of Thought](chapter-11-shape-of-thought.html)
- [12. The Name in the Mirror](chapter-12-name-in-mirror.html)
- [13. Firewood](chapter-13-firewood.html)
- [14. The Outline of the Name](chapter-14-outline-of-name.html)
- [15. The Complete Picture](chapter-15-complete-picture.html)

- [Epilogue: A Friend Named einlang](chapter-epilogue.html)
