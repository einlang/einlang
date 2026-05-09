---
layout: book
title: "The Name in the Bracket"
description: "The Name in the Bracket — a book about what notation hides, and what happens when you refuse to let it."
---

# The Name in the Bracket

Notation determines what you can notice. When a notation has no place for a fact, that fact becomes invisible—not just to the compiler, but to the act of reading and reasoning itself.

This book traces one idea—the coordinate audit—through sixteen chapters: from a Tuesday bug through gradients, recurrence, attention, and dynamic routing, to the four habits that organize them all. Einlang is the microscope. The habit is the payload.

[Download PDF](../the-name-in-the-bracket.pdf)

## Preface

This is a book of ideas, not a tool manual.

Every programmer who works with tensors has debugged a silent shape bug. The code runs. The shapes line up. The loss goes down. And yet the program is wrong—systematically, invisibly wrong—because an axis means `class` but the operation treated it as `batch`. The notation never recorded the difference. The compiler never checked it. The bug survived integration tests, code review, and the training dashboard. It was found at 3 AM by a human being who traced one number backward through twelve layers of a deployed model and realized that the axis had been silently swapped three weeks earlier.

This book is about the gap between what tensor notation records and what tensor programs mean. The central claim: notation determines what you can notice. When a notation has no place for a fact, that fact becomes invisible.

The book uses a small language called Einlang to make the demonstrations precise. You can type the examples. You can run them. The compiler will catch the shape-compatible wrong versions. But Einlang is not the argument. It is the microscope. The argument is the coordinate habit: four questions you can ask about any tensor expression, in any framework, that will tell you whether the notation preserved the facts that correctness depends on.

**Who this book is for.** If you build things below the level of an API call—a compiler pass, a numerical library, an autodiff engine, a tensor DSL, or a notation shared by a team—this book is written for you. If you have ever stared at `RuntimeError: mat1 and mat2 shapes cannot be multiplied` at 3 AM and wished the traceback told you which dimension was supposed to be `head` and which one was supposed to be `feature`, this book is written for you. If you mainly want another framework function to call, it will probably spend too much time under the floorboards. That is by design.

**How to read this book.** The Prologue is a ten-minute encounter with the notation and the habit. Do not skip it. Part I (Chapters 1–5) introduces the primitives—naming, permuting, reducing, broadcasting—one at a time, with einlang syntax arriving in Chapter 3. Part II (Chapters 6–9) teaches combination: coordinate-aware functions, refactoring safety, gradients, and parameter updates. Part III (Chapters 10–12) teaches abstraction: building your own coordinate primitives, recurrence through time, and the four habits formalized. Part IV (Chapters 13–16) applies everything: complex coordinate interactions, the full syntax panorama, and two capstone audits.

Every chapter opens with a bug story. The bug stories are not decoration. They are the scar tissue that earned each rule.

Three motifs run through the book. The **Hiding Law**—"do not hide a fact that later reasoning must recover"—is the criterion for what earns a name and what earns silence. The **coordinate audit**—survive, consume, omit—is the procedure for reading any tensor line. The **3 AM test**—could a tired colleague find this bug without the mental context you have right now?—is the pressure that makes the first two matter.

A note on language. Throughout this book, "we" means the author and the reader together—we are tracing the same cells, debugging the same bugs, asking the same four questions. The coordinate habit is learned collaboratively or not at all.

The book is an argument in the form of sixteen chapters and one idea. Turn the page.

## Contents

- [Prologue: The Error That Didn't Error](chapter-prologue.html)

### Part I: Primitives
- [1. Every Dimension Deserves a Name](chapter-01-naming-coordinates.html)
- [2. Moving Flowers Without Losing the Trail](chapter-02-permutation.html)
- [3. A Small Farewell](chapter-03-reduction.html)
- [4. Copy, and a Promise](chapter-04-broadcasting.html)
- [5. When Promises Chain Together](chapter-05-broadcast-selfcheck.html)

### Part II: Combinations
- [6. Putting the Pieces Together](chapter-06-meet-einlang.html)
- [7. The Refactoring Trap](chapter-07-refactoring-traps.html)
- [8. The Challenge of Walking Backward](chapter-08-gradients.html)
- [9. Updates, with Names](chapter-09-updates.html)

### Part III: Abstraction
- [10. Building Your Own Blocks](chapter-10-abstraction.html)
- [11. Names in Time](chapter-11-time.html)
- [12. The Guardian's Booklet](chapter-12-four-habits.html)

### Part IV: Applications and Graduation
- [13. Complex Terrain](chapter-13-complex-terrain.html)
- [14. The Complete Picture](chapter-14-syntax-panorama.html)
- [15. The Simulation That Looked Right](chapter-15-audit-non-ml.html)
- [16. The Night Before the Run](chapter-16-audit-ml.html)

- [Epilogue: A Friend Named einlang](chapter-epilogue.html)

