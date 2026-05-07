---
layout: book
title: "The Name in the Bracket"
description: "The Name in the Bracket — a book about what notation hides, and what happens when you refuse to let it."
---

# The Name in the Bracket

Notation determines what you can notice. When a notation has no place for a fact, that fact becomes invisible — not just to the compiler, but to the act of reading and reasoning itself.

This book traces one idea — the coordinate audit — through sixteen chapters: from a Tuesday bug through gradients, recurrence, attention, and dynamic routing, to the principle that organizes them all. Einlang is the microscope. The habit is the payload.

## Preface

This is a book of ideas, not a tool manual.

Every programmer who works with tensors has debugged a silent shape bug. The code runs. The shapes line up. The loss goes down. And yet the program is wrong — systematically, invisibly wrong — because an axis means `class` but the operation treated it as `batch`. The notation never recorded the difference. The compiler never checked it. The bug survived integration tests, code review, and the training dashboard. It was found at 3 AM by a human being who traced one number backward through twelve layers of a deployed model and realized that the axis had been silently swapped three weeks earlier.

This book is about the gap between what tensor notation records and what tensor programs mean. The central claim is simple enough to state in a sentence: notation determines what you can notice. When a notation has no place for a fact, that fact becomes invisible — not just to the compiler, but to the act of reading and reasoning itself. The rest of the book is an extended demonstration of what that sentence costs and what it buys.

The book uses a small language called Einlang to make the demonstrations precise. You can type the examples. You can run them. The compiler will catch the shape-compatible wrong versions. But Einlang is not the argument. It is the microscope. The argument is the coordinate habit: four questions you can ask about any tensor expression, in any framework, that will tell you whether the notation preserved the facts that correctness depends on.

**Who this book is for.** If you build things below the level of an API call — a compiler pass, a numerical library, an autodiff engine, a tensor DSL, or a notation shared by a team — this book is written for you. If you have ever stared at `RuntimeError: mat1 and mat2 shapes cannot be multiplied` at 3 AM and wished the traceback told you which dimension was supposed to be `head` and which one was supposed to be `feature`, this book is written for you. If you mainly want another framework function to call, it will probably spend too much time under the floorboards. That is by design.

**How to read this book.** The Introduction is a twenty-minute encounter with the notation and the habit. Do not skip it — the four questions that carry the entire book are introduced there, and every later chapter assumes you have asked them at least once. Parts I through IV each raise the pressure: more coordinates, more operations, more ways for a hidden role to produce a silent bug. Chapter 15 shows that every rule in the preceding fourteen chapters follows from one sentence. Chapter 16 brings the Tuesday story home.

Along the way, every chapter opens with a bug story and ends with a Try It section. The bug stories are not decoration. They are the scar tissue that earned each rule. The Try It sections are not quizzes. They are the moment where reading becomes doing — where you test whether the habit has become yours.

Three motifs run through the book. The **Hiding Law** — "do not hide a fact that later reasoning must recover" — is the criterion for what earns a name and what earns silence. The **coordinate audit** — survive, consume, omit — is the procedure for reading any tensor line. The **3 AM test** — could a tired colleague find this bug without the mental context you have right now? — is the pressure that makes the first two matter.

A note on language. Throughout this book, "we" means the author and the reader together — we are tracing the same cells, debugging the same bugs, asking the same four questions. The coordinate habit is learned collaboratively or not at all.

**What this book is not.** It is not a language reference. The appendices and the [documentation](../docs/) serve that purpose. It is not a production migration guide. Einlang is still growing. It is not neutral. Every chapter takes a side: when a coordinate role decides correctness, the source should be able to state it.

The book is an argument in the form of sixteen chapters, two appendices, and one idea. The idea was stated in the first sentence. Everything that follows is evidence.

Turn the page. Let us trace one cell together.

## Contents

- [Introduction: The Shape-Meanings Gap](chapter-00-why.html)

### Part I: Coordinates
- [1. What Can the Compiler Not See?](chapter-01-compiler-blindness.html)
- [2. Axis Roles Are Not Axis Positions](chapter-02-axis-roles.html)
- [3. Coordinate Maps in the Standard Library](chapter-03-coordinate-maps.html)
- [4. What Does Broadcasting Hide?](chapter-04-broadcasting.html)
- [5. The Index That Leaves](chapter-05-index-that-leaves.html)
- [6. Softmax Has Three Coordinate Roles](chapter-06-softmax-coordinate-roles.html)

### Part II: Derivatives
- [7. What Is a Gradient?](chapter-07-gradients.html)
- [8. Matrix Multiplication Teaches the Pullback](chapter-08-matmul-pullback.html)
- [9. Local Derivatives, Global Shape](chapter-09-local-derivatives-global-shape.html)

### Part III: Time and Recurrence
- [10. Time Steps Are Not Loops](chapter-10-time-steps.html)
- [11. Storage Follows Observation](chapter-11-storage-follows-observation.html)
- [12. An RNN Is a Dependency Graph](chapter-12-rnn-dependency-graph.html)

### Part IV: Full Applications
- [13. If Dimensions Had Names Everywhere](chapter-13-named-dimensions-everywhere.html)
- [14. Attention as Named Communication](chapter-14-attention-named-communication.html)
- [15. What the Notation Refuses to Hide](chapter-15-notation-refuses-to-hide.html)
- [16. Dynamic Routing and Low-Rank Communication](chapter-16-dynamic-routing-low-rank-communication.html)

### Appendix
- [Coordinate Diagnostics](appendix-coordinate-diagnostics.html)
- [Coordinate Reading Laws](appendix-coordinate-laws.html)
