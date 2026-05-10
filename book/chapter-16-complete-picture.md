---
layout: book
title: "Appendix · The Complete Picture"
---

# Appendix · The Complete Picture

> "The purpose of computing is insight, not numbers."
>
> — Richard Hamming

*Reference · Thought map and syntax at a glance*

---

This chapter contains no new syntax. It is a map of territory already explored.

The preceding fourteen chapters introduced Einlang's grammar piece by piece, each piece arriving when the concept it served had earned its introduction. The result is a working knowledge of the language, but the pieces are scattered across chapters. This chapter assembles them into one place, organized by category rather than by pedagogical necessity.

Think of it as the view from the summit. You climbed the mountain one trail at a time. Now you can see the whole range.

But before I show you my map, draw your own.

---

## Build Your Own Map

Take out a piece of paper. Or open a blank document. At the top, write: `dim=1 bug`.

Now draw arrows downward. Each arrow is a question the bug forced us to ask. *"Which coordinate did I just erase?"* → naming. *"Which coordinate is being reduced?"* → reduction bracket. *"Why can't the compiler check this?"* → coordinate contracts.

Don't look at the next section yet. Draw from memory. What were the big ideas? How do they connect? Which chapters depend on which?

Five minutes. Go.

---

Done? Good. Now look at your map and ask three questions:

1. **Which arrow did you forget?** Everyone forgets at least one. The arrow you forgot connects two ideas you hadn't realized were dependent on each other. That connection is the thing you haven't fully internalized yet.

2. **Which arrow did you draw but can't explain why it exists?** You remembered that A depends on B, but you can't articulate the dependency. That arrow is a memory, not an understanding. Go back to the chapter where that arrow was first drawn and reread the transition.

3. **Which idea has the most arrows pointing to it?** That idea is the load-bearing concept. In this book, it is almost certainly "a coordinate has a name." Everything else depends on it. If you had to explain the book in one sentence, that concept would be in it.

The map you drew is not the final answer. It is a snapshot of your understanding at this moment. A month from now, draw it again. The arrows will have moved. Some will have disappeared—their dependencies now obvious. Others will have appeared—connections you didn't see the first time.

Learning is not the accumulation of facts. It is the continuous redrawing of the map.

---

## The Thought Map (One Version)

Here is one version of the map. It is not the only version. Compare it to yours. Where do they agree? Where do they differ? The differences are not errors—they are perspectives.

Before the syntax reference, a map of how the ideas connect. Each arrow is a dependency: the idea at the tail must be understood before the idea at the head.

```
dim=1 bug (Prologue)
    │
    ▼
A coordinate has a name, a domain, a position (Ch1)
    │
    ├──► Permutation: names survive position changes (Ch1)
    │
    ├──► Reduction: the consumed coordinate is named (Ch2)
    │       │
    │       └──► Broadcasting: the omitted coordinate is visible (Ch2)
    │               │
    │               └──► Inversion Rule: broadcast ↔ reduction dual (Ch2, Ch7)
    │
    ├──► Coordinate-aware functions: names as type-level contracts (Ch3)
    │       │
    │       ├──► Square Matrix Test: when extents equal, only names differ (Ch3)
    │       │
    │       ├──► Pack polymorphism: ..batch absorbs unknown leading dims (Ch4)
    │       │
    │       └──► Normalization skeleton: one pattern, four functions (Ch4)
    │
    ├──► Recurrence: time as a directional coordinate (Ch5)
    │       │
    │       └──► Causality constraint: t-1 valid, t+1 rejected (Ch5)
    │
    ├──► Complex terrain: splits, arithmetic, disambiguation (Ch6)
    │
    ├──► Differentiation: the pullback reads the forward pass backward (Ch7)
    │       │
    │       └──► @fn: custom derivative rules carry coordinate contracts (Ch7)
    │
    ├──► Comparisons: same computation, two notations (Ch9–11)
    │       │
    │       ├──► Normalization: GroupNorm reshape chain vs named groups (Ch9)
    │       ├──► Attention: identical PyTorch, distinct Einlang signatures (Ch10)
    │       └──► Physics: integer field indices vs named field coordinates (Ch11)
    │
    └──► Compiler construction (Ch12–14)
            │
            ├──► IR: S-expressions preserve every name (Ch12)
            │
            ├──► Analysis: range → shape → type, five check rules (Ch13)
            │
            └──► Lowering: names → integers, three strategies (Ch14)
                    │
                    └──► Firewood: names burn, heat remains (Ch14)
```

Every path begins at the `dim=1` bug. Every arrow is a question the bug forced us to ask. The map is not the territory—but it shows how the trails connect.

---

## Declarations

**`let`** binds an immutable name to a value. *Introduced in Chapter 1.*

```rust
let x = 42;
let pi: f64 = 3.141592653589793;
let matrix: [f32; 2, 3] = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
```

Type annotations are optional. When present, the value must be compatible. All `let` bindings are immutable.

---

## Rectangular Declarations

A rectangular declaration binds a tensor by naming its coordinates. *Introduced in Chapter 1; extended with domains in Chapter 5.*

```rust
let C[i, j] = sum[k](A[i, k] * B[k, j]);
```

Index slots in the declaration bracket may be:
- A name: `i`, `j`, `batch` — the standard case.
- A name with an explicit domain: `t in 0..T` — for recurrences.
- A literal: `0` — used for base cases.
- A named rest: `..batch` — absorbs zero or more adjacent axes.

Expressions are not allowed in the declaration bracket. `let fib[n-1] = ...` is an error. The left side names what is being defined. The right side computes it.

---

## Reductions

A reduction consumes a coordinate. *Introduced in Chapter 2; selection reductions in Chapter 4.*

Operations: `sum`, `max`, `min`, `prod`.

```rust
let total = sum[i](data[i]);
let row_sums[i] = sum[j](matrix[i, j]);
```

Selection reductions return addresses rather than values:

```rust
let pred[b] = argmax[class](logits[b, class]);
```

The consumed coordinate is eliminated from the result shape. The reduction bracket names it explicitly—the reader does not need to infer which coordinate disappeared.

---

## Broadcasting

Broadcasting is an omission in the indexing pattern. *Introduced in Chapter 2; self-audit in Chapter 7.*

```rust
let out[i, j] = A[i, j] + bias[j];   // bias omits i → broadcast over i
```

The omitted coordinate is the one being broadcast over. The megaphone model: `bias` is silent on `i`, so the compiler copies it across all values of `i`. The silence is a semantic claim: `bias` does not depend on `i`.

The Inversion Rule: what broadcasts in the forward pass is reduced in the backward pass. `bias[j]` omits `i` forward → `d_bias[j] = sum[i](d_out[i, j])` backward.

---

## Named Rest Indices

`..name` stands for zero or more adjacent axes, collectively named. *Introduced in Chapter 2; pack polymorphism in Chapter 4.*

```rust
let result[..batch, j] = x[..batch, j] + bias[j];
let row_sum[..batch] = sum[j](x[..batch, j]);
```

The same rest name must describe the same axis span within an expression. Packs make functions rank-polymorphic: the same `layer_norm[feature]` works on 2D, 3D, or 4D inputs.

---

## Where Clauses

A where clause filters or binds. *Introduced in Chapter 2; backward behavior in Chapter 7.*

Boolean guards narrow the domain:

```rust
let pos_sum = sum[i](data[i]) where data[i] > 0;
let upper[i, j] = matrix[i, j] where i <= j;
```

Variable bindings name intermediate values:

```rust
let output[i, j] = activated
    where z = sum[k](input[i, k] * weight[k, j]) + bias[j],
          activated = if z > 0.0 { z } else { 0.0 };
```

In the backward pass, filtered elements receive zero gradient. The domain constraint applies symmetrically in both directions.

---

## Coordinate-Aware Functions

A function may declare coordinate parameters. *Introduced in Chapter 3; pack parameters in Chapter 4.*

```rust
fn softmax[j](x: [f32; ..left, j, ..right])
    -> [f32; ..left, j, ..right]
{ ... }
```

Call sites pass coordinate arguments in the bracket position:

```rust
let p[b, class] = softmax[class](logits[b, class]);
```

The compiler checks that `class` exists on `logits` and that the coordinate contract is satisfied. The bracketed name is part of the call contract, not a comment.

Packs (`..left`, `..right`, `..spatial`) make functions polymorphic over surrounding structure. A caller disambiguates by grouping: `softmax[(height, width)](x)`.

---

## Recurrence Relations

Self-referential declarations define sequences over time. *Introduced in Chapter 5.*

```rust
let u[t in 0..T, i] = initial[i];
let u[t in 1..T, i] = u[t-1, i] + f(u[t-1, i]);
```

Backward references only. `u[t+1, i]` on the right-hand side with declaration index `t` is a compile error. Causality is a syntactic constraint, not a convention.

The optimizer is a recurrence:

```rust
let w[t in 1..T, out, in] = w[t-1, out, in] - lr * grad[t-1, out, in];
```

---

## Automatic Differentiation

`@loss / @W` computes the gradient. *Introduced in Chapter 7.*

```rust
let dW = @loss / @W;
```

The gradient has the same shape as the denominator. The pullback is computed by reversing the forward graph: every forward reduction becomes a backward broadcast; every forward broadcast becomes a backward reduction. The shopping cart record, read in reverse.

Custom rules use `@fn`:

```rust
@fn relu(x) {
    if x > 0.0 { @x } else { 0.0 }
}
```

Coordinate-aware custom rules carry the same bracketed parameters as the primal function.

---

## Why the Compiler Reads Coordinates Too

The preceding sections catalogued syntax. But syntax is only half the story. Each compiler pass depends on coordinate names to do its job. *These passes are described in Chapters 12–14.*

**Shape inference** (Ch12–13) reads coordinate names to decide whether an expression is legal before it runs. `sum[k](A[i, k] * B[k, j])` succeeds if `k` appears in both `A` and `B`. Under names, the contract is: `i` survives from `A`, `j` survives from `B`, `k` appears in both and is consumed.

**Range analysis** (Ch13) finds the domain of every axis: from array shapes, from literals, or from explicit declarations. Every coordinate gets a concrete range before code generation.

**Five check rules** (Ch13) verify the IR: index existence, reduction consistency, broadcast recording, causality, and coordinate contract at call sites. Each catches a class of bug that positional notation silently accepts.

**Gradient lowering** (Ch14) reads coordinate names to build the backward pass. The rule: preserve the coordinates of `W`, sum over everything else. Set subtraction, applied to coordinate names, derives the pullback.

**Storage planning** (Ch14) reads coordinate names to decide which tensors can share memory. A recurrence creates a dependency chain; the compiler allocates a rolling buffer.

**Kernel fusion** (Ch14) reads coordinate names to decide which operations can be merged. Operations that share surviving coordinates can fuse; operations across a reduction boundary cannot.

---

## Error Codes

Three errors are especially relevant to the coordinate habit. You won't memorize error codes from a book. But reading them now means you'll recognize them when they appear:

- **E003 (Undefined Coordinate)**: a coordinate name is referenced but does not exist on the tensor. `softmax[nonexistent](logits)` — caught at the call site. The error message names the missing coordinate and the tensor that lacks it. This is the compiler version of "you wrote `dim=1` but the tensor has no dimension 1." Unlike the positional version, the error tells you *which name* was expected.

- **E004 (Coordinate Range Mismatch)**: two uses of the same coordinate name infer incompatible ranges. `A[i, k] * B[k, j]` where `k` has range 64 in `A` but 128 in `B`. The shapes would produce a runtime error. The compiler catches it at analysis time and tells you which tensor declares which range.

- **E006 (Coordinate Contract Violation)**: a function call supplies a coordinate argument that does not match the function's declared coordinate parameter layout. `softmax[batch](logits[batch, class])` where `softmax` expects `j` and preserves it in the return type, but `batch` is in `..left` — the contract is violated. The error message shows the expected layout and the actual layout side by side.

These errors catch the bugs that positional APIs leave to runtime or to silence. They exist because the coordinate names exist. No names → no E003. No coordinate-aware functions → no E006. The error codes are not arbitrary. They are the compiler saying, in structured form: "the name you wrote does not match the names the program declares."

---

## How to Use This Chapter

This chapter is built to be revisited. Not read cover to cover—opened to the section you need.

If you're writing a new Einlang function and can't remember the exact syntax for a recurrence declaration, open to "Recurrence Relations." If you're debugging a coordinate mismatch and want to re-derive the pullback rule, open to "Automatic Differentiation." If you're designing a new operation and want to check whether it fits the existing primitives, trace it through the Thought Map.

The syntax reference is the scaffolding. The thought map is the blueprint. Together they let you rebuild what you need without rereading the whole book.

But the most important section of this chapter is not the syntax. It is the four-question audit table. Those four questions work in any framework. They are the coordinate habit, reduced to its smallest portable form. Copy them. Tape them to your monitor. Use them on your next tensor bug.

---

## Five Principles, Restated

Before the syntax reference ends, the five principles that every chapter has demonstrated. They are not syntax. They are what the syntax serves.

**1. Coordinates have identities.** `batch`, `channel`, `time`, `feature` are not positions. They are names. A position records where. A name records what. When the layout changes, the position breaks. The name survives.

**2. Reductions must name what they consume.** `sum[class](x)` says "I am consuming `class`." `x.sum(dim=1)` says "I am consuming position 1." The first survives a transpose. The second does not.

**3. Broadcasts must name what they copy along.** `out[i, j] = A[i, j] + bias[j]` says "`bias` is silent on `i`." `A + bias` says nothing. The first records the omission. The second infers it from shapes.

**4. Functions must declare their coordinate contracts.** `fn softmax[j](x: [f32; ..left, j, ..right])` says "I operate on `j`." `def softmax(logits, dim=-1)` says "I operate on the last axis." The first is checked at every call site. The second is a convention.

**5. Gradients read the forward pass backward.** What consumed forward is broadcast backward. What was silent forward is summed backward. The Inversion Rule is not a separate mechanism. It is the forward pass, read in reverse, with the same coordinate names on both sides.

Five principles. They are not Einlang-specific. They apply in any notation that records coordinate identities. The notation can be brackets. It can be einops strings. It can be comments. What matters is that the identity is recorded somewhere the reader can see it and the compiler can check it.

The syntax will evolve. The thought map will grow. The habit—write the coordinate names, make the omissions explicit, let the compiler check the contracts—will outlast any particular syntax.

### Five Principles in Practice

Each principle is a claim about what the notation should record. But principles read differently when you see them applied to a single program. Here is one program—a linear layer with LayerNorm—written first without the principles, then with each applied in turn.

**Without any principles.** PyTorch:

```python
def forward(x, W, b, gamma, beta):
    h = x @ W.T + b
    mean = h.mean(dim=-1, keepdim=True)
    var = (h - mean).pow(2).mean(dim=-1, keepdim=True)
    return gamma * (h - mean) / torch.sqrt(var + 1e-5) + beta
```

Seven lines. The operations are correct. The axes are implicit. `dim=-1` appears twice. If `h` changes from `(batch, feature)` to `(batch, seq, feature)`, `dim=-1` silently changes meaning from `feature` to `seq`. The code runs. The normalization is over the wrong axis.

**Principle 1 applied: Coordinates have identities.**

```python
# h: (batch, feature) — feature is dim=-1
def forward(x, W, b, gamma, beta):
    h = x @ W.T + b
    mean = h.mean(dim=-1, keepdim=True)  # dim=-1 = feature
    var = (h - mean).pow(2).mean(dim=-1, keepdim=True)  # dim=-1 = feature
    return gamma * (h - mean) / torch.sqrt(var + 1e-5) + beta
```

The comments record identity. They can rot. But they are present—a reader six months later can see what `dim=-1` was supposed to mean. When `h` gains a `seq` dimension, the comments say `feature` but the code now normalizes over `seq`. The comment is wrong. The reader has a chance to notice the mismatch. Without comments, there is no mismatch to notice—the code changed silently.

**Principle 2 applied: Reductions must name what they consume.**

```python
# h: (batch, feature)
def forward(x, W, b, gamma, beta):
    h = x @ W.T + b
    mean = mean[feature](h[batch, feature])
    var = mean[feature]((h[batch, feature] - mean) ** 2)
    return gamma * (h[batch, feature] - mean) / sqrt(var + 1e-5) + beta
```

The reduction names `feature`. The name appears in the bracket, not in a comment. If `h` gains a `seq` dimension, its declaration becomes `h[batch, seq, feature]`. The reduction `mean[feature]` still names `feature`—it does not silently switch to `seq`. The name protects the reduction from the layout change.

**Principle 3 applied: Broadcasts must name what they copy along.**

```python
# mean[batch], var[batch] — silent on feature, broadcast back over it
# gamma[feature], beta[feature] — silent on batch, broadcast along batch
let normalized[batch, feature] = gamma[feature] * (h[batch, feature] - mean[batch]) / sqrt(var[batch] + 1e-5) + beta[feature];
```

Two broadcasts. `gamma` and `beta` silently copy over `batch`. `mean` and `var` silently copy over `feature`. Every omission is visible in the index patterns—the coordinate that is absent from the bracket is the coordinate the tensor broadcasts over. The backward pass will sum over the appropriate coordinate for each parameter.

**Principle 4 applied: Functions must declare their coordinate contracts.**

```python
fn layer_norm[feature](x: [f32; ..batch, feature], gamma: [f32; feature], beta: [f32; feature])
    -> [f32; ..batch, feature]
```

The coordinate parameter `feature` is part of the function's type. Every call site that passes `feature` is checked: does the tensor have a coordinate called `feature`? The contract is not a docstring. It is verified.

**Principle 5 applied: Gradients read the forward pass backward.**

```
Forward: mean[feature](h[batch, feature]) → mean[batch] (broadcasts over feature)
         gamma[feature] * ... (broadcasts over batch)
Backward: d_mean[batch] = sum[feature](d_norm[batch, feature] * ...)
          d_gamma[feature] = sum[batch](d_norm[batch, feature] * ...)
```

The backward sums are over the coordinates that were broadcast forward. `mean` consumed `feature` → backward sum consumes `feature`. `gamma` omitted `batch` → backward sum consumes `batch`. The Inversion Rule, applied mechanically from the forward coordinate sets.

---

### The Principles Stack

None of the five principles requires the others. You can apply Principle 1 (name the coordinates) without changing your framework—add comments. You can apply Principle 2 (name the reductions) by choosing reduction functions that accept axis names. You can apply Principle 5 (the Inversion Rule) as a manual check when debugging gradient shapes.

But the principles compose. When you name coordinates (1), you can name what reductions consume (2). When you name broadcasts (3), you can check the backward pass against the forward pass (5). When you declare coordinate contracts (4), the compiler can check every call site against every principle simultaneously.

The five principles are a ladder. Each rung makes the next possible. The first three rungs are available in any framework—they require only discipline, not tooling. The last two require compiler support. But the first three, practiced consistently, catch the majority of coordinate bugs at code-review time, if not at compile time.

The habit begins at rung one. Name the coordinate. The rest follows.

---

## One Table: The Coordinate Audit

Every tensor operation can be audited with four questions. They are not Einlang-specific. They work in any framework because they are questions about meaning, not syntax.

| Question | What it catches | Chapter |
|---|---|---|
| Which coordinate is consumed? | Reduction over wrong axis | 2, 8 |
| Which coordinate is copied along? | Broadcast over wrong axis | 2, 7 |
| Can you trace a coordinate from source to destination? | Silent permutation/transpose | 1, 6, 9 |
| Does the backward reduction match the forward broadcast? | Gradient shape mismatch | 7 |

Ask these four questions of any tensor line. The answers tell you whether the notation preserved the facts that correctness depends on.

---

## Debugging with the Audit

The audit table is also a debugging tool. When a bug manifests as a wrong output shape or a wrong gradient, walk the audit questions backward from the symptom.

**Symptom: gradient has wrong shape.** The backward reduction doesn't match the forward broadcast. Check Question 4: which coordinate was broadcast forward? Sum over it backward. If the backward sum is over a different coordinate, the shapes will differ at exactly that coordinate. Trace the forward broadcast. Find where the coordinate was omitted. The omission is the bug.

**Symptom: output values look normalized over the wrong axis.** A softmax output summing to 1.0 over rows instead of columns. Check Question 1: which coordinate was consumed by the softmax reduction? If it was `dim=-1` but the intended coordinate is not the last one, the consumption is wrong. The fix is a `dim` change or a transpose before the softmax. The audit question tells you what to look for.

**Symptom: loss is slightly worse after a refactoring, but all tensor shapes match.** A coordinate was silently permuted. Check Question 3: trace one coordinate from the data entry point (data loader) through every operation to the loss. Find where the coordinate's position changed without the code recording the change. The position change is the bug. The name that wasn't there is the root cause.

**Symptom: batch normalization behaves differently after adding a sequence dimension.** The batch statistics are computed over the wrong set of coordinates. Check Question 1: which coordinates are reduced by `mean`? If the reduction consumed `batch` (correct) but also consumed `seq` (wrong), the statistics are being pooled across the wrong dimensions. In a positional API, this is a `dim` tuple audit. In a named API, the reduction bracket names the consumed coordinates, and adding a dimension doesn't change the bracket.

The four questions are a checklist. Run through them in order. The answer to at least one will be "I don't know from reading the code." The I-don't-know is a gap. The gap is where the bug lives.

---

The syntax has a small surface area. Once you internalize the primitives—naming, reducing, broadcasting, recurring, differentiating—you can regenerate most of what you need from first principles. The thought map above shows how they connect. The syntax reference records what they are.

The syntax will evolve. The thought map will grow. The habit—write the coordinate names, make the omissions explicit, let the compiler check the contracts—will outlast any particular syntax.

---

## The Book's Vocabulary

This book built a naming system for the ideas it introduced. Here they are, gathered in one place.

**Megaphone.** A tensor speaks on the coordinates in its brackets and stays silent on all others. Broadcasting is the repetition of silence.

**Consume.** A reduction consumes a coordinate—eliminates it from the output. A broadcast consumes silence—repeats a value along a coordinate it does not have. A compiler consumes the name—burns it into an integer after all checks pass.

**Shopping cart / Ledger.** The forward pass records which coordinates each tensor omits. The backward pass reads the record in reverse: what was omitted forward becomes summed backward.

**Skeleton.** A normalization operation has a fixed coordinate structure: reduce some coordinates, broadcast statistics back, apply affine parameters. The skeleton is the same for BatchNorm, LayerNorm, InstanceNorm, GroupNorm, RMSNorm. Only which coordinates are reduced changes.

**Firewood.** A name is firewood for the compiler. It burns into an integer at lowering. A good abstraction is good firewood—its beauty is in the light the flame casts when it burns.

**Panorama.** The five forms of a name seen simultaneously: Source → IR → After Analysis → After Lowering → Generated Code. One name, five forms, zero loss of identity.

**Coordinate habit.** The reflex of pausing before a tensor operation and asking: which coordinate is being consumed, copied, or moved—and is its name in the code? Not a skill. A change in what you notice.

**Shape-meanings gap.** The shape says *how many*. The role says *which one*. Every framework knows the shape. None of them know the role. The gap is where the bugs live.

**Inversion Rule.** Forward broadcast becomes backward reduction. Forward reduction becomes backward broadcast. The coordinate names are the thread connecting the two directions.

**Five check rules.** Index Existence (Rule 1), Reduction Consistency (Rule 2), Broadcast Recording (Rule 3), Causality (Rule 4), Coordinate Contract (Rule 5). The five ways a name can be wrong, and the five questions the compiler asks to catch it.

**Lowering.** The final stage of the compiler: names become integers. The name is burned. The integer is correct because the name was verified.

These words are not decoration. They are the book's own coordinate system. Their job is the same as the job of the bracket: to give a fact a place to live, so it can be checked.

Turn the page.
