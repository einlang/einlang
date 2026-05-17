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

The preceding fourteen chapters introduced Einlang's grammar piece by piece, each piece arriving when the concept it served had earned its introduction. What follows assembles them into one place, organized by category rather than by pedagogical necessity.

Think of it as the view from the summit. You climbed the mountain one trail at a time. Now you can see the whole range.

But first, draw your own map.

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
    |
    v
A coordinate has a name, a domain, a position (Ch1)
    |
    |---> Permutation: names survive position changes (Ch1)
    |
    |---> Reduction: the consumed coordinate is named (Ch2)
    |       |
    |       +---> Broadcasting: the omitted coordinate is visible (Ch2)
    |               |
    |               +---> Inversion Rule: broadcast <-> reduction dual (Ch2, Ch4)
    |
    |---> Coordinate-aware functions: names as type-level contracts (Ch3)
    |       |
    |       |---> Square Matrix Test: when extents equal, only names differ (Ch3)
    |       |
    |       |---> Pack polymorphism: ..b absorbs unknown leading dims (Ch5)
    |       |
    |       +---> Normalization skeleton: one pattern, four functions (Ch5)
    |
    |---> Recurrence: time as a directional coordinate (Ch6)
    |       |
    |       +---> Causality constraint: t-1 valid, t+1 rejected (Ch6)
    |
    |---> Complex terrain: splits, arithmetic, disambiguation (Ch7)
    |
    |---> Differentiation: the pullback reads the forward pass backward (Ch8)
    |       |
    |       +---> @fn: custom derivative rules carry coordinate contracts (Ch8)
    |
    |---> Comparisons: same computation, two notations (Ch11-13)
    |       |
    |       |---> Normalization: GroupNorm reshape chain vs named groups (Ch11)
    |       |---> Attention: identical PyTorch, distinct Einlang signatures (Ch12)
    |       +---> Physics: integer field indices vs named field coordinates (Ch13)
    |
    +---> Compiler construction (Ch9-10)
            |
            |---> IR: S-expressions preserve every name (Ch9)
            |
            |---> Analysis: range -> shape -> type, five check rules (Ch9)
            |
            +---> Lowering: names -> integers, strategies (Ch9-10)
                    |
                    +---> Firewood: names burn, heat remains (Ch9)
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
- A named rest: `..b` — absorbs zero or more adjacent axes.

Expressions are not allowed in the declaration bracket. `let fib[n-1] = ...` is an error. The left side names what is being defined. The right side computes it.

---

## Reductions

A reduction consumes a coordinate. *Introduced in Chapter 2; selection reductions in Chapter 5.*

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

Broadcasting is an omission in the indexing pattern. *Introduced in Chapter 2; self-audit in Chapter 4.*

```rust
let out[i, j] = A[i, j] + bias[j];   // bias omits i → broadcast over i
```

The omitted coordinate is the one being broadcast over. The megaphone model: `bias` is silent on `i`, so the compiler copies it across all values of `i`. The silence is a semantic claim: `bias` does not depend on `i`.

The Inversion Rule: what broadcasts in the forward pass is reduced in the backward pass. `bias[j]` omits `i` forward → `d_bias[j] = sum[i](d_out[i, j])` backward.

---

## Named Rest Indices

`..name` stands for zero or more adjacent axes, collectively named. *Introduced in Chapter 2; pack polymorphism in Chapter 5.*

```rust
let result[..b, j] = x[..b, j] + bias[j];
let row_sum[..b] = sum[j](x[..b, j]);
```

The same rest name must describe the same axis span within an expression. Packs make functions rank-polymorphic: the same `layer_norm[feature]` works on 2D, 3D, or 4D inputs.

---

## Where Clauses

A where clause filters or binds. *Introduced in Chapter 2; backward behavior in Chapter 8.*

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

A function may declare coordinate parameters. *Introduced in Chapter 3; pack parameters in Chapter 5.*

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

Packs (`..left`, `..right`, `..s`) make functions polymorphic over surrounding structure. A caller disambiguates by grouping: `softmax[(height, width)](x)`.

---

## Recurrence Relations

Self-referential declarations define sequences over time. *Introduced in Chapter 6.*

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

`@loss / @W` computes the gradient. *Introduced in Chapter 8.*

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

The preceding sections catalogued syntax. But syntax is only half the story. Each compiler pass depends on coordinate names to do its job. *These passes are described in Chapters 9–10.*

**Shape inference** (Ch9–10) reads coordinate names to decide whether an expression is legal before it runs. `sum[k](A[i, k] * B[k, j])` succeeds if `k` appears in both `A` and `B`. Under names, the contract is: `i` survives from `A`, `j` survives from `B`, `k` appears in both and is consumed.

**Range analysis** (Ch10) finds the domain of every axis: from array shapes, from literals, or from explicit declarations. Every coordinate gets a concrete range before code generation.

**Five check rules** (Ch9) verify the IR: index existence, reduction consistency, broadcast recording, causality, and coordinate contract at call sites. Each catches a class of bug that positional notation silently accepts.

**Gradient lowering** (Ch9) reads coordinate names to build the backward pass. The rule: preserve the coordinates of `W`, sum over everything else. Set subtraction, applied to coordinate names, derives the pullback.

**Storage planning** (Ch9) reads coordinate names to decide which tensors can share memory. A recurrence creates a dependency chain; the compiler allocates a rolling buffer.

**Kernel fusion** (Ch9) reads coordinate names to decide which operations can be merged. Operations that share surviving coordinates can fuse; operations across a reduction boundary cannot.

---

## Error Codes

Three errors are especially relevant to the coordinate habit. You won't memorize error codes from a book. But reading them now means you'll recognize them when they appear:

- **E0425 (Undefined Coordinate)**: a coordinate name is referenced but does not exist on the tensor.
- **E0308 (Coordinate Range Mismatch)**: two uses of the same coordinate name infer incompatible ranges.
- **E0061 (Coordinate Contract Violation)**: a function call supplies a coordinate argument that does not match the function's declared coordinate parameter layout.

### Error Walkthrough

A list of error codes is a reference. A walkthrough is a skill. Here are the three errors as you would encounter them in practice: source, message, reading, and fix.

---

**E0425: The typo that silence would swallow.**

You write a softmax with a coordinate that doesn't exist:

```rust
let probs[batch, class] = softmax[clss](logits[batch, class]);
```

The compiler responds:

```
error[E0425]: coordinate `clss` not found in this scope
  → line 12, column 35
  `clss` is not declared on any tensor in the reduction body.
  Declared coordinates on `logits`: batch, class
  Hint: did you mean `class`?
```

Reading the error: the compiler names the offending coordinate (`clss`), shows you which coordinates actually exist on the tensor (`batch, class`), and suggests the correction. In a positional API, `softmax(logits, dim=1)` would run without error—and silently normalize over whatever axis happens to be at position 1.

Fix: `s/clss/class/`. One keystroke, caught at compile time.

---

**E0308: The shape mismatch that surfaces before runtime.**

You multiply two matrices with incompatible contraction dimensions:

```rust
let C[i, j] = sum[k](A[i, k] * B[k, j]);
```

If `A` has `k = 64` but `B` has `k = 128`:

```
error[E0308]: coordinate range mismatch for `k`
  → line 8, column 25
  `k` inferred as 64 from `A[i, k]` (declared at line 5)
  `k` inferred as 128 from `B[k, j]` (declared at line 6)
  These ranges must be equal for contraction.
```

Reading the error: the compiler tracked the size of `k` through both declarations, compared them at the reduction site, and found a contradiction. The error names both tensors and both ranges. In a positional API, this becomes `RuntimeError: size mismatch, m1: [32 x 64], m2: [128 x 64]`—which tells you the shapes but not which axis is wrong, which declaration caused it, or what the expected size should be.

Fix: align the declarations of `k`. The error tells you exactly where to look.

---

**E0061: The contract that positional APIs leave as a comment.**

You call a function with a coordinate parameter that violates its contract:

```rust
fn layer_norm[feature](x: [f32; batch, feature]) -> [f32; batch, feature] {
    let mean[batch] = mean[feature](x);
    let var[batch] = var[feature](x, mean[batch]);
    (x - mean[batch]) / sqrt(var[batch] + 1e-5)
}

let h[batch, channel] = layer_norm[batch](x[batch, channel]);
```

The compiler responds:

```
error[E0061]: coordinate contract violation in call to `layer_norm`
  → line 20, column 27
  `layer_norm` expects coordinate parameter `feature` (consumed by reduction)
  Called with `batch`, which appears in `..left` position of argument `x`
  Expected layout: feature is consumed; batch survives
  Actual layout:   batch would be consumed; feature is not in reduction
```

Reading the error: `layer_norm`'s contract says "I consume `feature` and preserve `batch`." The call says "consume `batch`." The two don't match. The compiler shows both the expected contract and the actual call site, so you can compare them side by side. In a positional API, `layer_norm(x, dim=-1)` would run—and normalize over `batch` instead of `feature` if the tensor happened to be transposed.

Fix: `layer_norm[feature](x[batch, channel])`. The coordinate parameter matches the contract.

---

These three errors share a structure. Each one: (1) names the coordinate involved, (2) shows where it was declared and where it was used, (3) states what was expected versus what was found. The structure is not Einlang-specific. It is the structure of any good type error. The coordinate names make it possible.

No names → no E0425. No coordinate-aware functions → no E0061. The error codes are not arbitrary. They are the compiler saying, in structured form: "the name you wrote does not match the names the program declares."

---

## How to Use This Chapter

This reference is built to be revisited—opened to the section you need.

If you're writing a new Einlang function and can't remember the exact syntax for a recurrence declaration, open to "Recurrence Relations." If you're debugging a coordinate mismatch and want to re-derive the pullback rule, open to "Automatic Differentiation." If you're designing a new operation and want to check whether it fits the existing primitives, trace it through the Thought Map.

The syntax reference is the scaffolding. The thought map is the blueprint. Together they let you rebuild what you need without rereading the whole book.

But the most important section here is not the syntax. It is the four-question audit table. Those four questions work in any framework. They are the coordinate habit, reduced to its smallest portable form. Copy them. Tape them to your monitor. Use them on your next tensor bug.

---

## Five Principles

**1. Coordinates have identities.** `batch`, `channel`, `time`, `feature` are not positions — they are names. A position records where; a name records what.

**2. Reductions must name what they consume.** `sum[class](x)` names the erased coordinate. `x.sum(dim=1)` erases a position.

**3. Broadcasts must name what they copy along.** `bias[j]` omits `i` — the omission records the broadcast.

**4. Functions must declare their coordinate contracts.** The bracketed parameter is part of the type. The compiler checks every call site.

**5. Gradients read the forward pass backward.** What consumed forward broadcasts backward. What omitted forward sums backward.

These principles are not Einlang-specific. They apply in any notation that records coordinate identities — brackets, einops strings, comments. The habit outlasts any particular syntax.

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

```rust
// h: (batch, feature)
let avg[batch] = mean[feature](h[batch, feature]);
let var[batch] = mean[feature]((h[batch, feature] - avg[batch]) ** 2.0);
let normalized[batch, feature] = gamma[feature] * (h[batch, feature] - avg[batch]) / (var[batch] ** 0.5 + 1e-5) + beta[feature];
```

The reduction names `feature`. The name appears in the bracket, not in a comment. If `h` gains a `seq` dimension, its declaration becomes `h[batch, seq, feature]`. The reduction `mean[feature]` still names `feature`—it does not silently switch to `seq`. The name protects the reduction from the layout change.

**Principle 3 applied: Broadcasts must name what they copy along.**

```rust
# mean[batch], var[batch] — silent on feature, broadcast back over it
# gamma[feature], beta[feature] — silent on batch, broadcast along batch
let normalized[batch, feature] = gamma[feature] * (h[batch, feature] - mean[batch]) / sqrt(var[batch] + 1e-5) + beta[feature];
```

Two broadcasts. `gamma` and `beta` silently copy over `batch`. `mean` and `var` silently copy over `feature`. Every omission is visible in the index patterns—the coordinate that is absent from the bracket is the coordinate the tensor broadcasts over. The backward pass will sum over the appropriate coordinate for each parameter.

**Principle 4 applied: Functions must declare their coordinate contracts.**

```rust
fn layer_norm[feature](x: [f32; ..b, feature], gamma: [f32; feature], beta: [f32; feature])
    -> [f32; ..b, feature]
```

The coordinate parameter `feature` is part of the function's type. Every call site that passes `feature` is checked: does the tensor have a coordinate called `feature`? The contract is not a docstring. It is verified.

**Principle 5 applied: Gradients read the forward pass backward.**

```rust
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
| Which coordinate is copied along? | Broadcast over wrong axis | 2, 4 |
| Can you trace a coordinate from source to destination? | Silent permutation/transpose | 1, 5, 10 |
| Does the backward reduction match the forward broadcast? | Gradient shape mismatch | 8 |

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

## Bug Bounty: Spot the Silent Bug

The best way to internalize what the coordinate habit catches is to try catching bugs yourself. Below are five real-world patterns. Each one compiles and runs without error in a positional API. Each one is wrong. For each: (1) find the bug, (2) explain why the positional compiler is silent, (3) write the Einlang version that would have caught it.

---

**Bug 1: The Shifting Axis.**

```python
def process(x, w):
    x = x.transpose(1, 2)          # swap spatial and channel
    out = torch.softmax(x, dim=1)   # softmax over... which axis now?
    return out @ w
```

The transpose changed which axis sits at position 1. `dim=1` refers to a different coordinate before and after the transpose. The code runs. The softmax normalizes over the wrong axis.

In Einlang: `softmax[channel]` — the name doesn't shift when the tensor is transposed.

---

**Bug 2: The Vanishing Dimension.**

```python
def aggregate(features):
    pooled = features.mean(dim=(2, 3))   # pool spatial dims
    return pooled @ classifier            # [b, c] @ [c, classes]
```

`features` is `(batch, channel, height, width)`. `mean(dim=(2, 3))` pools over positions 2 and 3—which are `height` and `width`. Correct. But six months later, someone adds a temporal dimension: `(batch, channel, time, height, width)`. `(2, 3)` now pools `time` and `height`. `width` survives. The code runs. The pooling is over the wrong coordinates.

In Einlang: `mean[height, width](features)` — the names are the same regardless of position. Adding `time` doesn't shift them.

---

**Bug 3: The Broadcast That Shouldn't Be.**

```python
def apply_mask(scores, mask):
    return scores * mask
```

`scores` is `(batch, heads, seq_q, seq_k)`. `mask` is `(batch, 1, 1, seq_k)`. Broadcasting expands `mask` along `heads` (dim=1) and `seq_q` (dim=2). The code runs. But should the mask broadcast over `heads`? If different heads should see different masks—for example, head 0 attends locally, head 1 attends globally—broadcasting is semantically wrong. The positional broadcast is silent on *why* `heads` and `seq_q` are omitted.

In Einlang: `scores[batch, head, q, k] * mask[batch, k]` — the absence of `head` and `q` from `mask`'s brackets is recorded. The compiler confirms: broadcast over `head` and `q`. If that broadcast is not justified, the brackets make the unjustified claim visible.

---

**Bug 4: The Gradient Gap.**

```python
def contrastive_loss(embeddings_a, embeddings_b):
    logits = embeddings_a @ embeddings_b.T   # [b, b]
    labels = torch.arange(logits.shape[0])
    return F.cross_entropy(logits, labels)
```

`cross_entropy` consumes `dim=-1` (the class dimension). The output is a scalar. But the user intended `loss[batch]` — per-sample loss for later weighting. `cross_entropy(logits, labels, reduction='none')` would preserve the batch dimension, but that's a keyword argument, not a coordinate specification. The default reduction consumed a coordinate silently.

In Einlang: `cross_entropy[class](logits[batch, class], labels[batch])` — the bracket says which coordinate is consumed. If you want `loss[batch]`, you see that `class` was consumed and `batch` survived. The output coordinates are explicit.

---

**Bug 5: The Contract That Was Only a Comment.**

```python
# x: (batch, seq, feature)
# mask: (batch, seq) — broadcasts over feature
def masked_mean(x, mask):
    return (x * mask.unsqueeze(-1)).sum(dim=1) / mask.sum(dim=1, keepdim=True)
```

The comment says `mask.sum(dim=1)` reduces over `seq`. Someone refactors: `x` now has shape `(batch, seq, num_heads, feature)`. The `unsqueeze(-1)` still adds one dimension. `dim=1` now refers to `seq`—but wait, is `num_heads` at position 2? The `dim=1` reduction silently shifted from `seq` to... still `seq`? It depends on whether the refactor inserted `num_heads` before or after `seq`. The comment doesn't enforce anything. The position depends on convention, and convention is not checked.

In Einlang: `fn masked_mean[seq](x: [f32; batch, seq, ..rest], mask: [f32; batch, seq])` — the contract declares that `seq` is consumed, and the compiler checks it at every call site. Inserting a new dimension doesn't change which coordinate `seq` refers to.

---

Five bugs. None of them throw an error in a positional API. All of them produce wrong results that pass silently into downstream computations, where the symptom will be a slightly worse metric, not a crash.

The compiler checks described in this book—the five rules, the error codes, the lowering verifications—exist to catch these five bugs before they reach runtime. The compiler is not a luxury. It is a tool for making the coordinate habit machine-checkable.

A `dim=` argument in a positional codebase is an integer. Which coordinate it refers to is a question the code cannot answer—the answer lives in the programmer's head, or in the data loader's output shape, or in the documentation comment that may or may not be up to date. When the answer is "run the program to find out," the integer has already won. The gap between the integer and the identity is the bug's hiding place.

---

## The Book's Vocabulary

This book built a naming system for the ideas it introduced. Here they are, gathered in one place.

**Megaphone.** A tensor speaks on the coordinates in its brackets and stays silent on all others. Broadcasting is the repetition of silence.

**Consume.** A reduction consumes a coordinate—eliminates it from the output. A broadcast consumes silence—repeats a value along a coordinate it does not have. A compiler consumes the name—burns it into an integer after all checks pass.

**Shopping cart / Ledger.** The forward pass records which coordinates each tensor omits. The backward pass reads the record in reverse: what was omitted forward becomes summed backward.

**Skeleton.** A normalization operation has a fixed coordinate structure: reduce some coordinates, broadcast statistics back, apply affine parameters. The skeleton is the same for BatchNorm, LayerNorm, InstanceNorm, GroupNorm, RMSNorm. Only which coordinates are reduced changes.

**Firewood.** A name is firewood for the compiler. It burns into an integer at lowering.

**Panorama.** The five forms of a name seen simultaneously: Source → IR → After Analysis → After Lowering → Generated Code. One name, five forms, zero loss of identity.

**Coordinate habit.** The reflex of pausing before a tensor operation and asking: which coordinate is being consumed, copied, or moved—and is its name in the code? Not a skill. A change in what you notice.

**Shape-meanings gap.** The shape says *how many*. The role says *which one*. Every framework knows the shape. None of them know the role. The gap is where the bugs live.

**Inversion Rule.** Forward broadcast becomes backward reduction. Forward reduction becomes backward broadcast. The coordinate names are the thread connecting the two directions.

**Five check rules.** Index Existence (Rule 1), Reduction Consistency (Rule 2), Broadcast Recording (Rule 3), Causality (Rule 4), Coordinate Contract (Rule 5). The five ways a name can be wrong, and the five questions the compiler asks to catch it.

**Lowering.** The final stage of the compiler: names become integers. The name is burned. The integer is correct because the name was verified.

These words are not decoration. They are the text's own coordinate system. Their job is the same as the job of the bracket: to give a fact a place to live, so it can be checked.

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

The Epilogue returns to the Tuesday bug from Chapter 1 and asks what changes when a name has a place to live.
