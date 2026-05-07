---
layout: book
title: "Axis Roles Are Not Axis Positions"
---

# Axis Roles Are Not Axis Positions

> "There are only two hard things in Computer Science: cache invalidation
> and naming things."
>
> — Phil Karlton

You are debugging a training run at 2 AM. The loss went down for three epochs,
then exploded. You trace it to a normalization layer. Your input has shape
`[32, 128, 768]`. The normalization is supposed to average over features—axis 2.
But someone transposed the tensor three functions up the call stack. Axis 2 is
now time steps. The shapes still match. The code still runs. The only thing wrong
is the meaning of the number `2`.

This chapter gives you a way to never debug this bug again. The idea is simple
enough to fit in one sentence: **an axis position is a number; an axis role is
the reason that number matters.** By the end of this chapter, you will know how
to tell the compiler about roles—and how that changes what the compiler can tell
you back.

### A parking lot analogy

Imagine a parking lot with rows labeled A through H and columns numbered 1
through 20. You park your car in slot D-7. Three hours later, someone repaints
the lot. Row D is now Row F. Your parking ticket still says D-7. You walk to the
wrong spot.

This is what happens in a tensor program when you write `x.transpose(1, 2)`. The
data moved. The meaning moved. But nothing in the code tells you that transpose
turned features into time steps. The shape `[32, 128, 768]` is like a ticket
that says "your car is in an 8×20 lot"—true, but it doesn't tell you which row.

Axis roles are like giving each row and column a permanent name that survives
repainting. Even if someone rearranges the rows, the name on your ticket still
matches the name painted on the ground.

In this book, a **role** means the semantic responsibility an axis carries at a
particular point in a computation. A role can be "batch example," "query
position," "class," "previous hidden unit," or "feature coordinate being
summed." The same physical axis may carry different local roles in different
expressions, and two roles may have the same extent without becoming
interchangeable.

In a tensor of shape `[16, 64, 128]`, position `0` might be batch, position `1`
might be feature, and position `2` might be a packed spatial or head dimension.
But the shape alone does not say that. It gives the extents, not the story.

This distinction matters because positions are easy to preserve while roles are
easy to lose. A reshape can keep the total number of elements correct. A
transpose can keep the rank correct. A flatten can produce the expected final
size. None of those facts prove that batch stayed batch or that feature stayed
feature.

This is where anonymous axes do their quiet work. They do not usually cause a
crash. They let two different stories fit the same tuple of numbers. The work
here is to separate the numeric slot from the model role. Once those two things
are apart, later chapters can ask better questions than "which axis
number was that?" Two axes can both have extent `32`; that does not make
batch, time, and feature interchangeable.

The question is not whether a shape tuple can describe the layout. It can.
The question is whether a shape tuple distinguishes batch from time when
both happen to equal 32. If the answer depends on a comment three lines
up -- a comment that may have drifted out of date -- then the notation has
already failed. A notation that cannot tell two roles apart in the source
is a notation that forces the reader to carry that distinction in memory.
And memory drifts.

## Position or Name

A language designer has an immediate choice. The first option is to keep axis
positions as the only source-level identity. This matches existing array
libraries and keeps notation short, but it makes every later operation depend
on memory of what position `0`, `1`, or `2` meant at that point in the program.

The second option is to keep positions in the implementation but attach
comments or type annotations outside the expression. That improves
documentation, but the operation itself can still say `axis=1` while the
meaning lives somewhere else. The compiler receives the number; the reader
supplies the role.

Existing named-dimension tools explore this middle ground in useful ways.
PyTorch Named Tensors and xarray can attach names to dimensions and use those
names for alignment. That is already better than a naked shape tuple. The
question this book presses is slightly different: does the operation's source
semantics actually use the role, or is the role still mostly metadata attached
to an array? A name that aligns two arrays is helpful. A name that also appears
inside `sum[feature]`, `softmax[class]`, `argmax[class]`, or
`hidden[t] <- hidden[t - 1]` gives the compiler a stronger fact: which role was
consumed, preserved, omitted, shifted, or returned as an address.

Shape-annotation tools make a neighboring move. A Python boundary can say:

```python
def normalize(x: Float[Array, "batch feature"]) -> Float[Array, "batch feature"]:
    return softmax(x, axis=-1)
```

That catches many lies at the edge of the helper. The remaining weakness is in
the body: the operation still says `axis=-1`. The role reached the boundary,
but the expression consumed a position. Einlang's version keeps the same name
in both places:

```rust
fn normalize[feature](x: [f32; batch, feature]) -> [f32; batch, feature] {
    softmax[feature](x)
}
```

The trap is to stop at decorative names. If a tensor remembers that two axes
are called `time` and `feature`, but a transpose-like operation can still move
them by position without checking the role-level claim, the name helped the
reader more than the program. Names matter most when operations are forced to
use them.

Coordinate functions are the smallest version of that requirement:

```rust
let probs[b, class] = softmax[class](logits[b, class]);
let prediction[b] = argmax[class](probs[b, class]);
```

Both calls are short, but neither asks the reader to remember that `class`
happens to be axis `1`. The operation names and the coordinate roles meet at
the call site.

The same rule scales beyond one coordinate. If a helper only needs the channel
role, the caller should not have to spell every surrounding spatial role:

```rust
fn move_channel[channel, ..spatial](x: [f32; channel, ..spatial])
    -> [f32; ..spatial, channel]

let image[channel, row, col] = load_image();
let y = move_channel[channel](image);
```

The word `channel` is the explicit choice. The pack `..spatial` is inferred as
`row, col`. That is the difference between a coordinate function and a
positional wrapper around `permute`: the function quantifies over roles, not
over slot numbers.

So the argument is not "names on dimensions are useless." It is that names
become much more powerful when operations are written in terms of them, rather
than merely carrying them beside positional operations.

We are not just naming for clarity. We are naming so the compiler can
catch the error before the model trains on it. A name attached to a tensor
as metadata may survive a reshape. A name used inside `sum[feature]` or
`softmax[class]` becomes part of the operation's contract -- and the
compiler can check that contract against the roles it infers from the
surrounding expressions. The first is documentation. The second is a fact
the compiler can act on.

The third option is to make the role part of the expression:

```text
image[b, channel, row, col]
```

This is the choice Einlang makes. It costs a few names, but it removes a
permanent ambiguity. Once a role is part of the expression, later code can ask
whether that role survived, moved, disappeared, or was used as a local
coordinate. Without the name, those questions arrive too late.

The danger is easiest to see with two equal-sized axes. Suppose `time` and
`feature` both have extent `128`. A transpose that swaps them may leave the
shape unchanged. A positional program can look undisturbed while the meaning
has moved. The role name is the local clue that the move was intended.

## A Small Naming Discipline

The smallest useful move is to write the role as a coordinate:

```text
image[b, channel, row, col] = ...
```

Now the program has four coordinates with four jobs. A later expression can
choose to preserve some, consume some, or map some into new coordinates:

```text
gray[b, row, col] =
    sum[channel](weights[channel] * image[b, channel, row, col])
```

The coordinate `channel` is local to the sum and disappears. The coordinates
`b`, `row`, and `col` survive. The result is more specific than "rank 3"; it is
a family over batch and spatial position.

Read one point:

```text
gray[2, 17, 9]
```

This point is the weighted sum of channels at one batch item and one pixel. The
coordinate names make that sentence possible without looking up an axis table.

The same line also shows the design boundary. The reducer does not need a
global declaration that "axis 1 is channel." It introduces `channel` exactly
where the role matters and removes it exactly where the result no longer has a
channel coordinate. The formula is doing the documentation and the check at
the same time.

## When Positions Drift

A common bug begins innocently:

```python
x = x.transpose(1, 2)
y = layer(x)
```

If `layer` expects `[batch, feature, time]`, the transpose may have changed
more than memory layout. It may have changed the meaning of position `1`.

Named coordinates make that kind of drift visible:

```rust
let x2[b, time, feature] = x[b, feature, time];
let y[b, time, out] = layer(x2[b, time, feature]);
```

The second line now has to confront the actual roles. If the layer expects
`feature` before `time`, the mismatch is a local fact in the formula rather
than a later shape surprise.

## Role Names as Local Documentation

Good tensor code often has comments that look like this:

```python
# x: [batch, time, feature]
```

The comment is useful because the code needs more than a shape. It needs a
contract for each position. The problem is that the comment is not part of the
expression language. It may be correct when written and stale after a refactor.
It may describe the input to a function but not the intermediate value three
lines later.

An indexed binding brings that contract closer to the operation:

```text
projected[b, time, out] =
    sum[feature](x[b, time, feature] * W[out, feature])
```

Now the role names are doing work. `feature` is the coordinate consumed by the
projection. `out` is the coordinate introduced by the weight matrix and
surviving in the result. `b` and `time` are carried along. The binding is a
small piece of documentation, but it is also the formula.

Read one concrete point:

```text
projected[3, 12, 7]
```

This means "for batch item 3 and time step 12, compute output feature 7 by
summing over input features." That sentence is not only a paraphrase. It is
the coordinate structure of the expression.

Read that sentence again: "summing over input features." Without the word
`feature` in the source, the reader must infer which axis was summed. A
positional API says `axis=1`; the reader must consult a shape comment
or a mental model of the pipeline. The named form puts the inference in
the line itself. The distinction between "the code runs" and "the code
says what it means" lives in that one word.

## Same Shape, Different Program

Two tensors can share a shape and still be different kinds of objects:

```text
scores[b, class]
tokens[b, time]
```

Both might have shape `[32, 128]`. But `class` and `time` are not
interchangeable. Reducing over `class` computes something like a per-example
normalization or loss. Reducing over `time` computes something like a sequence
summary. A generic shape checker sees two axes of length `128`; the source
roles say what the axes mean.

This is where named coordinates become more than decoration. They let code
review focus on meaning:

```rust
let loss[b] = -sum[class](target[b, class] * log_probs[b, class]);
```

If someone accidentally writes `sum[b]`, the formula is visibly wrong. If they
write `axis=0`, the reader has to remember what axis `0` meant at that exact
point in the pipeline.

The discipline is modest: name the coordinate where the operation depends on
the role. Do not name everything for ceremony. Name the dimensions whose
meaning would otherwise be carried by memory.

## What Survives the Pool?

Consider the quiet promise made by this expression:

```rust
let pooled[b, channel] = max[row, col](image[b, channel, row, col]);
```

The roles that survive are more than positions in the result. They are the
parts of the original object still available for later reasoning. The consumed
roles are not only removed axes; they are facts that mattered locally and
then disappeared. Positions can tell you where an axis sits. Roles tell you why
it is there.

## Naming Without Over-Naming

At this point you may be thinking: if I have to name every axis in every
expression, my code is going to look like a novel. That's a fair objection. There
is a real danger here: if every tiny expression gets covered in long coordinate
names, the notation becomes heavy. The goal is not ceremonial naming. The goal
is to name the roles that determine whether the program is correct.

A temporary scalar does not need a grand title. A local helper inside a small
formula may not need more than `i` or `j`. But a dimension that distinguishes
batch from time, feature from class, or head from within-head feature deserves
a name because confusing it changes the program.

A useful rule is:

```text
Name the coordinate when the role would otherwise live in a comment.
```

If the code needs a comment saying `[batch, time, feature]`, the roles are
important enough to appear in the notation. If a transformation depends on
which axis is time and which axis is feature, use names. If a reduction must
remove class but preserve batch, use names. If a shape is incidental and the
meaning is obvious from a local expression, keep the notation light.

This balance matters for a production language. Explicitness is valuable only
when it buys clarity. The goal is not maximal verbosity. The goal is to move
semantic roles out of the reader's memory and into the lines where those roles
are used.

A visible dimension is therefore more than a longer axis label. It is a role
that participates in a contract. Once that contract is written down, every
later operation can be read by asking which roles survive, which roles change,
and which roles disappear.

## Same Extent, Different Role

The implemented compiler does not need to believe that `batch` is a special
primitive type in order to benefit from the role. It only needs scoped names
and consistent uses. If `b` is introduced on the left side of:

```rust
let centered[b, t, f] = x[b, t, f] - mean[b, t];
```

then every occurrence of `b` inside the body refers to that coordinate. If a
term uses `mean[t, b]` instead, the address relation has changed. Depending on
the extents, shape analysis may catch the mismatch. Even when extents happen
to agree, the source now gives reviewers a local clue that the role order is
suspicious.

This is the modest power of axis roles. They do not prove the whole model
correct, but they make a class of wrong-but-shape-compatible programs visible.
They let each line state the role alignment it claims. Once those claims are
in the program, tests can focus on numerical behavior instead of trying to
infer which axis the author meant.

That shift changes code review. Instead of asking a reviewer to remember that
position `1` means time in this file but feature in another file, the line
itself carries the distinction. The reviewer can spend attention on the actual
claim: should this operation preserve time, consume it, or move it into a new
coordinate? That is a better use of human memory.

## Role Audit Under Pressure

Consider a batch of token embeddings:

```text
x[b, t, f] = ...
```

Here `b` is the example, `t` is the token position, and `f` is the feature
coordinate. A positional library may store the same value with shape
`[32, 128, 768]`. That shape is useful, but it does not say which axis can be
mixed without changing the meaning of the model.

A layer-normalization-style operation over features should keep `b` and `t`
fixed while reducing over `f`:

```rust
let mean[b, t] = sum[f](x[b, t, f]) / feature_count;
let centered[b, t, f] = x[b, t, f] - mean[b, t];
```

The role audit is immediate. The mean is addressed by `[b, t]`, so it computes
one value per example and token. The feature coordinate is local to the
reduction and does not survive into `mean`. The centered tensor restores `f`
because each feature value is shifted by the mean for its own token.

Now compare a batch normalization over examples:

```rust
let mean[t, f] = sum[b](x[b, t, f]) / batch_count;
let centered[b, t, f] = x[b, t, f] - mean[t, f];
```

This program has the same input and output shape as the previous one. Both
produce `centered[b, t, f]`. A shape-only review can easily miss the semantic
difference because the final tensor still has rank three and the same extents.
The coordinate names expose the difference: one program consumes `f`, the
other consumes `b`.

That is why roles cannot be reduced to positions. In a model where `batch` and
`time` both have length `32`, a mistaken normalization may even use an axis of
the right size. The error is not "dimension 0 has length 32 but expected 128."
The error is "the program treated examples as the distribution instead of
features." That message requires role information.

The same audit works for pooling. A temporal average over a sequence is:

```rust
let pooled[b, f] = sum[t](x[b, t, f]) / time_count;
```

The result keeps examples and features, but consumes time. A feature average
would be:

```rust
let pooled[b, t] = sum[f](x[b, t, f]) / feature_count;
```

Both are legal reductions. Neither is intrinsically more correct without the
model contract. The point is that the source should say which contract it is
using.

This is the concrete habit behind role names: read every operation as a small
audit.

```text
Which coordinates survive?
Which coordinates are consumed?
Which coordinate is being reused because it is absent from a term?
```

```
   Survive / Consume / Omit Ledger

   y[b, f] = x[b, f] + bias[f]

   x[b,f] ---- b --> survive ------> y[b,f]
         ---- f --> survive ------>
                                   y[b,f]
   bias[f] --- f --> survive ------>
            (b absent --> omit --> reused across every batch item)

   Three coordinate fates stated in one line.
   b: survive.  f: survive.  bias omits b -- one line, three facts.
```

If the answer matters to correctness, the name is carrying real work. It is
not there for style. It gives the compiler and the reader a shared object to
reason about before the program collapses into anonymous positions.

This is the Hiding Law applied to the axis level. A positional program
hides which axis means what. That fact is recovered from convention --
from the order the dimensions were originally declared, from a comment
at the top of the file, from the reader's memory of the data pipeline.
A coordinate program puts the fact in the expression. The difference is
not verbosity. It is whether the fact survives a refactor that changes
the order of dimensions without changing their meaning.

A tensor shaped `[32, 32, 128]` is a useful last example: give the two `32`
axes different roles, then imagine an operation that remains legal after they
are swapped. The diagnostic you would want is not "size 32". It is the name of
the role that crossed the line.

## Summary

An axis position is a number. An axis role is the reason that number matters.

Where Chapter 1 showed that shapes can be correct while withholding the reason
they are correct, this chapter gave the missing vocabulary a name. A role name
survives a transpose because it is not tied to a slot. It survives a reshape
because it travels with meaning, not with layout. And it fails visibly when the
wrong role is consumed: `sum[class](scores[b, class])` destroys `class`, while
`sum[time](scores[b, class])` is the wrong reduction entirely—visible as a
one-word difference in the source.

The chapter introduced four enduring ideas:

1. **Position vs. Role.** Two axes can both have extent 32. The shape is the
   same. The roles—batch, time, feature—determine whether an operation is correct.
2. **Survive, consume, omit.** Every expression can be audited by asking which
   coordinates survive in the result, which are consumed by a reduction, and
   which are omitted from a term (forcing a broadcast).
3. **Names inside operations, not beside them.** The power of a role name comes
   not from attaching it to a tensor, but from using it inside `sum[feature]`,
   `softmax[class]`, and `max[row, col]`. The operation itself carries the
   coordinate claim.
4. **Name what would live in a comment.** If a dimension's meaning would
   otherwise need a comment saying `[batch, time, feature]`, it earns its
   place in the notation. Ceremonial names for temporary coordinates do not.

The role audit—"which coordinates survive, which are consumed, which are
omitted?"—will return in every subsequent chapter, culminating in the attention
audit of Chapter 14. The next chapter applies this vocabulary to standard-library
coordinate maps: transpose, flatten, and `depth_to_space`.

### Where This Leads

An axis position is a number the compiler can count. An axis role is a fact the
notation either states or buries. When the notation has no place for the role,
the role becomes invisible — to the compiler, to the autodiff engine, to the next
person reading the code.

You have now felt the difference. A parking ticket with just a shape is an
invitation to walk the wrong row. A tensor declared as `x[batch, time, feature]`
carries its own ticket — it knows what it is, not just how big it is. The
longing for a name, once you have debugged a silent axis swap at 3 AM, is not
aesthetic. It is scar tissue.

A name is a bridge between a number and its reason. Without it, you cross alone.
With it, the compiler crosses with you — and can tell you when the bridge is
broken.

Chapter 1 showed you that anonymous axes hide the question "where did the
coordinate go?" This chapter gave you the vocabulary to answer: the coordinate
audit — survive, consume, omit. A role and a position are different facts. The
compiler can count positions. It cannot count what it does not have a name for.

Chapter 3 applies that vocabulary to the standard library, where coordinate maps
written as positional permutations hide the address equations that govern
transpose, flatten, and depth-to-space.

## Try It

A tensor `scores[b, class]` and a tensor `tokens[b, time]` both have shape
`[32, 128]`. For each of these operations, write the coordinate audit (survive,
consume, omit) and mark whether it is semantically correct.
`sum[class](scores[b, class])` computes a per-example normalization.
`sum[time](tokens[b, time])` computes a sequence summary.
`softmax[class](tokens[b, class])` attempts to normalize across something
`tokens` does not have—`tokens` does not carry a `class` coordinate, so the
expression would fail at name resolution.

Now the insight: `sum[class](scores)` and `sum[time](tokens)` look different in
named notation but would both be `sum(dim=1)` in a positional API. When `class`
and `time` both have extent 128, they are the same positional operation. The
named form distinguishes them in one character. The positional form needs a
comment. Write one expression that is shape-legal for both tensors but
semantically wrong for one, and ask yourself: which coordinate did you just
consume?

Your teammate writes a layer normalization as:

```python
# x has shape [batch, feature]
normalized = (x - x.mean(dim=-1, keepdim=True)) / x.std(dim=-1, keepdim=True)
```

After a pipeline refactor that transposes the input from `[batch, feature]` to
`[feature, batch]`, the code still runs. Because `dim=-1` still refers to the
last axis—which is now `batch`. The code silently normalizes over examples
instead of features. Write both versions with named coordinates:

```text
// Before refactor (correct)
let mu[b, 1] = mean[f](x[b, f]);
let sigma[b, 1] = std[f](x[b, f]);
let normed[b, f] = (x[b, f] - mu[b, 1]) / sigma[b, 1];

// After refactor (wrong, but runs)
let mu[f, 1] = mean[b](x[f, b]);
let sigma[f, 1] = std[b](x[f, b]);
let normed[f, b] = (x[f, b] - mu[f, 1]) / sigma[f, 1];
```

Read the second version carefully. `mean[b]` consumes batch. The result `mu[f,
1]` has only `f` surviving. Each feature gets its own mean—computed over all
batch examples. The normalization now asks "how unusual is this example relative
to the batch?" instead of "how unusual is this feature value relative to other
features?" The shapes are identical. The code runs. The model trains. The
coordinate audit reveals the semantic difference instantly: one consumes `f`, the
other consumes `b`. There is no positional equivalent of this audit—you would
need to trace the transpose, remember the refactor, and mentally remap `dim=-1`.
The coordinate name tells you in one look.

Design a coordinate function `normalize_over[coord](x)` where the bracketed
coordinate is the one consumed by the normalization:

```text
fn normalize_over[coord](x: [f32; ..left, coord, ..right])
    -> [f32; ..left, coord, ..right]
{
    let mu[..left, ..right] = mean[coord](x[..left, coord, ..right]);
    let sigma[..left, ..right] = std[coord](x[..left, coord, ..right]);
    let y[..left, coord, ..right] =
        (x[..left, coord, ..right] - mu[..left, ..right])
        / sigma[..left, ..right];
    y
}
```

Now write three calls that differ only in the bracketed coordinate:

```text
let normed_by_feature = normalize_over[feature](x[batch, time, feature]);
let normed_by_time    = normalize_over[time](x[batch, time, feature]);
let normed_by_batch   = normalize_over[batch](x[batch, time, feature]);
```

Three different normalization stories. Same input tensor. Same output shape. No
positional argument changed—only the name in the bracket. In a positional API,
these three calls would be `normalize_over(x, dim=f_idx)`, `normalize_over(x,
dim=t_idx)`, and `normalize_over(x, dim=b_idx)`. When `batch_count == time_count
== feature_count == 128`, the shape tuple is `[128, 128, 128]`. All three
positional calls produce the same output shape `[128, 128, 128]`. The positional
API relies on the reader to know that `dim=0` is batch, `dim=1` is time, and
`dim=2` is feature at this specific call site in this specific file. The
coordinate API encodes that knowledge in the bracket character itself. The
coordinate name tells you instantly which role is being consumed. You do not
reconstruct roles from axis numbers. The bracket is the answer.

Now write the same three calls in PyTorch and JAX. The tensor has shape
`[batch, time, feature]`, all 128. At this call site, axis 0 is batch, axis 1
is time, axis 2 is feature. The reader must know this:

```python
# PyTorch — which normalization is which?
x = torch.randn(128, 128, 128)

# Normalize over feature: dim=-1 (or dim=2)
normed_feature = F.layer_norm(x, (128,))

# Normalize over time: dim=1
mu_t = x.mean(dim=1, keepdim=True)
std_t = x.std(dim=1, keepdim=True)
normed_time = (x - mu_t) / std_t

# Normalize over batch: dim=0
mu_b = x.mean(dim=0, keepdim=True)
std_b = x.std(dim=0, keepdim=True)
normed_batch = (x - mu_b) / std_b
```

```python
# JAX — same positional logic, same risk
import jax.numpy as jnp
x = jnp.ones((128, 128, 128))

normed_feature = jax.nn.standardize(x, axis=2)   # or axis=-1
normed_time    = jax.nn.standardize(x, axis=1)
normed_batch   = jax.nn.standardize(x, axis=0)
```

All three produce `[128, 128, 128]`. The loss curves descend. If a refactor
swaps `time` and `feature` in the tensor layout, the positional calls `dim=1`
and `dim=2` silently swap their targets. The Einlang calls
`normalize_over[time]` and `normalize_over[feature]` do not — `time` is still
`time`, wherever it lives in the shape. The bracket is the anchor that a
position number can never be.

**Line to keep:** equal lengths do not make two axes semantically
interchangeable.
