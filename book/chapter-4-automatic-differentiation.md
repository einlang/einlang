---
layout: book
title: "Chapter 4: Automatic Differentiation"
---

# Chapter 4: Automatic Differentiation

Once tensor equations have names, the language can ask how one named value
changes with respect to another. This chapter presents `@` as an expression
construct: derivatives are requested inside the same program that defines the
primal computation, and the compiler keeps the resulting graph lazy until a
value is actually needed. This integration of differentiation into the language
itself transforms calculus from a separate concern into a natural part of
programming, enabling algorithms that compose differentiation with other
operations seamlessly.

## 4.1 Derivatives as Expressions

The `@` syntax elevates differentiation from a library function to a
first-class language construct, enabling derivatives to be requested as easily
as any other computation. This design choice makes differentiation composable
with the rest of the language, allowing complex optimization algorithms to be
expressed directly.

Formulas:

```text
y = x^2
softmax(x)_i = exp(x_i) / sum_j exp(x_j)
```

Einlang:

```rust
use std::math::exp;

let x = 3.0;
let y = x * x;
let dy_dx = @y / @x;

let s[i] = exp(logits[i]) / sum[j](exp(logits[j]));
let ds_dlogits = @s / @logits;
```

The `@` form is source syntax, not a host-language `grad(f)` wrapper. This
distinction is crucial: differentiation becomes part of the program's logic,
not an external analysis applied to it.
`@y / @x` asks for the derivative of named binding `y` with respect to named
binding `x`.

For tensor outputs and tensor inputs, the result may be represented lazily by a
Jacobian-backed runtime value. That lets a program request a derivative without
eagerly materializing every entry of a huge Jacobian.

The current documentation distinguishes two printing paths:

```rust
print(@y / @x);      // symbolic debugging path
let dy_dx = @y / @x;
print(dy_dx);        // numeric value path
```

### Derivative Requests Live in the Program

In many systems, differentiation is a transformation applied outside the
program:

```text
grad(function)(argument)
```

Einlang instead keeps the question in the same source environment:

```rust
let y = x * x;
let dy_dx = @y / @x;
```

This changes what can be named. We can ask for the derivative of a final loss,
but we can also ask about an intermediate value:

```rust
let h[i] = x[i] * x[i];
let loss = sum[i](h[i]);
let dloss_dx = @loss / @x;
let dh_dx = @h / @x;
```

The source-level rule is simple: bind the value first, then ask about the
binding. The compiler and runtime can then decide whether the derivative is a
scalar, a gradient-shaped tensor, or a lazy Jacobian-like value.

### The Shape of a Sensitivity

Derivative values also have shapes. If `loss` is scalar and `W` is a matrix,
then:

```rust
let dloss_dW = @loss / @W;
```

has the shape of `W`: one sensitivity per parameter element. If `y` is a vector
and `x` is a vector, then:

```rust
let dy_dx = @y / @x;
```

denotes a Jacobian-like object: each output component may depend on each input
component. The implementation may avoid materializing that object densely, but
the source-level meaning is still a shaped derivative value.

This is why Einlang treats `@y / @x` as an expression. It is not an instruction
to mutate gradients into side buffers. It is a value that can be named, printed,
passed onward, or differentiated again when supported.

## 4.2 Chain Rule and Attention

> **Think More.** The chain rule is not only a theorem about derivatives; in a
> compiler it is also a traversal strategy over dependencies. Which intermediate
> values should be cached? Which paths can be ignored? How would the answer
> change if a program asks for many related gradients instead of one? This turns
> calculus into questions about scheduling, memory, and reuse.

Formula:

```text
Attention(Q,K,V) = softmax(QK^T / sqrt(d)) V
```

Einlang:

```rust
use std::math::{exp, sqrt};

let scores[b, h, q, k] =
    sum[d](Q[b, h, q, d] * K[b, h, k, d]) / sqrt(D as f32);

let denom[b, h, q] = sum[k](exp(scores[b, h, q, k]));
let probs[b, h, q, k] = exp(scores[b, h, q, k]) / denom[b, h, q];
let out[b, h, q, d] = sum[k](probs[b, h, q, k] * V[b, h, k, d]);

let dout_dQ = @out / @Q;
```

The chain rule follows the named dependencies. `out` depends on `probs`,
`probs` depends on `scores`, and `scores` depends on `Q` and `K`. The derivative
request does not require reorganizing the program around a new function
boundary.

The implementation answers supported requests through compiler and runtime
autodiff machinery. Unsupported derivative paths should produce errors rather
than silent finite-difference approximations.

### Sharing Structure

The attention fragment contains meaningful intermediate bindings:

```rust
let denom[b, h, q] = sum[k](exp(scores[b, h, q, k]));
let probs[b, h, q, k] = exp(scores[b, h, q, k]) / denom[b, h, q];
```

By naming `denom`, the program gives both the reader and the compiler a place to
attach meaning. In a derivative computation, the denominator and probabilities
are not just temporary arithmetic; they are nodes in the dependency graph. A
runtime representation can cache or share such nodes when evaluating derivative
requests.

This is the same design lesson again: names expose structure that would
otherwise have to be rediscovered.

## 4.3 Higher-Order Derivatives

> **Think More.** Higher-order differentiation tests whether derivative requests
> are truly expressions. If `@y / @x` can be differentiated again, the language
> must avoid expanding everything too early. What representation keeps the
> second question meaningful without turning the program into a giant formula?
> Follow that question toward sparse Hessians, lazy graphs, and the difference
> between symbolic form and practical computation.

Formula:

```text
d^2 y / dx^2
```

Einlang:

```rust
let x = 2.0;
let y = x * x * x;
let dy_dx = @y / @x;
let d2y_dx2 = @dy_dx / @x;
```

Because a derivative request produces a value, it can be named and then used in
another derivative request when the involved operations are supported.

For large tensor programs, the right representation is often lazy or structured.
Materializing a dense Hessian is rarely the default one wants.

### Naming the Intermediate Derivative

The spelling:

```rust
let dy_dx = @y / @x;
let d2y_dx2 = @dy_dx / @x;
```

is intentionally more explicit than writing everything inline. The first
derivative becomes a named source value. The second derivative is then an
ordinary derivative request over that value.

This style makes higher-order differentiation a consequence of expression
composition. It also leaves room for the implementation to avoid expanding a
large symbolic expression all at once.

### A Worked Reading: A Small Loss

Consider:

```rust
let pred[i] = sum[j](x[i, j] * w[j]);
let err[i] = pred[i] - target[i];
let loss = sum[i](err[i] * err[i]);
let grad = @loss / @w;
```

The program first creates a vector of predictions. Then it creates a vector of
errors. Then it consumes the example axis `i` to produce a scalar loss. The
derivative request asks how that scalar changes with respect to each component
of `w`.

The compiler-visible dependency chain is:

```text
loss -> err -> pred -> w
```

The reduction over `i` does not erase the dependency on `w`; it combines all
example contributions into one scalar objective. This is exactly the structure
one wants for gradient-based fitting.

## 4.4 Gradient Descent as Recurrence

> **Think More.** Gradient descent is often presented as assignment:
> `theta = theta - lr * grad`. Mathematically, it is a recurrence over versions
> of `theta`. What becomes easier to analyze if optimization is written as time,
> and what becomes harder for a programmer used to mutation? This is a good
> point to compare clarity for the compiler with familiarity for the human.

Formula:

```text
theta_t = theta_{t-1} - eta * grad L(theta_{t-1})
```

Einlang:

```rust
let weights[0, i] = init[i];
let weights[t in 1..T, i] = {
    let prev[j] = weights[t - 1, j];
    let pred[n] = sum[j](X[n, j] * prev[j]);
    let loss = sum[n]((pred[n] - y[n]) * (pred[n] - y[n]));
    let grad = @loss / @prev;
    prev[i] - lr * grad[i]
};
```

The optimization loop is a recurrence. Each version of `weights` is a point in
time, not a mutation of a variable.

This source form gives the compiler a dependency fact: `weights[t]` reads
`weights[t - 1]`. If later code only observes the final weight vector, storage
analysis can choose a finite rolling window instead of retaining every
historical version. The derivative request remains local to the update rule.

### Optimization Without Mutation

The usual imperative description of gradient descent says "update the weights."
The recurrence version says something slightly different:

```text
weights[t] is computed from weights[t - 1]
```

This is not merely aesthetic. The recurrence preserves the history as a
mathematical object, while still allowing the implementation to store only the
history it needs. The source is denotational; the storage strategy is
operational.

## 4.5 Differentiation Algorithms and Patterns

> **Think More.** First-class differentiation makes gradients values that can be
> named and inspected. Which algorithms become clearer when derivative requests
> are explicit in source, and which details should still belong to the lowering
> strategy?

First-class differentiation enables direct expression of advanced optimization
algorithms.

### Newton's Method with Hessians

Second-order optimization using Hessian matrices:

```rust
let x = initial_guess;
let f = x * x * x - 2 * x - 5;  // Function to minimize
let grad = @f / @x;
let hess = @@f / @x / @x;  // Second derivative

let x_next = x - grad / hess;
```

Higher-order derivatives support optimization methods that need second-order
information.

### Jacobian Computations

Computing derivatives with respect to multiple variables:

```rust
let params[a in 0..3] = [1.0, 2.0, 3.0][a];
let inputs[b in 0..4] = [0.1, 0.2, 0.3, 0.4][b];

let output = sum[a](params[a] * inputs[a]) + params[2] * params[2];
let jacobian[a, b] = @output / @inputs[b] / @params[a];
```

Jacobians enable sensitivity analysis and advanced optimization.

### Custom Loss Functions

Complex loss functions with automatic differentiation:

```rust
let pred[i in 0..10] = model_output[i];
let target[i in 0..10] = true_labels[i];

// Huber loss (robust to outliers)
let residual[i] = pred[i] - target[i];
let abs_residual[i] = abs(residual[i]);
let huber_loss = sum[i](
    residual[i] * residual[i] / 2 where abs_residual[i] <= delta
    else delta * (abs_residual[i] - delta / 2)
);

let grad = @huber_loss / @pred;
```

Complex loss functions become straightforward with AD.

### Implementation Models for Automatic Differentiation

**Forward-Mode AD**: Computes derivatives alongside primal values. Efficient
when there are few outputs but many inputs.

**Reverse-Mode AD**: Builds a computation graph and traverses backward.
Efficient when there are few inputs but many outputs.

**Symbolic AD**: Manipulates expressions symbolically before evaluation.
Enables algebraic simplifications but may be complex.

**Source-Code AD**: Transforms the source program to compute derivatives.
Provides full control but requires a more capable compiler.

**Operator Overloading AD**: Uses overloaded operators to track derivatives.
Simple to implement but may have performance overhead.

Each model affects how differentiation algorithms are implemented and
optimized.

## 4.6 What a Derivative Value Really Is

One reason automatic differentiation is often mystifying is that many systems
teach it operationally before they teach it semantically. Users learn that a
gradient "appears" after a backward pass, or that some runtime object will
accumulate sensitivities in hidden buffers, but they are given less help in
thinking about what a derivative request means as part of the program itself.
Einlang chooses a different entry point. It treats the derivative as a value
described in the source.

That does not mean every derivative is eagerly materialized as a dense tensor.
In fact, large Jacobians or Hessians are often exactly what we do not want to
construct blindly. It means that the language grants derivative requests the
same status as other expressions. They can be named, reasoned about, passed
through later code, and in some cases differentiated again.

### Values With Structure

Suppose we ask for

```rust
let dloss_dW = @loss / @W;
```

The result is not an opaque side effect on `W`. It is a shaped sensitivity whose
entries correspond to the entries of `W`. If `W` is a matrix, the gradient is a
matrix-shaped value. If the output were itself a vector, the result might be
Jacobian-like. The language therefore encourages an important mental habit:
derivatives are not exceptional bookkeeping artifacts. They are structured
objects whose shapes follow from the same source relationships as ordinary
values.

This matters for readers because it keeps differentiation integrated with the
rest of the language. The same question that applies to any binding applies here
too: what does this value vary over, and how can it be observed? That continuity
is one of the strongest arguments for making autodiff a source-language feature.

### Naming Sensitivities Changes the Style of Programs

When derivative results can be bound explicitly, programs begin to read
differently. A training step stops being "call an optimizer on a hidden model
state." Instead it becomes a chain of named relationships:

```rust
let pred = ...;
let loss = ...;
let grad = @loss / @weights;
```

Each stage can be inspected independently. That is valuable for debugging and
teaching, but it is also valuable for systems work. Once the derivative is a
named expression, the implementation can ask whether it should be represented
eagerly, lazily, sparsely, or through some structured operator. The program
does not have to choose that representation too early in order to remain clear.

### Derivatives of Intermediates Are First-Class Questions

Another benefit of this model is that intermediate sensitivities become ordinary
questions rather than special tooling features. We may ask for the derivative of
the final loss with respect to parameters, but we may also ask for the
sensitivity of an internal normalization constant, an attention probability, or
a hidden state. In systems where differentiation is framed only as an external
transformation of whole functions, those intermediate questions often feel like
debugging hacks. In Einlang they are natural:

```rust
let probs = ...;
let dprobs_dscores = @probs / @scores;
```

That naturalness broadens what the language can be used for. It supports not
only training but analysis, interpretability, and investigation of internal
behavior.

## 4.7 Traversal Strategies: Forward, Reverse, and Mixed Perspectives

A language-level derivative request does not force one implementation strategy.
This is important because automatic differentiation is not one algorithm but a
family of related techniques. The same source expression may be answered by
forward accumulation, reverse accumulation, symbolic rewriting, graph
interpretation, or some hybrid strategy.

Forward-mode reasoning is often easiest to explain for scalar-to-scalar or
small-input problems. One imagines perturbations flowing alongside the primal
computation. Reverse-mode reasoning is often more attractive for scalar-loss,
many-parameter settings, because sensitivities flow backward from a final
objective through the dependency graph. Einlang's source notation does not make
the user choose between these perspectives at the level of everyday program
meaning. It instead names the mathematical question and leaves room for the
implementation to pick an answering strategy.

### Why Source Clarity Helps Implementation Flexibility

This separation is not trivial. In many ecosystems, users are taught the
implementation model first, and their style of programming becomes constrained
by that model. They learn which control flow "breaks the graph," which mutation
patterns are tolerated, and which APIs are differentiable by convention. A
source language that integrates differentiation more directly can aim for a
cleaner contract: if the operation is supported and the dependency relation is
clear, then the derivative question is meaningful, regardless of whether the
runtime later chooses a tape, a transformed function, or symbolic graph nodes.

That clean contract is especially useful for a language whose other chapters are
already about visible structure. If names, reductions, and recurrences are all
part of the source semantics, then autodiff can rely on those same semantics
instead of bolting on a separate mental model.

### Mixed Strategies in Real Systems

Real implementations often mix strategies. A compiler may lower some
differentiable fragments symbolically, call specialized kernels for others, and
use reverse-mode accumulation across large tensor regions. A Hessian-vector
product may involve a nested use of forward and reverse perspectives. The point
of the source notation is not to hide that such complexity exists. It is to
prevent that complexity from infecting the everyday reading of the program.

The user should be able to say "this is the derivative of `loss` with respect to
`weights`" without also having to encode the storage representation of the tape
or the exact traversal order of internal graph nodes. By keeping the question at
the source level and the strategy at the implementation level, the language
respects both clarity and performance.

## 4.8 Differentiation as a Language Design Choice

There is also a larger design argument here. Many tensor systems support
automatic differentiation as an external capability attached to a library. That
approach is practical and often powerful. But it tends to inherit the shape of
the host language rather than shaping the language around differentiation. Einlang
takes the opposite route for a narrow domain. It asks what changes if
differentiation is treated as one of the core questions the language should know
how to express.

Several consequences follow. Names matter more, because derivative requests
attach to named bindings. Intermediate structure matters more, because
subcomputations may become explicit derivative targets. Reductions matter more,
because sensitivity through contraction must remain legible. Recurrence matters
more, because optimization loops and recurrent models both involve temporal
dependency structure. In short, autodiff is not an isolated feature. It presses
on the whole language design.

### Why the Question Belongs in Source

The strongest argument for source-level differentiation is not that the syntax
is shorter. It is that the language can talk about the same object the user is
thinking about. If a researcher writes down a loss and asks "how does this
change with respect to those parameters?" that is already a semantic question
about named quantities. Encoding the question directly preserves the user's
intent instead of translating it into an indirect API pattern.

That directness also helps when differentiation is not the end goal. A user may
need gradients for optimization, yes, but they may also need local sensitivity
analysis, Jacobian structure, or confirmation that an intermediate variable
influences a result at all. A language that can ask such questions natively is
useful beyond training loops.

### The Cost of Integration

Of course integration comes with obligations. Unsupported operations cannot be
silently hand-waved away. The language and its implementation need a principled
story for what is differentiable, how errors are reported, and how derivative
values are represented. In a small language, that burden is easier to shoulder
because the semantic core is tighter. Einlang does not need to make every
possible host-language trick differentiable. It needs to make the tensor core
and its immediate boundaries legible and robust.

### The Chapter in Context

This chapter is therefore about more than calculus. It is about what kind of
programming environment emerges when derivative questions are allowed to sit in
the same world as names, indices, reductions, and recurrence. The answer is a
language where optimization is easier to describe, easier to inspect, and easier
to connect to the rest of the computational structure. That is the wider
motivation behind the `@` syntax. It is a small mark carrying a large claim: the
change of one named value with respect to another is part of the program's
meaning, not merely an after-the-fact tool pass.

## 4.9 Reading Optimization Systems as Derivative Narratives

A large optimization system can look intimidating when read as layers, helper
functions, and training utilities. It becomes more manageable when read as a
narrative of derivative relationships. Which value is the objective? Which
intermediate quantities feed that objective? Which parameters are asking to be
held responsible for changes in the objective? Once those questions are asked,
the surrounding infrastructure often becomes easier to separate from the core
mathematics.

This is one reason first-class derivative syntax matters beyond elegance. It
encourages the reader to locate the real semantic hinge of an optimization
program. The hinge is not merely "run backward now." It is the explicit question
of how one named value varies with respect to another. Seen that way, gradient
descent, sensitivity analysis, and model inspection become different uses of one
language idea rather than disconnected library features.

A minimal optimization spine often looks like:

```rust
let pred = model(input, weights);
let loss = objective(pred, target);
let grad = @loss / @weights;
```

For readers coming from machine learning practice, this can be a relief. It
replaces procedure with interpretation. For readers coming from programming
languages, it reveals autodiff as a design issue about values, names, and
representations rather than merely a trick of compiler engineering. That double
readability is part of what this chapter is trying to earn.

## 4.10 Discussion: How Differentiation Changes Reading

Differentiation changes what is worth naming. Intermediate values are not only
temporary arithmetic; they may become the targets or sources of sensitivity
questions. Reduction boundaries matter because gradients flow back through
them. Recurrences matter because optimization itself can be expressed as a
time-indexed computation. Autodiff therefore feeds back into how a program is
organized, not only how it is executed.

This is the source-level point of `@y / @x`. A derivative request is a statement
about named values in the program. It gives the reader a place to ask: what is
the objective, what is the target, and which intermediate relationships should
remain visible because they explain the derivative?

## 4.11 Discussion: Differentiable Programming Beyond Training

Differentiable programming is often introduced through machine learning
training loops, but the idea is broader. A differentiable program can state what
it computes and which sensitivity questions about those values are meaningful.
That includes optimization, but also analysis, debugging, robustness checks, and
inspection of intermediate quantities.

In compact form:

```text
named values -> dependency structure -> derivative question
```

Source-level derivative requests make that chain readable. The gradient is not
only an operation performed by a runtime; it is a relation the program names.
That relation can then interact with the same source structures used elsewhere
in the language: indices, reductions, recurrences, and intermediate bindings.

## 4.12 Reading Optimization Programs

When an optimization program feels complicated, isolate four bindings: the
parameters, the forward result, the objective, and the derivative request. Those
points usually reveal the semantic skeleton of the computation.

```rust
let pred = model(input, weights);
let loss = mse(pred, target);
let grad = @loss / @weights;
```

That reading habit is the practical payoff of source-level autodiff. A gradient
is not hidden in a training harness; it is a named relation between values in
the program.

## Summary

Automatic differentiation adds sensitivity questions to the same source model
used for tensors and reductions.

- Derivative requests refer to named bindings, making differentiation a
  relationship between specific values rather than abstract functions;
- Scalar gradients, tensor gradients, and Jacobian-like objects share one
  notation, enabling consistent reasoning across different mathematical
  structures;
- Lazy derivative values avoid unnecessary dense materialization, enabling
  efficient computation of sparse or selective derivatives;
- Intermediate bindings expose reusable graph structure, allowing
  subcomputations to be differentiated independently;
- Optimization loops can be expressed as recurrences with local derivative
  requests, turning iterative optimization into structured temporal patterns.

The next chapter introduces recurrence, where time-like dependence becomes an
index relationship that can be analyzed alongside tensors and derivatives.
