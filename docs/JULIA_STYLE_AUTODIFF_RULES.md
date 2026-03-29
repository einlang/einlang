# Julia-Style Autodiff Rules For Einlang

This document defines the Julia-style reverse-mode rules that Einlang is
targeting for tensor quotients and pullbacks.

It is intentionally a semantics document, not a ChainRules API document.
Einlang does not try to copy Julia's public autodiff interfaces verbatim.
Instead, it adopts the same core idea:

- reverse mode is a seeded pullback
- cotangents keep the structure of the primal
- tensor outputs are not scalarized for correctness

## Scope

This document covers the rules Einlang should implement for:

- `@x`
- `@num / @den`
- tensor-valued quotients
- scalar aliases of tensor slices
- reductions and indexed tensor expressions
- user function calls and nested composite expressions

This document does not define:

- the full Julia `ChainRulesCore` API
- `rrule`/`frule` surface syntax
- tangent object types like `ZeroTangent`, `NoTangent`, `Thunk`
- mutation rules or in-place array semantics

## Core Model

Given a primal expression `y = f(x)`, reverse mode computes a pullback:

- input: cotangent seed for `y`
- output: cotangent for `x`

In Einlang terms, the internal operation is:

- `build_seeded_pullback(expr, seed, wrt, ...)`

The seed is not optional. Every reverse-mode result is defined relative to a
chosen cotangent seed.

## Seed Rules

Julia-style seeded pullback means:

- scalar result: seed = `1`
- tensor result: seed = `ones(shape(result))`
- indexed tensor element: seed = `onehot(index, shape(result))`
- reduction result: start with the cotangent of the reduced result, then
  propagate it structurally through the reduction

For Einlang quotients this means:

- `@y / @x` where `y` is scalar uses seed `1`
- `@Y / @x` where `Y` is a tensor uses `ones(shape(Y))`
- `@(Y[i...]) / @x` uses a onehot seed at `i...`

## Structural Cotangent Rule

Cotangents must keep the same structure as the primal they correspond to.

That means:

- a tensor output produces a tensor cotangent seed
- indexing a tensor projects or scatters cotangents structurally
- reductions broadcast or contract cotangents structurally
- correctness must not depend on replacing tensor cotangents with scalars

This is the main semantic gap Einlang is closing relative to older mixed
symbolic/Jacobian shortcuts.

## Primitive Reverse Rules

For a seed `ȳ`:

- constant `c`
  - pullback = `0`

- identity `x`
  - pullback = `ȳ`

- alias `z = x`
  - pullback = `ȳ`

- addition `a + b`
  - `ā = ȳ`
  - `b̄ = ȳ`

- subtraction `a - b`
  - `ā = ȳ`
  - `b̄ = -ȳ`

- multiplication `a * b`
  - `ā = ȳ * b`
  - `b̄ = ȳ * a`

- division `a / b`
  - `ā = ȳ / b`
  - `b̄ = -ȳ * a / b^2`

- power `a ^ b`
  - `ā = ȳ * b * a^(b - 1)`
  - `b̄ = ȳ * a^b * log(a)`

- negation `-a`
  - `ā = -ȳ`

- smooth unary function `f(a)`
  - `ā = ȳ * f'(a)`

Examples:

- `exp(a)` -> `ā = ȳ * exp(a)`
- `log(a)` -> `ā = ȳ / a`
- `tanh(a)` -> `ā = ȳ * (1 - tanh(a)^2)`

## Indexing Rules

### Tensor Read

For `z = X[i...]`:

- the incoming cotangent for `z` is scattered back into `X`
- all non-selected positions get zero

In matrix form, this is not a full Jacobian build.
It is a structured scatter of the seed into the base tensor.

### Scalar Alias Of Tensor Slice

If:

- `e = X[i...]`

then Julia-style reverse semantics require:

- `@loss / @e`
  to equal
- `(@loss / @X)[i...]`

This rule is generic. It must not depend on specific op names or hand-written
special cases.

## Reduction Rules

Reductions do not destroy structure during reverse mode. They redistribute the
incoming cotangent over the reduced body.

### Sum

For:

- `y = sum[k](body[k])`

with incoming seed `ȳ`:

- broadcast `ȳ` across the reduced axes
- recurse into the body with that broadcasted seed

### Product

For:

- `y = prod[k](body[k])`

the pullback is equivalent to:

- `ȳ * prod(body) / body[k]`

distributed back through the body and then summed over the reduction axes.

### Max / Min

For:

- `y = max[k](body[k])`
- `y = min[k](body[k])`

the cotangent is sent only to the selected extremum location.

That means:

- pick the winning primal index
- send the incoming seed to the corresponding differentiated body slot
- all non-winning positions get zero

Tie behavior must follow the primal selection behavior used by Einlang's runtime.
Today that means "first argmax/argmin wins" where the backend already uses that
convention.

## Einstein / Tensor-Comprehension Rule

For an Einstein-style tensor definition:

- each output clause is pointwise in its own output indices
- the incoming seed must be aligned with those output indices
- then the clause body is differentiated with that aligned seed
- finally, contributions are summed over the output indices that were seeded

The key rule is:

- multiply or align by the output cotangent first
- recurse through the clause value second
- never scalarize the tensor output just to make the implementation simpler

## Conditional Rule

For:

- `if cond { a } else { b }`

reverse mode preserves the primal branch structure:

- reuse `cond`
- propagate the same seed through the taken branch expression

So the pullback is:

- `if cond { pullback(a, ȳ) } else { pullback(b, ȳ) }`

## Block / Let Rule

For a block:

- replay local bindings in program order
- propagate the incoming seed into the final expression
- local cotangents are computed through the replayed primal definitions

This is important because the reverse computation depends on the original local
primal values and local alias structure.

## Function Call Rule

For `y = f(args...)`, Julia-style reverse mode treats the call generically:

- resolve the callee
- build the pullback of the callee body
- substitute the primal arguments for the callee parameters
- propagate the incoming seed into the callee result
- return the cotangent with respect to the caller-side target

Correctness rules:

- no correctness dependence on callee names
- no correctness dependence on a whitelist like `softmax`, `relu`, `max_pool`
- no tensor-output scalarization shortcut

Optimizations are allowed, but only after the generic rule is correct.

## Function Rules With Known Derivatives

Julia-style semantics still allow primitive derivative rules for well-known
scalar functions, as long as they are used as local pullback rules, not as
name-based correctness hacks for tensor quotients.

In Einlang this corresponds to:

- `custom_diff_body` / `@fn` rules for scalar primitives
- applying the incoming seed to that local rule
- then continuing the reverse propagation structurally

## Zero Rule

If an expression does not depend on the variable being differentiated, its
pullback contribution is zero.

For tensors:

- zero means the zero tensor of the correct shape

For scalars:

- zero means scalar `0`

## What "Julia Alignment" Means For Einlang

For Einlang, Julia alignment specifically means:

- reverse mode is expressed as seeded pullback/VJP
- tensor quotients use structured seeds
- reductions propagate seeds structurally
- scalar aliases of tensor slices are handled generically
- function calls are differentiated generically through the callee body
- name-based backend fast paths are optional optimizations, not correctness
  mechanisms

It does not mean:

- reimplementing ChainRules itself
- exposing Julia tangent types at the Einlang surface
- matching every Julia package's public API

## Einlang Porting Checklist

The new Einlang system should satisfy all of these:

- `@tensor / @x` uses a seed of `ones(shape(tensor))`
- `@(tensor[i...]) / @x` uses a onehot seed
- `@loss / @slice_alias` equals projection of `@loss / @tensor_root`
- reductions propagate seeds structurally, not by scalarizing the callee result
- tensor calls are differentiated through the callee body, not by callee name
- `max`/`min` pullbacks respect Einlang's tie convention
- backend fast paths, if present, only preserve semantics already produced by
  the compiler

## Current Einlang Rule Inventory

The intended Julia-style rule coverage in Einlang's new system is:

- identifiers and aliases
- literals and zeros
- unary and binary scalar primitives
- rectangular indexing and slice aliases
- `sum`, `prod`, `max`, `min`
- `if`
- `block`
- Einstein/tensor clauses
- user function calls
- custom scalar derivative rules via `@fn`

The remaining work should be judged against this document, not against older
mixed symbolic shortcuts.
