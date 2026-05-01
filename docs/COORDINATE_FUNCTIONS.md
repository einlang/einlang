# Coordinate-Aware Functions and Selection Reductions

Status: implemented language surface, with some analysis metadata and backend
coverage still evolving.

Coordinate-aware functions are ordinary-looking calls with one extra piece of
syntax: bracketed coordinate arguments before the value arguments. They let
function boundaries preserve the coordinate facts that tensor code relies on,
without making every common operation expand into raw indexed formulas.

```rust
let p[b, class] = softmax[class](logits[b, class]);
let pred[b] = argmax[class](p[b, class]);
```

Read `softmax[class]` as "normalize over the `class` coordinate." Read
`argmax[class]` as "select an address in the `class` coordinate domain."
The bracket is not decoration and it is not an axis number. It is part of the
call's contract.

Ordinary pointwise functions do not need coordinate arguments:

```rust
let y[b, f] = relu(x[b, f]);
let z[b, f] = exp(y[b, f]);
```

Use coordinate arguments when the callee consumes, normalizes, selects, routes,
or otherwise needs to refer to a coordinate domain by identity. The rest of
this page is both user reference and implementation notes. For the short syntax
reference, see [reference](reference.md#coordinate-aware-functions).

## Design Goal

The user-facing syntax must be short enough to use in real model code.
Coordinate precision is valuable only if the common path stays light.

Einlang lets a library function hide an implementation, but not hide the
coordinate contract that makes the implementation meaningful.

Common calls look like this:

```rust
let h = relu(conv3d(x, weight, bias));
let p = softmax[class](logits);
let pred = argmax[class](logits);
let route = argmax[expert](gate_prob);
```

The guiding rule is:

```text
Callers name only the coordinates whose identity the callee must use.
```

The second rule is just as important:

```text
Convenience calls must lower to precise coordinate contracts.
```

If `argmax[class](logits)` resolves to the `class` coordinate, the result
carries the corresponding address-domain fact.

## Ergonomic Surface, Precise Core

The concise coordinate-argument form is the user-facing form:

```rust
softmax[class](x)
argmax[class](x)
```

When a standard-library helper has a legacy positional form, that form may
remain available, but the coordinate-aware form is the one that preserves role
meaning:

```rust
softmax[class](logits)
argmax[class](logits)
```

These calls are not wrappers around a more verbose public API. They are the
coordinate-aware API.

## Coordinate Facts Follow Values

Coordinate names should not float freely through lexical scope. A coordinate
name is valid at a call only when it can be grounded in a value or in an
expected result shape.

```rust
let logits[b, class] = model(x);
let p = softmax[class](logits);  // ok: logits carries class
```

Coordinate facts also flow through pointwise expressions:

```rust
let x[class in 0..3] = if class == 1 { -5.0 } else { class as f32 };
let pred = argmax[class](x ** 2.0);  // ok: x ** 2.0 still carries class
let energy = sum[class](x ** 2.0);   // reductions use the same rule
```

This is not legal unless `raw` already has a `class` coordinate fact:

```rust
let raw = model(x);
let p = softmax[class](raw);     // error: class is not grounded
```

If a value does not carry coordinate facts, make the body explicit:

```rust
let raw = [-1.0, 3.0, -5.0];
let pred = argmax[class](raw[class] ** 2.0);
```

Valid grounding sources are:

```text
1. An explicit indexed argument, such as logits[b, class].
2. The left side of the current binding, such as let p[b, class] = ...
3. Existing coordinate facts on a named value.
4. A function signature or return type.
```

The name `class` in `softmax[class](logits)` must resolve through
`logits` or through the expected result. It must not resolve through an
unrelated earlier binding that happened to use the same label.

## Same Name Means Unify

Within one expression or one function instance, the same coordinate name is a
request to use the same domain.

```rust
let z[b, class] = x[b, class] + y[b, class];
```

Here `x` and `y` must agree on both `b` and `class`. Matching extents are not
enough. The domains must be the same or must be made compatible by an explicit
cast or rename.

Different bindings may reuse a name independently until they are combined:

```rust
let train_logits[train_b, class] = ...;
let eval_logits[eval_b, class] = ...;
```

This is preferable to writing both batches as `b` when the two batch domains
are not actually the same.

## Selection Reductions

`argmax` and `argmin` are reduction-like forms, but they are not ordinary
numeric reductions.

```rust
let pred[b] = argmax[class](logits[b, class]);
let route[b, t] = argmax[e](gate_prob[b, t, e]);
```

`max[e](body)` returns a value. `argmax[e](body)` returns an address in the
`e` coordinate domain.

The relevant facts are:

```text
argmax[e](body) scans e.
The result is an integer tensor over the remaining coordinates.
The result values are addresses in the e coordinate domain.
The result is not an ordinary differentiable numeric value.
```

This distinction matters later:

```rust
let route[b, t] = argmax[e](gate_prob[b, t, e]);
let y[b, t, o] = expert_output[route[b, t], o];
```

The compiler records that the integer value `route[b, t]` is an
expert-domain address, not an arbitrary integer.

## Integer Indices With Domain Contracts

The design does not require a new `Index[...]` type constructor. Selection
operations can return ordinary integer tensors, while the function contract
records which coordinate domain those integer values address.

```rust
[i32; b]        // values address class
[i64; b, t]     // values address expert
[i32; ..rest]   // values address item
```

Example:

```rust
argmax[j](x[..left, j, ..right])  // [i32; ..left, ..right]
```

The result is still an integer tensor. The extra contract says that each
integer is valid as an address in the `j` domain. That address-domain fact is
part of the function's semantic contract, even though it is not expressed as a
new return type in this sketch. Later indexed reads can use the fact for range
checking, diagnostics, and optimization.

## Domain, Range, and Extent

This document uses `domain` deliberately.

```text
domain   the coordinate identity and legal address set, such as class or expert
range    a numeric iteration interval, such as 0..n
extent   the size of a coordinate domain
```

For example, `class` and `expert` may both have extent `1024`, but they are not
the same domain. Likewise, an integer result may have numeric range `0..1024`
while still needing the stronger fact that it addresses `class`.

This is why the selected coordinate in `argmax[j](...)` means a coordinate
domain, not just a numeric range.

## Same-Domain Local Binders

Function bodies often need a local scan coordinate over the same domain as a
coordinate parameter. The concise rule is:

```text
Inside a reduction or selection, a coordinate parameter may be rebound as the
local scan coordinate for that expression.
```

Stable softmax shows why this is needed:

```rust
fn softmax[j](x: [f32; ..left, j, ..right])
    -> [f32; ..left, j, ..right]
{
    let m[..left, ..right] =
        max[j](x[..left, j, ..right]);

    let e[..left, j, ..right] =
        exp(x[..left, j, ..right] - m[..left, ..right]);

    let z[..left, ..right] =
        sum[j](e[..left, j, ..right]);

    e[..left, j, ..right] / z[..left, ..right]
}
```

In the function signature, `j` names the selected coordinate domain. In
`max[j]` and `sum[j]`, `j` is rebound as a local scan coordinate over that same
domain. The rebinding is scoped only to the reduction body. Outside the
reduction, `j` still names the output coordinate position.

## Rest Packs

`..rest` is a rank-polymorphic coordinate pack. Functions that select or
normalize an axis should not be forced to place that axis last; they can use
packs on both sides of the selected coordinate.

```rust
fn center[j](x: [f32; ..left, j, ..right])
    -> [f32; ..left, j, ..right]
{
    let mean[..left, ..right] =
        sum[j](x[..left, j, ..right]) / extent(j);

    x[..left, j, ..right] - mean[..left, ..right]
}
```

Rules:

```text
The same pack name denotes the same coordinate sequence in one function
instance.
Different pack names may denote different coordinate sequences.
A pack is scoped to the function signature and body.
A pack is read-only; it cannot be rebound in the function body.
```

Coordinate parameter lists can bind packs directly:

```rust
fn id_axes[..axes](x: [f32; ..axes]) -> [f32; ..axes] {
    x
}

let y = id_axes[(height, width)](x);
```

The explicit coordinate argument list is strict but may be partial. Each
provided item binds one coordinate parameter in order; omitted coordinate
parameters are inferred from the ordinary arguments when the signature makes
that possible. Scalar parameters receive a bare coordinate:
`softmax[class](x)`. Pack parameters receive one parenthesized group when they
must be supplied explicitly: `id_axes[(height, width)](x)`.

```rust
fn move_channel[channel, ..spatial](x: [f32; channel, ..spatial])
    -> [f32; ..spatial, channel]
{
    let y[..spatial, channel] = x[channel, ..spatial];
    y
}

let y = move_channel[c](x);
```

Here `channel` is the only explicit choice. The spatial pack is whatever
remains after matching `[channel, ..spatial]` against the argument layout. The
parenthesized group rule avoids implicit splitting: `id_axes[height, width](x)`
has two coordinate arguments; it does not silently pack `height, width`.

This means:

```rust
fn add_same[j](x: [f32; ..left, j, ..right],
               y: [f32; ..left, j, ..right])
    -> [f32; ..left, j, ..right]
```

requires `x` and `y` to share the same `..left` and `..right` coordinates.

## Flexible Axis Placement

Coordinate-aware functions should behave like NumPy's `axis` argument in one
important sense: the selected coordinate may appear anywhere in the input, not
only in the last position.

```rust
fn argmax[j](x: [f32; ..left, j, ..right]) -> [i32; ..left, ..right]
```

Examples:

```rust
argmax[class](logits[b, class])       // left=[b], right=[]
argmax[class](logits[class, b])       // left=[],  right=[b]
argmax[class](logits[b, class, t])    // left=[b], right=[t]
```

All three calls select the `class` domain and preserve the other coordinates
in their original order. This is coordinate-based flexibility, not positional
guessing.

## Coordinate Parameters in Calls

The square brackets in a call are not comments. They are coordinate arguments
that must unify with the function signature.

```rust
fn softmax[j](x: [f32; ..left, j, ..right])
    -> [f32; ..left, j, ..right]
```

Legal:

```rust
let logits[b, class] = ...;
let p = softmax[class](logits);
```

Illegal:

```rust
let raw = ...;
let p = softmax[class](raw);
```

The second call has no way to prove that `raw` contains a `class` coordinate.
The caller should annotate, index, or cast the value first:

```rust
let logits[b, class] = raw[b, class];
let p = softmax[class](logits);
```

## When Calls Need Coordinates

Calls do not need coordinate arguments when the function preserves coordinate
structure pointwise:

```rust
relu(x)
sigmoid(x)
tanh(x)
exp(x)
log(x)
sqrt(x)
abs(x)
x + bias
```

Calls do need a resolvable coordinate argument when the function consumes,
selects, or normalizes over a coordinate:

```rust
sum[class](x[b, class])
argmax[class](logits[b, class])
softmax[class](logits)
```

The rule is:

```text
If the callee must refer to a coordinate domain by identity, that coordinate
must be named or inferred unambiguously.
```

## Convolution

Convolution is not pointwise, but ordinary callers should not have to write the
full indexed formula on every call.

This should be acceptable when the inputs already carry layout facts:

```rust
let y = conv3d(x, weight, bias, stride=(1, 1, 1), padding=(1, 1, 1));
```

with facts such as:

```text
x      : [b, ic, d, h, w]
weight : [oc, ic, kd, kh, kw]
bias   : [oc]
```

If layout is ambiguous, use a layout-specific wrapper:

```rust
conv3d_ncdhw(x, weight, bias)
```

The standard library definition should still expose the reference coordinate
contract internally: batch survives, input channels and kernel coordinates are
consumed, output channels survive, and output spatial coordinates are related
to input spatial coordinates by stride, padding, and dilation.

## Custom Differentiation

Einlang already has `@fn` for custom tangent rules. Coordinate-aware functions
do not add a new custom-differentiation primitive.

An STE helper can be a coordinate-aware standard-library function with an
attached `@fn` rule:

```rust
fn ste_top1[j](p: [f32; ..left, j, ..right]) -> [i32; ..left, ..right]
{
    argmax[j](p[..left, j, ..right])
}

@fn ste_top1[j](p: [f32; ..left, j, ..right]) {
    soft_surrogate_tangent[j](p, @p)
}
```

The important piece is not the ability to write a custom tangent. The important
piece is the source-level contract that the forward result is an address in
the `j` domain.

## Implementation Stages

The implementation is easiest to understand as two conceptual stages.

Stage 1: selection reductions and coordinate grounding.

```text
argmax[i](...)
argmin[i](...)
integer result domain facts for selection results
grounding checks for coordinate arguments
```

Stage 2: coordinate-aware function signatures.

```text
fn f[j](x: [T; ..left, j, ..right]) -> ...
rebind coordinate parameters in local reductions
builtin signatures for softmax and similar coordinate-aware helpers
```

## Implementation Design

The implementation should add coordinate awareness as a thin analysis layer over
the existing type, shape, and range passes. The core numeric type of an
`argmax` result remains `i32` or `i64`; the address-domain contract lives in
analysis metadata attached to expressions and bindings.

### Surface Forms to Parse

The implementation has to distinguish two source forms with the same token
shape:

```rust
argmax[class](logits)
softmax[class](logits)
```

The rule is name-based after parsing:

```text
all bracketed calls are parsed as calls

builtin coordinate intrinsics
    lower to reduction, selection, or intrinsic runtime forms

ordinary resolved callees with bracket arguments
    lower to coordinate-aware function calls
```

This keeps the source language uniform while still giving the compiler
special lowering hooks for reductions, selections, and axis-sensitive helpers.

### Parser and AST Changes

Update `src/einlang/frontend/grammar.lark`:

```text
function_def: "fn" NAME coord_params? "(" param_list? ")" ("->" type)? block
pub_function_def: "pub" "fn" NAME coord_params? "(" param_list? ")" ("->" type)? block
diff_rule_def: "@" "fn" NAME coord_params? "(" param_list? ")" block
coord_params: "[" NAME ("," NAME)* "]"
```

Add a neutral bracketed-call representation for:

```rust
NAME[coord_args](arguments)
```

During transformation, lower it to either `ReductionExpression` or
`FunctionCall` using the name rule above.

Extend the AST nodes:

```text
FunctionDefinition.coordinate_params: list[str]
DiffRuleDef.coordinate_params: list[str]
FunctionCall.coordinate_args: list[str]
ReductionExpression.loop_vars: already present
```

Extend rectangular type shapes to accept named rest packs:

```rust
[f32; ..left, j, ..right]
[i32; ..left, ..right]
```

The existing expression-level `IndexRest` machinery can be reused
conceptually, but type-shape packs should be represented separately so type
annotations do not pretend that `..left` is a runtime expression.

### IR Changes

Extend `FunctionCallIR` with:

```text
coordinate_args: tuple[IdentifierIR, ...]
```

Extend `FunctionDefIR` with:

```text
coordinate_params: tuple[DefId, ...]
signature_coordinate_layouts
```

`ReductionExpressionIR` already carries `operation`, `loop_vars`, `body`, and
loop ranges. Stage 1 only needs new operations:

```text
ReductionOp.ARGMAX
ReductionOp.ARGMIN
```

No new `Index` type is introduced.

### Coordinate Metadata

Add a small coordinate analysis result to the type context:

```text
value_layout[DefId] -> [CoordinateDomain]
expr_layout[ExpressionIR] -> [CoordinateDomain]
address_domain[ExpressionIR or DefId] -> CoordinateDomain
```

`value_layout` records the coordinate domains of tensor axes. For example:

```rust
let logits[b, class] = model(x);
```

records:

```text
logits -> [b, class]
```

`address_domain` records integer tensors whose values are valid addresses into
a coordinate domain:

```rust
let pred[b] = argmax[class](logits[b, class]);
```

records:

```text
pred layout        -> [b]
pred address value -> class
```

This metadata is not part of `RectangularType`. Types continue to describe
element type and rank; coordinate analysis describes axis identity and integer
address meaning.

### Stage 1 Lowering

Add `argmax` and `argmin` to `ReductionOp.parse` in
`src/einlang/shared/types.py`.

Support two equivalent source styles:

```rust
argmax[class](logits[b, class])
argmax[class](logits)
argmax[class](logits ** 2.0)
sum[class](logits ** 2.0)
```

The first form is an explicit reduction body. The second and third forms are
tensor-axis shorthand. After coordinate analysis proves that `logits` has a
`class` axis, they expand conceptually to:

```rust
argmax[class](logits[..left, class, ..right])
argmax[class](logits[..left, class, ..right] ** 2.0)
sum[class](logits[..left, class, ..right] ** 2.0)
```

The same shorthand applies to ordinary reductions such as `sum`, `prod`,
`min`, and `max`. For those reductions, multiple selected coordinates are
allowed:

```rust
let x[b, row, col] = ...;
let norm[b] = sum[row, col](x ** 2.0);
```

The expansion preserves the surrounding result context and indexes the reduced
coordinates explicitly:

```rust
sum[row, col](x[b, row, col] ** 2.0)
```

Nested reductions use the outer reduction coordinates as context for inner
shorthand:

```rust
let A[k, n] = ...;
let total = sum[k](max[n](A));
```

Conceptually, the inner reduction sees the outer `k` and expands to:

```rust
let total = sum[k](max[n](A[k, n]));
```

Selection reductions such as `argmax` and `argmin` currently select one
coordinate at a time because their result is a single address domain.
multi-axis addresses.

Shape behavior:

```rust
argmax[j](x: [f32; ..left, j, ..right]) -> [i32; ..left, ..right]
argmin[j](x: [f32; ..left, j, ..right]) -> [i32; ..left, ..right]
```

Backend behavior:

```text
NumPy backend: numpy.argmax / numpy.argmin along the selected axis, cast to i32
Scalar lowered path: track best_value and best_index in the loop accumulator
Autodiff: report that selection results are not differentiable unless wrapped by
          a custom @fn rule
```

The backend does not need to know coordinate domains. It only receives the
selected numeric axis. Coordinate analysis is responsible for mapping `class`
to that axis before lowering.

### Grounding Checks

A bracket coordinate argument must be grounded before lowering:

```rust
softmax[class](raw)
argmax[class](raw)
```

is legal only if `raw`, an indexed argument, the expected result layout, or the
callee signature proves that `class` exists.

The grounding pass should run after name resolution and before shape-sensitive
lowering. It should reject:

```text
unknown coordinate name
coordinate name present only in an unrelated binding
same label but different coordinate domains in one expression
selected coordinate absent from the value being reduced
```

### Stage 2 Function Matching

A coordinate-aware function is specialized by matching coordinate parameters
and rest packs against the actual argument layouts.

Definition:

```rust
fn softmax[j](x: [f32; ..left, j, ..right])
    -> [f32; ..left, j, ..right]
```

Call:

```rust
let p = softmax[class](logits);
```

If `logits` has layout:

```text
[b, class, t]
```

the matcher binds:

```text
j       = class
..left  = [b]
..right = [t]
```

The whole signature is instantiated from those bindings, not just the argument
list. The return type is interpreted with the same coordinate substitution:

```rust
fn top1[j](x: [f32; ..left, j, ..right]) -> [i32; ..left, ..right]

top1[class](logits[b, class, t])  // result coordinates: [b, t]
```

The same instantiation propagates rank and concrete dimensions when they are
known. If `logits` has shape `(B, C, T)`, the result of `top1[class](logits)`
has shape `(B, T)`.

This is a compiler-wide contract, not a frontend hint. Coordinate analysis,
shape analysis, and type inference use the same signature matcher, so
coordinate layout, `shape_info`, and rectangular return types agree about the
instantiated call. Later rewrites that rebuild calls must preserve
`coordinate_args`, `type_info`, and `shape_info`.

After coordinate analysis runs, the result is stamped directly onto IR
expressions:

```text
expr.coordinate_layout          // surviving named coordinates, e.g. ("b", "t")
expr.coordinate_address_domain  // for argmax/argmin, e.g. "class"
```

Later passes read those node fields. They should not rediscover coordinate
layout from scratch or depend on a side table except while coordinate analysis
itself is still constructing the facts.

The instantiated body then behaves like ordinary Einlang code with those
coordinate names substituted.

Inside the body, a coordinate parameter can be rebound only as the local binder
of a reduction or selection:

```rust
let m[..left, ..right] = max[j](x[..left, j, ..right]);
```

That local rebinding is expression-scoped. It must not create a mutable
coordinate variable in the surrounding block.

### Builtin Placement

Coordinate-sensitive reductions and helpers are not ordinary stdlib functions.
They should be builtin functions with explicit coordinate signatures and
intrinsic lowering:

```rust
builtin fn sum[j, T](x: [T; ..left, j, ..right]) -> [T; ..left, ..right];
builtin fn max[j, T](x: [T; ..left, j, ..right]) -> [T; ..left, ..right];
builtin fn min[j, T](x: [T; ..left, j, ..right]) -> [T; ..left, ..right];
builtin fn argmax[j, T](x: [T; ..left, j, ..right]) -> [i32; ..left, ..right];
builtin fn argmin[j, T](x: [T; ..left, j, ..right]) -> [i32; ..left, ..right];
builtin fn softmax[j](x: [f32; ..left, j, ..right])
    -> [f32; ..left, j, ..right];
```

An ordinary helper can still compose these intrinsics, but the axis-bearing
surface entry point should not live in `stdlib/`:

```rust
let p = softmax[class](logits);
let pred = argmax[class](logits);
```

This is the important split:

```text
builtin coordinate intrinsics own axis contracts and lowering
stdlib functions are ordinary functions without privileged coordinate meaning
```

### Pass Order

The recommended pass order is:

```text
parse
module collection
name resolution
AST to IR
coordinate grounding and layout analysis
type inference
shape analysis
rest-pack specialization for coordinate functions
range analysis
Einstein lowering
backend codegen or interpretation
```

If existing generic-function monomorphization already owns specialization,
coordinate-function specialization should plug into that service instead of
creating a parallel mechanism.

### Tests

Add parser tests for:

```rust
fn softmax[j](x: [f32; ..left, j, ..right]) -> [f32; ..left, j, ..right]
@fn ste_top1[j](p: [f32; ..left, j, ..right]) { ... }
softmax[class](logits)
argmax[class](logits)
```

Add analysis tests for:

```text
argmax over the first, middle, and last coordinate
argmax result has integer type and an address-domain contract
softmax[class](raw) fails when raw has no class coordinate
same label with different domains fails when combined
```

Add backend tests for:

```text
argmax[class] on [b, class]
argmax[class] on [class, b]
argmax[class] on [b, class, t]
```

## Diagnostics

Diagnostics should talk about coordinate facts, not only shapes.

Example:

```rust
softmax[class](raw)
```

Possible message:

```text
coordinate `class` is not grounded in argument `raw`.
Annotate raw, index the argument, or bind the result shape.
```

Example:

```rust
x[b, class] + y[b, class]
```

when the two `class` domains are different:

```text
coordinate `class` in x and y have the same label but different domains.
Use an explicit cast or rename if this alignment is intended.
```

## Summary

The design keeps the call-site burden low while preserving the facts that
make coordinate programs inspectable:

```text
Coordinate facts follow values.
Coordinate names do not float freely through scope.
Callers name only the coordinates used by the callee's meaning.
Selection returns integer tensors with coordinate-domain contracts, not bare
integers with no facts.
Standard-library functions may hide implementations, but not coordinate
contracts.
```
