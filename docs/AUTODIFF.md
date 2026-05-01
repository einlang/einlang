# Automatic differentiation

Einlang has autodiff built into the language. You write the primal program normally, bind the values you care about, and then ask for tangents or derivatives with `@`.

## The basic forms

```rust
let x = ...;
let y = ...;

let tx = @x;        // tangent of x
let dy_dx = @y / @x;
```

Core rules:

- `@x` asks for the tangent of the named binding `x`
- `@y / @x` asks for the derivative or Jacobian of `y` with respect to `x`
- bind the value first, then differentiate the binding

This is not a separate `grad(f)` API. The derivative request is part of the language itself.

## First examples

Scalar derivative:

```rust
let x = 3.0;
let y = x * x;
let dy_dx = @y / @x;
print(dy_dx);   // 6.0
```

Gradient of a loss:

```rust
let w = [1.0, 2.0, 3.0];
let loss = sum[i](w[i] * w[i]);
let dloss_dw = @loss / @w;
```

Tensor quotient:

```rust
let C[i, j] = sum[k](A[i, k] * B[k, j]);
let dC_dA = @C / @A;
```

## Custom derivative rules with `@fn`

Use `@fn` to attach a custom tangent rule to a function:

```rust
fn ratio(x, y) { x / y }

@fn ratio(x, y) {
    (y * @x - x * @y) / y ** 2.0
}
```

The rule name and parameters match the primal function. Inside the rule body,
`@x` is the tangent flowing through parameter `x`, `@y` is the tangent flowing
through parameter `y`, and the body returns the tangent of the function result.

This is how standard-library functions backed by external implementations stay
differentiable:

```rust
pub fn exp(x) {
    python::numpy::exp(x)
}

@fn exp(x) { exp(x) * @x }
```

An `@fn` rule is dormant unless an autodiff request reaches a call to that
function. Without a derivative request, the primal function runs normally.

Coordinate-aware functions can also have coordinate-aware custom rules. The
coordinate parameter list must match the primal function:

```rust
fn ste_top1[j](p: [f32; ..left, j, ..right]) -> [i32; ..left, ..right] {
    argmax[j](p[..left, j, ..right])
}

@fn ste_top1[j](p: [f32; ..left, j, ..right]) {
    soft_surrogate_tangent[j](p, @p)
}
```

This is useful for axis-sensitive helpers such as selections, routing, or
normalization, where the derivative rule must preserve the same coordinate
contract as the primal call. See [COORDINATE_FUNCTIONS](COORDINATE_FUNCTIONS.md)
for the full coordinate-function rules.

## What `@x` means

`@x` materializes the identity tangent of `x`.

- scalar `x` -> `1.0`
- tensor `x` -> a ones-like tangent with the same shape

This is mostly useful when you want to inspect or compose tangent behavior explicitly.

## Printing behavior

These two forms behave differently:

```rust
print(@y / @x);
```

That prints a symbolic relation for debugging.

```rust
let dy_dx = @y / @x;
print(dy_dx);
```

That prints the numeric result.

If you want the actual derivative value, bind it first.

## What works well

Autodiff is designed for the kinds of tensor programs Einlang is already good at:

- scalar arithmetic and elementwise tensor math
- reductions such as `sum`
- Einstein-style contractions such as matrix multiply
- expressions with `where` clauses
- many standard-library math and ML operations
- coordinate-aware calls such as `softmax[class](x)` when the called operation has a supported rule

In practice, if you can express the computation cleanly in Einlang, autodiff is often the first thing to try instead of finite-difference estimates.

## Practical expectations

- Tensor quotients may be represented lazily instead of materializing a dense Jacobian immediately.
- Some expressions and operations are better supported than others; an unsupported derivative path reports an error rather than silently approximating.
- Binding names matters. `@` works on named values, not arbitrary inline subexpressions.

## Example Groups

The small calculus examples introduce the syntax; the fitting and training
workloads show the same notation in larger programs.

From the repository root:

```bash
python3 -m einlang examples/autodiff_small.ein
python3 -m einlang examples/autodiff_matmul.ein
python3 -m einlang examples/applications/linear_regression_autodiff.ein
python3 -m einlang examples/applications/decay_calibration_autodiff.ein
python3 -m einlang examples/gradient_descent_autodiff.ein
cd examples/mnist && PYTHONPATH=../../src python3 -m einlang train_sklearn_digits.ein
cd examples/mnist && PYTHONPATH=../../src python3 -m einlang train_recurrence.ein
```

The examples fall into three groups:

- syntax and local sanity checks: `autodiff_small`, `autodiff_matmul`, `autodiff_chain`, `autodiff_loss`
- real fitting workflows: `applications/linear_regression_autodiff`, `applications/decay_calibration_autodiff`
- optimization loops: `gradient_descent_autodiff`
- model training showcase on separate train/test data: `mnist/train_sklearn_digits.ein`
- smaller training demos on bundled data: `mnist/train_recurrence.ein`, `mnist/train_full.ein`

If you want the clearest “autodiff is doing real ML training work” example, start with `mnist/train_sklearn_digits.ein`. It uses the full `sklearn.load_digits` train/test split and reaches strong held-out accuracy after multiple autodiff updates.

## Next

- For the full language rules, see [reference](reference.md).
- For coordinate-aware function contracts, see [COORDINATE_FUNCTIONS](COORDINATE_FUNCTIONS.md).
- For runnable programs, see [examples/README](../examples/README.md).
