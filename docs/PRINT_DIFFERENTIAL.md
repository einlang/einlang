# `print(@...)` behavior

**Status:** Current symbolic-print behavior.

`print(@...)` is the symbolic autodiff display path. It is not the same as binding an autodiff request to a value and printing that value later.

## Direct print

Examples:

```rust
let x = 3.0;
let y = x * x;

print(@x);         // @x
print(@y / @x);    // (@y / @x) · @x
```

Direct print is rewritten by `AutodiffPass` to symbolic autodiff intrinsics.

## Bound then printed

Examples:

```rust
let x = 3.0;
let y = x * x;

let dx = @x;
let dy_dx = @y / @x;

print(dx);      // 1.0
print(dy_dx);   // 6.0
```

Binding first switches to the numeric runtime path.

## Current guidance

- Use direct `print(@...)` when you want a symbolic relation
- Bind first when you want a numeric derivative/Jacobian value
- For executable autodiff requests, prefer named bindings such as `@y / @x`

## Code

- compiler rewrite: `src/einlang/passes/autodiff/__init__.py`
- symbolic runtime handling: `src/einlang/backends/numpy_ir_tensor_runtime.py`
