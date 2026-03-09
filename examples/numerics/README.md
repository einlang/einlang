# Numerics (reusable modules)

Reusable modules for ODEs, optimization, and dynamic programming. Use from your `.ein` files via `use numerics::diffeq`, etc. They follow the same patterns as Julia’s DifferentialEquations.jl, Optim.jl, and QuantEcon.jl.

| Module | What it provides |
|--------|-------------------|
| `diffeq` | `euler_decay_step`, `euler_decay` — explicit Euler for scalar decay |
| `optim` | `gradient_descent_step_2d`, `gradient_descent_2d` — gradient descent for quadratic 2D |
| `quantecon` | `bellman_step`, `value_iteration` — Bellman value iteration (3 states) |

## Run the demo

From repo root (runner lives in `examples/` so that `numerics::*` resolves):

```bash
python3 -m einlang examples/run_numerics.ein
```

## Use in your code

From any `.ein` file under `examples/`:

```rust
use numerics::diffeq;

let u = diffeq::euler_decay(1.0, 0.05, 0.1);
print(u);
```

Or use a single function:

```rust
use numerics::quantecon::{value_iteration};

let r = [0.0, 1.0, 2.0];
let P = [[0.5, 0.2, 0.1], [0.3, 0.5, 0.2], [0.2, 0.3, 0.7]];
let V = value_iteration(r, P, 0.95);
```

## Notes

- Small, fixed-size implementations (e.g. 50 steps, 2D or 3 states) so they compile without dynamic loop limits.
- For full solvers and adaptive methods, use Julia’s packages; here we provide the same *patterns* in Einlang.

**General numerics stdlib:** A proper stdlib numerics layer (no hardcoded sizes, like Julia) would require language support for variable-length recurrence and shape-agnostic state. See [Numerics stdlib design](../docs/NUMERICS_STDLIB_DESIGN.md).
