
# Numerics stdlib: design and constraints

**Goal:** A general-purpose numerics standard library (Julia-like: no hardcoded sizes, no fixed iteration counts). This doc explains why the current `examples/numerics/` modules are not suitable for stdlib as-is and what would be required for a proper `std::numerics` (or `std::ode`, `std::optim`, etc.).

## Constraint: no hardcoding in stdlib

Stdlib must be **general-purpose**:

- **No fixed dimensions** — e.g. not “3 states” or “2D only”; state size and shape must come from parameters or input.
- **No fixed iteration counts** — e.g. not “50 steps”; step count (or convergence) must be a parameter or derived from inputs.
- **No fixed problem sizes** — ODE state, optimization variable, and DP state space are user-defined.

So the current `examples/numerics/` code (fixed 51 steps, 2D, 3 states) is appropriate as **examples** that mirror Julia usage, but it must **not** be copied into stdlib without generalizing it.

## What a general numerics stdlib would provide

Aligned with Julia’s DifferentialEquations.jl, Optim.jl, and QuantEcon.jl:

1. **ODEs**
   - One-step steppers: e.g. `euler_step(u, t, dt, rhs)` where `u` and `rhs` are arbitrary-shaped; or specialized steppers (decay, linear) that take parameters.
   - Solvers: e.g. `solve_ivp_euler(u0, tspan, n_steps, rhs)` with `n_steps` (and optionally `tspan`) as parameters.
   - No hardcoded state dimension or step count.

2. **Optimization**
   - One step: e.g. `gradient_descent_step(x, grad, alpha)` for any `x`/`grad` shape.
   - Iteration: e.g. `gradient_descent(f, grad_f, x0, options)` with `max_iter` and possibly convergence tolerance as parameters.
   - No hardcoded 2D or 31 steps.

3. **Dynamic programming (QuantEcon-style)**
   - One Bellman step: `bellman_step(V, r, P, beta)` where `V`, `r`, `P` have shapes derived from parameters (e.g. `n_states`).
   - Value iteration: e.g. `value_iteration(r, P, beta; n_states, max_iter)` with no fixed 3 states or 50 iterations.

## What the language and compiler need

To implement the above in Einlang **without** hardcoding:

1. **Variable-length recurrence**
   - Recurrence over `t in 1..n_steps` where `n_steps` is a function parameter (or input).
   - Backend already evaluates ranges at runtime in some paths; this must work reliably for Einstein recurrence so that “number of steps” is not a literal.

2. **Shape-parameterized or shape-agnostic arrays**
   - Function parameters like `u: [f64; *]` or `u: [f64; n]` where `n` is a parameter, so ODE state and optimizer state can be arbitrary size.
   - Stdlib already uses `[f32; *]` and `[f32; ?, ?]` in places; recurrence and shape inference must support these in the same way.

3. **Optional: first-class or typed “rhs” for ODEs**
   - In Julia, `solve(ode, u0, tspan)` works with `ode(u, p, t)` as a function. In Einlang we don’t have first-class function parameters.
   - Alternatives: (a) fixed set of built-in ODE forms (e.g. linear, decay) with parameters; (b) add a way to pass a function reference or a small DSL for `du/dt`; (c) keep only one-step steppers in stdlib and let the user write the loop (with their chosen `n_steps`) in their code.

## Recommended path

1. **Do not** move the current `examples/numerics/` code into stdlib as-is; it is intentionally small and fixed-size for demos.
2. **Do** treat a general numerics stdlib as a target: document it (this file), and track language/compiler work needed (variable recurrence bounds, shape-agnostic recurrence).
3. **Short term:** If we add anything to stdlib, limit it to **building blocks** that are already general:
   - Single-step operations that don’t hide recurrence: e.g. `euler_decay_step(u, k, dt)` (scalar) is fine; a general `euler_step(u, dt, rhs)` would require a way to pass `rhs` (see above).
   - Document in this file or in `docs/stdlib.md` that “full” solvers (multi-step, variable length) are planned once the language supports variable-bound recurrence and generic shapes in that context.
4. **Later:** Once variable recurrence and shapes are robust, add `std::numerics` (or per-domain modules) with APIs that take `n_steps`, `n_states`, shapes from parameters, etc., and no hardcoded sizes.

## References

- Julia: [DifferentialEquations.jl](https://docs.sciml.ai/DiffEqDocs/stable/), [Optim.jl](https://github.com/JuliaNLSolvers/Optim.jl), [QuantEcon.jl](https://quantecon.org/quantecon-jl/).
- Current Einlang demos: `examples/numerics/` (fixed-size patterns only).
