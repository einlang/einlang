---
layout: book
title: "Chapter 11 · Comparison: Physics"
---

# Chapter 11 · Comparison: Physics

> "The map is not the territory—but when the map says 'temperature' and the territory says 'pressure,' you have a problem that no amount of correct math can fix."
>
> — The author

*Comparisons · Heat equations and field components in two notations*

---

You are simulating a two-dimensional thermodynamic system: temperature, pressure, velocity-x, velocity-y, evolving on a spatial grid over time. You store them in a tensor of shape `(T, N, 4)`. Integer `2` is velocity-x. Integer `3` is velocity-y. You're writing field-specific equations, slicing `[:,:,2]` and `[:,:,3]` throughout the code. It works.

A colleague adds a magnetic field. Instead of appending it at index 4, they insert it at index 0—"put the most important field first." Every `[:,:,2]` now silently reads pressure instead of velocity-x. Every `[:,:,3]` reads velocity-x instead of velocity-y. The shapes are correct. The code runs. The physics is wrong. The simulation produces results that look plausible—pressure gradients where velocity should be, velocity where magnetic flux should be—and you discover the error three weeks later, staring at a contour plot that shouldn't be symmetric.

The first two comparison chapters looked at machine learning primitives—normalization and attention. This chapter looks at something older: physical simulation. The heat equation, fluid dynamics, multi-field coupling. Computations where coordinates carry physical meaning—temperature, pressure, velocity—and confusing them means solving the wrong physics.

---

## The Heat Equation

The one-dimensional heat equation describes how temperature diffuses through a rod over time:

$$u_t = \alpha \cdot u_{xx}$$

In explicit Euler stepping, each point's new temperature is a weighted average of its neighbors:

$$u[t, i] = u[t-1, i] + \alpha \cdot (u[t-1, i+1] - 2 \cdot u[t-1, i] + u[t-1, i-1])$$

**NumPy:**

```python
def heat_diffusion(initial, alpha, T):
    N = len(initial)
    u = np.zeros((T, N))
    u[0] = initial
    for t in range(1, T):
        u[t, 1:-1] = u[t-1, 1:-1] + alpha * (
            u[t-1, 2:] - 2 * u[t-1, 1:-1] + u[t-1, :-2])
    return u
```

The code works. But the Laplacian—the discrete second derivative—is spread across three slice expressions: `u[t-1, 2:]`, `u[t-1, 1:-1]`, and `u[t-1, :-2]`. The relationship between them ("these three terms form a stencil over the spatial coordinate") is invisible. If you swap `alpha` from 0.1 to 0.5 (violating the CFL condition), the code still runs—it just produces physically impossible results.

**Einlang:**

```
let u[t in 0..T, i] = initial[i];
let u[t in 1..T, i] = u[t-1, i] + alpha * (
    u[t-1, i+1] - 2.0 * u[t-1, i] + u[t-1, i-1]
);
```

The Laplacian is a single expression: `u[t-1, i+1] - 2*u[t-1, i] + u[t-1, i-1]`. The index arithmetic `i+1` and `i-1` makes the stencil visible. `i` is the spatial coordinate, and the offsets `+1` and `-1` are relative to it. The declaration bracket says `t in 1..T, i`—time runs from 1 to T-1, space runs over the whole domain. The recurrence is a fact about the coordinate `t`, stated in the bracket.

---

## Multi-Field Coupling

Real simulations track multiple physical fields—temperature, pressure, velocity components—coupled through partial differential equations. In a positional array, these fields are stored along an integer axis:

```python
# state shape: (T, N, 4)
# state[..., 0] = temperature
# state[..., 1] = pressure
# state[..., 2] = velocity_x
# state[..., 3] = velocity_y

def coupled_step(state, t, alpha, beta):
    temp = state[t, :, 0]
    press = state[t, :, 1]
    vx = state[t, :, 2]
    vy = state[t, :, 3]
    # ... coupled equations ...
```

`state[t, :, 0]` extracts temperature. `state[t, :, 1]` extracts pressure. The mapping from integer to physical quantity is in the comments. If a new field is added—say, humidity—it becomes `state[..., 4]`. If the order changes—temperature moves from index 0 to index 2—every `[:, 0]` silently becomes wrong. The code runs. The numbers change. No error is raised.

**Einlang:**

```
let state[t in 0..T, i, field] = init_field(field, i);

let temp[t, i] = state[t, i, field=0];
let press[t, i] = state[t, i, field=1];
let vx[t, i] = state[t, i, field=2];
let vy[t, i] = state[t, i, field=3];
```

`field` is a coordinate. Its values are named: `field=0` is temperature, `field=1` is pressure. If humidity is added, it becomes `field=4`—a new coordinate value, not a new integer to remember. If the field order changes, the name `field=0` still means temperature, regardless of where it sits in the array.

But the einlang version does more: it names the *physical coordinate* `i` and the *field coordinate* `field` separately. The coupling equations can reference them by name. A term that depends on temperature reads `state[t, i, field=0]`. A term that depends on the spatial gradient reads `state[t, i+1, field=0] - state[t, i-1, field=0]`. The code says which field and which spatial offset.

---

## Adding a New Field

Suppose the simulation is extended to include humidity. In the positional version:

```python
# Before: state shape (T, N, 4)
# After:  state shape (T, N, 5)
# Every [..., 0:4] slice must be audited.
# Every equation that referenced field indices must be checked.
state = np.zeros((T, N, 5))
temp = state[:, :, 0]      # unchanged — luckily
press = state[:, :, 1]     # unchanged — luckily
vx = state[:, :, 2]        # unchanged — luckily
vy = state[:, :, 3]        # unchanged — luckily
humidity = state[:, :, 4]  # new
```

Every integer index must be verified. The compiler provides no help. If humidity was inserted at index 0 instead of appended at index 4, every subsequent index shifts by one.

In the einlang version:

```
let state[t in 0..T, i, field] = init_field(field, i);

let temp[t, i] = state[t, i, field=0];
let press[t, i] = state[t, i, field=1];
let vx[t, i] = state[t, i, field=2];
let vy[t, i] = state[t, i, field=3];
let humidity[t, i] = state[t, i, field=4];  // new line
```

The existing field assignments are unchanged. `field=0` is still temperature, regardless of whether humidity is `field=4` or `field=0` with everything else shifted. The coordinate names are stable under insertions because they are names, not positions.

---

## The Coupled Burgers Equation

The 1D coupled Burgers equation for velocity `v` and temperature `T`:

$$v_t + v \cdot v_x = \nu \cdot v_{xx} + \beta \cdot T_x$$

Each term has a specific coordinate interpretation: `v_t` is the time derivative (difference along `t`), `v_x` is the spatial derivative (difference along `i`), `v_{xx}` is the second spatial derivative, and `T_x` is the temperature gradient driving the velocity.

**NumPy:**

```python
for t in range(1, T):
    v_xx = (v[t-1, 2:] - 2*v[t-1, 1:-1] + v[t-1, :-2]) / dx**2
    v_x = (v[t-1, 2:] - v[t-1, :-2]) / (2*dx)
    T_x = (T[t-1, 2:] - T[t-1, :-2]) / (2*dx)
    v[t, 1:-1] = (v[t-1, 1:-1]
                  + dt * (nu * v_xx
                          - v[t-1, 1:-1] * v_x
                          + beta * T_x))
```

The field identity (`v` vs `T`) is in variable names. The coordinate identity (`t` vs `i`) is in bracket positions. The stencil structure is in the slicing patterns.

**Einlang:**

```
let v[t in 1..T, i] = v[t-1, i]
    + dt * (nu * (v[t-1, i+1] - 2.0*v[t-1, i] + v[t-1, i-1]) / (dx**2)
            - v[t-1, i] * (v[t-1, i+1] - v[t-1, i-1]) / (2.0*dx)
            + beta * (T[t-1, i+1] - T[t-1, i-1]) / (2.0*dx));
```

The terms are identifiable by their coordinate arithmetic: `i+1` and `i-1` are spatial derivatives. `t-1` is the time recurrence. `v[...]` and `T[...]` are different fields, named as different tensors. The equation reads like the PDE it discretizes.

---

## What the Comparison Reveals

Physical simulation exposes a different failure mode than machine learning. In ML, wrong coordinates produce wrong gradients and degraded performance. In physics, wrong coordinates produce physically impossible results—negative temperatures, violated conservation laws—that may look plausible at a glance.

The integer field index (`state[..., 2]`) is the weakest link in the positional chain. Stop and test yourself. If you saw `state[..., 2]` in a Fortran simulation code—no comments, no documentation, just the integer index—could you be 100% certain it is velocity-x? If you answered yes, ask yourself: is that certainty coming from the code, or from a convention you memorized?

A convention is a fact that lives outside the notation. It is correct until someone reorganizes the field order, or inserts a new field at index 0, or reuses the same integer for a different field in a different function. The convention drifts. The integer stays the same. The bug is not in the arithmetic—it is in the gap between what the integer means and what the code records.

In einlang, `field=0` and `field=1` are names. They survive reorganization because the name `field=0` is tied to the coordinate value, not to its position. If a new field is inserted at `field=0`, the compiler flags the conflict—two fields cannot both be `field=0`. The integer `2` would silently become a different field. The name `field=0` would refuse.

---

## Adding a Magnetic Field

Now extend the simulation with a magnetic field `B` as a fifth physical quantity. In the positional version, this means changing the state shape from `(T, N, 4)` to `(T, N, 5)` and auditing every occurrence of the integer `4` and every slice that assumed four fields:

```python
# Before: state shape (T, N, 4)
# After:  state shape (T, N, 5)
# Must audit: every [..., 0:4], every [..., 3], every hardcoded "4"
B = state[:, :, 4]  # new — but is it really at index 4?
```

If `B` was inserted at index 0, every subsequent index shifts: `temp` moves from 0 to 1, `press` from 1 to 2, and so on. The compiler provides no help. The integer `state[..., 0]` does not know it was supposed to be temperature.

In the einlang version:

```
let state[t in 0..T, i, field] = init_field(field, i);

let temp[t, i] = state[t, i, field=0];
let press[t, i] = state[t, i, field=1];
let vx[t, i] = state[t, i, field=2];
let vy[t, i] = state[t, i, field=3];
let B[t, i] = state[t, i, field=4];     // new line — one change
```

The existing field assignments are unchanged. `field=0` is still temperature. If `B` were inserted at `field=0`, the compiler would flag every subsequent `field=N` reference as referring to a shifted domain—because each field value now maps to a different physical quantity. The coordinate name `field` is stable under insertions. The integer index `2` is not.

The cost difference is not in writing the initial code. It is in every modification made after the original author has forgotten which integer meant what.

---

## The Wave Equation: A Stencil in Two Notations

The 1D wave equation describes how a displacement propagates through a medium:

$$u_{tt} = c^2 \cdot u_{xx}$$

In explicit finite differences, it becomes a three-point stencil in space and a two-point stencil in time:

$$u[t, i] = 2 \cdot u[t-1, i] - u[t-2, i] + c^2 \cdot (u[t-1, i+1] - 2 \cdot u[t-1, i] + u[t-1, i-1])$$

**NumPy:**

```python
def wave_step(u, t, c):
    u[t, 1:-1] = (2 * u[t-1, 1:-1] - u[t-2, 1:-1]
                   + c**2 * (u[t-1, 2:] - 2 * u[t-1, 1:-1] + u[t-1, :-2]))
```

The time index `t` is in variable position `u[t, ...]`. The spatial stencil `i-1, i, i+1` is distributed across three slices: `u[t-1, 2:]`, `u[t-1, 1:-1]`, `u[t-1, :-2]`. The second derivative structure `(f[i+1] - 2*f[i] + f[i-1])` is visible only if you mentally align the three slices.

**Einlang:**

```
let u[t in 0..1, i] = initial[i] + dt * v_initial[i];
let u[t in 2..T, i] =
    2.0 * u[t-1, i] - u[t-2, i]
    + c**2 * (u[t-1, i+1] - 2.0 * u[t-1, i] + u[t-1, i-1]);
```

The Laplacian stencil is a single expression: `u[t-1, i+1] - 2*u[t-1, i] + u[t-1, i-1]`. The index arithmetic names the stencil offsets: `i+1` is the right neighbor, `i-1` is the left neighbor. The time recurrence names `t-1` (one step back) and `t-2` (two steps back). If someone accidentally writes `i+2` instead of `i+1`, the stencil would be wrong—but the error is a single character in a named expression, not a misaligned slice that the reader must reconstruct.

---

## The Inventory: What the Physics Chapter Found

Physical simulation exposes a failure mode distinct from machine learning. In ML, wrong coordinates degrade performance—the loss is worse, the BLEU score drops. In physics, wrong coordinates produce physically impossible results: negative absolute temperatures, violated conservation laws, waves that amplify instead of propagating. The symptoms may look plausible at a glance—the contour plot has the right general shape, the time series has the right range. Only a physicist's eye catches them.

The integer field index (`state[..., 2]`) and the positional stencil slices (`u[t-1, 2:]`) share the same root cause: **the mapping from integer to meaning lives outside the notation.** The integer `2` is velocity-x because the comment says so. The slice `2:` is the right neighbor because the reshape put it there. When the mapping drifts—a new field inserted, a dimension reordered—the integer stays the same and the meaning changes.

In einlang, `field=2` is velocity-x because the coordinate value `2` is bound to the name `field`. If the field order changes, `field=2` still refers to the same physical quantity—or the compiler catches the inconsistency. The name is tied to the coordinate, not to its position. The stencil `i+1` is the right neighbor because `i` is the spatial coordinate and `+1` is the rightward offset. If the spatial dimension moves, `i` still means spatial—the name doesn't change.

---

## The Navier-Stokes Skeleton

Fluid dynamics is the grand challenge of computational physics. The Navier-Stokes equations couple velocity, pressure, and vorticity across three spatial dimensions and time. The codebase is typically hundreds of thousands of lines of Fortran or C++, with integer dimension indices scattered throughout. The most common bugs are coordinate swaps—confusing `x` for `y` velocity, or the `x` momentum equation for the `y` momentum equation.

Here is a simplified 2D Navier-Stokes time step in einlang, using the same coordinate conventions from the heat equation and Burgers equation:

```
let u[t in 1..T, i, j] = u[t-1, i, j]
    + dt * (nu * (u[t-1, i+1, j] - 2.0*u[t-1, i, j] + u[t-1, i-1, j]) / dx**2
           + nu * (u[t-1, i, j+1] - 2.0*u[t-1, i, j] + u[t-1, i, j-1]) / dy**2
           - u[t-1, i, j] * (u[t-1, i+1, j] - u[t-1, i-1, j]) / (2.0*dx)
           - v[t-1, i, j] * (u[t-1, i, j+1] - u[t-1, i, j-1]) / (2.0*dy)
           - (p[t-1, i+1, j] - p[t-1, i-1, j]) / (2.0*dx));
```

The terms are recognizable: the first two lines are the viscous diffusion (Laplacian in `i` and `j`), the third line is the advection (velocity convecting itself), the fourth line is the pressure gradient. Each term names its coordinates and offsets. `i+1` and `i-1` are always the x-differences. `j+1` and `j-1` are always the y-differences. The fields `u`, `v`, `p` are separate tensors with separate names.

In the positional Fortran/C++ version, the same code uses array indices like `U(I+1, J)`, `U(I, J+1)`, `P(I+1, J)`—the coordinate names `i` and `j` are loop variables, not part of the tensor structure. The field identity is in the variable name (`U`, `V`, `P`). The stencil is distributed across multiple array access expressions. If an index is typed wrong—`U(I, J+1)` where `U(I+1, J)` was intended—the compiler cannot catch it because both are valid array accesses. The bug survives compilation and produces physically plausible but incorrect results.

The einlang version separates three concerns that the Fortran version merges:
1. **Field identity**: `u`, `v`, `p` are different tensors, not different array names pointing into the same multi-field state tensor.
2. **Coordinate identity**: `i` is the x-coordinate, `j` is the y-coordinate. The offsets `+1` and `-1` say which direction.
3. **Stencil structure**: the finite difference terms are grouped by physical meaning (diffusion, advection, pressure).

In Fortran, all three concerns are compressed into `U(I+1, J)`. The compression works. But it makes every stencil access look like every other stencil access. When they differ, only the reader's eye catches the difference.

---

## Comparison with JAX and Functional Physics

JAX's functional approach to physics simulation—pure functions, no mutation, explicit state passing—shares philosophical ground with einlang. In JAX, a simulation step is:

```python
def step(state, dt):
    u, v, p = state['u'], state['v'], state['p']
    u_new = u + dt * (nu * laplacian(u) - advection(u, v) - grad_x(p))
    return {'u': u_new, 'v': v_new, 'p': p_new}
```

The fields are named dictionary keys. The stencils are in the function names (`laplacian`, `advection`, `grad_x`). The coordinate structure—which stencil operates on which axis—is inside each function, not at the call site. This is a design choice: encapsulate the stencil, expose the physics.

Einlang's contribution is a third axis of naming: not just the field (dictionary key) and the operation (function name), but the *coordinate*. `u[t-1, i+1, j]` names the field (`u`), the time recurrence (`t-1`), and the spatial offset (`i+1, j`). JAX names the field and the operation. Einlang adds the coordinate. When the coordinate is wrong—when `i+1` should be `j+1`—the name is visible at the point of error. In JAX, the error is inside `grad_x`, and the call site only knows it called `grad_x`, not which coordinate `grad_x` operates on.

The three comparison chapters—normalization, attention, physics—converge on the same finding. Each notation makes different facts visible. Positional notation makes shapes visible. JAX's functional notation makes data dependencies visible. Einlang's coordinate notation makes coordinate identities visible. None is strictly better. But when the bug is a coordinate identity error—and a surprising fraction of tensor bugs are—the notation that records identities is the notation that catches the error.

---

The three comparison chapters—normalization, attention, physics—have shown a consistent pattern. Positional notation is not incorrect. It is *underspecified*. It records shapes. It does not record identities. When identities matter—and they always matter—the difference between a correct program and a wrong one is a name that was never written down.


---

*If you saw `state[..., 2]` in a simulation code with no comments, could you be 100% certain it is velocity-x? If your answer is "yes, because of the convention"—ask yourself whether the convention survived the last refactoring. The integer stays the same. The meaning drifts. Only a name can anchor it.*

Now we enter the book's most critical section. We stop comparing notations and start building: the engine that gives names their power. How does a compiler represent, analyze, and lower a program written in named coordinates? You are about to see what einlang looks like from the inside—and build the machinery that makes the first ten chapters checkable.
