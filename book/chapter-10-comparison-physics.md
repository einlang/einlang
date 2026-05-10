---
layout: book
title: "Chapter 10 · Comparison: Physics"
---

# Chapter 10 · Comparison: Physics

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

The integer field index (`state[..., 2]`) is the weakest link in the positional chain. It carries no semantic information. It is correct only by convention. When the convention changes—a new field is added, the order is reorganized—the code breaks silently. In einlang, `field=0` and `field=1` are names. They survive reorganization.

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

The three comparison chapters—normalization, attention, physics—have shown a consistent pattern. Positional notation is not incorrect. It is *underspecified*. It records shapes. It does not record identities. When identities matter—and they always matter—the difference between a correct program and a wrong one is a name that was never written down.

Now we enter the book's most critical section. We stop comparing notations and start building: the engine that gives names their power. How does a compiler represent, analyze, and lower a program written in named coordinates? You are about to see what einlang looks like from the inside—and build the machinery that makes the first ten chapters checkable.
