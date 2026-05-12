---
layout: book
title: "Chapter 12 · Comparison: Physics"
---

# Chapter 12 · Comparison: Physics

> "The map is not the territory—but when the map says 'temperature' and the territory says 'pressure,' you have a problem that no amount of correct math can fix."
>
> — The author

*Comparisons · Heat equations and field components in two notations*

---

Normalization and attention revealed the same pattern: when two dimensions share the same extent, positional code loses the ability to distinguish them. The Square Matrix Test from Chapter 3—`batch_size == num_classes`—returned in attention as `seq_q == seq_k`. In both cases, the positional notation was correct-by-coincidence. The names made the distinction checkable.

This pattern's oldest domain is physical simulation, which predates machine learning by decades. Fortran physicists have been writing `U(I+1, J)` since before the term "tensor" entered our vocabulary—and if you ask one of them what `state[:,:,2]` means, they will give you the correct answer. Then they will tell you about the bug they fixed in 1997 where `2` was actually `3` and the simulation ran for two weeks before anyone noticed. The integer field index—`state[:,:,2]` for velocity-x—is the original ghost in the name.

It asks: when coordinates represent temperature, pressure, and velocity fields, and confusing them means solving the wrong physics, what does each notation make visible?

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

But the Einlang version does more: it names the *physical coordinate* `i` and the *field coordinate* `field` separately. The coupling equations can reference them by name. A term that depends on temperature reads `state[t, i, field=0]`. A term that depends on the spatial gradient reads `state[t, i+1, field=0] - state[t, i-1, field=0]`. The code says which field and which spatial offset. This is the megaphone model at the level of physical quantities: `state` speaks on `t`, `i`, and `field`; operations that only care about `i` omit `t` and `field` from their brackets, and the omission is the claim that the stencil is spatial, not temporal or field-specific.

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

In the Einlang version:

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

The integer field index (`state[..., 2]`) is the weakest link in the positional chain. If you saw `state[..., 2]` in a Fortran simulation code—no comments, no documentation, just the integer index—could you be 100% certain it is velocity-x? If you answered yes, notice: is that certainty coming from the code, or from a convention you memorized?

A convention is a fact that lives outside the notation. It is correct until someone reorganizes the field order, or inserts a new field at index 0, or reuses the same integer for a different field in a different function. The convention drifts. The integer stays the same. The bug is not in the arithmetic—it is in the gap between what the integer means and what the code records.

In Einlang, `field=0` and `field=1` are names. They survive reorganization because the name `field=0` is tied to the coordinate value, not to its position. If a new field is inserted at `field=0`, the compiler flags the conflict—two fields cannot both be `field=0`. The integer `2` would silently become a different field. The name `field=0` would refuse.

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

In the Einlang version:

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

In Einlang, `field=2` is velocity-x because the coordinate value `2` is bound to the name `field`. If the field order changes, `field=2` still refers to the same physical quantity—or the compiler catches the inconsistency. The name is tied to the coordinate, not to its position. The stencil `i+1` is the right neighbor because `i` is the spatial coordinate and `+1` is the rightward offset. If the spatial dimension moves, `i` still means spatial—the name doesn't change.

---

## The Navier-Stokes Skeleton

Fluid dynamics is the grand challenge of computational physics. The Navier-Stokes equations couple velocity, pressure, and vorticity across three spatial dimensions and time. The codebase is typically hundreds of thousands of lines of Fortran or C++, with integer dimension indices scattered throughout. The most common bugs are coordinate swaps—confusing `x` for `y` velocity, or the `x` momentum equation for the `y` momentum equation.

Here is a simplified 2D Navier-Stokes time step in Einlang, using the same coordinate conventions from the heat equation and Burgers equation:

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

The Einlang version separates three concerns that the Fortran version merges:
1. **Field identity**: `u`, `v`, `p` are different tensors, not different array names pointing into the same multi-field state tensor.
2. **Coordinate identity**: `i` is the x-coordinate, `j` is the y-coordinate. The offsets `+1` and `-1` say which direction.
3. **Stencil structure**: the finite difference terms are grouped by physical meaning (diffusion, advection, pressure).

In Fortran, all three concerns are compressed into `U(I+1, J)`. The compression works. But it makes every stencil access look like every other stencil access. When they differ, only the reader's eye catches the difference.

---

## Comparison with JAX and Functional Physics

JAX's functional approach to physics simulation—pure functions, no mutation, explicit state passing—shares philosophical ground with Einlang. In JAX, a simulation step is:

```python
def step(state, dt):
    u, v, p = state['u'], state['v'], state['p']
    u_new = u + dt * (nu * laplacian(u) - advection(u, v) - grad_x(p))
    return {'u': u_new, 'v': v_new, 'p': p_new}
```

The fields are named dictionary keys. The stencils are in the function names (`laplacian`, `advection`, `grad_x`). The coordinate structure—which stencil operates on which axis—is inside each function, not at the call site. This is a design choice: encapsulate the stencil, expose the physics.

Einlang's contribution is a third axis of naming: not just the field (dictionary key) and the operation (function name), but the *coordinate*. `u[t-1, i+1, j]` names the field (`u`), the time recurrence (`t-1`), and the spatial offset (`i+1, j`). JAX names the field and the operation. Einlang adds the coordinate. When the coordinate is wrong—when `i+1` should be `j+1`—the name is visible at the point of error. In JAX, the error is inside `grad_x`, and the call site only knows it called `grad_x`, not which coordinate `grad_x` operates on.

If the magnetic field index moves from 4 to 0, how many lines of code do you need to change?

In JAX: every line that indexes `state[..., 4]` must be found and updated. In Einlang: zero. The coordinate name `Bx` stays `Bx` regardless of field order. The compiler maps the name to the new position. The question tests whether the coordinate identity lives in the code or in the convention.

---

## Two Notations, One Task

The three comparison chapters end here. Positional notation is concise, universal, and runs directly on every accelerator—when coordinates are genuinely anonymous, `dim=-1` is all you need. For a ReLU activation or an element-wise addition where no coordinate identity is at stake, the two notations cost the same keystrokes and the same thought. The named notation earns its keep where identities diverge: `class` vs `batch`, `seq_q` vs `seq_k`, `velocity-x` vs `pressure`. Named notation makes coordinate identities checkable—when `class` and `batch` have the same extent, only the name distinguishes them. The difference is not efficiency. It is whether the code records the facts that correctness depends on.

There is a subtler argument for positional notation, and it deserves to be stated plainly: **sometimes `dim=-1` is correct by construction, and that correctness is not an accident.** A softmax that normalizes over the last dimension will be correct for any tensor whose last dimension happens to be the class dimension—and in many codebases, that invariant is genuinely stable. The fully-connected layer `linear(x, W)` is insensitive to whether `x` is `(batch, feature)` or `(feature, batch)` as long as `W` is transposed accordingly. Positional notation's "ambiguity" can be a form of flexibility: the same function works on different layouts because it only cares about relative position, not absolute identity.

The coordinate habit does not deny this. It asks a narrower question: *when the operation depends on which coordinate is which, is that dependency recorded?* If your codebase has a stable convention that the last dimension is always the feature dimension, and that convention is enforced by assertions or code review, `dim=-1` is not a bug waiting to happen—it is a shorthand for a well-understood invariant. The problem is not `dim=-1` itself. The problem is `dim=-1` in a codebase where the invariant is undocumented, unenforced, and assumed.

Names are not a replacement for conventions. They are a way to make conventions checkable. The coordinate habit says: if a convention exists, record it. If it doesn't, the name is where you discover that.

If the magnetic field index moves from 4 to 0, how many lines of code do you need to change?

---

*If you saw `state[..., 2]` in a simulation code with no comments, could you be 100% certain it is velocity-x? If your answer is "yes, because of the convention"—ask yourself whether the convention survived the last refactoring. The integer stays the same. The meaning drifts. Only a name can anchor it.*

---

## The Coordinate Audit in Practice

Physical simulation code is dense with integer indices, each one a claim about identity. `state[..., 0]` claims "density lives at position 0." `u[i+1]` claims "the right neighbor lies one index ahead." These claims live in comments, enums, and variable names—never in the notation itself.

Pick `temperature`. Find every place it's read and written. If the code uses `state[..., 3]`, you must verify every `3` corresponds to temperature. If a new diagnostic field is inserted at position 2, every `3` must be re-evaluated: some should become `4`, others should stay. The refactoring requires manual verification of every integer index. No compiler helps.

In Einlang, `state[..., temp]` stays `temp` regardless of field order. The coordinate name is the anchor. The integer is the implementation detail. The difference is whether the anchor is in the code or in your head.
