---
layout: book
title: "Chapter 13 · Comparison: Physics"
---

# Chapter 13 · Comparison: Physics

> "The first principle is that you must not fool yourself — and you are the easiest person to fool."
>
> — Richard Feynman

*Comparisons · Heat equations and field components in two notations*

---

Chapters 11 and 12 showed the pattern in machine learning: normalization proved it holds, attention proved it reveals. This chapter asks the final question: **what does the pattern prevent?**

The domain is physical simulation, which predates machine learning by decades. Fortran physicists have been writing `U(I+1, J)` since before the term "tensor" entered our vocabulary—and if you ask one of them what `state[:,:,2]` means, they will give you the correct answer. Then they will tell you about the bug they fixed in 1997 where `2` was actually `3` and the simulation ran for two weeks before anyone noticed. The integer field index—`state[:,:,2]` for velocity-x—is the original ghost in the name.

The stakes are higher here. In ML, a coordinate swap degrades a metric. In physics, a coordinate swap produces negative absolute temperatures, violated conservation laws, waves that amplify instead of propagating. The results look plausible—the contour plot has the right shape, the time series has the right range. Only a physicist's eye catches them. The question is not *does the code run?* It is *does the code solve the right equations?*

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

```rust
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

```rust
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

```rust
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

```rust
let v[t in 1..T, i] = v[t-1, i]
    + dt * (nu * (v[t-1, i+1] - 2.0*v[t-1, i] + v[t-1, i-1]) / (dx**2)
            - v[t-1, i] * (v[t-1, i+1] - v[t-1, i-1]) / (2.0*dx)
            + beta * (T[t-1, i+1] - T[t-1, i-1]) / (2.0*dx));
```

The terms are identifiable by their coordinate arithmetic: `i+1` and `i-1` are spatial derivatives. `t-1` is the time recurrence. `v[...]` and `T[...]` are different fields, named as different tensors. The equation reads like the PDE it discretizes.

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

```rust
let u[t in 0..1, i] = initial[i] + dt * v_initial[i];
let u[t in 2..T, i] =
    2.0 * u[t-1, i] - u[t-2, i]
    + c**2 * (u[t-1, i+1] - 2.0 * u[t-1, i] + u[t-1, i-1]);
```

The Laplacian stencil is a single expression: `u[t-1, i+1] - 2*u[t-1, i] + u[t-1, i-1]`. The index arithmetic names the stencil offsets: `i+1` is the right neighbor, `i-1` is the left neighbor. The time recurrence names `t-1` (one step back) and `t-2` (two steps back). If someone accidentally writes `i+2` instead of `i+1`, the stencil would be wrong—but the error is a single character in a named expression, not a misaligned slice that the reader must reconstruct.

---

## The Navier-Stokes Skeleton

Fluid dynamics is the grand challenge of computational physics. The Navier-Stokes equations couple velocity, pressure, and vorticity across three spatial dimensions and time. The codebase is typically hundreds of thousands of lines of Fortran or C++, with integer dimension indices scattered throughout. The most common bugs are coordinate swaps—confusing `x` for `y` velocity, or the `x` momentum equation for the `y` momentum equation.

Here is a simplified 2D Navier-Stokes time step in Einlang, using the same coordinate conventions from the heat equation and Burgers equation:

```rust
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

## The Inventory

Three chapters, three domains, one finding.

Normalization showed the pattern holds. Four variants, one skeleton. The coordinate name absorbs layout changes that silently corrupt a positional `dim=`. LayerNorm with `dim=-1` broke when `feature` moved. `mean[feature]` didn't.

Attention showed what the pattern reveals. Self-attention and cross-attention—different semantics, different gradient flows, different architectural implications—are the same Python function. The distinction lives in runtime shapes, not in source code. Named coordinates made the invisible visible: `seq` shared vs `seq_q`/`seq_k` separate.

Physics showed what the pattern prevents. The bugs here are older than machine learning—integer field indices silently swapping since Fortran, stencil slices misaligned since the first finite difference code. The symptoms look plausible. Negative temperatures that still form contour plots. Waves that amplify but the time series has the right range. Only a physicist's eye catches them.

In every domain, the root cause is the same: **the mapping from integer to meaning lives outside the notation.** `dim=-1` is `feature` because the layout convention says so. `state[..., 2]` is velocity-x because the comment says so. `u[t-1, 2:]` is the right neighbor because the reshape put it there. When the layout changes, the meaning drifts. The integer does not change. Only the meaning does.

In Einlang, the coordinate name is the anchor. `mean[feature]` stays `mean[feature]` regardless of layout. `field=2` stays velocity-x regardless of field order. `i+1` stays the right neighbor regardless of which position `i` maps to. The name is tied to the coordinate, not to its position. The integer is the implementation detail.

---

Here is a PDE stencil. Before reading the commentary, find the coordinate that carries recurrence:

```rust
let u[t, i, j] = u[t-1, i, j]
    + c * (u[t-1, i+1, j] - 2.0 * u[t-1, i, j] + u[t-1, i-1, j])
    + c * (u[t-1, i, j+1] - 2.0 * u[t-1, i, j] + u[t-1, i, j-1]);
```

`t` only appears as `t-1` on the right — never as `t+1`. The recurrence arrow constrains `t`. The compiler enforces that `t+1` cannot appear on the right-hand side of a definition for `t`. This expression passes.

Now imagine a colleague accidentally swaps `i` and `j` in the second line — writes `u[t-1, j+1, i]` instead of `u[t-1, i, j+1]`. In a positional `u[t-1, :, :]`, the swap is invisible — the code runs, produces numbers, the contour plot looks right. But the x-derivative and y-derivative have been exchanged. The physics is wrong. In the named version, `i` and `j` are different names. `u[t-1, j+1, i]` puts a `j+1` expression where `i` is declared — the coordinate mismatch is in the source. The stencil doesn't silently swap. The names won't let it.

---

## Two Notations, One Task

The three comparison chapters end here. A fair assessment requires stating what positional notation does well.

Positional notation is concise, universal, and runs directly on every accelerator. When coordinates are genuinely anonymous—a ReLU activation, an element-wise addition—the two notations cost the same keystrokes and the same thought. Named notation earns its keep where identities diverge: `class` vs `batch`, `seq_q` vs `seq_k`, `velocity-x` vs `pressure`.

There is a subtler argument for positional notation: **sometimes `dim=-1` is correct by construction.** A softmax that normalizes over the last dimension will be correct for any tensor whose last dimension happens to be the correct one—and in many codebases, that invariant is genuinely stable. Positional notation's "ambiguity" can be a form of flexibility: the same function works on different layouts because it only cares about relative position, not absolute identity.

The coordinate habit does not deny this. It asks a narrower question: *when the operation depends on which coordinate is which, is that dependency recorded?* If your codebase enforces the convention that the last dimension is always `feature`, `dim=-1` is a shorthand for a well-understood invariant, not a bug waiting to happen. The problem is `dim=-1` in a codebase where the invariant is undocumented, unenforced, and assumed.

Names are not a replacement for conventions. They are a way to make conventions checkable. The coordinate habit says: if a convention exists, record it. If it doesn't, the name is where you discover that.

If the magnetic field index moves from 4 to 0, how many lines of code do you need to change?

The comparison chapters are not an argument that positional notation is bad. They are a demonstration that positional notation is incomplete. The integer records a position. The name records an identity. Both are facts. Only one is in the source code.

Now the question the book has been circling since Chapter 1: if names are so useful, what can they NOT do? Every tool has a boundary.

---

Chapter 14 asks the necessary question: what can names *not* check? Every system has a boundary. The boundary is not a flaw. It is a map. And a good map tells you where the boundaries are.
