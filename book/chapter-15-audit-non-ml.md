---
layout: book
title: "Chapter 15 · The Simulation That Looked Right"
---

# Chapter 15 · The Simulation That Looked Right

> "The great enemy of knowledge is not error, but the illusion of knowledge."
>
> — Daniel J. Boorstin

*Diagnosis · A physicist audits a heat-equation simulation with the four habits*

---

Late on a Thursday, a physicist you know—call her Mira—stares at two plots on her screen. Both show the temperature field of a 2D plate after a thousand time steps. One was computed by a Fortran reference she trusts. The other came from a new einlang implementation her group wrote to prototype coupled physics on GPUs.

The contours match. The color scale matches. The maximum temperature—1.03 in the Fortran, 1.04 in the einlang version—is close enough to attribute to floating-point differences.

But the total energy is drifting. After 500 steps the drift is 0.3%. After 1000 steps it is 1.1%. This is not floating-point noise. This is a bug.

Mira opens the source. It is 75 lines. Every tensor in it has the correct shape. Her shape checker, which she trusts, reports no mismatches. She has been staring at this code for two hours.

"Walk me through it," you say. "But this time, don't check the shapes. Check the names."

---

## The Code, Plainly Stated

The simulation evolves four physical fields—temperature, pressure, velocity-x, velocity-y—on a 2D spatial grid, across time, for a batch of independent initial conditions. Here is the code Mira has been reading:

```rust
// Domain setup
let grid[height, width] = init_grid(height, width);
let fields[field] = [0.0, 1.0, 0.0, 0.0];   // T, P, Vx, Vy initial

// Initial condition: temperature spike at center
let init_temp[b, h, w, field] = if h == height/2 && w == width/2
    { if field == 0 { 100.0 } else { fields[field] } }
    else { fields[field] };

// Main simulation recurrence
let state[0, b, h, w, f] = init_temp[b, h, w, f];

// Laplacian operator for diffusion
let laplacian[t, b, h, w, f] =
    state[t, b, h-1, w, f] +
    state[t, b, h+1, w, f] +
    state[t, b, h, w-1, f] +
    state[t, b, h, w+1, f] -
    4.0 * state[t, b, h, w, f];

// Coupling: temperature affects pressure
let coupling[t, b, h, w] =
    state[t, b, h, w, 0] * thermal_expansion;

// State update
let state[t in 1..T, b, h, w, f] =
    if f == 0 {
        // Temperature: diffusion + source
        state[t-1, b, h, w, f] + alpha * laplacian[t-1, b, h, w, f]
    } else if f == 1 {
        // Pressure: diffusion + coupling from temperature
        state[t-1, b, h, w, f] + beta * laplacian[t-1, b, h, w, f]
            + gamma * coupling[t-1, b, h, w]
    } else if f == 2 {
        // Velocity-x: advection
        state[t-1, b, h, w, f] -
            state[t-1, b, h, w, 2] *
            (state[t-1, b, h+1, w, 2] - state[t-1, b, h-1, w, 2]) / (2.0 * dx)
    } else {
        // Velocity-y: advection
        state[t-1, b, h, w, f] -
            state[t-1, b, h, w, 3] *
            (state[t-1, b, h, w+1, 3] - state[t-1, b, h, w-1, 3]) / (2.0 * dy)
    };

// Diagnostics
let avg_temp[t] = mean[b, h, w](state[t, b, h, w, 0]);
let total_energy[t] = sum[b, h, w, f](state[t, b, h, w, f]);
```

"It runs," Mira says. "Every shape matches. Nobody on the team can find the bug."

You point to the first reduction. "Let's start there. Not with what the code does. With what it claims."

---

## First Habit: Eliminate with a Name

You read the diagnostic line aloud:

```rust
let avg_temp[t] = mean[b, h, w](state[t, b, h, w, 0]);
```

"What does this produce?" you ask.

"Average temperature at each time step," Mira says.

"One number per time step?"

"Yes. The mean collapses over batch, height, and width."

"Then what is the coordinate of the result?"

Mira looks at the declaration. The result has coordinate `[t]`. One number per time step. "That's correct," she says, but she says it slowly, because she has just noticed something.

"Three days ago," she says, "I added a second batch element—a different initial condition for comparison. The average was supposed to be per-batch, per-time. I never updated this line."

She types the correction:

```rust
let avg_temp[t, b] = mean[h, w](state[t, b, h, w, 0]);
```

The reduction now explicitly consumes `h` and `w`. The coordinate `b` is absent from the reduction bracket, so it survives. The declaration says `avg_temp[t, b]`—two coordinates, not one.

This is the first habit applied: **which coordinate does this operation eliminate?** The original code said `mean[b, h, w]`—it claimed to eliminate batch, height, and width. If the intent was to preserve batch, the reduction bracket was wrong. The shape was correct either way—a 1D result `[t]` and a 2D result `[t, b]` are both legal. But the *meaning* was wrong, and a shape checker could not tell.

One bug found. Four remain.

---

## Second Habit: Copy with a Signature

You move your finger to the coupling term:

```rust
let coupling[t, b, h, w] =
    state[t, b, h, w, 0] * thermal_expansion;
```

"Four coordinates," you say. "Now look at where it's used."

```rust
} else if f == 1 {
    // Pressure: diffusion + coupling from temperature
    state[t-1, b, h, w, f] + beta * laplacian[t-1, b, h, w, f]
        + gamma * coupling[t-1, b, h, w]
```

"`coupling` has no `f` coordinate," you say. "`state[... f]` has one. What happens when you add them?"

Mira sees it immediately. "The coupling term is broadcast over `f`."

"Is that what you want?"

She shakes her head. "The coupling is only supposed to affect pressure—field 1. That's why we have the `f == 1` guard. But `coupling` doesn't have a `f` coordinate at all, so it broadcasts over *all* fields. The guard `f == 1` means we only *add* it to the pressure equation. But when the compiler lays out memory—when it sees `coupling[t, b, h, w]` added to `state[t, b, h, w, f]`—it must decide whether `f` is being independently replicated."

She pauses. "In this case, the `f == 1` branch protects against wrong *values*. But if someone later uses `coupling` in a reduction over `f`, the missing coordinate will silently broadcast across all fields. The intent—'this only matters for pressure'—is invisible in the coordinate structure."

She writes the correction:

```rust
let coupling[t, b, h, w, field] = if field == 1
    { state[t, b, h, w, 0] * thermal_expansion }
    else { 0.0 };
```

Now `coupling` carries the `field` coordinate explicitly. For pressure it carries the coupling value. For all other fields it carries zero. The broadcast is no longer implicit. The coordinate structure says what the physics says: coupling is a per-field quantity, and only one field is affected.

This is the second habit: **which coordinate does this operation copy along?** When tensors of different ranks interact, something is being broadcast. The question is whether the broadcast is justified by the physics, and whether the code makes it visible.

---

## Third Habit: Permute with a Source

"Let me look at the advection terms more carefully," you say. You point to the velocity-x branch:

```rust
} else if f == 2 {
    // Velocity-x: advection
    state[t-1, b, h, w, f] -
        state[t-1, b, h, w, 2] *
        (state[t-1, b, h+1, w, 2] - state[t-1, b, h-1, w, 2]) / (2.0 * dx)
```

"Field 2 is velocity-x," Mira says. "The advection of velocity-x by itself. That's nonlinear self-advection. It's physically correct—the Navier-Stokes equations have a `u · ∇u` term."

"Agreed. Now read the velocity-y branch."

```rust
} else {
    // Velocity-y: advection
    state[t-1, b, h, w, f] -
        state[t-1, b, h, w, 3] *
        (state[t-1, b, h, w+1, 3] - state[t-1, b, h, w-1, 3]) / (2.0 * dy)
```

Mira reads it. Then reads it again.

"The derivative is along `w`," she says. "That's width. But velocity-y is supposed to be advected along height. The derivative should be `state[t-1, b, h+1, w, 3] - state[t-1, b, h-1, w, 3]`."

She traces her finger across the indices. "The first factor uses `state[t-1, b, h, w, 3]`—velocity-y at the current position. The derivative uses `state[t-1, b, h, w+1, 3] - state[t-1, b, h, w-1, 3]`—the difference along width. But velocity-y's advection should involve the gradient along height. Someone copied the velocity-x branch and changed the field index from 2 to 3 but forgot to change the spatial direction."

"This is why the total energy drifted," she says. "The advection was moving velocity-y in the wrong direction. The magnitude was right. The shape was right. The physics was wrong."

She writes the correction:

```rust
} else {
    // Velocity-y: advection along height
    state[t-1, b, h, w, f] -
        state[t-1, b, h, w, 3] *
        (state[t-1, b, h+1, w, 3] - state[t-1, b, h-1, w, 3]) / (2.0 * dy)
```

This is the third habit in action: **where did this coordinate come from, and where is it going?** The integer literal `2` hides an identity. When you copy and paste `2` to `3` in three places, one of the three might be the wrong change—if you are not tracing the coordinate through the expression. The coordinate names `h` and `w` were right there in the code. The field literals `2` and `3` were indistinguishable tokens. The bug survived because the human brain processes repeated integers as "the same number" and spatial indices as "just indices"—the notation gave no signal that `w+1` was the wrong dimension for velocity-y.

---

## Fourth Habit: The Boundaries

"One more," you say. "The Laplacian."

```rust
let laplacian[t, b, h, w, f] =
    state[t, b, h-1, w, f] +
    state[t, b, h+1, w, f] +
    state[t, b, h, w-1, f] +
    state[t, b, h, w+1, f] -
    4.0 * state[t, b, h, w, f];
```

Mira looks at the domain. "The recurrence defines `state[t in 1..T, b, h, w, f]`. But the Laplacian is used inside that recurrence, and it references `h-1` and `h+1` at every point. At the boundaries—`h == 0` and `h == height-1`—those references go out of bounds."

"What happens?" you ask.

"In most tensor frameworks, out-of-bounds indices either wrap around or read uninitialized memory. Either way, the boundary cells see values from the wrong side of the grid. The shape is correct—the Laplacian has the same shape as the state. But the boundary physics is wrong, and after enough time steps, the boundary errors propagate inward."

She writes the correction:

```rust
let laplacian[t, b, h, w, f] =
    state[t, b, h-1, w, f] +
    state[t, b, h+1, w, f] +
    state[t, b, h, w-1, f] +
    state[t, b, h, w+1, f] -
    4.0 * state[t, b, h, w, f]
    where h > 0 && h < height-1 && w > 0 && w < width-1;
```

With the `where` clause, the Laplacian is only defined for interior cells. The compiler now knows that boundary cells need a separate definition—perhaps a Dirichlet condition (fixed temperature) or a Neumann condition (zero flux). The boundary condition is no longer an accident of array indexing. It is a deliberate choice, visible in the source.

This is the fourth habit—**forward and backward, symmetric**—applied to spatial boundaries rather than gradient flow. The forward computation has a coordinate domain. When operations step outside that domain, the missing guard is a missing coordinate fact. The `where` clause restores it.

---

## The Fifth Error: Integer Literals as Coordinate Names

You lean back. "There's one more. It's not a bug in the execution. It's a bug in the reading."

You point to the lines that appear throughout the code: `field == 0`, `field == 1`, `field == 2`, `field == 3`. And the corresponding array accesses: `state[... 0]`, `state[... 2]`, `state[... 3]`.

"What is field 2?" you ask.

"Velocity-x."

"What is field 3?"

"Velocity-y."

"How many times did you have to look that up while debugging?"

Mira laughs. "Every time."

"The integers are correct. They produce the right indices. But they carry no meaning. Compare them to the spatial coordinates `h` and `w`—when you see `h`, you know it's height. When you see `w`, you know it's width. When you see `2`, you have to perform a mental lookup: '2 equals velocity-x.' Every. Single. Time."

She nods slowly. "And the copy-paste error—the derivative along `w` instead of `h`—was easier to make because `2` and `3` look the same to the eye. If the field names were `T`, `P`, `Vx`, `Vy`, the asymmetry would have jumped out."

You smile. "The four habits don't require named field indices. Einlang doesn't have an enum syntax for coordinates. But the habit—**eliminate with a name**—applies here too. The coordinate `field` has four positions. Each position has an identity. If the identity matters to the physics, the identity belongs in the code."

Mira opens a new file and begins to refactor. The field integers become named constants. The coordinate `field` keeps its name in every tensor. The physics and the notation realign.

---

## What the Four Habits Found

Mira leans back. "Five errors. Not one of them would have been caught by a shape checker."

She counts them on her fingers:

1. **The diagnostic mean** consumed `b` when it should have preserved it. The shape was legal either way. The reduction bracket told the truth about what was consumed—but the truth was wrong.

2. **The coupling term** was missing the `field` coordinate, broadcasting silently across all fields. The broadcast was harmless inside the `f == 1` guard but the *fact* of the broadcast was invisible.

3. **The velocity-y advection** used the spatial derivative along `w` instead of `h`. A copy-paste of integer literals where coordinate names would have made the asymmetry visible.

4. **The Laplacian at boundaries** read out-of-bounds indices without a `where` guard. The boundary condition—a physical fact—was an accident of array indexing rather than a deliberate statement in the source.

5. **The integer field indices** (`0, 1, 2, 3`) hid the identities of the physical quantities. The code ran correctly only as long as the reader remembered which integer meant what.

"These are not exotic bugs," she says. "These are ordinary, everyday bugs that happen when the notation cannot record the facts that matter."

You nod. "That is the whole argument of the book you just lived through. The coordinate habit is not magic. It is a discipline: when a fact about identity, direction, or domain determines correctness, put that fact in the source where the compiler and the reader can see it. The four habits are four ways of asking whether you have done so."

---

Mira's corrected simulation runs. The total energy is conserved to machine precision. The Fortran reference agrees to five decimal places. The plots overlay perfectly.

But the real correction is not in the output. It is in the source. The code now says which coordinates it eliminates, which it copies along, how they flow from one operation to the next, and where the boundaries lie. The next person who reads it—even if that person is Mira, six months from now—will see the physics in the indices.

That is what the coordinate habit builds. Not just correct programs. Programs that stay correct when you are not there to remember what the integers meant.

When Mira adds a magnetic field `B` as a fifth field component, the named-field-index approach—`let B_field = 4`—means she changes the index definition in one place and adds one branch to the recurrence. In the integer-literal version, she would have to find and update every occurrence of the number `4` and every array slice that assumed four fields. The named-field approach makes the semantics of each index explicit, so the compiler can flag any place where the new field is missing. Mira's simulation uses a recurrence for `state[t, ...]`. The gradient `@total_energy[T-1] / @state[0, b, h, w, f]` computes the sensitivity of the final energy to the initial state—and yes, the gradient respects the `where` clause on the Laplacian, because the backward pass inherits the forward pass's domain constraints. Mira's colleague proposes `fn laplacian[..spatial](x: [f32; ..batch, ..spatial]) -> [f32; ..batch, ..spatial]`. The advantage is reuse and a named abstraction. The cost is that the inline version shows the stencil arithmetic directly—the coordinate indices inside the Laplacian are part of the physics. Abstraction hides them. Whether that trade is worth it depends on whether the reader needs to see the stencil or just the fact that a Laplacian was applied.
