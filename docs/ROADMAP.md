# Roadmap

High-level direction: **NumPy backend (now)** → MLIR via Python (next) → native/GPU. This page tracks **language and ergonomics** items that are not yet first-class, especially the ones that matter for scientific and training workflows built around autodiff.

---

## Nested recurrence + inner Einstein (per-step `print` and clearer training loops)

**Motivation:** Today, a single clause like `W[step, i, j] = { ... }` fuses the time step with the weight grid. That makes **per-step logging** awkward (guards on `step` are easy to get wrong when some indices are vectorized). It also interacts with **recurrence analysis** when the body uses a reduction index that “collides” with a clause index (e.g. `W[step-1, a, j]` with `a` in `sum[a]` marks `i` as recurrence-related), which steers the runtime toward **inner recurrence** execution paths.

**Target shape (conceptual):** outer step as its own Einstein slice, inner grid as a separate clause inside a block:

```text
let W[step in 1..6] = {
    let n = (step - 1) % 10;
    let W_step[i in 0..784, j in 0..10] = {
        let logit_j = sum[a in 0..784](X[n, a] * W[step - 1, a, j]);
        let loss_b = (logit_j - Y[n, j]) ** 2.0;
        let w_ij = W[step - 1, i, j];
        let g = @loss_b / @w_ij;
        w_ij - lr * g
    };
    W_step
};
```

**Requirements / open work:**

- **Typing and storage:** `W[step]` must denote a **rank-3** tensor (or equivalent) so each step holds a full matrix; surface syntax and shape inference must agree.
- **Initialization:** retain an explicit `W[0, i, j] = …` (or equivalent) and any reads of `W[5, …]` after training.
- **Autodiff:** nested Einstein inside blocks already lowers; verify **quotient and recurrence** rules for this pattern end-to-end (training correctness).
- **Scientific debugging:** this would make stepwise diagnostics and autodiff-based sensitivity checks easier to use than ad hoc finite-difference probes inside a vectorized loop.
- **Diagnostics:** with the outer step isolated, `print(...)` (or a small helper) inside the outer block becomes a natural **per-epoch / per-step** hook without fighting vectorized `i`/`j`.

This item is **not** a commitment to a particular syntax keyword; the roadmap goal is **ergonomics + debuggability** for recurrent training loops and other scientific update patterns where autodiff is the preferred replacement for finite-difference gradients.

---

## Natural multi-value recurrence (coupled state updates)

**Motivation:** Some training loops naturally update multiple recurrent states together from the same previous-step snapshot, e.g. `W[step]` and `B[step]` in a linear/CNN head. Today, users may pack these into a single tensor as a workaround (e.g. appending bias as an extra row), which works but is not the most readable source shape.

**Target shape (conceptual):**

```text
let W[0, ...] = ...;
let B[0, ...] = ...;
let W[step in 1..T, ...] = f(W[step - 1, ...], B[step - 1, ...], ...);
let B[step in 1..T, ...] = g(W[step - 1, ...], B[step - 1, ...], ...);
```

**Requirements / open work:**

- **Language semantics:** define grouped/mutually-coupled recurrence so multiple arrays can be updated per step from the same prior-step values without introducing circular same-step dependencies.
- **Lowering/IR:** represent recurrence groups explicitly (or equivalent) so ordering is deterministic and backend execution uses one coherent previous-step snapshot.
- **Autodiff correctness:** ensure quotient/differential rules remain correct through coupled recurrence updates.
- **Vectorization parity:** avoid regressions in vectorization behavior compared with single-array recurrence.
- **Examples cleanup:** if MNIST training returns, avoid the old packed-weight workaround from the removed `examples/mnist/train.ein`.

**Current status:** supported via workaround (parameter packing) in MNIST training; first-class natural multi-value recurrence is planned.

---

## See also

- [README.md](../README.md) — “Docs and roadmap” summary  
- [RECURRENCE_ORDER_DESIGN.md](RECURRENCE_ORDER_DESIGN.md) — recurrence iteration order  
- [VECTORIZATION_DESIGN.md](VECTORIZATION_DESIGN.md) — Einstein execution and debug flags  
