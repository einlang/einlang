# Roadmap

High-level direction: **NumPy backend (now)** → MLIR via Python (next) → native/GPU. This page tracks **language and ergonomics** items that are not yet first-class.

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
- **Diagnostics:** with the outer step isolated, `print(...)` (or a small helper) inside the outer block becomes a natural **per-epoch / per-step** hook without fighting vectorized `i`/`j`.

This item is **not** a commitment to a particular syntax keyword; the roadmap goal is **ergonomics + debuggability** for recurrent training loops.

---

## See also

- [README.md](../README.md) — “Docs and roadmap” summary  
- [RECURRENCE_ORDER_DESIGN.md](RECURRENCE_ORDER_DESIGN.md) — recurrence iteration order  
- [VECTORIZATION_DESIGN.md](VECTORIZATION_DESIGN.md) — Einstein execution and debug flags  
