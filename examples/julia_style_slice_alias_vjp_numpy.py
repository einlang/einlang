"""NumPy reference: scalar slice view — generic Julia / Zygote-style AD (no IR pattern matching).

Julia AD (Zygote, ChainRules): ``getindex`` (or a scalar view) has rrule that takes the cotangent
on the *element* and accumulates into the parent storage. Equivalently for a scalar loss ``ℓ`` and
view ``e = W[i0,j0]``:

  ∂ℓ/∂e  =  (∂ℓ/∂W)[i0, j0]

i.e. **full cotangent on storage**, then **project with the same indices** — not a hand-built
Kronecker chain inside the reduction over all elements.

Wrong approach: sum over reduction indices with a mask that accidentally identifies ``e`` with every
loop slot (yields ``sum_{ij} 2 W[i,j]`` instead of ``2 W[i0,j0]``).

Port: ``AutodiffPass`` quotient expansion uses ``build_seeded_pullback(..., wrt=root_W)`` plus
``RectangularAccessIR`` at the slice indices (same as ``(@loss / @W)[i,j]``).
"""

import numpy as np


def loss_sum_squares(W: np.ndarray) -> float:
    W = np.asarray(W, dtype=np.float64)
    return float(np.sum(W * W))


def pullback_loss_wrt_W(W: np.ndarray) -> np.ndarray:
    """Storage cotangent ∂(sum W²)/∂W — the generic backward for this loss."""
    W = np.asarray(W, dtype=np.float64)
    return 2.0 * W


def dloss_de_via_projection(W: np.ndarray, i0: int, j0: int) -> float:
    """Julia-style: ∂ℓ/∂e with e = W[i0,j0] equals (∂ℓ/∂W)[i0, j0]."""
    g_W = pullback_loss_wrt_W(W)
    return float(g_W[i0, j0])


def dloss_de_wrong_sum_all_slots(W: np.ndarray, i0: int, j0: int) -> float:
    """Bug: treat every storage cell as tied to e (mask always on)."""
    del i0, j0
    W = np.asarray(W, dtype=np.float64)
    return float(np.sum(2.0 * W))


if __name__ == "__main__":
    W = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    i0, j0 = 1, 0
    g_proj = pullback_loss_wrt_W(W)[i0, j0]
    g_julia = dloss_de_via_projection(W, i0, j0)
    g_bug = dloss_de_wrong_sum_all_slots(W, i0, j0)
    assert g_julia == g_proj == 6.0
    assert g_bug == 20.0
    assert loss_sum_squares(W) == 30.0
    print("ok: pullback + projection g == 6; wrong masked-sum gives", g_bug)
