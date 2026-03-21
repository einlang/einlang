"""
Forward-mode AD prototype for softmax building blocks.

Demonstrates the correct tangent propagation for:
  1. e[i] = exp(x[i])         → de/dx (diag)
  2. s = sum[k](exp(x[k]))    → ds/dx (vector)
  3. s = sum[k](e[k])          → ds/dx via chain rule through e
  4. y[i] = e[i] / s           → dy/dx (full softmax Jacobian)

Two modes:
  print(@y)  — tangent with @x left as identity (all leaves unevaluated)
  @y / @x    — same Jacobian, just extracted per-seed column

Convention: for a variable v of shape (n,), its tangent dv has shape (n, n_seeds).
            Seed j corresponds to @x_j = 1, all others 0.
            So dv[:, j] = ∂v/∂x_j.
"""
import numpy as np

x = np.array([1.0, 2.0, 3.0])
n = len(x)

# -- Primals --
e = np.exp(x)       # shape (n,)
s = np.sum(e)        # scalar
y = e / s            # shape (n,) — softmax

print("=== Primal values ===")
print(f"x = {x}")
print(f"e = exp(x) = {e}")
print(f"s = sum(e) = {s}")
print(f"y = e/s    = {y}")

# =====================================================================
# Forward-mode AD: dx = I  (all leaves unevaluated)
# =====================================================================
dx = np.eye(n)  # dx[i, j] = ∂x_i / ∂x_j = δ_{ij}

# --- Part 1: e[i] = exp(x[i]) ---
# @fn exp(x) { exp(x) * @x }  →  de[i,j] = exp(x[i]) * dx[i,j]
de = e[:, None] * dx  # (n, n_seeds)

print("\n=== Part 1: de = exp(x) * dx (elementwise @fn rule) ===")
print(f"@e (tangent, leaves unevaluated) =\n{de}")
print(f"@e/@x (Jacobian) =\n{de}")
# Expected: diag(exp(x)) = diag([2.718, 7.389, 20.086])

# --- Part 2: s = sum[k](exp(x[k])) (direct, no intermediate e) ---
# d(sum[k](f(x[k]))) = sum[k](df(x[k])) = sum[k](exp(x[k]) * dx[k,:])
ds_direct = np.sum(de, axis=0)  # (n_seeds,)

print("\n=== Part 2: ds = sum_k(de[k]) (sum-of-exp, direct) ===")
print(f"@s (tangent) = {ds_direct}")
print(f"@s/@x        = {ds_direct}")
# Expected: [exp(1), exp(2), exp(3)] = [2.718, 7.389, 20.086]

# --- Part 3: s = sum[k](e[k]) (chain rule through named e) ---
# d(sum[k](e[k])) = sum[k](de[k,:])  — same result, just via chain rule
ds_chain = np.sum(de, axis=0)  # (n_seeds,)

print("\n=== Part 3: ds = sum_k(de[k]) (chain rule through e) ===")
print(f"@s (tangent) = {ds_chain}")
print(f"@s/@x        = {ds_chain}")

# --- Part 4: y[i] = e[i] / s (quotient rule) ---
# d(a/b) = (da*b - a*db) / b^2
# dy[i,j] = (de[i,j]*s - e[i]*ds[j]) / s^2
dy = (de * s - e[:, None] * ds_chain[None, :]) / (s ** 2)  # (n, n_seeds)

print("\n=== Part 4: dy = (de*s - e*ds) / s^2 (softmax Jacobian) ===")
print(f"@y (tangent, leaves unevaluated) =\n{dy}")
print(f"@y/@x (Jacobian) =\n{dy}")

# Verify against known softmax Jacobian: J[i,j] = y[i]*(δ_{ij} - y[j])
J_ref = np.diag(y) - np.outer(y, y)
print(f"\nReference softmax Jacobian y*(I - y^T) =\n{J_ref}")
print(f"Max error: {np.max(np.abs(dy - J_ref)):.2e}")

# =====================================================================
# Summary: what einlang should generate for each binding's tangent
# =====================================================================
print("\n=== What the autodiff pass must generate ===")
print("let ∂x = <seed>                           # leaf: identity or 1")
print("let ∂e[i] = exp(x[i]) * ∂x[i]            # @fn rule applied inside Einstein clause")
print("let ∂s   = sum[k](∂e[k])                  # d(sum) = sum(d)")
print("let ∂y[i] = (∂e[i]*s - e[i]*∂s) / s**2   # d(a/b) quotient rule")
print()
print("For print(@y):  evaluate ∂y with ∂x = identity → full Jacobian")
print("For @y/@x:      same computation, or symbolic d(expr)/d(x)")
