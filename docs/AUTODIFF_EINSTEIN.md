# Autodiff for Einstein-style expressions (math)

**Status:** Standalone math specification for differentiating tensor expressions in Einstein notation. Reference for [AUTODIFF_ALGORITHM.md](AUTODIFF_ALGORITHM.md) and the implementation in `passes/autodiff.py` (`_diff_einstein_wrt`).

---

## 1. Einstein notation and sum-of-products

### 1.1 Index notation

We consider **indexed arrays** (tensors) over a finite index set per dimension, e.g. A_{ik}, B_{kj} where indices run over ranges (e.g. i \in [0,n), k \in [0,p), j \in [0,m)).

**Einstein convention:** A repeated index in an expression implies **summation** over that index. So

C_{ij} = A_{ik} B_{kj}

means

C_{ij} = \sum_k A_{ik} B_{kj}.

We will always write the sum explicitly when we need to differentiate.

### 1.2 Sum-of-products form (with where-clause)

A single **clause** in our setting has the form

Y_{I} = \sum_{K\mid\phi(I,K)} \prod_{\ell=1}^{L} X^{(\ell)}*{I*\ell},

where:

- I is the tuple of **output indices** (e.g. (i,j) for C_{ij}),
- K is the tuple of **reduction indices** (e.g. k),
- **\phi(I,K)** is an optional **where-clause**: a predicate on the indices so that the sum runs only over K (and I) satisfying \phi. If absent, all indices in range are included.
- Each X^{(\ell)} is an array and I_\ell is the tuple of indices used for that array (a subset of I \cup K).

So the sum is over all (I,K) in range such that \phi(I,K) holds; for each such (I,K) we add the product term. **The where-clause is part of the clause definition** and must be respected when we differentiate.

**Example (matrix multiply, no where):**

C_{ij} = \sum_k A_{ik} B_{kj}.

Here I = (i,j), K = (k), L=2, X^{(1)} = A with I_1 = (i,k), X^{(2)} = B with I_2 = (k,j), and no \phi (or \phi \equiv \text{true}).

**Example (with where):** e.g. upper-triangular part only: sum over k where i \le k \le j.

### 1.3 Multi-clause form

A single **Einstein tensor** can be defined by **multiple clauses** that are **added** to form the same output. So

Y_{I} = \text{(contribution of clause 1)} + \text{(contribution of clause 2)} + \cdots.

Formally, with clauses c = 1,\ldots,C:

Y_{I} = \sum_{c=1}^{C} \Bigl( \sum_{K_c \mid \phi_c(I,K_c)} \prod_{\ell=1}^{L_c} X^{(\ell,c)}*{I*{\ell,c}} \Bigr).

Each clause c has its own reduction indices K_c, where-clause \phi_c, and product of L_c factors. All clauses write into the **same** output indices I (same shape); the backend adds their contributions.

**Example:** C_{ij} = \sum_k A_{ik}B_{kj} + \sum_k D_{ik}E_{kj} (two clauses, same output C).

---

## 2. Derivative of a sum-of-products w.r.t. one array

We want the **partial derivative** of the scalar (or tensor) defined by the sum-of-products with respect to **one** of the arrays, say X^{(w)} (the “wrt” array). That derivative is a **tensor** with indices: the original output indices I plus the indices of X^{(w)}.

### 2.1 One factor equal to the “wrt” array

Assume the clause has the form (with where-clause \phi)

Y_I = \sum_{K\mid\phi(I,K)} \bigl( X^{(w)}*{J} \cdot \prod*{\ell \neq w} X^{(\ell)}*{I*\ell} \bigr),

where J is the index tuple for X^{(w)} (e.g. (i,k) for A_{ik}). The sum is only over (I,K) satisfying \phi.

**Definition:** The derivative of Y with respect to X^{(w)} is the tensor

\frac{\partial Y_I}{\partial X^{(w)}_R}

where R runs over the same index positions as J (e.g. R = (r,s) for a 2-index array).

### 2.2 Pointwise derivative of the product

For a single term in the sum (fixed K), the product is

P = X^{(w)}*{J} \cdot M,\qquad M = \prod*{\ell \neq w} X^{(\ell)}*{I*\ell}.

M does not depend on X^{(w)}. So

\frac{\partial P}{\partial X^{(w)}*R} = \frac{\partial}{\partial X^{(w)}R}\bigl( X^{(w)}{J} \cdot M \bigr) = M \cdot \frac{\partial X^{(w)}*{J}}{\partial X^{(w)}_R}.


### 2.3 Kronecker delta

X^{(w)}*{J} is just one entry of the array X^{(w)}, so

\frac{\partial X^{(w)}*{J}}{\partial X^{(w)}_R} = \begin{cases} 1 & \text{if } R = J  0 & \text{otherwise}. \end{cases}

We write this as **\delta_{R,J}** (Kronecker delta: 1 when the index tuples R and J are equal, 0 otherwise). So

\frac{\partial P}{\partial X^{(w)}*R} = M \cdot \delta*{R,J}.


### 2.4 Sum over reduction indices (with where-clause)


\frac{\partial Y_I}{\partial X^{(w)}*R} = \sum*{K\mid\phi(I,K)} \frac{\partial P}{\partial X^{(w)}*R} = \sum*{K\mid\phi(I,K)} \bigl( M \cdot \delta_{R,J} \bigr).


So the derivative **inherits the same where-clause \phi**: we only sum over K (and I) that satisfy the original clause constraint. Within that, for each K, J is determined by I and K. So \delta_{R,J} is 1 only when J = R. That picks out **at most one** value of K (the one that makes J = R) among those satisfying \phi. If no such K exists, the sum is 0.

**Result (single “wrt” factor, with where-clause):**

\boxed{
\frac{\partial Y_I}{\partial X^{(w)}*R}
= \sum*{K\mid\phi(I,K)} \bigl( \delta_{R,J} \cdot \prod_{\ell \neq w} X^{(\ell)}*{I*\ell} \bigr).
}

In words: same reduction, **same where-clause \phi**, and same other factors; the factor X^{(w)} is replaced by \delta_{R,J}. So we still have one sum over K subject to \phi, and the term is nonzero only when J = R (and \phi(I,K) holds).

---

## 3. Example: matrix multiply C = AB


C_{ij} = \sum_k A_{ik} B_{kj}.

Differentiate w.r.t. A. Here X^{(w)} = A with J = (i,k).


\frac{\partial C_{ij}}{\partial A_{rs}}
= \sum_k \delta_{(r,s),(i,k)}  B_{kj}.

\delta_{(r,s),(i,k)} = 1 iff r = i and s = k. So the only contribution is when k = s (and i = r):

\frac{\partial C_{ij}}{\partial A_{rs}} = \begin{cases} B_{sj} & \text{if } i = r,  0 & \text{otherwise}. \end{cases}


So the 4-tensor \partial C / \partial A has shape (n,m,n,p) (if C is n\times m and A is n\times p) and

\biggl(\frac{\partial C}{\partial A}\biggr)*{ijrs} = B*{sj}\ \ \text{when}\ \ i=r;\quad 0\ \ \text{otherwise}.


---

## 4. Multi-clause: derivative is the sum of per-clause derivatives

By **linearity** of the derivative,

\frac{\partial Y_I}{\partial X^{(w)}_R}
= \frac{\partial}{\partial X^{(w)}*R} \sum*{c=1}^{C} \text{(clause}*c\text{)} = \sum*{c=1}^{C} \frac{\partial \text{(clause}_c\text{)}}{\partial X^{(w)}_R}.


- **Clauses that contain X^{(w)}:** For each such clause, apply the single-clause rule (§2): same reduction and where-clause, replace the wrt factor by \delta_{R,J}, sum over K_c \mid \phi_c. That gives one derivative term per such clause.
- **Clauses that do not contain X^{(w)}:** Their derivative w.r.t. X^{(w)} is zero; they contribute nothing to \partial Y / \partial X^{(w)}.

So the full derivative tensor is the **sum** of the derivative tensors from each clause that depends on the wrt array. All have the same shape (output indices I plus derivative indices R), so they add element-wise.

**Implementation:** Build one derivative clause for each original clause that has a factor equal to the wrt array; collect them in a single EinsteinIR with the same output shape. The IR semantics (multiple clauses → add into same tensor) then implements the sum of per-clause derivatives.

---

## 5. Multiple factors equal to the “wrt” array (within one clause)

If the product contains X^{(w)} more than once, e.g. A_{ik}A_{kj}, then

\frac{\partial}{\partial A_{rs}} \bigl( A_{ik} A_{kj} \bigr) = \delta_{(r,s),(i,k)} A_{kj} + A_{ik} \delta_{(r,s),(k,j)}.

So we get **one term per occurrence** of the wrt array, each term being (delta at that occurrence) × (product of the rest). Summing over K gives the full derivative. The generic rule is: for each occurrence of X^{(w)} in the product, add a term where that occurrence is replaced by \delta_{R,J} and the rest unchanged; then sum over the reduction indices.

---

## 6. Implementation mapping (IR)

- **Multi-clause:** For each original clause that contains the wrt array, build one derivative clause (§4). Return a single EinsteinIR whose `clauses` list is all such derivative clauses; the backend adds their contributions into the same output tensor.
- **Output indices:** The derivative tensor has indices I \cup R: original clause output indices plus one new index per position in the “wrt” array (e.g. i,j,r,s).
- **Where-clause (two parts):**
  1. **Original clause where-clause \phi:** The derivative reduction must apply the **same** \phi(I,K) as the original clause, so we only sum over indices that the original clause allowed. In IR: preserve the clause’s `where_clause` when building the derivative clause.
  2. **Delta \delta_{R,J}:** Implement the delta as **additional** constraints on the reduction: restrict so that each index in J equals the corresponding index in R (e.g. “index at position 1 = r”, “index at position 2 = s”). So the full reduction runs over K subject to **\phi(I,K) \land (J = R)**.
- **Body:** The reduction body is the product of the **other** factors (the wrt factor is effectively replaced by 1 when the delta holds). So we build a reduction with the same loop vars, body = product of non-wrt factors, and where-clause = **original \phi conjoined with (J = R)**.

**Note:** In the IR, a where-clause can live on the **clause** (filter on output/reduction indices) and/or on the **reduction** (filter inside the sum). The derivative must respect **both**: the reduction’s where-clause is conjoined with the delta (J = R), and the clause’s where-clause is preserved on the derivative clause. So the full condition for the derivative sum is \phi_{\text{clause}}(I,K) \land \phi_{\text{red}}(I,K) \land (J = R).

This yields an Einstein-style expression (same IR shape) for \partial Y / \partial X^{(w)} that respects the original where-clause(s) and can be lowered and executed like any other Einstein expression.

---

## 7. References

- Standard multivariable calculus: partial derivative of a sum = sum of partials; derivative of a product uses the product rule.
- Kronecker delta: \delta_{ab} = 1 if a=b, else 0; for index tuples, \delta_{R,J} = 1 iff R = J element-wise.
- [AUTODIFF_ALGORITHM.md](AUTODIFF_ALGORITHM.md) for the overall pass and quotient semantics; [AUTODIFF_OPS.md](AUTODIFF_OPS.md) for scalar/binary rules.

