# Vectorized Reduction Failure Analysis

When the vectorized reduction path in `execute_reduction_with_loops` was enabled (using `_try_vectorized_reduction` when `parallel_shape` is set), the following tests failed. This document summarizes the failure patterns and the intended behavior of the vectorized path.

---

## Map to NumPy fancy indexing (correct mental model)

For each example, the **correct** vectorized evaluation is: give each index a shape so that broadcasting yields `(parallel_dims..., reduction_dims)`, evaluate the body once, then `sum(axis=reduction_axes)`.

### Example 1: 2D matmul — `C[i,k] = sum[j](A[i,j] * B[j,k])`

- **Parallel:** `i` (axis 0), `k` (axis 1). **Reduction:** `j` (axis 2).
- Index shapes: `i` → (2,1,1), `k` → (1,2,1), `j` → (1,1,2). Body result shape (2,2,2) with `body[i,k,j] = A[i,j]*B[j,k]`.

```python
i = np.arange(2).reshape(2, 1, 1)
k = np.arange(2).reshape(1, 2, 1)
j = np.arange(2).reshape(1, 1, 2)
C = (A[i, j] * B[j, k]).sum(axis=-1)   # sum over j
# C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0] = 19 ✓
```

### Example 2: Row sums — `row_sums[i] = sum[j](A[i, j])`

- **Parallel:** `i` (axis 0). **Reduction:** `j` (axis 1).
- Index shapes: `i` → (3,1), `j` → (1,3). Body result shape (3,3) with `body[i,j] = A[i,j]`.

```python
i = np.arange(3).reshape(3, 1)
j = np.arange(3).reshape(1, 3)
row_sums = A[i, j].sum(axis=-1)   # sum over j
# row_sums[0] = A[0,0]+A[0,1]+A[0,2] = 3, row_sums[1]=6, row_sums[2]=9 ✓
```

### Example 3a: First reduction — `B[i] = sum[j](A[i, j])`

Same as Example 2: one parallel dim `i`, one reduction dim `j`.

```python
i = np.arange(3).reshape(3, 1)
j = np.arange(3).reshape(1, 3)
B = A[i, j].sum(axis=-1)   # shape (3,) → B = [0, 3, 6] ✓
```

### Example 3b: Second reduction — `C = sum[i](B[i] * B[i])`

- **No parallel dims** (single scalar output). **Reduction:** `i` (one axis).
- Index shape: `i` → (3,). Body result shape (3,) with `body[i] = B[i]*B[i]`. Then sum over the only axis.

```python
i = np.arange(3)
C = (B[i] * B[i]).sum()   # 0² + 3² + 6² = 45 ✓
```

**Rule:** Parallel indices get shapes that each occupy one leading axis (with 1s elsewhere); the reduction index gets shape that occupies the **last** axis(s). So `body` has layout `(parallel..., reduction...)` and we always `sum(axis=tuple(range(-n_red, 0)))`.

---

## What the vectorized path does

1. **Build reduction index arrays:** For each reduction variable (e.g. `j`), build an array of size `reduction_size` and reshape to `red_shape` (e.g. `(3,)` for one var).
2. **Broadcast:** `ctx[defid] = broadcast_to(red_arr, parallel_shape + red_shape)` so the reduction var has shape `(parallel_dims..., reduction_dims...)`.
3. **Evaluate body once:** `result = body_evaluator(ctx)`. Parallel vars (e.g. `i`, `k`) are already in the backend env from the clause, with shapes like `(2,1)` and `(1,2)`.
4. **Reduce:** `reduced = result.sum(axis=tuple(range(-n, 0)))` (last `n` axes), then reshape to `parallel_shape`.

The code **assumes** the body result has shape exactly `parallel_shape + expected_shape` with **semantic** layout: `result[parallel_dims..., reduction_dims...]` = body value at those indices. Wrong layout or wrong axes → wrong numbers.

---

## Failure 1: 2D matmul-like — wrong scalar in one cell

**Source:**
```text
let C[i in 0..2, k in 0..2] = sum[j](A[i,j] * B[j,k]);
```
**A = [[1,2],[3,4]], B = [[5,6],[7,8]].**

**Expected:** C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0] = 1*5 + 2*7 = **19**.

**Actual (with vectorized reduction):** C[0,0] = **21**.

**Interpretation:** 21 = 1*6 + 2*8 = A[0,0]*B[0,1] + A[0,1]*B[1,1], i.e. one of the terms uses the wrong column of B (k mixed with j). So the reduction is combining the right rows of A with the **wrong** columns of B — i.e. the **axis mapping** for the reduction index vs. the parallel indices in the body does not match what we assume. The body result layout is likely `result[i, k, j]` in memory but with j and k dimensions swapped in how they were filled, so summing over the last axis does not correspond to “sum over j.”

---

## Failure 2: Partial (row) reduction — same value every row

**Source:**
```text
let A[i in 0..3, j in 0..3] = i + j;
let row_sums[i in 0..3] = sum[j in 0..3](A[i, j]);
```
**A = [[0,1,2],[1,2,3],[2,3,4]].**

**Expected:** row_sums = [0+1+2, 1+2+3, 2+3+4] = **[3, 6, 9]**.

**Actual (with vectorized reduction):** row_sums = **[6, 6, 6]**.

**Interpretation:** Total sum of A is 3+6+9 = 18; 18/3 = 6. So we get one scalar (18) and then the same value (6) in every parallel slot — as if we reduced over **all** axes (or over the parallel axis) and then repeated. So the reduction is being applied over the wrong axes: either we are summing over axis 0 (or both axes) instead of only over the reduction axis, or the result layout is (reduction, parallel) instead of (parallel, reduction), and we sum over the last axis (which is the parallel index), giving one number per j that we then broadcast.

---

## Failure 3: Nested reductions — wrong B and C

**Source:**
```text
let A[i in 0..3, j in 0..3] = i * j;
let B[i in 0..3] = sum[j in 0..3](A[i, j]);
let C = sum[i in 0..3](B[i] * B[i]);
```
**A = [[0,0,0],[0,1,2],[0,2,4]].**

**Expected:** B = [0, 3, 6], C = 0² + 3² + 6² = **45**.

**Actual (with vectorized reduction):** B = **[5, 5, 5]**, C = **75**.

**Interpretation:**  
- 5 is the mean of 0, 3, 6 (or (0+3+6)/3 ≈ 3, but we see 5). Actually 0+3+6 = 9; 9/3 = 3. So 5 might be (0+1+2+3+4+6)/something or a wrong reduction. 5 also equals 1+2+2 (one column sum of A) or 0+1+2+2 (first row + one extra). So B is getting a **single** value (or wrong per-row reduction) repeated.  
- 75 = 3 * 5² (three 5s squared and summed). So B really is [5,5,5], and C = sum(B[i]²) = 3*25 = 75. So the first reduction (B[i] = sum_j A[i,j]) is wrong in the same way as Failure 2: we get one number (5) and broadcast it to every i. So again, reduction is applied over the wrong axes or the result layout is swapped so that “sum over last axis” is not “sum over j.”

---

## Common pattern

Across all three:

1. **Wrong axis reduced:** Either we reduce over axes that include the parallel dimensions, or the array we reduce has (reduction_dims, parallel_dims) instead of (parallel_dims, reduction_dims).
2. **Single value repeated:** When the result should be per parallel index (e.g. per row), we get one scalar repeated (e.g. [6,6,6], [5,5,5]), consistent with reducing over the wrong dimensions and then reshaping/broadcasting to `parallel_shape`.
3. **Cross-term mix-up (matmul):** One index (e.g. j vs k) is used in the wrong position, so the reduction sums over the wrong pairing (e.g. A[i,j]*B[k,j] instead of A[i,j]*B[j,k]).

So the **root cause** is that the **layout / axis order** of the body result does not match the assumption “last `n` axes = reduction; leading axes = parallel.” That can happen if:

- Parallel vars are set with shapes that, when broadcast with `(parallel_shape + red_shape)`, make the body output dimensions in a different order (e.g. reduction dim in the middle).
- The reduction var is broadcast to `parallel_shape + red_shape` but the backend evaluates the body with a different convention for which axis is “reduction” (e.g. defid remap or multiple reduction vars).
- For multiple reduction dimensions, `red_shape` and `expected_shape` force a fixed order, but the body might produce the reduction dimensions in another order.

Fixing the vectorized path would require:

1. **Defining** the exact layout contract: which axes of the body result correspond to which parallel indices and which reduction indices (including when there are multiple of each).
2. **Ensuring** the clause sets parallel vars to shapes that, together with the broadcast reduction vars, produce that layout (e.g. parallel var for dim 0 has shape `(size0, 1, 1, ...)`, etc.).
3. **Reducing** only over the axes that correspond to reduction indices (e.g. by mapping from reduction defid to axis index in the result), instead of blindly using the last `n` axes.

Until that is done, the vectorized reduction path remains disabled and reductions in 2a run as scalar for correctness.
