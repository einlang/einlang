# `test_print_at.py` — study-only cases (`STUDY_SKIP_CASES`)

Same programs as `pytest.mark.skip` entries in `tests/unit/test_autodiff_pass.py::_PRINT_DIFF_ML_OPS`. Run `python3 scripts/test_print_at.py --study-only` to attempt compile+exec (diagnostic; no exit failure).

---

## `softmax`

- **Pytest skip reason:** softmax autodiff not yet supported without @fn rule
- **Math reference:** y_i = exp(x_i)/sum_k exp(x_k); ∂y_i/∂x_j = y_i (δ_ij − y_j) (row i of Jacobian).

```
use std::ml;
let x = [[1.0, 2.0, 3.0]];
let y = std::ml::softmax(x);
print(@y);
```

---

## `log_softmax`

- **Pytest skip reason:** log_softmax autodiff not yet supported without @fn rule
- **Math reference:** log_softmax(x)_i = x_i − log(sum_k exp(x_k)); ∂/∂x_j = δ_ij − softmax(x)_j.

```
use std::ml;
let x = [[1.0, 2.0, 3.0]];
let y = std::ml::log_softmax(x);
print(@y);
```

---

## `reduce_sum`

- **Pytest skip reason:** print(@y) for multi-step inlined function: intermediate var out of scope
- **Math reference:** y = sum_ij x_ij; ∂y/∂x is all 1s (same shape as x).

```
use std::ml;
let x = [[1.0, 2.0, 3.0]];
let y = std::ml::reduce_sum(x);
print(@y);
```

---

## `reduce_mean`

- **Pytest skip reason:** print(@y) for multi-step inlined function: intermediate var out of scope
- **Math reference:** y = (1/N) sum x; ∂y/∂x is constant 1/N on each element.

```
use std::ml;
let x = [[1.0, 2.0, 3.0]];
let y = std::ml::reduce_mean(x);
print(@y);
```

---

## `reduce_l1`

- **Pytest skip reason:** Einstein clause body with function call not yet supported
- **Math reference:** y = sum |x|; ∂y/∂x = sign(x) (subgradient at 0).

```
use std::ml;
let x = [[1.0, -2.0, 3.0]];
let y = std::ml::reduce_l1(x);
print(@y);
```

---

## `reduce_l2`

- **Pytest skip reason:** print(@y) for multi-step inlined function: intermediate var out of scope
- **Math reference:** y = ||x||_2; ∂y/∂x = x / ||x||_2 (for x ≠ 0).

```
use std::ml;
let x = [[3.0, 4.0]];
let y = std::ml::reduce_l2(x);
print(@y);
```

---

## `reduce_sum_square`

- **Pytest skip reason:** Einstein clause body with power not yet supported
- **Math reference:** y = sum x^2; ∂y/∂x = 2x elementwise.

```
use std::ml;
let x = [[1.0, 2.0, 3.0]];
let y = std::ml::reduce_sum_square(x);
print(@y);
```

---

## `reduce_log_sum`

- **Pytest skip reason:** print(@y) for multi-step inlined function: intermediate var out of scope
- **Math reference:** y = log(sum_ij exp(x_ij)); ∂y/∂x = softmax(x) flattened to x's shape.

```
use std::ml;
let x = [[1.0, 2.0, 3.0]];
let y = std::ml::reduce_log_sum(x);
print(@y);
```

---

## `reduce_log_sum_exp`

- **Pytest skip reason:** print(@y) for multi-step inlined function: intermediate var out of scope
- **Math reference:** same as log-sum-exp: ∂y/∂x = softmax(x).

```
use std::ml;
let x = [[1.0, 2.0, 3.0]];
let y = std::ml::reduce_log_sum_exp(x);
print(@y);
```

---

## `linear`

- **Pytest skip reason:** print(@y) for multi-step inlined function: intermediate var out of scope
- **Math reference:** y = x W^T + b; ∂y/∂x = W, ∂y/∂W = x, ∂y/∂b = 1 (layout as in einlang).

```
use std::ml;
let x = [[1.0, 2.0]];
let W = [[0.5, 0.3], [0.2, 0.4]];
let b = [0.1, 0.2];
let y = std::ml::linear(x, W, b);
print(@y);
```

---

## `matmul`

- **Pytest skip reason:** matmul shape inference error in print(@y)
- **Math reference:** C = A B; ∂L/∂A = (∂L/∂C) B^T, ∂L/∂B = A^T (∂L/∂C) (VJP form for scalar L).

```
use std::ml;
let A = [[1.0, 2.0], [3.0, 4.0]];
let B = [[5.0, 6.0], [7.0, 8.0]];
let C = std::ml::matmul(A, B);
print(@C);
```

---

## `mse_loss`

- **Pytest skip reason:** print(@y) for multi-step inlined function: intermediate var out of scope
- **Math reference:** mean (pred−target)^2; ∂/∂pred = (2/N)(pred − target) (per reduction in impl).

```
use std::ml;
let pred = [[1.0, 2.0, 3.0]];
let target = [[1.5, 2.5, 3.5]];
let y = std::ml::mse_loss(pred, target);
print(@y);
```

---

## `mae_loss`

- **Pytest skip reason:** print(@y) for multi-step inlined function: intermediate var out of scope
- **Math reference:** mean |pred−target|; ∂/∂pred = sign(pred−target) / N (subgradient at 0).

```
use std::ml;
let pred = [[1.0, 2.0, 3.0]];
let target = [[1.5, 2.5, 3.5]];
let y = std::ml::mae_loss(pred, target);
print(@y);
```

---

## `huber_loss`

- **Pytest skip reason:** print(@y) for multi-step inlined function: intermediate var out of scope
- **Math reference:** quadratic near 0, linear far; ∂/∂pred is piecewise (pred−target) or ±δ.

```
use std::ml;
let pred = [[1.0, 2.0, 3.0]];
let target = [[1.5, 2.5, 3.5]];
let y = std::ml::huber_loss(pred, target, 1.0);
print(@y);
```

---

## `binary_cross_entropy`

- **Pytest skip reason:** print(@y) for multi-step inlined function: intermediate var out of scope
- **Math reference:** −(t log p + (1−t)log(1−p)); ∂/∂pred = (p−t)/(p(1−p)) per element (with stable impl variants).

```
use std::ml;
let pred = [[0.8, 0.3, 0.9]];
let target = [[1.0, 0.0, 1.0]];
let y = std::ml::binary_cross_entropy(pred, target);
print(@y);
```

---

## `cosine_similarity`

- **Pytest skip reason:** print(@y) for multi-step inlined function: intermediate var out of scope
- **Math reference:** dot(a,b)/(||a|| ||b||); ∂/∂a, ∂/∂b are projections orthogonal to a,b (vector calculus).

```
use std::ml;
let a = [[1.0, 2.0, 3.0]];
let b = [[4.0, 5.0, 6.0]];
let y = std::ml::cosine_similarity(a, b);
print(@y);
```

---

