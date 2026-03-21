# Study-skip cases: expected math vs actual result

Programs from [`scripts/test_print_at.py`](../scripts/test_print_at.py) (`STUDY_SKIP_CASES`).

**Regenerate:** `python3 scripts/gen_study_skip_compare.py`

## Summary

- **COMPILE FAIL:** 4
- **COMPILE OK, EXEC FAIL:** 12
- **EXEC OK (got symbolic print):** 0

Symbolic `print(@y)` output is only comparable to the math reference when compile and exec both succeed.

## Comparison table

| Case | Pytest skip reason | Expected math (reference) | Outcome | Actual (error or print) |
|------|--------------------|-----------------------------|---------|-------------------------|
| `softmax` | softmax autodiff not yet supported without @fn … | y_i = exp(x_i)/sum_k exp(x_k); ∂y_i/∂x_j = y_i (δ_ij − y_j) (row i of J… | **EXEC FAIL** | rectangular_access: expected ndarray, list, or str, got NoneType |
| `log_softmax` | log_softmax autodiff not yet supported without … | log_softmax(x)_i = x_i − log(sum_k exp(x_k)); ∂/∂x_j = δ_ij − softmax(x… | **EXEC FAIL** | rectangular_access: expected ndarray, list, or str, got NoneType |
| `reduce_sum` | print(@y) for multi-step inlined function: inte… | y = sum_ij x_ij; ∂y/∂x is all 1s (same shape as x). | **EXEC FAIL** | Variable not found (defid=0:4251). Name: x |
| `reduce_mean` | print(@y) for multi-step inlined function: inte… | y = (1/N) sum x; ∂y/∂x is constant 1/N on each element. | **EXEC FAIL** | rectangular_access: expected ndarray, list, or str, got NoneType |
| `reduce_l1` | Einstein clause body with function call not yet… | y = sum \|x\|; ∂y/∂x = sign(x) (subgradient at 0). | **COMPILE FAIL** | ['error: Autodiff: Einstein clause body is not a product of indexed arrays\n --> <test>… |
| `reduce_l2` | print(@y) for multi-step inlined function: inte… | y = \|\|x\|\|_2; ∂y/∂x = x / \|\|x\|\|_2 (for x ≠ 0). | **COMPILE FAIL** | ['error: Autodiff: Einstein clause body is not a product of indexed arrays\n --> <test>… |
| `reduce_sum_square` | Einstein clause body with power not yet support… | y = sum x^2; ∂y/∂x = 2x elementwise. | **COMPILE FAIL** | ['error: Autodiff: Einstein clause body is not a product of indexed arrays\n --> <test>… |
| `reduce_log_sum` | print(@y) for multi-step inlined function: inte… | y = log(sum_ij exp(x_ij)); ∂y/∂x = softmax(x) flattened to x's shape. | **EXEC FAIL** | rectangular_access: expected ndarray, list, or str, got NoneType |
| `reduce_log_sum_exp` | print(@y) for multi-step inlined function: inte… | same as log-sum-exp: ∂y/∂x = softmax(x). | **COMPILE FAIL** | ['error: Autodiff: Einstein clause body is not a product of indexed arrays\n --> <test>… |
| `linear` | print(@y) for multi-step inlined function: inte… | y = x W^T + b; ∂y/∂x = W, ∂y/∂W = x, ∂y/∂b = 1 (layout as in einlang). | **EXEC FAIL** | Variable not found (defid=0:4057). Name: x |
| `matmul` | matmul shape inference error in print(@y) | C = A B; ∂L/∂A = (∂L/∂C) B^T, ∂L/∂B = A^T (∂L/∂C) (VJP form for scalar … | **EXEC FAIL** | error[E0007]: tuple index out of range |
| `mse_loss` | print(@y) for multi-step inlined function: inte… | mean (pred−target)^2; ∂/∂pred = (2/N)(pred − target) (per reduction in … | **EXEC FAIL** | Variable not found (defid=0:4617). Name: predictions |
| `mae_loss` | print(@y) for multi-step inlined function: inte… | mean \|pred−target\|; ∂/∂pred = sign(pred−target) / N (subgradient at 0… | **EXEC FAIL** | Variable not found (defid=0:4623). Name: predictions |
| `huber_loss` | print(@y) for multi-step inlined function: inte… | quadratic near 0, linear far; ∂/∂pred is piecewise (pred−target) or ±δ. | **EXEC FAIL** | rectangular_access: expected ndarray, list, or str, got NoneType |
| `binary_cross_entropy` | print(@y) for multi-step inlined function: inte… | −(t log p + (1−t)log(1−p)); ∂/∂pred = (p−t)/(p(1−p)) per element (with … | **EXEC FAIL** | rectangular_access: expected ndarray, list, or str, got NoneType |
| `cosine_similarity` | print(@y) for multi-step inlined function: inte… | dot(a,b)/(\|\|a\|\| \|\|b\|\|); ∂/∂a, ∂/∂b are projections orthogonal t… | **EXEC FAIL** | rectangular_access: expected ndarray, list, or str, got NoneType |

## Full expected math (same as `STUDY_MATH_REFERENCE`)

- **`softmax`:** y_i = exp(x_i)/sum_k exp(x_k); ∂y_i/∂x_j = y_i (δ_ij − y_j) (row i of Jacobian).
- **`log_softmax`:** log_softmax(x)_i = x_i − log(sum_k exp(x_k)); ∂/∂x_j = δ_ij − softmax(x)_j.
- **`reduce_sum`:** y = sum_ij x_ij; ∂y/∂x is all 1s (same shape as x).
- **`reduce_mean`:** y = (1/N) sum x; ∂y/∂x is constant 1/N on each element.
- **`reduce_l1`:** y = sum |x|; ∂y/∂x = sign(x) (subgradient at 0).
- **`reduce_l2`:** y = ||x||_2; ∂y/∂x = x / ||x||_2 (for x ≠ 0).
- **`reduce_sum_square`:** y = sum x^2; ∂y/∂x = 2x elementwise.
- **`reduce_log_sum`:** y = log(sum_ij exp(x_ij)); ∂y/∂x = softmax(x) flattened to x's shape.
- **`reduce_log_sum_exp`:** same as log-sum-exp: ∂y/∂x = softmax(x).
- **`linear`:** y = x W^T + b; ∂y/∂x = W, ∂y/∂W = x, ∂y/∂b = 1 (layout as in einlang).
- **`matmul`:** C = A B; ∂L/∂A = (∂L/∂C) B^T, ∂L/∂B = A^T (∂L/∂C) (VJP form for scalar L).
- **`mse_loss`:** mean (pred−target)^2; ∂/∂pred = (2/N)(pred − target) (per reduction in impl).
- **`mae_loss`:** mean |pred−target|; ∂/∂pred = sign(pred−target) / N (subgradient at 0).
- **`huber_loss`:** quadratic near 0, linear far; ∂/∂pred is piecewise (pred−target) or ±δ.
- **`binary_cross_entropy`:** −(t log p + (1−t)log(1−p)); ∂/∂pred = (p−t)/(p(1−p)) per element (with stable impl variants).
- **`cosine_similarity`:** dot(a,b)/(||a|| ||b||); ∂/∂a, ∂/∂b are projections orthogonal to a,b (vector calculus).

## See also

- [TEST_PRINT_AT_STUDY_SKIP_DUMP.md](TEST_PRINT_AT_STUDY_SKIP_DUMP.md)
- [study_skip_ir/README.md](study_skip_ir/README.md) — IR S-expr + `*.meta.txt` per case (`python3 scripts/dump_study_skip_ir.py`)
- [PRINT_DIFFERENTIAL.md](PRINT_DIFFERENTIAL.md)

