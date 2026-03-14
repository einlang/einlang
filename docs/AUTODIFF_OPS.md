# Autodiff: differentiation rules by op

This document gives **concrete differentiation rules** for a potential autodiff implementation. It extends [AUTODIFF_FUTURE.md](AUTODIFF_FUTURE.md) with op-level VJPs (vector–Jacobian products) and gradient shapes. **Status:** design only; no autodiff in the language today.

**Conventions:**
- \(y = f(x)\): forward. \(\bar{x}\) = gradient of loss \(L\) w.r.t. \(x\), i.e. \(\partial L / \partial x\) (same shape as \(x\)).
- VJP: given \(\bar{y} = \partial L / \partial y\), we give \(\bar{x} = \partial L / \partial x\) for each input \(x\).
- All shapes are preserved: \(\bar{x}\) has the same shape as \(x\).

**Forward vs reverse (math vs backprop):** In math we write \(\mathrm{d}y = \frac{\mathrm{d}y}{\mathrm{d}x}\,\mathrm{d}x\): a change \(\mathrm{d}x\) propagates **forward** to \(\mathrm{d}y\). That’s **forward-mode** autodiff (tangents flow \(x \to y\)). Here we use **reverse-mode**: we are given the gradient *at the output* (\(\bar{y} = \partial L/\partial y\)) and compute the gradient *at the input* (\(\bar{x} = \partial L/\partial x\)). So the flow is **\(\bar{y} \to \bar{x}\)** (adjoints flow backward, from \(y\) toward \(x\)). If we ignore \(L\) and think in differentials only, this is like propagating **\(1/\mathrm{d}y \to 1/\mathrm{d}x\)**, with \(\frac{1}{\mathrm{d}x} = \frac{\mathrm{d}y}{\mathrm{d}x} \cdot \frac{1}{\mathrm{d}y}\) (same derivative, reverse direction). Example: if \(\mathrm{d}y = 10\,\mathrm{d}x\) then \(\frac{1}{\mathrm{d}y} = \frac{1}{10}\,\frac{1}{\mathrm{d}x}\), so \(\frac{1}{\mathrm{d}x} = 10\cdot\frac{1}{\mathrm{d}y}\) ⇒ \(\bar{x} = 10\cdot\bar{y}\). The formulas below are all in that sense: “given \(\bar{y}\), here is \(\bar{x}\)”.

---

## 1. Elementwise unary ops

| Op | Forward \(y\) | VJP \(\bar{x}\) | Notes |
|----|----------------|------------------|--------|
| **neg** | \(y = -x\) | \(\bar{x} = -\bar{y}\) | |
| **pos** | \(y = x\) | \(\bar{x} = \bar{y}\) | |
| **exp** | \(y = \exp(x)\) | \(\bar{x} = \bar{y} \cdot y\) | \(y\) from forward |
| **ln** | \(y = \ln(x)\) | \(\bar{x} = \bar{y} / x\) | \(x \neq 0\) |
| **sqrt** | \(y = \sqrt{x}\) | \(\bar{x} = \bar{y} / (2 y)\) | \(y\) from forward |
| **sin** | \(y = \sin(x)\) | \(\bar{x} = \bar{y} \cdot \cos(x)\) | |
| **cos** | \(y = \cos(x)\) | \(\bar{x} = -\bar{y} \cdot \sin(x)\) | |
| **tanh** | \(y = \tanh(x)\) | \(\bar{x} = \bar{y} \cdot (1 - y^2)\) | \(y\) from forward |
| **sigmoid** \(\sigma\) | \(y = 1/(1+e^{-x})\) | \(\bar{x} = \bar{y} \cdot y \cdot (1-y)\) | \(y\) from forward |
| **abs** | \(y = \lvert x\rvert\) | \(\bar{x} = \bar{y} \cdot \mathrm{sign}(x)\) | subgradient at 0 (e.g. 0) |
| **relu** | \(y = \max(0,x)\) | \(\bar{x} = \bar{y}\) where \(x>0\), else \(0\) | subgradient at 0 (e.g. 0) |
| **leaky_relu** | \(y = x\) if \(x>0\) else \(\alpha x\) | \(\bar{x} = \bar{y}\) where \(x>0\), else \(\alpha \bar{y}\) | |

**ReLU (and abs) at zero:** Not differentiable in the strict sense. Use a defined subgradient (e.g. 0 or 0.5) and document it.

---

## 2. Elementwise binary ops

| Op | Forward | VJP \(\bar{a}\), \(\bar{b}\) | Notes |
|----|---------|------------------------------|--------|
| **add** | \(y = a + b\) | \(\bar{a} = \bar{y},\quad \bar{b} = \bar{y}\) | broadcast: same as forward broadcast |
| **sub** | \(y = a - b\) | \(\bar{a} = \bar{y},\quad \bar{b} = -\bar{y}\) | |
| **mul** | \(y = a \cdot b\) | \(\bar{a} = \bar{y} \cdot b,\quad \bar{b} = \bar{y} \cdot a\) | |
| **div** | \(y = a / b\) | \(\bar{a} = \bar{y} / b,\quad \bar{b} = -\bar{y} \cdot a / b^2\) | \(b \neq 0\) |
| **pow** | \(y = a^b\) | \(\bar{a} = \bar{y} \cdot b \cdot a^{b-1},\quad \bar{b} = \bar{y} \cdot a^b \ln(a)\) | \(a > 0\) for \(\bar{b}\); special cases for \(b\) constant |

**Broadcasting:** Gradients w.r.t. broadcasted operands are **summed** over the broadcast dimensions so that \(\bar{a}\) and \(\bar{b}\) have the same shapes as \(a\) and \(b\).

---

## 3. Matrix multiply (matmul)

**Forward:** \(C = A B\) with \(A \in \mathbb{R}^{m\times k}\), \(B \in \mathbb{R}^{k\times n}\), \(C \in \mathbb{R}^{m\times n}\).

**VJP:**
- \(\bar{A} = \bar{C} B^\top\)  (shape \(m\times k\))
- \(\bar{B} = A^\top \bar{C}\)  (shape \(k\times n\))

**Batched:** Same per batch element: `output[..batch, i, j] = sum[k](a[..batch, i, k] * b[..batch, k, j])`. Backward: \(\bar{a}[b,i,k] += \bar{C}[b,i,:] \cdot B[b,:,k]\), \(\bar{b}[b,k,j] += A[b,:,k] \cdot \bar{C}[b,:,j]\).

---

## 4. Linear layer

**Forward:** \(y = x W^\top + b\) (Einlang `linear(x, weights, bias)` with `weights` shape `(out_features, in_features)`).

So \(y_{ij} = \sum_k x_{ik} W_{jk} + b_j\).

**VJP:**
- \(\bar{x} = \bar{y} W\)  (same shape as \(x\))
- \(\bar{W} = \bar{y}^\top x\)  (same shape as \(W\))
- \(\bar{b} = \mathrm{sum}(\bar{y},\ \mathrm{axis}=\mathrm{batch})\)  (same shape as \(b\))

---

## 5. 2D Convolution (conv2d)

**Forward (single batch):**  
\(y[b,c_o,i,j] = \sum_{c_i,m,n} x[b,c_i, i\cdot s_h - p_h + m\cdot d_h, j\cdot s_w - p_w + n\cdot d_w]\; w[c_o,c_i,m,n] + b[c_o]\).

**VJP (standard conv backward):**
- **grad_input** \(\bar{x}\): transpose-convolve \(\bar{y}\) with the kernel (same padding/stride/dilation as forward, no bias).
- **grad_weight** \(\bar{w}\): convolve input \(x\) with \(\bar{y}\) (layout and indexing per standard conv backward for weight).
- **grad_bias** \(\bar{b}\): \(\bar{b}[c_o] = \sum_{b,i,j} \bar{y}[b,c_o,i,j]\).

Implementation-wise: same as PyTorch’s `conv2d` backward (conv_transpose2d for \(\bar{x}\), conv2d for \(\bar{w}\), sum over spatial+batch for \(\bar{b}\)).

---

## 6. Reductions (Einstein sum / scalar output)

**sum:** \(y = \sum_i x_i\).  
\(\bar{x}_i = \bar{y}\) (broadcast scalar \(\bar{y}\) to shape of \(x\)).

**sum over subset of indices:** e.g. `let y[i] = sum[j](x[i,j])`.  
\(\bar{x}[i,j] = \bar{y}[i]\) (broadcast \(\bar{y}\) along \(j\)).

**max:** \(y = \max_i x_i\).  
\(\bar{x}_i = \bar{y}\) where \(i = \mathrm{argmax}(x)\), else 0 (subgradient; one-hot or normalized if multiple argmax).

**min:** Same idea with argmin.

---

## 7. Softmax and log_softmax

**softmax:** \(p_i = e^{x_i - m} / \sum_j e^{x_j - m}\) with \(m = \max(x)\).

**VJP:** \(\bar{x} = p \cdot (\bar{p} - \sum_k p_k \bar{p}_k) = p \cdot (\bar{p} - (p \cdot \bar{p}))\). So: \(s = \sum_i p_i \bar{p}_i\); \(\bar{x}_i = p_i (\bar{p}_i - s)\).

**log_softmax:** \(\ell_i = x_i - \mathrm{logsumexp}(x)\).

**VJP:** \(\bar{x}_i = \bar{\ell}_i - p_i \sum_k \bar{\ell}_k\) where \(p = \mathrm{softmax}(x)\). So: \(s = \sum_k \bar{\ell}_k\); \(\bar{x}_i = \bar{\ell}_i - p_i s\).

---

## 8. IR mapping (what to differentiate)

| IR / construct | Differentiable? | Backward source |
|----------------|------------------|------------------|
| `BinaryOpIR` ADD, SUB, MUL, DIV, POW | Yes | Section 2 |
| `UnaryOpIR` NEG, POS | Yes | Section 1 |
| `UnaryOpIR` NOT, BOOL_NOT | No (boolean) | — |
| `FunctionCallIR` to math (exp, ln, sqrt, sin, cos, tanh, …) | Yes | Section 1 |
| `LoweredEinsteinIR` (element-wise bracket) | Yes | Composition of elementwise |
| `LoweredEinsteinIR` with single reduction (sum) | Yes | Section 6 |
| `LoweredEinsteinIR` matmul pattern | Yes | Section 3 |
| `LoweredEinsteinIR` conv pattern | Yes | Section 5 |
| `LoweredRecurrenceIR` | Yes | Backward recurrence (reverse-time VJP) |
| Comparisons, conditionals (e.g. `if x > 0`) | Subgradient / defined branch | ReLU, etc. |
| `max`/`min` over indices | Yes | Section 6 |

---

## 9. Chain rule and pass structure

- **Reverse mode:** For each op in the forward pass, store any needed forward values (e.g. \(y\) for tanh, \(p\) for softmax). In the backward pass, given \(\bar{y}\), compute \(\bar{x}\) for each input and **add** (accumulate) into the gradient buffer for \(x\) when \(x\) is used in multiple places.
- **Recurrence:** Forward is \(x_0\) given; \(x_t = f(x_{t-1}, \ldots)\). Backward: run a loop from last time step to first; at each step compute VJP of \(f\) w.r.t. \(x_{t-1}\) (and other args) and accumulate into \(\bar{x}_{t-1}\).
- **Shape rule:** Every \(\bar{x}\) has exactly the shape of \(x\); the compiler can check this at the IR level.

---

## 10. Non-differentiable and edge cases

- **Comparison ops** (EQ, NE, LT, LE, GT, GE): not differentiable; used only in conditionals or for masking; gradient through a branch uses the chosen branch’s gradient (e.g. ReLU).
- **MOD, integer DIV:** Not differentiable in the usual sense; either exclude or define a subgradient (e.g. 0).
- **POW** when exponent is not constant: \(\bar{b} = \bar{y} \cdot a^b \ln(a)\); undefined for \(a \le 0\).
- **Division by zero:** Undefined; implementation should avoid or document (e.g. clamp denominator).

This document is the single reference for op-level autodiff rules if the project adds autodiff; the high-level design and constraints remain in [AUTODIFF_FUTURE.md](AUTODIFF_FUTURE.md).

---

## 11. Examples: science and ML (same autodiff)

The same gradient primitive and VJP rules apply in both scientific and ML code. **Math formula first** (see [AUTODIFF_FUTURE.md](AUTODIFF_FUTURE.md)): syntax is formula-like, ASCII only, **no operator reuse** — we use **`grad[x](loss)`** ("gradient w.r.t. x of loss"); bracket = "with respect to", like indices. We do not use `d(loss)/d(x)` because `/` is division. Below we use `grad[x](loss)` in examples.

### 11.1 Science: parameter sensitivity / calibration

**Goal:** Given a scalar loss \(L(\theta)\) (e.g. squared error between model output and data), compute \(\partial L / \partial \theta\) to drive an optimizer or to report sensitivity.

**Example — decay calibration.** Model: \(u(t; k) = u_0 e^{-k t}\). Loss over data points \((t_i, y_i)\):  
\(L(k) = \sum_i (u(t_i; k) - y_i)^2\).  
We want \(\bar{k} = \partial L / \partial k\).

- Forward: for each \(i\), \(u_i = u_0 \exp(-k\,t_i)\); \(L = \sum_i (u_i - y_i)^2\).
- Backward: \(\bar{u}_i = 2(u_i - y_i)\); then \(\bar{k}\) gets contributions from each \(u_i\) via the exp VJP: \(\mathrm{d}u_i/\mathrm{d}k = -u_0 t_i e^{-k t_i} = -t_i u_i\), so \(\bar{k} \mathrel{+}= \bar{u}_i \cdot (-t_i u_i)\).

**In Einlang (hypothetical):**
```text
let u(i) = u0 * exp(-k * t[i]);
let loss = sum(i)( (u(i) - y[i])^2 );
let dL_dk = grad[k](loss);   // gradient w.r.t. k of loss (same as dL/dk)
// optimizer: k_next = k - alpha * dL_dk
```

Same autodiff: elementwise (exp, mul, sub, pow) and reduction (sum) use the rules in §§1–2, 6.

---

**Example — least-squares / linear sensitivity.** Model: \(y = X \beta\). Loss \(L(\beta) = \|X\beta - b\|^2\).  
\(\bar{\beta} = X^\top (2(X\beta - b))\). So we need matmul forward and its VJP (Section 3): \(\bar{X} = \bar{y} \beta^\top\), \(\bar{\beta} = X^\top \bar{y}\). Here \(\bar{y} = 2(X\beta - b)\), so \(\bar{\beta} = X^\top \bar{y}\).

**In Einlang (hypothetical):**
```text
let y = matmul(X, beta);   // or linear layer
let residual = y - b;
let loss = sum(i)( residual[i]^2 );
let dL_dbeta = grad[beta](loss);   // gradient w.r.t. beta, same shape as beta
```

Same matmul VJP; no separate “science” vs “ML” rule.

---

### 11.2 Science: adjoint of conv (iterative inverse)

**Goal:** Solve “undo blur” or inverse problem where the forward model is \(y = C x\) (convolution). Iterative solvers (e.g. conjugate gradient) need \(C\) and \(C^\top\). So we use the same **conv** for \(C\) and **conv_transpose** (adjoint) for \(C^\top\).

**In Einlang (hypothetical):**
```text
// Forward: y = conv(x, w, stride, pad, ...)
let y = conv(blurred_input, kernel, ...);

// In CG solver we need: residual = y - conv(x_est, w, ...);  and  C^T * v:
let Ct_v = conv_transpose(v, w, ...);   // same kernel, same stride/pad — one op
```

So “deconv” is not a second syntax: it’s the single **adjoint** op used both as the backward of conv (below) and as \(C^\top\) in scientific solvers.

---

### 11.3 ML: training a small conv net

**Goal:** One forward pass through one conv layer + ReLU, then scalar loss; compute gradients w.r.t. input, weights, and bias for a training step.

**Forward:**
```text
let h = conv(x, w, bias, stride, pad, ...);
let a = relu(h);
let loss = sum(...)( (a - target)^2 );
```

**Backward (what autodiff produces from the same VJP rules):**
- \(\bar{a} = 2(a - \mathrm{target})\) (elementwise and sum).
- \(\bar{h} = \mathrm{relu}\_{\mathrm{back}}(\bar{a}, h)\): \(\bar{h}_i = \bar{a}_i\) where \(h_i > 0\), else 0 (Section 1).
- \(\bar{x} = \mathrm{conv}\_{\mathrm{transpose}}(\bar{h}, w, ...)\), \(\bar{w} = \mathrm{conv}(x, \bar{h}, ...)\), \(\bar{b} = \mathrm{sum}(\bar{h})\) (Section 5).

So the **same** conv_transpose (adjoint) op that science uses for \(C^\top\) is exactly the grad_input of conv. One op, two uses.

**In Einlang (hypothetical):**
```text
let loss = ...;   // as above
let dL_dx   = grad[x](loss);
let dL_dw   = grad[w](loss);
let dL_db   = grad[bias](loss);
// then: w_next = w - lr * dL_dw;  etc.
```

No extra syntax for “ML backward”: the compiler derives it from the same VJP table; conv’s backward is conv_transpose for \(\bar{x}\).

---

### 11.4 Summary

| Use case | Forward | What we need from autodiff | Same op? |
|---------|--------|----------------------------|----------|
| Science: calibration | \(L(\theta)\) (e.g. decay, least-squares) | \(\bar{\theta} = \mathrm{grad}[\theta](L)\) | elementwise + sum + matmul VJPs |
| Science: inverse (CG) | \(y = C x\), solve \(C x = y\) | \(C^\top v\) for solver | conv_transpose = adjoint |
| ML: train conv layer | conv → relu → loss | \(\bar{x}, \bar{w}, \bar{b}\) | conv backward uses same conv_transpose |

One formula-like syntax: `d(loss)/d(x)` (same as dL/dx on the board) and one table of VJPs. Conv’s backward is the same adjoint (conv_transpose) used in scientific iterative inversion.
