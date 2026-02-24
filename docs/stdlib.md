# Einlang Standard Library

All functions are implemented in pure Einlang (`.ein` files in `stdlib/`).

---

## `std::math`

### `std::math::basic`

```einlang
use std::math::basic::{abs, sqrt, min, max, sign, pow, floor, ceil, round, trunc,
                        fmod, mod, gcd, lcm, factorial, square, neg, reciprocal};

let x = abs(-5.0);          // 5.0
let r = sqrt(16.0);         // 4.0
let n = gcd(12, 8);         // 4
let f = factorial(5);        // 120
```

### `std::math::trig`

```einlang
use std::math::trig::{sin, cos, tan, asin, acos, atan, atan2};

let angle = pi() / 4;
let s = sin(angle);          // ~0.707
let a = atan2(1.0, 1.0);    // ~0.785
```

### `std::math::exp`

```einlang
use std::math::exp::exp;

let e_power = exp(1.0);     // ~2.718
```

### `std::math::log`

```einlang
use std::math::log::{log, ln, log10, log2, log1p, expm1, rsqrt};

let natural = ln(10.0);     // ~2.303
let base10 = log10(1000.0); // ~3.0
```

### `std::math::hyperbolic`

```einlang
use std::math::hyperbolic::{sinh, cosh, tanh, asinh, acosh, atanh};

let h = tanh(1.0);          // ~0.762
```

### `std::math::clamp`

```einlang
use std::math::clamp::{clamp, clamp_min, clamp_max, saturate};

let c = clamp(15.0, 0.0, 10.0);  // 10.0
let s = saturate(-0.5);           // 0.0 (clamps to 0..1)
```

### `std::math::constants`

```einlang
use std::math::constants::{pi, e, tau, phi, sqrt2, sqrt3,
                            ln2, ln10, log2e, log10e,
                            infinity, nan, epsilon, max_float, min_float};

let circumference = 2.0 * pi() * radius;
```

### `std::math::special`

```einlang
use std::math::special::erf;

let x = erf(1.0);           // ~0.843
```

### `std::math::sum`

```einlang
use std::math::sum::sum;

let total = sum([1, 2, 3, 4]);  // 10
```

---

## `std::array`

```einlang
use std::array::{flatten, transpose, sum, concatenate,
                  argmax, argmin, argmax_all, argmin_all,
                  partition, partition_2d, topk_extract, topk_with_indices_extract};

let matrix = [[1, 2], [3, 4]];
let t = transpose(matrix);       // [[1, 3], [2, 4]]
let f = flatten(matrix);         // [1, 2, 3, 4]
let c = concatenate([1, 2], [3, 4]); // [1, 2, 3, 4]
let idx = argmax([3, 1, 4, 1, 5]);   // 4
```

---

## `std::ml`

### Activations (`std::ml::activations`)

```einlang
use std::ml::activations::{relu, sigmoid, softmax, log_softmax, leaky_relu,
                            elu, gelu, swish, selu, softplus, mish,
                            hardtanh, relu6, prelu, hardsigmoid, hardswish,
                            softsign, tanhshrink, softshrink, hardshrink,
                            threshold, thresholded_relu, celu, gelu_tanh};

let out = relu([[-1, 0, 1], [2, -2, 0.5]]);  // [[0, 0, 1], [2, 0, 0.5]]
let probs = softmax([1.0, 2.0, 3.0]);
```

### Layers (`std::ml::layers`)

```einlang
use std::ml::layers::{linear, gemm, conv2d};

let out = linear(input, weights, bias);
let mm = gemm(A, B, C, 1.0, 0.0, 0, 0);
```

### Convolution (`std::ml::conv_ops`)

```einlang
use std::ml::conv_ops::{conv, conv_transpose, depthwise_conv};

let out = conv(X, W, B, [1, 1], [0, 0, 0, 0], [1, 1]);
```

### Normalization (`std::ml::norm_ops`)

```einlang
use std::ml::norm_ops::{batch_normalization, instance_normalization,
                         layer_normalization, lrn, lp_normalization,
                         mean_variance_normalization};

let bn = batch_normalization(X, scale, B, mean, var, 1e-5);
let ln = layer_normalization(X, scale, B, 1e-5, -1);
```

### Pooling (`std::ml::pool_ops`)

```einlang
use std::ml::pool_ops::{max_pool, average_pool, global_average_pool,
                         global_max_pool, lp_pool, max_roi_pool};

let pooled = max_pool(X, [2, 2], [2, 2], [0, 0, 0, 0]);
let gap = global_average_pool(X);
```

### Loss Functions (`std::ml::ml_ex`)

```einlang
use std::ml::ml_ex::{mse_loss, mae_loss, cross_entropy_loss,
                      binary_cross_entropy, softmax_cross_entropy_loss,
                      huber_loss, cosine_similarity};

let loss = mse_loss(predictions, targets);
```

Also in `std::ml::ml_ex`: linear algebra utilities — `eye`, `diag_extract`, `diag_construct`, `trace`, `frobenius_norm`, `outer`, `kron`, `tril`, `triu`, `roll`, `repeat_interleave`, `flip`, `image_scaler`.

### Attention (`std::ml::attention_ops`)

```einlang
use std::ml::attention_ops::{attention_dummy, multi_head_attention_simple,
                              multi_head_attention};

let out = multi_head_attention(query, key, value, 8, scale, mask);
```

### Recurrent (`std::ml::recurrent_ops`)

```einlang
use std::ml::recurrent_ops::{rnn, lstm, gru};

let out = lstm(X, W, R, B, initial_h, initial_c, hidden_size, "forward", 0.0);
```

### Reduction (`std::ml::reduction_ops`)

```einlang
use std::ml::reduction_ops::{reduce_mean, reduce_sum, reduce_max, reduce_min,
                              reduce_l1, reduce_l2, reduce_sum_square,
                              reduce_log_sum, reduce_log_sum_exp, reduce_prod};

let mean = reduce_mean(tensor);
```

### Shape (`std::ml::shape_ops`)

```einlang
use std::ml::shape_ops::{reshape, squeeze, unsqueeze, split, expand, shape};

let reshaped = reshape(data, [2, 3]);
let squeezed = squeeze(data, [1]);
```

### Transform (`std::ml::transform_ops`)

```einlang
use std::ml::transform_ops::{pad, depth_to_space, space_to_depth,
                              range, constant_of_shape, concat, tile,
                              transpose, flatten};

let padded = pad(data, [1, 1, 1, 1], 0.0);
let seq = range(0, 10, 1);
```

### Indexing (`std::ml::indexing_ops`)

```einlang
use std::ml::indexing_ops::{gather, gather_elements, scatter_elements,
                             onehot, gather_nd, scatter, scatter_nd};

let gathered = gather(data, indices, 0);
let encoded = onehot(indices, 10, [0.0, 1.0]);
```

### Linear Algebra (`std::ml::linalg_ops`)

```einlang
use std::ml::linalg_ops::{matmul, batch_matmul};

let product = matmul(A, B);
```

### Comparison & Logic

```einlang
use std::ml::comparison_ops::{equal, greater, less, greater_or_equal,
                               less_or_equal, not_equal, not};
use std::ml::logical_ops::{logical_and, logical_or, logical_xor, logical_not};

let mask = greater(tensor, 0.0);
```

### Selection (`std::ml::selection_ops`)

```einlang
use std::ml::selection_ops::{topk, nonzero, argmax, argmin};

let top = topk(X, 5, 0);
```

### Math (`std::ml::math_ops`)

```einlang
use std::ml::math_ops::{add, subtract, multiply, divide, rsqrt};
```

### Trig (`std::ml::trig_ops`)

```einlang
use std::ml::trig_ops::{tanh, sinh, cosh};
```

### Special (`std::ml::special_ops`)

```einlang
use std::ml::special_ops::{is_nan, is_inf, einsum};
```

### Utility (`std::ml::utility_ops`)

```einlang
use std::ml::utility_ops::{where, identity, constant, dropout,
                            l2_normalize, numel, size, cast, slice, cumsum};

let selected = where(condition, x, y);
let sliced = slice(data, [0], [5], [0], [1]);
```

---

## `std::io`

```einlang
use std::io::{current_dir, list_dir, file_exists, read_file, write_file,
               append_file, delete_file, create_dir, remove_dir,
               copy_file, move_file, file_size, is_file, is_dir,
               join_path, split_path, basename, dirname,
               absolute_path, relative_path};

let files = list_dir(".");
let path = join_path(["/home", "user", "data.txt"]);
let dir = dirname("/home/user/file.txt");   // "/home/user"
let name = basename("/home/user/file.txt"); // "file.txt"
```

---

See [Language Reference](reference.md) for syntax and semantics.
