# Fair train-level comparison sources

These files are side-by-side source candidates for any future train-level
`LoC` comparison in the paper.

Rules for using them:

- Compare only files that express the same immediate semantic object.
- Count nonblank, noncomment lines.
- Exclude import lines if the table says imports are excluded.
- Do not compare an explicit Einlang tensor-definition artifact against a
  higher-level adjacent-system helper such as `nn.Module`, `torch.optim`,
  `optax`, trainer wrappers, or dataset boilerplate.

The current pairs are:

- `train_mlp_one_step.ein`
- `train_mlp_one_step_torch.py`
- `train_mlp_one_step_jax.py`
- `train_linear_recurrence.ein`
- `train_linear_recurrence_torch.py`
- `train_linear_recurrence_jax.py`
- `kalman_filter_core.ein`
- `kalman_filter_core_torch.py`
- `kalman_filter_core_jax.py`

They intentionally stay at the same abstraction level:

- explicit parameter tensors
- explicit forward equations
- explicit squared-error loss
- explicit gradient request
- explicit SGD-style update
- explicit evaluation values
