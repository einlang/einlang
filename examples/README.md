# Einlang Examples

This is the example map for users. If you only want a few good entry points, start here instead of browsing every directory.

## Best first runs

Run these from the repository root:

```bash
python3 -m einlang examples/hello.ein
python3 -m einlang examples/demos/matrix_operations.ein
python3 -m einlang examples/autodiff_small.ein
python3 -m einlang examples/recurrence/recurrence_suite.ein
python3 -m einlang examples/ode/ode_suite.ein
```

That sequence covers the language basics, tensor notation, autodiff, recurrences, and numerics.

## By goal

| Goal | Where to look | Run |
|------|---------------|-----|
| Learn the language | `examples/basics/`, `examples/demos/` | `python3 -m einlang examples/basics/variables_demo.ein` |
| See feature-sized syntax examples | `examples/units/` | [units/README](units/README.md) |
| Try autodiff | root autodiff examples | `python3 -m einlang examples/autodiff_small.ein` |
| Explore recurrences and dynamic programs | `examples/recurrence/`, `examples/value_iteration/`, `examples/job_search/` | `python3 -m einlang examples/recurrence/recurrence_suite.ein` |
| Run numerical simulations | `examples/ode/`, `examples/pde_1d/`, `examples/wave_2d/`, `examples/brusselator/` | `python3 -m einlang examples/ode/ode_suite.ein` |
| Run optimization and workflow examples | `examples/optimization/`, `examples/applications/`, `examples/time_series/`, `examples/finance/` | `python3 -m einlang examples/optimization/optimization_suite.ein` |
| Run model examples | `examples/mnist/`, `examples/mnist_quantized/`, `examples/deit_tiny/`, `examples/whisper_tiny/` | `python3 -m einlang examples/mnist/main.ein` |

## Directories worth knowing

| Directory | What it contains |
|-----------|------------------|
| `examples/basics/` | Small language primers |
| `examples/demos/` | Matrix and tensor examples |
| `examples/units/` | Feature-by-feature lookup examples |
| `examples/recurrence/` | Recurrence patterns and discrete dynamics |
| `examples/ode/`, `examples/pde_1d/`, `examples/wave_2d/`, `examples/brusselator/` | Numerical simulation examples |
| `examples/optimization/`, `examples/value_iteration/`, `examples/job_search/`, `examples/applications/` | Optimization, DP, economics, and workflow examples |
| `examples/mnist/`, `examples/mnist_quantized/`, `examples/deit_tiny/`, `examples/whisper_tiny/` | Larger model examples |

## Detailed guides kept on purpose

Most example directories no longer have their own README. The ones that remain are the ones that provide setup details or real lookup value:

- [units/README](units/README.md)
- [mnist/README](mnist/README.md)
- [mnist_quantized/README](mnist_quantized/README.md)
- [deit_tiny/README](deit_tiny/README.md)
- [whisper_tiny/README](whisper_tiny/README.md)
- [applications/kalman_filter/README](applications/kalman_filter/README.md)

## Setup notes

- `mnist`, `mnist_quantized`, `deit_tiny`, and `whisper_tiny` have extra weight or sample setup; use their README files above.
- Many of the smaller examples run immediately after `pip install -e .`.

## Related docs

- [Getting started](../docs/GETTING_STARTED.md)
- [Language reference](../docs/reference.md)
- [Standard library](../docs/stdlib.md)
- [Autodiff guide](../docs/AUTODIFF.md)
