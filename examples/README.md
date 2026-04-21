# Einlang Examples

This is the example map for users. If you only want a few good entry points, start here instead of browsing every directory.

## Suggested path

These are the lowest-friction paths through the repo. They all run from the repository root after `pip install -e .`.

| Time | Run | Why it is a good stop |
|------|-----|-----------------------|
| 30 seconds | `python3 -m einlang examples/hello.ein` | Confirms the install and shows the CLI on a tiny program. |
| 5 minutes | `python3 -m einlang examples/autodiff_small.ein`<br>`python3 -m einlang examples/demos/matrix_operations.ein` | Shows the core autodiff syntax and the tensor notation quickly. |
| 30 minutes | `python3 -m einlang examples/applications/linear_regression_autodiff.ein`<br>`python3 -m einlang examples/recurrence/recurrence_suite.ein`<br>`python3 -m einlang examples/ode/ode_suite.ein`<br>`python3 -m einlang examples/optimization/optimization_suite.ein` | Shows that the same language reaches fitting, recurrences, numerics, and optimization workflows. |
| After that | [examples/README](README.md), [mnist/README](mnist/README.md), [deit_tiny/README](deit_tiny/README.md), [whisper_tiny/README](whisper_tiny/README.md) | Use the directory guides once you want the heavier showcases or extra setup. |

If you only run one thing after `hello.ein`, run `autodiff_small.ein`.

## By goal

| Goal | Where to look | Run |
|------|---------------|-----|
| Learn the language | `examples/basics/`, `examples/demos/` | `python3 -m einlang examples/basics/variables_demo.ein` |
| See feature-sized syntax examples | `examples/units/` | [units/README](units/README.md) |
| Try autodiff basics | root autodiff examples | `python3 -m einlang examples/autodiff_small.ein` |
| See autodiff on real fitting tasks | `examples/applications/`, `examples/gradient_descent_autodiff.ein`, `examples/mnist/` | `python3 -m einlang examples/applications/linear_regression_autodiff.ein` |
| See the main sklearn-backed training showcase | `examples/mnist/train_sklearn_digits.ein` | [mnist/README](mnist/README.md) |
| See estimator-style testing around an Einlang model | `examples/applications/linear_regression_sklearn_style_checks.py` | `python3 examples/applications/linear_regression_sklearn_style_checks.py` |
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
| `examples/applications/`, `examples/gradient_descent_autodiff.ein`, `examples/mnist/` | Real-world autodiff examples: fitting, calibration, optimization, and training |
| `examples/recurrence/` | Recurrence patterns and discrete dynamics |
| `examples/ode/`, `examples/pde_1d/`, `examples/wave_2d/`, `examples/brusselator/` | Numerical simulation examples |
| `examples/optimization/`, `examples/value_iteration/`, `examples/job_search/`, `examples/applications/` | Optimization, DP, economics, and workflow examples |
| `examples/mnist/`, `examples/mnist_quantized/`, `examples/deit_tiny/`, `examples/whisper_tiny/` | Larger model examples, including the full-split sklearn digits training showcase |

## Detailed guides kept on purpose

Most example directories no longer have their own README. The ones that remain are the ones that provide setup details or real lookup value:

- [units/README](units/README.md)
- [mnist/README](mnist/README.md)
- [mnist_quantized/README](mnist_quantized/README.md)
- [deit_tiny/README](deit_tiny/README.md)
- [whisper_tiny/README](whisper_tiny/README.md)
- [applications/kalman_filter/README](applications/kalman_filter/README.md)

## Setup notes

- The first-run paths above are intentionally root-run and low-friction.
- High-setup directories keep their exact commands in their local README files instead of front-loading that complexity here.
- `mnist`, `mnist_quantized`, `deit_tiny`, and `whisper_tiny` have extra weight or sample setup; use their README files above.
- Many of the smaller examples run immediately after `pip install -e .`.
- Compiler repro programs and isolated pattern fixtures live under `tests/fixtures/`; they are intentionally kept out of the curated starting path.

## Related docs

- [Getting started](../docs/GETTING_STARTED.md)
- [Language reference](../docs/reference.md)
- [Standard library](../docs/stdlib.md)
- [Autodiff guide](../docs/AUTODIFF.md)
