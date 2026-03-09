# Julia demos and how Einlang maps to them

We position Einlang in the same **numerical simulation + ML** space as [Julia](https://julialang.org/). **Coming from Julia?** Start with [Einlang for Julia programmers](EINLANG_FOR_JULIA_PROGRAMMERS.md) (indexing, ODEs, recurrence, mental model). This page lists well-known Julia demos and case studies, with links, and maps them to our examples — in the spirit of [Julia’s documentation](https://docs.julialang.org/) and [SciML showcases](https://docs.sciml.ai/Overview/stable/showcase/): **problem first**, then code; each of our simulation examples includes a **Julia equivalent** (1-based) in comments and is accuracy-tested against NumPy or analytical references.

**Learn from real applications, not just samples.** The table below maps **short demos** (tutorials, minimal runs) to Einlang equivalents so you can compare line-by-line. The real target is Julia's **production use**: [JuliaHub case studies](https://juliahub.com/case-studies) (Aviva, Betterment, AOT), full QuantEcon.jl and SciML workflows, and ecosystem adoption. Our examples are entry points; we aim to support the same class of problems and patterns as those real applications.

**Examples by domain:** [Scientific simulation](../../examples/README.md#scientific-simulation-odes--pdes) (ode, pde_1d, wave_2d, brusselator) · [Discrete dynamics](../../examples/README.md#discrete-dynamics--recurrence) (recurrence) · [Finance](../../examples/README.md#finance) · [Economics / optimization](../../examples/README.md#economics--optimization) (value_iteration, job_search, optimization) · [Time series](../../examples/README.md#time-series) (time_series) · [Computer vision](../../examples/README.md#computer-vision) · [Speech](../../examples/README.md#speech--sequence).

**Summary:** Julia’s *numerical* PDE/ODE use cases (explicit time-stepping, stencil + recurrence) align with Einlang’s [ode](../../examples/ode/) (decay, linear, Lorenz, Lotka–Volterra, **pendulum**, **van der Pol**, **SIR**, **harmonic**, **fitzhugh_nagumo**, **lorenz96**), [pde_1d](../../examples/pde_1d/) (heat, advection), [wave_2d](../../examples/wave_2d/), [brusselator](../../examples/brusselator/), [recurrence/](../../examples/recurrence/) (**Markov stationary**, **logistic**), [optimization/](../../examples/optimization/) (**gradient_descent**, **power_iteration**, **projected_gradient**, **rosenbrock**), [value_iteration/](../../examples/value_iteration/) (Bellman), [job_search/](../../examples/job_search/) (**McCall**), [finance/](../../examples/finance/) (**savings**), and [time_series/](../../examples/time_series/) (**exponential_smoothing**). Same kind of simulation; we use Einstein notation and compile-time shape checking instead of a separate symbolic layer.

---

## Try first

New to the simulation examples? Start with one ODE and the matching Julia material:


| Run this                                    | Julia parallel                                                                                                                                                 |
| ------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `python3 -m einlang examples/ode/decay.ein` | [DifferentialEquations.jl: ODE tutorial](https://docs.sciml.ai/DiffEqDocs/stable/getting_started/#ode_example) — same idea: define the equation, step in time. |


Then explore the table below for the full mapping.

---

## Julia demos and equivalents


| Julia demo / ecosystem                                    | Domain                                             | Link                                                                                                                                                                            | Einlang equivalent                                                                                                                                                                                                                                                                                                                                           |
| --------------------------------------------------------- | -------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **Brusselator PDE** (reaction–diffusion)                  | PDE, numerical simulation                          | [SciML showcase: Brusselator](https://docs.sciml.ai/Overview/stable/showcase/brusselator/)                                                                                      | [brusselator/](../../examples/brusselator/)                                                                                                                                                                                                                                                                                                                     |
| **DifferentialEquations.jl** (ODEs, SDEs, PDEs)           | Numerical integration, time-stepping               | [DiffEqDocs](https://docs.sciml.ai/DiffEqDocs/stable/)                                                                                                                          | [ode/](../../examples/ode/) (decay, linear, Lorenz, Lotka–Volterra, **pendulum**, **van der Pol**, **SIR**, **harmonic**; **fitzhugh_nagumo.ein**, **lorenz96.ein**); heat, wave, Brusselator (explicit PDE time-step)                                                                                                                                           |
| **MethodOfLines.jl** (PDE → ODE, then solve)              | PDE discretization + simulation                    | [MethodOfLines](https://docs.sciml.ai/MethodOfLines/stable/)                                                                                                                    | Same idea: we write the discrete update in Einstein notation ([wave_2d/main.ein](../../examples/wave_2d/main.ein), [pde_1d/heat_1d.ein](../../examples/pde_1d/heat_1d.ein), [run_heat_1d.py](../../examples/pde_1d/run_heat_1d.py))                                                                                                                                   |
| **Quantitative Economics (QuantEcon.jl)**                 | Dynamic programming, Markov chains, linear algebra | [QuantEcon.jl](https://quantecon.org/quantecon-jl/), [Quantitative Economics with Julia](https://julia.quantecon.org/)                                                          | [value_iteration/](../../examples/value_iteration/) (Bellman); [job_search/mccall.ein](../../examples/job_search/mccall.ein) ([McCall search model](https://julia.quantecon.org/dynamic_programming/mccall_model.html), reservation wage); [recurrence/markov_stationary.ein](../../examples/recurrence/markov_stationary.ein) (finite [Markov chains](https://julia.quantecon.org/introduction_dynamics/finite_markov.html)) |
| **JuMP** / **Optim.jl** (optimization)                    | Optimization, control (e.g. MPC)                   | [JuMP](https://jump.dev/), [Optim.jl](https://julianlsolvers.github.io/Optim.jl/), [Optimization.jl Rosenbrock](https://docs.sciml.ai/Optimization/stable/examples/rosenbrock/) | [optimization/](../../examples/optimization/): [gradient_descent.ein](../../examples/optimization/gradient_descent.ein), [power_iteration.ein](../../examples/optimization/power_iteration.ein), [projected_gradient.ein](../../examples/optimization/projected_gradient.ein), [rosenbrock.ein](../../examples/optimization/rosenbrock.ein). **Reusable libraries:** [numerics/](../../examples/numerics/) (DiffEq stepper, Optim 2D gradient descent, QuantEcon value iteration) — run `python3 -m einlang examples/run_numerics.ein` |
| **StateSpaceModels.jl** / **TimeSeries.jl** (forecasting) | Exponential smoothing, ETS                         | [StateSpaceModels.jl](https://github.com/JuliaStats/StateSpaceModels.jl), [TimeSeries.jl](https://juliastats.org/TimeSeries.jl/)                                                | [time_series/exponential_smoothing.ein](../../examples/time_series/exponential_smoothing.ein)                                                                                                                                                                                                                                                                   |
| **SciML** (scientific ML, neural ODEs)                    | Parameter estimation, neural DEs                   | [SciML](https://docs.sciml.ai/Overview/stable/)                                                                                                                                 | ML examples: [mnist](../../examples/mnist/), [deit_tiny](../../examples/deit_tiny/), [whisper_tiny](../../examples/whisper_tiny/)                                                                                                                                                                                                                                     |
| **Aviva (Solvency II)**                                   | Insurance, risk, numerical                         | [JuliaHub case study: Aviva](https://juliahub.com/industries/case-studies/aviva)                                                                                                | Same *style* of code: tensor math + shape safety                                                                                                                                                                                                                                                                                                             |
| **Betterment**                                            | Finance, projections                               | [JuliaHub: Betterment](https://juliahub.com/case-studies/betterment)                                                                                                            | [value_iteration/](../../examples/value_iteration/) (Bellman); [finance/savings.ein](../../examples/finance/savings.ein) (compound interest / savings)                                                                                                                                                                                                             |


---

## What we cover today

- **ODE/PDE numerical simulation (Julia-style):** [ode/](../../examples/ode/) (exponential decay, linear A·u, Lorenz, Lotka–Volterra, **pendulum**, **van der Pol**, **SIR**, **harmonic**, **fitzhugh_nagumo**, **lorenz96**). [pde_1d/](../../examples/pde_1d/) (heat, advection). 2D wave, Brusselator reaction–diffusion. [recurrence/](../../examples/recurrence/) (**Markov stationary**, **logistic**). [optimization/](../../examples/optimization/) (**gradient_descent**, **power_iteration**, **projected_gradient**, **rosenbrock**). [job_search/](../../examples/job_search/) (**McCall** reservation wage). [finance/](../../examples/finance/) (**savings**). [time_series/](../../examples/time_series/) (**exponential_smoothing**). Explicit recurrence in time, vectorized stencil in space where applicable — the same class of problems as Julia’s DifferentialEquations.jl, QuantEcon.jl, Optim.jl, and StateSpaceModels.jl demos.
- **Numerics (reusable modules):** [numerics/](../../examples/numerics/) — DiffEq (Euler decay stepper + trajectory), Optim (gradient descent 2D), QuantEcon (Bellman value iteration). Use from your code via `use numerics::diffeq`; run demo: `python3 -m einlang examples/run_numerics.ein`. See [numerics/README.md](../../examples/numerics/README.md).
- **Real applications (calibration, scenarios):** [calibration/](../../examples/calibration/) (fit decay to data, grid search over parameters); [applications/](../../examples/applications/) (savings over multiple interest-rate scenarios). Same patterns as SciML parameter estimation, Optim.jl, and [JuliaHub case studies](https://juliahub.com/case-studies) (sensitivity, scenario analysis).
- **ML / vision / speech:** MNIST CNN, quantized CNN, ViT (DeiT-tiny), Whisper-tiny. Same language and shape checks as the simulations.

For the **symbolic** side of Julia (e.g. ModelingToolkit, Symbolics.jl), Einlang does not aim to replace that; we focus on the **numerical simulation** part: you write the discrete equations, we check shapes and run them.

---

## How we showcase (learning from Julia)

We follow patterns from Julia’s docs and SciML:

- **Problem first, then code** — Each example states the equation or goal (in comments or README), then the `.ein` code. Same idea as [SciML showcases](https://docs.sciml.ai/Overview/stable/showcase/brusselator/) (e.g. Brusselator: problem setup → symbolic definition → solve).
- **One folder per Julia source** — Like [QuantEcon.jl](https://quantecon.org/quantecon-jl/) or [DiffEqDocs](https://docs.sciml.ai/DiffEqDocs/stable/), we group by ecosystem: [ode/](../../examples/ode/) ↔ DifferentialEquations.jl, [value_iteration/](../../examples/value_iteration/) ↔ QuantEcon.jl, etc.
- **Julia equivalent in comments** — Each simulation `.ein` has a 1-based Julia snippet in the file so you can compare line-by-line with [Julia’s learning resources](https://julialang.org/learning/).
- **Accuracy-tested** — Outputs are checked against NumPy or analytical references (see [tests/examples/test_simulation_accuracy.py](../../tests/examples/test_simulation_accuracy.py)).

---

## Links (Julia)

- [Julia language](https://julialang.org/) · [Learning](https://julialang.org/learning/) · [Documentation](https://docs.julialang.org/en/v1/)
- [SciML (Scientific Machine Learning)](https://docs.sciml.ai/Overview/stable/) · [Showcases](https://docs.sciml.ai/Overview/stable/showcase/brusselator/)
- [DifferentialEquations.jl](https://docs.sciml.ai/DiffEqDocs/stable/) · [ODE tutorial](https://docs.sciml.ai/DiffEqDocs/stable/getting_started/#ode_example)
- [QuantEcon.jl](https://quantecon.org/quantecon-jl/) · [Quantitative Economics with Julia](https://julia.quantecon.org/)
- [JuMP](https://jump.dev/)
- [JuliaHub case studies](https://juliahub.com/case-studies) (Aviva, Betterment, AOT, etc.)

