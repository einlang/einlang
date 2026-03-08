# Julia demos and how Einlang maps to them

We position Einlang in the same **numerical simulation + ML** space as [Julia](https://julialang.org/). This page lists well-known Julia demos and case studies, with links, and maps them to our examples. Each simulation example includes a **Julia equivalent** (1-based) in comments in the `.ein` file and is accuracy-tested against NumPy or analytical references.

**Summary:** Julia’s *numerical* PDE/ODE use cases (explicit time-stepping, stencil + recurrence) align with Einlang’s [ode](../examples/ode/) (decay, linear, Lorenz, Lotka–Volterra), [heat](../examples/heat_animation.py), [wave_2d](../examples/wave_2d/), and [brusselator](../examples/brusselator/) — same kind of simulation; we use Einstein notation and compile-time shape checking instead of a separate symbolic layer.

---

## Julia demos and equivalents

| Julia demo / ecosystem | Domain | Link | Einlang equivalent |
|------------------------|--------|------|--------------------|
| **Brusselator PDE** (reaction–diffusion) | PDE, numerical simulation | [SciML showcase: Brusselator](https://docs.sciml.ai/Overview/stable/showcase/brusselator/) | [brusselator/](../examples/brusselator/) |
| **DifferentialEquations.jl** (ODEs, SDEs, PDEs) | Numerical integration, time-stepping | [DiffEqDocs](https://docs.sciml.ai/DiffEqDocs/stable/) | [ode/](../examples/ode/) (decay, linear, Lorenz, Lotka–Volterra); heat, wave, Brusselator (explicit PDE time-step) |
| **MethodOfLines.jl** (PDE → ODE, then solve) | PDE discretization + simulation | [MethodOfLines](https://docs.sciml.ai/MethodOfLines/stable/) | Same idea: we write the discrete update in Einstein notation ([wave_2d/main.ein](../examples/wave_2d/main.ein), [heat_animation.py](../examples/heat_animation.py)) |
| **Quantitative Economics (QuantEcon.jl)** | Dynamic programming, linear algebra | [QuantEcon.jl](https://quantecon.org/quantecon-jl/), [Quantitative Economics with Julia](https://julia.quantecon.org/) | [value_iteration/](../examples/value_iteration/) (Bellman value iteration) |
| **JuMP** (optimization) | Optimization, control (e.g. MPC) | [JuMP](https://jump.dev/) | — (no direct example yet) |
| **SciML** (scientific ML, neural ODEs) | Parameter estimation, neural DEs | [SciML](https://docs.sciml.ai/Overview/stable/) | ML examples: [mnist](../examples/mnist/), [deit_tiny](../examples/deit_tiny/), [whisper_tiny](../examples/whisper_tiny/) |
| **Aviva (Solvency II)** | Insurance, risk, numerical | [JuliaHub case study: Aviva](https://juliahub.com/industries/case-studies/aviva) | Same *style* of code: tensor math + shape safety |
| **Betterment** | Finance, projections | [JuliaHub: Betterment](https://juliahub.com/case-studies/betterment) | — |

---

## What we cover today

- **ODE/PDE numerical simulation (Julia-style):** [ode/](../examples/ode/) (exponential decay, linear A·u, Lorenz, Lotka–Volterra). Heat diffusion, 2D wave, Brusselator reaction–diffusion. Explicit recurrence in time, vectorized stencil in space where applicable — the same class of problems as Julia’s DifferentialEquations.jl demos.
- **ML / vision / speech:** MNIST CNN, quantized CNN, ViT (DeiT-tiny), Whisper-tiny. Same language and shape checks as the simulations.

For the **symbolic** side of Julia (e.g. ModelingToolkit, Symbolics.jl), Einlang does not aim to replace that; we focus on the **numerical simulation** part: you write the discrete equations, we check shapes and run them.

---

## Links (Julia)

- [Julia language](https://julialang.org/)
- [SciML (Scientific Machine Learning)](https://docs.sciml.ai/Overview/stable/)
- [DifferentialEquations.jl](https://docs.sciml.ai/DiffEqDocs/stable/)
- [QuantEcon.jl](https://quantecon.org/quantecon-jl/)
- [JuMP](https://jump.dev/)
- [JuliaHub case studies](https://juliahub.com/case-studies) (Aviva, Betterment, AOT, etc.)
