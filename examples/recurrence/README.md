# Recurrence (small examples)

Simple recurrences — base case(s) plus inductive step. QuantEcon-style [finite Markov chains](https://julia.quantecon.org/introduction_dynamics/finite_markov.html) (stationary distribution). For optimization (gradient descent, power iteration) see [../optimization/](../optimization/).

| File | What it does | Run |
|------|--------------|-----|
| [fibonacci.ein](fibonacci.ein) | fib[n] = fib[n-1] + fib[n-2] | `python3 -m einlang examples/recurrence/fibonacci.ein` |
| [random_walk.ein](random_walk.ein) | x[t] = x[t-1] + steps[t-1] (fixed steps) | `python3 -m einlang examples/recurrence/random_walk.ein` |
| [markov_stationary.ein](markov_stationary.ein) | ψ = ψ P (stationary distribution by power iteration) | `python3 -m einlang examples/recurrence/markov_stationary.ein` |
| [logistic.ein](logistic.ein) | Logistic map x[n] = r·x[n-1]·(1−x[n-1]) | `python3 -m einlang examples/recurrence/logistic.ein` |
