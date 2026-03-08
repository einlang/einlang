# Value function iteration (QuantEcon-style)

**Discrete Bellman** — V[k,s] = r[s] + β · Σ_{s'} P(s'|s) V[k−1,s']. Recurrence over iteration index; same idea as [QuantEcon.jl](https://quantecon.org/quantecon-jl/) / [Quantitative Economics with Julia](https://julia.quantecon.org/).

## Run

```bash
python3 -m einlang examples/value_iteration/main.ein
```

Output: `V` shape (50, 3) — value per state over 50 iterations.
