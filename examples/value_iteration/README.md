
# Value iteration & policy iteration (QuantEcon-style)

**Value iteration:** V[k,s] = r[s] + β · Σ_{s'} P(s'|s) V[k−1,s']. Single recurrence to convergence.

**Policy iteration (Howard):** Alternate (1) policy evaluation — solve V^π for current π; (2) policy improvement — π'(s) = argmax_a [ r(s,a) + β Σ_{s'} P(s'|s,a) V(s') ]. More involved; fewer outer iterations.

Same ideas as [QuantEcon.jl](https://quantecon.org/quantecon-jl/) / [Quantitative Economics with Julia](https://julia.quantecon.org/).

## Run

```bash
python3 -m einlang examples/value_iteration/main.ein
python3 -m einlang examples/value_iteration/policy_iteration.ein
```

- `main.ein`: output `V` shape (50, 3) — value per state over 50 iterations.
- `policy_iteration.ein`: 3-state 2-action MDP; 4 outer phases, 30 inner steps each; outputs `V_final`, `policy_final`.
