# Real applications (multi-step workflows)

Examples that mirror **real application** patterns: multiple scenarios, simple pipelines, or model + decision step — not just a single small demo. Inspired by [JuliaHub case studies](https://juliahub.com/case-studies) and production use (sensitivity analysis, scenario runs, calibration).

## Run

```bash
python3 -m einlang examples/applications/savings_scenarios.ein
```

## Files

| File | What it does |
|------|--------------|
| `savings_scenarios.ein` | Run the same savings (compound interest) model for several interest-rate scenarios; output final balance per scenario. Pattern: one model, many parameter sets. |

## Julia / real-app parallel

- [Betterment](https://juliahub.com/case-studies/betterment): finance projections over scenarios.
- [QuantEcon](https://julia.quantecon.org/): solve once per parameter or state; compare outcomes.
