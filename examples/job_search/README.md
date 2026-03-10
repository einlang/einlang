
# Job search (QuantEcon)

Dynamic programming examples from [Quantitative Economics with Julia](https://julia.quantecon.org/).

| File | Description | Run |
|------|--------------|-----|
| [mccall.ein](mccall.ein) | McCall search model: value function iteration, reservation wage | `python3 -m einlang examples/job_search/mccall.ein` |

- **Accuracy-tested:** [tests/examples/test_simulation_accuracy.py](https://github.com/einlang/einlang/blob/main/tests/examples/test_simulation_accuracy.py) compares output to a NumPy reference.
- No weights or data files; run from anywhere (e.g. repo root with `PYTHONPATH=src`).
