
# Paper & citation

Short description and how to cite Einlang in papers or technical writing.

---

## Description (for papers / abstracts)

**Einlang** is a domain-specific language for tensor computation. Programs use **Einstein notation** as first-class syntax; indices and shapes are checked at **compile time**. The language supports where-clauses (index algebra, guards), recurrence relations (e.g. for RNNs), and a standard library of 300+ functions. Implementations include a NumPy backend (current) with a path to MLIR and native/GPU. Real models (CNN, quantized CNN, Vision Transformer, Whisper-style encoder–decoder) are written in Einlang and share the same type and shape guarantees.

**One-line:** Einlang is a language for tensor math in Einstein notation with compile-time shape checking.

---

## Repository and license

- **Repository:** [github.com/einlang/einlang](https://github.com/einlang/einlang)
- **License:** Apache 2.0 — see [LICENSE](https://github.com/einlang/einlang/blob/main/LICENSE) in the repository.

---

## How to cite

If you use Einlang in academic work, cite the repository:

```bibtex
@software{einlang,
  title = {Einlang: A language for tensor computation in Einstein notation},
  author = {Einlang contributors},
  year = {2025},
  url = {https://github.com/einlang/einlang},
  note = {Apache 2.0}
}
```

Adjust the year to the version you used. For a specific release, add `version = {...}`.

---

## Technical reference

- **Language:** [Language reference](https://github.com/einlang/einlang/blob/main/docs/reference.md)
- **Standard library:** [Standard library](https://github.com/einlang/einlang/blob/main/docs/stdlib.md)
- **Install and run:** [README — Install & run](https://github.com/einlang/einlang/blob/main/README.md#install--run)
