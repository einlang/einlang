---
layout: default
title: Release readiness
---

# Release readiness — before a wide audience

**One chance:** Visitors who have a bad first experience rarely come back. Use this checklist before promoting the repo to a wide audience.

---

## 1. First 30 seconds (try-it must work)

- [ ] **Try-it command works on a clean clone.** From repo root after `pip install -e .`:  
  `python3 -m einlang -c "let x = 1+1; print(x);"`  
  prints `2` with no error.
- [ ] **hello.ein runs.**  
  `python3 -m einlang examples/hello.ein`  
  prints the matrix multiply output. No missing env vars, no “loop limit exceeded” for this tiny example.
- [ ] **README “Try it” block is copy-paste safe.** No placeholder URLs, no typos in commands. Mention Python 3.7+ and `pip install -e .` before the `-c` command.
- [ ] **Badge is green.** [Tests](https://github.com/einlang/einlang/actions/workflows/tests.yml) workflow passes on main so the badge doesn’t scare people.

---

## 2. Docs: no broken or misleading copy-paste

- [ ] **Every code snippet in README, GETTING_STARTED, reference, MATH, SYNTAX_COMPARISON, UNSUPPORTED is valid.**  
  Recurrence: range **in bracket** (`let fib[n in 2..8] = ...`), not in `where`. See [UNSUPPORTED](UNSUPPORTED.md#9-index-range-in-where-all-cases-invalid).
- [ ] **No “where n in 2..20” in any doc** unless it’s explicitly marked as invalid.
- [ ] **Internal links work.** All `[text](path)` and `#anchor` links in docs/ and README resolve (e.g. on GitHub or your doc host).
- [ ] **“Install & run” is the single source.** Only README has the canonical install/CLI/Python API; other docs link to it.

---

## 3. Examples: first-click wins

- [ ] **Every example linked from README “Examples” table runs** from repo root with:  
  `python3 -m einlang examples/...`  
  No extra setup (or document it clearly: e.g. “run download_weights.py first” for whisper).
- [ ] **Heavy examples don’t hit the loop limit by default.**  
  Default is 5000 (config.DEFAULT_EINSTEIN_LOOP_MAX), enough for simulation demos and whisper_tiny (3000 steps).
- [ ] **examples/README.md** learning path matches what actually runs (basics → demos → mnist → mnist_quantized → deit_tiny → whisper_tiny; simulation: ode, pde_1d, wave_2d, brusselator, recurrence, finance, value_iteration, job_search, optimization, time_series).

---

## 4. Discoverability and trust

- [ ] **Repo description and topics.** On GitHub: short description + topics (e.g. `tensor`, `einstein-notation`, `dsl`, `compiler`, `python`) so people find you.
- [ ] **LICENSE** is present and correct (e.g. Apache 2.0).
- [ ] **CONTRIBUTING.md** is welcoming and points to GETTING_STARTED and doc index; no “ask permission first” for small fixes.
- [ ] **No secrets or junk in the tree.** `.gitignore` is sane; no accidental API keys or large generated files committed.

---

## 5. Python and install

- [ ] **pyproject.toml** lists all runtime deps (numpy, lark, sexpdata, typing_extensions). Optional `[test]` for pytest.
- [ ] **Editable install works:** `pip install -e .` then `python3 -m einlang --help` (or run hello.ein). If your build requires a modern setuptools, note “setuptools>=61” or “pip>=21” in README if needed.
- [ ] **CI runs on 3.9 and 3.12** (or your supported range) so the badge reflects reality.

---

## 6. What to fix right before release (summary)

| Priority | Item |
|----------|------|
| **P0** | Try-it and hello.ein work on clean clone + `pip install -e .` |
| **P0** | All doc code uses valid syntax (recurrence range in bracket, not in `where`) |
| **P0** | README “Examples” table: every listed example runs or has a one-line “run X first” |
| **P1** | Internal doc links and anchors checked (no 404) |
| **P2** | Repo description + topics; CONTRIBUTING welcoming; LICENSE present |

---

After you’ve run through this, do one final pass: clone in a fresh env, run the README try-it and two or three examples, and open GETTING_STARTED and the doc index. If that path is smooth, you’re in good shape for that one chance with a wide audience.
