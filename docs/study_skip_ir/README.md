# IR dumps: `STUDY_SKIP_CASES`

S-expr snapshots from `CompilerDriver.compile` for each study-skip program.

- **`*.sexpr`:** `serialize_ir(result.ir)` when `result.ir` is not `None` (often **~2 MB** per case because stdlib is present in IR). **`*.sexpr` is gitignored** — generate locally.
- **`*.meta.txt`:** compile success flag, pytest skip reason, compiler errors (if any).
- **`INDEX.txt`:** one-line summary per case (byte size of `.sexpr` when present).

Partial IR: on compile failure (e.g. autodiff `ValueError`), the driver returns IR **as of the last completed pass** before the failure (typically pre-autodiff or pre-error pass).

Regenerate:

```bash
python3 scripts/dump_study_skip_ir.py
```

See also [TEST_PRINT_AT_STUDY_SKIP_COMPARE.md](../TEST_PRINT_AT_STUDY_SKIP_COMPARE.md).

**What the dumps mean (size bands, `print` literal vs `differential`, etc.):** [STUDY_SKIP_IR_ANALYSIS.md](../STUDY_SKIP_IR_ANALYSIS.md).
