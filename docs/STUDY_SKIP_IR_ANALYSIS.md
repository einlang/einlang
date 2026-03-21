# Analysis: dumped IR for `STUDY_SKIP_CASES`

This summarizes what the **S-expr dumps** under [`study_skip_ir/`](study_skip_ir/) show, using heuristics on `serialize_ir` output (substring counts, `builtin-call "print"` shape, file sizes). Regenerate dumps with `python3 scripts/dump_study_skip_ir.py`.

## 1. Two size bands

| Band (approx.) | Cases | Typical `compile_success` |
|----------------|--------|---------------------------|
| **~0.48–0.61 MB** | `reduce_sum`, `reduce_mean`, `reduce_log_sum`, `linear`, `matmul`, `mse_loss`, `mae_loss`, `binary_cross_entropy`, `cosine_similarity` | All **True** |
| **~1.9–2.1 MB** | `softmax`, `log_softmax`, `huber_loss`, `reduce_l1`, `reduce_l2`, `reduce_sum_square`, `reduce_log_sum_exp` | Mixed (see below) |

Larger files correlate with **more `lowered-einstein` / clause text** in the serialized program (rough substring counts: ~190–270 vs ~730–1300), not only “failure vs success”.

## 2. `print(@y)` / `print(@C)` in IR: literal vs differential

For each case there is exactly **one** user-facing `builtin-call "print"` in the small test program. The **first argument** after lowering distinguishes pipelines:

| First arg to `print` | Meaning | Cases |
|----------------------|---------|--------|
| **`literal`** | Autodiff expanded `print(@…)` to `print("…")` with a symbolic tangent **string** | All **12** `compile_success: True` cases |
| **`differential`** | Autodiff **did not** finish on this program; IR is the snapshot **before** the failing `AutodiffPass` step | `reduce_l1`, `reduce_l2`, `reduce_sum_square`, `reduce_log_sum_exp` (4 cases; all `compile_success: False`) |

So the **four compile-fail** dumps are the important ones for debugging autodiff/Einstein: they still contain **`(builtin-call "print" ((differential …)))`** for the study program.

## 3. Compile success vs IR size (nuance)

- **Success + smaller IR:** mostly **single std::ml op** on a small tensor, **plus** full pipeline (Autodiff → Einstein lowering → later passes → `tree_shake`). The expanded `print` literal and compact lowered body dominate; less leftover graph than in softmax-class ops.
- **Success + larger IR (~1.9 MB):** `softmax`, `log_softmax`, `huber_loss` — **compile cleanly** but the lowered/autodiff result and/or retained stdlib surface is **much bigger** (more bindings, more `lowered-einstein` structure). This is *not* the same mechanism as the 2 MB **pre-autodiff** failures (those four are “stopped with `print(differential …)` still there”).

## 4. `matmul`

- `compile_success: True`, `print` arg is **`literal`** (symbolic tangent string was emitted).
- Serialized text may **not** contain `@y` because the program prints **`@C`** (`print(@C)` in the study source).

## 5. Heuristic “differential” counts in raw text

Rough count of substrings matching `differential` / `Differential` in the `.sexpr` file:

- **~2** for most **compile-ok** programs (typical ∂ bindings / printer usage).
- **~15** for the **four compile-fail** programs (more unfinished / duplicated differential structure around the stuck `print` path).

Use this only as a **quick fingerprint**, not a formal IR metric.

## 6. Relation to runtime failures

**Compile success** in `*.meta.txt` means the **full compiler pipeline** accepted the program and produced IR suitable for codegen. The study table in [TEST_PRINT_AT_STUDY_SKIP_COMPARE.md](TEST_PRINT_AT_STUDY_SKIP_COMPARE.md) shows many of these still **fail at execute** (`rectangular_access`, missing variables, etc.). So:

- **IR dump** = what the **compiler** produced.
- **Compare doc** = whether **NumPy runtime** can run that IR.

## 7. How to re-derive this analysis

With `.sexpr` files present:

```bash
# sizes
wc -c docs/study_skip_ir/*.sexpr

# print arg kind (first user print heuristic)
python3 -c "
import re, pathlib
for p in sorted(pathlib.Path('docs/study_skip_ir').glob('*.sexpr')):
    t = p.read_text(encoding='utf-8', errors='replace')
    kinds = sorted(set(re.findall(r'\\(builtin-call\\s+\"print\"\\s+\\(\\((\\w+)', t)))
    print(p.name, kinds)
"
```

## See also

- [study_skip_ir/README.md](study_skip_ir/README.md) — dump layout, partial IR on failure  
- [TEST_PRINT_AT_STUDY_SKIP_COMPARE.md](TEST_PRINT_AT_STUDY_SKIP_COMPARE.md) — math vs compile/exec  
- [PRINT_DIFFERENTIAL.md](PRINT_DIFFERENTIAL.md) — `print(@y)` stringification  
