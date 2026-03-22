"""Test print(@y) symbolic output: @y is the differential ∂y (tangent), not dy/dx unless you use @y/@x.
Core rules, std::math @fn, and a few Einstein/reduction cases.

Usage:
  python3 scripts/test_print_at.py              # golden string checks only
  python3 scripts/test_print_at.py --report     # markdown table: all golden + study_skip prints vs math refs (exit 1 if any golden mismatch)
  python3 scripts/test_print_at.py --study      # golden + study: flushed logs, math refs, vs print(@y)
  python3 scripts/test_print_at.py --study-only # skip goldens; only study_skip cases (same live diagnostics as --study)
  python3 scripts/test_print_at.py --dump-study-only  # markdown dump of STUDY_SKIP_CASES to stdout
  python3 scripts/test_print_at.py --help

`--report` alone prints the table and exits. With `--study` or `--study-only`, the report runs first, then the
interactive study block (exit status still reflects golden mismatches from the report).

Study mode prints each case incrementally (flush=True), shows the expected calculus (how ∂y relates to ∂x),
then the compiler/runtime result and a short heuristic compare to the reference.
"""
from pathlib import Path
from io import StringIO
import sys
from typing import Dict, List, NamedTuple, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from einlang.compiler.driver import CompilerDriver
from einlang.runtime.runtime import EinlangRuntime

REPO = Path(__file__).resolve().parent.parent

failures = []

# Programs marked pytest.mark.skip in tests/unit/test_autodiff_pass.py::_PRINT_DIFF_ML_OPS
# (same sources). Used for local diagnosis only — compile/exec may fail; output is printed.
STUDY_SKIP_CASES = [
    (
        "softmax",
        """
use std::ml;
let x = [[1.0, 2.0, 3.0]];
let y = std::ml::softmax(x);
print(@y);
""",
        "softmax autodiff not yet supported without @fn rule",
    ),
    (
        "log_softmax",
        """
use std::ml;
let x = [[1.0, 2.0, 3.0]];
let y = std::ml::log_softmax(x);
print(@y);
""",
        "log_softmax autodiff not yet supported without @fn rule",
    ),
    (
        "reduce_l2",
        """
use std::ml;
let x = [[3.0, 4.0]];
let y = std::ml::reduce_l2(x);
print(@y);
""",
        "print(@y) for multi-step inlined function: intermediate var out of scope",
    ),
    (
        "reduce_log_sum",
        """
use std::ml;
let x = [[1.0, 2.0, 3.0]];
let y = std::ml::reduce_log_sum(x);
print(@y);
""",
        "print(@y) for multi-step inlined function: intermediate var out of scope",
    ),
    (
        "reduce_log_sum_exp",
        """
use std::ml;
let x = [[1.0, 2.0, 3.0]];
let y = std::ml::reduce_log_sum_exp(x);
print(@y);
""",
        "print(@y) for multi-step inlined function: intermediate var out of scope",
    ),
    (
        "matmul",
        """
use std::ml;
let A = [[1.0, 2.0], [3.0, 4.0]];
let B = [[5.0, 6.0], [7.0, 8.0]];
let C = std::ml::matmul(A, B);
print(@C);
""",
        "matmul shape inference error in print(@y)",
    ),
    (
        "huber_loss",
        """
use std::ml;
let pred = [[1.0, 2.0, 3.0]];
let target = [[1.5, 2.5, 3.5]];
let y = std::ml::huber_loss(pred, target, 1.0);
print(@y);
""",
        "print(@y) for multi-step inlined function: intermediate var out of scope",
    ),
    (
        "cosine_similarity",
        """
use std::ml;
let a = [[1.0, 2.0, 3.0]];
let b = [[4.0, 5.0, 6.0]];
let y = std::ml::cosine_similarity(a, b);
print(@y);
""",
        "print(@y) for multi-step inlined function: intermediate var out of scope",
    ),
]

# Expected tangent @y for each study-skip case (forward-mode differential, not Jacobian).
STUDY_MATH_REFERENCE = {
    "softmax": (
        "y_i = exp(x_i)/S, S=Σexp(x_k) ⇒ @y_i = Σ_j (y_i·(δ_ij−y_j))·@x_j."
    ),
    "log_softmax": (
        "y_i = x_i−log S ⇒ @y_i = @x_i − (Σ_j softmax_j·@x_j)."
    ),
    "reduce_l2": "y = √(Σx²) ⇒ @y = Σ x_j·@x_j / y.",
    "reduce_log_sum": "y = log(Σexp(x)) ⇒ @y = Σ softmax(x)_j·@x_j.",
    "reduce_log_sum_exp": "y = log(Σexp(x)) ⇒ @y = Σ softmax(x)_j·@x_j.",
    "matmul": "C = A·B ⇒ @C = @A·B + A·@B (forward tangent of both inputs).",
    "huber_loss": "y = mean huber(p−t,δ) ⇒ @y = Σ ((p−t)·@p/N if |p−t|≤δ, else sign(p−t)·δ·@p/N).",
    "cosine_similarity": (
        "y = dot(a,b)/(‖a‖‖b‖) ⇒ @y = (@a·b + a·@b)/(‖a‖‖b‖) − y·(a·@a/‖a‖² + b·@b/‖b‖²)."
    ),
}

# One-line differential description for each golden print(@y) case.
# Each entry describes the tangent @y in terms of input tangents (@x, @a, …),
# matching forward-mode AD: @y is the directional derivative, not the Jacobian.
GOLDEN_CALCULUS: Dict[str, str] = {
    "constant": "y = const ⇒ @y = 0.",
    "identity": "y = x ⇒ @y = @x.",
    "add": "y = x + x ⇒ @y = 2·@x.",
    "sub": "y = x − c ⇒ @y = @x.",
    "product": "y = x² ⇒ @y = 2x·@x.",
    "product_two_vars": "y = a·b ⇒ @y = b·@a + a·@b.",
    "quotient": "y = a/b ⇒ @y = (b·@a − a·@b) / b².",
    "power_const": "y = x³ ⇒ @y = 3x²·@x.",
    "neg": "y = −x ⇒ @y = −@x.",
    "chain_let": "y = 2z, z = x² ⇒ @z = 2x·@x, @y = 2z·@z = 4x·x·@x.",
    "exp_scalar": "y = eˣ ⇒ @y = eˣ·@x.",
    "exp_einstein": "y[i] = eˣ⁽ⁱ⁾ ⇒ @y[i] = eˣ⁽ⁱ⁾·@x[i].",
    "sum_reduction": "y = Σ e_k ⇒ @y = Σ @e_k.",
    "softmax_quotient": "y[i] = e[i]/S, S = Σ e_k ⇒ @y[i] = (S·@e[i] − e[i]·@S) / S².",
    "if_else": "y = if x>0 then x else 0 ⇒ @y = @x if x>0, else 0.",
    "scalar_mul": "y = 2x ⇒ @y = 2·@x.",
    "compound": "y = x² + x ⇒ @y = (2x + 1)·@x.",
    "log_scalar": "y = ln x ⇒ @y = @x / x.",
    "sin_scalar": "y = sin x ⇒ @y = cos(x)·@x.",
    "cos_scalar": "y = cos x ⇒ @y = −sin(x)·@x.",
    "tan_scalar": "y = tan x ⇒ @y = @x / cos²(x).",
    "log1p_scalar": "y = ln(1+x) ⇒ @y = @x / (1+x).",
    "expm1_scalar": "y = eˣ−1 ⇒ @y = eˣ·@x.",
    "atan_scalar": "y = arctan x ⇒ @y = @x / (1+x²).",
    "asin_scalar": "y = arcsin x ⇒ @y = @x / √(1−x²).",
    "acos_scalar": "y = arccos x ⇒ @y = −@x / √(1−x²).",
    "atan2_two_vars": "y = atan2(y,x) ⇒ @y = (x·@y − y·@x) / (x²+y²).",
    "sqrt_via_pow": "y = √x = x^0.5 ⇒ @y = 0.5·x^−0.5·@x.",
    "mod_scalar": "y = x mod c ⇒ @y = @x (discontinuities ignored).",
    "quotient_chain": "y = x/(x+1) ⇒ @y = ((x+1)·@x − x·@x) / (x+1)².",
    "einstein_square": "y[i] = x[i]² ⇒ @y[i] = 2x[i]·@x[i].",
    "prod_reduction": "y = ∏ x_j ⇒ @y = Σ_j (y/x_j)·@x_j.",
    "reduce_sum": "y = Σ x ⇒ @y = Σ @x.",
    "reduce_l1": "y = Σ|x| ⇒ @y = Σ sign(x)·@x.",
    "reduce_sum_square": "y = Σ x² ⇒ @y = Σ 2x·@x.",
    "reduce_mean": "y = Σx/N ⇒ @y = Σ@x/N.",
    "linear": "y = xW^T+b ⇒ @y = W·@x + x·@W^T + @b (sum of tangent terms).",
    "mse_loss": "y = mean(p−t)² ⇒ @y = Σ 2(p−t)·@p/N + Σ 2(p−t)·(−@t)/N.",
    "mae_loss": "y = mean|p−t| ⇒ @y = Σ sign(p−t)·@p/N (via sqrt chain).",
    "binary_cross_entropy": "y = −mean(t·log p+(1−t)log(1−p)) ⇒ @y via clipped_pred chain.",
}


class PrintReportRow(NamedTuple):
    """One row: compile/exec/print vs golden and vs calculus reference."""

    label: str
    category: str  # "golden" | "study_skip"
    compile_ok: bool
    exec_ok: bool
    printed: str
    golden_expected: str
    golden_match: Optional[bool]  # None when no golden string (study-only)
    math_reference: str
    error: str
    heuristic_note: str


def _compile_exec_capture(source: str) -> Tuple[bool, bool, str, str]:
    """Returns (compile_ok, exec_ok, printed_stdout, error_message)."""
    compiler = CompilerDriver()
    result = compiler.compile(source.strip(), source_file="<test>", root_path=REPO)
    if not result.success:
        return False, False, "", _short_err(result.get_errors())
    runtime = EinlangRuntime(backend="numpy")
    buf = StringIO()
    old_stdout = sys.stdout
    sys.stdout = buf
    try:
        exec_result = runtime.execute(result)
    finally:
        sys.stdout = old_stdout
    if not exec_result.success:
        err = getattr(exec_result, "error", None) or exec_result.errors or "exec failed"
        return True, False, "", _short_err(err)
    return True, True, buf.getvalue().strip(), ""


def _heuristic_math_note(label: str, printed: str) -> str:
    """Short note: does printed output plausibly match STUDY_MATH_REFERENCE / known patterns?"""
    if not printed:
        return "(no output — cannot compare to math)"
    if "?" in printed and "=" in printed:
        return (
            "Contains '?' — printer missing IR rule; check math ref for intended ∂y."
        )
    low = printed.lower()
    if label == "softmax" and ("@" in printed) and ("sum" in low or "max" in low or "exp" in low):
        return "Plausible softmax Jacobian (sum/max/exp present)."
    if label == "reduce_sum" and "sum" in low and "@" in printed:
        return "Plausible ∂sum/∂x = 1 (sum of @x-like terms)."
    if label == "reduce_mean" and "sum" in low and ("/" in printed or "*" in printed):
        return "Plausible 1/N scaling vs sum."
    if label in ("reduce_log_sum", "reduce_log_sum_exp") and "sum" in low:
        return "Log-sum-exp ↔ softmax; look for exp/sum structure."
    if label == "matmul" and "@" in printed and ("sum" in low or "[" in printed):
        return "Plausible contraction / VJP for matmul."
    return "Compare symbols manually to math_reference column."


def collect_all_print_results() -> List[PrintReportRow]:
    """Run every golden case and every study-skip ML case; collect print output vs math refs."""
    rows: List[PrintReportRow] = []
    for label, source, expected in GOLDEN_PRINT_CASES:
        c_ok, e_ok, out, err = _compile_exec_capture(source)
        match: Optional[bool] = None
        if c_ok and e_ok:
            match = out == expected
        calc = GOLDEN_CALCULUS.get(label, "See golden_expected (symbolic tangent).")
        if match is True:
            note = _heuristic_math_note(label, out)
        elif match is False:
            note = f"golden mismatch: expected {expected!r}, got {out!r}"
        else:
            note = err if err else "(compile or exec failed)"
        rows.append(
            PrintReportRow(
                label=label,
                category="golden",
                compile_ok=c_ok,
                exec_ok=e_ok,
                printed=out,
                golden_expected=expected,
                golden_match=match,
                math_reference=calc,
                error=err,
                heuristic_note=note,
            )
        )
    for label, source, pytest_reason in STUDY_SKIP_CASES:
        c_ok, e_ok, out, err = _compile_exec_capture(source)
        ref = STUDY_MATH_REFERENCE.get(label, "")
        note = _heuristic_math_note(label, out)
        rows.append(
            PrintReportRow(
                label=label,
                category="study_skip",
                compile_ok=c_ok,
                exec_ok=e_ok,
                printed=out,
                golden_expected="",
                golden_match=None,
                math_reference=ref or pytest_reason,
                error=err,
                heuristic_note=note,
            )
        )
    return rows


def format_report_markdown(rows: List[PrintReportRow]) -> str:
    """Markdown table: all print results vs golden (if any) and calculus reference."""
    lines: List[str] = []
    lines.append("# print(@…) report: output vs math\n")
    lines.append(
        "| label | category | compile | exec | golden_match | printed (abridged) | math reference | note |"
    )
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for r in rows:
        abridged = (r.printed[:120] + "…") if len(r.printed) > 120 else r.printed
        abridged = abridged.replace("|", "\\|").replace("\n", " ⏎ ")
        gm = "" if r.golden_match is None else ("yes" if r.golden_match else "no")
        mref = (r.math_reference[:100] + "…") if len(r.math_reference) > 100 else r.math_reference
        mref = mref.replace("|", "\\|")
        note = (r.heuristic_note[:80] + "…") if len(r.heuristic_note) > 80 else r.heuristic_note
        note = note.replace("|", "\\|")
        err_cell = r.error[:60].replace("|", "\\|") if r.error else ""
        lines.append(
            f"| {r.label} | {r.category} | {r.compile_ok} | {r.exec_ok} | {gm} | `{abridged}` | {mref} | {note} |"
        )
        if r.error and not r.exec_ok:
            lines.append(f"| | | | | | *error:* `{err_cell}` | | |")
    n_golden = sum(1 for r in rows if r.category == "golden")
    n_golden_ok = sum(1 for r in rows if r.category == "golden" and r.golden_match is True)
    n_study_ok = sum(1 for r in rows if r.category == "study_skip" and r.exec_ok)
    n_study = sum(1 for r in rows if r.category == "study_skip")
    lines.append("")
    lines.append(
        f"**Summary:** golden {n_golden_ok}/{n_golden} exact string match; "
        f"study_skip executed {n_study_ok}/{n_study}."
    )
    return "\n".join(lines)


def run_report_to_stdout() -> int:
    """Print full markdown report. Exit 0 iff all golden string matches."""
    rows = collect_all_print_results()
    sys.stdout.write(format_report_markdown(rows))
    bad = [r for r in rows if r.category == "golden" and r.golden_match is False]
    return 1 if bad else 0


GOLDEN_PRINT_CASES: List[Tuple[str, str, str]] = [
    (
        "constant",
        """
let x = 3.0;
let y = 5.0;
print(@y);
""",
        "let @y = 0.0;",
    ),
    (
        "identity",
        """
let x = 3.0;
let y = x;
print(@y);
""",
        "let @y = @x;",
    ),
    (
        "add",
        """
let x = 3.0;
let y = x + x;
print(@y);
""",
        "let @y = 2.0 * @x;",
    ),
    (
        "sub",
        """
let x = 3.0;
let y = x - 1.0;
print(@y);
""",
        "let @y = @x;",
    ),
    (
        "product",
        """
let x = 3.0;
let y = x * x;
print(@y);
""",
        "let @y = 2.0 * x * @x;",
    ),
    (
        "product_two_vars",
        """
let a = 3.0;
let b = 4.0;
let y = a * b;
print(@y);
""",
        "let @y = a * @b + b * @a;",
    ),
    (
        "quotient",
        """
let a = 3.0;
let b = 4.0;
let y = a / b;
print(@y);
""",
        "let @y = (b * @a - a * @b) / b ** 2.0;",
    ),
    (
        "power_const",
        """
let x = 2.0;
let y = x ** 3.0;
print(@y);
""",
        "let @y = 3.0 * x ** 2.0 * @x;",
    ),
    (
        "neg",
        """
let x = 3.0;
let y = -x;
print(@y);
""",
        "let @y = -@x;",
    ),
    (
        "chain_let",
        """
let x = 2.0;
let z = x * x;
let y = z + z;
print(@y);
""",
        "let @z = 2.0 * x * @x;\nlet @y = 2.0 * @z;",
    ),
    (
        "exp_scalar",
        """
let x = 1.0;
let y = std::math::exp(x);
print(@y);
""",
        "let @y = {\n    let _@exp_x: f32 = exp(x) * @x;\n    _@exp_x\n};",
    ),
    (
        "exp_einstein",
        """
let x = [1.0, 2.0, 3.0];
let e[i] = std::math::exp(x[i]);
print(@e);
""",
        "let @e[i] = {\n    let _@exp: f32 = exp(x[i]) * @x[i];\n    _@exp\n};",
    ),
    (
        "sum_reduction",
        """
let x = [1.0, 2.0, 3.0];
let e[i] = std::math::exp(x[i]);
let s = sum[k](e[k]);
print(@s);
""",
        "let @e[i] = exp(x[i]) * @x[i];\nlet @s = sum[k](@e[k]);",
    ),
    (
        "softmax_quotient",
        """
let x = [1.0, 2.0, 3.0];
let e[i] = std::math::exp(x[i]);
let s = sum[k](e[k]);
let y[i] = e[i] / s;
print(@y);
""",
        "let @e[i] = exp(x[i]) * @x[i];\n"
        "let @s = sum[k](@e[k]);\n"
        "let @y[i] = (s * @e[i] - e[i] * @s) / s ** 2.0;",
    ),
    (
        "if_else",
        """
let x = 3.0;
let y = if x > 0.0 { x } else { 0.0 };
print(@y);
""",
        "let @y = if x > 0.0 { @x } else { 0.0 };",
    ),
    (
        "scalar_mul",
        """
let x = 3.0;
let y = 2.0 * x;
print(@y);
""",
        "let @y = 2.0 * @x;",
    ),
    (
        "compound",
        """
let x = 3.0;
let y = x * x + x;
print(@y);
""",
        "let @y = 2.0 * x * @x + @x;",
    ),
    (
        "log_scalar",
        """
let x = 2.0;
let y = std::math::log(x);
print(@y);
""",
        "let @y = {\n    let _@log_x: f32 = 1.0 / x * @x;\n    _@log_x\n};",
    ),
    (
        "sin_scalar",
        """
let x = 1.0;
let y = std::math::sin(x);
print(@y);
""",
        "let @y = cos(x) * @x;",
    ),
    (
        "cos_scalar",
        """
let x = 1.0;
let y = std::math::cos(x);
print(@y);
""",
        "let @y = -sin(x) * @x;",
    ),
    (
        "tan_scalar",
        """
let x = 0.5;
let y = std::math::tan(x);
print(@y);
""",
        "let @y = 1.0 / (cos(x) * cos(x)) * @x;",
    ),
    (
        "log1p_scalar",
        """
let x = 0.5;
let y = std::math::log1p(x);
print(@y);
""",
        "let @y = 1.0 / (1.0 + x) * @x;",
    ),
    (
        "expm1_scalar",
        """
let x = 0.5;
let y = std::math::expm1(x);
print(@y);
""",
        "let @y = exp(x) * @x;",
    ),
    (
        "atan_scalar",
        """
let x = 0.5;
let y = std::math::atan(x);
print(@y);
""",
        "let @y = 1.0 / (1.0 + x * x) * @x;",
    ),
    (
        "asin_scalar",
        """
let x = 0.5;
let y = std::math::asin(x);
print(@y);
""",
        "let @y = 1.0 / (1.0 - x * x) ** 0.5 * @x;",
    ),
    (
        "acos_scalar",
        """
let x = 0.5;
let y = std::math::acos(x);
print(@y);
""",
        "let @y = -1.0 / (1.0 - x * x) ** 0.5 * @x;",
    ),
    (
        "atan2_two_vars",
        """
let y = 1.0;
let x = 2.0;
let z = std::math::atan2(y, x);
print(@z);
""",
        "let @z = x / (x * x + y * y) * @y + -y / (x * x + y * y) * @x;",
    ),
    (
        "sqrt_via_pow",
        """
let x = 4.0;
let y = x ** 0.5;
print(@y);
""",
        "let @y = 0.5 * x ** -0.5 * @x;",
    ),
    (
        "mod_scalar",
        """
let x = 7.0;
let y = x % 3.0;
print(@y);
""",
        "let @y = @x;",
    ),
    (
        "quotient_chain",
        """
let x = 3.0;
let y = x / (x + 1.0);
print(@y);
""",
        "let @y = ((x + 1.0) * @x - x * @x) / (x + 1.0) ** 2.0;",
    ),
    (
        "einstein_square",
        """
let x = [1.0, 2.0, 3.0];
let t[i] = x[i] * x[i];
print(@t);
""",
        "let @t[i] = 2.0 * x[i] * @x[i];",
    ),
    (
        "prod_reduction",
        """
let x = [1.0, 2.0, 3.0];
let p = prod[j](x[j]);
print(@p);
""",
        "let @p = prod[j](x[j]) / x[j] * @x[j];",
    ),
    (
        "reduce_sum",
        """
use std::ml;
let x = [[1.0, 2.0, 3.0]];
let y = std::ml::reduce_sum(x);
print(@y);
""",
        "let @y = {\n    let _@reduce_sum_x: [f32; ?] = {\n        let _@result[batch.0] = sum[j](@x[batch.0, j]);\n        _@result\n    };\n    _@reduce_sum_x\n};",
    ),
    (
        "reduce_l1",
        """
use std::ml;
let x = [[1.0, -2.0, 3.0]];
let y = std::ml::reduce_l1(x);
print(@y);
""",
        "let @y = {\n    let _@reduce_l1_x: [f32; *] = {\n        let _@result[batch.0] = sum[j](if x[batch.0, j] as f32 >= 0.0 { @x[batch.0, j] } else { -@x[batch.0, j] });\n        _@result\n    };\n    _@reduce_l1_x\n};",
    ),
    (
        "reduce_sum_square",
        """
use std::ml;
let x = [[1.0, 2.0, 3.0]];
let y = std::ml::reduce_sum_square(x);
print(@y);
""",
        "let @y = {\n    let _@reduce_sum_square_x: [f32; *] = {\n        let _@result[batch.0] = sum[j](2.0 * x[batch.0, j] * @x[batch.0, j]);\n        _@result\n    };\n    _@reduce_sum_square_x\n};",
    ),
    (
        "reduce_mean",
        """
use std::ml;
let x = [[1.0, 2.0, 3.0]];
let y = std::ml::reduce_mean(x);
print(@y);
""",
        "let @y = {\n    let _@reduce_mean_x: [f32; ?] = {\n        let _@sum_val[batch.0] = sum[j](@x[batch.0, j]);\n        let _@mean[batch.0] = _@sum_val[batch.0] / len(x[0]) as f32;\n        _@mean\n    };\n    _@reduce_mean_x\n};",
    ),
    (
        "linear",
        """
use std::ml;
let x = [[1.0, 2.0]];
let W = [[0.5, 0.3], [0.2, 0.4]];
let b = [0.1, 0.2];
let y = std::ml::linear(x, W, b);
print(@y);
""",
        "let @y = {\n    let _@linear_call: [f32; ?, ?] = {\n        let _@output[batch.0, j] = sum[k](x[batch.0, k] * 0.0[j, k] + W[j, k] * @x[batch.0, k]) + 0.0[j];\n        let _@output[batch.0, j] = sum[k](x[batch.0, k] * @W[j, k] + W[j, k] * 0.0[batch.0, k]) + 0.0[j];\n        let _@output[batch.0, j] = sum[k](x[batch.0, k] * 0.0[j, k] + W[j, k] * 0.0[batch.0, k]) + @b[j];\n        _@output + _@output + _@output\n    };\n    _@linear_call\n};",
    ),
    (
        "mse_loss",
        """
use std::ml;
let pred = [[1.0, 2.0, 3.0]];
let target = [[1.5, 2.5, 3.5]];
let y = std::ml::mse_loss(pred, target);
print(@y);
""",
        "let @y = {\n    let _@mse_loss_call: [f32; ?] = {\n        let _@loss[batch.0] = sum[j](2.0 * (pred[batch.0, j] - target[batch.0, j]) * (@pred[batch.0, j] - 0.0[batch.0, j])) / len(pred[0]) as f32;\n        let _@loss[batch.0] = sum[j](2.0 * (pred[batch.0, j] - target[batch.0, j]) * (0.0[batch.0, j] - @target[batch.0, j])) / len(pred[0]) as f32;\n        _@loss + _@loss\n    };\n    _@mse_loss_call\n};",
    ),
    (
        "mae_loss",
        """
use std::ml;
let pred = [[1.0, 2.0, 3.0]];
let target = [[1.5, 2.5, 3.5]];
let y = std::ml::mae_loss(pred, target);
print(@y);
""",
        "let @y = {\n    let _@mae_loss_call: [f32; ?] = {\n        let _@loss[batch.0] = sum[j](0.5 * ((pred[batch.0, j] - target[batch.0, j]) ** 2.0) ** -0.5 * 2.0 * (pred[batch.0, j] - target[batch.0, j]) * (@pred[batch.0, j] - 0.0[batch.0, j])) / len(pred[0]) as f32;\n        let _@loss[batch.0] = sum[j](0.5 * ((pred[batch.0, j] - target[batch.0, j]) ** 2.0) ** -0.5 * 2.0 * (pred[batch.0, j] - target[batch.0, j]) * (0.0[batch.0, j] - @target[batch.0, j])) / len(pred[0]) as f32;\n        _@loss + _@loss\n    };\n    _@mae_loss_call\n};",
    ),
    (
        "binary_cross_entropy",
        """
use std::ml;
let pred = [[0.8, 0.3, 0.9]];
let target = [[1.0, 0.0, 1.0]];
let y = std::ml::binary_cross_entropy(pred, target);
print(@y);
""",
        "let @y = {\n    let _@binary_cross_entropy_call: [f32; ?] = {\n        let _@clipped_pred[batch.0, j] = if pred[batch.0, j] < 1e-07 { 0.0 } else { if pred[batch.0, j] > 1.0 - 1e-07 { 0.0 - 0.0 } else { @pred[batch.0, j] } };\n        let _@loss[batch.0] = -sum[j](target[batch.0, j] * 1.0 / if pred[batch.0, j] < 1e-07 { 1e-07 } else { if pred[batch.0, j] > 1.0 - 1e-07 { 1.0 - 1e-07 } else { pred[batch.0, j] } }[batch.0, j] * 0.0[batch.0, j] + ln(if pred[batch.0, j] < 1e-07 { 1e-07 } else { if pred[batch.0, j] > 1.0 - 1e-07 { 1.0 - 1e-07 } else { pred[batch.0, j] } }[batch.0, j]) * 0.0[batch.0, j] + (1.0 - target[batch.0, j]) * 1.0 / (1.0 - if pred[batch.0, j] < 1e-07 { 1e-07 } else { if pred[batch.0, j] > 1.0 - 1e-07 { 1.0 - 1e-07 } else { pred[batch.0, j] } }[batch.0, j]) * (0.0 - 0.0[batch.0, j]) + ln(1.0 - if pred[batch.0, j] < 1e-07 { 1e-07 } else { if pred[batch.0, j] > 1.0 - 1e-07 { 1.0 - 1e-07 } else { pred[batch.0, j] } }[batch.0, j]) * (0.0 - 0.0[batch.0, j]));\n        let _@clipped_pred[batch.0, j] = if pred[batch.0, j] < 1e-07 { 0.0 } else { if pred[batch.0, j] > 1.0 - 1e-07 { 0.0 - 0.0 } else { 0.0[batch.0, j] } };\n        let _@loss[batch.0] = -sum[j](target[batch.0, j] * 1.0 / if pred[batch.0, j] < 1e-07 { 1e-07 } else { if pred[batch.0, j] > 1.0 - 1e-07 { 1.0 - 1e-07 } else { pred[batch.0, j] } }[batch.0, j] * 0.0[batch.0, j] + ln(if pred[batch.0, j] < 1e-07 { 1e-07 } else { if pred[batch.0, j] > 1.0 - 1e-07 { 1.0 - 1e-07 } else { pred[batch.0, j] } }[batch.0, j]) * @target[batch.0, j] + (1.0 - target[batch.0, j]) * 1.0 / (1.0 - if pred[batch.0, j] < 1e-07 { 1e-07 } else { if pred[batch.0, j] > 1.0 - 1e-07 { 1.0 - 1e-07 } else { pred[batch.0, j] } }[batch.0, j]) * (0.0 - 0.0[batch.0, j]) + ln(1.0 - if pred[batch.0, j] < 1e-07 { 1e-07 } else { if pred[batch.0, j] > 1.0 - 1e-07 { 1.0 - 1e-07 } else { pred[batch.0, j] } }[batch.0, j]) * (0.0 - @target[batch.0, j]));\n        _@loss + _@loss\n    };\n    _@binary_cross_entropy_call\n};",
    ),
]


def dump_study_skip_cases_markdown() -> str:
    """Full markdown listing: label, pytest skip reason, math ref, source (STUDY_SKIP_CASES)."""
    parts: List[str] = []
    parts.append("# `test_print_at.py` — study-only cases (`STUDY_SKIP_CASES`)\n\n")
    parts.append(
        "Same programs as `pytest.mark.skip` entries in "
        "`tests/unit/test_autodiff_pass.py::_PRINT_DIFF_ML_OPS`. "
        "Run `python3 scripts/test_print_at.py --study-only` to attempt compile+exec (diagnostic; no exit failure).\n\n"
        "---\n\n"
    )
    for label, source, reason in STUDY_SKIP_CASES:
        parts.append(f"## `{label}`\n\n")
        parts.append(f"- **Pytest skip reason:** {reason}\n")
        ref = STUDY_MATH_REFERENCE.get(label, "")
        if ref:
            parts.append(f"- **Math reference:** {ref}\n")
        parts.append("\n```\n" + source.strip() + "\n```\n\n")
        parts.append("---\n\n")
    return "".join(parts)


def _sp(*args, **kwargs) -> None:
    """Print with flush so study output appears immediately (long compiles per case)."""
    kwargs.setdefault("flush", True)
    print(*args, **kwargs)


def run(label, source, expected):
    compiler = CompilerDriver()
    result = compiler.compile(source.strip(), source_file="<test>", root_path=REPO)
    if not result.success:
        failures.append((label, f"COMPILE FAIL: {result.get_errors()}"))
        print(f"  FAIL  {label}: COMPILE FAIL")
        return
    runtime = EinlangRuntime(backend="numpy")
    buf = StringIO()
    old_stdout = sys.stdout
    sys.stdout = buf
    try:
        exec_result = runtime.execute(result)
    finally:
        sys.stdout = old_stdout
    if not exec_result.success:
        failures.append((label, f"EXEC FAIL: {getattr(exec_result, 'error', None) or exec_result.errors}"))
        print(f"  FAIL  {label}: EXEC FAIL")
        return
    printed = buf.getvalue().strip()
    if printed == expected:
        print(f"  OK    {label}: {printed}")
    else:
        failures.append((label, f"got '{printed}', expected '{expected}'"))
        print(f"  FAIL  {label}: got '{printed}', expected '{expected}'")


def _short_err(obj, limit: int = 600) -> str:
    if obj is None:
        return ""
    s = str(obj)
    if len(s) > limit:
        return s[:limit] + "..."
    return s


def _study_math_compare_line(label: str, printed: str) -> None:
    """Heuristic note: does the printed string plausibly reflect the reference?"""
    _sp("  -> compare: " + _heuristic_math_note(label, printed))


def study_skipped_pytest_cases() -> None:
    """Try compile + execute for each pytest-skipped print(@y) ML case; never affects exit code."""
    _sp()
    _sp("=" * 72)
    _sp("STUDY: pytest-skipped cases (tests/unit/test_autodiff_pass.py::_PRINT_DIFF_ML_OPS)")
    _sp("      Each block prints math reference first, then live status (flush).")
    _sp("      (diagnostic only — failures here do not fail this script)")
    _sp("=" * 72)
    for label, source, pytest_reason in STUDY_SKIP_CASES:
        _sp()
        _sp(f"  [{label}]")
        _sp(f"  math (reference): {STUDY_MATH_REFERENCE.get(label, '(no reference)')}")
        _sp(f"  pytest skip reason: {pytest_reason}")
        _sp("  -> compiling …")
        compiler = CompilerDriver()
        result = compiler.compile(source.strip(), source_file="<test>", root_path=REPO)
        if not result.success:
            _sp("  -> COMPILE: fail")
            _sp(f"     {_short_err(result.get_errors())}")
            continue
        _sp("  -> COMPILE: ok; executing …")
        runtime = EinlangRuntime(backend="numpy")
        buf = StringIO()
        old_stdout = sys.stdout
        sys.stdout = buf
        try:
            exec_result = runtime.execute(result)
        finally:
            sys.stdout = old_stdout
        if not exec_result.success:
            _sp("  -> EXEC: fail")
            _sp(f"     {_short_err(getattr(exec_result, 'error', None) or exec_result.errors)}")
            continue
        out = buf.getvalue().strip()
        _sp(f"  -> EXEC: ok")
        _sp(f"  -> print(@y) output: {out!r}")
        _study_math_compare_line(label, out)


def golden_tests() -> None:
    failures.clear()
    for label, source, expected in GOLDEN_PRINT_CASES:
        run(label, source, expected)


def main() -> None:
    argv = sys.argv[1:]
    if "-h" in argv or "--help" in argv:
        print(__doc__)
        sys.exit(0)
    if "--dump-study-only" in argv:
        sys.stdout.write(dump_study_skip_cases_markdown())
        sys.exit(0)

    study_only = "--study-only" in argv
    do_study = "--study" in argv or study_only
    report_code = 0
    if "--report" in argv:
        report_code = run_report_to_stdout()
        print()
        if not do_study:
            sys.exit(report_code)

    if not study_only:
        golden_tests()
        print()
        if failures:
            print(f"{len(failures)} FAILED:")
            for label, msg in failures:
                print(f"  {label}: {msg}")
            sys.exit(1)
        print("All passed.")
    else:
        print("(golden tests skipped: --study-only)")
        print()

    if do_study:
        study_skipped_pytest_cases()

    sys.exit(report_code)


if __name__ == "__main__":
    main()
