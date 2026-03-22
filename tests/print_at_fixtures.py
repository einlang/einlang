"""Goldens and helpers for ``print(@…)`` stdout (``scripts/test_print_at.py`` + pytest)."""

from __future__ import annotations

from io import StringIO
from pathlib import Path
import re
import sys
from typing import Dict, List, NamedTuple, Optional, Tuple

from einlang.compiler.driver import CompilerDriver
from einlang.runtime.runtime import EinlangRuntime

REPO_ROOT = Path(__file__).resolve().parent.parent


def short_err_print_at(obj: object, limit: int = 600) -> str:
    if obj is None:
        return ""
    s = str(obj)
    if len(s) > limit:
        return s[:limit] + "..."
    return s


def compile_exec_capture_print_at(source: str) -> Tuple[bool, bool, str, str]:
    """Compile ``source``, run with numpy backend, return ``(compile_ok, exec_ok, stdout.strip(), err)``."""
    compiler = CompilerDriver()
    result = compiler.compile(source.strip(), source_file="<test>", root_path=REPO_ROOT)
    if not result.success:
        return False, False, "", short_err_print_at(result.get_errors())
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
        return True, False, "", short_err_print_at(err)
    return True, True, buf.getvalue().strip(), ""


# Captured ``print(@y)`` for ``huber_loss`` golden (multi-input callee: three spine copies, final ``_@loss + …``).
_HUBER_LOSS_PRINT_AT_GOLDEN = """let @y = {
    let _@huber_loss_call: [f32; ?] = {
        let n = len(pred[0]) as f32;
        let diff[batch.0, j] = pred[batch.0, j] - target[batch.0, j];
        let abs_diff[batch.0, j] = abs(diff[batch.0, j]);
        let _@diff[batch.0, j] = @pred[batch.0, j] - 0.0[batch.0, j];
        let _@abs_diff[batch.0, j] = if diff[batch.0, j] as f32 >= 0.0 { 0.0[batch.0, j] } else { -0.0[batch.0, j] };
        let _@huber_elem[batch.0, j] = if abs_diff[batch.0, j] <= 1.0 { 0.5 * diff[batch.0, j] * _@diff[batch.0, j] + diff[batch.0, j] * (0.5 * _@diff[batch.0, j] + diff[batch.0, j] * 0.0) } else { 1.0 * (_@abs_diff[batch.0, j] - (0.5 * 0.0 + 1.0 * 0.0)) + (abs_diff[batch.0, j] - 0.5 * 1.0) * 0.0 };
        let _@loss[batch.0] = sum[j](_@huber_elem[batch.0, j]) / n;
        let _@diff[batch.0, j] = 0.0[batch.0, j] - @target[batch.0, j];
        let _@abs_diff[batch.0, j] = if diff[batch.0, j] as f32 >= 0.0 { 0.0[batch.0, j] } else { -0.0[batch.0, j] };
        let _@huber_elem[batch.0, j] = if abs_diff[batch.0, j] <= 1.0 { 0.5 * diff[batch.0, j] * _@diff[batch.0, j] + diff[batch.0, j] * (0.5 * _@diff[batch.0, j] + diff[batch.0, j] * 0.0) } else { 1.0 * (_@abs_diff[batch.0, j] - (0.5 * 0.0 + 1.0 * 0.0)) + (abs_diff[batch.0, j] - 0.5 * 1.0) * 0.0 };
        let _@loss[batch.0] = sum[j](_@huber_elem[batch.0, j]) / n;
        let _@diff[batch.0, j] = 0.0[batch.0, j] - 0.0[batch.0, j];
        let _@abs_diff[batch.0, j] = if diff[batch.0, j] as f32 >= 0.0 { 0.0[batch.0, j] } else { -0.0[batch.0, j] };
        let _@huber_elem[batch.0, j] = if abs_diff[batch.0, j] <= 1.0 { 0.5 * diff[batch.0, j] * _@diff[batch.0, j] + diff[batch.0, j] * (0.5 * _@diff[batch.0, j] + diff[batch.0, j] * 0.0) } else { 1.0 * (_@abs_diff[batch.0, j] - (0.5 * 0.0 + 1.0 * 0.0)) + (abs_diff[batch.0, j] - 0.5 * 1.0) * 0.0 };
        let _@loss[batch.0] = sum[j](_@huber_elem[batch.0, j]) / n;
        _@loss + _@loss + _@loss
    };
    _@huber_loss_call
};"""

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
    "softmax": "y_i = exp(x_i)/S, S=Σexp(x_k) ⇒ @y_i = Σ_j (y_i·(δ_ij−y_j))·@x_j.",
    "log_softmax": "y_i = x_i−log S ⇒ @y_i = @x_i − (Σ_j softmax_j·@x_j).",
    "reduce_l2": "y = √(Σx²) ⇒ @y = Σ x_j·@x_j / y.",
    "reduce_log_sum": "y = log(Σ x) (not log-sum-exp) ⇒ @y = (Σ @x) / (Σ x).",
    "reduce_log_sum_exp": "y = log(Σexp(x)) ⇒ @y = Σ softmax(x)_j·@x_j.",
    "cosine_similarity": "y = dot(a,b)/(‖a‖‖b‖) ⇒ @y = (@a·b + a·@b)/(‖a‖‖b‖) − y·(a·@a/‖a‖² + b·@b/‖b‖²).",
    "matmul": "C = A·B ⇒ @C = @A·B + A·@B (2×2 matmul; batched tensors use batch_matmul).",
    # print(@y) for y = batch_matmul(A,B): symbolic @y is the JVP in @A,@B (same as matmul per batch).
    "batch_matmul": (
        "Primal C[b,i,j]=Σ_k A[b,i,k]B[b,k,j]. "
        "print(@C) (forward-mode tangent): @C[b,i,j]=Σ_k (A[b,i,k]·@B[b,k,j] + B[b,k,j]·@A[b,i,k]); "
        "commutative reorder of the two product-rule terms matches golden "
        "sum[k](A[batch.0,i,k]*@B[batch.0,k,j] + B[batch.0,k,j]*@A[batch.0,i,k]) "
        "with batch.0 ≡ b. Outer let @C = { _@batch_matmul_call: … { _@result[…]; _@result }; … } is printer framing only."
    ),
    "huber_loss": (
        "Primal L[b]=(1/n)Σ_j h(diff_j), diff=pred−target, h=0.5·diff² if |diff|≤δ else δ(|diff|−½δ). "
        "Forward-mode print(@y): @L = (∂L/∂pred)[@pred] + (∂L/∂target)[@target]; implementation repeats the "
        "loss spine with tangents (@pred,0), (0,−@target), (0,0) and sums _@loss+_@loss+_@loss. "
        "When |diff|≤δ: ∂h/∂pred=diff, ∂h/∂target=−diff; blocks give mean(diff⊙@pred) and mean(diff⊙(−@target)) "
        "= mean((−diff)⊙@target), i.e. ∂h/∂pred·@pred + ∂h/∂target·@target per element — matches 0.5·diff² chain rule."
    ),
}


class PrintReportRow(NamedTuple):
    """One row: compile/exec/print vs golden and vs calculus reference."""

    label: str
    compile_ok: bool
    exec_ok: bool
    printed: str
    golden_expected: str
    golden_match: bool
    math_reference: str
    error: str
    heuristic_note: str


# Strip ``[… ; … ? …]`` shape placeholders so ``?`` in ``[f32; ?, ?]`` is not treated as IR holes.
_SHAPE_QUESTION_SLOT = re.compile(r"\[[^\]]*;\s*[^\]]*\?[^\]]*\]")


def _printed_stray_question_mark(printed: str) -> bool:
    return "?" in _SHAPE_QUESTION_SLOT.sub("", printed)


def _structural_math_fit(label: str, printed: str) -> str:
    """After exec ok: one-line structural read vs expected tangent shape (not a proof)."""
    low = printed.lower()
    if "_@" not in printed and "@y" not in printed and "@c" not in low:
        return "Unexpected: no _@ / @y style tangent bindings in capture."

    if _printed_stray_question_mark(printed):
        return "Stray '?' outside ``[type; … ? …]`` shapes — inspect printer vs math ref."

    if label == "softmax":
        if "exp" in low and "sum" in low and "/" in printed and "@x" in printed:
            return "Fit: exp + sum + quotient + @x — consistent with softmax JVP (see math ref)."
        return "Check softmax: expect exp, sums, normalization fraction, @x (see math ref)."

    if label == "log_softmax":
        if "@x" in printed and ("ln(" in low or "log" in low) and "sum" in low:
            return "Fit: @x, log-mass via sum/ln — consistent with log_softmax ref."
        return "Check log_softmax: @x minus weighted log-sum of exp (see math ref)."

    if label == "reduce_l2":
        if "sum" in low and "@x" in printed and ("** -0.5" in printed or "**-0.5" in printed.replace(" ", "")):
            return "Fit: sum of squares + sqrt chain (**-0.5) + @x — matches @y = Σx·@x / ‖x‖."
        return "Check L2: expect sum x², sqrt/ power -0.5 chain, @x (see math ref)."

    if label == "reduce_log_sum":
        if ("ln(" in low or "log(" in low) and "@x" in printed and ("1.0 /" in printed or "/ sum" in low):
            return "Fit: ln(sum x) chain — 1/sum · Σ@x visible; matches y=log(Σx) ref."
        return "Check reduce_log_sum: ln(sum x) ⇒ @y = (Σ@x)/(Σx); not log-sum-exp (see ref)."

    if label == "reduce_log_sum_exp":
        if "exp" in low and "sum" in low and ("max" in low or "shifted" in low) and "@x" in printed:
            return "Fit: max-shift + exp + sum + ln — stable log-sum-exp; softmax weights in @y (see ref)."
        return "Check log-sum-exp: max, shifted, exp, sum, ln + @x (see math ref)."

    if label == "matmul":
        if "sum" in low and "@" in printed:
            return "Fit: sum contraction + @ bindings — consistent with @C = @A·B + A·@B ref."
        return "Expect sum over shared index and tangents on A and B (see math ref)."

    if label == "batch_matmul":
        if "sum" in low and "@a" in low and "@b" in low and "batch" in low:
            return (
                "Fit: print(@C) shows sum[k] over k with A*@B + B*@A on [batch.0,…] — "
                "matches @C[b,i,j]=Σ_k(A[b,i,k]@B[b,k,j]+B[b,k,j]@A[b,i,k]) (GOLDEN_CALCULUS)."
            )
        if "sum" in low and "@A" in printed and "@B" in printed:
            return (
                "Fit: print(@C) sum[k] + @A/@B — same JVP as 2D matmul per batch (see GOLDEN_CALCULUS batch_matmul)."
            )
        return (
            "Check print(@C): batch axis on A,B; inner sum[k](A*@B+B*@A) vs math ref "
            "@C[b,i,j]=Σ_k(A[b,i,k]@B[b,k,j]+B[b,k,j]@A[b,i,k])."
        )

    if label == "huber_loss":
        if "@pred" in printed and "huber" in low and "if " in low:
            return "Fit: piecewise huber + @pred/@target style — compare branches to huber ref."
        return "Check huber: if/else on |p−t|, @pred terms, mean / n (see math ref)."

    if label == "cosine_similarity":
        if "dot_product" in low and "norm" in low and "/ " in printed and ("@a" in printed or "@b" in printed):
            return "Fit: dot + norms + quotient **2 — consistent with cosine quotient rule ref."
        return "Check cosine: dot, ‖a‖‖b‖, quotient rule on @a/@b (see math ref)."

    # Golden / generic ML prints
    if label == "reduce_sum":
        if "sum" in low and "@x" in printed:
            return "Fit: sum over @x — matches @y = Σ@x."
    if label == "reduce_mean":
        if "sum" in low and "@x" in printed and ("/" in printed or "len(" in low):
            return "Fit: sum @x with 1/N style — matches mean ref."
    if label == "reduce_l1":
        if "sum" in low and "@x" in printed:
            return "Fit: sum with sign/abs branch on @x — matches L1 ref."
    if label == "reduce_sum_square":
        if "2.0" in printed and "@x" in printed and "sum" in low:
            return "Fit: Σ 2x·@x pattern — matches ref."
    if label == "linear":
        if "@w" in low or "@W" in printed:
            if "@x" in printed or "@b" in printed:
                return "Fit: @output with @W, @x, @b — matches linear JVP ref."
    if label == "mse_loss":
        if "@pred" in printed or "@target" in printed:
            if "2.0" in printed and "sum" in low:
                return "Fit: 2(p−t)·(@p−@t) in sum — matches MSE ref."
    if label == "mae_loss":
        if ("@pred" in printed or "@target" in printed) and "sum" in low:
            return "Fit: sqrt/abs chain on (p−t) — compare to MAE ref."
    if label == "binary_cross_entropy":
        if "clipped_pred" in low and "@pred" in printed:
            return "Fit: clipped_pred + ln + @pred/@target — matches BCE chain ref."

    return "Exec produced tangent text — line-by-line compare to math_reference / GOLDEN_CALCULUS."


def study_math_alignment_note(
    label: str,
    printed: str,
    math_ref: str,
    *,
    compile_ok: bool,
    exec_ok: bool,
    golden_exact_match: Optional[bool] = None,
) -> str:
    """Human note: exec gate first, then structural fit vs calculus line."""
    if not compile_ok:
        return "no comparison: compile failed."
    if not exec_ok:
        return "no comparison: exec failed."
    stripped = (printed or "").strip()
    if not stripped:
        return "exec ok but no stdout — nothing to compare to math."

    fit = _structural_math_fit(label, stripped)
    if golden_exact_match is True:
        return fit + " (golden string exact)"
    if golden_exact_match is False:
        return "golden text differs — " + fit
    return fit


def collect_all_print_results() -> List[PrintReportRow]:
    """Run every golden case; collect print output vs math refs."""
    rows: List[PrintReportRow] = []
    for label, source, expected in GOLDEN_PRINT_CASES:
        c_ok, e_ok, out, err = compile_exec_capture_print_at(source)
        match = (out == expected) if (c_ok and e_ok) else False
        calc = GOLDEN_CALCULUS.get(label, "See golden_expected (symbolic tangent).")
        if match:
            note = study_math_alignment_note(
                label,
                out,
                calc,
                compile_ok=c_ok,
                exec_ok=e_ok,
                golden_exact_match=True,
            )
        elif c_ok and e_ok:
            note = f"golden mismatch: expected {expected!r}, got {out!r}"
        else:
            note = study_math_alignment_note(
                label, out, calc, compile_ok=c_ok, exec_ok=e_ok
            )
        rows.append(
            PrintReportRow(
                label=label,
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
    return rows


def format_report_markdown(rows: List[PrintReportRow]) -> str:
    """Markdown table: golden print results vs expected string and calculus reference."""
    lines: List[str] = []
    lines.append("# print(@…) report: output vs math\n")
    lines.append(
        "| label | compile | exec | golden_match | printed (abridged) | math reference | note |"
    )
    lines.append("| --- | --- | --- | --- | --- | --- | --- |")
    for r in rows:
        abridged = (r.printed[:120] + "…") if len(r.printed) > 120 else r.printed
        abridged = abridged.replace("|", "\\|").replace("\n", " ⏎ ")
        gm = "yes" if r.golden_match else "no"
        mref = (r.math_reference[:100] + "…") if len(r.math_reference) > 100 else r.math_reference
        mref = mref.replace("|", "\\|")
        note = (r.heuristic_note[:80] + "…") if len(r.heuristic_note) > 80 else r.heuristic_note
        note = note.replace("|", "\\|")
        err_cell = r.error[:60].replace("|", "\\|") if r.error else ""
        lines.append(
            f"| {r.label} | {r.compile_ok} | {r.exec_ok} | {gm} | `{abridged}` | {mref} | {note} |"
        )
        if r.error and not r.exec_ok:
            lines.append(f"| | | | | *error:* `{err_cell}` | | |")
    n_ok = sum(1 for r in rows if r.golden_match)
    lines.append("")
    lines.append(f"**Summary:** {n_ok}/{len(rows)} golden stdout strings match exactly.")
    return "\n".join(lines)


def run_report_to_stdout() -> int:
    """Print full markdown report. Exit 0 iff all golden string matches."""
    rows = collect_all_print_results()
    sys.stdout.write(format_report_markdown(rows))
    bad = [r for r in rows if not r.golden_match]
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
        "let @y = exp(x) * @x;",
    ),
    (
        "exp_einstein",
        """
let x = [1.0, 2.0, 3.0];
let e[i] = std::math::exp(x[i]);
print(@e);
""",
        "let @e[i] = exp(x[i]) * @x[i];",
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
        "let @y = 1.0 / x * @x;",
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
        "let @y = {\n    let _@reduce_mean_x: [f32; ?] = {\n        let count = len(x[0]) as f32;\n        let _@sum_val[batch.0] = sum[j](@x[batch.0, j]);\n        let _@mean[batch.0] = _@sum_val[batch.0] / count;\n        _@mean\n    };\n    _@reduce_mean_x\n};",
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
        "let @y = {\n    let _@linear_call: [f32; ?, ?] = {\n        let _@output[batch.0, j] = sum[k](x[batch.0, k] * @W[j, k] + W[j, k] * @x[batch.0, k]) + @b[j];\n        _@output\n    };\n    _@linear_call\n};",
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
        "let @y = {\n    let _@mse_loss_call: [f32; ?] = {\n        let n = len(pred[0]) as f32;\n        let _@loss[batch.0] = sum[j](2.0 * (pred[batch.0, j] - target[batch.0, j]) * (@pred[batch.0, j] - @target[batch.0, j])) / n;\n        _@loss\n    };\n    _@mse_loss_call\n};",
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
        "let @y = {\n    let _@mae_loss_call: [f32; ?] = {\n        let n = len(pred[0]) as f32;\n        let _@loss[batch.0] = sum[j](0.5 * ((pred[batch.0, j] - target[batch.0, j]) ** 2.0) ** -0.5 * 2.0 * (pred[batch.0, j] - target[batch.0, j]) * (@pred[batch.0, j] - @target[batch.0, j])) / n;\n        _@loss\n    };\n    _@mae_loss_call\n};",
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
        _HUBER_LOSS_PRINT_AT_GOLDEN,
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
        "let @y = {\n    let _@binary_cross_entropy_call: [f32; ?] = {\n        let eps = 1e-07;\n        let clipped_pred[batch.0, j] = if pred[batch.0, j] < eps { eps } else { if pred[batch.0, j] > 1.0 - eps { 1.0 - eps } else { pred[batch.0, j] } };\n        let _@clipped_pred[batch.0, j] = if pred[batch.0, j] < eps { 0.0 } else { if pred[batch.0, j] > 1.0 - eps { 0.0 - 0.0 } else { @pred[batch.0, j] } };\n        let _@loss[batch.0] = -sum[j](target[batch.0, j] * 1.0 / clipped_pred[batch.0, j] * _@clipped_pred[batch.0, j] + ln(clipped_pred[batch.0, j]) * @target[batch.0, j] + (1.0 - target[batch.0, j]) * 1.0 / (1.0 - clipped_pred[batch.0, j]) * (0.0 - _@clipped_pred[batch.0, j]) + ln(1.0 - clipped_pred[batch.0, j]) * (0.0 - @target[batch.0, j]));\n        _@loss\n    };\n    _@binary_cross_entropy_call\n};",
    ),
    (
        "softmax",
        """
use std::ml;
let x = [[1.0, 2.0, 3.0]];
let y = std::ml::softmax(x);
print(@y);
""",
        "let @y = {\n    let _@softmax_x: [f32; ?, ?] = {\n        let max_val[batch.0] = max[j](x[batch.0, j]);\n        let shifted[batch.0, j] = x[batch.0, j] - max_val[batch.0];\n        let exp_vals[batch.0, j] = exp(shifted[batch.0, j]);\n        let sums[batch.0] = sum[k](exp_vals[batch.0, k]);\n        let _@max_val[batch.0] = @x[batch.0, j] at argmax[j](x[batch.0, j]);\n        let _@shifted[batch.0, j] = @x[batch.0, j] - _@max_val[batch.0];\n        let _@exp_vals[batch.0, j] = exp(shifted[batch.0, j]) * _@shifted[batch.0, j];\n        let _@sums[batch.0] = sum[k](_@exp_vals[batch.0, k]);\n        let _@output[batch.0, j] = (sums[batch.0] * _@exp_vals[batch.0, j] - exp_vals[batch.0, j] * _@sums[batch.0]) / sums[batch.0] ** 2.0;\n        _@output\n    };\n    _@softmax_x\n};",
    ),
    (
        "log_softmax",
        """
use std::ml;
let x = [[1.0, 2.0, 3.0]];
let y = std::ml::log_softmax(x);
print(@y);
""",
        "let @y = {\n    let _@log_softmax_x: [f32; ?, ?] = {\n        let max_val[batch.0] = max[j](x[batch.0, j]);\n        let shifted[batch.0, j] = x[batch.0, j] - max_val[batch.0];\n        let exp_vals[batch.0, j] = exp(shifted[batch.0, j]);\n        let sum_exp[batch.0] = sum[k](exp_vals[batch.0, k]);\n        let _@max_val[batch.0] = @x[batch.0, j] at argmax[j](x[batch.0, j]);\n        let _@shifted[batch.0, j] = @x[batch.0, j] - _@max_val[batch.0];\n        let _@exp_vals[batch.0, j] = exp(shifted[batch.0, j]) * _@shifted[batch.0, j];\n        let _@sum_exp[batch.0] = sum[k](_@exp_vals[batch.0, k]);\n        let _@log_sum[batch.0] = 1.0 / sum_exp[batch.0] * _@sum_exp[batch.0] + _@max_val[batch.0];\n        let _@output[batch.0, j] = @x[batch.0, j] - _@log_sum[batch.0];\n        _@output\n    };\n    _@log_softmax_x\n};",
    ),
    (
        "reduce_l2",
        """
use std::ml;
let x = [[3.0, 4.0]];
let y = std::ml::reduce_l2(x);
print(@y);
""",
        "let @y = {\n    let _@reduce_l2_x: [f32; *] = {\n        let sum_squares[batch.0] = sum[j](x[batch.0, j] ** 2.0);\n        let _@sum_squares[batch.0] = sum[j](2.0 * x[batch.0, j] * @x[batch.0, j]);\n        let _@result[batch.0] = 0.5 * sum_squares[batch.0] ** -0.5 * _@sum_squares[batch.0];\n        _@result\n    };\n    _@reduce_l2_x\n};",
    ),
    (
        "reduce_log_sum",
        """
use std::ml;
let x = [[1.0, 2.0, 3.0]];
let y = std::ml::reduce_log_sum(x);
print(@y);
""",
        "let @y = {\n    let _@reduce_log_sum_x: [f32; ?] = {\n        let sum_val[batch.0] = sum[j](x[batch.0, j]);\n        let _@sum_val[batch.0] = sum[j](@x[batch.0, j]);\n        let _@result[batch.0] = 1.0 / sum_val[batch.0] * _@sum_val[batch.0];\n        _@result\n    };\n    _@reduce_log_sum_x\n};",
    ),
    (
        "reduce_log_sum_exp",
        """
use std::ml;
let x = [[1.0, 2.0, 3.0]];
let y = std::ml::reduce_log_sum_exp(x);
print(@y);
""",
        "let @y = {\n    let _@reduce_log_sum_exp_x: [f32; ?] = {\n        let max_val[batch.0] = max[j](x[batch.0, j]);\n        let shifted[batch.0, j] = x[batch.0, j] - max_val[batch.0];\n        let sum_exp[batch.0] = sum[j](exp(shifted[batch.0, j]));\n        let _@max_val[batch.0] = @x[batch.0, j] at argmax[j](x[batch.0, j]);\n        let _@shifted[batch.0, j] = @x[batch.0, j] - _@max_val[batch.0];\n        let _@sum_exp[batch.0] = sum[j](exp(shifted[batch.0, j]) * _@shifted[batch.0, j]);\n        let _@result[batch.0] = _@max_val[batch.0] + 1.0 / sum_exp[batch.0] * _@sum_exp[batch.0];\n        _@result\n    };\n    _@reduce_log_sum_exp_x\n};",
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
        "let @y = {\n    let _@cosine_similarity_call: [f32; ?] = {\n        let dot_product[batch.0] = sum[j](a[batch.0, j] * b[batch.0, j]);\n        let norm_a_sq[batch.0] = sum[j](a[batch.0, j] * a[batch.0, j]);\n        let norm_b_sq[batch.0] = sum[j](b[batch.0, j] * b[batch.0, j]);\n        let norm_a[batch.0] = sqrt(norm_a_sq[batch.0]);\n        let norm_b[batch.0] = sqrt(norm_b_sq[batch.0]);\n        let _@dot_product[batch.0] = sum[j](a[batch.0, j] * @b[batch.0, j] + b[batch.0, j] * @a[batch.0, j]);\n        let _@norm_a_sq[batch.0] = sum[j](2.0 * a[batch.0, j] * @a[batch.0, j]);\n        let _@norm_b_sq[batch.0] = sum[j](2.0 * b[batch.0, j] * @b[batch.0, j]);\n        let _@norm_a[batch.0] = 0.5 * norm_a_sq[batch.0] ** -0.5 * _@norm_a_sq[batch.0];\n        let _@norm_b[batch.0] = 0.5 * norm_b_sq[batch.0] ** -0.5 * _@norm_b_sq[batch.0];\n        let _@similarity[batch.0] = (norm_a[batch.0] * norm_b[batch.0] * _@dot_product[batch.0] - dot_product[batch.0] * (norm_a[batch.0] * _@norm_b[batch.0] + norm_b[batch.0] * _@norm_a[batch.0])) / (norm_a[batch.0] * norm_b[batch.0]) ** 2.0;\n        _@similarity\n    };\n    _@cosine_similarity_call\n};",
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
        "let @C = {\n    let _@matmul_call: [f32; ?, ?] = {\n        let _@output[i, j] = sum[k](A[i, k] * @B[k, j] + B[k, j] * @A[i, k]);\n        _@output\n    };\n    _@matmul_call\n};",
    ),
    (
        # Golden string is print(@C); math: @C[b,i,j]=Σ_k(A[b,i,k]@B[b,k,j]+B[b,k,j]@A[b,i,k]) — see GOLDEN_CALCULUS["batch_matmul"].
        "batch_matmul",
        """
use std::ml;
let A = [[[1.0, 2.0], [3.0, 4.0]], [[0.5, 0.5], [0.1, 0.2]]];
let B = [[[5.0, 6.0], [7.0, 8.0]], [[1.0, 1.0], [1.0, 1.0]]];
let C = std::ml::batch_matmul(A, B);
print(@C);
""",
        "let @C = {\n    let _@batch_matmul_call: [f32; ?, ?, ?] = {\n        let _@result[batch.0, i, j] = sum[k](A[batch.0, i, k] * @B[batch.0, k, j] + B[batch.0, k, j] * @A[batch.0, i, k]);\n        _@result\n    };\n    _@batch_matmul_call\n};",
    ),
]