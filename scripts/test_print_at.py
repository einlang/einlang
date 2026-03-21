"""Test print(@y) symbolic output for core rules, std::math @fn, and a few Einstein/reduction cases.

Usage:
  python3 scripts/test_print_at.py              # golden string checks only
  python3 scripts/test_print_at.py --study      # golden + study: flushed logs, math refs, vs print(@y)
  python3 scripts/test_print_at.py --study-only # study only (math reference printed before each compile)
  python3 scripts/test_print_at.py --dump-study-only  # markdown dump of STUDY_SKIP_CASES to stdout
  python3 scripts/test_print_at.py --help

Study mode prints each case incrementally (flush=True), shows the expected calculus/Jacobian,
then the compiler/runtime result and a short heuristic compare to the reference.
"""
from pathlib import Path
from io import StringIO
import sys
from typing import List

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
        "reduce_sum",
        """
use std::ml;
let x = [[1.0, 2.0, 3.0]];
let y = std::ml::reduce_sum(x);
print(@y);
""",
        "print(@y) for multi-step inlined function: intermediate var out of scope",
    ),
    (
        "reduce_mean",
        """
use std::ml;
let x = [[1.0, 2.0, 3.0]];
let y = std::ml::reduce_mean(x);
print(@y);
""",
        "print(@y) for multi-step inlined function: intermediate var out of scope",
    ),
    (
        "reduce_l1",
        """
use std::ml;
let x = [[1.0, -2.0, 3.0]];
let y = std::ml::reduce_l1(x);
print(@y);
""",
        "Einstein clause body with function call not yet supported",
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
        "reduce_sum_square",
        """
use std::ml;
let x = [[1.0, 2.0, 3.0]];
let y = std::ml::reduce_sum_square(x);
print(@y);
""",
        "Einstein clause body with power not yet supported",
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
        "linear",
        """
use std::ml;
let x = [[1.0, 2.0]];
let W = [[0.5, 0.3], [0.2, 0.4]];
let b = [0.1, 0.2];
let y = std::ml::linear(x, W, b);
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
        "mse_loss",
        """
use std::ml;
let pred = [[1.0, 2.0, 3.0]];
let target = [[1.5, 2.5, 3.5]];
let y = std::ml::mse_loss(pred, target);
print(@y);
""",
        "print(@y) for multi-step inlined function: intermediate var out of scope",
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
        "print(@y) for multi-step inlined function: intermediate var out of scope",
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
        "binary_cross_entropy",
        """
use std::ml;
let pred = [[0.8, 0.3, 0.9]];
let target = [[1.0, 0.0, 1.0]];
let y = std::ml::binary_cross_entropy(pred, target);
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

# Expected calculus / Jacobian shape for each skipped case (for manual compare to print(@y) output).
STUDY_MATH_REFERENCE = {
    "softmax": (
        "y_i = exp(x_i)/sum_k exp(x_k); ∂y_i/∂x_j = y_i (δ_ij − y_j) (row i of Jacobian)."
    ),
    "log_softmax": (
        "log_softmax(x)_i = x_i − log(sum_k exp(x_k)); ∂/∂x_j = δ_ij − softmax(x)_j."
    ),
    "reduce_sum": "y = sum_ij x_ij; ∂y/∂x is all 1s (same shape as x).",
    "reduce_mean": "y = (1/N) sum x; ∂y/∂x is constant 1/N on each element.",
    "reduce_l1": "y = sum |x|; ∂y/∂x = sign(x) (subgradient at 0).",
    "reduce_l2": "y = ||x||_2; ∂y/∂x = x / ||x||_2 (for x ≠ 0).",
    "reduce_sum_square": "y = sum x^2; ∂y/∂x = 2x elementwise.",
    "reduce_log_sum": (
        "y = log(sum_ij exp(x_ij)); ∂y/∂x = softmax(x) flattened to x's shape."
    ),
    "reduce_log_sum_exp": "same as log-sum-exp: ∂y/∂x = softmax(x).",
    "linear": "y = x W^T + b; ∂y/∂x = W, ∂y/∂W = x, ∂y/∂b = 1 (layout as in einlang).",
    "matmul": "C = A B; ∂L/∂A = (∂L/∂C) B^T, ∂L/∂B = A^T (∂L/∂C) (VJP form for scalar L).",
    "mse_loss": "mean (pred−target)^2; ∂/∂pred = (2/N)(pred − target) (per reduction in impl).",
    "mae_loss": "mean |pred−target|; ∂/∂pred = sign(pred−target) / N (subgradient at 0).",
    "huber_loss": "quadratic near 0, linear far; ∂/∂pred is piecewise (pred−target) or ±δ.",
    "binary_cross_entropy": (
        "−(t log p + (1−t)log(1−p)); ∂/∂pred = (p−t)/(p(1−p)) per element (with stable impl variants)."
    ),
    "cosine_similarity": (
        "dot(a,b)/(||a|| ||b||); ∂/∂a, ∂/∂b are projections orthogonal to a,b (vector calculus)."
    ),
}


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
    ref = STUDY_MATH_REFERENCE.get(label, "")
    if not printed:
        _sp("  -> compare: (no print output) — cannot check against math reference.")
        return
    if "?" in printed and "=" in printed:
        _sp(
            "  -> compare: output contains '?' — _expr_to_diff_source has no rule for this IR; "
            "math reference above still states the correct ∂y; fix printer/AD to match."
        )
    low = printed.lower()
    notes = []
    if label == "softmax" and ("@" in printed) and ("sum" in low or "max" in low or "exp" in low):
        notes.append("mentions sum/max/exp-style pieces — plausible for softmax Jacobian.")
    if label == "reduce_sum" and "sum" in low and "@" in printed:
        notes.append("sum of @x-like terms — matches ∂sum/∂x = 1.")
    if label == "reduce_mean" and "sum" in low and ("/" in printed or "*" in printed):
        notes.append("sum scaled — plausible for 1/N factor.")
    if label in ("reduce_log_sum", "reduce_log_sum_exp") and "sum" in low:
        notes.append("log-sum-exp derivative involves softmax; look for exp/sum structure.")
    if label == "matmul" and "@" in printed and ("sum" in low or "[" in printed):
        notes.append("tensor contraction — plausible for matrix multiply VJP.")
    if not notes:
        notes.append("compare printed symbols to the math reference above (manual).")
    _sp("  -> compare: " + " ".join(notes))


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
    # ── Constant rule: d(c) = 0 ──
    run("constant", """
let x = 3.0;
let y = 5.0;
print(@y);
""", "@y = 0")

    # ── Identity rule: d(x) = @x ──
    run("identity", """
let x = 3.0;
let y = x;
print(@y);
""", "@y = @x")

    # ── Addition rule: d(a+b) = da + db ──
    run("add", """
let x = 3.0;
let y = x + x;
print(@y);
""", "@y = 2 * @x")

    # ── Subtraction rule: d(a-b) = da - db ──
    run("sub", """
let x = 3.0;
let y = x - 1.0;
print(@y);
""", "@y = @x")

    # ── Product rule: d(u*v) = du*v + u*dv ──
    run("product", """
let x = 3.0;
let y = x * x;
print(@y);
""", "@y = 2 * x * @x")

    # ── Product (two vars): d(a*b) = da*b + a*db ──
    run("product_two_vars", """
let a = 3.0;
let b = 4.0;
let y = a * b;
print(@y);
""", "@y = a * @b + b * @a")

    # ── Quotient rule: d(u/v) = (du*v - u*dv)/v^2 ──
    run("quotient", """
let a = 3.0;
let b = 4.0;
let y = a / b;
print(@y);
""", "@y = (b * @a - a * @b) / b ** 2")

    # ── Power (constant exponent): d(x^n) = n*x^(n-1)*dx ──
    run("power_const", """
let x = 2.0;
let y = x ** 3.0;
print(@y);
""", "@y = 3 * x ** 2 * @x")

    # ── Unary negation: d(-x) = -dx ──
    run("neg", """
let x = 3.0;
let y = -x;
print(@y);
""", "@y = -@x")

    # ── Chain rule through let: d(y) when y=f(z), z=g(x) ──
    run("chain_let", """
let x = 2.0;
let z = x * x;
let y = z + z;
print(@y);
""", "@y = 2 * @z")

    # ── @fn rule (exp): d(exp(x)) = exp(x)*dx ──
    run("exp_scalar", """
let x = 1.0;
let y = std::math::exp(x);
print(@y);
""", "@y = exp(x) * @x")

    # ── @fn rule (exp) inside Einstein: d(exp(x[i])) ──
    run("exp_einstein", """
let x = [1.0, 2.0, 3.0];
let e[i] = std::math::exp(x[i]);
print(@e);
""", "@e[i] = exp(x[i]) * @x[i]")

    # ── Sum reduction: d(sum) = sum(d) ──
    run("sum_reduction", """
let x = [1.0, 2.0, 3.0];
let e[i] = std::math::exp(x[i]);
let s = sum[k](e[k]);
print(@s);
""", "@s = sum[k](@e[k])")

    # ── Quotient rule in Einstein (softmax): d(e[i]/s) ──
    run("softmax_quotient", """
let x = [1.0, 2.0, 3.0];
let e[i] = std::math::exp(x[i]);
let s = sum[k](e[k]);
let y[i] = e[i] / s;
print(@y);
""", "@y[i] = (s * @e[i] - e[i] * @s) / s ** 2")

    # ── If/else (piecewise): d(if c {a} else {b}) = if c {da} else {db} ──
    run("if_else", """
let x = 3.0;
let y = if x > 0.0 { x } else { 0.0 };
print(@y);
""", "@y = if x > 0 { @x } else { 0 }")

    # ── Scalar mul: d(c*x) = c*dx ──
    run("scalar_mul", """
let x = 3.0;
let y = 2.0 * x;
print(@y);
""", "@y = 2 * @x")

    # ── Compound: d(x*x + x) = 2*x*dx + dx ──
    run("compound", """
let x = 3.0;
let y = x * x + x;
print(@y);
""", "@y = (2 * x + 1) * @x")

    # ── More std::math @fn rules (scalar) ──
    run("log_scalar", """
let x = 2.0;
let y = std::math::log(x);
print(@y);
""", "@y = 1 / x * @x")

    run("sin_scalar", """
let x = 1.0;
let y = std::math::sin(x);
print(@y);
""", "@y = cos(x) * @x")

    run("cos_scalar", """
let x = 1.0;
let y = std::math::cos(x);
print(@y);
""", "@y = -sin(x) * @x")

    run("tan_scalar", """
let x = 0.5;
let y = std::math::tan(x);
print(@y);
""", "@y = 1 / (cos(x) * cos(x)) * @x")

    run("log1p_scalar", """
let x = 0.5;
let y = std::math::log1p(x);
print(@y);
""", "@y = 1 / (1 + x) * @x")

    run("expm1_scalar", """
let x = 0.5;
let y = std::math::expm1(x);
print(@y);
""", "@y = exp(x) * @x")

    run("atan_scalar", """
let x = 0.5;
let y = std::math::atan(x);
print(@y);
""", "@y = 1 / (1 + x * x) * @x")

    run("asin_scalar", """
let x = 0.5;
let y = std::math::asin(x);
print(@y);
""", "@y = 1 / (1 - x * x) ** 0.5 * @x")

    run("acos_scalar", """
let x = 0.5;
let y = std::math::acos(x);
print(@y);
""", "@y = -1 / (1 - x * x) ** 0.5 * @x")

    run("atan2_two_vars", """
let y = 1.0;
let x = 2.0;
let z = std::math::atan2(y, x);
print(@z);
""", "@z = x / (x * x + y * y) * @y + -y / (x * x + y * y) * @x")

    # ── Power chain: d(x**0.5) ──
    run("sqrt_via_pow", """
let x = 4.0;
let y = x ** 0.5;
print(@y);
""", "@y = 0.5 * x ** -0.5 * @x")

    # ── Mod: autodiff treats remainder w.r.t. first operand as @x ──
    run("mod_scalar", """
let x = 7.0;
let y = x % 3.0;
print(@y);
""", "@y = @x")

    # ── Quotient with non-trivial denominator ──
    run("quotient_chain", """
let x = 3.0;
let y = x / (x + 1.0);
print(@y);
""", "@y = ((x + 1) * @x - x * @x) / (x + 1) ** 2")

    # ── Einstein: product rule on indexed access ──
    run("einstein_square", """
let x = [1.0, 2.0, 3.0];
let t[i] = x[i] * x[i];
print(@t);
""", "@t[i] = 2 * x[i] * @x[i]")

    # ── Prod reduction: d(prod x)/d x_i = (prod / x_i) * @x_i ──
    run("prod_reduction", """
let x = [1.0, 2.0, 3.0];
let p = prod[j](x[j]);
print(@p);
""", "@p = prod[j](x[j]) / x[j] * @x[j]")


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


if __name__ == "__main__":
    main()
