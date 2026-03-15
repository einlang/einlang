#!/usr/bin/env python3
"""Run autodiff examples and print the results; check accuracy of derivatives."""
from pathlib import Path
import sys
from typing import List, Optional, Any, Dict, Tuple

# Allow importing einlang from repo
repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root / "src"))

from einlang.compiler.driver import CompilerDriver
from einlang.runtime.runtime import EinlangRuntime

# Tolerance for float comparison
ATOL = 1e-5
RTOL = 1e-5


def _scalar(x: Any) -> float:
    if hasattr(x, "item"):
        return float(x.item())
    return float(x)


def _allclose(a: Any, b: Any, atol: float = ATOL, rtol: float = RTOL) -> bool:
    try:
        import numpy as np
        return bool(np.allclose(np.asarray(a), np.asarray(b), atol=atol, rtol=rtol))
    except Exception:
        return abs(_scalar(a) - _scalar(b)) <= atol + rtol * abs(_scalar(b))


def run_and_print(ein_path: Path, title: str, vars_to_show: Optional[List[str]] = None) -> Dict[str, Any]:
    """Compile and run an .ein file, then print selected (or all) output variables. Returns outputs dict."""
    source = ein_path.read_text()
    driver = CompilerDriver()
    result = driver.compile(
        source, str(ein_path), root_path=ein_path.parent
    )
    if not result.success:
        print(f"Compilation failed for {ein_path}:")
        if result.tcx and result.tcx.reporter.has_errors():
            print(result.tcx.reporter.format_all_errors())
        return {}
    runtime = EinlangRuntime(backend="numpy")
    exec_result = runtime.execute(result, inputs={})
    if exec_result.error:
        print(f"Runtime error: {exec_result.error}")
        return {}
    outputs = exec_result.outputs or {}
    print()
    print("=" * 60)
    print(title)
    print("=" * 60)
    if vars_to_show:
        for name in vars_to_show:
            if name in outputs:
                val = outputs[name]
                print(f"  {name} = {val}")
            else:
                print(f"  {name} = (not in outputs)")
    else:
        for name, val in sorted(outputs.items()):
            print(f"  {name} = {val}")
    print()
    return outputs


def check_accuracy(example_name: str, outputs: Dict[str, Any], checks: List[Tuple[str, Any, str]]) -> List[str]:
    """Check outputs against expected values. Returns list of error messages (empty if all pass)."""
    errors: List[str] = []
    for var_name, expected, desc in checks:
        if var_name not in outputs:
            errors.append(f"  {example_name}: missing output {var_name}")
            continue
        actual = outputs[var_name]
        if not _allclose(actual, expected):
            errors.append(f"  {example_name}: {var_name} = {actual} (expected {expected}) — {desc}")
    return errors


def main() -> None:
    examples_dir = Path(__file__).resolve().parent
    all_errors: List[str] = []

    # autodiff_small.ein: x=1, y=2 → z=3, dz_dx=1, dz_dy=1; w=2, dw_dx=y=2, dw_dy=x=1; u=-1, du_dx=1, du_dy=-1; v=0.5, dv_dx=1/y=0.5, dv_dy=-x/y²=-0.25
    out_small = run_and_print(
        examples_dir / "autodiff_small.ein",
        "autodiff_small.ein — scalar derivatives (z=x+y, w=x*y, u=x-y, v=x/y)",
        vars_to_show=[
            "z", "dz_dx", "dz_dy",
            "w", "dw_dx", "dw_dy",
            "u", "du_dx", "du_dy",
            "v", "dv_dx", "dv_dy",
        ],
    )
    all_errors.extend(check_accuracy("autodiff_small", out_small, [
        ("z", 3.0, "z = x+y"),
        ("dz_dx", 1.0, "∂z/∂x"),
        ("dz_dy", 1.0, "∂z/∂y"),
        ("w", 2.0, "w = x*y"),
        ("dw_dx", 2.0, "∂w/∂x = y"),
        ("dw_dy", 1.0, "∂w/∂y = x"),
        ("u", -1.0, "u = x-y"),
        ("du_dx", 1.0, "∂u/∂x"),
        ("du_dy", -1.0, "∂u/∂y"),
        ("v", 0.5, "v = x/y"),
        ("dv_dx", 0.5, "∂v/∂x = 1/y"),
        ("dv_dy", -0.25, "∂v/∂y = -x/y²"),
    ]))

    # autodiff_matmul.ein: C = A @ B (2x2), dC_dA has shape (2,2,2,2); C[0,0]=19, C[1,0]=43, etc.
    out_matmul = run_and_print(
        examples_dir / "autodiff_matmul.ein",
        "autodiff_matmul.ein — C = A @ B, dC_dA = ∂C/∂A",
        vars_to_show=["C", "dC_dA"],
    )
    if out_matmul:
        import numpy as np
        C = np.asarray(out_matmul.get("C"))
        if C.shape == (2, 2) and abs(float(C[0, 0]) - 19.0) <= ATOL and abs(float(C[1, 1]) - 50.0) <= ATOL:
            pass  # C correct
        else:
            all_errors.append(f"  autodiff_matmul: C = {C} (expected [[19,22],[43,50]])")
        dC_dA = out_matmul.get("dC_dA")
        if dC_dA is not None and hasattr(dC_dA, "shape") and dC_dA.shape == (2, 2, 2, 2):
            pass  # dC_dA shape correct; values B repeated per row of A
        elif dC_dA is None:
            all_errors.append("  autodiff_matmul: missing dC_dA")
        else:
            all_errors.append(f"  autodiff_matmul: dC_dA shape = {getattr(dC_dA, 'shape', type(dC_dA))} (expected (2,2,2,2))")

    # autodiff_chain.ein: a=3, b=9, c=10; dc_da = 6 (chain rule through let bindings)
    out_chain = run_and_print(
        examples_dir / "autodiff_chain.ein",
        "autodiff_chain.ein — chain rule (c = b+1, b = a²) → dc_da = 2*a",
        vars_to_show=["a", "b", "c", "dc_da"],
    )
    all_errors.extend(check_accuracy("autodiff_chain", out_chain, [
        ("a", 3.0, "a = 3"),
        ("b", 9.0, "b = a²"),
        ("c", 10.0, "c = b+1"),
        ("dc_da", 6.0, "∂c/∂a = 2*a (chain rule)"),
    ]))

    # autodiff_user_fn.ein: a=3, b=9, db_da = 2*a = 6
    out_user = run_and_print(
        examples_dir / "autodiff_user_fn.ein",
        "autodiff_user_fn.ein — b = sq(a), db_da = @b/@a = 2*a (6 at a=3)",
        vars_to_show=["a", "b", "db_da"],
    )
    all_errors.extend(check_accuracy("autodiff_user_fn", out_user, [
        ("a", 3.0, "a = 3"),
        ("b", 9.0, "b = sq(a)"),
        ("db_da", 6.0, "∂b/∂a = 2*a"),
    ]))

    # autodiff_loss.ein: pred=2, target=1, loss=1, d_loss_d_pred = 2*(pred-target) = 2
    out_loss = run_and_print(
        examples_dir / "autodiff_loss.ein",
        "autodiff_loss.ein — loss = (pred - target)², d_loss_d_pred = 2*(pred - target)",
        vars_to_show=["pred", "target", "loss", "d_loss_d_pred"],
    )
    all_errors.extend(check_accuracy("autodiff_loss", out_loss, [
        ("pred", 2.0, "pred = 2"),
        ("target", 1.0, "target = 1"),
        ("loss", 1.0, "loss = (pred-target)²"),
        ("d_loss_d_pred", 2.0, "∂loss/∂pred = 2*(pred-target)"),
    ]))

    # Accuracy summary
    print("=" * 60)
    print("Accuracy check")
    print("=" * 60)
    if all_errors:
        for e in all_errors:
            print(e)
        print(f"\nTotal: {len(all_errors)} check(s) failed.")
        sys.exit(1)
    print("  All autodiff example outputs match expected values.")
    print()


if __name__ == "__main__":
    main()
