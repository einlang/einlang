#!/usr/bin/env python3
"""Run autodiff examples and print the results.

The AutodiffPass replaces program.statements with only bindings (no print calls),
so we run the compiled program and print the computed values from execution outputs.
"""
from pathlib import Path
import sys
from typing import List, Optional

# Allow importing einlang from repo
repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root / "src"))

from einlang.compiler.driver import CompilerDriver
from einlang.runtime.runtime import EinlangRuntime


def run_and_print(ein_path: Path, title: str, vars_to_show: Optional[List[str]] = None) -> None:
    """Compile and run an .ein file, then print selected (or all) output variables."""
    source = ein_path.read_text()
    driver = CompilerDriver()
    result = driver.compile(
        source, str(ein_path), root_path=ein_path.parent
    )
    if not result.success:
        print(f"Compilation failed for {ein_path}:")
        if result.tcx and result.tcx.reporter.has_errors():
            print(result.tcx.reporter.format_all_errors())
        return
    runtime = EinlangRuntime(backend="numpy")
    exec_result = runtime.execute(result, inputs={})
    if exec_result.error:
        print(f"Runtime error: {exec_result.error}")
        return
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


def main() -> None:
    examples_dir = Path(__file__).resolve().parent

    # autodiff_small.ein: scalars z = x+y, w = x*y, u = x-y, v = x/y and their derivatives
    run_and_print(
        examples_dir / "autodiff_small.ein",
        "autodiff_small.ein — scalar derivatives (z=x+y, w=x*y, u=x-y, v=x/y)",
        vars_to_show=[
            "z", "dz_dx", "dz_dy",
            "w", "dw_dx", "dw_dy",
            "u", "du_dx", "du_dy",
            "v", "dv_dx", "dv_dy",
        ],
    )

    # autodiff_matmul.ein: matrix multiply and gradient dC_dA
    run_and_print(
        examples_dir / "autodiff_matmul.ein",
        "autodiff_matmul.ein — C = A @ B, dC_dA = ∂C/∂A",
        vars_to_show=["C", "dC_dA"],
    )


if __name__ == "__main__":
    main()
