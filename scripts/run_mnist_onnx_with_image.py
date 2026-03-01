#!/usr/bin/env python3
"""Run MNIST inference via Einlang.

Usage:
  python3 scripts/run_mnist_onnx_with_image.py 3
  python3 scripts/run_mnist_onnx_with_image.py path/to/digit.pgm
"""
import os
import subprocess
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
demos_dir    = project_root / "examples" / "demos"
samples_dir  = demos_dir / "mnist_samples"
weights_dir  = demos_dir / "mnist_weights"
arch_ein     = demos_dir / "mnist_onnx_arch.ein"


def find_pgm(arg: str) -> Path:
    if arg.isdigit() and 0 <= int(arg) <= 9:
        p = samples_dir / f"{arg}.pgm"
        if p.exists():
            return p
        for pattern in [
            project_root / "data" / "tensorrt" / "tensorrt_sample_data_20260203" / "mnist",
            *project_root.glob("data/tensorrt/*/mnist"),
        ]:
            p = Path(pattern) / f"{arg}.pgm"
            if p.exists():
                return p
        if os.environ.get("TRT_DATADIR"):
            p = Path(os.environ["TRT_DATADIR"]) / "mnist" / f"{arg}.pgm"
            if p.exists():
                return p
        print(f"digit {arg}.pgm not found in mnist_samples/ or data/")
        return None
    for candidate in [(project_root / arg).resolve(), Path(arg).resolve()]:
        if candidate.exists():
            return candidate
    print(f"file not found: {arg}")
    return None


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 scripts/run_mnist_onnx_with_image.py <digit 0-9 or path to .pgm>")
        return 1

    pgm = find_pgm(sys.argv[1])
    if pgm is None:
        return 1

    weights_needed = ["conv1_w", "conv1_b", "conv2_w", "conv2_b", "fc_w", "fc_b"]
    missing = [n for n in weights_needed if not (weights_dir / f"{n}.npy").exists()]
    if missing:
        print(f"Missing weight files: {missing}")
        print("Run: python3 scripts/export_mnist_weights_npy.py")
        return 1

    result = subprocess.run(
        [sys.executable, "-m", "einlang", str(arch_ein)],
        capture_output=True, text=True, cwd=demos_dir,
        env={**os.environ,
             "PYTHONPATH": str(project_root / "src"),
             "EINLANG_MNIST_INPUT": str(pgm)},
        timeout=300,
    )
    if result.returncode != 0:
        sys.stderr.write(result.stderr or result.stdout)
        return 1
    print(result.stdout.strip())
    return 0


if __name__ == "__main__":
    sys.exit(main())
