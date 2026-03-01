#!/usr/bin/env python3
"""
Parametrized demos tests - loads all file contents together upfront for speed.
"""

import subprocess
import sys
import pytest
from pathlib import Path
from tests.test_utils import compile_and_execute


# Load all file contents once at module import time
_DEMOS_CACHE = {}
_DEMOS_PATHS = {}

def _load_all_demos():
    """Load all demos file contents into cache once"""
    if _DEMOS_CACHE:
        return

    project_root = Path(__file__).parent.parent.parent
    demos_dir = project_root / "examples" / "demos"
    # Skip demos that use unsupported syntax or require external files at runtime
    unsupported = ['enum ', 'type ', 'while ', 'tensor[', '-> tensor', 'scan[+](', 'data = [',
                   'python::']

    if demos_dir.exists():
        for f in sorted(demos_dir.glob("*.ein")):
            with open(f, 'r', encoding='utf-8') as fp:
                content = fp.read()
            if not any(kw in content for kw in unsupported):
                _DEMOS_CACHE[f.stem] = content
                _DEMOS_PATHS[f.stem] = str(f)

# Trigger load at import
_load_all_demos()


def get_demos_params():
    """Get parametrized test cases with content already loaded"""
    return [pytest.param(name, id=name) for name in _DEMOS_CACHE.keys()]


class TestDemos:
    """Tests for demos tutorial files - content pre-loaded for speed"""

    @pytest.mark.parametrize("demo_name", get_demos_params())
    def test_execution(self, compiler, runtime, demo_name):
        """Test demo execution"""
        content = _DEMOS_CACHE[demo_name]
        source_file = _DEMOS_PATHS[demo_name]

        expected_fail = "EXPECTED TO FAIL" in content

        try:
            result = compile_and_execute(content, compiler, runtime, source_file=source_file)

            if result is None or not getattr(result, 'success', False):
                if expected_fail:
                    return
                errors = getattr(result, 'errors', ['Unknown']) if result else ['No result']
                pytest.fail(f"{demo_name} failed: {errors}")
        except Exception as e:
            if expected_fail:
                return
            pytest.fail(f"{demo_name} exception: {e}")

    def test_mnist_onnx_arch(self):
        """Verify mnist_onnx_arch.ein runs and outputs a digit prediction 0..9.

        Requires pre-exported weights in examples/demos/mnist_weights/ and a
        previously saved input.npy (written by run_mnist_onnx_with_image.py).
        Skipped when those files are absent.
        """
        project_root = Path(__file__).parent.parent.parent
        demos_dir = project_root / "examples" / "demos"
        weights_dir = demos_dir / "mnist_weights"
        arch_ein = demos_dir / "mnist_onnx_arch.ein"

        required = [weights_dir / n for n in
                    ("conv1_w.npy", "conv1_b.npy", "conv2_w.npy", "conv2_b.npy",
                     "fc_w.npy", "fc_b.npy", "input.npy")]
        missing = [str(p) for p in required if not p.exists()]
        if missing:
            pytest.skip(f"mnist_weights not ready: {missing}")

        result = subprocess.run(
            [sys.executable, "-m", "einlang", str(arch_ein)],
            capture_output=True, text=True, cwd=demos_dir,
            env={**__import__("os").environ, "PYTHONPATH": str(project_root / "src")},
            timeout=300,
        )
        assert result.returncode == 0, result.stderr or result.stdout
        pred = result.stdout.strip()
        assert pred.lstrip("-").isdigit(), f"expected a digit, got: {pred!r}"
        assert 0 <= int(pred) <= 9, f"prediction out of range: {pred}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
