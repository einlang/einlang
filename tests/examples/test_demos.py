#!/usr/bin/env python3
"""
Parametrized demos tests - loads all file contents together upfront for speed.
"""

import re
import subprocess
import sys
import pytest
from pathlib import Path
from tests.test_utils import compile_and_execute


def _parse_vectorize_counts(output: str):
    """Extract (vectorized, scalar, hybrid, call_scalar) from --debug-vectorize output."""
    m = re.search(
        r"\[vectorize\] Einstein clauses: (\d+) vectorized, (\d+) scalar, (\d+) hybrid, (\d+) call-scalar",
        output,
    )
    if m is None:
        return None
    return int(m.group(1)), int(m.group(2)), int(m.group(3)), int(m.group(4))


def _assert_vectorize_counts(output: str, min_vectorized: int, max_scalar: int, label: str):
    counts = _parse_vectorize_counts(output)
    assert counts is not None, f"{label}: --debug-vectorize summary line not found in output"
    vectorized, scalar, hybrid, call_scalar = counts
    assert vectorized >= min_vectorized, (
        f"{label}: vectorized count regressed: {vectorized} < {min_vectorized}"
    )
    assert scalar <= max_scalar, (
        f"{label}: scalar count increased: {scalar} > {max_scalar}"
    )


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


def _ensure_weights_on_demand(project_root, example_dir, required_paths, script_name,
                             script_args=None, timeout=300):
    """If any required path is missing, run script_name in example_dir; fail if still missing."""
    missing = [p for p in required_paths if not p.exists()]
    if not missing:
        return
    script = example_dir / script_name
    if not script.is_file():
        pytest.fail(
            f"{example_dir.name}: required {script_name} missing (required files: "
            f"{[p.name for p in required_paths[:3]]}{'...' if len(required_paths) > 3 else ''})"
        )
    env = {**__import__("os").environ, "PYTHONPATH": str(project_root / "src")}
    result = subprocess.run(
        [sys.executable, str(script)] + (script_args or []),
        capture_output=True, text=True, cwd=str(example_dir), env=env, timeout=timeout,
    )
    if result.returncode != 0:
        pytest.fail(
            f"{example_dir.name}: {script_name} failed (exit {result.returncode})\n"
            f"stdout: {result.stdout}\nstderr: {result.stderr}"
        )
    still_missing = [p for p in required_paths if not p.exists()]
    if still_missing:
        pytest.fail(
            f"{example_dir.name} still missing after {script_name}: {still_missing}\n"
            f"stdout: {result.stdout}\nstderr: {result.stderr}"
        )


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

            if result is None or not result.success:
                if expected_fail:
                    return
                errors = result.errors if result else ['No result']
                pytest.fail(f"{demo_name} failed: {errors}")
        except Exception as e:
            if expected_fail:
                return
            pytest.fail(f"{demo_name} exception: {e}")

    def test_mnist(self):
        """Run examples/mnist/main.ein and verify 10/10 digit predictions."""
        project_root = Path(__file__).parent.parent.parent
        mnist_dir = project_root / "examples" / "mnist"
        main_ein = mnist_dir / "main.ein"

        required = [mnist_dir / "weights" / n for n in
                    ("conv1_w.npy", "conv1_b.npy", "conv2_w.npy", "conv2_b.npy",
                     "fc_w.npy", "fc_b.npy")]
        required += [mnist_dir / "samples" / f"{i}.pgm" for i in range(10)]
        _ensure_weights_on_demand(project_root, mnist_dir, required, "download_weights.py")

        result = subprocess.run(
            [sys.executable, "-m", "einlang", str(main_ein), "--debug-vectorize"],
            capture_output=True, text=True, cwd=mnist_dir,
            env={**__import__("os").environ, "PYTHONPATH": str(project_root / "src")},
            timeout=300,
        )
        assert result.returncode == 0, result.stderr or result.stdout
        full_output = result.stdout.strip()
        output = "\n".join(l for l in full_output.split("\n") if not l.startswith("[vectorize]")).strip()
        assert output == "[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]", f"unexpected output: {output!r}"
        _assert_vectorize_counts(full_output, min_vectorized=290, max_scalar=0, label="mnist")

    def test_mnist_quantized(self):
        """Run examples/mnist_quantized/main.ein and verify 10/10 digit predictions."""
        project_root = Path(__file__).parent.parent.parent
        quant_dir = project_root / "examples" / "mnist_quantized"
        mnist_dir = project_root / "examples" / "mnist"
        main_ein = quant_dir / "main.ein"

        weight_names = [
            "conv1_w_q.npy", "conv1_w_s.npy",
            "conv2_w_q.npy", "conv2_w_s.npy",
            "fc_w_q.npy", "fc_w_s.npy",
            "conv1_b.npy", "conv2_b.npy", "fc_b.npy",
            "act1_s.npy", "flat_s.npy",
        ]
        required = [quant_dir / "weights" / n for n in weight_names]
        required += [quant_dir / "samples" / f"{i}.pgm" for i in range(10)]
        # prepare_weights.py creates weights (reads from ../mnist/weights); copy samples from mnist if missing
        quant_samples = quant_dir / "samples"
        if not (quant_samples / "0.pgm").exists() and (mnist_dir / "samples" / "0.pgm").exists():
            quant_samples.mkdir(parents=True, exist_ok=True)
            for i in range(10):
                src = mnist_dir / "samples" / f"{i}.pgm"
                if src.exists():
                    (quant_samples / f"{i}.pgm").write_bytes(src.read_bytes())
        _ensure_weights_on_demand(project_root, quant_dir, required, "prepare_weights.py")

        result = subprocess.run(
            [sys.executable, "-m", "einlang", str(main_ein)],
            capture_output=True, text=True, cwd=quant_dir,
            env={**__import__("os").environ, "PYTHONPATH": str(project_root / "src")},
            timeout=300,
        )
        assert result.returncode == 0, result.stderr or result.stdout
        output = result.stdout.strip()
        assert output == "[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]", f"unexpected output: {output!r}"

    def test_deit_tiny(self):
        """Run examples/deit_tiny/main.ein and verify ImageNet predictions."""
        project_root = Path(__file__).parent.parent.parent
        deit_dir = project_root / "examples" / "deit_tiny"
        main_ein = deit_dir / "main.ein"

        weight_names = [
            "patch_proj_w.npy", "patch_proj_b.npy", "cls_token.npy", "pos_embed.npy",
            "norm_w.npy", "norm_b.npy", "head_w.npy", "head_b.npy",
            "blk_ln1_w.npy", "blk_ln1_b.npy", "blk_qkv_w.npy", "blk_qkv_b.npy",
            "blk_proj_w.npy", "blk_proj_b.npy", "blk_ln2_w.npy", "blk_ln2_b.npy",
            "blk_fc1_w.npy", "blk_fc1_b.npy", "blk_fc2_w.npy", "blk_fc2_b.npy",
        ]
        required = [deit_dir / "weights" / n for n in weight_names]
        required += [deit_dir / "samples" / f"{i}.npy" for i in range(3)]
        _ensure_weights_on_demand(project_root, deit_dir, required, "download_weights.py", timeout=600)

        result = subprocess.run(
            [sys.executable, "-m", "einlang", str(main_ein), "--debug-vectorize"],
            capture_output=True, text=True, cwd=deit_dir,
            env={**__import__("os").environ, "PYTHONPATH": str(project_root / "src")},
            timeout=1800,
        )
        assert result.returncode == 0, result.stderr or result.stdout
        full_output = result.stdout.strip()
        output = "\n".join(l for l in full_output.split("\n") if not l.startswith("[vectorize]")).strip()
        assert output == "['Egyptian Mau', 'Golden Retriever', 'strawberry']", f"unexpected output: {output!r}"
        _assert_vectorize_counts(full_output, min_vectorized=963, max_scalar=0, label="deit_tiny")

    def test_whisper_tiny(self):
        """Run examples/whisper_tiny/main.ein and assert transcript matches golden_ref.txt."""
        project_root = Path(__file__).parent.parent.parent
        whisper_dir = project_root / "examples" / "whisper_tiny"
        golden = whisper_dir / "golden_ref.txt"
        if not golden.is_file():
            pytest.fail("whisper_tiny: required golden_ref.txt missing")
        main_ein = whisper_dir / "main.ein"
        if not main_ein.exists():
            pytest.fail("whisper_tiny: required main.ein missing")
        required = [
            whisper_dir / "weights" / "enc_conv1_w.npy",
            whisper_dir / "samples" / "jfk.npy",
        ]
        _ensure_weights_on_demand(
            project_root, whisper_dir, required,
            "download_weights.py", script_args=["--skip-verify"], timeout=300,
        )
        golden_text = golden.read_text(encoding="utf-8").strip()

        result = subprocess.run(
            [sys.executable, "-m", "einlang", str(main_ein), "--debug-vectorize"],
            capture_output=True, text=True, cwd=str(whisper_dir),
            env={**__import__("os").environ, "PYTHONPATH": str(project_root / "src")},
            timeout=3600,
        )
        assert result.returncode == 0, result.stderr or result.stdout or "no output"
        full_output = result.stdout.strip()
        output = "\n".join(l for l in full_output.split("\n") if not l.startswith("[vectorize]")).strip()
        if output != golden_text:
            print(f"\nwhisper_tiny transcription:\n  golden:  {golden_text!r}\n  einlang: {output!r}")
            pytest.fail(
                f"Transcription mismatch:\n  golden:  {golden_text!r}\n  einlang: {output!r}\n"
                "Possible causes: (1) different audio sample (e.g. download_weights used 440Hz sine fallback "
                "if JFK URLs failed) -> remove samples/jfk.npy and re-run download_weights.py with network; "
                "(2) numerical/implementation difference -> if einlang output is correct, update golden_ref.txt "
                "with: echo -n '<output>' > examples/whisper_tiny/golden_ref.txt"
            )
        _assert_vectorize_counts(full_output, min_vectorized=13811, max_scalar=4, label="whisper_tiny")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
