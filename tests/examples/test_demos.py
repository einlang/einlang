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

    def test_mnist(self):
        """Run examples/mnist/main.ein and verify 10/10 digit predictions."""
        project_root = Path(__file__).parent.parent.parent
        mnist_dir = project_root / "examples" / "mnist"
        main_ein = mnist_dir / "main.ein"

        required = [mnist_dir / "weights" / n for n in
                    ("conv1_w.npy", "conv1_b.npy", "conv2_w.npy", "conv2_b.npy",
                     "fc_w.npy", "fc_b.npy")]
        required += [mnist_dir / "samples" / f"{i}.pgm" for i in range(10)]
        missing = [str(p) for p in required if not p.exists()]
        assert not missing, f"mnist data missing: {missing}"

        result = subprocess.run(
            [sys.executable, "-m", "einlang", str(main_ein)],
            capture_output=True, text=True, cwd=mnist_dir,
            env={**__import__("os").environ, "PYTHONPATH": str(project_root / "src")},
            timeout=300,
        )
        assert result.returncode == 0, result.stderr or result.stdout
        output = result.stdout.strip()
        assert output == "[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]", f"unexpected output: {output!r}"

    def test_mnist_quantized(self):
        """Run examples/mnist_quantized/main.ein and verify 10/10 digit predictions."""
        project_root = Path(__file__).parent.parent.parent
        quant_dir = project_root / "examples" / "mnist_quantized"
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
        missing = [str(p) for p in required if not p.exists()]
        assert not missing, f"mnist_quantized data missing: {missing}"

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
        missing = [str(p) for p in required if not p.exists()]
        assert not missing, f"deit_tiny data missing: {missing}"

        result = subprocess.run(
            [sys.executable, "-m", "einlang", str(main_ein)],
            capture_output=True, text=True, cwd=deit_dir,
            env={**__import__("os").environ, "PYTHONPATH": str(project_root / "src")},
            timeout=1800,
        )
        assert result.returncode == 0, result.stderr or result.stdout
        output = result.stdout.strip()
        assert output == "['Egyptian Mau', 'Golden Retriever', 'strawberry']", f"unexpected output: {output!r}"


    def test_whisper_tiny(self):
        """Run examples/whisper_tiny/main.ein and verify speech transcription.
        When run with EINLANG_EINSTEIN_LOOP_MAX=100, must pass without override (fully vectorized)."""
        project_root = Path(__file__).parent.parent.parent
        whisper_dir = project_root / "examples" / "whisper_tiny"
        main_ein = whisper_dir / "main.ein"

        required = [whisper_dir / "samples" / "jfk.npy",
                    whisper_dir / "tokenizer.json"]
        weight_prefixes = [
            "enc_conv1_w", "enc_conv1_b", "enc_conv2_w", "enc_conv2_b",
            "enc_pos_emb", "enc_ln_w", "enc_ln_b",
            "dec_tok_emb", "dec_pos_emb", "dec_ln_w", "dec_ln_b",
        ]
        required += [whisper_dir / "weights" / f"{n}.npy" for n in weight_prefixes]
        missing = [str(p) for p in required if not p.exists()]
        assert not missing, f"whisper_tiny data missing: {missing}"

        result = subprocess.run(
            [sys.executable, "-m", "einlang", str(main_ein)],
            capture_output=True, text=True, cwd=whisper_dir,
            env={**__import__("os").environ, "PYTHONPATH": str(project_root / "src")},
            timeout=3600,
        )
        assert result.returncode == 0, result.stderr or result.stdout
        output = result.stdout.strip().lower()
        assert len(output) > 5, f"whisper output too short: {output!r}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
