#!/usr/bin/env python3
"""
Parametrized demos tests - loads all file contents together upfront for speed.
"""

import subprocess
import sys
import os
import json
import importlib
import time
from collections import Counter
from contextlib import contextmanager
import pytest
from pathlib import Path
import numpy as np

from einlang.compiler.driver import CompilerDriver
from einlang.ir.nodes import IRNode, LoweredReductionIR
from einlang.runtime.runtime import EinlangRuntime
from tests.test_utils import compile_and_execute, load_example_sources, project_root as repo_root


def _assert_vectorize_counts_dict(
    counts,
    min_vectorized: int,
    max_scalar: int,
    label: str,
    *,
    max_hybrid: int = 0,
    max_call_scalar: int = 0,
):
    vectorized = int(counts.get("vectorized", 0))
    scalar = int(counts.get("scalar", 0))
    hybrid = int(counts.get("hybrid", 0))
    call_scalar = int(counts.get("call_scalar", 0))
    assert vectorized >= min_vectorized, (
        f"{label}: vectorized count regressed: {vectorized} < {min_vectorized}"
    )
    assert scalar <= max_scalar, (
        f"{label}: scalar count increased: {scalar} > {max_scalar}"
    )
    assert hybrid <= max_hybrid, (
        f"{label}: hybrid count increased: {hybrid} > {max_hybrid}"
    )
    assert call_scalar <= max_call_scalar, (
        f"{label}: call-scalar count increased: {call_scalar} > {max_call_scalar}"
    )


def _output_as_list(exec_result, output_name: str):
    return np.asarray(exec_result.outputs.get(output_name)).tolist()


def _copy_mnist_samples_if_needed(src_dir: Path, dst_dir: Path, count: int = 10):
    if (dst_dir / "0.pgm").exists() or not (src_dir / "0.pgm").exists():
        return
    dst_dir.mkdir(parents=True, exist_ok=True)
    for i in range(count):
        src = src_dir / f"{i}.pgm"
        if src.exists():
            (dst_dir / f"{i}.pgm").write_bytes(src.read_bytes())


_DEMO_SOURCES = load_example_sources("examples/demos")


@contextmanager
def _asset_prepare_lock(example_dir: Path, timeout: int):
    """Serialize on-demand example asset generation across xdist workers."""
    lock_dir = example_dir / ".pytest-asset-lock"
    deadline = time.time() + timeout
    while True:
        try:
            lock_dir.mkdir()
            break
        except FileExistsError:
            if time.time() >= deadline:
                pytest.fail(f"{example_dir.name}: timed out waiting for asset lock {lock_dir.name}")
            time.sleep(0.2)
    try:
        yield
    finally:
        try:
            lock_dir.rmdir()
        except OSError:
            pass


def _ensure_weights_on_demand(project_root, example_dir, required_paths, script_name,
                             script_args=None, timeout=300):
    """If any required path is missing, run script_name in example_dir; fail if still missing."""
    missing = [p for p in required_paths if not p.exists()]
    if not missing:
        return
    with _asset_prepare_lock(example_dir, timeout=timeout):
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
            cwd=str(example_dir), env=env, timeout=timeout,
        )
        if result.returncode != 0:
            pytest.fail(
                f"{example_dir.name}: {script_name} failed (exit {result.returncode})"
            )
        still_missing = [p for p in required_paths if not p.exists()]
        if still_missing:
            pytest.fail(f"{example_dir.name} still missing after {script_name}: {still_missing}")


def _run_file_with_stats(path: Path):
    source = path.read_text(encoding="utf-8")
    compiler = CompilerDriver()
    runtime = EinlangRuntime(backend="numpy")
    with _example_runtime_context(path.parent):
        result = compiler.compile(source, source_file=str(path), root_path=path.parent)
        assert result.success, result.get_errors() or "compile failed"
        exec_result = runtime.execute(result)
    assert exec_result.success, exec_result.error or exec_result.errors
    return exec_result, runtime.get_last_vectorize_counts()


def _run_file_with_stats_subprocess(path: Path, *, timeout: int = 300):
    repo = repo_root()
    child_env = dict(os.environ)
    script = (
        "import json, sys\n"
        "from pathlib import Path\n"
        f"repo = Path({str(repo)!r})\n"
        "sys.path.insert(0, str(repo / 'src'))\n"
        "sys.path.insert(0, str(repo))\n"
        "from tests.examples.test_demo import _run_file_with_stats\n"
        f"path = Path({str(path)!r})\n"
        "exec_result, counts = _run_file_with_stats(path)\n"
        "predictions = exec_result.outputs.get('predictions')\n"
        "if hasattr(predictions, 'tolist'):\n"
        "    predictions = predictions.tolist()\n"
        "print(json.dumps({'predictions': predictions, 'counts': counts}))\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", script],
        cwd=str(repo),
        env=child_env,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    assert proc.returncode == 0, (
        f"child process failed for {path} (exit {proc.returncode})\n"
        f"STDERR:\n{proc.stderr}\nSTDOUT:\n{proc.stdout}"
    )
    lines = [line for line in proc.stdout.splitlines() if line.strip()]
    assert lines, "mnist subprocess produced no output"
    payload = json.loads(lines[-1])
    return payload["predictions"], payload["counts"]


def _assert_mnist_main_example(path: Path, *, timeout: int = 600):
    """Shared slow-path check for examples/mnist/main.ein."""
    predictions, counts = _run_file_with_stats_subprocess(path, timeout=timeout)
    assert predictions == list(range(10)), f"unexpected output: {predictions!r}"
    _assert_vectorize_counts_dict(counts, min_vectorized=1, max_scalar=4, label="mnist")


def _walk_ir(node):
    if node is None:
        return
    if isinstance(node, (list, tuple)):
        for item in node:
            yield from _walk_ir(item)
        return
    if not isinstance(node, IRNode):
        return
    yield node
    slots = []
    for cls in type(node).__mro__:
        cls_slots = getattr(cls, "__slots__", ())
        if isinstance(cls_slots, str):
            slots.append(cls_slots)
        else:
            slots.extend(cls_slots)
    seen = set()
    for slot in slots:
        if slot in seen:
            continue
        seen.add(slot)
        yield from _walk_ir(getattr(node, slot, None))


def _compile_reduction_strategy_counts(path: Path):
    source = path.read_text(encoding="utf-8")
    compiler = CompilerDriver()
    with _example_runtime_context(path.parent):
        result = compiler.compile(source, source_file=str(path), root_path=path.parent)
    assert result.success, result.get_errors() or "compile failed"
    assert result.ir is not None
    return Counter(
        getattr(node, "execution_strategy", None)
        for node in _walk_ir(result.ir)
        if isinstance(node, LoweredReductionIR)
    )


@contextmanager
def _example_runtime_context(example_dir: Path):
    """Run compile/execute with the example directory acting like CLI cwd/import root."""
    example_dir_str = str(example_dir)
    old_cwd = os.getcwd()
    inserted = False
    if example_dir_str not in sys.path:
        sys.path.insert(0, example_dir_str)
        inserted = True
    importlib.invalidate_caches()
    os.chdir(example_dir_str)
    try:
        yield
    finally:
        os.chdir(old_cwd)
        if inserted:
            try:
                sys.path.remove(example_dir_str)
            except ValueError:
                pass


class TestD:
    """Tests for demos tutorial files - content pre-loaded for speed"""

    @pytest.mark.parametrize("demo_source", _DEMO_SOURCES, ids=lambda source: source.name)
    def test_execution(self, compiler, runtime, demo_source):
        """Test demo execution"""
        content = demo_source.content
        source_file = str(demo_source.path)

        expected_fail = "EXPECTED TO FAIL" in content

        try:
            result = compile_and_execute(content, compiler, runtime, source_file=source_file)

            if result is None or not result.success:
                if expected_fail:
                    return
                errors = result.errors if result else ['No result']
                pytest.fail(f"{demo_source.name} failed: {errors}")
        except Exception as e:
            if expected_fail:
                return
            pytest.fail(f"{demo_source.name} exception: {e}")

    def test_mnist_quantized(self):
        """Run examples/mnist_quantized/main.ein and verify 10/10 digit predictions."""
        root = repo_root()
        quant_dir = root / "examples" / "mnist_quantized"
        mnist_dir = root / "examples" / "mnist"
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
        _copy_mnist_samples_if_needed(mnist_dir / "samples", quant_dir / "samples")
        _ensure_weights_on_demand(root, quant_dir, required, "prepare_weights.py")

        exec_result, _ = _run_file_with_stats(main_ein)
        predictions = _output_as_list(exec_result, "predictions")
        assert predictions == list(range(10)), f"unexpected output: {predictions!r}"

    def test_deit_tiny(self):
        """Run examples/deit_tiny/main.ein and verify ImageNet predictions."""
        root = repo_root()
        deit_dir = root / "examples" / "deit_tiny"
        main_ein = deit_dir / "main.ein"
        reduction_strategies = _compile_reduction_strategy_counts(main_ein)

        weight_names = [
            "patch_proj_w.npy", "patch_proj_b.npy", "cls_token.npy", "pos_embed.npy",
            "norm_w.npy", "norm_b.npy", "head_w.npy", "head_b.npy",
            "blk_ln1_w.npy", "blk_ln1_b.npy", "blk_qkv_w.npy", "blk_qkv_b.npy",
            "blk_proj_w.npy", "blk_proj_b.npy", "blk_ln2_w.npy", "blk_ln2_b.npy",
            "blk_fc1_w.npy", "blk_fc1_b.npy", "blk_fc2_w.npy", "blk_fc2_b.npy",
        ]
        required = [deit_dir / "weights" / n for n in weight_names]
        required += [deit_dir / "samples" / f"{i}.npy" for i in range(3)]
        _ensure_weights_on_demand(root, deit_dir, required, "download_weights.py", timeout=600)

        exec_result, counts = _run_file_with_stats(main_ein)
        names = _output_as_list(exec_result, "names")
        assert names == ["Egyptian Mau", "Golden Retriever", "strawberry"], f"unexpected output: {names!r}"
        assert reduction_strategies.get("windowed_sumprod", 0) >= 2, reduction_strategies
        assert reduction_strategies.get("matmul_sumprod", 0) >= 14, reduction_strategies
        assert reduction_strategies.get("scalar", 0) == 0, reduction_strategies
        _assert_vectorize_counts_dict(counts, min_vectorized=963, max_scalar=0, label="deit_tiny")

    def test_whisper_tiny(self):
        """Run examples/whisper_tiny/main.ein and assert transcript matches golden_ref.txt."""
        root = repo_root()
        whisper_dir = root / "examples" / "whisper_tiny"
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
            root, whisper_dir, required,
            "download_weights.py", script_args=["--skip-verify"], timeout=300,
        )
        golden_text = golden.read_text(encoding="utf-8").strip()

        exec_result, counts = _run_file_with_stats(main_ein)

        output = exec_result.outputs.get("text")
        if isinstance(output, np.ndarray) and output.ndim == 0:
            output = output.item()
        output = "" if output is None else str(output).strip()
        if output != golden_text:
            print(f"\nwhisper_tiny transcription:\n  golden:  {golden_text!r}\n  einlang: {output!r}")
            pytest.fail(
                f"Transcription mismatch:\n  golden:  {golden_text!r}\n  einlang: {output!r}\n"
                "Possible causes: (1) different audio sample (e.g. download_weights used 440Hz sine fallback "
                "if JFK URLs failed) -> remove samples/jfk.npy and re-run download_weights.py with network; "
                "(2) numerical/implementation difference -> if einlang output is correct, update golden_ref.txt "
                "with: echo -n '<output>' > examples/whisper_tiny/golden_ref.txt"
            )
        # The tuple-valued decode_state recurrence now uses one recurrence-object
        # hybrid step for correctness; the old hidden per-slot scalar fallback
        # inside nested bindings is still fixed.
        _assert_vectorize_counts_dict(
            counts,
            min_vectorized=5722,
            max_scalar=0,
            max_hybrid=1,
            label="whisper_tiny",
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
