"""Test that whisper_tiny Einlang transcription matches the numpy reference."""

import os
import subprocess
import sys
from pathlib import Path

import pytest


def _get_reference_transcription():
    """Return (text, tokens) from numpy reference, or (None, None) if data missing."""
    project_root = Path(__file__).resolve().parent.parent.parent
    whisper_dir = project_root / "examples" / "whisper_tiny"
    if str(whisper_dir) not in sys.path:
        sys.path.insert(0, str(whisper_dir))
    from download_weights import get_reference_transcription
    return get_reference_transcription()


def test_whisper_tiny_accuracy():
    """Run main.ein and assert printed transcription equals numpy reference.
    Skips if weights/samples/tokenizer are missing. Run from repo root with data present to verify accuracy."""
    ref_text, _ = _get_reference_transcription()
    if ref_text is None:
        pytest.skip("whisper_tiny data missing (run: cd examples/whisper_tiny && python3 download_weights.py)")

    project_root = Path(__file__).resolve().parent.parent.parent
    whisper_dir = project_root / "examples" / "whisper_tiny"
    main_ein = whisper_dir / "main.ein"
    env = {**os.environ, "PYTHONPATH": str(project_root / "src")}

    result = subprocess.run(
        [sys.executable, "-m", "einlang", str(main_ein)],
        capture_output=True,
        text=True,
        cwd=str(whisper_dir),
        env=env,
        timeout=3600,
    )
    assert result.returncode == 0, (result.stderr or result.stdout or "no output")
    einlang_text = result.stdout.strip()
    assert einlang_text == ref_text, (
        f"Transcription mismatch:\n  reference: {ref_text!r}\n  einlang:   {einlang_text!r}"
    )
