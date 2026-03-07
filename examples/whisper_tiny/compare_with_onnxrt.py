#!/usr/bin/env python3
"""Compare NumPy/Einlang Whisper-tiny with ONNX Runtime.

Requires: weights and samples (run download_weights.py first).
Optional: pip install onnxruntime transformers optimum[onnxruntime]
  for ONNX RT comparison. Without optimum, compares only NumPy vs Einlang.

Usage:
  python3 compare_with_onnxrt.py [--no-einlang] [--no-onnx]
  (from examples/whisper_tiny, or set PYTHONPATH to repo root for Einlang)
"""

import os
import sys
import time
import subprocess
import argparse

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))


def run_numpy():
    sys.path.insert(0, SCRIPT_DIR)
    from download_weights import get_reference_transcription
    t0 = time.perf_counter()
    text, tokens = get_reference_transcription()
    elapsed = time.perf_counter() - t0
    if text is None:
        return None, None, "missing data"
    return text, elapsed, None


def run_einlang():
    main_ein = os.path.join(SCRIPT_DIR, "main.ein")
    env = {**os.environ, "PYTHONPATH": os.path.join(REPO_ROOT, "src")}
    t0 = time.perf_counter()
    r = subprocess.run(
        [sys.executable, "-m", "einlang", main_ein],
        capture_output=True, text=True, cwd=SCRIPT_DIR, env=env, timeout=600,
    )
    elapsed = time.perf_counter() - t0
    if r.returncode != 0:
        return None, elapsed, r.stderr or r.stdout or "failed"
    return r.stdout.strip(), elapsed, None


def run_onnxrt():
    """Run Whisper-tiny via ONNX Runtime (optimum) or fallback to PyTorch pipeline."""
    audio_path = os.path.join(SCRIPT_DIR, "samples", "audio.wav")
    if not os.path.isfile(audio_path):
        return None, None, "samples/audio.wav not found (run download_weights.py, then delete samples/jfk.npy and re-run to save audio)"
    try:
        from transformers import pipeline
    except ImportError:
        return None, None, "pip install transformers"
    # Prefer ONNX Runtime via Optimum
    backend = "pytorch"
    try:
        from optimum.onnxruntime import ORTModelForSpeechSeq2Seq
        model = ORTModelForSpeechSeq2Seq.from_pretrained(
            "openai/whisper-tiny", export=True,
        )
        backend = "onnxruntime"
    except Exception:
        model = "openai/whisper-tiny"
    pipe = pipeline("automatic-speech-recognition", model=model, device=-1)
    t0 = time.perf_counter()
    out = pipe(audio_path, return_timestamps=False)
    elapsed = time.perf_counter() - t0
    text = (out.get("text") or "").strip()
    if not text and isinstance(out.get("chunks"), list) and out["chunks"]:
        text = (out["chunks"][0].get("text") or "").strip()
    if not text:
        text = str(out)
    return text, elapsed, backend


def main():
    ap = argparse.ArgumentParser(description="Compare NumPy / Einlang / ONNX RT")
    ap.add_argument("--no-einlang", action="store_true", help="Skip Einlang run")
    ap.add_argument("--no-onnx", action="store_true", help="Skip ONNX RT run")
    args = ap.parse_args()

    print("=== Whisper-tiny: NumPy vs Einlang vs ONNX RT ===\n")

    # NumPy reference
    print("Running NumPy reference ...")
    txt_np, time_np, err_np = run_numpy()
    if err_np:
        print(f"  NumPy: {err_np}\n")
        time_np = None
    else:
        print(f"  Time: {time_np:.2f}s")
        print(f"  Text: {txt_np!r}\n")

    # Einlang
    if not args.no_einlang:
        print("Running Einlang (main.ein) ...")
        txt_ein, time_ein, err_ein = run_einlang()
        if err_ein:
            print(f"  Einlang: {err_ein}\n")
            time_ein = None
        else:
            print(f"  Time: {time_ein:.2f}s")
            print(f"  Text: {txt_ein!r}\n")
    else:
        txt_ein, time_ein, err_ein = None, None, None

    # ONNX RT
    if not args.no_onnx:
        print("Running ONNX Runtime (or PyTorch fallback) ...")
        txt_onnx, time_onnx, err_onnx = run_onnxrt()
        if isinstance(err_onnx, str) and (txt_onnx is None or time_onnx is None):
            print(f"  ONNX: {err_onnx}\n")
            time_onnx, txt_onnx = None, None
        else:
            backend = err_onnx if isinstance(err_onnx, str) else "onnx"
            print(f"  Backend: {backend}")
            print(f"  Time: {time_onnx:.2f}s")
            print(f"  Text: {txt_onnx!r}\n")
    else:
        txt_onnx, time_onnx = None, None

    # Summary table
    print("--- Summary ---")
    rows = [("NumPy (reference)", time_np, txt_np)]
    if not args.no_einlang and time_ein is not None:
        rows.append(("Einlang", time_ein, txt_ein))
    if not args.no_onnx and time_onnx is not None:
        rows.append(("ONNX RT", time_onnx, txt_onnx))
    for name, t, txt in rows:
        if t is not None:
            print(f"  {name}: {t:.2f}s  |  {txt[:60]}{'...' if len(txt or '') > 60 else ''}")
    if time_np and time_ein is not None:
        print(f"\n  Einlang / NumPy time ratio: {time_ein / time_np:.2f}x")
    if time_np and time_onnx is not None:
        print(f"  ONNX RT / NumPy time ratio: {time_onnx / time_np:.2f}x")


if __name__ == "__main__":
    main()
