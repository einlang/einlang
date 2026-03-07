#!/usr/bin/env python3
"""Compare Einlang (and optional NumPy) output to golden reference.

No PyTorch/transformers/ONNX. Requires: weights and samples (run download_weights.py first).
Golden reference: golden_ref.txt (expected transcript for JFK sample).

Usage:
  python3 compare_with_golden.py [--no-numpy] [--no-einlang]
"""

import os
import sys
import time
import subprocess
import argparse

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
GOLDEN_PATH = os.path.join(SCRIPT_DIR, "golden_ref.txt")


def load_golden():
    if not os.path.isfile(GOLDEN_PATH):
        return None
    return open(GOLDEN_PATH, "r", encoding="utf-8").read().strip()


def run_numpy():
    sys.path.insert(0, SCRIPT_DIR)
    from download_weights import get_reference_transcription
    t0 = time.perf_counter()
    text, _ = get_reference_transcription()
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


def main():
    ap = argparse.ArgumentParser(description="Compare to golden reference")
    ap.add_argument("--no-numpy", action="store_true", help="Skip NumPy run")
    ap.add_argument("--no-einlang", action="store_true", help="Skip Einlang run")
    args = ap.parse_args()

    golden = load_golden()
    if not golden:
        print(f"Golden reference not found: {GOLDEN_PATH}")
        sys.exit(1)
    print("=== Whisper-tiny: compare with golden reference ===\n")
    print(f"Golden ref: {golden!r}\n")

    results = []

    if not args.no_numpy:
        print("Running NumPy reference ...")
        txt, t, err = run_numpy()
        if err:
            print(f"  NumPy: {err}\n")
        else:
            match = "OK" if txt == golden else "MISMATCH"
            print(f"  Time: {t:.2f}s  Match: {match}")
            print(f"  Text: {txt!r}\n")
            results.append(("NumPy", t, txt == golden, txt))

    if not args.no_einlang:
        print("Running Einlang (main.ein) ...")
        txt, t, err = run_einlang()
        if err:
            print(f"  Einlang: {err}\n")
        else:
            match = "OK" if txt == golden else "MISMATCH"
            print(f"  Time: {t:.2f}s  Match: {match}")
            print(f"  Text: {txt!r}\n")
            results.append(("Einlang", t, txt == golden, txt))

    print("--- Summary ---")
    for name, t, ok, txt in results:
        status = "PASS" if ok else "FAIL"
        print(f"  {name}: {t:.2f}s  {status}  |  {(txt or '')[:55]}...")
    all_ok = all(r[2] for r in results)
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
