#!/usr/bin/env python3
"""
Profile all examples and report vectorized / hybrid / scalar counts.

Run from repo root:
  PYTHONPATH=src python3 examples/profile_vectorize.py

Runs: ode, wave_2d, reaction_diffusion, heat, mnist, mnist_quantized, deit_tiny, whisper_tiny.
Requires EINLANG_PROFILE_* and EINLANG_DEBUG_VECTORIZE=1 so the backend
prints "[vectorize] Einstein clauses: V vectorized, S scalar, H hybrid, C call-scalar (total T)".
"""

import os
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def run_and_capture_vectorize(env: dict, cmd: list, cwd: Path) -> dict:
    """Run command; parse last '[vectorize] Einstein clauses: ...' line. Return dict with v,s,h,c,total."""
    full_env = {**os.environ, **env}
    try:
        result = subprocess.run(
            cmd,
            cwd=cwd,
            env=full_env,
            capture_output=True,
            text=True,
            timeout=120,
        )
        combined = (result.stdout or "") + (result.stderr or "")
    except subprocess.TimeoutExpired:
        return {"error": "timeout", "vectorized": 0, "scalar": 0, "hybrid": 0, "call_scalar": 0, "total": 0}
    except Exception as e:
        return {"error": str(e), "vectorized": 0, "scalar": 0, "hybrid": 0, "call_scalar": 0, "total": 0}

    # Parse: [vectorize] Einstein clauses: 4 vectorized, 0 scalar, 2 hybrid, 0 call-scalar (total 6)
    m = re.search(
        r"\[vectorize\] Einstein clauses:\s*(\d+)\s+vectorized,\s*(\d+)\s+scalar,\s*(\d+)\s+hybrid,\s*(\d+)\s+call-scalar\s*\(total\s+(\d+)\)",
        combined,
    )
    if m:
        return {
            "vectorized": int(m.group(1)),
            "scalar": int(m.group(2)),
            "hybrid": int(m.group(3)),
            "call_scalar": int(m.group(4)),
            "total": int(m.group(5)),
        }
    if result.returncode != 0:
        return {"error": f"exit {result.returncode}", "vectorized": 0, "scalar": 0, "hybrid": 0, "call_scalar": 0, "total": 0}
    return {"error": "no [vectorize] line", "vectorized": 0, "scalar": 0, "hybrid": 0, "call_scalar": 0, "total": 0}


def main():
    profile_env = {
        "EINLANG_PROFILE_STATEMENTS": "1",
        "EINLANG_PROFILE_FUNCTIONS": "1",
        "EINLANG_PROFILE_BLOCKS": "1",
        "EINLANG_PROFILE_REDUCTIONS": "1",
        "EINLANG_DEBUG_VECTORIZE": "1",
    }
    py = sys.executable
    src = REPO_ROOT / "src"
    env_pythonpath = {"PYTHONPATH": str(src)}

    examples = [
        ("ode", [py, "-m", "einlang", str(REPO_ROOT / "examples/ode/decay.ein")], REPO_ROOT, env_pythonpath),
        ("wave_2d", [py, str(REPO_ROOT / "examples/wave_2d/run_wave.py"), "--profile-einlang"], REPO_ROOT, {}),
        (
            "reaction_diffusion",
            [py, str(REPO_ROOT / "examples/reaction_diffusion/run_rd.py"), "--profile-einlang"],
            REPO_ROOT,
            {},
        ),
        ("heat", [py, str(REPO_ROOT / "examples/heat_animation.py"), "--profile-einlang"], REPO_ROOT, {}),
        ("mnist", [py, "-m", "einlang", "main.ein"], REPO_ROOT / "examples" / "mnist", env_pythonpath),
        ("mnist_quantized", [py, "-m", "einlang", "main.ein"], REPO_ROOT / "examples" / "mnist_quantized", env_pythonpath),
        ("deit_tiny", [py, "-m", "einlang", "main.ein"], REPO_ROOT / "examples" / "deit_tiny", env_pythonpath),
        ("whisper_tiny", [py, "-m", "einlang", "main.ein"], REPO_ROOT / "examples" / "whisper_tiny", env_pythonpath),
    ]

    print("Profile: all examples (vectorized / hybrid / scalar)\n")
    rows = []
    for name, cmd, cwd, extra_env in examples:
        env = {**os.environ, **profile_env, **extra_env}
        r = run_and_capture_vectorize(env, cmd, cwd)
        err = r.get("error")
        v, s, h, c, t = r.get("vectorized", 0), r.get("scalar", 0), r.get("hybrid", 0), r.get("call_scalar", 0), r.get("total", 0)
        if err:
            rows.append((name, err, "-", "-", "-", "-"))
        else:
            all_vec = (t > 0 and s == 0 and c == 0)
            status = "all vectorized or hybrid" if (all_vec and (v + h == t)) else ("scalar" if s else "hybrid+vectorized")
            rows.append((name, str(v), str(h), str(s), str(c), status))
    # Table
    print(f"{'Example':<22} {'V':>4} {'H':>4} {'S':>4} {'C':>4}  Status")
    print("-" * 55)
    for row in rows:
        print(f"{row[0]:<22} {row[1]:>4} {row[2]:>4} {row[3]:>4} {row[4]:>4}  {row[5]}")
    print("\nV=vectorized, H=hybrid, S=scalar, C=call-scalar.")
    print("'all vectorized or hybrid' = no scalar path.")
    print("ode is scalar: single loop dim (t) is recurrence, so 0 < len(rec_dims) < len(loops) is false -> no vectorize-over-other-dims; body reads u[t-1] so we cannot vectorize over t -> scalar loop.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
