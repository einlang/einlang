#!/usr/bin/env python3
"""
Run Gray-Scott reaction-diffusion (main.ein) and write HTML animation.

From repo root:
  python3 examples/reaction_diffusion/run_rd.py
  python3 examples/reaction_diffusion/run_rd.py --html rd.html
  python3 examples/reaction_diffusion/run_rd.py --profile-einlang   # print vectorized/hybrid/scalar path
"""

import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

os.environ.setdefault("EINLANG_EINSTEIN_LOOP_MAX", "5000000")

import numpy as np
from einlang.compiler.driver import CompilerDriver
from einlang.runtime.runtime import EinlangRuntime

EXAMPLE_DIR = Path(__file__).resolve().parent
MAIN_EIN = EXAMPLE_DIR / "main.ein"


def write_html_animation(v_frames: np.ndarray, path: str, interval_ms: int = 60) -> None:
    """v_frames: (n_frames, ny, nx) - V concentration to display."""
    v_frames = np.asarray(v_frames)
    n_frames, ny, nx = v_frames.shape
    vmin = float(v_frames.min())
    vmax = float(v_frames.max())
    if vmax <= vmin:
        vmax = vmin + 1.0
    frames = [v_frames[t].tolist() for t in range(n_frames)]
    html = """<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Gray-Scott (Einlang)</title></head>
<body style="margin:1em;font-family:sans-serif">
<h1>Gray-Scott reaction-diffusion (V)</h1>
<p>t = <span id="t">0</span> / %(n_frames)s</p>
<canvas id="c" width="%(nx)s" height="%(ny)s" style="image-rendering:pixelated;image-rendering:crisp-edges;width:320px;height:320px"></canvas>
<script>
var frames = %(frames)s;
var vmin = %(vmin)s, vmax = %(vmax)s;
var n = frames.length;
var nx = frames[0][0].length, ny = frames[0].length;
var c = document.getElementById("c");
c.width = nx; c.height = ny;
var ctx = c.getContext("2d");
var idx = 0;
function rdColor(v) {
  var t = Math.max(0, Math.min(1, (v - vmin) / (vmax - vmin)));
  var r = Math.floor(255 * t);
  var g = Math.floor(80 * t);
  var b = Math.floor(200 * (1 - t));
  return "rgb(" + r + "," + g + "," + b + ")";
}
function draw() {
  var grid = frames[idx];
  var img = ctx.createImageData(nx, ny);
  for (var j = 0; j < ny; j++)
    for (var i = 0; i < nx; i++) {
      var v = grid[j][i];
      var k = (j * nx + i) * 4;
      var rgb = rdColor(v).match(/\\d+/g);
      img.data[k]=parseInt(rgb[0]); img.data[k+1]=parseInt(rgb[1]); img.data[k+2]=parseInt(rgb[2]); img.data[k+3]=255;
    }
  ctx.putImageData(img, 0, 0);
  document.getElementById("t").textContent = idx;
  idx = (idx + 1) %% n;
}
setInterval(draw, %(interval_ms)s);
draw();
</script>
</body></html>
""" % {
        "n_frames": n_frames,
        "nx": nx,
        "ny": ny,
        "frames": json.dumps(frames),
        "vmin": vmin,
        "vmax": vmax,
        "interval_ms": interval_ms,
    }
    Path(path).write_text(html, encoding="utf-8")


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Run Gray-Scott reaction-diffusion and write HTML animation")
    ap.add_argument("--html", type=str, default="rd.html", help="Output HTML path")
    ap.add_argument("--profile-einlang", action="store_true", help="Print backend path: vectorized / hybrid / scalar")
    args = ap.parse_args()

    if args.profile_einlang:
        os.environ["EINLANG_DEBUG_VECTORIZE"] = "1"
        os.environ["EINLANG_PROFILE_STATEMENTS"] = "1"

    source = MAIN_EIN.read_text(encoding="utf-8")
    compiler = CompilerDriver()
    runtime = EinlangRuntime()
    result = compiler.compile(source, str(MAIN_EIN), root_path=EXAMPLE_DIR)
    if not result.success:
        err = getattr(result, "tcx", None)
        if err and getattr(err, "reporter", None):
            print(err.reporter.format_all_errors(), file=sys.stderr)
        sys.exit(1)
    exec_result = runtime.execute(result)
    if exec_result.error:
        print(exec_result.error, file=sys.stderr)
        sys.exit(1)

    state = np.asarray(exec_result.outputs.get("state"))
    if state is None or state.ndim != 4:
        print("Expected 4D array state[t, c, i, j]", file=sys.stderr)
        sys.exit(1)

    # Animate V (channel 1)
    v = state[:, 1, :, :]
    write_html_animation(v, args.html)
    print("Gray-Scott result: shape state =", state.shape)
    print("Open in browser: file://%s" % Path(args.html).resolve())


if __name__ == "__main__":
    main()
