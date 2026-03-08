#!/usr/bin/env python3
"""
Run 2D wave equation (main.ein) and write HTML animation.

From repo root:
  python3 examples/wave_2d/run_wave.py
  python3 examples/wave_2d/run_wave.py --html wave.html
  python3 examples/wave_2d/run_wave.py --profile-einlang   # per-clause time + vectorized/hybrid/scalar summary
"""

import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np
from einlang.compiler.driver import CompilerDriver
from einlang.runtime.runtime import EinlangRuntime

# Allow enough recurrence steps for 200 time levels
if "EINLANG_EINSTEIN_LOOP_MAX" not in os.environ:
    os.environ["EINLANG_EINSTEIN_LOOP_MAX"] = "500"

EXAMPLE_DIR = Path(__file__).resolve().parent
MAIN_EIN = EXAMPLE_DIR / "main.ein"


def write_html_animation(h: np.ndarray, path: str, interval_ms: int = 80) -> None:
    h = np.asarray(h)
    n_frames, ny, nx = h.shape
    vmin = float(h.min())
    vmax = float(h.max())
    if vmax <= vmin:
        vmax = vmin + 1.0
    frames = [h[t].tolist() for t in range(n_frames)]
    html = """<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>2D wave equation (Einlang)</title></head>
<body style="margin:1em;font-family:sans-serif">
<h1>2D wave &part;&sup2;h/&part;t&sup2; = c&sup2;&nabla;&sup2;h</h1>
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
function waveColor(v) {
  var t = (v - vmin) / (vmax - vmin);
  var r, g, b;
  if (t < 0.33) {
    r = Math.floor(255 * (t / 0.33));
    g = 0;
    b = Math.floor(120 * (1 - t / 0.33));
  } else if (t < 0.66) {
    r = 255;
    g = Math.floor(255 * ((t - 0.33) / 0.33));
    b = 0;
  } else {
    r = 255;
    g = 255;
    b = Math.floor(255 * ((t - 0.66) / 0.34));
  }
  return "rgb(" + r + "," + g + "," + b + ")";
}
function draw() {
  var grid = frames[idx];
  var img = ctx.createImageData(nx, ny);
  for (var j = 0; j < ny; j++)
    for (var i = 0; i < nx; i++) {
      var v = grid[j][i];
      var k = (j * nx + i) * 4;
      var rgb = waveColor(v).match(/\\d+/g);
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
    ap = argparse.ArgumentParser(description="Run 2D wave equation and write HTML animation")
    ap.add_argument("--html", type=str, default="wave.html", help="Output HTML path")
    ap.add_argument("--profile", action="store_true", help="Print per-clause profile (L12, L13, recurrence)")
    ap.add_argument("--profile-einlang", action="store_true", help="Per-clause time + vectorized/hybrid/scalar summary")
    args = ap.parse_args()

    if args.profile or args.profile_einlang:
        os.environ["EINLANG_PROFILE_STATEMENTS"] = "1"
    if args.profile_einlang:
        os.environ["EINLANG_DEBUG_VECTORIZE"] = "1"

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

    h = np.asarray(exec_result.outputs.get("h"))
    if h is None or h.ndim != 3:
        print("Expected 3D array h[t,i,j]", file=sys.stderr)
        sys.exit(1)

    write_html_animation(h, args.html)
    print("Wave equation result: shape h =", h.shape)
    print("Open in browser: file://%s" % Path(args.html).resolve())


if __name__ == "__main__":
    main()
