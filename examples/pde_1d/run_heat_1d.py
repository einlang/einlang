#!/usr/bin/env python3
"""
Run 1D heat equation (heat_1d.ein) and write HTML heatmap (time vs space).

From repo root:
  python3 examples/pde_1d/run_heat_1d.py
  python3 examples/pde_1d/run_heat_1d.py --html heat_1d.html
"""

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np
from einlang.compiler.driver import CompilerDriver
from einlang.runtime.runtime import EinlangRuntime

EXAMPLE_DIR = Path(__file__).resolve().parent
HEAT_EIN = EXAMPLE_DIR / "heat_1d.ein"


def write_html_heatmap(u: np.ndarray, path: str) -> None:
    """u: (n_frames, nx). Static heatmap: rows = time, cols = space."""
    u = np.asarray(u)
    n_frames, nx = u.shape
    umin = float(u.min())
    umax = float(u.max())
    if umax <= umin:
        umax = umin + 1.0
    grid = u.tolist()
    html = """<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>1D heat (Einlang)</title></head>
<body style="margin:1em;font-family:sans-serif">
<h1>1D heat equation</h1>
<p>Rows = time (0..%(n_frames)s), cols = space (0..%(nx)s)</p>
<canvas id="c" width="%(nx)s" height="%(n_frames)s" style="image-rendering:pixelated;width:400px;height:200px"></canvas>
<script>
var grid = %(grid)s;
var umin = %(umin)s, umax = %(umax)s;
var ny = grid.length, nx = grid[0].length;
var c = document.getElementById("c");
c.width = nx; c.height = ny;
var ctx = c.getContext("2d");
for (var j = 0; j < ny; j++)
  for (var i = 0; i < nx; i++) {
    var t = (grid[j][i] - umin) / (umax - umin);
    ctx.fillStyle = "rgb(" + Math.floor(255*t) + ",0," + Math.floor(255*(1-t)) + ")";
    ctx.fillRect(i, j, 1, 1);
  }
</script>
</body></html>
""" % {
        "n_frames": n_frames,
        "nx": nx,
        "grid": json.dumps(grid),
        "umin": umin,
        "umax": umax,
    }
    Path(path).write_text(html, encoding="utf-8")


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Run 1D heat and write HTML")
    ap.add_argument("--html", type=str, default="heat_1d.html", help="Output HTML path")
    args = ap.parse_args()

    source = HEAT_EIN.read_text(encoding="utf-8")
    compiler = CompilerDriver()
    runtime = EinlangRuntime()
    result = compiler.compile(source, str(HEAT_EIN), root_path=EXAMPLE_DIR)
    if not result.success:
        err = getattr(result, "tcx", None)
        if err and getattr(err, "reporter", None):
            print(err.reporter.format_all_errors(), file=sys.stderr)
        sys.exit(1)
    exec_result = runtime.execute(result)
    if exec_result.error:
        print(exec_result.error, file=sys.stderr)
        sys.exit(1)

    u = np.asarray(exec_result.outputs.get("u"))
    if u is None or u.ndim != 2:
        print("Expected 2D array u[t, i]", file=sys.stderr)
        sys.exit(1)

    write_html_heatmap(u, args.html)
    print("1D heat result: shape u =", u.shape)
    print("Open in browser: file://%s" % Path(args.html).resolve())


if __name__ == "__main__":
    main()
