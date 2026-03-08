#!/usr/bin/env python3
"""
Animate 2D heat equation ∂u/∂t = α∇²u computed in Einlang.

Run from repo root:
  python3 examples/heat_animation.py          # open HTML in browser (no pip)
  python3 examples/heat_animation.py --html heat.html
  python3 examples/heat_animation.py --save heat.gif   # needs matplotlib + pillow
  python3 examples/heat_animation.py --profile        # print cProfile top 30
  python3 examples/heat_animation.py --profile-out heat.prof   # save for snakeviz
  python3 examples/heat_animation.py --profile-einlang   # backend path (vectorized/hybrid/scalar) + cProfile einlang only
"""

import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import argparse
import numpy as np

# Einlang
from einlang.compiler.driver import CompilerDriver
from einlang.runtime.runtime import EinlangRuntime


# Hybrid path: only t is scalar (one loop), i,j vectorized.
# Allow more t steps by raising the recurrence loop limit for this script.
if "EINLANG_EINSTEIN_LOOP_MAX" not in os.environ:
    os.environ["EINLANG_EINSTEIN_LOOP_MAX"] = "2000"

# Run long enough that the heat propagates to the boundary (center to edge ~20 cells, r=0.2 => ~1200 steps).
HEAT_SOURCE = """
let r = 0.2;
let cx = 20;
let cy = 20;
let R2 = 196.0;
let u[0, i in 0..40, j in 0..40] = if ((i - cx) * (i - cx) + (j - cy) * (j - cy)) as f32 <= R2 { 10.0 * (1.0 - (((i - cx) * (i - cx) + (j - cy) * (j - cy)) as f32) / R2) } else { 0.0 };
let u[t in 1..1200, i in 1..39, j in 1..39] = u[t - 1, i, j] + r * (u[t - 1, i - 1, j] + u[t - 1, i + 1, j] + u[t - 1, i, j - 1] + u[t - 1, i, j + 1] - 4.0 * u[t - 1, i, j]);
u;
"""
def write_html_animation(u: np.ndarray, path: str, interval_ms: int = 120) -> None:
    """Write a self-contained HTML file that animates u[t,i,j]. No matplotlib/pip required."""
    u = np.asarray(u)
    n_frames, ny, nx = u.shape
    vmin = 0.0
    vmax = float(max(u.max(), 1.0))
    # Serialize frames as list of row-major grids
    frames = [u[t].tolist() for t in range(n_frames)]
    html = """<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>2D heat equation (Einlang)</title></head>
<body style="margin:1em;font-family:sans-serif">
<h1>2D heat equation &part;u/&part;t = &alpha;&nabla;&sup2;u</h1>
<p>t = <span id="t">0</span> / %(n_frames)s</p>
<canvas id="c" width="%(nx)s" height="%(ny)s" style="image-rendering:pixelated;image-rendering:crisp-edges;width:300px;height:300px"></canvas>
<script>
var frames = %(frames)s;
var vmin = %(vmin)s, vmax = %(vmax)s;
var n = frames.length;
var nx = frames[0][0].length, ny = frames[0].length;
var c = document.getElementById("c");
c.width = nx; c.height = ny;
var ctx = c.getContext("2d");
var idx = 0;
function heat(v) {
  var t = (v - vmin) / (vmax - vmin);
  var r, g, b;
  if (t < 0.33) {
    r = Math.floor(255 * (t / 0.33));
    g = 0;
    b = Math.floor(60 * (1 - t / 0.33));
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
      var rgb = heat(v).match(/\\d+/g);
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
    ap = argparse.ArgumentParser(description="Animate 2D heat diffusion from Einlang result")
    ap.add_argument("--save", type=str, default=None, help="Save as GIF (needs matplotlib+Pillow)")
    ap.add_argument("--html", type=str, default=None, help="Save as self-contained HTML (no pip)")
    ap.add_argument("--interval", type=int, default=120, help="Frame interval in ms")
    ap.add_argument("--profile", action="store_true", help="Run under cProfile and print top 30 by cumulative time")
    ap.add_argument("--profile-out", type=str, default=None, help="Save profile to FILE (e.g. heat.prof for snakeviz)")
    ap.add_argument("--profile-einlang", action="store_true", help="Enable backend statement profile (vectorized/hybrid/scalar) and cProfile filtered to einlang")
    args = ap.parse_args()

    if args.profile_einlang:
        os.environ["EINLANG_PROFILE_STATEMENTS"] = "1"
        os.environ["EINLANG_DEBUG_VECTORIZE"] = "1"
    if args.profile or args.profile_out or args.profile_einlang:
        import cProfile
        import pstats
        prof = cProfile.Profile()
        prof.enable()

    compiler = CompilerDriver()
    runtime = EinlangRuntime()
    result = compiler.compile(HEAT_SOURCE, "<heat_animation>", root_path=REPO_ROOT)
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
    if u is None or u.ndim != 3:
        print("Expected 3D array u[t,i,j]", file=sys.stderr)
        sys.exit(1)

    # HTML path: explicit --html or default when not using matplotlib
    html_path = args.html
    use_matplotlib = args.save is not None
    if use_matplotlib:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            from matplotlib.animation import FuncAnimation, PillowWriter
        except ImportError:
            use_matplotlib = False
            if html_path is None:
                html_path = "heat.html"
    if not use_matplotlib:
        out = html_path or "heat.html"
        write_html_animation(u, out, args.interval)
        print("Heat equation result: shape u =", u.shape)
        print("Open in browser: file://%s" % Path(out).resolve())
        if args.save:
            print("(GIF skipped: install matplotlib and Pillow for --save)")
        if args.profile or args.profile_out or args.profile_einlang:
            prof.disable()
            if args.profile_out:
                prof.dump_stats(args.profile_out)
                print("Profile saved to %s (e.g. python3 -m snakeviz %s)" % (args.profile_out, args.profile_out))
            ps = pstats.Stats(prof)
            ps.sort_stats(pstats.SortKey.CUMULATIVE)
            if args.profile_einlang:
                print("\n--- cProfile: einlang runtime/backend (vectorized vs scalar) ---")
                ps.print_stats("einlang", 40)
            else:
                ps.print_stats(30)
        return

    n_frames = u.shape[0]
    vmin, vmax = float(u.min()), float(u.max())
    if vmax <= vmin:
        vmax = vmin + 1.0

    fig, ax = plt.subplots()
    im = ax.imshow(u[0], cmap="hot", aspect="equal", vmin=vmin, vmax=vmax, origin="lower")
    plt.colorbar(im, ax=ax, label="u")
    ax.set_title("2D heat equation (Einlang)")
    ax.set_xlabel("j")
    ax.set_ylabel("i")

    def update(frame):
        im.set_data(u[frame])
        ax.set_title(f"2D heat equation  t = {frame}")
        return [im]

    anim = FuncAnimation(fig, update, frames=n_frames, interval=args.interval, blit=True)
    anim.save(args.save, writer=PillowWriter(fps=1000 // args.interval))
    print("Saved to", args.save)

    if args.profile or args.profile_out or args.profile_einlang:
        prof.disable()
        if args.profile_out:
            prof.dump_stats(args.profile_out)
            print("Profile saved to %s (e.g. python3 -m snakeviz %s)" % (args.profile_out, args.profile_out))
        ps = pstats.Stats(prof)
        ps.sort_stats(pstats.SortKey.CUMULATIVE)
        if args.profile_einlang:
            print("\n--- cProfile: einlang runtime/backend (vectorized vs scalar) ---")
            ps.print_stats("einlang", 40)
        else:
            ps.print_stats(30)


if __name__ == "__main__":
    main()
