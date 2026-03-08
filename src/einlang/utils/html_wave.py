"""
Write a self-contained HTML file that animates a 3D array [time, y, x] as a wave visualization.
Used by the CLI --html option and by examples (wave_2d, etc.) when they delegate to this.
"""

import json
from pathlib import Path
from typing import Union

import numpy as np


def write_wave_html(
    data_3d: Union[np.ndarray, list],
    path: Union[str, Path],
    interval_ms: int = 80,
    title: str = "2D wave equation (Einlang)",
) -> None:
    """
    Write HTML that animates data_3d[t, i, j] on a canvas.

    Args:
        data_3d: 3D array (n_frames, ny, nx). Will be converted via np.asarray.
        path: Output file path.
        interval_ms: Milliseconds between frames.
        title: Page title.
    """
    h = np.asarray(data_3d)
    if h.ndim != 3:
        raise ValueError(f"write_wave_html expects 3D array [time, y, x], got ndim={h.ndim}")
    n_frames, ny, nx = h.shape
    vmin = float(h.min())
    vmax = float(h.max())
    if vmax <= vmin:
        vmax = vmin + 1.0
    frames = [h[t].tolist() for t in range(n_frames)]
    html = """<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>%(title)s</title></head>
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
        "title": title,
        "n_frames": n_frames,
        "nx": nx,
        "ny": ny,
        "frames": json.dumps(frames),
        "vmin": vmin,
        "vmax": vmax,
        "interval_ms": interval_ms,
    }
    Path(path).write_text(html, encoding="utf-8")
