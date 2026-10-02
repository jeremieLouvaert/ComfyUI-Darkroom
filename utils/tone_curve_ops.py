"""
Freeform tone curve + Contrast / Shadows / Midtones / Highlights, as used by
Film Stock (Color) on top of the film look (docs/film-stock-tone-curve.md).

The JS editor (web/darkroom_freeform_curve.js) implements the SAME functions
below, line for line, so the curve drawn on the node is the curve applied
(tools/test_film_tone_curve.py checks JS against Python under Node).

Composition, on display-referred values, per channel (like Capture One's RGB
curve):  T(x) = bumps( contrast( spline(x) ) ), then forced non-decreasing and
clipped to [0, 1].
"""

import math
import re

import numpy as np
import torch

IDENTITY_POINTS = "0,0;1,1"
TABLE_SIZE = 4096
BUMP_HALF_WIDTH = 0.25       # shadows 0-0.5, midtones 0.25-0.75, highlights 0.5-1
BUMP_AMOUNT = 0.15           # output shift at the bump centre for a slider at 100
CONTRAST_K = 0.6             # S-curve strength at contrast 100 (|k| < 1 keeps it monotonic)
_DEC = re.compile(r"^\s*[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?\s*$")   # same rule as the JS twin


def parse_points(text):
    """'x,y;x,y;...' -> sorted list of (x, y) clamped to [0, 1]; identity on any error."""
    try:
        pts = []
        for chunk in str(text).split(";"):
            if chunk.strip():
                x, y = chunk.split(",")
                if not (_DEC.match(x) and _DEC.match(y)):
                    raise ValueError(f"not a plain number: {chunk!r}")
                pts.append((min(max(float(x), 0.0), 1.0), min(max(float(y), 0.0), 1.0)))
        pts.sort(key=lambda p: p[0])
        dedup = []
        for p in pts:                      # drop points sharing an x (keep the last)
            if dedup and abs(p[0] - dedup[-1][0]) < 1e-6:
                dedup[-1] = p
            else:
                dedup.append(p)
        if len(dedup) < 2 or not all(math.isfinite(v) for p in dedup for v in p):
            raise ValueError("need at least two points")
        return dedup
    except Exception as e:
        print(f"[Darkroom] tone curve: could not read points {text!r} ({e}); using identity")
        return [(0.0, 0.0), (1.0, 1.0)]


def natural_cubic(points, x):
    """Natural cubic spline through points, evaluated at x (numpy). Flat beyond
    the end points, like Capture One."""
    xs = np.array([p[0] for p in points], dtype=np.float64)
    ys = np.array([p[1] for p in points], dtype=np.float64)
    n = len(xs)
    x = np.asarray(x, dtype=np.float64)
    if n == 2:
        t = np.clip((x - xs[0]) / max(xs[1] - xs[0], 1e-9), 0.0, 1.0)
        return ys[0] + t * (ys[1] - ys[0])
    h = np.diff(xs)
    # second derivatives M, natural ends (M0 = Mn-1 = 0): tridiagonal solve
    a = h[:-1].copy()
    b = 2.0 * (h[:-1] + h[1:])
    c = h[1:].copy()
    d = 6.0 * ((ys[2:] - ys[1:-1]) / h[1:] - (ys[1:-1] - ys[:-2]) / h[:-1])
    m = len(b)
    for i in range(1, m):                  # Thomas algorithm
        w = a[i] / b[i - 1]
        b[i] -= w * c[i - 1]
        d[i] -= w * d[i - 1]
    M_in = np.zeros(m)
    M_in[-1] = d[-1] / b[-1]
    for i in range(m - 2, -1, -1):
        M_in[i] = (d[i] - c[i] * M_in[i + 1]) / b[i]
    M = np.concatenate([[0.0], M_in, [0.0]])
    xc = np.clip(x, xs[0], xs[-1])
    k = np.clip(np.searchsorted(xs, xc, side="right") - 1, 0, n - 2)
    hk = h[k]
    t1 = xs[k + 1] - xc
    t0 = xc - xs[k]
    return (M[k] * t1 ** 3 / (6 * hk) + M[k + 1] * t0 ** 3 / (6 * hk)
            + (ys[k] / hk - M[k] * hk / 6) * t1 + (ys[k + 1] / hk - M[k + 1] * hk / 6) * t0)


def _bump(y, centre):
    d = np.abs(y - centre)
    return np.where(d < BUMP_HALF_WIDTH, 0.5 * (1.0 + np.cos(np.pi * d / BUMP_HALF_WIDTH)), 0.0)


def compose(points, contrast=0.0, shadows=0.0, midtones=0.0, highlights=0.0, size=TABLE_SIZE):
    """The full transfer curve sampled at `size` points on [0, 1] (numpy float64)."""
    x = np.linspace(0.0, 1.0, size)
    y = natural_cubic(points, x)
    k = CONTRAST_K * contrast / 100.0
    y = y - k * np.sin(2.0 * np.pi * y) / (2.0 * np.pi)
    y = (y + BUMP_AMOUNT * (shadows / 100.0 * _bump(y, 0.25)
                            + midtones / 100.0 * _bump(y, 0.5)
                            + highlights / 100.0 * _bump(y, 0.75)))
    return np.clip(np.maximum.accumulate(y), 0.0, 1.0)


def is_identity(points, contrast, shadows, midtones, highlights):
    return (len(points) == 2 and points[0] == (0.0, 0.0) and points[1] == (1.0, 1.0)
            and contrast == 0 and shadows == 0 and midtones == 0 and highlights == 0)


def apply_table(img, table):
    """Apply a sampled curve (numpy, values on [0,1]) to every channel of img (torch)."""
    t = torch.as_tensor(table.astype(np.float32), device=img.device)
    n = t.numel() - 1
    pos = img.clamp(0.0, 1.0) * n
    i0 = pos.floor().clamp(max=n - 1).long()
    f = pos - i0
    return t[i0] * (1.0 - f) + t[i0 + 1] * f
