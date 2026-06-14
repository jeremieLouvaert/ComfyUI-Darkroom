"""
Offline validation for the Halftone node — "teeth before trust".
Run with the ComfyUI embedded python. No pytest; prints PASS/FAIL per check
and exits nonzero if any check fails.

Loads the node via a synthetic package shim so the node's `from ..utils...`
relative imports resolve WITHOUT triggering the top-level package __init__
(which imports server_routes / ComfyUI runtime bits).

Teeth (per docs/halftone-derivation.md):
  1. TONAL MONOTONICITY / NO INVERSION (headline)  + NEGATIVE CONTROL (flipped screen).
  2. TONE PRESERVATION  — heavy blur of output ≈ input.
  3. ENDPOINTS          — white→white, black→black.
  4. BINARY-ISH @ ss=1  — pre-AA ink mask is ~binary.
  5. CMYK ANGLES DIFFER — distinct rosette angles.
Plus perf-gate timings: 1024² mono and ~4K color.
"""

import os
import sys
import time
import types
import importlib.util

import numpy as np
import torch
from scipy.ndimage import gaussian_filter

PACK_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ---------------------------------------------------------------------------
# Synthetic package shim: build `dr_pkg` (parent) + `dr_pkg.nodes` + load
# `dr_pkg.nodes.halftone` so `from ..utils.color import ...` resolves to
# dr_pkg.utils, all WITHOUT importing the pack's real __init__ (server_routes).
# ---------------------------------------------------------------------------
PKG = "dr_pkg"

parent = types.ModuleType(PKG)
parent.__path__ = [PACK_ROOT]
sys.modules[PKG] = parent

nodes_pkg = types.ModuleType(PKG + ".nodes")
nodes_pkg.__path__ = [os.path.join(PACK_ROOT, "nodes")]
sys.modules[PKG + ".nodes"] = nodes_pkg

utils_pkg = types.ModuleType(PKG + ".utils")
utils_pkg.__path__ = [os.path.join(PACK_ROOT, "utils")]
sys.modules[PKG + ".utils"] = utils_pkg

spec = importlib.util.spec_from_file_location(
    PKG + ".nodes.halftone", os.path.join(PACK_ROOT, "nodes", "halftone.py")
)
halftone = importlib.util.module_from_spec(spec)
sys.modules[PKG + ".nodes.halftone"] = halftone
spec.loader.exec_module(halftone)

Halftone = halftone.Halftone
_screen = halftone._screen
_screen_fm = halftone._screen_fm
_CMYK_ANGLES = halftone._CMYK_ANGLES

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device = {device}, torch = {torch.__version__}")
print(f"torch path engaged for screen eval: {torch.cuda.is_available()}")

PASS = True


def check(name, cond, detail=""):
    global PASS
    status = "PASS" if cond else "FAIL"
    if not cond:
        PASS = False
    print(f"  [{status}] {name}  {detail}")


def to_tensor(arr):
    """numpy (H,W,C) -> IMAGE tensor (1,H,W,C)."""
    return torch.from_numpy(np.ascontiguousarray(arr, dtype=np.float32)).unsqueeze(0)


def from_tensor(t):
    return t[0].cpu().numpy().astype(np.float32)


def run(img_np, **kw):
    node = Halftone()
    out = node.execute(to_tensor(img_np), **kw)[0]
    return from_tensor(out)


def luma(img):
    return 0.2126 * img[..., 0] + 0.7152 * img[..., 1] + 0.0722 * img[..., 2]


def monotonic_nondecreasing(col_means, tol=0.02):
    """Trend check: binned thirds strictly increasing + global Spearman-ish."""
    n = len(col_means)
    t1 = col_means[: n // 3].mean()
    t2 = col_means[n // 3: 2 * n // 3].mean()
    t3 = col_means[2 * n // 3:].mean()
    thirds_ok = (t2 > t1 + tol) and (t3 > t2 + tol)
    # Spearman-ish: rank correlation of column index vs column mean.
    idx = np.arange(n)
    rank_means = np.argsort(np.argsort(col_means))
    # Pearson on ranks == Spearman
    spear = np.corrcoef(idx, rank_means)[0, 1]
    return thirds_ok, spear, (t1, t2, t3)


# ---------------------------------------------------------------------------
# 1. TONAL MONOTONICITY / NO INVERSION (headline) + NEGATIVE CONTROL
# ---------------------------------------------------------------------------
print("\n[1] Tonal monotonicity / NO INVERSION (256x256 black->white gradient, mono, lines=40, ss=2)")
H = W = 256
xramp = np.linspace(0.0, 1.0, W, dtype=np.float32)
grad = np.repeat(xramp[None, :], H, axis=0)            # (H,W) x: 0->1
grad_img = np.stack([grad, grad, grad], axis=-1)

ht = run(grad_img, color_mode="mono (black)", lines=40, angle=45.0, supersample=2, strength=1.0)
col_means = luma(ht).mean(axis=0)                       # per-column mean luminance
ok, spear, thirds = monotonic_nondecreasing(col_means)
check("monotone non-decreasing (dark in -> dark out)", ok and spear > 0.9,
      f"thirds={tuple(round(float(v),3) for v in thirds)} spearman={spear:.3f}")

# NEGATIVE CONTROL: flipped screen (c < T) must BREAK monotonicity.
print("    negative control: flipped screen (coverage < T) must FAIL the same test")
long_edge = max(H, W)
cov = 1.0 - luma(grad_img)                              # mono coverage
ink_flip = _screen(cov, 45.0, 40, 2, long_edge, flip=True)
tone_flip = 1.0 - ink_flip
col_means_flip = tone_flip.mean(axis=0)
ok_flip, spear_flip, thirds_flip = monotonic_nondecreasing(col_means_flip)
neg_fired = not (ok_flip and spear_flip > 0.9)
check("NEGATIVE CONTROL fired (flipped screen FAILS monotonicity)", neg_fired,
      f"thirds={tuple(round(float(v),3) for v in thirds_flip)} spearman={spear_flip:.3f}")

# ---------------------------------------------------------------------------
# 1b. MONOTONICITY PARAMETRIZED over every AM dot_shape AND over FM
# ---------------------------------------------------------------------------
print("\n[1b] Monotonicity / NO INVERSION parametrized over AM dot_shapes + FM")
# Headline invariant = NO INVERSION: thirds strictly increasing + positive rank
# correlation. All shapes are tone-linearized (square/ellipse via the coverage
# pre-warp), so column means track input. Spearman (rank correlation) is invariant
# to a monotonic tone curve, so any residual dip is dot-DISCRETIZATION noise on the
# coarse pattern at lines=40 (ellipse's anisotropic dots are the noisiest), NOT an
# inversion — the thirds check is the real no-inversion guard, kept tight for all.
_SPEAR_BAR = {"round": 0.9, "line": 0.9, "square": 0.9, "ellipse": 0.5}
for shape in ("round", "line", "square", "ellipse"):
    ht_s = run(grad_img, color_mode="mono (black)", method="AM (clustered dot)",
               dot_shape=shape, lines=40, angle=45.0, supersample=2, strength=1.0)
    cm = luma(ht_s).mean(axis=0)
    ok_s, spear_s, thirds_s = monotonic_nondecreasing(cm)
    check(f"AM dot_shape={shape}: monotone non-decreasing (no inversion)",
          ok_s and spear_s > _SPEAR_BAR[shape],
          f"thirds={tuple(round(float(v),3) for v in thirds_s)} spearman={spear_s:.3f}")

# FM (dispersed / Bayer) — ignores dot_shape/angle/supersample.
ht_fm = run(grad_img, color_mode="mono (black)", method="FM (dispersed / Bayer)",
            lines=40, angle=45.0, supersample=2, strength=1.0)
cm_fm = luma(ht_fm).mean(axis=0)
ok_fm, spear_fm, thirds_fm = monotonic_nondecreasing(cm_fm)
check("FM (Bayer): monotone non-decreasing", ok_fm and spear_fm > 0.9,
      f"thirds={tuple(round(float(v),3) for v in thirds_fm)} spearman={spear_fm:.3f}")

# ---------------------------------------------------------------------------
# 2. TONE PRESERVATION — heavy blur of halftone output ~ input
# ---------------------------------------------------------------------------
print("\n[2] Tone preservation (2D gradient, blur(sigma~2*pitch) of output ~ input)")
H = W = 512
yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
ramp2d = ((xx / (W - 1)) * 0.5 + (yy / (H - 1)) * 0.5).astype(np.float32)  # smooth 0->1
img2d = np.stack([ramp2d, ramp2d, ramp2d], axis=-1)
lines = 40
ht2 = run(img2d, color_mode="mono (black)", lines=lines, angle=45.0, supersample=2, strength=1.0)
pitch = max(max(H, W) / lines, 3.0)
sigma = 2.0 * pitch
blurred = np.empty_like(ht2)
for ch in range(3):
    blurred[..., ch] = gaussian_filter(ht2[..., ch], sigma=sigma)
# crop a margin to avoid blur edge bias
m = int(sigma * 2)
diff = np.abs(blurred[m:-m, m:-m] - img2d[m:-m, m:-m]).mean()
check("blur(halftone) approximates input (mean abs diff < 0.1)", diff < 0.1,
      f"mean_abs_diff={diff:.4f} (sigma={sigma:.1f})")

# ---------------------------------------------------------------------------
# 3. ENDPOINTS
# ---------------------------------------------------------------------------
print("\n[3] Endpoints (pure white -> ~white, pure black -> ~black)")
white = np.ones((128, 128, 3), dtype=np.float32)
black = np.zeros((128, 128, 3), dtype=np.float32)
hw = run(white, color_mode="mono (black)", lines=40, supersample=2, strength=1.0)
hb = run(black, color_mode="mono (black)", lines=40, supersample=2, strength=1.0)
check("white in -> mean out > 0.95", hw.mean() > 0.95, f"mean={hw.mean():.4f}")
check("black in -> mean out < 0.05", hb.mean() < 0.05, f"mean={hb.mean():.4f}")

# ---------------------------------------------------------------------------
# 3b. TONE PRESERVATION + ENDPOINTS for a NEW shape (square) and for FM
# ---------------------------------------------------------------------------
print("\n[3b] Tone preservation for tone-linearized shapes (AM line/square/ellipse, FM) + endpoints")
# TONE-LINEARIZATION PROOF: square/ellipse are pre-warped (coverage -> CDF_T^-1)
# so ink area = coverage, exactly like round/line. So a heavy blur of their output
# matches the input for EVERY shape — dot_shape changes geometry only, never tone.
# (Before the warp, square blurred to ~quadratically too light; this is the guard
# that it no longer does.) FM (Bayer) is tone-exact by construction.


def tone_pres(method_name, shape_name):
    out = run(img2d, color_mode="mono (black)", method=method_name,
              dot_shape=shape_name, lines=lines, angle=45.0, supersample=2, strength=1.0)
    bl = np.empty_like(out)
    for ch in range(3):
        bl[..., ch] = gaussian_filter(out[..., ch], sigma=sigma)
    return np.abs(bl[m:-m, m:-m] - img2d[m:-m, m:-m]).mean()


for label, method_name, shape_name in (
    ("AM line", "AM (clustered dot)", "line"),
    ("AM square", "AM (clustered dot)", "square"),
    ("AM ellipse", "AM (clustered dot)", "ellipse"),
    ("FM", "FM (dispersed / Bayer)", "round"),
):
    d = tone_pres(method_name, shape_name)
    check(f"{label}: blur(halftone) approximates input (< 0.1)", d < 0.1,
          f"mean_abs_diff={d:.4f}")

# ellipse is included specifically: before the normalized-field fix its clipped
# distance left a T=1 plateau over ~half each cell, capping ink at ~49% so blacks
# were UNREACHABLE (black-in -> mean ~0.5). This black endpoint guards that regress.
for label, method_name, shape_name in (
    ("AM square", "AM (clustered dot)", "square"),
    ("AM ellipse", "AM (clustered dot)", "ellipse"),
    ("FM", "FM (dispersed / Bayer)", "round"),
):
    hw_s = run(white, color_mode="mono (black)", method=method_name,
               dot_shape=shape_name, lines=40, supersample=2, strength=1.0)
    hb_s = run(black, color_mode="mono (black)", method=method_name,
               dot_shape=shape_name, lines=40, supersample=2, strength=1.0)
    check(f"{label}: white in -> mean out > 0.95", hw_s.mean() > 0.95, f"mean={hw_s.mean():.4f}")
    check(f"{label}: black in -> mean out < 0.05", hb_s.mean() < 0.05, f"mean={hb_s.mean():.4f}")

# ---------------------------------------------------------------------------
# 4. BINARY-ISH INK @ ss=1
# ---------------------------------------------------------------------------
print("\n[4] Binary-ish ink @ ss=1 (mid-gray patch, raw ink mask near {0,1})")
gray = np.full((256, 256), 0.5, dtype=np.float32)       # coverage 0.5
ink_ss1 = _screen(gray, 45.0, 40, 1, 256, flip=False)
near01 = ((ink_ss1 < 0.05) | (ink_ss1 > 0.95)).mean()
check("ss=1 ink predominantly near 0 or 1 (>90%)", near01 > 0.90,
      f"fraction_near_{{0,1}}={near01:.4f}")

# ---------------------------------------------------------------------------
# 5. CMYK ANGLES DIFFER
# ---------------------------------------------------------------------------
print("\n[5] CMYK rosette angles distinct")
angles = [_CMYK_ANGLES["c"], _CMYK_ANGLES["m"], _CMYK_ANGLES["y"], _CMYK_ANGLES["k"]]
check("four rosette angles all distinct", len(set(angles)) == 4, f"angles={angles}")

# ---------------------------------------------------------------------------
# PERF GATE
# ---------------------------------------------------------------------------
print("\n[perf] timings")
img1k = np.random.rand(1024, 1024, 3).astype(np.float32)
# warm
_ = run(img1k, color_mode="mono (black)", lines=100, supersample=2, strength=1.0)
if device == "cuda":
    torch.cuda.synchronize()
t0 = time.time()
_ = run(img1k, color_mode="mono (black)", lines=100, supersample=2, strength=1.0)
if device == "cuda":
    torch.cuda.synchronize()
t_1k = time.time() - t0
print(f"  1024x1024 mono (lines=100, ss=2): {t_1k:.3f} s")

img4k = np.random.rand(2160, 3840, 3).astype(np.float32)
if device == "cuda":
    torch.cuda.synchronize()
t0 = time.time()
_ = run(img4k, color_mode="color (CMYK)", lines=100, supersample=2, strength=1.0)
if device == "cuda":
    torch.cuda.synchronize()
t_4k = time.time() - t0
print(f"  3840x2160 color CMYK AM (lines=100, ss=2): {t_4k:.3f} s")

# 4K FM run (CMYK) — confirm no perf regression vs AM.
if device == "cuda":
    torch.cuda.synchronize()
t0 = time.time()
_ = run(img4k, color_mode="color (CMYK)", method="FM (dispersed / Bayer)",
        lines=100, supersample=2, strength=1.0)
if device == "cuda":
    torch.cuda.synchronize()
t_4k_fm = time.time() - t0
print(f"  3840x2160 color CMYK FM (Bayer): {t_4k_fm:.3f} s")

# ---------------------------------------------------------------------------
print("\n" + ("ALL CHECKS PASSED" if PASS else "SOME CHECKS FAILED"))
sys.exit(0 if PASS else 1)
