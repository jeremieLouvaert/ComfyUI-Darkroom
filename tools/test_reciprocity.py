"""
Offline validation for the Reciprocity Failure node — "teeth before trust".
Run with the ComfyUI embedded python. No pytest; prints PASS/FAIL per check
and exits nonzero if any check fails.

Loads the node via a synthetic package shim so the node's `from ..utils...`
relative imports resolve WITHOUT triggering the top-level package __init__.

Teeth (per docs/reciprocity-derivation.md):
  1. CAST DIRECTION (headline) — E100 -> CYAN/BLUE, Provia -> MAGENTA, B&W -> no
     chroma. + NEGATIVE CONTROL: flipped-direction build makes E100 go RED.
  2. MONOTONIC WITH TIME + onset — cast magnitude grows with exposure; short t ~ identity.
  3. LUMINANCE ~PRESERVED — cast-only render keeps mean luma within ~0.05.
  4. SHADOW CRUSH — dark patch drops more (relative) than bright patch.
  5. NEUTRAL RAMP, NO INVERSION — a 0->1 grey ramp stays monotonic.
  6. strength=0 / short-time -> identity (early-exit).
"""

import os
import sys
import types
import importlib.util

import numpy as np
import torch

PACK_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ---------------------------------------------------------------------------
# Synthetic package shim (mirrors test_halftone.py).
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
    PKG + ".nodes.reciprocity", os.path.join(PACK_ROOT, "nodes", "reciprocity.py")
)
reciprocity = importlib.util.module_from_spec(spec)
sys.modules[PKG + ".nodes.reciprocity"] = reciprocity
spec.loader.exec_module(reciprocity)

Reciprocity = reciprocity.Reciprocity
_interp_log_time = reciprocity._interp_log_time
_cast_gain = reciprocity._cast_gain
FILM_TABLE = reciprocity.FILM_TABLE

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device = {device}, torch = {torch.__version__}")

PASS = True


def check(name, cond, detail=""):
    global PASS
    status = "PASS" if cond else "FAIL"
    if not cond:
        PASS = False
    print(f"  [{status}] {name}  {detail}")


def to_tensor(arr):
    return torch.from_numpy(np.ascontiguousarray(arr, dtype=np.float32)).unsqueeze(0)


def from_tensor(t):
    return t[0].cpu().numpy().astype(np.float32)


def run(img_np, **kw):
    node = Reciprocity()
    out = node.execute(to_tensor(img_np), **kw)[0]
    return from_tensor(out)


def grey(val, h=64, w=64):
    return np.full((h, w, 3), val, dtype=np.float32)


def luma(img):
    return 0.2126 * img[..., 0] + 0.7152 * img[..., 1] + 0.0722 * img[..., 2]


# ---------------------------------------------------------------------------
# 1. CAST DIRECTION (headline) + NEGATIVE CONTROL
# ---------------------------------------------------------------------------
print("\n[1] CAST DIRECTION on neutral grey 0.5 @ long exposure")

g = grey(0.5)

# Ektachrome E100 @ 100s -> CYAN/BLUE: (G+B)/2 > R.
e100 = run(g, film="Kodak Ektachrome E100", exposure_time=100.0, strength=1.0)
r, gg, b = e100[..., 0].mean(), e100[..., 1].mean(), e100[..., 2].mean()
check("Ektachrome E100 -> CYAN/BLUE ((G+B)/2 > R)", (gg + b) / 2.0 > r,
      f"R={r:.4f} G={gg:.4f} B={b:.4f}")

# Fuji Provia 100F @ 240s -> MAGENTA: (R+B)/2 > G.
prov = run(g, film="Fuji Provia 100F", exposure_time=240.0, strength=1.0)
pr, pg, pb = prov[..., 0].mean(), prov[..., 1].mean(), prov[..., 2].mean()
check("Fuji Provia 100F -> MAGENTA ((R+B)/2 > G)", (pr + pb) / 2.0 > pg,
      f"R={pr:.4f} G={pg:.4f} B={pb:.4f}")

# B&W (general) @ 100s -> no chroma: R ~ G ~ B.
bw = run(g, film="B&W (general)", exposure_time=100.0, strength=1.0)
br, bgr, bb = bw[..., 0].mean(), bw[..., 1].mean(), bw[..., 2].mean()
chroma = max(br, bgr, bb) - min(br, bgr, bb)
check("B&W (general) -> no chroma (max-min channel < 0.005)", chroma < 0.005,
      f"R={br:.4f} G={bgr:.4f} B={bb:.4f} spread={chroma:.5f}")

# NEGATIVE CONTROL: flip the cast direction -> E100 must go RED not cyan.
print("    negative control: flipped-direction E100 must go RED (cast test FAILS on flip)")
_, cast_e100 = _interp_log_time(FILM_TABLE["Kodak Ektachrome E100"], 100.0)
gains_flip = _cast_gain(cast_e100, flip=True)   # negated densities
# Apply the flipped gains directly to neutral grey in linear, luma-normalized.
lin = reciprocity.srgb_to_linear(grey(0.5))
flip_out = lin.copy()
flip_out[..., 0] *= gains_flip[0]
flip_out[..., 1] *= gains_flip[1]
flip_out[..., 2] *= gains_flip[2]
fr, fg, fb = flip_out[..., 0].mean(), flip_out[..., 1].mean(), flip_out[..., 2].mean()
neg_fired = fr > (fg + fb) / 2.0   # flipped => RED dominant => cyan test would fail
check("NEGATIVE CONTROL fired (flipped E100 is RED, R > (G+B)/2)", neg_fired,
      f"R={fr:.4f} G={fg:.4f} B={fb:.4f}")

# ---------------------------------------------------------------------------
# 2. MONOTONIC WITH TIME + onset
# ---------------------------------------------------------------------------
print("\n[2] Cast magnitude monotonic with exposure_time + onset ~ identity")


def cast_mag(film, t):
    """Max channel-gain deviation from neutral (1.0) at time t."""
    _, cast = _interp_log_time(FILM_TABLE[film], t)
    gains = _cast_gain(cast)
    return float(np.max(np.abs(gains - 1.0)))


mags = [cast_mag("Kodak Ektachrome E100", t) for t in (1.0, 10.0, 100.0)]
mono = mags[0] <= mags[1] <= mags[2] and mags[2] > mags[0]
check("E100 cast magnitude increases over [1,10,100]s", mono,
      f"mags={[round(m,4) for m in mags]}")

# Onset: at a short time (<= Portra onset 1s) the output ~ identity.
short = run(grey(0.5), film="Kodak Portra 400", exposure_time=0.5, strength=1.0)
ident_diff = np.abs(short - grey(0.5)).mean()
check("Portra @ 0.5s (below onset) ~ identity (mean abs diff < 1e-4)",
      ident_diff < 1e-4, f"mean_abs_diff={ident_diff:.6f}")

# ---------------------------------------------------------------------------
# 3. LUMINANCE ~PRESERVED (cast only, shadow crush aside)
# ---------------------------------------------------------------------------
print("\n[3] Cast-only render preserves mean luminance (renormalization works)")
for film, t in (("Kodak Ektachrome E100", 100.0), ("Fuji Provia 100F", 240.0),
                ("Fuji Velvia 50", 120.0)):
    _, cast = _interp_log_time(FILM_TABLE[film], t)
    gains = _cast_gain(cast)
    lin = reciprocity.srgb_to_linear(grey(0.5))
    cast_only = lin.copy()
    cast_only[..., 0] *= gains[0]
    cast_only[..., 1] *= gains[1]
    cast_only[..., 2] *= gains[2]
    out = reciprocity.linear_to_srgb(np.clip(cast_only, 0.0, 1.0))
    dl = abs(luma(out).mean() - luma(grey(0.5)).mean())
    check(f"{film} cast-only luma within 0.05 of input", dl < 0.05,
          f"delta_luma={dl:.4f}")

# ---------------------------------------------------------------------------
# 4. SHADOW CRUSH — dark patch drops more (relative) than bright patch
# ---------------------------------------------------------------------------
print("\n[4] Shadow crush — dark patch drops more (relative) than bright patch")
dark_in, bright_in = 0.1, 0.9
dark_out = run(grey(dark_in), film="Kodak Ektachrome E100",
               exposure_time=100.0, strength=1.0)
bright_out = run(grey(bright_in), film="Kodak Ektachrome E100",
                 exposure_time=100.0, strength=1.0)
dark_rel = (dark_in - luma(dark_out).mean()) / dark_in
bright_rel = (bright_in - luma(bright_out).mean()) / bright_in
check("dark (0.1) relative drop > bright (0.9) relative drop",
      dark_rel > bright_rel,
      f"dark_rel_drop={dark_rel:.4f} bright_rel_drop={bright_rel:.4f}")

# ---------------------------------------------------------------------------
# 5. NEUTRAL RAMP, NO INVERSION
# ---------------------------------------------------------------------------
print("\n[5] Neutral 0->1 grey ramp stays monotonic (no inversion)")
W = 256
xramp = np.linspace(0.0, 1.0, W, dtype=np.float32)
ramp = np.repeat(xramp[None, :], 16, axis=0)
ramp_img = np.stack([ramp, ramp, ramp], axis=-1)
out_ramp = run(ramp_img, film="Kodak Ektachrome E100", exposure_time=120.0, strength=1.0)
col = luma(out_ramp).mean(axis=0)
diffs = np.diff(col)
# Allow tiny float noise; require non-decreasing.
mono_ramp = bool(np.all(diffs >= -1e-4))
check("ramp luma non-decreasing across the ramp (no inversion)", mono_ramp,
      f"min_step={diffs.min():.6f} endpoints=({col[0]:.4f},{col[-1]:.4f})")

# ---------------------------------------------------------------------------
# 6. strength=0 / short-time -> identity (early-exit)
# ---------------------------------------------------------------------------
print("\n[6] strength=0 and short-time -> identity (early-exit)")
s0 = run(grey(0.5), film="Kodak Ektachrome E100", exposure_time=100.0, strength=0.0)
check("strength=0 -> exact identity", np.array_equal(s0, grey(0.5)),
      f"mean_abs_diff={np.abs(s0 - grey(0.5)).mean():.6f}")
st = run(grey(0.5), film="Kodak Ektachrome E100", exposure_time=0.5, strength=1.0)
# E100 first table point is 1s; 0.5s is below onset -> identity.
check("E100 @ 0.5s (below onset) -> exact identity", np.array_equal(st, grey(0.5)),
      f"mean_abs_diff={np.abs(st - grey(0.5)).mean():.6f}")

# ---------------------------------------------------------------------------
print("\n" + ("ALL CHECKS PASSED" if PASS else "SOME CHECKS FAILED"))
sys.exit(0 if PASS else 1)
