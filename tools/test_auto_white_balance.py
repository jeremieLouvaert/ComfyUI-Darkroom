"""
Offline validation for the Auto White Balance node — "teeth before trust".
Run with the ComfyUI embedded python. No pytest; prints PASS/FAIL per check
and exits nonzero if any check fails.

Loads the node via a synthetic package shim so the node's `from ..utils...`
relative imports resolve WITHOUT triggering the top-level package __init__
(which imports server_routes / ComfyUI runtime bits).

Teeth (per docs/auto-white-balance-derivation.md):
  1. CAST RECOVERY (headline) + NEGATIVE CONTROL (strength=0 must NOT recover).
  2. GRAY WORLD EXACTNESS — 3 corrected channel means equal (in linear).
  3. NEUTRAL STAYS NEUTRAL — balanced image -> output ~ input, gains ~ 1.
  4. BRIGHTNESS PRESERVED — mean Rec.709 luminance ~ unchanged.
  5. ALL METHODS valid in [0,1], no NaN; flat-gray -> no NaN / no wild change.

SPACE NOTE: the illuminant cast and ALL imbalance / gray-world-exactness
measurements are done in LINEAR light. That is the space the von Kries
correction operates in, the space gray-world equalizes, and the space the cast
is physically a per-channel multiply in. Measuring there is the honest place to
assert the algorithm's defining properties. (Final sRGB output is only used for
the [0,1] / NaN validity checks.)
"""

import os
import sys
import types
import importlib.util

import numpy as np
import torch

PACK_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ---------------------------------------------------------------------------
# Synthetic package shim (mirrors tools/test_halftone.py).
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
    PKG + ".nodes.auto_white_balance",
    os.path.join(PACK_ROOT, "nodes", "auto_white_balance.py"),
)
awb_mod = importlib.util.module_from_spec(spec)
sys.modules[PKG + ".nodes.auto_white_balance"] = awb_mod
spec.loader.exec_module(awb_mod)

AutoWhiteBalance = awb_mod.AutoWhiteBalance

# color utils from the same shimmed package, for measuring in linear space.
color_spec = importlib.util.spec_from_file_location(
    PKG + ".utils.color", os.path.join(PACK_ROOT, "utils", "color.py")
)
color = importlib.util.module_from_spec(color_spec)
sys.modules[PKG + ".utils.color"] = color
color_spec.loader.exec_module(color)
srgb_to_linear = color.srgb_to_linear
luminance_rec709 = color.luminance_rec709

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
    node = AutoWhiteBalance()
    out = node.execute(to_tensor(img_np), **kw)[0]
    return from_tensor(out)


def channel_means_linear(img_srgb):
    """Mean of each channel in LINEAR light."""
    lin = srgb_to_linear(img_srgb)
    return np.array([lin[..., c].mean() for c in range(3)], dtype=np.float64)


def imbalance_linear(img_srgb):
    m = channel_means_linear(img_srgb)
    return float(m.max() / m.min())


# ---------------------------------------------------------------------------
# Build a balanced base image (roughly equal channel means) + a known cast.
# Cast is applied in LINEAR light (physical illuminant), then back to sRGB so
# the node sees a normal sRGB image. Imbalance measured in linear throughout.
# ---------------------------------------------------------------------------
rng = np.random.default_rng(42)
H = W = 256
base = rng.uniform(0.15, 0.85, size=(H, W, 3)).astype(np.float32)
# Force the base to be near-balanced in linear so the cast is the only imbalance.
base_lin = srgb_to_linear(base)
target_mean = base_lin.reshape(-1, 3).mean(axis=0).mean()
for c in range(3):
    base_lin[..., c] *= target_mean / base_lin[..., c].mean()
base = color.linear_to_srgb(np.clip(base_lin, 0.0, 1.0))

CAST = np.array([1.3, 1.0, 0.7], dtype=np.float32)  # warm: R up, B down
cast_lin = np.clip(srgb_to_linear(base) * CAST[None, None, :], 0.0, 1.0)
cast_img = color.linear_to_srgb(cast_lin)

base_imb = imbalance_linear(base)
cast_imb = imbalance_linear(cast_img)

# ---------------------------------------------------------------------------
# 1. CAST RECOVERY (headline) + NEGATIVE CONTROL
# ---------------------------------------------------------------------------
print("\n[1] CAST RECOVERY (linear-space imbalance) + NEGATIVE CONTROL")
print(f"    base imbalance (should be ~1)       = {base_imb:.4f}")
print(f"    cast imbalance (R*1.3,B*0.7 linear) = {cast_imb:.4f}")

for method in ("Gray World", "Shades of Gray"):
    corr = run(cast_img, method=method, strength=1.0)
    corr_imb = imbalance_linear(corr)
    check(f"{method}: corrected imbalance < 1.10 and < cast",
          corr_imb < 1.10 and corr_imb < cast_imb,
          f"corrected_imbalance={corr_imb:.4f} (cast={cast_imb:.4f})")

# NEGATIVE CONTROL: strength=0 must NOT recover (imbalance ~ unchanged).
neg = run(cast_img, method="Shades of Gray", strength=0.0)
neg_imb = imbalance_linear(neg)
neg_unchanged = abs(neg_imb - cast_imb) < 1e-4
check("NEGATIVE CONTROL: strength=0 does NOT recover (imbalance unchanged)",
      neg_unchanged,
      f"neg_imbalance={neg_imb:.4f} vs cast={cast_imb:.4f} (delta={abs(neg_imb-cast_imb):.2e})")

# ---------------------------------------------------------------------------
# 2. GRAY WORLD EXACTNESS (linear): 3 corrected channel means equal.
# ---------------------------------------------------------------------------
print("\n[2] GRAY WORLD EXACTNESS (corrected channel means equal in LINEAR)")
gw = run(cast_img, method="Gray World", strength=1.0)
gw_means = channel_means_linear(gw)
spread = float(gw_means.max() - gw_means.min())
check("Gray World: 3 corrected channel means equal within 1e-3", spread < 1e-3,
      f"means={[round(float(x),5) for x in gw_means]} spread={spread:.2e}")

# ---------------------------------------------------------------------------
# 3. NEUTRAL STAYS NEUTRAL: balanced image -> output ~ input, gains ~ 1.
# ---------------------------------------------------------------------------
print("\n[3] NEUTRAL STAYS NEUTRAL (balanced image unchanged)")
for method in ("Gray World", "White Patch", "Shades of Gray", "Gray Edge"):
    out = run(base, method=method, strength=1.0)
    mad = float(np.abs(out - base).mean())
    # gains ~1 <=> means unchanged: compare linear means before/after.
    m_in = channel_means_linear(base)
    m_out = channel_means_linear(out)
    gain_dev = float(np.abs(m_out / m_in - 1.0).max())
    check(f"{method}: balanced output ~ input (mad<0.02, gain dev<0.05)",
          mad < 0.02 and gain_dev < 0.05,
          f"mean_abs_diff={mad:.4f} max_gain_dev={gain_dev:.4f}")

# ---------------------------------------------------------------------------
# 4. BRIGHTNESS PRESERVED: mean Rec.709 luminance ~ cast input's.
# ---------------------------------------------------------------------------
print("\n[4] BRIGHTNESS PRESERVED (mean Rec.709 luminance ~ cast input)")
lum_cast = float(luminance_rec709(cast_img).mean())
for method in ("Gray World", "White Patch", "Shades of Gray", "Gray Edge"):
    out = run(cast_img, method=method, strength=1.0)
    lum_out = float(luminance_rec709(out).mean())
    check(f"{method}: |lum_out - lum_cast| < 0.1", abs(lum_out - lum_cast) < 0.1,
          f"lum_cast={lum_cast:.4f} lum_out={lum_out:.4f} d={abs(lum_out-lum_cast):.4f}")

# ---------------------------------------------------------------------------
# 5. ALL METHODS valid [0,1] / no NaN; flat-gray -> no NaN / no wild change.
# ---------------------------------------------------------------------------
print("\n[5] ALL METHODS valid + flat-gray div-zero guard")
flat = np.full((128, 128, 3), 0.5, dtype=np.float32)
for method in ("Gray World", "White Patch", "Shades of Gray", "Gray Edge"):
    out = run(cast_img, method=method, strength=1.0)
    valid = (not np.isnan(out).any()) and out.min() >= 0.0 and out.max() <= 1.0
    check(f"{method}: output in [0,1], no NaN (cast image)", valid,
          f"min={out.min():.4f} max={out.max():.4f} nan={bool(np.isnan(out).any())}")

    fout = run(flat, method=method, strength=1.0)
    no_nan = not np.isnan(fout).any()
    near_unchanged = float(np.abs(fout - flat).mean()) < 0.02
    check(f"{method}: flat-gray no NaN + ~unchanged (gains~1)",
          no_nan and near_unchanged,
          f"mean_abs_diff={float(np.abs(fout-flat).mean()):.5f} nan={bool(np.isnan(fout).any())}")

# ---------------------------------------------------------------------------
print("\n" + ("ALL CHECKS PASSED" if PASS else "SOME CHECKS FAILED"))
sys.exit(0 if PASS else 1)
