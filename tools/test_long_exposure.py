"""
Teeth for the Long Exposure node.

Physics and mathematics, not pixels -- the GPU path is float32 and reorders
reductions, so bit-equality is the wrong bar everywhere except where the operator
is provably an identity.

  T1  zero exposure is a bit-exact identity
  T2  the integral conserves energy: a uniform field integrates to itself, and the
      operator can never brighten or darken
  T3  gathering leaves NO holes and cannot tear -- every output pixel is a convex
      combination of real input pixels, so the output stays inside the input's range
  T4  THE HONEST ONE: the synthesised path reproduces the MEASURED human tremor
      spectrum, ~90% of displacement in 1-3.5Hz and ~24% in 7.5-12.5Hz. This is what
      separates a derived path from a tuned one, and it is what a first guess of
      0.45 tremor weight failed (59.5% in the tremor band)
  T5  roll varies with radius: exactly zero at the centre, growing outward
  T6  determinism, and the seed actually changes the shake
  T7  the mask scales motion per pixel, so a masked-out region is untouched
  T8  the GPU and numpy paths agree in the mean
  NC  five negative controls that MUST fail

Run:
  python_embeded/python.exe ComfyUI-Darkroom/tools/test_long_exposure.py
"""

import math
import os
import sys
import time
import types

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

import numpy as np
import torch

from importlib import import_module

# The pack's __init__ decorates against a live PromptServer; same shim as the
# other Darkroom suites.
if "server" not in sys.modules:
    class _Routes:
        def __getattr__(self, _n):
            return lambda *a, **k: (lambda f: f)
    _m = types.ModuleType("server")
    _m.PromptServer = type("PromptServer", (), {
        "instance": types.SimpleNamespace(routes=_Routes(),
                                          send_sync=lambda *a, **k: None)})
    sys.modules["server"] = _m

_pkg = os.path.basename(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LE = import_module(f"{_pkg}.utils.long_exposure")
NodeMod = import_module(f"{_pkg}.nodes.long_exposure")

PASSED, FAILED = [], []


def check(name, ok, detail=""):
    (PASSED if ok else FAILED).append(name)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}  {detail}")


def must_fail(name, ok, detail=""):
    (PASSED if not ok else FAILED).append(name)
    print(f"  [{'PASS' if not ok else 'FAIL'}] {name}  {detail}"
          f"{'' if not ok else '   <-- CONTROL DID NOT FIRE'}")


rng = np.random.default_rng(7)
IMG = rng.random((96, 128, 3))

print("=" * 78)
print("LONG EXPOSURE — teeth")
print("=" * 78)
t_start = time.time()

# --- T1 ---------------------------------------------------------------------
print("\nT1  zero exposure is a bit-exact identity")
tx, ty, rl = LE.shake_path(32, seed=1)
out = LE.integrate(IMG, tx, ty, rl, amp_px=0.0, roll_deg=0.0)
check("T1 amp 0, roll 0 returns the source", float(np.abs(out - IMG).max()) < 1e-12,
      f"max |diff| {float(np.abs(out - IMG).max()):.2e}")

# --- T2 ---------------------------------------------------------------------
print("\nT2  the integral conserves energy")
flat = np.full((64, 64, 3), 0.42)
o = LE.integrate(flat, tx, ty, rl, amp_px=25.0, roll_deg=3.0)
check("T2 a uniform field integrates to itself", float(np.abs(o - 0.42).max()) < 1e-9,
      f"max |diff| {float(np.abs(o - 0.42).max()):.2e}")
o2 = LE.integrate(IMG, tx, ty, rl, amp_px=30.0, roll_deg=2.0)
check("T2 mean brightness is preserved", abs(float(o2.mean() - IMG.mean())) < 0.02,
      f"{IMG.mean():.4f} -> {o2.mean():.4f}")

# --- T3 ---------------------------------------------------------------------
print("\nT3  gathering cannot tear or leave holes")
check("T3 output stays inside the input's range",
      float(o2.min()) >= float(IMG.min()) - 1e-9 and float(o2.max()) <= float(IMG.max()) + 1e-9,
      f"in [{IMG.min():.3f},{IMG.max():.3f}] -> out [{o2.min():.3f},{o2.max():.3f}]")
check("T3 no NaN or Inf anywhere", bool(np.isfinite(o2).all()))

# --- T4 ---------------------------------------------------------------------
print("\nT4  the path reproduces the MEASURED human tremor spectrum")
lo, hi = LE.band_shares(96, steadiness=1.0)
check("T4 drift band 1-3.5Hz carries ~90% of displacement", 0.82 <= lo <= 0.95,
      f"{100*lo:.1f}% (human ~90%)")
check("T4 tremor band 7.5-12.5Hz carries ~24%", 0.18 <= hi <= 0.32,
      f"{100*hi:.1f}% (human ~24%)")
lo0, hi0 = LE.band_shares(96, steadiness=0.0)
check("T4 steadiness 0 suppresses the tremor band", hi0 < hi,
      f"{100*hi:.1f}% -> {100*hi0:.1f}% with a perfectly steady hand")

# --- T4b/T4c the two defects Jeremie found in the render --------------------
# Both were reported from looking at the output, not caught by this suite, so both
# get a check. Neither is subtle once measured -- which is the point.
print("\nT4b  the movement must have a DIRECTION, not be square by construction")
ratios = []
for sd in (0, 1, 4, 17, 99, 123):
    a, b, _ = LE.shake_path(96, seed=sd, variety=1.0)
    ratios.append((a.max() - a.min()) / max(b.max() - b.min(), 1e-9))
spread = max(ratios) / max(min(ratios), 1e-9)
check("T4b axis ratio varies across seeds", spread > 3.0,
      f"x/y ratio {min(ratios):.2f} to {max(ratios):.2f} ({spread:.1f}x spread); "
      f"independently-normalised axes gave 0.97-1.03 for EVERY seed")

print("\nT4c  the speed must CHANGE during the exposure (dwell, then swoop)")


def _speed_spread(variety, seed=4, n=192):
    a, b, _ = LE.shake_path(n, seed=seed, variety=variety)
    v = np.hypot(np.diff(a), np.diff(b))
    q = len(v) // 4
    qs = [float(v[i * q:(i + 1) * q].mean()) for i in range(4)]
    return max(qs) / max(min(qs), 1e-12)


s_uniform, s_varied = _speed_spread(0.0), _speed_spread(1.0)
check("T4c variety produces a non-stationary path", s_varied > 2.5,
      f"fastest/slowest quarter {s_varied:.1f}x at variety 1.0")
check("T4c variety 0 stays uniform, as documented", s_uniform < s_varied,
      f"{s_uniform:.1f}x at variety 0 vs {s_varied:.1f}x at 1.0")

# --- T5 ---------------------------------------------------------------------
print("\nT5  roll varies across the frame, and is zero at the centre")
H, W = 129, 129
ys, xs, dsy, dsx = LE._pose_offsets(H, W, tx, ty, rl, 3, 0.0, 6.0)
d = np.hypot(dsy, dsx)
cy, cx = (H - 1) // 2, (W - 1) // 2
check("T5 displacement is exactly zero on the rotation axis", d[cy, cx] < 1e-9,
      f"centre {d[cy, cx]:.2e}px")
check("T5 corners travel further than mid-radius", d[0, 0] > d[cy // 2, cx // 2] > 0,
      f"corner {d[0,0]:.2f}px vs mid {d[cy//2,cx//2]:.2f}px")

# --- T6 ---------------------------------------------------------------------
print("\nT6  determinism, and the seed matters")
a1 = LE.shake_path(48, seed=5)[0]
a2 = LE.shake_path(48, seed=5)[0]
a3 = LE.shake_path(48, seed=6)[0]
check("T6 same seed gives the same path", float(np.abs(a1 - a2).max()) < 1e-15)
check("T6 a different seed gives a different path", float(np.abs(a1 - a3).max()) > 0.05,
      f"max |diff| {float(np.abs(a1 - a3).max()):.3f}")

# --- T7 ---------------------------------------------------------------------
print("\nT7  the mask scales motion per pixel")
mask = np.zeros((96, 128))
mask[:, 64:] = 1.0
om = LE.integrate(IMG, tx, ty, rl, amp_px=30.0, roll_deg=0.0, mask=mask)
still = float(np.abs(om[:, :60] - IMG[:, :60]).max())
moved = float(np.abs(om[:, 70:] - IMG[:, 70:]).mean())
check("T7 masked-out region is untouched", still < 1e-12, f"max |diff| {still:.2e}")
check("T7 masked-in region does move", moved > 0.02, f"mean |diff| {moved:.4f}")

# --- T8 ---------------------------------------------------------------------
print("\nT8  GPU and numpy agree")
if LE._torch_ok():
    g = LE.integrate_gpu(IMG, tx, ty, rl, amp_px=30.0, roll_deg=2.0)
    check("T8 gpu matches numpy in the mean", float(np.abs(g - o2).mean()) < 0.02,
          f"mean |diff| {float(np.abs(g - o2).mean()):.5f}")
else:
    print("  [SKIP] no CUDA available")

# --- node level -------------------------------------------------------------
print("\nT9  node level")
node = NodeMod.LongExposure()
batch = torch.rand(2, 64, 48, 3)
img_out, = node.execute(batch, exposure_px=25.0, hand_steadiness=1.0, variety=1.0,
                        roll_deg=1.5, seed=0, poses=24)
check("T9 shape and dtype survive",
      img_out.shape == batch.shape and img_out.dtype == torch.float32,
      f"{tuple(img_out.shape)} {img_out.dtype}")
check("T9 finite and in range", bool(torch.isfinite(img_out).all())
      and float(img_out.min()) >= 0.0 and float(img_out.max()) <= 1.0)
zero, = node.execute(batch, exposure_px=0.0, hand_steadiness=1.0, variety=1.0,
                     roll_deg=0.0, seed=0, poses=24)
check("T9 zero exposure is an identity at node level",
      float((zero - batch).abs().max()) < 1e-7,
      f"max |diff| {float((zero - batch).abs().max()):.2e}")
mk = torch.zeros(1, 64, 48)
masked, = node.execute(batch, exposure_px=30.0, hand_steadiness=1.0, variety=1.0,
                       roll_deg=2.0, seed=0, poses=24, subject_mask=mk)
# float32 bar, stated once and deliberately. The node may run the GPU path, where
# grid_sample normalises coordinates in float32 and costs ~1e-6 even at exactly
# identity positions. The property itself is EXACT and is tested as such against the
# numpy reference in T7 (8.88e-16); this is the same property at the precision the
# device can hold it, not a weaker claim about the maths.
check("T9 an all-zero mask disables the effect entirely (float32 bar)",
      float((masked - batch).abs().max()) < 2e-5,
      f"max |diff| {float((masked - batch).abs().max()):.2e}; "
      f"numpy path proves it exact in T7")

# --- negative controls ------------------------------------------------------
print("\nNC  negative controls — each MUST fail")
f = np.fft.rfftfreq(96, d=1.0 / 96).copy()
f[0] = 1e-9
white = np.ones_like(f)
white[0] = 0
tot = math.sqrt(float((white ** 2).sum()))
w_lo = math.sqrt(float((white[(f >= 1.0) & (f <= 3.5)] ** 2).sum())) / tot
must_fail("NC1 a WHITE-noise path matches the human drift share", 0.82 <= w_lo <= 0.95,
          f"white noise puts only {100*w_lo:.1f}% in 1-3.5Hz vs the human ~90%")
_, _, r0 = LE.shake_path(32, seed=1)
_, _, _, dsx0 = LE._pose_offsets(65, 65, tx, ty, r0, 3, 20.0, 0.0)
must_fail("NC2 zero roll still varies across the frame",
          float(dsx0.max() - dsx0.min()) > 1e-9,
          "a pure shift must displace every pixel identically")
must_fail("NC3 a uniform field can be brightened by the integral",
          abs(float(LE.integrate(flat, tx, ty, rl, amp_px=40.0).mean()) - 0.42) > 1e-9)
must_fail("NC4 an all-ONE mask leaves the image untouched",
          float(np.abs(LE.integrate(IMG, tx, ty, rl, amp_px=30.0,
                                    mask=np.ones((96, 128))) - IMG).max()) < 1e-9)
must_fail("NC5 steadiness 4 still matches the human spectrum",
          0.18 <= LE.band_shares(96, steadiness=4.0)[1] <= 0.32,
          f"a jittery hand puts {100*LE.band_shares(96, steadiness=4.0)[1]:.0f}% "
          "in the tremor band, well outside human")

print("\n" + "=" * 78)
print(f"{len(PASSED)} passed, {len(FAILED)} failed  ({time.time()-t_start:.1f}s)")
if FAILED:
    print("FAILURES: " + ", ".join(FAILED))
print("=" * 78)
sys.exit(1 if FAILED else 0)
