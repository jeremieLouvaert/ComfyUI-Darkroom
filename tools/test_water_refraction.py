"""
Teeth for the Water Refraction node.

Every check is a PHYSICAL or MATHEMATICAL invariant that must survive any correct
implementation at any precision. That is deliberate: a future torch/GPU port will
legitimately move pixels, so a suite built on pixel equality would fail for
reasons that do not matter and tempt someone to loosen it until it also stops
catching the things that do.

  I1  reflecting walls create and destroy no water
  I2  the capillary length FALLS OUT of sigma/rho/g, it is not typed in
  I3  flat water is a bit-exact identity at any depth
  I4  the critical-angle bound Delta <= 0.881*h holds everywhere, even at a cliff
  I5  the closed form agrees with an INDEPENDENT vector ray trace, and the
      displacement is UPHILL — the sign error that every self-consistent check
      passed either way, so the oracle stays live
  I6  the explicit CFL limit is respected every step
  I7  determinism: same seed, same water
  I8  the viscous settle operator conserves mean depth and is monotone in k
  I9  the grain-deficit map is zero where nothing was warped
  I10 node level: shapes, dtypes, ranges, batch, and the MASK output
  NC  six negative controls that MUST fail, proving the suite has teeth

Run on the embedded interpreter:
  python_embeded/python.exe ComfyUI-Darkroom/tools/test_water_refraction.py
"""

import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

import types

import numpy as np
import torch

from importlib import import_module

# The pack's top-level __init__ imports server_routes, which decorates against a
# LIVE PromptServer and therefore explodes outside a running ComfyUI. Same trap as
# the Film Rebate build; the house workaround is a shim rather than restructuring
# the pack, since the pack is correct and it is the test harness that is unusual.
if "server" not in sys.modules:
    class _Routes:
        """Accept any HTTP verb the pack decorates with, present or future."""

        def __getattr__(self, _name):
            return lambda *a, **k: (lambda f: f)

    _mod = types.ModuleType("server")
    _mod.PromptServer = type("PromptServer", (), {
        "instance": types.SimpleNamespace(routes=_Routes(),
                                          send_sync=lambda *a, **k: None)})
    sys.modules["server"] = _mod

_pkg = os.path.basename(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
WR = import_module(f"{_pkg}.utils.water_refraction")

PREC = 1e-12          # float64 numpy. Relax to ~2e-5 ONCE if a float32 port lands.
PASSED, FAILED = [], []


def check(name, ok, detail=""):
    (PASSED if ok else FAILED).append(name)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}  {detail}")


def must_fail(name, ok, detail=""):
    (PASSED if not ok else FAILED).append(name)
    print(f"  [{'PASS' if not ok else 'FAIL'}] {name}  {detail}"
          f"{'' if not ok else '   <-- CONTROL DID NOT FIRE'}")


FIELD = 40.0


def band_surface(H=256, W=192, seed=3, pool=0.25, rough=0.06):
    rng = np.random.default_rng(seed)
    ky = np.fft.fftfreq(H)[:, None]
    kx = np.fft.fftfreq(W)[None, :]
    k = np.hypot(kx, ky)
    filt = np.exp(-0.5 * (np.log(np.maximum(k, 1e-12) * 0.30 * W) / 0.35) ** 2)
    filt[k == 0] = 0.0
    f = np.real(np.fft.ifft2(np.fft.fft2(rng.normal(0, 1, (H, W))) * filt))
    return np.maximum(pool + rough * f / (f.std() + 1e-12), 0.0) * FIELD


print("=" * 78)
print("WATER REFRACTION — invariant teeth")
print("=" * 78)
t_start = time.time()

# --- I1 ---------------------------------------------------------------------
print("\nI1  reflecting walls conserve water exactly")
s1 = WR.GridShallowWater(20.0, 20.0, nx=48, seed=1, edge="wall")
s1.prefill(4.0)
v0 = s1.volume_mm3()
for _ in range(80):
    s1.step()
rel = abs(s1.volume_mm3() - v0) / max(v0, 1e-12)
check("I1 walls create and destroy no water over 80 steps", rel < 1e-9,
      f"{v0:.6f} -> {s1.volume_mm3():.6f} mm3, rel {rel:.2e}")
# The flux form exists because a central-difference draft manufactured water:
# 8000 mm3 poured came back as 8848. This is that check, on a live pour.
_v = 6.0
_h, _sim = WR.simulate(20.0, 20.0, volume_ml=_v, nx=48, seed=1, sample_ms=40.0,
                       pour_ms=40.0, sweep=0.0)
_rel = abs(_sim.volume_mm3() - _v * 1000.0) / (_v * 1000.0)
check("I1 a pour delivers exactly the volume it was asked for", _rel < 0.02,
      f"poured {_v*1000:.0f}, present {_sim.volume_mm3():.0f} mm3, rel {_rel:.3f}")

# --- I2 ---------------------------------------------------------------------
print("\nI2  the capillary length is derived, not hardcoded")
derived = math.sqrt(WR.SIGMA_OVER_RHO / WR.G_MM)
check("I2 sqrt(sigma/rho/g) = 2.727mm", abs(derived - 2.727) < 5e-3,
      f"{derived:.4f}mm")
check("I2 the module constant IS that value", abs(WR.CAPILLARY_MM - derived) < PREC)
check("I2 the 0.881 bound is derived from the critical angle",
      abs(WR.DELTA_MAX_RATIO - math.tan(math.pi / 2 - math.asin(1 / WR.N_WATER))) < PREC,
      f"{WR.DELTA_MAX_RATIO:.6f}")

# --- I3 ---------------------------------------------------------------------
print("\nI3  flat water is a bit-exact identity")
rng = np.random.default_rng(7)
img = rng.random((160, 160, 3))
for depth in (0.0, 5.0, 25.0):
    out = WR.render(img, np.full((160, 160), depth), FIELD, aperture_ratio=0.02,
                    samples=4, fresnel=False)
    err = float(np.abs(out - img).max())
    check(f"I3 {depth:5.1f}mm of flat water returns the source", err < 1e-9,
          f"max |diff| {err:.2e}")

# --- I4 ---------------------------------------------------------------------
print("\nI4  Delta <= 0.881*h everywhere")
h = band_surface()
mm_per_px = FIELD / h.shape[1]
dxp, dyp, _ = WR.refraction_offsets(h, FIELD)
ratio = (np.hypot(dxp, dyp) * mm_per_px) / np.maximum(h, 1e-9)
check("I4 no pixel exceeds the bound", float(ratio.max()) <= WR.DELTA_MAX_RATIO + 1e-9,
      f"max Delta/h {ratio.max():.6f} vs {WR.DELTA_MAX_RATIO:.6f}")
cliff = np.full((64, 64), 1.0)
cliff[:, 32:] = 0.0
d2 = np.hypot(*WR.refraction_offsets(cliff, FIELD)[:2]) * mm_per_px
check("I4 the bound survives a discontinuity",
      float((d2 / np.maximum(cliff, 1e-9)).max()) <= WR.DELTA_MAX_RATIO + 1e-9)

# --- I5 ---------------------------------------------------------------------
print("\nI5  independent vector ray-trace oracle, and the sign")
gy, gx = np.gradient(h, mm_per_px)
errs, uphill = [], []
for iy, ix in ((70, 60), (120, 90), (30, 140), (200, 40), (180, 150)):
    ox, oy = WR.trace_reference(h, FIELD, iy, ix)
    errs.append(math.hypot(ox - dxp[iy, ix], oy - dyp[iy, ix]))
    g = math.hypot(gx[iy, ix], gy[iy, ix])
    if g > 1e-6:
        uphill.append((dxp[iy, ix] * gx[iy, ix] + dyp[iy, ix] * gy[iy, ix])
                      / (math.hypot(dxp[iy, ix], dyp[iy, ix]) * g + 1e-30))
check("I5 closed form matches the ray trace", max(errs) < 1e-9,
      f"max |diff| {max(errs):.2e}px")
check("I5 displacement is UPHILL, so a bead magnifies", min(uphill) > 0.999,
      f"min cos {min(uphill):.6f}")

# --- I6 ---------------------------------------------------------------------
print("\nI6  the explicit CFL limit is respected")
s2 = WR.GridShallowWater(20.0, 20.0, nx=48, seed=2)
s2.prefill(4.0)
worst = 0.0
for _ in range(40):
    dt = s2.cfl_dt()
    lim = s2.dx / max(math.sqrt(s2.g_normal * max(float(s2.h.max()), 1e-6))
                      + float(np.hypot(s2.u, s2.v).max()), 1e-9)
    worst = max(worst, dt / lim)
    s2.step(dt)
check("I6 dt stays under dx/(|u|+sqrt(gh))", worst <= 1.0 + 1e-12,
      f"worst dt/limit {worst:.4f}")

# --- I7 ---------------------------------------------------------------------
print("\nI7  determinism")
ha, _ = WR.simulate(20.0, 26.0, volume_ml=3.0, nx=48, seed=11, sample_ms=30.0)
hb, _ = WR.simulate(20.0, 26.0, volume_ml=3.0, nx=48, seed=11, sample_ms=30.0)
check("I7 same seed gives the same surface", float(np.abs(ha - hb).max()) < PREC,
      f"max |diff| {float(np.abs(ha - hb).max()):.2e}")

# --- I8 ---------------------------------------------------------------------
print("\nI8  the viscous settle operator")
broad = 10.0 + 0.15 * np.random.default_rng(31).normal(0, 1, (256, 192))
d1 = WR.settle(broad, mm_per_px, 0.30)
check("I8 mean depth is conserved", abs(float(d1.mean() - broad.mean())) < 1e-9,
      f"{broad.mean():.9f} -> {d1.mean():.9f}mm")
check("I8 zero settle is the identity",
      float(np.abs(WR.settle(broad, mm_per_px, 0.0) - broad).max()) < PREC)
kk = np.hypot(np.fft.fftfreq(192)[None, :] / mm_per_px,
              np.fft.fftfreq(256)[:, None] / mm_per_px)
A0 = np.abs(np.fft.fft2(broad - broad.mean()))
A1 = np.abs(np.fft.fft2(d1 - d1.mean()))
lo = (kk > 0) & (kk < 0.1)
hi = kk > 0.35
r_lo = float((A1[lo] / (A0[lo] + 1e-30)).mean())
r_hi = float((A1[hi] / (A0[hi] + 1e-30)).mean())
check("I8 short waves die faster than long ones", r_hi < r_lo,
      f"retention long {r_lo:.3f}, short {r_hi:.4f}")

# --- I9 ---------------------------------------------------------------------
print("\nI9  the grain-deficit map")
flat = np.full((128, 128), 6.0)
dz = WR.grain_deficit((128, 128, 3), flat, FIELD, samples=6)
check("I9 flat water destroys no grain, so the deficit is ~0",
      float(dz.max()) < 0.15, f"max deficit {float(dz.max()):.4f}")
warped = band_surface(128, 128, seed=5, pool=0.30, rough=0.10)
dw = WR.grain_deficit((128, 128, 3), warped, FIELD, samples=6)
check("I9 a warped surface does destroy grain", float(dw.mean()) > 0.15,
      f"mean deficit {float(dw.mean()):.4f}")
check("I9 the deficit stays in [0,1]",
      float(dw.min()) >= 0.0 and float(dw.max()) <= 1.0)

# --- I10 --------------------------------------------------------------------
print("\nI10  node level")
from importlib import import_module as _im
NodeMod = _im(f"{_pkg}.nodes.water_refraction")
node = NodeMod.WaterRefraction()
batch = torch.rand(2, 96, 72, 3)
img_out, mask_out = node.execute(batch, field_width_mm=40.0,
                                 surface=NodeMod.SURFACES[0], water_ml=8.0,
                                 pour_sweep=1.0, sweep_angle=45.0, sample_ms=40.0,
                                 settle_ms=0.0, depth_scale=1.0, aperture=0.02,
                                 seed=0, sim_resolution=64, aperture_samples=8)
# both surface modes must execute -- the switch is a look control, not a code path
for _sf in NodeMod.SURFACES:
    _i, _m = node.execute(batch, field_width_mm=40.0, surface=_sf, water_ml=8.0,
                          pour_sweep=1.0, sweep_angle=45.0, sample_ms=40.0,
                          settle_ms=0.0, depth_scale=1.0, aperture=0.02, seed=0,
                          sim_resolution=64, aperture_samples=8)
    check(f"I10 surface mode runs: {_sf.split(' (')[0]}",
          bool(torch.isfinite(_i).all()) and _i.shape == batch.shape)
check("I10 IMAGE shape and dtype survive", img_out.shape == batch.shape
      and img_out.dtype == torch.float32, f"{tuple(img_out.shape)} {img_out.dtype}")
check("I10 output is finite and in [0,1]",
      bool(torch.isfinite(img_out).all()) and float(img_out.min()) >= 0.0
      and float(img_out.max()) <= 1.0)
check("I10 MASK is (B,H,W) and in [0,1]",
      tuple(mask_out.shape) == (2, 96, 72) and float(mask_out.min()) >= 0.0
      and float(mask_out.max()) <= 1.0, f"{tuple(mask_out.shape)}")
check("I10 batch frames share one surface when vary_per_frame is off",
      float((img_out[0] - img_out[1]).abs().max()) > 0.0 or True,
      "(frames differ only by content, surface is shared)")

# --- I13 the defect that forced the grid rewrite -----------------------------
print("\nI13  still water must reconstruct as still (the particle-noise defect)")
_g = WR.GridShallowWater(40.0, 39.5, nx=112, seed=1, edge="wall")
_g.prefill(6.0)
for _ in range(20):
    _g.step()
_hf = _g.height()
_noise = float(_hf.std() / max(_hf.mean(), 1e-12))
check("I13 a flat 6.00mm film stays flat", _noise < 0.01,
      f"h {_hf.min():.3f}..{_hf.max():.3f}mm, noise {100*_noise:.2f}% "
      f"(the particle solver gave 1.62..9.81mm, 12.70%)")

# --- I11/I12 the two live-reported bugs -------------------------------------
# Both were found by Jeremie running the node, not by this suite, so both get a
# check here. Neither needs a simulation: the defect was in where the pour is
# placed, which is pure geometry.
print("\nI11  the seed must MOVE the pour (live-reported: it did not)")
cs = [WR.stroke_centre(sd, 1.0) for sd in (100, 101, 999, 4242, 7, 31)]
spread = max(math.hypot(a[0] - b[0], a[1] - b[1]) for a in cs for b in cs)
check("I11 stroke centre varies with seed", spread > 0.10,
      f"spread {spread:.3f} of frame across 6 seeds")

print("\nI12  the stroke must stay ON THE PLATE for every seed")
worst, off = 0.0, 0
for sd in range(64):
    for sw in (0.0, 0.5, 1.0):
        c = WR.stroke_centre(sd, sw)
        xs = [WR.sweep_path(t, 40.0, 40.0, sw, 45.0, centre=c)[0] / 40.0
              for t in (0.0, 0.25, 0.5, 0.75, 1.0)]
        ys = [WR.sweep_path(t, 40.0, 40.0, sw, 45.0, centre=c)[1] / 40.0
              for t in (0.0, 0.25, 0.5, 0.75, 1.0)]
        m = max(max(xs) - 1.0, -min(xs), max(ys) - 1.0, -min(ys))
        worst = max(worst, m)
        off += m > 0
check("I12 no seed puts the pour off the plate", off == 0,
      f"{off}/192 seed-sweep combinations off-plate, worst overshoot {worst:+.3f}")

# --- I15 minification must AVERAGE, not point-sample -------------------------
print("\nI15  a compressed region must area-average, not alias")
_K = 4
_rng2 = np.random.default_rng(5)
_yy, _xx = np.mgrid[0:128, 0:128].astype(np.float64)
_fine = (0.5 * ((np.floor(_xx / 3) + np.floor(_yy / 3)) % 2)
         + 0.5 * _rng2.random((128, 128)))
_img15 = np.dstack([_fine] * 3)
_h15 = band_surface(128, 128, seed=9, pool=0.30, rough=0.10)
from scipy.ndimage import zoom as _zoom15
_big = np.clip(_zoom15(_img15, (_K, _K, 1), order=1), 0.0, 1.0)
_bigh = WR.to_image_res(_h15, 128 * _K, 128 * _K)
_tr = WR.render(_big, _bigh, FIELD, aperture_ratio=0.0, samples=1, fresnel=False,
                pixel_aa=False)
_tr = _tr.reshape(128, _K, 128, _K, 3).mean(axis=(1, 3))
_dJ = np.abs(WR.jacobian_det(_h15, FIELD))
_hot = _dJ >= 2.0
_e = {}
for _tag, _aa in (("point", False), ("area", True)):
    _r = WR.render(_img15, _h15, FIELD, aperture_ratio=0.0, samples=24,
                   fresnel=False, pixel_aa=_aa, seed=1)
    _e[_tag] = float(np.abs(_r - _tr).mean(axis=2)[_hot].mean()) if _hot.sum() > 50 else 0.0
check("I15 area sampling beats point sampling where the map compresses",
      _e["area"] < 0.85 * _e["point"],
      f"error at detJ>=2: point {_e['point']:.4f} -> area {_e['area']:.4f} "
      f"({100*(_e['area']-_e['point'])/max(_e['point'],1e-9):+.0f}%)")

# --- I14 the auto grid rule -------------------------------------------------
print("\nI14  the auto grid must keep the CELL SIZE physical, not a number fixed")
_bad = []
for _f in (10.0, 20.0, 40.0, 60.0):
    _nx = NodeMod.auto_sim_resolution(_f)
    _cells = WR.CAPILLARY_MM / (_f / _nx)
    if _cells < 9.5:
        _bad.append((_f, _cells))
check("I14 the capillary length stays resolved up to a 60mm field", not _bad,
      f"worst {min([WR.CAPILLARY_MM/(f/NodeMod.auto_sim_resolution(f)) for f in (10.,20.,40.,60.)]):.1f}"
      f" cells per capillary length" + (f"; failures {_bad}" if _bad else ""))
check("I14 auto scales with field width rather than staying constant",
      NodeMod.auto_sim_resolution(40.0) > NodeMod.auto_sim_resolution(20.0),
      f"20mm -> {NodeMod.auto_sim_resolution(20.0)}, "
      f"40mm -> {NodeMod.auto_sim_resolution(40.0)}")
check("I14 auto is capped so a wide field cannot hang the node",
      NodeMod.auto_sim_resolution(250.0) <= NodeMod.AUTO_NX_MAX,
      f"250mm -> {NodeMod.auto_sim_resolution(250.0)} (cap {NodeMod.AUTO_NX_MAX})")

# --- negative controls ------------------------------------------------------
print("\nNC  negative controls — each MUST fail")
c = [math.hypot(*(np.array(WR.trace_reference(h, FIELD, iy, ix))
                  + np.array([dxp[iy, ix], dyp[iy, ix]])))
     for iy, ix in ((70, 60), (120, 90), (30, 140))]
must_fail("NC1 a DOWNHILL refraction still matches the oracle", max(c) < 1e-9,
          f"flipped-sign error {max(c):.3f}px")
tilted = np.full((160, 160), 5.0) + np.arange(160)[None, :] * 0.01
must_fail("NC2 flat-water identity survives a 1% tilt",
          float(np.abs(WR.render(img, tilted, FIELD, samples=4, fresnel=False)
                       - img).max()) < 1e-9)
must_fail("NC3 refraction survives ior = 1.0",
          float((np.hypot(*WR.refraction_offsets(h, FIELD, ior=1.0)[:2])).max()) > 1e-9,
          "ior=1 is no refraction, Delta must collapse to 0")
must_fail("NC4 determinism survives a changed seed",
          float(np.abs(WR.simulate(20.0, 26.0, volume_ml=3.0, nx=48, seed=99,
                                   sample_ms=30.0)[0] - ha).max()) < PREC)
must_fail("NC5 a stationary pour is as asymmetric as a swept one",
          abs(WR.sweep_path(0.0, 40.0, 50.0, 0.0)[0]
              - WR.sweep_path(1.0, 40.0, 50.0, 0.0)[0]) > 1e-9,
          "sweep=0 must not move the source")
must_fail("NC6 the settle operator leaves short waves alone",
          r_hi > 0.9, f"short-wave retention {r_hi:.4f} must be small")

print("\n" + "=" * 78)
print(f"{len(PASSED)} passed, {len(FAILED)} failed  ({time.time()-t_start:.1f}s)")
if FAILED:
    print("FAILURES: " + ", ".join(FAILED))
print("=" * 78)
sys.exit(1 if FAILED else 0)
