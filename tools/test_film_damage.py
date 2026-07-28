"""
Teeth for the Film Damage node (T1-T8, docs/film-damage-derivation.md sec 7).

Run with the embedded python:
  python_embeded/python.exe tools/test_film_damage.py

Every physics check carries a NEGATIVE CONTROL that must FAIL for the check to
mean anything. Promoted from _film_damage_spike/checks.py against the shipped
utils/film_damage.py, so the tests exercise production code, not a copy.
"""

import importlib.util
import os
import sys
import types

import numpy as np
import torch
from scipy.ndimage import gaussian_filter

PACK_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Synthetic package shim (cf. test_halation.py / test_film_rebate.py): lets the
# node's `from ..utils...` relative imports resolve WITHOUT executing the
# top-level package __init__, which needs the live ComfyUI runtime.
PKG = "dr_pkg_film_damage"
for name, path in ((PKG, PACK_ROOT),
                   (PKG + ".nodes", os.path.join(PACK_ROOT, "nodes")),
                   (PKG + ".utils", os.path.join(PACK_ROOT, "utils"))):
    m = types.ModuleType(name)
    m.__path__ = [path]
    sys.modules[name] = m


def _load(modname, relpath):
    spec = importlib.util.spec_from_file_location(
        PKG + "." + modname, os.path.join(PACK_ROOT, *relpath))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[PKG + "." + modname] = mod
    spec.loader.exec_module(mod)
    return mod


_fd_utils = _load("utils.film_damage", ("utils", "film_damage.py"))
_color = _load("utils.color", ("utils", "color.py"))
_node_mod = _load("nodes.film_damage", ("nodes", "film_damage.py"))

build_tau = _fd_utils.build_tau
composite = _fd_utils.composite
defect_mask = _fd_utils.defect_mask
dye_tau_weights = _fd_utils.dye_tau_weights
srgb_to_linear = _color.srgb_to_linear
linear_to_srgb = _color.linear_to_srgb
DarkroomFilmDamage = _node_mod.DarkroomFilmDamage
ORIGINS = _node_mod.ORIGINS
FILM_TYPES = _node_mod.FILM_TYPES

PASS, FAIL = [], []
GAMMA = 2.0


def check(name, ok, detail=""):
    (PASS if ok else FAIL).append(name)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))
    return ok


print("=" * 74)
print("FILM DAMAGE — teeth")
print("=" * 74)

H = W = 512
tau = build_tau(H, W, 7, density=0.6, scratch_count=0, origin="negative")
mask = tau[..., 1] < 0.97
flat = np.full((H, W, 3), 0.35)
lin_flat = srgb_to_linear(flat)


# ---------------------------------------------------------------------------
print("\nT1 — chain-parity sign")
# ---------------------------------------------------------------------------
neg = composite(lin_flat, tau, "negative", GAMMA)
pos = composite(lin_flat, tau, "positive", GAMMA)
check("negative-plane defect prints LIGHTER (white dust)",
      neg[mask].mean() > lin_flat[mask].mean(),
      f"dL={neg[mask].mean()-lin_flat[mask].mean():+.4f}")
check("positive-plane defect reads DARKER",
      pos[mask].mean() < lin_flat[mask].mean(),
      f"dL={pos[mask].mean()-lin_flat[mask].mean():+.4f}")
naive = np.clip(lin_flat * tau, 0, 1)          # parity-blind model
check("NEG CONTROL parity-blind model cannot produce white dust",
      not (naive[mask].mean() > lin_flat[mask].mean()))


# ---------------------------------------------------------------------------
print("\nT2 — exponent magnitude and clipping endpoints")
# ---------------------------------------------------------------------------
opaque = np.full((4, 4, 3), 1e-4)
mid = np.full((4, 4, 3), 0.35)
check("opaque defect on negative -> paper-base WHITE",
      np.allclose(composite(mid, opaque, "negative", GAMMA), 1.0))
check("opaque defect on positive -> BLACK (tau-floor limited)",
      composite(mid, opaque, "positive", GAMMA).max() < 1e-4)
# substrate 0.05, not mid-grey: at 0.35 a gamma=2 lift CLIPS and the test would
# measure the clip instead of the exponent (the halation test-integrity trap)
dark = np.full((4, 4, 3), 0.05)
half = np.full((4, 4, 3), 0.5)
d1 = np.log10(composite(dark, half, "negative", 1.0) / dark)
d2 = np.log10(composite(dark, half, "negative", 2.0) / dark)
check("density change scales linearly with print_gamma",
      np.allclose(d2 / d1, 2.0, rtol=1e-9), f"ratio={float((d2/d1).mean()):.6f}")


# ---------------------------------------------------------------------------
print("\nT3 — tone-invariance (THE KILL-TEST)")
# ---------------------------------------------------------------------------
# A defect is a constant DENSITY offset, so at fixed tau the log-luminance
# ratio does not depend on the substrate tone. Exact, not statistical.
ramp = np.tile(np.linspace(0.02, 0.98, 400)[None, :, None], (8, 1, 3))
# linear ramp built directly in float64: utils/color.srgb_to_linear casts to
# float32 by pack convention, which would cap the measurable precision at
# float32 epsilon and hide whether the invariance is exact or merely close.
lin_ramp = np.linspace(0.0005, 0.95, 400)[None, :, None] * np.ones((8, 1, 3))
tau_fix = np.full_like(lin_ramp, 0.6)
for origin in ("negative", "positive"):
    o = composite(lin_ramp, tau_fix, origin, GAMMA)
    un = (o[..., 1] < 0.999) & (o[..., 1] > 1e-6)
    lr = np.log(o[..., 1] / lin_ramp[..., 1])[un]
    check(f"{origin}-plane log-luminance offset is tone-INVARIANT (float64, exact)",
          float(lr.std()) < 1e-12, f"mean={lr.mean():+.5f} std={lr.std():.2e}")
# and the same invariance survives the shipped float32 path, to float32 epsilon
lin32 = srgb_to_linear(ramp)
o32 = composite(lin32, np.full_like(lin32, 0.6, dtype=np.float64), "negative", GAMMA)
un32 = (o32[..., 1] < 0.999) & (o32[..., 1] > 1e-6)
lr32 = np.log(o32[..., 1] / lin32[..., 1])[un32]
check("invariance survives the shipped float32 colour path",
      float(lr32.std()) < 1e-6, f"std={lr32.std():.2e} (float32 eps ~1.2e-7)")
for label, paint, thresh in (("white", 1.0, 0.5), ("dark", 0.0, 0.1)):
    a = 0.4
    ob = srgb_to_linear(np.clip(ramp * (1 - a) + paint * a, 0, 1))
    # reference must be lin32 (the SAME sRGB ramp taken to linear), not the
    # synthetic float64 ramp above, or the control measures the wrong ratio
    lrb = np.log((ob[..., 1] + 1e-12) / (lin32[..., 1] + 1e-12))
    cv = abs(float(lrb.std() / lrb.mean()))
    check(f"NEG CONTROL sRGB {label} alpha-blend breaks tone-invariance",
          cv > thresh, f"CV={cv:.3f} vs physics ~1e-16")

# visibility profile asymmetry (statistical half; needs a dense rig)
tau_p = build_tau(512, 512, 3, density=22.0, scratch_count=0, origin="negative")
mp = tau_p[..., 1] < 0.97
ramp2 = np.tile(np.linspace(0.02, 0.98, 512)[None, :, None], (512, 1, 3))
lin2 = srgb_to_linear(ramp2)


def _centroid(origin):
    out = linear_to_srgb(composite(lin2, tau_p, origin, GAMMA))
    delta = np.abs(out - ramp2).mean(axis=2)
    v, x = [], []
    for b in range(12):
        lo, hi = 0.02 + b * 0.08, 0.02 + (b + 1) * 0.08
        sel = mp & (ramp2[..., 0] >= lo) & (ramp2[..., 0] < hi)
        if sel.sum() > 200:
            v.append(delta[sel].mean())
            x.append(b / 11.0)
    v, x = np.array(v), np.array(x)
    return float((v * x).sum() / v.sum()), v


c_neg, v_neg = _centroid("negative")
c_pos, _ = _centroid("positive")
check("negative-plane visibility is shadow-shifted vs positive-plane",
      c_neg < c_pos - 0.12, f"centroid {c_neg:.3f} vs {c_pos:.3f}")
check("negative-plane collapses in blown highlights (clips to paper white)",
      v_neg[-1] < v_neg.max() * 0.5, f"top bin {v_neg[-1]:.3f} vs peak {v_neg.max():.3f}")


# ---------------------------------------------------------------------------
print("\nT4 — dye-depth colour table and reversal complementarity")
# ---------------------------------------------------------------------------
grey = np.full((4, 4, 3), 0.45)
lin_g = np.full((4, 4, 3), 0.17, dtype=np.float64)   # float64: see T3 note
ok, cos_all = True, []
for depth in (0.15, 0.5, 0.9):
    t = np.ones((4, 4, 3)) * np.array(dye_tau_weights(depth, "c41"))[None, None, :]
    d_neg = np.log(lin_g * t ** (-GAMMA)) - np.log(lin_g)
    d_rev = np.log(lin_g * t) - np.log(lin_g)
    a, b = d_neg[0, 0], d_rev[0, 0]
    cos_all.append(float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b))))
    # residual form, never a ratio: unremoved channels give 0/0 = NaN
    ok &= float(np.abs(d_neg + GAMMA * d_rev).max()) < 1e-12
check("negative and reversal log-deltas exactly anti-parallel (ratio -gamma_p)",
      ok and all(abs(c + 1.0) < 1e-9 for c in cos_all),
      f"cos={['%.6f' % c for c in cos_all]}")


def mark(depth, film, origin):
    t = np.ones((1, 1, 3)) * np.array(dye_tau_weights(depth, film))[None, None, :]
    return linear_to_srgb(composite(lin_g[:1, :1], t, origin, GAMMA))[0, 0]


y = mark(0.15, "c41", "negative")
r = mark(0.55, "c41", "negative")
k = mark(0.98, "c41", "negative")
check("shallow C-41 emulsion scratch reads YELLOW", y[2] < y[0] and y[2] < y[1],
      f"RGB {np.round(y,3)}")
check("mid-depth C-41 scratch reads RED/ORANGE", r[0] > r[1] > r[2],
      f"RGB {np.round(r,3)}")
check("full-depth C-41 gouge reads near-BLACK", k.max() < 0.2, f"RGB {np.round(k,3)}")
bl = mark(0.15, "reversal", "positive")
cy = mark(0.55, "reversal", "positive")
wh = mark(0.98, "reversal", "positive")
check("shallow reversal scratch reads BLUE", bl[2] > bl[1] and bl[2] > bl[0],
      f"RGB {np.round(bl,3)}")
check("mid reversal scratch reads CYAN", cy[2] > cy[0] and cy[1] > cy[0],
      f"RGB {np.round(cy,3)}")
check("full-depth reversal gouge reads near-WHITE", wh.min() > 0.75,
      f"RGB {np.round(wh,3)}")
bw = dye_tau_weights(0.6, "bw")
check("B&W emulsion scratch stays NEUTRAL (no dye layers)",
      abs(bw[0] - bw[1]) < 1e-12 and abs(bw[1] - bw[2]) < 1e-12, f"{np.round(bw,4)}")


# ---------------------------------------------------------------------------
print("\nT5 — resolution independence (ref-px @1024)")
# ---------------------------------------------------------------------------
t1 = build_tau(1024, 1024, 99, density=0.5, hair_amount=0.0, scratch_count=0)
t2 = build_tau(2048, 2048, 99, density=0.5, hair_amount=0.0, scratch_count=0)
o1 = np.clip(1.0 - t1[..., 1], 0, None)
o2 = np.clip(1.0 - t2[..., 1], 0, None).reshape(1024, 2, 1024, 2).mean(axis=(1, 3))
corr = float(np.corrcoef(gaussian_filter(o1, 3.0).ravel(),
                         gaussian_filter(o2, 3.0).ravel())[0, 1])
check("layout agrees across resolution (downsample + correlate)", corr > 0.85,
      f"Pearson r={corr:.4f}")
# integrated MASS, not a thresholded pixel count: the softness blur conserves
# mass but spreads it, so a count-based test flags correct behaviour as drift
m1 = float(o1.sum()) / (1024 * 1024)
m2 = float(np.clip(1.0 - t2[..., 1], 0, None).sum()) / (2048 * 2048)
check("integrated opacity mass stable across resolution",
      abs(m1 - m2) / max(m1, 1e-9) < 0.10, f"{m1:.6f} vs {m2:.6f}")
m_lo = float(np.clip(1.0 - build_tau(512, 512, 99, density=0.5, hair_amount=0.0,
                                     scratch_count=0)[..., 1], 0, None).sum()) / (512 * 512)
check("KNOWN LIMIT sub-1024 inflates mass (sub-pixel motes), documented",
      m_lo > m1, f"512px {m_lo:.6f} vs 1024px {m1:.6f}")


# ---------------------------------------------------------------------------
print("\nT6 — determinism and transport axis")
# ---------------------------------------------------------------------------
check("same seed reproduces exactly",
      np.array_equal(build_tau(256, 256, 5, density=0.4),
                     build_tau(256, 256, 5, density=0.4)))
check("different seed differs",
      not np.array_equal(build_tau(256, 256, 5, density=0.4),
                         build_tau(256, 256, 6, density=0.4)))
# auto axis follows the LONG edge: landscape -> horizontal streaks
land = 1.0 - build_tau(256, 512, 4, density=0.0, dust_amount=0.0, dirt_amount=0.0,
                       hair_amount=0.0, scratch_count=6,
                       transport_axis="auto")[..., 1]
port = 1.0 - build_tau(512, 256, 4, density=0.0, dust_amount=0.0, dirt_amount=0.0,
                       hair_amount=0.0, scratch_count=6,
                       transport_axis="auto")[..., 1]
land_h = land.sum(axis=1).std() / (land.sum(axis=1).mean() + 1e-9)
land_v = land.sum(axis=0).std() / (land.sum(axis=0).mean() + 1e-9)
port_h = port.sum(axis=1).std() / (port.sum(axis=1).mean() + 1e-9)
port_v = port.sum(axis=0).std() / (port.sum(axis=0).mean() + 1e-9)
check("auto transport axis follows the long edge (landscape -> horizontal)",
      land_h > land_v and port_v > port_h,
      f"landscape row/col var {land_h:.2f}/{land_v:.2f}, portrait {port_h:.2f}/{port_v:.2f}")


# ---------------------------------------------------------------------------
print("\nT7 — node-level identity, batch, mask, dtype")
# ---------------------------------------------------------------------------
node = DarkroomFilmDamage()
img = torch.rand(2, 96, 128, 3, dtype=torch.float32)

out, msk = node.execute(img, density=0.0)
check("density 0 = bit-exact passthrough",
      torch.equal(out, img), f"max|d|={float((out-img).abs().max()):.2e}")
check("passthrough still returns a correctly shaped empty mask",
      tuple(msk.shape) == (2, 96, 128) and float(msk.abs().max()) == 0.0)

out2, msk2 = node.execute(img, density=1.0, seed=3)
check("output shape and dtype preserved",
      out2.shape == img.shape and out2.dtype == torch.float32)
check("output stays in [0,1] and is finite",
      bool(torch.isfinite(out2).all()) and float(out2.min()) >= 0.0
      and float(out2.max()) <= 1.0)
check("mask shape is (B, H, W) with real coverage",
      tuple(msk2.shape) == (2, 96, 128) and 0.0 < float(msk2.max()) <= 1.0,
      f"max={float(msk2.max()):.3f}")
check("vary_per_frame gives batch frames DIFFERENT defect fields",
      not torch.equal(out2[0], out2[1]))
out3, _ = node.execute(img, density=1.0, seed=3, vary_per_frame=False)
d0 = (out3[0] - img[0]).abs().sum(dim=2)
d1 = (out3[1] - img[1]).abs().sum(dim=2)
check("vary_per_frame OFF reuses one field across the batch",
      float(((d0 > 1e-6) ^ (d1 > 1e-6)).float().mean()) < 0.01)
outa, _ = node.execute(img, density=1.0, seed=11)
outb, _ = node.execute(img, density=1.0, seed=11)
check("node execute is deterministic for a fixed seed", torch.equal(outa, outb))

# the headline behaviour, end to end through the node
grey_t = torch.full((1, 64, 64, 3), 0.45, dtype=torch.float32)
n_out, n_m = node.execute(grey_t, density=2.0, seed=5, defect_origin=ORIGINS[0])
p_out, _ = node.execute(grey_t, density=2.0, seed=5, defect_origin=ORIGINS[1])
sel = n_m[0] > 0.05
check("NODE-LEVEL: same seed, origin flip inverts the defect sign",
      float(n_out[0][sel].mean()) > 0.45 > float(p_out[0][sel].mean()),
      f"neg={float(n_out[0][sel].mean()):.3f} pos={float(p_out[0][sel].mean()):.3f}")


# ---------------------------------------------------------------------------
print("\nT8 — perf budget")
# ---------------------------------------------------------------------------
import time
for (h, w, label, budget) in ((1024, 1024, "1024^2", 2.0), (3840, 2160, "4K", 12.0)):
    t0 = time.time()
    node.execute(torch.full((1, h, w, 3), 0.4, dtype=torch.float32),
                 density=0.5, seed=1)
    el = time.time() - t0
    check(f"perf {label} under {budget}s", el < budget, f"{el:.2f}s")


print("\n" + "=" * 74)
print(f"RESULT: {len(PASS)} passed, {len(FAIL)} failed")
for f in FAIL:
    print(f"   FAILED: {f}")
print("=" * 74)
sys.exit(1 if FAIL else 0)
