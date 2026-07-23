"""
Offline validation for the Halation node -- "teeth before trust". Run with the
ComfyUI embedded python. No pytest; prints PASS/FAIL per check and exits
nonzero if any check fails.

Loads the node via the namespace-shim pattern (cf. tools/test_eberhard.py) so
the node's `from ..utils...` relative imports resolve WITHOUT triggering the
top-level package __init__ (server_routes / ComfyUI runtime).

Teeth (per docs/halation-derivation.md "Teeth" + plan lovely-coalescing-rain.md):
  1. RING GEOMETRY (headline): point source -> radial profile peaks off-center
     at r_c (+-15%), rim/center > 2.5. NEGATIVE: a gaussian kernel of the same
     characteristic radius has its max AT r=0 (the fake-glow class this model
     is not).
  2. ENERGY: kernel sums to 1 (1e-9); mean-lift == strength_frac * mean(H) in
     the unclipped regime (1e-6).
  3. TINT ORDERING: CineStill halo annulus w_R > w_G > w_B. NEGATIVE: flipped
     weights (Custom preset, B/G/R order) fail the same ordering check.
  4. KILL SWITCHES: ah_strength=1 -> output == input (early-exit, (1-ah)<=0);
     strength=0 -> bit-identical early-exit (same tensor object).
  5. RESOLUTION INDEPENDENCE: ring_radius held constant in ref-px @1024, point
     source through the node at 512 vs 2048 -> measured peak radius scales
     ~4x (+-0.6), the long_edge/1024 convention.
  6. MONOTONICITY: strength 0.2 < 0.5 < 0.8 -> halo-region energy strictly
     increasing.
  7. PERF: 1024^2 and 3840x2160 timings printed (budget: warn if >5s@1024^2,
     >25s@4K -- non-fatal).

Also saves prod_night_default.png + prod_photo_default.png (if
../_sabattier_spike/00_original.png exists) into ../_halation_spike/ for
eyeball.
"""

import os
import sys
import time
import types
import importlib.util

import numpy as np
import torch
from PIL import Image

PACK_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ---------------------------------------------------------------------------
# Synthetic package shim (cf. test_eberhard.py).
# ---------------------------------------------------------------------------
PKG = "dr_pkg_halation"

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
    PKG + ".nodes.halation", os.path.join(PACK_ROOT, "nodes", "halation.py")
)
halation_mod = importlib.util.module_from_spec(spec)
sys.modules[PKG + ".nodes.halation"] = halation_mod
spec.loader.exec_module(halation_mod)

Halation = halation_mod.Halation

# utils.image is now loaded as a side effect of the node's relative import;
# grab it directly too, for teeth that need the raw kernel builder.
import importlib
dr_image = importlib.import_module(PKG + ".utils.image")
halation_psf = dr_image.halation_psf

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
    node = Halation()
    out = node.execute(to_tensor(img_np), **kw)[0]
    return from_tensor(out), out


def srgb_to_linear(x):
    """Mirror of utils.color.srgb_to_linear -- measure in the space the node
    actually composites in (linear light)."""
    x = np.clip(x, 0.0, 1.0).astype(np.float32)
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4).astype(np.float32)


def radial_profile(k):
    c = k.shape[0] // 2
    y, x = np.mgrid[:k.shape[0], :k.shape[1]]
    r = np.hypot(x - c, y - c).astype(int)
    prof = np.bincount(r.ravel(), weights=k.ravel())
    cnt = np.bincount(r.ravel())
    return prof / np.maximum(cnt, 1)


def point_source(size, value=0.5):
    """Dirac-ish point-source image: a single bright pixel on a zero field.
    Zero field means srgb_to_linear(0)=0, so no knee interaction, and the
    convolution response IS the kernel itself (times w_c), scaled by frac.
    value defaults to 0.5 (not 1.0): the source pixel itself sits at r=0, and
    at value=1.0 it is already at the clip ceiling, so ANY additive halation
    there silently clips to a zero delta -- an artifact that fakes a
    Fresnel-suppressed center rather than measuring it."""
    img = np.zeros((size, size, 3), dtype=np.float32)
    c = size // 2
    img[c, c] = value
    return img


CINESTILL = "CineStill 800T"


# ---------------------------------------------------------------------------
# 1. RING GEOMETRY (headline) + NEGATIVE gaussian control
# ---------------------------------------------------------------------------
print("\n[1] Ring geometry: point source through the node -> radial peak at r_c, rim/center > 2.5")
SZ = 257
r_c = 20.0
img_pt = point_source(SZ, value=0.5)
out_pt, _ = run(img_pt, preset=CINESTILL, strength=1.0, ring_radius=r_c,
                 tail_length=40.0, ring_tail_balance=1.0, ah_strength=0.0,
                 highlight_knee=0.0)
# node ran at long_edge=257 -> scale = 257/1024, so widget ring_radius=r_c
# lands at physical ring_px = r_c * 257/1024. Work in PHYSICAL px throughout.
scale = SZ / 1024.0
ring_px_actual = r_c * scale

lin_in = srgb_to_linear(img_pt)
lin_out = srgb_to_linear(out_pt)
resp = (lin_out - lin_in)[..., 0]   # R channel: w_R=1.0 for CineStill
prof = radial_profile(resp)
peak = int(np.argmax(prof[1:]) + 1)
check("radial response peaks off-center at ~ring_radius (physical px)",
      abs(peak - ring_px_actual) <= 0.15 * ring_px_actual + 1,
      f"peak at r={peak}, expected~{ring_px_actual:.1f}")
check("center is Fresnel-suppressed vs rim (rim/center > 2.5)",
      prof[peak] > 2.5 * max(prof[0], 1e-12),
      f"rim/center = {prof[peak] / max(prof[0], 1e-12):.2f}")

print("    NEGATIVE CONTROL: a gaussian kernel of the same radius has its max AT r=0 (the fake-glow class)")
yy, xx = np.mgrid[-60:61, -60:61].astype(np.float64)
g = np.exp(-(xx ** 2 + yy ** 2) / (2 * (ring_px_actual ** 2)))
g /= g.sum()
gp = radial_profile(g)
check("gaussian control max at r=0 (NOT off-center -- confirms this test has teeth)",
      np.argmax(gp) == 0, f"gaussian argmax r={np.argmax(gp)}")

# ---------------------------------------------------------------------------
# 2. ENERGY CONSERVATION
# ---------------------------------------------------------------------------
print("\n[2] Energy: kernel sums to 1; mean-lift == strength_frac * mean(H) (unclipped regime)")
k_direct = halation_psf(20.0, 40.0, 0.65, tir_on=True)
check("kernel sums to 1 (1e-9)", abs(k_direct.sum() - 1.0) < 1e-9,
      f"sum={k_direct.sum():.12f}")

rng = np.random.default_rng(42)
test_img = (rng.random((256, 256, 3)) * 0.5).astype(np.float32)  # kept low -> unclipped
strength_ui = 0.4
ring_r, tail_r, bal = 12.0, 30.0, 0.65
out_e, _ = run(test_img, preset=CINESTILL, strength=strength_ui, ring_radius=ring_r,
               tail_length=tail_r, ring_tail_balance=bal, ah_strength=0.0,
               highlight_knee=0.0)
lin_in_e = srgb_to_linear(test_img)
lin_out_e = srgb_to_linear(out_e)
lift = (lin_out_e - lin_in_e).mean()

# Predict independently via the production kernel builder (mirrors the node's
# own math, not a re-derivation) to check the END-TO-END node against it.
sc = 256 / 1024.0
k_pred = halation_psf(ring_r * sc, tail_r * sc, bal, tir_on=True,
                       max_ksize=max(256 - 1, 1))
from scipy.signal import fftconvolve
w_pred = np.array([1.00, 0.32, 0.10])
H_pred = np.empty_like(lin_in_e)
for c in range(3):
    H_pred[..., c] = fftconvolve(lin_in_e[..., c], k_pred, mode="same") * w_pred[c]
frac_pred = strength_ui * 0.25 * (1.0 - 0.0) ** 2
pred_lift = frac_pred * H_pred.mean()
check("mean lift == strength_frac * mean(H) (unclipped, 1e-6)",
      abs(lift - pred_lift) < 1e-6, f"lift={lift:.8f} pred={pred_lift:.8f}")

# ---------------------------------------------------------------------------
# 3. TINT ORDERING + NEGATIVE CONTROL
# ---------------------------------------------------------------------------
print("\n[3] Tint ordering: CineStill halo annulus R > G > B")
img_pt2 = point_source(257, value=0.5)
out_c, _ = run(img_pt2, preset=CINESTILL, strength=0.5, ring_radius=20.0,
               tail_length=40.0, ring_tail_balance=0.65, ah_strength=0.0,
               highlight_knee=0.0)
lin_in_c = srgb_to_linear(img_pt2)
lin_out_c = srgb_to_linear(out_c)
H_c = lin_out_c - lin_in_c
c0 = 257 // 2
yy2, xx2 = np.mgrid[:257, :257]
ann = np.hypot(xx2 - c0, yy2 - c0) > 10
means = [H_c[..., ch][ann].mean() for ch in range(3)]
check("halo annulus R > G > B", means[0] > means[1] > means[2],
      f"R={means[0]:.2e} G={means[1]:.2e} B={means[2]:.2e}")

print("    NEGATIVE CONTROL: flipped weights (Custom, B>G>R order) -> ordering check FAILS")
out_flip, _ = run(img_pt2, preset="Custom", strength=0.5, ring_radius=20.0,
                   tail_length=40.0, ring_tail_balance=0.65, ah_strength=0.0,
                   halo_r=0.10, halo_g=0.32, halo_b=1.00, highlight_knee=0.0)
lin_out_flip = srgb_to_linear(out_flip)
H_flip = lin_out_flip - lin_in_c
means_flip = [H_flip[..., ch][ann].mean() for ch in range(3)]
flipped_ordering_holds = means_flip[0] > means_flip[1] > means_flip[2]
check("flipped-weight ordering does NOT satisfy R>G>B (control fires)",
      not flipped_ordering_holds,
      f"R={means_flip[0]:.2e} G={means_flip[1]:.2e} B={means_flip[2]:.2e}")

# ---------------------------------------------------------------------------
# 4. KILL SWITCHES
# ---------------------------------------------------------------------------
print("\n[4] Kill switches: ah_strength=1 -> output==input; strength=0 -> bit-identical early-exit")
img_kill = (rng.random((64, 64, 3)).astype(np.float32))
t_kill = to_tensor(img_kill)

node = Halation()
out_ah1 = node.execute(t_kill, preset=CINESTILL, strength=0.7, ring_radius=8.0,
                        tail_length=40.0, ring_tail_balance=0.65, ah_strength=1.0)[0]
diff_ah1 = (out_ah1 - t_kill).abs().max().item()
check("ah_strength=1 -> output == input (1e-6)", diff_ah1 < 1e-6, f"max_abs_diff={diff_ah1:.2e}")

out_s0 = node.execute(t_kill, preset=CINESTILL, strength=0.0, ring_radius=8.0,
                       tail_length=40.0, ring_tail_balance=0.65, ah_strength=0.0)[0]
check("strength=0 -> bit-identical early-exit (same tensor object)",
      out_s0 is t_kill, f"is same object: {out_s0 is t_kill}")

# ---------------------------------------------------------------------------
# 5. RESOLUTION INDEPENDENCE
# ---------------------------------------------------------------------------
print("\n[5] Resolution independence: ring_radius constant (ref-px@1024), point source at 512 vs 2048")


def measured_peak(size, ring_radius=8.0):
    img = point_source(size, value=0.5)
    out, _ = run(img, preset=CINESTILL, strength=1.0, ring_radius=ring_radius,
                 tail_length=40.0, ring_tail_balance=1.0, ah_strength=0.0,
                 highlight_knee=0.0)
    lin_in_r = srgb_to_linear(img)
    lin_out_r = srgb_to_linear(out)
    resp_r = (lin_out_r - lin_in_r)[..., 0]
    prof_r = radial_profile(resp_r)
    return int(np.argmax(prof_r[1:]) + 1)


pk_512 = measured_peak(512)
pk_2048 = measured_peak(2048)
ratio = pk_2048 / max(pk_512, 1)
check("ring peak scales with long_edge/1024 (512->2048 ratio ~4.0, +-0.6)",
      abs(ratio - 4.0) <= 0.6, f"pk_512={pk_512} pk_2048={pk_2048} ratio={ratio:.2f}")

# ---------------------------------------------------------------------------
# 6. MONOTONICITY
# ---------------------------------------------------------------------------
print("\n[6] Monotonicity: strength 0.2 < 0.5 < 0.8 -> halo-region energy strictly increasing")
img_mono = point_source(257, value=0.5)
lin_in_m = srgb_to_linear(img_mono)
energies = []
for s in (0.2, 0.5, 0.8):
    out_m, _ = run(img_mono, preset=CINESTILL, strength=s, ring_radius=20.0,
                    tail_length=40.0, ring_tail_balance=0.65, ah_strength=0.0,
                    highlight_knee=0.0)
    lin_out_m = srgb_to_linear(out_m)
    H_m = lin_out_m - lin_in_m
    energies.append(np.abs(H_m[ann]).sum())
check("halo-region energy strictly increasing with strength",
      energies[0] < energies[1] < energies[2],
      f"E(0.2)={energies[0]:.4f} E(0.5)={energies[1]:.4f} E(0.8)={energies[2]:.4f}")

# ---------------------------------------------------------------------------
# 7. PERF
# ---------------------------------------------------------------------------
print("\n[7] Perf")
img1k = np.random.rand(1024, 1024, 3).astype(np.float32)
_ = run(img1k, preset=CINESTILL, strength=0.5)  # warm
t0 = time.time()
_ = run(img1k, preset=CINESTILL, strength=0.5)
t_1k = time.time() - t0
warn_1k = " *** WARN: over 5s budget ***" if t_1k > 5.0 else ""
print(f"  1024x1024: {t_1k:.3f} s{warn_1k}")

img4k = np.random.rand(2160, 3840, 3).astype(np.float32)
t0 = time.time()
_ = run(img4k, preset=CINESTILL, strength=0.5)
t_4k = time.time() - t0
warn_4k = " *** WARN: over 25s budget ***" if t_4k > 25.0 else ""
print(f"  3840x2160: {t_4k:.3f} s{warn_4k}")

# ---------------------------------------------------------------------------
# Save eyeball PNGs into _halation_spike/
# ---------------------------------------------------------------------------
print("\n[save] eyeball PNGs")
SPIKE = os.path.join(os.path.dirname(PACK_ROOT), "_halation_spike")
PHOTO = os.path.join(os.path.dirname(PACK_ROOT), "_sabattier_spike", "00_original.png")


def save(img, name):
    Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8)).save(os.path.join(SPIKE, name))
    print(f"  saved {name}")


night = np.full((512, 512, 3), 0.02, dtype=np.float32)
for (cy, cx, rad, val) in [(100, 100, 6, 1.0), (100, 300, 10, 1.0),
                            (300, 100, 4, 0.6), (300, 300, 14, 1.0),
                            (400, 420, 2, 1.0), (180, 420, 8, 0.35)]:
    yy3, xx3 = np.mgrid[:512, :512]
    night[np.hypot(yy3 - cy, xx3 - cx) < rad] = val
out_night, _ = run(night, preset=CINESTILL, strength=0.35, ring_radius=8.0,
                    tail_length=40.0, ring_tail_balance=0.65, ah_strength=0.0)
save(out_night, "prod_night_default.png")

if os.path.exists(PHOTO):
    photo = np.asarray(Image.open(PHOTO).convert("RGB"), dtype=np.float32) / 255.0
    out_photo, _ = run(photo, preset=CINESTILL, strength=0.35, ring_radius=8.0,
                        tail_length=40.0, ring_tail_balance=0.65, ah_strength=0.0)
    save(out_photo, "prod_photo_default.png")
else:
    print(f"  (photo not found at {PHOTO}, skipped)")

# ---------------------------------------------------------------------------
print("\n" + ("ALL CHECKS PASSED" if PASS else "SOME CHECKS FAILED"))
sys.exit(0 if PASS else 1)
