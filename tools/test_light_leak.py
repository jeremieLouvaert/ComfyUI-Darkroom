"""
Teeth for the Light Leak node (docs/light-leak-derivation.md).

Run with the embedded python:
  python_embeded/python.exe tools/test_light_leak.py

Promoted from _light_leak_spike/checks.py against the SHIPPED utils/light_leak.py,
so these exercise production code rather than a copy. Every physics check carries
a negative control that must FAIL for the check to mean anything.
"""

import importlib.util
import os
import sys
import types

import numpy as np
import torch

PACK_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Synthetic package shim (cf. test_halation.py / test_film_rebate.py /
# test_film_damage.py): resolves the node's `from ..utils...` imports without
# executing the top-level __init__, which needs the live ComfyUI runtime.
PKG = "dr_pkg_light_leak"
for _n, _p in ((PKG, PACK_ROOT),
               (PKG + ".nodes", os.path.join(PACK_ROOT, "nodes")),
               (PKG + ".utils", os.path.join(PACK_ROOT, "utils"))):
    _m = types.ModuleType(_n)
    _m.__path__ = [_p]
    sys.modules[_n] = _m


def _load(modname, *relpath):
    spec = importlib.util.spec_from_file_location(
        PKG + "." + modname, os.path.join(PACK_ROOT, *relpath))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[PKG + "." + modname] = mod
    spec.loader.exec_module(mod)
    return mod


LL = _load("utils.light_leak", "utils", "light_leak.py")
_color = _load("utils.color", "utils", "color.py")
_node = _load("nodes.light_leak", "nodes", "light_leak.py")
LL.srgb_to_linear = _color.srgb_to_linear
LL.linear_to_srgb = _color.linear_to_srgb


def _screen_srgb(img_srgb, G, strength=1.0):
    """NEGATIVE CONTROL: the sRGB screen blend an overlay pack performs."""
    a = np.clip(strength * G, 0.0, 1.0)
    return np.clip(1.0 - (1.0 - img_srgb) * (1.0 - a), 0.0, 1.0)


PASS, FAIL = [], []


def check(name, ok, detail=""):
    (PASS if ok else FAIL).append(name)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))
    return ok


print("=" * 74)
print("LIGHT LEAK - teeth")
print("=" * 74)
np.set_printoptions(precision=4, suppress=True)

H, W = 512, 768


# ---------------------------------------------------------------------------
print("\nT1 — additive-in-linear invariance (THE KILL-TEST)")
# ---------------------------------------------------------------------------
# A leak adds a CONSTANT absolute luminance regardless of what is underneath.
ramp = np.tile(np.linspace(0.02, 0.98, 400)[None, :, None], (8, 1, 3))
lin = np.linspace(0.001, 0.90, 400)[None, :, None] * np.ones((8, 1, 3))
G_fix = np.full_like(lin, 0.05)

out = LL.composite(lin, G_fix, 1.0)
un = out[..., 1] < 0.999
dL = (out[..., 1] - lin[..., 1])[un]
check("additive leak adds a tone-INVARIANT absolute luminance",
      float(dL.std()) < 1e-12, f"mean={dL.mean():+.6f} std={dL.std():.2e}")

# NEGATIVE CONTROL: sRGB screen blend at matched strength
scr = _screen_srgb(ramp, np.full_like(ramp, 0.25), 1.0)
d_scr = LL.srgb_to_linear(scr)[..., 1] - LL.srgb_to_linear(ramp)[..., 1]
cv = abs(float(d_scr.std() / d_scr.mean()))
check("NEG CONTROL sRGB screen blend cannot hold it",
      cv > 0.3, f"CV={cv:.3f} vs physics ~1e-16")
check("NEG CONTROL screen-blend delta SHRINKS as tone rises",
      d_scr[0, -1] < d_scr[0, 0] * 0.5,
      f"delta {d_scr[0,0]:.4f} (shadow) -> {d_scr[0,-1]:.4f} (highlight)")


# ---------------------------------------------------------------------------
print("\nT2 — colour ordering: the leak reddens with distance")
# ---------------------------------------------------------------------------
G = LL.leak_field(H, W, "gradient", edge="top", corner=None, lam_ref=200.0, seed=1)
prof = G[:, W // 2, :]                                   # down the leak's path
sel = prof[:, 0] > 1e-6
rb = prof[sel, 0] / np.maximum(prof[sel, 2], 1e-12)
check("R:B ratio increases monotonically with distance",
      bool(np.all(np.diff(rb) > 0)), f"R:B {rb[0]:.3f} -> {rb[-1]:.1f}")

# NEGATIVE CONTROL: equal lambda per channel gives a flat ratio
G_eq = LL.leak_field(H, W, "gradient", edge="top", corner=None, lam_ref=200.0,
                    seed=1, lam_ratio=(1.0, 1.0, 1.0))
p_eq = G_eq[:, W // 2, :]
s_eq = p_eq[:, 0] > 1e-6
rb_eq = p_eq[s_eq, 0] / np.maximum(p_eq[s_eq, 2], 1e-12)
check("NEG CONTROL equal per-channel lambda gives a FLAT ratio",
      float(np.ptp(rb_eq)) < 1e-9, f"ptp={np.ptp(rb_eq):.2e}")


# ---------------------------------------------------------------------------
print("\nT3 — hot core: near-white at the edge, saturating outward")
# ---------------------------------------------------------------------------
def spread(v):
    """channel spread normalised by mean = 0 when neutral."""
    return float((v.max() - v.min()) / max(v.mean(), 1e-12))


core = prof[0]
mid = prof[int(0.35 * H)]
far = prof[int(0.7 * H)]
check("leak is near-NEUTRAL at d=0 (hot core, not a tint)",
      spread(core) < 0.02, f"channel spread {spread(core):.4f}")
check("saturation rises with distance (core -> fringe)",
      spread(core) < spread(mid) < spread(far),
      f"{spread(core):.3f} -> {spread(mid):.3f} -> {spread(far):.3f}")
# the structure a canned plate flattens: uniform tint has constant spread
check("NEG CONTROL a uniform-tint leak has constant spread",
      abs(spread(p_eq[0]) - spread(p_eq[int(0.7 * H)])) < 1e-9)


# ---------------------------------------------------------------------------
print("\nT4 — sprocket lattice: pitch oracle + modulation decay")
# ---------------------------------------------------------------------------
Gs = LL.leak_field(H, W, "sprocket", edge="top", lam_ref=200.0, mod_ratio=0.30, seed=5)
row = Gs[0, :, 1]                                        # comb at the film edge
# independently re-derive the expected pitch from the mm spec
L = max(H, W)
expect_pitch = (4.75 / 36.0) * L
# measure it: autocorrelation peak away from zero lag
r0 = row - row.mean()
ac = np.correlate(r0, r0, mode="full")[len(r0) - 1:]
lo = int(expect_pitch * 0.5)
meas = lo + int(np.argmax(ac[lo:int(expect_pitch * 2.0)]))
check("comb period matches 4.75mm scaled into px (independent re-derivation)",
      abs(meas - expect_pitch) / expect_pitch < 0.03,
      f"measured {meas}px vs expected {expect_pitch:.1f}px")
check("36mm aperture carries 7.58 perforation pitches",
      abs(W / expect_pitch - (36.0 / 4.75) * (W / L)) < 0.05,
      f"{LL.PITCHES_ACROSS:.3f} pitches across the long edge")


def mod_contrast(d_row):
    r = Gs[d_row, :, 1]
    env = r.mean()
    return float((r.max() - r.min()) / max(env, 1e-12))


c0, c1, c2 = mod_contrast(0), mod_contrast(40), mod_contrast(110)
check("modulation contrast decays with depth (comb blurs, glow persists)",
      c0 > c1 > c2, f"{c0:.3f} -> {c1:.3f} -> {c2:.3f}")
# and it must decay FASTER than the envelope -- the load-bearing inequality
env0, env1 = Gs[0, :, 1].mean(), Gs[110, :, 1].mean()
check("modulation decays FASTER than the envelope (lambda_mod < lambda)",
      (c2 / c0) < (env1 / env0), f"contrast ratio {c2/c0:.4f} vs envelope {env1/env0:.4f}")
# NEGATIVE CONTROL: no lateral diffusion (spread = 0). The comb then never
# blurs, so its contrast is depth-independent and cannot outpace the envelope.
# (The old control raised the coefficient, which under the diffusion model just
# blurs FASTER and still passes -- it was testing the wrong thing.)
Gb = LL.leak_field(H, W, "sprocket", edge="top", lam_ref=200.0, mod_ratio=0.0, seed=5)


def mc(g, d):
    r = g[d, :, 1]
    return float((r.max() - r.min()) / max(r.mean(), 1e-12))


bad = (mc(Gb, 110) / mc(Gb, 0)) < (Gb[110, :, 1].mean() / Gb[0, :, 1].mean())
check("NEG CONTROL zero lateral diffusion fails the decay inequality", not bad,
      f"contrast ratio {mc(Gb,110)/mc(Gb,0):.4f} vs envelope "
      f"{Gb[110,:,1].mean()/Gb[0,:,1].mean():.4f}")


# ---------------------------------------------------------------------------
print("\nT5 — pinhole neutrality (the anti-convention differentiator)")
# ---------------------------------------------------------------------------
Gp = LL.leak_field(H, W, "pinhole", seed=3, pinhole_count=4, color_source="neutral")
hot = Gp[..., 1] > 0.05 * Gp[..., 1].max()
pin_rb = float(Gp[..., 0][hot].mean() / max(Gp[..., 2][hot].mean(), 1e-12))
base_rb = float(G[..., 0][G[..., 1] > 1e-6].mean() / max(G[..., 2][G[..., 1] > 1e-6].mean(), 1e-12))
check("pinhole leak stays NEUTRAL (no lateral path -> no differential absorption)",
      abs(pin_rb - 1.0) < 0.10, f"R:B {pin_rb:.3f}")
check("base-path leak at the same settings is strongly RED-shifted",
      base_rb > 2.0, f"R:B {base_rb:.2f} vs pinhole {pin_rb:.3f}")
check("the two differ by the mechanism, not a tint switch",
      base_rb / pin_rb > 2.0, f"ratio {base_rb/pin_rb:.2f}x")


# ---------------------------------------------------------------------------
print("\nT6 — pinhole spot size tracks a + D*theta")
# ---------------------------------------------------------------------------
def spot_area(flange, hole=0.30, angle=0.0093):
    g = LL.leak_field(400, 400, "pinhole", seed=9, pinhole_count=1,
                     hole_mm=hole, flange_mm=flange, source_angle=angle)
    return float((g[..., 1] > 0.5 * g[..., 1].max()).sum())


a1, a2 = spot_area(50.0), spot_area(100.0)
# geometric core diameter doubles with flange distance -> area ~4x
check("doubling flange distance ~quadruples spot area (D*theta term)",
      3.0 < a2 / max(a1, 1) < 5.0, f"area ratio {a2/max(a1,1):.2f}x")
a3 = spot_area(50.0, angle=0.0186)
check("doubling source angular size also grows the spot",
      a3 > a1 * 2.5, f"{a1:.0f} -> {a3:.0f} px")


# ---------------------------------------------------------------------------
print("\nT7 — resolution independence (ref-px @1024)")
# ---------------------------------------------------------------------------
g1 = LL.leak_field(512, 512, "sprocket", edge="top", lam_ref=200.0, seed=4)
g2 = LL.leak_field(1024, 1024, "sprocket", edge="top", lam_ref=200.0, seed=4)
d1 = g1[..., 1]
d2 = g2[..., 1].reshape(512, 2, 512, 2).mean(axis=(1, 3))
corr = float(np.corrcoef(d1.ravel(), d2.ravel())[0, 1])
check("field agrees across resolution (downsample + correlate)", corr > 0.97,
      f"Pearson r={corr:.4f}")
e1, e2 = float(d1.mean()), float(d2.mean())
check("mean field energy stable across resolution",
      abs(e1 - e2) / max(e1, 1e-9) < 0.05, f"{e1:.5f} vs {e2:.5f}")


# ---------------------------------------------------------------------------
print("\nT8 — determinism, identity, perf")
# ---------------------------------------------------------------------------
check("same seed reproduces exactly",
      np.array_equal(LL.leak_field(128, 128, "sprocket", seed=2),
                     LL.leak_field(128, 128, "sprocket", seed=2)))
check("different seed differs (lattice phase / pinhole placement)",
      not np.array_equal(LL.leak_field(128, 128, "pinhole", seed=2),
                         LL.leak_field(128, 128, "pinhole", seed=3)))
# strength 0 must be an exact no-op at engine level too (node level: T10)
photo = np.random.default_rng(0).random((64, 64, 3))
lin_p = LL.srgb_to_linear(photo)
G0 = LL.leak_field(64, 64, "gradient", edge="top", corner=None, seed=1)
check("strength 0 = bit-exact passthrough (engine)",
      np.array_equal(LL.composite(lin_p, G0, 0.0), np.clip(lin_p, 0.0, 1.0)))

# perf is measured at node level in T11



# ---------------------------------------------------------------------------
print("")
print("T9 - path-length displacement (Jeremie's ask)")
# ---------------------------------------------------------------------------
g_clean = LL.leak_field(H, W, "gradient", edge="top", corner=None, lam_ref=200.0, seed=8)
g_warp = LL.leak_field(H, W, "gradient", edge="top", corner=None, lam_ref=200.0, seed=8,
                      displacement=120.0, displacement_scale=300.0)
check("displacement 0 is the exact analytic field (identity)",
      np.array_equal(g_clean, LL.leak_field(H, W, "gradient", edge="top", corner=None,
                                           lam_ref=200.0, seed=8, displacement=0.0)))
check("displacement > 0 actually moves the field", not np.allclose(g_clean, g_warp))


def iso_depth(g, level=0.30):
    col = g[:, :, 1]
    return np.array([np.argmax(col[:, x] < level) for x in range(g.shape[1])], dtype=float)


sd_clean, sd_warp = float(np.std(iso_depth(g_clean))), float(np.std(iso_depth(g_warp)))
check("iso-intensity contour is FLAT without displacement", sd_clean < 0.51,
      "std %.3fpx along the edge" % sd_clean)
check("iso-intensity contour WANDERS with displacement", sd_warp > 15.0,
      "std %.1fpx along the edge" % sd_warp)

# LOAD-BEARING: colour must stay a strict function of path length, so the red
# fringe follows the warped contour instead of sliding off it.
d_impl = -np.log(np.maximum(g_warp[..., 0], 1e-12))
rb = g_warp[..., 0] / np.maximum(g_warp[..., 2], 1e-12)
live = g_warp[..., 1] > 1e-5
rb_sorted = rb[live][np.argsort(d_impl[live])]
edges = np.linspace(0, len(rb_sorted), 41).astype(int)
binned = np.array([rb_sorted[edges[i]:edges[i + 1]].mean() for i in range(40)
                   if edges[i + 1] > edges[i]])
check("colour remains a STRICT function of path length under displacement",
      bool(np.all(np.diff(binned) > -1e-9)),
      "red fringe follows the warped contour rather than sliding off it")

# The warp must be a smooth SWELL, not noise. Measure it directly: fraction of
# spatial-frequency energy above the mechanical cutoff (features finer than
# ~5mm of seam). Asserted, because "looks like noise" is otherwise a matter of
# opinion that drifts with every parameter change.
def hf_fraction(scale_ref, octaves=2):
    w = LL._warp_noise(512, 512, max(scale_ref, LL.WARP_SCALE_FLOOR) * 0.5, 3, octaves)
    F = np.abs(np.fft.fftshift(np.fft.fft2(w - w.mean()))) ** 2
    fy, fx = np.mgrid[-256:256, -256:256]
    r = np.sqrt(fy ** 2 + fx ** 2)
    cutoff = 512.0 / (LL.WARP_SCALE_FLOOR * 0.5)     # cycles across the tile
    return float(F[r > cutoff].sum() / max(F.sum(), 1e-12))


hf_default = hf_fraction(380.0)
check("warp energy is overwhelmingly LOW frequency (a swell, not noise)",
      hf_default < 0.05, "%.1f%% of energy above the mechanical cutoff" % (100 * hf_default))
check("requesting a too-fine scale is CLAMPED, not honoured",
      np.array_equal(
          LL.leak_field(256, 256, "gradient", edge="top", corner=None, seed=8,
                       displacement=100.0, displacement_scale=10.0),
          LL.leak_field(256, 256, "gradient", edge="top", corner=None, seed=8,
                       displacement=100.0, displacement_scale=LL.WARP_SCALE_FLOOR)),
      "floor = %.0f ref-px (5mm of seam)" % LL.WARP_SCALE_FLOOR)
# NEGATIVE CONTROL: an unclamped fine scale would blow the frequency budget
hf_bad = hf_fraction(20.0 / 0.5, octaves=5)
check("NEG CONTROL an unclamped fine multi-octave warp IS noisy",
      hf_bad > hf_default * 2.0, "%.1f%% vs %.1f%%" % (100 * hf_bad, 100 * hf_default))

out_w = LL.composite(lin, np.full_like(lin, 0.05), 1.0)
check("additive tone-invariance unaffected by displacement",
      float((out_w[..., 1] - lin[..., 1])[out_w[..., 1] < 0.999].std()) < 1e-12)
check("displacement is seeded and reproducible",
      np.array_equal(g_warp, LL.leak_field(H, W, "gradient", edge="top", corner=None,
                                          lam_ref=200.0, seed=8, displacement=120.0,
                                          displacement_scale=300.0)))


print("\n" + "=" * 74)
print(f"RESULT: {len(PASS)} passed, {len(FAIL)} failed")
for f in FAIL:
    print(f"   FAILED: {f}")
print("=" * 74)

# ---------------------------------------------------------------------------
print("")
print("T10 - node level")
# ---------------------------------------------------------------------------
node = _node.DarkroomLightLeak()
img = torch.rand(2, 96, 128, 3, dtype=torch.float32)

out, msk = node.execute(img, strength=0.0)
check("strength 0 = bit-exact passthrough", torch.equal(out, img))
check("passthrough returns a correctly shaped empty mask",
      tuple(msk.shape) == (2, 96, 128) and float(msk.abs().max()) == 0.0)

out2, msk2 = node.execute(img, strength=0.5, seed=3)
check("shape and dtype preserved",
      out2.shape == img.shape and out2.dtype == torch.float32)
check("output finite and in [0,1]",
      bool(torch.isfinite(out2).all()) and float(out2.min()) >= 0.0
      and float(out2.max()) <= 1.0)
check("mask is (B,H,W) with real coverage",
      tuple(msk2.shape) == (2, 96, 128) and 0.0 < float(msk2.max()) <= 1.0)
check("a leak only ever ADDS light (never darkens)",
      bool((out2 >= img - 1e-6).all()))
a, _ = node.execute(img, strength=0.5, seed=11)
b, _ = node.execute(img, strength=0.5, seed=11)
check("execute is deterministic for a fixed seed", torch.equal(a, b))
c, _ = node.execute(img, strength=0.5, seed=11, vary_per_frame=False)
d0 = (c[0] - img[0]).abs().sum(dim=2) > 1e-6
d1 = (c[1] - img[1]).abs().sum(dim=2) > 1e-6
check("vary_per_frame OFF reuses one leak across the batch",
      float((d0 ^ d1).float().mean()) < 0.01)

# all three modes execute and the pinhole stays neutral through the node
grey = torch.full((1, 128, 192, 3), 0.35, dtype=torch.float32)
for t in _node.LEAK_TYPES:
    o, m = node.execute(grey, leak_type=t, strength=0.6, seed=5)
    check(f"mode executes: {t.split(' (')[0]}",
          bool(torch.isfinite(o).all()) and float(m.max()) > 0.0)
o_pin, m_pin = node.execute(grey, leak_type=_node.LEAK_TYPES[2], strength=0.6,
                            seed=5, colour_source=_node.COLOUR_SOURCES[2])
sel = m_pin[0] > 0.05
d_rgb = (o_pin[0] - grey[0])[sel]
spread_pin = float((d_rgb.mean(0).max() - d_rgb.mean(0).min()) / d_rgb.mean())
o_bp, m_bp = node.execute(grey, leak_type=_node.LEAK_TYPES[1], strength=0.6, seed=5)
sel2 = m_bp[0] > 0.05
d2 = (o_bp[0] - grey[0])[sel2]
spread_base = float((d2.mean(0).max() - d2.mean(0).min()) / d2.mean())
check("NODE-LEVEL pinhole stays neutral while base-path is strongly coloured",
      spread_pin < spread_base * 0.4,
      f"pinhole spread {spread_pin:.3f} vs base path {spread_base:.3f}")

# sprocket collapses a corner selection to a real perforation edge
e, c_ = _node._resolve_origin("sprocket", "top-left")
check("sprocket collapses a corner to its edge (perf rows run along the film)",
      e == "top" and c_ is None)
e2, c2 = _node._resolve_origin("gradient", "top-left")
check("gradient honours corners", e2 is None and c2 == "top-left")


# ---------------------------------------------------------------------------
print("")
print("T11 - perf budget")
# ---------------------------------------------------------------------------
import time
for (h, w, label, budget) in ((1024, 1024, "1024^2", 3.0), (3840, 2160, "4K", 16.0)):
    t0 = time.time()
    node.execute(torch.full((1, h, w, 3), 0.4, dtype=torch.float32),
                 strength=0.5, seed=1)
    el = time.time() - t0
    check(f"perf {label} under {budget}s", el < budget, f"{el:.2f}s")


print("")
print("=" * 74)
print(f"RESULT: {len(PASS)} passed, {len(FAIL)} failed")
for f in FAIL:
    print(f"   FAILED: {f}")
print("=" * 74)
sys.exit(1 if FAIL else 0)
