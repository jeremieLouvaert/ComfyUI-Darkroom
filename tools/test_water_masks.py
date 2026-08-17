"""
Teeth for native MASK input on the Water Refraction node, per
docs/water-mask-derivation.md (v2, post-adversarial-attack).

IMPLEMENTATION-BLIND: written from the spec, the harness conventions of
tools/test_water_refraction.py and tools/test_lens_masks.py, and the FROZEN
pre-build engine (tools/_frozen_wr_1200.py, vendored verbatim from shipped
v1.20.0) only. This suite never reads utils/water_refraction.py,
nodes/water_refraction.py, or _water_mask_dryrun/dryrun.py -- it imports the
live modules at runtime, through the same PromptServer shim the other
Darkroom suites use, purely to call the public API (execute(), render(),
render_gpu(), render_auto(), grain_deficit()) and the two declared private
seams (spec Sec 7.8: _masked_offsets, _masked_offsets_gpu).

  W1   mask-absent is today's node          (engine-pinned oracle + golden)
  W2   zero mask is identity                 (raw input passthrough, RGBA-safe)
  W3   locality, BOTH outputs                ({out!=in} subset of {m>0}, exact)
  W4   full-res ones ~= absent                (tau4 = 1e-5, not bitwise)
  W5   the multiply is the spec's             (per-engine seam recompute)
  W6   jitter from the MASKED Jacobian        (interior-zero, mean-monotone)
  W7   Fresnel gated                          (flat-water analytic oracle)
  W8   deficit composition                    (exact-0 at m=0, evaluated!=multiplied)
  W9   resize plumbing                        (mode load-bearing, no dead strip)
  W10  batch semantics                        (M=2/B=3 pairing, M=0, all-zero)
  W11  solver isolation                       (h is mask-blind; console unmasked)
  W12  clamp + NaN                            (2.0=1.0, -0.5=0, NaN->0)
  W13  ride-along Sec 7.12a                   (deficit depth_scale forwarding)
  NC   a negative control threaded through every invariant above, MUST fire

Run:
  python_embeded/python.exe ComfyUI-Darkroom/tools/test_water_masks.py

If the mask feature has not landed yet, node.execute()/render()/etc calls
below raise TypeError (unexpected keyword 'mask') -- each row is isolated by
run_block(), so that surfaces as an explicit FAIL per row rather than killing
the whole suite's signal.
"""

import contextlib
import importlib.util
import io
import os
import sys
import time
import types

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

import numpy as np
import torch

from importlib import import_module

# Same trap, same shim as every other Darkroom suite: the pack's __init__
# imports server_routes, which decorates against a live PromptServer.
if "server" not in sys.modules:
    class _Routes:
        def __getattr__(self, _name):
            return lambda *a, **k: (lambda f: f)
    _mod = types.ModuleType("server")
    _mod.PromptServer = type("PromptServer", (), {
        "instance": types.SimpleNamespace(routes=_Routes(),
                                          send_sync=lambda *a, **k: None)})
    sys.modules["server"] = _mod

_pkg = os.path.basename(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
WR = import_module(f"{_pkg}.utils.water_refraction")
NodeMod = import_module(f"{_pkg}.nodes.water_refraction")

# FROZEN oracle -- verbatim pre-build v1.20.0, vendored in tools/ (whitelisted:
# this is the shipped past, not the build under test). tools/ has no
# __init__.py, so load it by file path rather than as a package member.
_FROZEN_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_frozen_wr_1200.py")
_frozen_spec = importlib.util.spec_from_file_location("frozen_wr_1200", _FROZEN_PATH)
FROZEN = importlib.util.module_from_spec(_frozen_spec)
_frozen_spec.loader.exec_module(FROZEN)

_PACK_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GOLD_DIR = os.path.join(_PACK_ROOT, "_water_mask_dryrun", "golden")

PASSED, FAILED = [], []


def check(name, ok, detail=""):
    (PASSED if ok else FAILED).append(name)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}  {detail}")


def must_fail(name, ok, detail=""):
    (PASSED if not ok else FAILED).append(name)
    print(f"  [{'PASS' if not ok else 'FAIL'}] {name}  {detail}"
          f"{'' if not ok else '   <-- CONTROL DID NOT FIRE'}")


def run_block(title, fn):
    """Isolate one invariant's block: an exception here (build not landed,
    a seam named differently than guessed) is an explicit FAIL, not a crash."""
    print(f"\n{title}")
    try:
        fn()
    except Exception as e:
        FAILED.append(f"{title} (exception)")
        print(f"  [FAIL] {title} raised {type(e).__name__}: {e}")


# =============================================================================
# Constants -- the spec's own awkward pins (Sec 8): field 37.0, depth_scale
# 1.3, aperture 0.023, seed 1234. H,W is the spec's primary knife-edge
# (509x767: odd, non-pow2, non-(2^n+1) -- the identity regime NC configs must
# avoid). Numpy engine rows are budget-capped to this size and samples<=4
# (numpy 512^2 measured 4.6s at full samples; GPU is ~0.04s at 512^2 so GPU
# rows can run richer sample counts and larger sizes cheaply).
# =============================================================================

FIELD = 37.0
APERTURE = 0.023
SEED = 1234
H, W = 509, 767
MM_PER_PX = FIELD / W
NUMPY_SAMPLES = 4
GPU_SAMPLES = 8
TAU4 = 1e-5
HAS_CUDA = torch.cuda.is_available()


def band_surface(H=256, W=192, seed=3, pool=0.25, rough=0.06):
    """Copied verbatim (helper conventions only) from tools/test_water_refraction.py,
    with that suite's hardcoded FIELD=40.0 replaced by this suite's module-level
    FIELD=37.0 via the ordinary closure/global lookup -- same function, this
    suite's pinned field width."""
    rng = np.random.default_rng(seed)
    ky = np.fft.fftfreq(H)[:, None]
    kx = np.fft.fftfreq(W)[None, :]
    k = np.hypot(kx, ky)
    filt = np.exp(-0.5 * (np.log(np.maximum(k, 1e-12) * 0.30 * W) / 0.35) ** 2)
    filt[k == 0] = 0.0
    f = np.real(np.fft.ifft2(np.fft.fft2(rng.normal(0, 1, (H, W))) * filt))
    return np.maximum(pool + rough * f / (f.std() + 1e-12), 0.0) * FIELD


def node_cfg(**overrides):
    """Shared node.execute() kwargs, kept small for the 5-minute budget: solver
    nx=48, short pour, few aperture samples. Every field-name here is verified
    against tools/test_water_refraction.py's I10 block and
    _water_mask_dryrun/golden_capture.py's CONFIGS (both non-forbidden)."""
    cfg = dict(field_width_mm=FIELD, surface=NodeMod.SURFACES[0], water_ml=9.0,
               pour_sweep=0.7, sweep_angle=33.0, sample_ms=50.0, settle_ms=0.0,
               depth_scale=1.0, aperture=APERTURE, seed=SEED, sim_resolution=48,
               aperture_samples=4, grain_restore=1.0, mask_min=0.0,
               mask_gamma=1.0)
    # mask_min=0.0 + mask_gamma=1.0 pin the HARD-GATE LINEAR regime: every
    # m==0 exactness row (W2, W3, W8a, the selects) is conditional on it per
    # spec section 11. The shipping defaults (0.15 floor, 2.0 contrast,
    # Jeremie's gate calls) are covered by W14.
    cfg.update(overrides)
    return cfg


# --- mask fixtures (numpy, for utils-level calls) ----------------------------

def np_const(h, w, val):
    return np.full((h, w), float(val), dtype=np.float64)


def np_zeros(h, w):
    return np.zeros((h, w), dtype=np.float64)


def np_ones(h, w):
    return np.ones((h, w), dtype=np.float64)


def np_hard_half(h, w):
    m = np.zeros((h, w), dtype=np.float64)
    m[:, w // 2:] = 1.0
    return m


def np_blob(h, w, feather_px=40.0):
    """A real feathered mask with FRACTIONAL (non-{0,1}) values -- needed by
    W5's m^2 negative control, since m^2 == m identically on a binary mask."""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    cy, cx = h / 2.0, w / 2.0
    r = np.hypot(yy - cy, xx - cx)
    radius = 0.3 * min(h, w)
    return np.clip((radius - r) / feather_px + 0.5, 0.0, 1.0)


# --- mask fixtures (torch, for node-level calls) ------------------------------

def t_const(h, w, val):
    return torch.full((h, w), float(val))


def t_zeros(h, w):
    return torch.zeros(h, w)


def t_ones(h, w):
    return torch.ones(h, w)


def t_hard_half(h, w):
    m = torch.zeros(h, w)
    m[:, w // 2:] = 1.0
    return m


def t_blob(h, w, feather_px=31.0):
    yy, xx = torch.meshgrid(torch.arange(h, dtype=torch.float32),
                             torch.arange(w, dtype=torch.float32), indexing='ij')
    cy, cx = h / 2.0, w / 2.0
    r = torch.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    radius = 0.3 * min(h, w)
    return ((radius - r) / feather_px + 0.5).clamp(0.0, 1.0)


def t_nan_mask(h, w, frac=0.02, seed=99):
    g = torch.Generator().manual_seed(seed)
    m = torch.rand(h, w, generator=g)
    nan_positions = torch.rand(h, w, generator=g) < frac
    m2 = m.clone()
    m2[nan_positions] = float('nan')
    return m2, nan_positions


print("=" * 78)
print("WATER MASKS -- teeth (implementation-blind, docs/water-mask-derivation.md v2)")
print("=" * 78)
print(f"CUDA available: {HAS_CUDA}")
t_start = time.time()


# =============================================================================
# W1 -- mask-absent is today's node.
# (a) same-engine numpy: frozen render()/grain_deficit() vs live render()/
#     grain_deficit() with mask=None, BOTH engines pinned to numpy via
#     render_auto monkeypatch (frozen.render_auto normally dispatches to CUDA
#     on this box, which is not the same-engine comparison W1a needs). Pinned
#     at depth_scale=1.0 ONLY: the depth_scale-forwarding ride-along (Sec
#     7.12a/W13) makes live grain_deficit(depth_scale!=1) diverge from frozen
#     BY DESIGN, so W1's bitwise row must sit where that ride-along is neutral.
# (b) node-level: golden capture (_water_mask_dryrun/golden/) was taken at
#     depth_scale=1.3, where the deficit forwarding fix changes the OUTPUT
#     on purpose (Sec 7.12a) -- so a bitwise compare against golden is not the
#     right tooth there. The real W1b tooth is determinism (mask=None run
#     twice must be bitwise identical); the golden delta is reported as an
#     informational number only, per the task's own resolution of this.
# =============================================================================

def w1a():
    img = np.random.default_rng(SEED).random((H, W, 3))
    h_mm = band_surface(H, W, seed=SEED, pool=0.30, rough=0.08)
    shape = (H, W, 3)
    orig_frozen_ra, orig_live_ra = FROZEN.render_auto, WR.render_auto
    try:
        FROZEN.render_auto = FROZEN.render
        WR.render_auto = WR.render

        img_frozen = FROZEN.render(img, h_mm, FIELD, aperture_ratio=APERTURE,
                                    samples=NUMPY_SAMPLES, depth_scale=1.0,
                                    fresnel=True, seed=SEED)
        img_live = WR.render(img, h_mm, FIELD, aperture_ratio=APERTURE,
                              samples=NUMPY_SAMPLES, depth_scale=1.0,
                              fresnel=True, seed=SEED, mask=None)
        check("W1a numpy render(mask=None) bitwise == frozen oracle",
              np.array_equal(img_frozen, img_live),
              f"max|diff| {float(np.abs(img_frozen - img_live).max()):.2e}")

        dz_frozen = FROZEN.grain_deficit(shape, h_mm, FIELD, aperture_ratio=APERTURE,
                                          samples=NUMPY_SAMPLES, seed=SEED)
        dz_live = WR.grain_deficit(shape, h_mm, FIELD, aperture_ratio=APERTURE,
                                    samples=NUMPY_SAMPLES, seed=SEED,
                                    depth_scale=1.0, mask=None)
        check("W1a numpy grain_deficit(mask=None) bitwise == frozen oracle",
              np.array_equal(dz_frozen, dz_live),
              f"max|diff| {float(np.abs(dz_frozen.astype(np.float64) - dz_live.astype(np.float64)).max()):.2e}")

        m_half = np_const(H, W, 0.5)
        img_half = WR.render(img, h_mm, FIELD, aperture_ratio=APERTURE,
                              samples=NUMPY_SAMPLES, depth_scale=1.0,
                              fresnel=True, seed=SEED, mask=m_half)
        must_fail("W1a NC const-0.5 mask matches frozen oracle (image)",
                  np.array_equal(img_half, img_frozen))
        dz_half = WR.grain_deficit(shape, h_mm, FIELD, aperture_ratio=APERTURE,
                                    samples=NUMPY_SAMPLES, seed=SEED,
                                    depth_scale=1.0, mask=m_half)
        must_fail("W1a NC const-0.5 mask matches frozen oracle (deficit)",
                  np.array_equal(dz_half, dz_frozen))
    finally:
        FROZEN.render_auto, WR.render_auto = orig_frozen_ra, orig_live_ra


def w1b():
    node = NodeMod.WaterRefraction()
    g1_path = os.path.join(GOLD_DIR, "G1_pool.npz")
    input_path = os.path.join(GOLD_DIR, "input_batch.npz")
    if not (os.path.exists(g1_path) and os.path.exists(input_path)):
        print("  [INFO] golden files not present -- skipping golden-informational read")
        batch = torch.rand(2, 192, 256, 3)
    else:
        batch = torch.from_numpy(np.load(input_path)["batch"])

    cfg = dict(field_width_mm=37.0, surface=NodeMod.SURFACES[0], water_ml=9.5,
               pour_sweep=0.7, sweep_angle=33.0, sample_ms=150.0, settle_ms=0.0,
               depth_scale=1.3, aperture=0.023, seed=1234, sim_resolution=64,
               aperture_samples=4, grain_restore=1.0)
    img1, mask1 = node.execute(batch, mask=None, **cfg)
    img2, mask2 = node.execute(batch, mask=None, **cfg)
    check("W1b node-level mask=None is deterministic (run twice, bitwise)",
          torch.equal(img1, img2) and torch.equal(mask1, mask2))

    if os.path.exists(g1_path):
        g1 = np.load(g1_path)
        delta_img = float(np.abs(img1.numpy() - g1["image"]).max())
        delta_mask = float(np.abs(mask1.numpy() - g1["mask"]).max())
        print(f"  [INFO] vs pre-build golden G1_pool (ds=1.3 -- NOT bitwise by design, "
              f"Sec 7.12a ride-along changed this output): image delta {delta_img:.4f}, "
              f"mask delta {delta_mask:.4f}")


run_block("W1a  mask-absent same-engine oracle (numpy, ds=1.0)", w1a)
run_block("W1b  mask-absent node-level (determinism + golden-informational)", w1b)


# =============================================================================
# W2 -- zero mask is identity. A 4-channel, out-of-range ([0,1.4]) input pins
# BOTH reasons at once: the RGBA slice (image[...,:3]) and the unclamped select
# passthrough (Sec 1c reason 3 -- render()'s clip(0,1) alone would clamp a 1.40
# input by 0.40; the select must pass the ORIGINAL through unclamped).
# =============================================================================

def w2():
    node = NodeMod.WaterRefraction()
    torch.manual_seed(SEED)
    Hs, Ws = 160, 200
    batch4 = torch.rand(1, Hs, Ws, 4) * 1.4
    z = t_zeros(Hs, Ws)
    img_out, mask_out = node.execute(batch4, mask=z, **node_cfg())
    check("W2 IMAGE bitwise == input[...,:3] (zeros mask, out-of-range 1.4, 4ch)",
          torch.equal(img_out, batch4[..., :3]),
          f"max|diff| {float((img_out - batch4[..., :3]).abs().max()):.2e}")
    check("W2 deficit exactly 0 at zeros mask", bool((mask_out == 0.0).all()),
          f"max {float(mask_out.max()):.3e}")

    o = t_ones(Hs, Ws)
    img_o, mask_o = node.execute(batch4, mask=o, **node_cfg())
    must_fail("W2 NC ones mask matches input (image)", torch.equal(img_o, batch4[..., :3]))
    must_fail("W2 NC ones mask matches zero deficit", bool((mask_o == 0.0).all()))


run_block("W2  zero mask is identity", w2)


# =============================================================================
# W3 -- locality, BOTH outputs, node level with grain_restore=1.0, hard-half
# mask at the pinned 509x767 knife-edge. NCs: (a) numpy raw render (the RAW
# masked optics, without the node's post-select) leaks into the m=0 boundary
# column via the one-pixel Jacobian overhang (Sec 1c reason 1) -- a ZEROS-mask
# NC is dead on numpy (measured bitwise-safe at every size except an EDGE), so
# the hard-half mask is what makes this fire; (b) GPU raw render leaks a small
# grid_sample residual even on a zeros mask at this exact size (Sec 1c reason
# 2 -- exactly 0 only in the (size-1)-pow-2 regime, which 509x767 is not).
# =============================================================================

def w3():
    node = NodeMod.WaterRefraction()
    torch.manual_seed(SEED)
    batch = torch.rand(1, H, W, 3)
    m = t_hard_half(H, W)
    cfg = node_cfg(grain_restore=1.0)
    img_out, mask_out = node.execute(batch, mask=m, **cfg)

    zero_region = (m == 0)
    zr_img = zero_region.unsqueeze(0).unsqueeze(-1).expand_as(img_out)
    ok_img = torch.equal(img_out[zr_img], batch[zr_img])
    check("W3 locality IMAGE: {out!=in} subset of {m>0} (hard-half, node level)", ok_img)
    zr_mask = zero_region.unsqueeze(0).expand_as(mask_out)
    ok_mask = bool((mask_out[zr_mask] == 0.0).all())
    check("W3 locality DEFICIT: {deficit>0} subset of {m>0} (hard-half, node level)", ok_mask)

    # NC (a): raw numpy render, no node-level select, at the same hard-half
    # mask and size -- the overhang column should leak.
    img_np = batch[0].numpy().astype(np.float64)
    h_mm = band_surface(H, W, seed=SEED, pool=0.30, rough=0.08)
    m_np = np_hard_half(H, W)
    orig_ra = WR.render_auto
    try:
        WR.render_auto = WR.render
        raw = WR.render(img_np, h_mm, FIELD, mask=m_np, aperture_ratio=APERTURE,
                         samples=NUMPY_SAMPLES, depth_scale=1.0, seed=SEED)
    finally:
        WR.render_auto = orig_ra
    boundary_col = W // 2 - 1
    diff_boundary = float(np.abs(raw[:, boundary_col] - img_np[:, boundary_col]).max())
    must_fail("W3 NC numpy raw render (no select) preserves the m=0 boundary column",
              diff_boundary == 0.0,
              f"boundary col max|diff| {diff_boundary:.3f} (spec measured up to 0.52)")

    node_boundary = img_out[0, :, boundary_col].numpy().astype(np.float64)
    input_boundary = batch[0, :, boundary_col].numpy().astype(np.float64)
    check("W3 sanity: NODE output (WITH select) is exact at that same boundary column",
          bool(np.array_equal(node_boundary, input_boundary)))

    # NC (b): GPU raw render, zeros mask, same 509x767 size -- grid_sample
    # residual leaks (measured 8.1e-5); dead on numpy at every size, per spec.
    if HAS_CUDA:
        z_np = np_zeros(H, W)
        raw_gpu = WR.render_gpu(img_np, h_mm, FIELD, mask=z_np, aperture_ratio=APERTURE,
                                 samples=GPU_SAMPLES, depth_scale=1.0, seed=SEED)
        diff_gpu = float(np.abs(raw_gpu - img_np).max())
        must_fail("W3 NC GPU raw render (no select) preserves zeros-mask identity at 509x767",
                  diff_gpu < 1e-8, f"max|diff| {diff_gpu:.2e} (spec measured 8.1e-5)")
    else:
        print("  [SKIP] W3 NC(b) -- no CUDA available")


run_block("W3  locality: {out!=in} subset of {m>0}", w3)


# =============================================================================
# W4 -- full-res ones mask ~= mask absent, node level. Expected bitwise
# (measured 0.0 across a 40-combination box per spec); tooth kept at
# tau4=1e-5 as the stated fallback (the lucky-sample scar).
# =============================================================================

def w4():
    node = NodeMod.WaterRefraction()
    torch.manual_seed(SEED)
    batch = torch.rand(1, H, W, 3)
    cfg = node_cfg()
    img_none, mask_none = node.execute(batch, mask=None, **cfg)
    img_ones, mask_ones = node.execute(batch, mask=t_ones(H, W), **cfg)
    d_img = float((img_none - img_ones).abs().max())
    d_mask = float((mask_none - mask_ones).abs().max())
    check("W4 full-res ones ~= mask=None (IMAGE)", d_img <= TAU4,
          f"max|diff| {d_img:.2e} <= tau4={TAU4}")
    check("W4 full-res ones ~= mask=None (DEFICIT)", d_mask <= TAU4,
          f"max|diff| {d_mask:.2e} <= tau4={TAU4}")

    img_half, _ = node.execute(batch, mask=t_const(H, W, 0.5), **cfg)
    d_half = float((img_none - img_half).abs().max())
    must_fail("W4 NC const-0.5 mask within tau4 of mask=None", d_half <= TAU4,
              f"max|diff| {d_half:.3f} > tau4 (spec measured ~0.93)")


run_block("W4  full-res ones ~= mask absent", w4)


# =============================================================================
# W5 -- the multiply is the spec's: per-engine seam recompute, bitwise against
# the SAME engine only (numpy-vs-GPU jamp_m differs by design, Sec 7.8). The
# mask carries fractional values (np_blob) because on a binary mask m^2==m,
# which would make the m-squared NC unable to discriminate real op-order from
# the defect -- same reasoning test_lens_masks.py uses for its I5 NC.
# =============================================================================

def w5_numpy():
    fn = getattr(WR, "_masked_offsets", None)
    if fn is None:
        check("W5 numpy seam _masked_offsets found", False, "not present on utils module")
        return
    h_mm = band_surface(H, W, seed=SEED, pool=0.30, rough=0.08)
    m = np_blob(H, W)
    dxp_m, dyp_m, jamp_m = fn(h_mm, FIELD, m, ior=WR.N_WATER, depth_scale=1.0)
    dxp, dyp, _ = WR.refraction_offsets(h_mm, FIELD, ior=WR.N_WATER, depth_scale=1.0)

    rec_dxp, rec_dyp = m * dxp, m * dyp
    check("W5 numpy m*dxp bitwise == seam's dxp_m", np.array_equal(rec_dxp, dxp_m))
    check("W5 numpy m*dyp bitwise == seam's dyp_m", np.array_equal(rec_dyp, dyp_m))

    xs0 = np.arange(W)[None, :].astype(np.float64)
    ys0 = np.arange(H)[:, None].astype(np.float64)
    sx_m, sy_m = dxp_m + xs0, dyp_m + ys0
    sxy, sxx = np.gradient(sx_m)
    syy, syx = np.gradient(sy_m)
    detJ_m = sxx * syy - sxy * syx
    rec_jamp = np.sqrt(np.maximum(1.0 - 1.0 / np.maximum(np.abs(detJ_m), 1e-9), 0.0))
    check("W5 numpy recomputed jamp_m (np.gradient of m*D + identity) bitwise == seam's jamp_m",
          np.array_equal(rec_jamp, jamp_m))

    m_sq = m * m
    dxp_m_sq = m_sq * dxp
    must_fail("W5 NC numpy m-squared variant matches seam's dxp_m",
              np.array_equal(dxp_m_sq, dxp_m))


def w5_gpu():
    if not HAS_CUDA:
        print("  [SKIP] no CUDA available")
        return
    fn = getattr(WR, "_masked_offsets_gpu", None)
    if fn is None:
        check("W5 GPU seam _masked_offsets_gpu found", False, "not present on utils module")
        return
    h_mm = band_surface(H, W, seed=SEED, pool=0.30, rough=0.08)
    m = np_blob(H, W)
    dxp_m, dyp_m, jamp_m = fn(h_mm, FIELD, m, ior=WR.N_WATER, depth_scale=1.0)
    dev = dxp_m.device

    hh = torch.tensor(np.ascontiguousarray(h_mm), dtype=torch.float32, device=dev)
    yy, xx = torch.meshgrid(torch.arange(H, device=dev, dtype=torch.float32),
                            torch.arange(W, device=dev, dtype=torch.float32), indexing="ij")

    def grad(a):
        # The engine's own op sequence (render_gpu's internal `grad`, visible
        # in the whitelisted frozen file and named by spec Sec 7.8 as "the
        # literal op sequence render_gpu uses").
        gy, gx = torch.zeros_like(a), torch.zeros_like(a)
        gy[1:-1, :] = (a[2:, :] - a[:-2, :]) / (2 * MM_PER_PX)
        gy[0, :] = (a[1, :] - a[0, :]) / MM_PER_PX
        gy[-1, :] = (a[-1, :] - a[-2, :]) / MM_PER_PX
        gx[:, 1:-1] = (a[:, 2:] - a[:, :-2]) / (2 * MM_PER_PX)
        gx[:, 0] = (a[:, 1] - a[:, 0]) / MM_PER_PX
        gx[:, -1] = (a[:, -1] - a[:, -2]) / MM_PER_PX
        return gy, gx

    gy_, gx_ = grad(hh)
    g = torch.sqrt(gx_ * gx_ + gy_ * gy_)
    th_i = torch.atan(g)
    th_t = torch.asin(torch.clamp(torch.sin(th_i) / WR.N_WATER, -1.0, 1.0))
    dpx = (hh * torch.tan(th_i - th_t)) / MM_PER_PX
    inv = torch.where(g > 1e-12, 1.0 / torch.clamp(g, min=1e-12), torch.zeros_like(g))
    dxp_u, dyp_u = dpx * gx_ * inv, dpx * gy_ * inv

    m_t = torch.tensor(m, dtype=torch.float32, device=dev)
    rec_dxp, rec_dyp = m_t * dxp_u, m_t * dyp_u
    check("W5 GPU m*dxp bitwise == seam's dxp_m", torch.equal(rec_dxp, dxp_m),
          f"max|diff| {(rec_dxp - dxp_m).abs().max().item():.2e}")
    check("W5 GPU m*dyp bitwise == seam's dyp_m", torch.equal(rec_dyp, dyp_m),
          f"max|diff| {(rec_dyp - dyp_m).abs().max().item():.2e}")

    sx_m, sy_m = rec_dxp + xx, rec_dyp + yy
    sxy, sxx = torch.gradient(sx_m, dim=(0, 1))
    syy, syx = torch.gradient(sy_m, dim=(0, 1))
    detJ_m = sxx * syy - sxy * syx
    rec_jamp = torch.sqrt(torch.clamp(1.0 - 1.0 / torch.clamp(detJ_m.abs(), min=1e-9), min=0.0))
    check("W5 GPU recomputed jamp_m bitwise == seam's jamp_m", torch.equal(rec_jamp, jamp_m),
          f"max|diff| {(rec_jamp - jamp_m).abs().max().item():.2e}")

    m_sq_t = m_t * m_t
    dxp_m_sq = m_sq_t * dxp_u
    must_fail("W5 NC GPU m-squared variant matches seam's dxp_m",
              torch.equal(dxp_m_sq, dxp_m))


run_block("W5  displacement scales linearly (numpy seam)", w5_numpy)
run_block("W5  displacement scales linearly (GPU seam)", w5_gpu)


# =============================================================================
# W6 -- jitter from the MASKED Jacobian. Interior m=0 exactly 0 (excluding the
# one boundary column carrying the documented overhang, Sec 2); mean jitter
# monotone at c=0.25/0.5/1.0 (structural ordering, not the spec's exact
# numbers -- those depend on the spec's own pinned surface/config which this
# blind suite cannot reproduce bit-for-bit). NO pointwise claim (the det-J
# parabola bulges past both endpoints at 54.9% of a typical frame -- recorded
# so nobody "fixes" it later; this suite does not test it either).
# =============================================================================

def w6():
    fn = getattr(WR, "_masked_offsets", None)
    if fn is None:
        check("W6 numpy seam _masked_offsets found", False, "not present on utils module")
        return
    h_mm = band_surface(H, W, seed=SEED, pool=0.30, rough=0.08)
    m = np_hard_half(H, W)
    _, _, jamp_m = fn(h_mm, FIELD, m, ior=WR.N_WATER, depth_scale=1.0)

    interior = jamp_m[:, :W // 2 - 1]  # excludes the last m=0 column (the overhang)
    check("W6 interior m=0 jamp_m exactly 0", bool((interior == 0.0).all()),
          f"interior max {float(np.abs(interior).max()):.3e}")

    means = {}
    for c in (0.25, 0.5, 1.0):
        _, _, jamp_c = fn(h_mm, FIELD, np_const(H, W, c), ior=WR.N_WATER, depth_scale=1.0)
        means[c] = float(jamp_c.mean())
    check("W6 mean jitter monotone in uniform c (0.25 < 0.5 < 1.0)",
          means[0.25] < means[0.5] < means[1.0], f"means {means}")

    zero_region = (m == 0)
    _, _, jamp_ones = fn(h_mm, FIELD, np_ones(H, W), ior=WR.N_WATER, depth_scale=1.0)
    frac_nonzero = float((jamp_ones[zero_region] > 0.1).mean())
    must_fail("W6 NC inherit-unmasked-jamp stays ~0 on the hard-half m=0 zone",
              frac_nonzero < 0.3,
              f"fraction of m=0 zone with jamp>0.1: {frac_nonzero:.3f} (spec measured 0.932)")


run_block("W6  jitter from the MASKED Jacobian", w6)


# =============================================================================
# W7 -- Fresnel gated. Flat water (D=0, R=R0 exactly) is the analytic anchor
# (Sec 3): out = clip((1-m*R0)*img + m*R0*env*env_strength). tau7n=1e-12
# numpy; tau7c size-scaled for CUDA float32 coordinate quantisation.
# =============================================================================

def w7_numpy():
    h_flat = np.full((H, W), 6.0, dtype=np.float64)
    img = 0.3 + 0.4 * np.random.default_rng(SEED).random((H, W, 3))  # mid-tone
    m = np_const(H, W, 0.5)
    env_color, env_strength = (0.75, 0.78, 0.82), 1.0

    orig_ra = WR.render_auto
    try:
        WR.render_auto = WR.render
        out = WR.render(img, h_flat, FIELD, mask=m, aperture_ratio=APERTURE,
                         samples=NUMPY_SAMPLES, depth_scale=1.0, fresnel=True,
                         env_color=env_color, env_strength=env_strength, seed=SEED)
    finally:
        WR.render_auto = orig_ra

    R0 = WR.R0
    env = np.array(env_color)[None, None, :] * env_strength
    expected = np.clip((1.0 - m[..., None] * R0) * img + m[..., None] * R0 * env, 0.0, 1.0)
    err = float(np.abs(out - expected).max())
    TAU7N = 1e-12
    check("W7 numpy flat-water Fresnel analytic oracle", err <= TAU7N,
          f"max err {err:.2e} <= tau7n={TAU7N}")

    ungated = np.clip((1.0 - R0) * img + R0 * env, 0.0, 1.0)
    gap = float(np.abs(out - ungated).max())
    must_fail("W7 NC numpy ungated matches gated output at m=0.5 mid-tone",
              gap < 1e-3, f"gap {gap:.4f} (spec measured 8.35e-3, forbidden dead configs avoided)")


def w7_cuda():
    if not HAS_CUDA:
        print("  [SKIP] no CUDA available")
        return
    for tag, Hc, Wc in (("192x256", 192, 256), ("509x767", H, W), ("1080p", 1080, 1920)):
        h_flat = np.full((Hc, Wc), 6.0, dtype=np.float64)
        img = 0.3 + 0.4 * np.random.default_rng(SEED + 1).random((Hc, Wc, 3))
        m = np_const(Hc, Wc, 0.5)
        env_color, env_strength = (0.75, 0.78, 0.82), 1.0
        out = WR.render_gpu(img, h_flat, FIELD, mask=m, aperture_ratio=APERTURE,
                             samples=GPU_SAMPLES, depth_scale=1.0, fresnel=True,
                             env_color=env_color, env_strength=env_strength, seed=SEED)
        R0 = WR.R0
        env = np.array(env_color)[None, None, :] * env_strength
        expected = np.clip((1.0 - m[..., None] * R0) * img + m[..., None] * R0 * env, 0.0, 1.0)
        err = float(np.abs(out - expected).max())
        tau7c = max(3e-5, 1.5e-7 * max(Hc, Wc))
        check(f"W7 CUDA flat-water Fresnel analytic oracle [{tag}]", err <= tau7c,
              f"max err {err:.2e} <= tau7c={tau7c:.2e}")


run_block("W7  Fresnel gated (numpy analytic anchor)", w7_numpy)
run_block("W7  Fresnel gated (CUDA, size-scaled tolerance)", w7_cuda)


# =============================================================================
# W8 -- deficit composition: (a) exact 0 at m=0 (utils + node level);
# (b) evaluated-through-the-masked-warp != c*unmasked, floor gap>0.10 at
# c=0.5 (the control IS the multiplied variant; no monotonicity claim, Sec 4).
# =============================================================================

def w8():
    h_mm = band_surface(H, W, seed=SEED, pool=0.30, rough=0.10)
    shape = (H, W, 3)

    dz0 = WR.grain_deficit(shape, h_mm, FIELD, aperture_ratio=APERTURE, samples=GPU_SAMPLES,
                            seed=SEED, depth_scale=1.0, mask=np_zeros(H, W))
    check("W8a grain_deficit exact 0 at m=0 (utils level)",
          bool((dz0 == 0.0).all()), f"max {float(dz0.max()):.3e}")

    node = NodeMod.WaterRefraction()
    batch = torch.rand(1, H, W, 3)
    _, mask_out = node.execute(batch, mask=t_zeros(H, W), **node_cfg())
    check("W8a node-level deficit exact 0 at m=0",
          bool((mask_out == 0.0).all()), f"max {float(mask_out.max()):.3e}")

    c = 0.5
    mc = np_const(H, W, c)
    deficit_eval = WR.grain_deficit(shape, h_mm, FIELD, aperture_ratio=APERTURE, samples=GPU_SAMPLES,
                                     seed=SEED, depth_scale=1.0, mask=mc)
    deficit_unmasked = WR.grain_deficit(shape, h_mm, FIELD, aperture_ratio=APERTURE, samples=GPU_SAMPLES,
                                         seed=SEED, depth_scale=1.0, mask=None)
    multiplied = c * deficit_unmasked.astype(np.float64)
    gap = float(np.abs(deficit_eval.astype(np.float64) - multiplied).mean())
    check("W8b evaluated-vs-multiplied floor gap > 0.10 at c=0.5",
          gap > 0.10, f"mean gap {gap:.4f} (spec measured 0.468, seed-spread 0.07)")
    must_fail("W8b NC multiplied variant matches evaluated deficit",
              gap < 0.02, f"mean gap {gap:.4f}")


run_block("W8  deficit composition", w8)


# =============================================================================
# W9 -- resize plumbing. Numpy-only engine (Sec 5): zoom(order=1,
# mode='nearest') + [:H,:W] + clip -- mode is load-bearing (patterns.md
# 2026-08-16). All rows go through the NODE (the node resizes the mask before
# handing it to render/grain_deficit; render() itself expects an
# already-full-resolution mask, so resize can only be observed node-level).
# =============================================================================

def w9():
    node = NodeMod.WaterRefraction()
    torch.manual_seed(SEED)
    batch = torch.rand(1, H, W, 3)
    cfg = node_cfg()

    lo, full = t_const(64, 64, 0.7), t_const(H, W, 0.7)
    _, mo_lo = node.execute(batch, mask=lo, **cfg)
    _, mo_full = node.execute(batch, mask=full, **cfg)
    d = float((mo_lo - mo_full).abs().max())
    TAU9 = 1e-9
    check("W9 const-0.7 64x64 mask ~= full-res const-0.7 mask", d <= TAU9,
          f"max|diff| {d:.2e} <= tau9={TAU9}")
    mismatched = t_const(64, 64, 0.3)
    _, mo_mismatch = node.execute(batch, mask=mismatched, **cfg)
    d_wrong = float((mo_mismatch - mo_full).abs().max())
    must_fail("W9 NC const-0.3 (low-res) matches const-0.7 (full-res)", d_wrong <= TAU9,
              f"max|diff| {d_wrong:.3f} > tau9")

    torch.manual_seed(SEED)
    batch_big = torch.rand(1, 1024, 1536, 3)
    ones_512, ones_full = t_ones(512, 512), t_ones(1024, 1536)
    img_resized, _ = node.execute(batch_big, mask=ones_512, **cfg)
    img_full, _ = node.execute(batch_big, mask=ones_full, **cfg)
    d_img = float((img_resized - img_full).abs().max())
    check("W9 ones 512x512-resized ~= true full-res ones (image, no dead strip)",
          d_img < 5e-3, f"max|diff| {d_img:.2e}")
    edge = torch.cat([img_resized[:, 0:2].flatten(), img_resized[:, -2:].flatten(),
                       img_resized[:, :, 0:2].flatten(), img_resized[:, :, -2:].flatten()])
    edge_full = torch.cat([img_full[:, 0:2].flatten(), img_full[:, -2:].flatten(),
                           img_full[:, :, 0:2].flatten(), img_full[:, :, -2:].flatten()])
    d_edge = float((edge - edge_full).abs().max())
    check("W9 no dead edge strip at the frame boundary", d_edge < 5e-3,
          f"max edge|diff| {d_edge:.2e}")

    z_lo = t_zeros(64, 64)
    img_out, mask_out = node.execute(batch, mask=z_lo, **cfg)
    check("W9 zeros 64x64 -> 509x767 (upscale) exact identity",
          bool(torch.equal(img_out, batch)) and bool((mask_out == 0.0).all()))
    z_hi = t_zeros(1024, 1536)
    img_out2, mask_out2 = node.execute(batch, mask=z_hi, **cfg)
    check("W9 zeros 1024x1536 -> 509x767 (downscale) exact identity",
          bool(torch.equal(img_out2, batch)) and bool((mask_out2 == 0.0).all()))

    from scipy.ndimage import zoom as _zoom
    # The edge-zero trap fires at only 9/95 size pairs and 64^2-> upscales are
    # typically NOT among them (patterns.md 2026-08-16, the lucky-sample scar --
    # this NC's first draft used 64^2->509x767 and was dead). Pin the spec's own
    # measured-firing pair: 512^2 -> 1024x1536, factors (2.0, 3.0).
    lo_ones = np.ones((512, 512), dtype=np.float64)
    default_resized = np.clip(_zoom(lo_ones, (2.0, 3.0), order=1)[:1024, :1536],
                              0.0, 1.0)  # scipy DEFAULT mode
    edge_min = min(default_resized[0, :].min(), default_resized[-1, :].min(),
                   default_resized[:, 0].min(), default_resized[:, -1].min())
    must_fail("W9 NC default-mode zoom preserves ones-min>1-eps at the edge",
              edge_min > 1.0 - 1e-12,
              f"edge min {edge_min:.6f} (spec: mode='nearest' is load-bearing)")


run_block("W9  resize plumbing", w9)


# =============================================================================
# W10 -- batch semantics. M=2/B=3, both vary_per_frame settings: frame_i
# bitwise == a SOLO run with mask[min(i,M-1)]; frame0 != frame1 (guards
# against the v1 bug where frames could match for surface-cache reasons);
# M=0 treated as absent + console warn; all-zero mask -> full no-op.
# =============================================================================

def w10():
    node = NodeMod.WaterRefraction()
    Hs, Ws, B = 128, 160, 3
    torch.manual_seed(SEED)
    batch = torch.rand(B, Hs, Ws, 3)

    for vpf_tag, vpf in (("vary_per_frame=False", False), ("vary_per_frame=True", True)):
        mask0, mask1 = t_hard_half(Hs, Ws), t_blob(Hs, Ws)
        m2 = torch.stack([mask0, mask1], dim=0)
        cfg = node_cfg(vary_per_frame=vpf)
        img_batch, mask_batch_out = node.execute(batch, mask=m2, **cfg)
        expected_pairing = [mask0, mask1, mask1]  # min(i, M-1), M=2
        for i in range(B):
            # Under vary_per_frame=True, batch frame i pours with seed
            # s + 7919*i (the node's own stride); the solo run's frame 0 gets
            # offset 0, so its seed widget must carry the stride explicitly or
            # the comparison tests pour seeds, not mask pairing.
            cfg_solo = dict(cfg)
            if vpf:
                cfg_solo["seed"] = cfg_solo["seed"] + 7919 * i
            solo_img, solo_mask = node.execute(batch[i:i + 1], mask=expected_pairing[i], **cfg_solo)
            ok_i = torch.equal(img_batch[i:i + 1], solo_img) and torch.equal(mask_batch_out[i:i + 1], solo_mask)
            check(f"W10 M=2/B=3 {vpf_tag} frame {i} bitwise == solo run with mask[min(i,M-1)]", ok_i)
        check(f"W10 {vpf_tag} frame0 != frame1 (not a surface-cache coincidence)",
              not torch.equal(img_batch[0], img_batch[1]))

    m_empty = torch.zeros(0, Hs, Ws)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        img_m0, mask_m0 = node.execute(batch, mask=m_empty, **node_cfg())
    img_none, mask_none = node.execute(batch, mask=None, **node_cfg())
    check("W10 M=0 output bitwise == mask=None output",
          torch.equal(img_m0, img_none) and torch.equal(mask_m0, mask_none))
    printed = buf.getvalue()
    check("W10 M=0 emits a console warning", len(printed.strip()) > 0,
          f"captured: {printed.strip()[:160]!r}")

    z = t_zeros(Hs, Ws)
    img_z, mask_z = node.execute(batch, mask=z, **node_cfg())
    check("W10 all-zero mask -> image bitwise == input", torch.equal(img_z, batch))
    check("W10 all-zero mask -> deficit exactly 0", bool((mask_z == 0.0).all()))

    shuffled = torch.stack([t_blob(Hs, Ws), t_hard_half(Hs, Ws)], dim=0)  # swapped vs above
    img_shuf, _ = node.execute(batch, mask=shuffled, **node_cfg(vary_per_frame=False))
    img_orig, _ = node.execute(batch, mask=torch.stack([t_hard_half(Hs, Ws), t_blob(Hs, Ws)], dim=0),
                               **node_cfg(vary_per_frame=False))
    must_fail("W10 NC shuffled M=2 pairing matches the original pairing",
              torch.equal(img_shuf, img_orig))


run_block("W10  batch / mask-batch pairing semantics", w10)


# =============================================================================
# W14 -- the mask_min intensity floor (spec section 11, Jeremie's gate call).
# m_eff = mask_min + m*(1-mask_min) at node level, exact op order; 0 restores
# the hard gate bitwise, 1 is ones, the default is 0.3.
# =============================================================================

def w14():
    node = NodeMod.WaterRefraction()
    Hs, Ws = 128, 160
    torch.manual_seed(SEED)
    batch = torch.rand(1, Hs, Ws, 3)

    opt = NodeMod.WaterRefraction.INPUT_TYPES().get("optional", {})
    mm_decl = opt.get("mask_min", (None, {}))[1]
    check("W14 mask_min widget declared, default 0.15",
          abs(float(mm_decl.get("default", -1)) - 0.15) < 1e-12,
          f"declared default {mm_decl.get('default')!r}")
    mg_decl = opt.get("mask_gamma", (None, {}))[1]
    check("W14 mask_gamma widget declared, default 2.0",
          abs(float(mg_decl.get("default", -1)) - 2.0) < 1e-12,
          f"declared default {mg_decl.get('default')!r}")

    # (b) the remap law, bitwise: node(mask=m, mask_min=c) == node(mask=
    # c + m*(1-c) computed in float64 with the node's own prep ops, mask_min=0)
    m_raw = t_blob(Hs, Ws)
    c = 0.4
    m64 = np.clip(np.nan_to_num(m_raw.numpy().astype(np.float64), nan=0.0), 0.0, 1.0)
    pre = torch.from_numpy(c + m64 * (1.0 - c))          # float64 tensor
    a_img, a_mask = node.execute(batch, mask=m_raw, **node_cfg(mask_min=c))
    b_img, b_mask = node.execute(batch, mask=pre, **node_cfg(mask_min=0.0))
    check("W14 remap law bitwise: mask_min=0.4 == pre-remapped mask at mask_min=0",
          torch.equal(a_img, b_img) and torch.equal(a_mask, b_mask))
    wrong = torch.from_numpy(0.5 + m64 * (1.0 - 0.5))
    w_img, _ = node.execute(batch, mask=wrong, **node_cfg(mask_min=0.0))
    must_fail("W14 NC wrong-constant remap (0.5) matches the 0.4 run",
              torch.equal(a_img, w_img))

    # (b2) the gamma law, bitwise: gamma applies BEFORE the floor, the node's
    # exact op order (np.power on the clamped mask, then the affine floor)
    g = 1.7
    pre_g = torch.from_numpy(c + (1.0 - c) * np.power(m64, g))
    ag_img, ag_mask = node.execute(batch, mask=m_raw,
                                   **node_cfg(mask_min=c, mask_gamma=g))
    bg_img, bg_mask = node.execute(batch, mask=pre_g, **node_cfg(mask_min=0.0))
    check("W14 gamma law bitwise: gamma=1.7 == pre-shaped mask at gamma=1",
          torch.equal(ag_img, bg_img) and torch.equal(ag_mask, bg_mask))
    must_fail("W14 NC gamma=1.7 run matches the linear gamma=1 run",
              torch.equal(ag_img, a_img))

    # (c) mask_min=1 with an arbitrary mask == mask absent, bitwise
    c_img, c_mask = node.execute(batch, mask=m_raw, **node_cfg(mask_min=1.0))
    n_img, n_mask = node.execute(batch, mask=None, **node_cfg())
    check("W14 mask_min=1.0 bitwise == mask absent (image + deficit)",
          torch.equal(c_img, n_img) and torch.equal(c_mask, n_mask))

    # (d) behaviour at the shipping default: black zone still moves (subtle),
    # and moves LESS than the white zone (intense) -- his sentence as a row.
    # METRIC NOTE (first draft failed here): on a noise frame at strong
    # settings |out-in| saturates once displacement exceeds the image's
    # correlation length, so BOTH zones read ~0.44 and the ordering is noise
    # (the 2026-07-27 metric scar). A smooth ramp at shallow settings keeps
    # |out-in| proportional to displacement, which is the quantity mask_min
    # actually scales.
    # SECOND METRIC FIX: a hard-half zone comparison confounds mask intensity
    # with the water's own spatial non-uniformity (the pour lands where it
    # lands). Compare the SAME frame under floor strength vs full strength --
    # zeros mask (m_eff = mask_min) vs ones mask (m_eff = 1), identical water.
    ramp = torch.linspace(0.0, 1.0, Ws).view(1, 1, Ws, 1).expand(1, Hs, Ws, 3).contiguous()
    cfg_d = node_cfg(mask_min=0.3, depth_scale=0.25, water_ml=4.0,
                     grain_restore=0.0)
    lo_img, lo_mask = node.execute(ramp, mask=t_zeros(Hs, Ws), **cfg_d)
    hi_img, _ = node.execute(ramp, mask=t_ones(Hs, Ws), **cfg_d)
    d_lo = float((lo_img - ramp).abs().mean())
    d_hi = float((hi_img - ramp).abs().mean())
    check("W14 default floor: zeros mask is NOT identity (subtle effect present)",
          float((lo_img - ramp).abs().max()) > 0.0,
          f"max|diff| {float((lo_img - ramp).abs().max()):.4f}")
    check("W14 default floor: floor strength changes less than full strength",
          d_lo < d_hi, f"mean|diff| floor {d_lo:.5f} < full {d_hi:.5f}")
    check("W14 default floor: deficit floor everywhere (no exact-zero region)",
          float(lo_mask.min()) > 0.0, f"deficit min {float(lo_mask.min()):.4f}")


run_block("W14  mask_min intensity floor", w14)


# =============================================================================
# W11 -- solver isolation: the mask scales the OPTICS, never the solver (Sec
# 9d law). Observable proxy: deficit at mask=ones is bitwise == deficit at
# mask=None (same h, same probe, same warp -- Sec 1's ones-bitwise-neutral
# argument extended to the deficit output); the console's fold/depth report
# is identical whether or not a mask is connected (Sec 6: it describes the
# WATER, not the applied effect). NC: a deliberately WRONG solver-side h*m
# variant (fed into grain_deficit as if it were h_mm) must diverge from the
# real masked-optics deficit, proving the isolation is not vacuous.
# =============================================================================

def w11():
    node = NodeMod.WaterRefraction()
    Hs, Ws = 160, 200
    torch.manual_seed(SEED)
    batch = torch.rand(1, Hs, Ws, 3)
    cfg = node_cfg(depth_scale=1.0)

    _, mask_ones = node.execute(batch, mask=t_ones(Hs, Ws), **cfg)
    _, mask_none = node.execute(batch, mask=None, **cfg)
    check("W11 deficit at mask=ones bitwise == mask=None (same h, same warp)",
          torch.equal(mask_ones, mask_none))

    buf_masked = io.StringIO()
    with contextlib.redirect_stdout(buf_masked):
        node.execute(batch, mask=t_hard_half(Hs, Ws), **cfg)
    buf_unmasked = io.StringIO()
    with contextlib.redirect_stdout(buf_unmasked):
        node.execute(batch, mask=None, **cfg)

    def fold_depth_lines(s):
        return [ln for ln in s.splitlines()
                if ("fold" in ln.lower() or "depth" in ln.lower() or "dmax" in ln.lower())]

    lm, lu = fold_depth_lines(buf_masked.getvalue()), fold_depth_lines(buf_unmasked.getvalue())
    check("W11 console fold/depth report identical masked vs unmasked",
          lm == lu and len(lm) > 0, f"masked={lm}  unmasked={lu}")

    h_mm = band_surface(Hs, Ws, seed=SEED, pool=0.30, rough=0.10)
    shape = (Hs, Ws, 3)
    m = np_hard_half(Hs, Ws)
    deficit_correct = WR.grain_deficit(shape, h_mm, FIELD, aperture_ratio=APERTURE,
                                        samples=GPU_SAMPLES, seed=SEED, depth_scale=1.0, mask=m)
    h_bad = h_mm * m  # WRONG: a hypothetical solver-side h*m variant
    deficit_bad = WR.grain_deficit(shape, h_bad, FIELD, aperture_ratio=APERTURE,
                                    samples=GPU_SAMPLES, seed=SEED, depth_scale=1.0, mask=None)
    d = float(np.abs(deficit_correct.astype(np.float64) - deficit_bad.astype(np.float64)).mean())
    must_fail("W11 NC solver-side h*m variant matches the correct masked-optics deficit",
              d < 0.02, f"mean|diff| {d:.4f}")


run_block("W11  solver isolation", w11)


# =============================================================================
# W12 -- clamp + NaN. Values clamp to [0,1] (Sec 5); NaN pixels become 0
# (nan_to_num, "not painted", Sec 1c).
# =============================================================================

def w12():
    node = NodeMod.WaterRefraction()
    Hs, Ws = 160, 200
    torch.manual_seed(SEED)
    batch = torch.rand(1, Hs, Ws, 3)
    cfg = node_cfg()

    img_2, mask_2 = node.execute(batch, mask=t_const(Hs, Ws, 2.0), **cfg)
    img_1, mask_1 = node.execute(batch, mask=t_const(Hs, Ws, 1.0), **cfg)
    check("W12 mask=2.0 clamps bitwise to mask=1.0",
          torch.equal(img_2, img_1) and torch.equal(mask_2, mask_1))

    img_n, mask_n = node.execute(batch, mask=t_const(Hs, Ws, -0.5), **cfg)
    img_0, mask_0 = node.execute(batch, mask=t_const(Hs, Ws, 0.0), **cfg)
    check("W12 mask=-0.5 clamps bitwise to mask=0.0",
          torch.equal(img_n, img_0) and torch.equal(mask_n, mask_0))

    img_h, _ = node.execute(batch, mask=t_const(Hs, Ws, 0.5), **cfg)
    must_fail("W12 NC mask=0.5 matches mask=1.0", torch.equal(img_h, img_1))

    nan_mask, _ = t_nan_mask(Hs, Ws)
    img_nan, mask_nan = node.execute(batch, mask=nan_mask, **cfg)
    n_nan_img = int(torch.isnan(img_nan).sum())
    n_nan_mask = int(torch.isnan(mask_nan).sum())
    check("W12 NaN-pixel mask -> zero NaN in IMAGE output", n_nan_img == 0, f"{n_nan_img} NaN px")
    check("W12 NaN-pixel mask -> zero NaN in MASK/deficit output", n_nan_mask == 0, f"{n_nan_mask} NaN px")


run_block("W12  clamp + NaN", w12)


# =============================================================================
# W13 -- ride-along Sec 7.12a: grain_deficit gains depth_scale and forwards
# it to the probe warp. Live grain_deficit(depth_scale=0.25) must differ
# substantially from the frozen (no-depth_scale) form on a SHALLOW surface;
# at the default depth_scale=1.0 the fix is neutral, so live must stay
# bitwise identical to frozen there (the I9-preservation half of the row).
# =============================================================================

def w13():
    h_mm = band_surface(H, W, seed=SEED, pool=0.10, rough=0.03)  # shallow pool
    shape = (H, W, 3)
    orig_frozen_ra, orig_live_ra = FROZEN.render_auto, WR.render_auto
    try:
        FROZEN.render_auto = FROZEN.render
        WR.render_auto = WR.render

        dz_frozen = FROZEN.grain_deficit(shape, h_mm, FIELD, aperture_ratio=APERTURE,
                                          samples=NUMPY_SAMPLES, seed=SEED)
        dz_live_ds025 = WR.grain_deficit(shape, h_mm, FIELD, aperture_ratio=APERTURE,
                                          samples=NUMPY_SAMPLES, seed=SEED,
                                          depth_scale=0.25, mask=None)
        gap = float(np.abs(dz_frozen.astype(np.float64) - dz_live_ds025.astype(np.float64)).max())
        check("W13 live grain_deficit(depth_scale=0.25) differs from frozen (shallow surface)",
              gap > 0.2, f"max|diff| {gap:.3f} (spec measured up to 0.50 pointwise)")
        must_fail("W13 NC shipped no-depth_scale form matches the ds=0.25-forwarded live form",
                  gap < 0.02, f"max|diff| {gap:.3f}")

        dz_live_ds1 = WR.grain_deficit(shape, h_mm, FIELD, aperture_ratio=APERTURE,
                                        samples=NUMPY_SAMPLES, seed=SEED,
                                        depth_scale=1.0, mask=None)
        check("W13 live grain_deficit(depth_scale=1.0) bitwise == frozen (I9-preservation)",
              np.array_equal(dz_live_ds1, dz_frozen),
              f"max|diff| {float(np.abs(dz_live_ds1.astype(np.float64) - dz_frozen.astype(np.float64)).max()):.2e}")
    finally:
        FROZEN.render_auto, WR.render_auto = orig_frozen_ra, orig_live_ra


run_block("W13  ride-along Sec 7.12a (deficit depth_scale forwarding)", w13)


# =============================================================================
print("\n" + "=" * 78)
print(f"{len(PASSED)} passed, {len(FAILED)} failed  ({time.time() - t_start:.1f}s)")
if FAILED:
    print("FAILURES: " + ", ".join(FAILED))
print("=" * 78)
sys.exit(1 if FAILED else 0)
