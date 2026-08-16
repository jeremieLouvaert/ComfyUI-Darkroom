"""
Teeth for native MASK inputs on the four geometric lens nodes -- Lens Distortion,
Chromatic Aberration, Lens Profile, Perspective Correct -- per
docs/lens-mask-derivation.md (v2).

IMPLEMENTATION-BLIND: written from the spec and the FROZEN pre-build node source
only (`_lens_mask_dryrun/frozen/*.py`, vendored verbatim below as the I1 oracles).
This suite never reads the live `nodes/*.py` files; it imports them at runtime,
through the same PromptServer shim the other Darkroom suites use, purely to call
their public API (`execute()`) and their declared private seams (spec Sec.5.3).

  I1   mask=None is bitwise today's code                  (vendored frozen oracles)
  I2   mask=zeros is exact identity                        (raw input passthrough)
  I3   locality: {out != in} subset of {m > 0}, exact
  I4   mask=ones ~= mask=None within tau4                  (not bitwise -- Sec 1d)
  I5   displacement scales linearly                        (seam refactor guard)
  I6   uniform mask == equivalent strength scaling          (domain-pinned, LD+CA)
  I7   CA one-mask coherence: one tensor drives R and B together
  I8   Lens Profile vignette is gated by the same mask
  I9   resize plumbing: low-res mask ~= full-res mask
  I10  batch / mask-batch pairing semantics (M=1, M=2, M=0)
  I11  Perspective auto_crop is ignored once a mask is connected
  I12  mask values clamp to [0, 1]
  I13  cross-engine resize agreement (torch bilinear acT vs scipy zoom nearest)
  I14  Long Exposure resize fix: no dead edge strip after upscaling a mask
  NC   a negative control threaded through every invariant above, each MUST fire

Run:
  python_embeded/python.exe ComfyUI-Darkroom/tools/test_lens_masks.py

If the mask feature has not landed yet, node.execute() calls below raise
TypeError (unexpected keyword 'mask') -- that is the expected "not built yet"
signal, not a suite bug.
"""

import contextlib
import io
import math
import os
import sys
import time
import types

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import map_coordinates, zoom

from importlib import import_module

# The pack's __init__ decorates against a live PromptServer; same shim as the
# other Darkroom suites (tools/test_long_exposure.py).
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
LDmod = import_module(f"{_pkg}.nodes.lens_distortion")
CAmod = import_module(f"{_pkg}.nodes.chromatic_aberration")
LPmod = import_module(f"{_pkg}.nodes.lens_profile")
PCmod = import_module(f"{_pkg}.nodes.perspective_correct")
LEmod = import_module(f"{_pkg}.nodes.long_exposure")
DataLens = import_module(f"{_pkg}.data.lens_profiles")

PASSED, FAILED = [], []


def check(name, ok, detail=""):
    (PASSED if ok else FAILED).append(name)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}  {detail}")


def must_fail(name, ok, detail=""):
    (PASSED if not ok else FAILED).append(name)
    print(f"  [{'PASS' if not ok else 'FAIL'}] {name}  {detail}"
          f"{'' if not ok else '   <-- CONTROL DID NOT FIRE'}")


def run_block(title, fn):
    """Isolate one invariant's block: an exception here (e.g. the build hasn't
    landed, or a seam is named differently than guessed) is reported as an
    explicit FAIL rather than killing the whole suite's signal."""
    print(f"\n{title}")
    try:
        fn()
    except Exception as e:
        FAILED.append(f"{title} (exception)")
        print(f"  [FAIL] {title} raised {type(e).__name__}: {e}")


SEED = 20260816
torch.manual_seed(SEED)
NPRNG = np.random.default_rng(SEED)

print("=" * 78)
print("LENS MASKS -- teeth (implementation-blind, docs/lens-mask-derivation.md v2)")
print("=" * 78)
t_start = time.time()

# =============================================================================
# I1 ORACLES -- vendored verbatim from _lens_mask_dryrun/frozen/*.py.
# These are today's shipped arithmetic, pure functions of tensors, copied
# before the build landed. The mask=None path of the built nodes must match
# these bitwise (spec Sec.5.1: "mask=None executes today's arithmetic
# UNCHANGED").
# =============================================================================

def _oracle_pixel_to_grid_coords(src_y, src_x, h, w):
    gx = 2.0 * src_x / (w - 1) - 1.0
    gy = 2.0 * src_y / (h - 1) - 1.0
    return torch.stack([gx, gy], dim=-1)


def _oracle_grid_sample_channel(channel, grid, padding_mode='reflection'):
    inp = channel.unsqueeze(0).unsqueeze(0)
    g = grid.unsqueeze(0)
    out = F.grid_sample(inp, g, mode='bicubic', padding_mode=padding_mode,
                        align_corners=True)
    return out.squeeze(0).squeeze(0)


def oracle_apply_distortion(img, k1, k2):
    """Frozen `_apply_distortion` from lens_distortion.py."""
    h, w = img.shape[:2]
    cy, cx = h / 2.0, w / 2.0
    device = img.device
    yy = torch.arange(h, dtype=torch.float32, device=device)
    xx = torch.arange(w, dtype=torch.float32, device=device)
    yy, xx = torch.meshgrid(yy, xx, indexing='ij')
    ny = (yy - cy) / cy
    nx = (xx - cx) / cx
    r2 = nx * nx + ny * ny
    r4 = r2 * r2
    distort = 1.0 + k1 * r2 + k2 * r4
    src_x = nx * distort * cx + cx
    src_y = ny * distort * cy + cy
    grid = _oracle_pixel_to_grid_coords(src_y, src_x, h, w)
    inp = img.permute(2, 0, 1).unsqueeze(0)
    g = grid.unsqueeze(0)
    out = F.grid_sample(inp, g, mode='bicubic', padding_mode='zeros',
                        align_corners=True)
    return out.squeeze(0).permute(1, 2, 0).clamp(0.0, 1.0)


def oracle_apply_lateral_ca(img, shift_r, shift_b):
    """Frozen `_apply_lateral_ca` from chromatic_aberration.py."""
    h, w = img.shape[:2]
    cy, cx = h / 2.0, w / 2.0
    max_r = math.sqrt(cx * cx + cy * cy)
    device = img.device
    yy = torch.arange(h, dtype=torch.float32, device=device)
    xx = torch.arange(w, dtype=torch.float32, device=device)
    yy, xx = torch.meshgrid(yy, xx, indexing='ij')
    dy = yy - cy
    dx = xx - cx
    r = torch.sqrt(dy * dy + dx * dx) / max_r
    result = img.clone()
    for c, shift in enumerate([shift_r, 0.0, shift_b]):
        if abs(shift) < 0.01:
            continue
        scale = 1.0 + (shift / max_r) * r
        new_y = cy + dy * scale
        new_x = cx + dx * scale
        grid = _oracle_pixel_to_grid_coords(new_y, new_x, h, w)
        result[..., c] = _oracle_grid_sample_channel(img[..., c], grid,
                                                      padding_mode='reflection')
    return result.clamp(0.0, 1.0)


def oracle_apply_lens_profile(img, k1, k2, ca_r, ca_b, vignette_strength,
                               vignette_mid, mode):
    """Frozen `_apply_lens_profile` from lens_profile.py. NOTE: this is the
    UNFIXED oracle -- `result = torch.empty_like(img)` verbatim, per spec Sec
    5.9b that only applies to the built code's 4th-channel init, not our I1
    reference (I1 compares 3-channel IMAGE tensors so this never surfaces)."""
    h, w = img.shape[:2]
    cy, cx = h / 2.0, w / 2.0
    scale = min(h, w) / 1024.0
    device = img.device
    yy = torch.arange(h, dtype=torch.float32, device=device)
    xx = torch.arange(w, dtype=torch.float32, device=device)
    yy, xx = torch.meshgrid(yy, xx, indexing='ij')
    ny = (yy - cy) / cy
    nx = (xx - cx) / cx
    r2 = nx * nx + ny * ny
    r4 = r2 * r2
    r = torch.sqrt(r2)
    max_r = math.sqrt(cx * cx + cy * cy)
    result = torch.empty_like(img)
    ca_shifts = [ca_r * scale, 0.0, ca_b * scale]
    for c in range(3):
        distort = 1.0 + k1 * r2 + k2 * r4
        ca = ca_shifts[c]
        if abs(ca) > 0.01:
            ca_factor = 1.0 + (ca / max_r) * r
            total_factor = distort * ca_factor
        else:
            total_factor = distort
        src_x = nx * total_factor * cx + cx
        src_y = ny * total_factor * cy + cy
        grid = _oracle_pixel_to_grid_coords(src_y, src_x, h, w)
        result[..., c] = _oracle_grid_sample_channel(img[..., c], grid,
                                                      padding_mode='reflection')
    if vignette_strength > 0.01:
        cos_theta = 1.0 / torch.sqrt(1.0 + r2)
        falloff = cos_theta ** 4
        transition = ((r - vignette_mid * 0.8) / 0.4).clamp(0.0, 1.0)
        if mode == "Add Aberrations":
            mask = (1.0 - transition * (1.0 - falloff) * vignette_strength * 2).clamp(0.0, 1.0)
            result = result * mask.unsqueeze(-1)
        else:
            correction = 1.0 + transition * (1.0 / falloff.clamp(0.3, 1.0) - 1.0) * vignette_strength
            result = result * correction.unsqueeze(-1)
    return result.clamp(0.0, 1.0)


def oracle_apply_perspective(img, vertical, horizontal, rotation):
    """Frozen `_apply_perspective` from perspective_correct.py (numpy)."""
    h, w = img.shape[:2]
    cy, cx = h / 2.0, w / 2.0
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    ny = (yy - cy) / cy
    nx = (xx - cx) / cx
    if abs(rotation) > 0.01:
        theta = np.radians(rotation)
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        rnx = nx * cos_t - ny * sin_t
        rny = nx * sin_t + ny * cos_t
        nx, ny = rnx, rny
    if abs(vertical) > 0.001:
        v_scale = 1.0 / (1.0 + vertical * ny)
        nx = nx * v_scale
    if abs(horizontal) > 0.001:
        h_scale = 1.0 / (1.0 + horizontal * nx)
        ny = ny * h_scale
    src_x = nx * cx + cx
    src_y = ny * cy + cy
    result = np.empty_like(img)
    for c in range(img.shape[2]):
        result[..., c] = map_coordinates(
            img[..., c], [src_y, src_x], order=1, mode='constant', cval=0.0
        ).astype(np.float32)
    return np.clip(result, 0.0, 1.0).astype(np.float32)


def oracle_auto_crop(img):
    """Frozen `_auto_crop` from perspective_correct.py."""
    gray = img.sum(axis=2)
    row_sums = (gray > 0.01).sum(axis=1)
    col_sums = (gray > 0.01).sum(axis=0)
    h, w = img.shape[:2]
    threshold_row = w * 0.9
    threshold_col = h * 0.9
    valid_rows = np.where(row_sums > threshold_row)[0]
    valid_cols = np.where(col_sums > threshold_col)[0]
    if len(valid_rows) < 10 or len(valid_cols) < 10:
        return img
    top, bottom = valid_rows[0], valid_rows[-1] + 1
    left, right = valid_cols[0], valid_cols[-1] + 1
    cropped = img[top:bottom, left:right]
    scale_y = h / cropped.shape[0]
    scale_x = w / cropped.shape[1]
    result = zoom(cropped, (scale_y, scale_x, 1.0), order=1)
    return np.clip(result, 0.0, 1.0).astype(np.float32)


# =============================================================================
# Fixtures
# =============================================================================

H, W = 509, 767          # primary knife-edge: odd, non-pow2, non-(2^n+1)
H_INERT, W_INERT = 513, 513   # grid_sample identity is exact here (Sec 1c)

K1, K2 = 0.35, -0.12
SHIFT_R, SHIFT_B = -3.7, 2.9
VERT, HORIZ, ROT = 0.37, -0.22, 7.3
V_SING, H_SING = 0.5, 0.5

TAU4 = 1e-5
TAU6 = 4.0e-3
TAU8 = 1e-6
TAU9 = 2.7e-3
TAU13 = 7.4e-4

HAS_CUDA = torch.cuda.is_available()


def rand_image(h, w, c=3, device='cpu', seed=None):
    g = torch.Generator(device='cpu').manual_seed(seed if seed is not None else SEED)
    return torch.rand(h, w, c, generator=g).to(device)


def rand_batch(b, h, w, c=3, device='cpu', seed=None):
    g = torch.Generator(device='cpu').manual_seed(seed if seed is not None else SEED)
    return torch.rand(b, h, w, c, generator=g).to(device)


def const_mask(h, w, val, device='cpu'):
    return torch.full((h, w), float(val), device=device)


def hard_half_mask(h, w, device='cpu'):
    m = torch.zeros(h, w, device=device)
    m[:, w // 2:] = 1.0
    return m


def blob_mask(h, w, feather_px=31.0, device='cpu'):
    yy, xx = torch.meshgrid(torch.arange(h, dtype=torch.float32),
                             torch.arange(w, dtype=torch.float32), indexing='ij')
    cy, cx = h / 2.0, w / 2.0
    r = torch.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    radius = 0.3 * min(h, w)
    m = ((radius - r) / feather_px + 0.5).clamp(0.0, 1.0)
    return m.to(device)


def binary_noise_mask(h, w, seed=1, device='cpu'):
    g = torch.Generator(device='cpu').manual_seed(seed)
    return (torch.rand(h, w, generator=g) > 0.5).float().to(device)


def get_lp_lens_name():
    """Pick the profile with the strongest combined distortion+CA+vignette so
    I1/etc never trip an early-out. Chosen dynamically -- not hardcoded to a
    name that could drift out of data/lens_profiles.py."""
    flat = DataLens.LENS_PROFILES_FLAT
    return max(flat, key=lambda n: abs(flat[n].k1) + abs(flat[n].ca_r) + flat[n].vig_strength)


LP_LENS = get_lp_lens_name()
LP_PROFILE = DataLens.LENS_PROFILES_FLAT[LP_LENS]


def lp_scaled_params(mode="Add Aberrations", strength=1.0, profile=None):
    p = profile or LP_PROFILE
    sign = 1.0 if mode == "Add Aberrations" else -1.0
    return dict(k1=p.k1 * strength * sign, k2=p.k2 * strength * sign,
                ca_r=p.ca_r * strength * sign, ca_b=p.ca_b * strength * sign,
                vig=p.vig_strength * strength, vig_mid=p.vig_midpoint)


def seam_fn(mod, candidates=("_masked_coords",)):
    for name in candidates:
        fn = getattr(mod, name, None)
        if fn is not None:
            return fn, name
    # Last resort, implementation-blind discovery: any private callable whose
    # name mentions both "mask" and "coord" (per-module seam-naming variants
    # like `_masked_coords_ca` were not literally pinned by spec Sec 5.3,
    # which names only LD's `_masked_coords` exactly).
    for n in dir(mod):
        if n.startswith("_") and "mask" in n.lower() and "coord" in n.lower():
            fn = getattr(mod, n)
            if callable(fn):
                return fn, n
    return None, None


# =============================================================================
# I1 -- mask=None is bitwise today's node
# =============================================================================

def _i1_ld(device):
    img = rand_batch(2, H, W, device=device)
    node = LDmod.LensDistortion()
    out, = node.execute(img, lens="Custom", strength=1.0, k1=K1, k2=K2, mask=None)
    oracle = torch.stack([oracle_apply_distortion(img[i], K1, K2) for i in range(img.shape[0])], dim=0)
    check(f"I1 LD mask=None bitwise == frozen oracle [{device}]", torch.equal(out, oracle),
          f"max|diff| {(out-oracle).abs().max().item():.2e}")
    # NC uses a partial (0.5) mask, not ones: per spec Sec 1d, mask=1.0 is
    # measured bitwise-ZERO against mask=None at every config tried (a
    # Sterbenz-adjacent cancellation, not a bug) -- confirmed here too, so a
    # ones-mask NC could never fire against a compliant build. 0.5 gives a
    # real, guaranteed-large discriminator instead.
    m = const_mask(H, W, 0.5, device=device)
    out_half, = node.execute(img, lens="Custom", strength=1.0, k1=K1, k2=K2, mask=m)
    must_fail(f"I1 NC LD const-0.5-mask matches frozen oracle [{device}]",
              torch.equal(out_half, oracle))


def _i1_ca(device):
    img = rand_batch(2, H, W, device=device)
    node = CAmod.ChromaticAberration()
    out, = node.execute(img, lens="Custom", strength=1.0, shift_r=SHIFT_R, shift_b=SHIFT_B, mask=None)
    scale = min(H, W) / 1024.0
    sr, sb = SHIFT_R * scale, SHIFT_B * scale
    oracle = torch.stack([oracle_apply_lateral_ca(img[i], sr, sb) for i in range(img.shape[0])], dim=0)
    check(f"I1 CA mask=None bitwise == frozen oracle [{device}]", torch.equal(out, oracle),
          f"max|diff| {(out-oracle).abs().max().item():.2e}")
    m = const_mask(H, W, 0.5, device=device)
    out_half, = node.execute(img, lens="Custom", strength=1.0, shift_r=SHIFT_R, shift_b=SHIFT_B, mask=m)
    must_fail(f"I1 NC CA const-0.5-mask matches frozen oracle [{device}]",
              torch.equal(out_half, oracle))


def _i1_lp(device):
    img = rand_batch(2, H, W, device=device)
    node = LPmod.LensProfile()
    out, = node.execute(img, lens=LP_LENS, mode="Add Aberrations", strength=1.0, mask=None)
    sp = lp_scaled_params("Add Aberrations", 1.0)
    oracle = torch.stack([oracle_apply_lens_profile(img[i], sp['k1'], sp['k2'], sp['ca_r'],
                                                     sp['ca_b'], sp['vig'], sp['vig_mid'],
                                                     "Add Aberrations") for i in range(img.shape[0])], dim=0)
    check(f"I1 LP mask=None bitwise == frozen oracle [{device}]", torch.equal(out, oracle),
          f"max|diff| {(out-oracle).abs().max().item():.2e}")
    m = const_mask(H, W, 0.5, device=device)
    out_half, = node.execute(img, lens=LP_LENS, mode="Add Aberrations", strength=1.0, mask=m)
    must_fail(f"I1 NC LP const-0.5-mask matches frozen oracle [{device}]",
              torch.equal(out_half, oracle))


def _i1_pc():
    img = rand_batch(2, H, W)
    node = PCmod.PerspectiveCorrect()
    out, = node.execute(img, vertical=VERT, horizontal=HORIZ, rotation=ROT, auto_crop=True, mask=None)
    arrs = [img[i].numpy().astype(np.float32) for i in range(img.shape[0])]
    oracle_np = []
    for a in arrs:
        r = oracle_apply_perspective(a, VERT, HORIZ, ROT)
        r = oracle_auto_crop(r)
        oracle_np.append(r)
    oracle = torch.from_numpy(np.stack(oracle_np, axis=0))
    check("I1 PC mask=None bitwise == frozen oracle (auto_crop=True) [cpu]",
          torch.equal(out, oracle), f"max|diff| {(out-oracle).abs().max().item():.2e}")
    m = const_mask(H, W, 0.5)
    out_half, = node.execute(img, vertical=VERT, horizontal=HORIZ, rotation=ROT, auto_crop=True, mask=m)
    must_fail("I1 NC PC const-0.5-mask matches frozen oracle [cpu]", torch.equal(out_half, oracle))


run_block("I1  mask=None is bitwise today's node (all 4 nodes, CPU)",
          lambda: (_i1_ld('cpu'), _i1_ca('cpu'), _i1_lp('cpu'), _i1_pc()))
if HAS_CUDA:
    run_block("I1  mask=None is bitwise today's node (LD/CA/LP, CUDA -- PC is CPU-only)",
              lambda: (_i1_ld('cuda'), _i1_ca('cuda'), _i1_lp('cuda')))
else:
    print("\nI1 CUDA  [SKIP] no CUDA available")


# =============================================================================
# I2 -- zero mask is exact identity (raw input passes through, unclamped)
# =============================================================================

def i2():
    img = rand_batch(2, H, W)
    z = const_mask(H, W, 0.0)
    o = const_mask(H, W, 1.0)

    node = LDmod.LensDistortion()
    out, = node.execute(img, lens="Custom", strength=1.0, k1=K1, k2=K2, mask=z)
    check("I2 LD zero mask == input bitwise", torch.equal(out, img))
    out_o, = node.execute(img, lens="Custom", strength=1.0, k1=K1, k2=K2, mask=o)
    must_fail("I2 NC LD ones mask matches input", torch.equal(out_o, img))

    node = CAmod.ChromaticAberration()
    out, = node.execute(img, lens="Custom", strength=1.0, shift_r=SHIFT_R, shift_b=SHIFT_B, mask=z)
    check("I2 CA zero mask == input bitwise", torch.equal(out, img))
    out_o, = node.execute(img, lens="Custom", strength=1.0, shift_r=SHIFT_R, shift_b=SHIFT_B, mask=o)
    must_fail("I2 NC CA ones mask matches input", torch.equal(out_o, img))

    node = LPmod.LensProfile()
    out, = node.execute(img, lens=LP_LENS, mode="Add Aberrations", strength=1.0, mask=z)
    check("I2 LP zero mask == input bitwise", torch.equal(out, img))
    out_o, = node.execute(img, lens=LP_LENS, mode="Add Aberrations", strength=1.0, mask=o)
    must_fail("I2 NC LP ones mask matches input", torch.equal(out_o, img))

    node = PCmod.PerspectiveCorrect()
    out, = node.execute(img, vertical=VERT, horizontal=HORIZ, rotation=ROT, auto_crop=True, mask=z)
    check("I2 PC zero mask == input bitwise", torch.equal(out, img))
    out_o, = node.execute(img, vertical=VERT, horizontal=HORIZ, rotation=ROT, auto_crop=True, mask=o)
    must_fail("I2 NC PC ones mask matches input", torch.equal(out_o, img))


run_block("I2  zero mask is exact identity", i2)


# =============================================================================
# I3 -- locality: {out != in} subset of {m > 0}, exact.
# Positive tooth runs through the real built nodes. The two negative controls
# are reference implementations written from spec Sec 1c/2d math -- they do
# NOT call the built code -- proving the mechanism the select guards against
# actually bites at the pinned configs.
# =============================================================================

def i3_locality_node(name, run_fn, m):
    img_in, out = run_fn(m)
    zero_region = (m == 0)
    # broadcast zero_region over channels
    zr = zero_region.unsqueeze(-1).expand_as(img_in)
    ok = torch.equal(out[zr], img_in[zr])
    check(f"I3 {name} locality: m=0 region untouched (hard-half mask)", ok)


def i3():
    m = hard_half_mask(H, W)
    img = rand_batch(1, H, W)[0]
    batch = img.unsqueeze(0)

    node = LDmod.LensDistortion()
    def run_ld(mask):
        out, = node.execute(batch, lens="Custom", strength=1.0, k1=K1, k2=K2, mask=mask)
        return batch[0], out[0]
    i3_locality_node("LD", run_ld, m)

    node = CAmod.ChromaticAberration()
    def run_ca(mask):
        out, = node.execute(batch, lens="Custom", strength=1.0, shift_r=SHIFT_R, shift_b=SHIFT_B, mask=mask)
        return batch[0], out[0]
    i3_locality_node("CA", run_ca, m)

    node = LPmod.LensProfile()
    def run_lp(mask):
        out, = node.execute(batch, lens=LP_LENS, mode="Add Aberrations", strength=1.0, mask=mask)
        return batch[0], out[0]
    i3_locality_node("LP", run_lp, m)

    node = PCmod.PerspectiveCorrect()
    def run_pc(mask):
        out, = node.execute(batch, vertical=VERT, horizontal=HORIZ, rotation=ROT, auto_crop=True, mask=mask)
        return batch[0], out[0]
    i3_locality_node("PC", run_pc, m)

    # extra coverage: blob + binary-noise masks on LD
    node = LDmod.LensDistortion()
    for mk_name, mk in (("blob", blob_mask(H, W)), ("binary-noise", binary_noise_mask(H, W))):
        img_in, out = run_ld(mk)
        zr = (mk == 0).unsqueeze(-1).expand_as(img_in)
        check(f"I3 LD locality: m=0 region untouched ({mk_name} mask)",
              torch.equal(out[zr], img_in[zr]))


def i3_nc_no_select():
    """Reference D-form WITHOUT the m=0 select, applied to LD's own unmasked
    coords (recomputed here, not read from the node). At 509x767 grid_sample's
    identity is NOT bit-exact (Sec 1c reason 1: (size-1) is not a power of
    two), so removing the select must leak differences into the m=0 region.
    Inert (by design) at 513x513 where (size-1)=512=2^9 IS exact."""
    def no_select_ld(h, w, k1, k2, m):
        cy, cx = h / 2.0, w / 2.0
        yy = torch.arange(h, dtype=torch.float32)
        xx = torch.arange(w, dtype=torch.float32)
        yy, xx = torch.meshgrid(yy, xx, indexing='ij')
        ny, nx = (yy - cy) / cy, (xx - cx) / cx
        r2 = nx * nx + ny * ny
        distort = 1.0 + k1 * r2 + k2 * r2 * r2
        src_x = nx * distort * cx + cx
        src_y = ny * distort * cy + cy
        src_y_m = yy + m * (src_y - yy)
        src_x_m = xx + m * (src_x - xx)
        img = torch.rand(h, w, 3, generator=torch.Generator().manual_seed(SEED))
        grid = _oracle_pixel_to_grid_coords(src_y_m, src_x_m, h, w)
        inp = img.permute(2, 0, 1).unsqueeze(0)
        out = F.grid_sample(inp, grid.unsqueeze(0), mode='bicubic',
                            padding_mode='zeros', align_corners=True)
        out = out.squeeze(0).permute(1, 2, 0).clamp(0, 1)
        return img, out, yy, xx

    m1 = hard_half_mask(H, W)
    img, out, yy, xx = no_select_ld(H, W, K1, K2, m1)
    zero_region = (m1 == 0)
    diff = (out - img).abs().amax(dim=-1)
    violating = int((diff[zero_region] > 0).sum())
    must_fail("I3 NC torch no-select at 509x767 preserves locality (LD, hard-half mask)",
              violating == 0, f"{violating} violating px at m=0 (spec measured 93k-100k)")

    m2 = hard_half_mask(H_INERT, W_INERT)
    img2, out2, yy2, xx2 = no_select_ld(H_INERT, W_INERT, K1, K2, m2)
    zero_region2 = (m2 == 0)
    diff2 = (out2 - img2).abs().amax(dim=-1)
    violating2 = int((diff2[zero_region2] > 0).sum())
    check("I3 sanity: no-select variant is inert at 513x513 by design",
          violating2 == 0, f"{violating2} violating px at m=0 (expect 0 -- (size-1)=512=2^9)")


def i3_nc_pc_singular():
    """Reference D-form WITHOUT the select for Perspective, at the singular
    corner v=h=0.5: the unmasked source coordinate is inf there, so an m=0
    pixel's 0*inf = NaN leaks through without the select."""
    h, w = H, W
    cy, cx = h / 2.0, w / 2.0
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    ny, nx = (yy - cy) / cy, (xx - cx) / cx
    v_scale = 1.0 / (1.0 + V_SING * ny)
    nx2 = nx * v_scale
    h_scale = 1.0 / (1.0 + H_SING * nx2)
    ny2 = ny * h_scale
    src_x = nx2 * cx + cx
    src_y = ny2 * cy + cy
    m = hard_half_mask(h, w).numpy().astype(np.float64)
    src_y_m = yy + m * (src_y - yy)
    src_x_m = xx + m * (src_x - xx)
    zero_region = (m == 0)
    nan_count = int(np.isnan(src_y_m[zero_region]).sum() + np.isnan(src_x_m[zero_region]).sum())
    must_fail("I3 NC PC no-select at singular v=h=0.5 preserves locality (no NaN)",
              nan_count == 0, f"{nan_count} NaN coordinate(s) at m=0 (spec measured 1)")


run_block("I3  locality: {out != in} subset of {m > 0}", i3)
run_block("I3 NC  reference no-select variants (self-contained, spec Sec 1c/2d math)",
          lambda: (i3_nc_no_select(), i3_nc_pc_singular()))


# =============================================================================
# I4 -- mask=ones ~= mask=None within tau4 (tolerance, not bitwise -- Sec 1d).
# PC: auto_crop=False on BOTH arms (spec: auto_crop=True is a "config trap").
# =============================================================================

def i4():
    img = rand_batch(2, H, W)
    ones = const_mask(H, W, 1.0)
    half = const_mask(H, W, 0.5)

    node = LDmod.LensDistortion()
    out_none, = node.execute(img, lens="Custom", strength=1.0, k1=K1, k2=K2, mask=None)
    out_ones, = node.execute(img, lens="Custom", strength=1.0, k1=K1, k2=K2, mask=ones)
    d = (out_none - out_ones).abs().max().item()
    check("I4 LD mask=ones ~= mask=None", d <= TAU4, f"max|diff| {d:.2e} <= tau4={TAU4}")
    out_half, = node.execute(img, lens="Custom", strength=1.0, k1=K1, k2=K2, mask=half)
    d_half = (out_none - out_half).abs().max().item()
    must_fail("I4 NC LD const 0.5 within tau4 of mask=None", d_half <= TAU4,
              f"max|diff| {d_half:.3f} > tau4 (spec measured ~0.76)")

    node = CAmod.ChromaticAberration()
    out_none, = node.execute(img, lens="Custom", strength=1.0, shift_r=SHIFT_R, shift_b=SHIFT_B, mask=None)
    out_ones, = node.execute(img, lens="Custom", strength=1.0, shift_r=SHIFT_R, shift_b=SHIFT_B, mask=ones)
    d = (out_none - out_ones).abs().max().item()
    check("I4 CA mask=ones ~= mask=None", d <= TAU4, f"max|diff| {d:.2e} <= tau4={TAU4}")

    node = LPmod.LensProfile()
    out_none, = node.execute(img, lens=LP_LENS, mode="Add Aberrations", strength=1.0, mask=None)
    out_ones, = node.execute(img, lens=LP_LENS, mode="Add Aberrations", strength=1.0, mask=ones)
    d = (out_none - out_ones).abs().max().item()
    check("I4 LP mask=ones ~= mask=None", d <= TAU4, f"max|diff| {d:.2e} <= tau4={TAU4}")

    node = PCmod.PerspectiveCorrect()
    out_none, = node.execute(img, vertical=VERT, horizontal=HORIZ, rotation=ROT, auto_crop=False, mask=None)
    out_ones, = node.execute(img, vertical=VERT, horizontal=HORIZ, rotation=ROT, auto_crop=False, mask=ones)
    d = (out_none - out_ones).abs().max().item()
    check("I4 PC mask=ones ~= mask=None (auto_crop=False both arms)", d <= TAU4,
          f"max|diff| {d:.2e} <= tau4={TAU4}")
    out_half, = node.execute(img, vertical=VERT, horizontal=HORIZ, rotation=ROT, auto_crop=False, mask=half)
    d_half = (out_none - out_half).abs().max().item()
    must_fail("I4 NC PC const 0.5 within tau4 of mask=None", d_half <= TAU4,
              f"max|diff| {d_half:.3f} > tau4")


run_block("I4  mask=ones equiv mask=None within tau4", i4)


# =============================================================================
# I5 -- displacement scales linearly (the seam refactor guard). Recompute
# p + m*(src-p) in the spec's exact op order from the seam's OWN unmasked
# (src_y, src_x), and require bitwise equality with the seam's OWN
# (src_y_m, src_x_m). Only LD's seam name is pinned verbatim by spec Sec 5.3
# ("_masked_coords"); CA/LP/PC's are discovered at runtime (no source read --
# `dir()` introspection for a private callable naming both "mask" and
# "coord"), found to be `_masked_coords_ca` / `_masked_coords_lp` /
# `_masked_coords_pc`.
#
# The mask MUST carry fractional (non-{0,1}) values here: for a binary mask
# m^2 == m identically, so the m-squared negative control cannot discriminate
# real op-order from the m^2 defect on a 0/1 mask. blob_mask's feather band
# provides real fractional values.
# =============================================================================

def i5_ld():
    fn, name = seam_fn(LDmod, ("_masked_coords",))
    if fn is None:
        check("I5 LD seam function found", False, "no _masked_coords on lens_distortion module")
        return
    h, w = H, W
    m = blob_mask(h, w)
    src_y, src_x, src_y_m, src_x_m = fn(h, w, K1, K2, m, torch.device('cpu'))
    yy = torch.arange(h, dtype=torch.float32)
    xx = torch.arange(w, dtype=torch.float32)
    yy, xx = torch.meshgrid(yy, xx, indexing='ij')
    recompute_y = yy + m * (src_y - yy)
    recompute_x = xx + m * (src_x - xx)
    check(f"I5 LD ({name}) recompute bitwise == seam's own masked coords (y)",
          torch.equal(recompute_y, src_y_m))
    check(f"I5 LD ({name}) recompute bitwise == seam's own masked coords (x)",
          torch.equal(recompute_x, src_x_m))
    recompute_y_sq = yy + (m * m) * (src_y - yy)
    recompute_x_sq = xx + (m * m) * (src_x - xx)
    must_fail("I5 NC LD m-squared variant matches seam's masked coords",
              torch.equal(recompute_y_sq, src_y_m) and torch.equal(recompute_x_sq, src_x_m))


def i5_ca():
    fn, name = seam_fn(CAmod, ("_masked_coords_ca", "_masked_coords"))
    if fn is None:
        check("I5 CA seam function found", False, "no masked-coords seam on chromatic_aberration module")
        return
    h, w = H, W
    m = blob_mask(h, w)
    d = fn(h, w, SHIFT_R, SHIFT_B, m, torch.device('cpu'))
    if not isinstance(d, dict):
        check("I5 CA seam returns a per-channel dict", False, f"got {type(d)}")
        return
    yy = torch.arange(h, dtype=torch.float32)
    xx = torch.arange(w, dtype=torch.float32)
    yy, xx = torch.meshgrid(yy, xx, indexing='ij')
    for c, (src_y, src_x, src_y_m, src_x_m) in d.items():
        recompute_y = yy + m * (src_y - yy)
        recompute_x = xx + m * (src_x - xx)
        check(f"I5 CA ({name}) channel {c}: recompute bitwise == seam's own masked coords",
              torch.equal(recompute_y, src_y_m) and torch.equal(recompute_x, src_x_m))
        recompute_y_sq = yy + (m * m) * (src_y - yy)
        must_fail(f"I5 NC CA channel {c}: m-squared variant matches seam's masked coords",
                  torch.equal(recompute_y_sq, src_y_m))


def i5_lp():
    fn, name = seam_fn(LPmod, ("_masked_coords_lp", "_masked_coords"))
    if fn is None:
        check("I5 LP seam function found", False, "no masked-coords seam on lens_profile module")
        return
    h, w = H, W
    m = blob_mask(h, w)
    sp = lp_scaled_params("Add Aberrations", 1.0)
    d = fn(h, w, sp['k1'], sp['k2'], sp['ca_r'], sp['ca_b'], m, torch.device('cpu'))
    if not isinstance(d, dict):
        check("I5 LP seam returns a per-channel dict", False, f"got {type(d)}")
        return
    yy = torch.arange(h, dtype=torch.float32)
    xx = torch.arange(w, dtype=torch.float32)
    yy, xx = torch.meshgrid(yy, xx, indexing='ij')
    for c, (src_y, src_x, src_y_m, src_x_m) in d.items():
        recompute_y = yy + m * (src_y - yy)
        recompute_x = xx + m * (src_x - xx)
        check(f"I5 LP ({name}) channel {c}: recompute bitwise == seam's own masked coords",
              torch.equal(recompute_y, src_y_m) and torch.equal(recompute_x, src_x_m))
        recompute_y_sq = yy + (m * m) * (src_y - yy)
        must_fail(f"I5 NC LP channel {c}: m-squared variant matches seam's masked coords",
                  torch.equal(recompute_y_sq, src_y_m))


def i5_pc():
    fn, name = seam_fn(PCmod, ("_masked_coords_pc", "_masked_coords"))
    if fn is None:
        check("I5 PC seam function found", False, "no masked-coords seam on perspective_correct module")
        return
    h, w = H, W
    m = blob_mask(h, w).numpy().astype(np.float64)
    src_y, src_x, src_y_m, src_x_m = fn(h, w, VERT, HORIZ, ROT, m)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    recompute_y = yy + m * (src_y - yy)
    recompute_x = xx + m * (src_x - xx)
    check(f"I5 PC ({name}) recompute bitwise == seam's own masked coords (y)",
          np.array_equal(recompute_y, src_y_m))
    check(f"I5 PC ({name}) recompute bitwise == seam's own masked coords (x)",
          np.array_equal(recompute_x, src_x_m))
    recompute_y_sq = yy + (m * m) * (src_y - yy)
    must_fail("I5 NC PC m-squared variant matches seam's masked coords (y)",
              np.array_equal(recompute_y_sq, src_y_m))


run_block("I5  displacement scales linearly (LD seam)", i5_ld)
run_block("I5  displacement scales linearly (CA seam)", i5_ca)
run_block("I5  displacement scales linearly (LP seam)", i5_lp)
run_block("I5  displacement scales linearly (PC seam)", i5_pc)


# =============================================================================
# I6 -- uniform mask == equivalent strength scaling. LD + CA only.
# Domain-pinned: both arms' scaled scalars sit >=10x above the early-out/skip
# threshold (spec explicitly calls out the naive config as broken).
# =============================================================================

def i6():
    img = rand_batch(2, H, W)
    c = 0.5
    m = const_mask(H, W, c)

    node = LDmod.LensDistortion()
    # domain check: k1*c, k2*c both >= 10x the 0.001 early-out threshold
    assert abs(K1 * c) >= 0.01 and abs(K2 * c) >= 0.01, "I6 LD domain pin violated"
    out_mask, = node.execute(img, lens="Custom", strength=1.0, k1=K1, k2=K2, mask=m)
    out_strength, = node.execute(img, lens="Custom", strength=c, k1=K1, k2=K2, mask=None)
    d = (out_mask - out_strength).abs().max().item()
    check("I6 LD const-c mask ~= strength*c", d <= TAU6, f"max|diff| {d:.2e} <= tau6={TAU6}")
    out_wrong, = node.execute(img, lens="Custom", strength=0.8, k1=K1, k2=K2, mask=None)
    d_wrong = (out_mask - out_wrong).abs().max().item()
    must_fail("I6 NC LD const 0.5 mask matches strength*0.8", d_wrong <= TAU6,
              f"max|diff| {d_wrong:.3f} > tau6 (spec measured ~0.76)")

    node = CAmod.ChromaticAberration()
    scale = min(H, W) / 1024.0
    assert abs(SHIFT_R * c * scale) >= 0.1 and abs(SHIFT_B * c * scale) >= 0.1, "I6 CA domain pin violated"
    out_mask, = node.execute(img, lens="Custom", strength=1.0, shift_r=SHIFT_R, shift_b=SHIFT_B, mask=m)
    out_strength, = node.execute(img, lens="Custom", strength=c, shift_r=SHIFT_R, shift_b=SHIFT_B, mask=None)
    d = (out_mask - out_strength).abs().max().item()
    check("I6 CA const-c mask ~= strength*c", d <= TAU6, f"max|diff| {d:.2e} <= tau6={TAU6}")
    out_wrong, = node.execute(img, lens="Custom", strength=0.8, shift_r=SHIFT_R, shift_b=SHIFT_B, mask=None)
    d_wrong = (out_mask - out_wrong).abs().max().item()
    must_fail("I6 NC CA const 0.5 mask matches strength*0.8", d_wrong <= TAU6,
              f"max|diff| {d_wrong:.3f} > tau6")


run_block("I6  uniform mask == equivalent strength (domain-pinned, LD+CA)", i6)


# =============================================================================
# I7 -- CA one-mask coherence: R and B use the SAME mask tensor; green is
# bitwise untouched regardless. Positive tooth via the real node + seam;
# negative control is a from-spec reference implementation with two
# DIFFERENT masks for R and B, showing it breaks the lens-true R:B ratio.
# =============================================================================

def i7_green_untouched():
    img = rand_batch(1, H, W)
    m = hard_half_mask(H, W)
    node = CAmod.ChromaticAberration()
    out, = node.execute(img, lens="Custom", strength=1.0, shift_r=SHIFT_R, shift_b=SHIFT_B, mask=m)
    check("I7 CA green channel bitwise untouched with mask connected",
          torch.equal(out[..., 1], img[..., 1]))


def i7_seam_same_mask():
    fn, name = seam_fn(CAmod, ("_masked_coords_ca", "_masked_coords"))
    if fn is None:
        check("I7 CA seam function found", False, "no masked-coords seam on chromatic_aberration module")
        return
    h, w = H, W
    m = hard_half_mask(h, w)
    d = fn(h, w, SHIFT_R, SHIFT_B, m, torch.device('cpu'))
    check("I7 CA seam returns a dict keyed by active channel", isinstance(d, dict) and 1 not in d,
          f"keys={sorted(d.keys()) if isinstance(d, dict) else type(d)}")
    if not isinstance(d, dict):
        return
    yy = torch.arange(h, dtype=torch.float32)
    xx = torch.arange(w, dtype=torch.float32)
    yy, xx = torch.meshgrid(yy, xx, indexing='ij')
    for c, vals in d.items():
        src_y, src_x, src_y_m, src_x_m = vals
        rec_y = yy + m * (src_y - yy)
        rec_x = xx + m * (src_x - xx)
        check(f"I7 CA seam channel {c}: recompute bitwise == seam masked coords",
              torch.equal(rec_y, src_y_m) and torch.equal(rec_x, src_x_m))


def i7_nc_per_channel_masks():
    """Reference (self-written, not the built node): use DIFFERENT masks for
    R and B. Lateral CA's R:B displacement ratio is shift_r/shift_b, constant
    across the frame under one mask; two masks must break it wherever they
    differ."""
    h, w = H, W
    cy, cx = h / 2.0, w / 2.0
    max_r = math.sqrt(cx * cx + cy * cy)
    yy = torch.arange(h, dtype=torch.float32)
    xx = torch.arange(w, dtype=torch.float32)
    yy, xx = torch.meshgrid(yy, xx, indexing='ij')
    dy, dx = yy - cy, xx - cx
    r = torch.sqrt(dy * dy + dx * dx) / max_r
    scale = min(h, w) / 1024.0
    sr, sb = SHIFT_R * scale, SHIFT_B * scale

    scale_r = 1.0 + (sr / max_r) * r
    scale_b = 1.0 + (sb / max_r) * r
    Dx_r = dx * scale_r - dx
    Dx_b = dx * scale_b - dx

    mask_r = hard_half_mask(h, w)
    mask_b = torch.zeros(h, w); mask_b[:h // 2, :] = 1.0  # a DIFFERENT region

    Dxr_m = mask_r * Dx_r
    true_ratio = sr / sb
    differ_region = (mask_r != mask_b) & (Dx_b.abs() > 1e-6)
    achieved_ratio = Dxr_m[differ_region] / Dx_b[differ_region]
    gap = (achieved_ratio - true_ratio).abs().max().item()
    must_fail("I7 NC per-channel masks preserve the lens-true R:B ratio",
              gap < 5e-2, f"max ratio gap {gap:.3f} on {int(differ_region.sum())} px "
                          "(spec measured ~0.91 with one-mask violated)")


run_block("I7  CA one-mask coherence: green untouched", i7_green_untouched)
run_block("I7  CA one-mask coherence: seam recompute", i7_seam_same_mask)
run_block("I7 NC  per-channel-mask reference breaks the R:B ratio", i7_nc_per_channel_masks)


# =============================================================================
# I8 -- Lens Profile vignette is gated by the same mask. A synthetic
# vignette-only profile (k1=k2=ca_r=ca_b=0) is injected into
# data.lens_profiles.LENS_PROFILES_FLAT so the geometric warp of a FLAT gray
# image is a mathematical no-op (constant field survives resampling exactly),
# isolating the vignette term to float-noise precision -- letting tau8=1e-6
# actually mean something.
# =============================================================================

def i8():
    synth = DataLens.LensProfile(
        name="__I8 synthetic vignette-only__", brand="test", focal_length="0mm",
        max_aperture="f/0", k1=0.0, k2=0.0, ca_r=0.0, ca_b=0.0,
        vig_strength=0.45, vig_midpoint=0.4, description="teeth fixture")
    DataLens.LENS_PROFILES_FLAT["__I8_SYNTH__"] = synth

    h, w = H, W
    cy, cx = h / 2.0, w / 2.0
    yy = torch.arange(h, dtype=torch.float32)
    xx = torch.arange(w, dtype=torch.float32)
    yy, xx = torch.meshgrid(yy, xx, indexing='ij')
    ny, nx = (yy - cy) / cy, (xx - cx) / cx
    r2 = nx * nx + ny * ny
    rr = torch.sqrt(r2)
    idx = torch.argmin((rr - 0.75).abs())
    iy, ix = int(idx // w), int(idx % w)
    assert 0.70 < float(rr[iy, ix]) < 0.80, "I8 mid-radius sample point drifted"

    vig_strength, vig_mid = synth.vig_strength, synth.vig_midpoint
    cos_theta = 1.0 / torch.sqrt(1.0 + r2)
    falloff = cos_theta ** 4
    transition = ((rr - vig_mid * 0.8) / 0.4).clamp(0.0, 1.0)
    V_add = (1.0 - transition * (1.0 - falloff) * vig_strength * 2).clamp(0.0, 1.0)
    V_correct = 1.0 + transition * (1.0 / falloff.clamp(0.3, 1.0) - 1.0) * vig_strength

    gray_val = 0.5
    m_const = 0.5
    img = torch.full((1, h, w, 3), gray_val)
    m = const_mask(h, w, m_const)

    node = LPmod.LensProfile()
    for mode, V in (("Add Aberrations", V_add), ("Correct Aberrations", V_correct)):
        v_here = float(V[iy, ix])
        gated = 1.0 + m_const * (v_here - 1.0)
        predicted = gray_val * gated
        assert 0.0 < predicted < 1.0, f"I8 {mode} gated sample not strictly interior: {predicted}"

        out, = node.execute(img, lens="__I8_SYNTH__", mode=mode, strength=1.0, mask=m)
        actual = float(out[0, iy, ix, 0])
        d = abs(actual - predicted)
        check(f"I8 LP vignette gated ({mode}) at mid-radius sample",
              d <= TAU8, f"actual {actual:.8f} predicted {predicted:.8f} diff {d:.2e} <= tau8={TAU8}")

        ungated_predicted = gray_val * v_here
        gap = abs(actual - ungated_predicted)
        must_fail(f"I8 NC ungated variant ({mode}) matches real gated output",
                  gap < 5e-3, f"gap {gap:.3f} (spec measured ~0.21 Add / 0.24 Correct)")


run_block("I8  Lens Profile vignette gated by the same mask", i8)


# =============================================================================
# I9 -- resize plumbing: a low-res constant mask must resize internally to
# (near-)match a full-res constant mask of the same value. Kernel
# discrimination lives in I13; this is plumbing-only.
# =============================================================================

def i9_pair(name, run_fn):
    lo = const_mask(64, 64, 0.7)
    full = const_mask(H, W, 0.7)
    out_lo = run_fn(lo)
    out_full = run_fn(full)
    d = (out_lo - out_full).abs().max().item()
    check(f"I9 {name} 64x64 const 0.7 mask ~= full-res const 0.7 mask", d <= TAU9,
          f"max|diff| {d:.2e} <= tau9={TAU9}")
    mismatched = const_mask(64, 64, 0.3)
    out_mismatch = run_fn(mismatched)
    d_wrong = (out_mismatch - out_full).abs().max().item()
    must_fail(f"I9 NC {name} const 0.3 (low-res) matches const 0.7 (full-res)",
              d_wrong <= TAU9, f"max|diff| {d_wrong:.3f} > tau9")


def i9():
    img = rand_batch(1, H, W)

    node = LDmod.LensDistortion()
    i9_pair("LD", lambda m: node.execute(img, lens="Custom", strength=1.0, k1=K1, k2=K2, mask=m)[0])

    node = CAmod.ChromaticAberration()
    i9_pair("CA", lambda m: node.execute(img, lens="Custom", strength=1.0, shift_r=SHIFT_R, shift_b=SHIFT_B, mask=m)[0])

    node = LPmod.LensProfile()
    i9_pair("LP", lambda m: node.execute(img, lens=LP_LENS, mode="Add Aberrations", strength=1.0, mask=m)[0])

    node = PCmod.PerspectiveCorrect()
    i9_pair("PC", lambda m: node.execute(img, vertical=VERT, horizontal=HORIZ, rotation=ROT, auto_crop=True, mask=m)[0])


run_block("I9  resize plumbing (low-res mask ~= full-res mask)", i9)


# =============================================================================
# I10 -- batch / mask-batch pairing semantics.
#   M=1, B=3: broadcast, each frame bitwise == its solo run
#   M=2, B=3: pairing is min(i, M-1) -- frame 2 gets mask 1 -- bitwise vs
#             explicit solo runs
#   M=0: treated as absent, with a console warning
# Representative nodes: LD (torch engine) and PC (numpy engine).
# =============================================================================

def i10_torch(node_factory, exec_kwargs, name):
    node = node_factory()
    B = 3
    batch = rand_batch(B, 96, 128)

    # M=1 broadcast
    m1 = torch.stack([hard_half_mask(96, 128)], dim=0)  # (1, H, W)
    out_batch, = node.execute(batch, mask=m1, **exec_kwargs)
    for i in range(B):
        solo, = node.execute(batch[i:i + 1], mask=m1, **exec_kwargs)
        check(f"I10 {name} M=1/B=3 frame {i} bitwise == its solo run",
              torch.equal(out_batch[i:i + 1], solo))

    # M=2, B=3: pairing min(i, M-1) -> frame2 gets mask1
    mask0 = hard_half_mask(96, 128)
    mask1 = blob_mask(96, 128)
    m2 = torch.stack([mask0, mask1], dim=0)  # (2, H, W)
    out_batch2, = node.execute(batch, mask=m2, **exec_kwargs)
    expected_pairing = [mask0, mask1, mask1]
    for i in range(B):
        solo, = node.execute(batch[i:i + 1], mask=expected_pairing[i].unsqueeze(0), **exec_kwargs)
        check(f"I10 {name} M=2/B=3 frame {i} pairing bitwise == explicit min(i,M-1) run",
              torch.equal(out_batch2[i:i + 1], solo))

    # NC: shuffled pairing must differ
    shuffled_pairing = [mask1, mask0, mask0]
    shuffled_out = []
    for i in range(B):
        solo, = node.execute(batch[i:i + 1], mask=shuffled_pairing[i].unsqueeze(0), **exec_kwargs)
        shuffled_out.append(solo)
    shuffled_out = torch.cat(shuffled_out, dim=0)
    must_fail(f"I10 NC {name} shuffled pairing matches real min(i,M-1) batch output",
              torch.equal(out_batch2, shuffled_out))

    # M=0: treated as absent, with a console warning
    m0 = torch.zeros(0, 96, 128)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        out_m0, = node.execute(batch, mask=m0, **exec_kwargs)
    out_none, = node.execute(batch, mask=None, **exec_kwargs)
    check(f"I10 {name} M=0 (empty mask batch) treated as absent",
          torch.equal(out_m0, out_none))
    check(f"I10 {name} M=0 prints a console warning", "warn" in buf.getvalue().lower()
          or "empty" in buf.getvalue().lower() or "M=0" in buf.getvalue(),
          f"captured: {buf.getvalue()!r}"[:200])


def i10_pc():
    node = PCmod.PerspectiveCorrect()
    B = 3
    batch = rand_batch(B, 96, 128)
    kwargs = dict(vertical=VERT, horizontal=HORIZ, rotation=ROT, auto_crop=False)

    m1 = torch.stack([hard_half_mask(96, 128)], dim=0)
    out_batch, = node.execute(batch, mask=m1, **kwargs)
    for i in range(B):
        solo, = node.execute(batch[i:i + 1], mask=m1, **kwargs)
        check(f"I10 PC M=1/B=3 frame {i} bitwise == its solo run",
              torch.equal(out_batch[i:i + 1], solo))

    mask0 = hard_half_mask(96, 128)
    mask1 = blob_mask(96, 128)
    m2 = torch.stack([mask0, mask1], dim=0)
    out_batch2, = node.execute(batch, mask=m2, **kwargs)
    expected_pairing = [mask0, mask1, mask1]
    for i in range(B):
        solo, = node.execute(batch[i:i + 1], mask=expected_pairing[i].unsqueeze(0), **kwargs)
        check(f"I10 PC M=2/B=3 frame {i} pairing bitwise == explicit min(i,M-1) run",
              torch.equal(out_batch2[i:i + 1], solo))

    m0 = torch.zeros(0, 96, 128)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        out_m0, = node.execute(batch, mask=m0, **kwargs)
    out_none, = node.execute(batch, mask=None, **kwargs)
    check("I10 PC M=0 (empty mask batch) treated as absent", torch.equal(out_m0, out_none))
    check("I10 PC M=0 prints a console warning",
          "warn" in buf.getvalue().lower() or "empty" in buf.getvalue().lower()
          or "M=0" in buf.getvalue(), f"captured: {buf.getvalue()!r}"[:200])


run_block("I10  batch / mask-batch pairing semantics (LD, torch engine)",
          lambda: i10_torch(LDmod.LensDistortion,
                             dict(lens="Custom", strength=1.0, k1=K1, k2=K2), "LD"))
run_block("I10  batch / mask-batch pairing semantics (PC, numpy engine)", i10_pc)


# =============================================================================
# I11 -- Perspective auto_crop is ignored once a mask is connected.
# =============================================================================

def i11():
    img = rand_batch(2, H, W)
    m = hard_half_mask(H, W)
    node = PCmod.PerspectiveCorrect()

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        out_true, = node.execute(img, vertical=VERT, horizontal=HORIZ, rotation=ROT, auto_crop=True, mask=m)
    out_false, = node.execute(img, vertical=VERT, horizontal=HORIZ, rotation=ROT, auto_crop=False, mask=m)
    check("I11 PC mask connected: auto_crop=True bitwise == auto_crop=False",
          torch.equal(out_true, out_false))
    check("I11 PC prints the mask-connected/auto_crop-ignored console note",
          "auto_crop" in buf.getvalue() and "ignored" in buf.getvalue(),
          f"captured: {buf.getvalue()!r}"[:200])

    out_true_nomask, = node.execute(img, vertical=VERT, horizontal=HORIZ, rotation=ROT,
                                    auto_crop=True, mask=None)
    out_false_nomask, = node.execute(img, vertical=VERT, horizontal=HORIZ, rotation=ROT,
                                     auto_crop=False, mask=None)
    d = (out_true_nomask - out_false_nomask).abs().max().item()
    must_fail("I11 NC mask=None: auto_crop=True bitwise == auto_crop=False",
              d < 1e-9, f"max|diff| {d:.3f} (spec measured ~0.96 apart)")


run_block("I11  PC auto_crop ignored once a mask is connected", i11)


# =============================================================================
# I12 -- mask values clamp to [0, 1].
# =============================================================================

def i12():
    img = rand_batch(1, H, W)

    for name, node, kwargs in (
        ("LD", LDmod.LensDistortion(), dict(lens="Custom", strength=1.0, k1=K1, k2=K2)),
        ("PC", PCmod.PerspectiveCorrect(), dict(vertical=VERT, horizontal=HORIZ, rotation=ROT, auto_crop=False)),
    ):
        m_2 = const_mask(H, W, 2.0)
        m_1 = const_mask(H, W, 1.0)
        out2, = node.execute(img, mask=m_2, **kwargs)
        out1, = node.execute(img, mask=m_1, **kwargs)
        check(f"I12 {name} mask=2.0 clamps bitwise == mask=1.0", torch.equal(out2, out1))

        m_neg = const_mask(H, W, -0.5)
        m_0 = const_mask(H, W, 0.0)
        outneg, = node.execute(img, mask=m_neg, **kwargs)
        out0, = node.execute(img, mask=m_0, **kwargs)
        check(f"I12 {name} mask=-0.5 clamps bitwise == mask=0.0", torch.equal(outneg, out0))

        m_half = const_mask(H, W, 0.5)
        outhalf, = node.execute(img, mask=m_half, **kwargs)
        must_fail(f"I12 NC {name} mask=0.5 matches mask=1.0", torch.equal(outhalf, out1))


run_block("I12  mask clamp to [0, 1]", i12)


# =============================================================================
# I13 -- cross-engine resize agreement. Fully self-contained: both resize
# paths are implemented here exactly per spec Sec 3's table, independent of
# the built nodes.
#   torch nodes:  F.interpolate(mode='bilinear', align_corners=True) + clamp
#   Perspective:  scipy zoom(order=1, mode='nearest') + [:H,:W] crop + clip
# =============================================================================

def torch_resize_mask(mask_np, H_, W_):
    t = torch.from_numpy(mask_np).float().unsqueeze(0).unsqueeze(0)
    out = F.interpolate(t, size=(H_, W_), mode='bilinear', align_corners=True)
    return out.squeeze(0).squeeze(0).clamp(0.0, 1.0).numpy()


def torch_resize_mask_nearest_NC(mask_np, H_, W_):
    """NC variant: nearest-KERNEL torch resize (wrong per spec Sec 3)."""
    t = torch.from_numpy(mask_np).float().unsqueeze(0).unsqueeze(0)
    out = F.interpolate(t, size=(H_, W_), mode='nearest')
    return out.squeeze(0).squeeze(0).clamp(0.0, 1.0).numpy()


def scipy_resize_mask(mask_np, H_, W_):
    h0, w0 = mask_np.shape
    scale_y, scale_x = H_ / h0, W_ / w0
    out = zoom(mask_np, (scale_y, scale_x), order=1, mode='nearest')
    out = out[:H_, :W_]
    return np.clip(out, 0.0, 1.0)


def disc_mask_np(h, w, r_frac=0.35):
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    cy, cx = h / 2.0, w / 2.0
    rr = np.hypot((yy - cy) / cy, (xx - cx) / cx)
    return (rr < r_frac).astype(np.float64)


def binary_noise_mask_np(h, w, seed=1):
    rng = np.random.default_rng(seed)
    return (rng.random((h, w)) > 0.5).astype(np.float64)


I13_SIZE_PAIRS = [
    ((512, 512), (1024, 1536)),
    ((64, 64), (H, W)),
    ((W, H), (H, W)),
    ((1080, 1920), (2160, 3840)),
    ((300, 300), (301, 301)),
]


def i13():
    max_gap = 0.0
    for (h0, w0), (Ht, Wt) in I13_SIZE_PAIRS:
        for maker_name, maker in (("disc", disc_mask_np), ("binary-noise", binary_noise_mask_np)):
            src = maker(h0, w0)
            a = torch_resize_mask(src, Ht, Wt)
            b = scipy_resize_mask(src, Ht, Wt)
            gap = float(np.abs(a - b).max())
            max_gap = max(max_gap, gap)
    check("I13 cross-engine resize agreement across 5 pinned size pairs",
          max_gap <= TAU13, f"max gap {max_gap:.2e} <= tau13={TAU13}")

    # exact-zero region stays exactly zero under both engines (corner far from disc)
    src = disc_mask_np(512, 512)
    A = torch_resize_mask(src, 1024, 1536)
    B = scipy_resize_mask(src, 1024, 1536)
    corner_a, corner_b = A[:20, :20], B[:20, :20]
    check("I13 exact-zero region stays exactly zero (torch)", bool(np.all(corner_a == 0.0)))
    check("I13 exact-zero region stays exactly zero (scipy)", bool(np.all(corner_b == 0.0)))

    # NC: nearest-KERNEL torch variant vs scipy nearest, on the disc
    src = disc_mask_np(512, 512)
    a_nc = torch_resize_mask_nearest_NC(src, 1024, 1536)
    b = scipy_resize_mask(src, 1024, 1536)
    gap_nc = float(np.abs(a_nc - b).max())
    must_fail("I13 NC nearest-kernel torch resize agrees with scipy within tau13",
              gap_nc <= TAU13, f"gap {gap_nc:.3f} > tau13 (kernel discrimination)")


run_block("I13  cross-engine resize agreement (self-contained)", i13)


# =============================================================================
# I14 -- Long Exposure resize fix, at node level: a 512x512 all-ones
# subject_mask on a 1024x1536 frame at a large exposure must show shake at
# EVERY frame-edge pixel. NC: a vendored copy of the OLD resize (zoom default
# mode='constant') produces a dead edge strip (min==0) in the resized mask.
# =============================================================================

def i14():
    node = LEmod.LongExposure()
    img = rand_batch(1, 1024, 1536)
    subject_mask = torch.ones(512, 512)
    out, = node.execute(img, exposure_px=60.0, hand_steadiness=1.0, variety=1.0,
                        roll_deg=5.0, seed=0, poses=24, subject_mask=subject_mask)
    diff = (out - img).abs().amax(dim=-1)[0]  # (H, W)
    edge = torch.cat([diff[0, :], diff[-1, :], diff[:, 0], diff[:, -1]])
    check("I14 every frame-edge pixel shows shake with an all-ones mask",
          bool((edge > 1e-4).all()), f"min edge |diff| {float(edge.min()):.2e}")

    # NC: vendored OLD resize (zoom default mode) leaves a dead edge strip
    mask512 = np.ones((512, 512), dtype=np.float64)
    h0, w0 = mask512.shape
    Ht, Wt = 1024, 1536
    scale_y, scale_x = Ht / h0, Wt / w0
    old_resized = zoom(mask512, (scale_y, scale_x), order=1)  # default mode='constant', cval=0.0
    must_fail("I14 NC old resize (zoom default mode) has no dead edge strip",
              old_resized.min() > 1e-12, f"min {old_resized.min():.3f} == 0.0 (the shipped bug)")

    new_resized = zoom(mask512, (scale_y, scale_x), order=1, mode='nearest')
    check("I14 sanity: fixed resize (mode='nearest') has no dead strip",
          new_resized.min() > 1.0 - 1e-12, f"min {new_resized.min():.12f}")


run_block("I14  Long Exposure resize fix: no dead edge strip", i14)


# =============================================================================
print("\n" + "=" * 78)
print(f"{len(PASSED)} passed, {len(FAILED)} failed  ({time.time() - t_start:.1f}s)")
if FAILED:
    print("FAILURES:")
    for f in FAILED:
        print(f"  - {f}")
print("=" * 78)
sys.exit(1 if FAILED else 0)
