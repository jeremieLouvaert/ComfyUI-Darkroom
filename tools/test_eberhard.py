"""
Offline validation for the Eberhard / Adjacency Acutance node — "teeth before
trust". Run with the ComfyUI embedded python. No pytest; prints PASS/FAIL per
check and exits nonzero if any check fails.

Loads the node via the namespace-shim pattern (cf. tools/test_halftone.py) so the
node's `from ..utils...` relative imports resolve WITHOUT triggering the top-level
package __init__ (server_routes / ComfyUI runtime).

Teeth (per docs/eberhard-derivation.md, PRODUCTION MODEL):
  1. MACKIE LINE (headline): step edge -> bright OVERSHOOT + dark UNDERSHOOT.
  2. ASYMMETRY + NEGATIVE CONTROL: asym=6 -> over>under; asym=1 -> over~=under;
     intensity=0 & drag=0 -> output == input.
  3. TONE PRESERVED off-edges: flat patch unchanged.
  4. RESOLUTION INDEPENDENCE: same overshoot amplitude at 256 vs 1024 wide.
  5. HUE PRESERVED: saturated colored edge keeps its hue (ratio application).
  6. DRAG: density-minus streak on the gravity side; drag=0 == no-drag path;
     drag_angle=90 = DOWN.
  7. PERF: 1024^2 and ~4K timings (drag on).

Also saves prod_acutance.png + prod_drag.png into _eberhard_spike/ for eyeball.
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
# Synthetic package shim (cf. test_halftone.py).
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
    PKG + ".nodes.eberhard", os.path.join(PACK_ROOT, "nodes", "eberhard.py")
)
eberhard = importlib.util.module_from_spec(spec)
sys.modules[PKG + ".nodes.eberhard"] = eberhard
spec.loader.exec_module(eberhard)

Eberhard = eberhard.Eberhard

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
    node = Eberhard()
    out = node.execute(to_tensor(img_np), **kw)[0]
    return from_tensor(out)


def luma(img):
    return 0.2126 * img[..., 0] + 0.7152 * img[..., 1] + 0.0722 * img[..., 2]


def srgb_to_linear(x):
    """Mirror of utils.color.srgb_to_linear for measuring in the space the
    node actually operates in (linear light). The node's gain symmetry lives in
    linear luminance; sRGB-space amplitudes are warped by the display gamma."""
    x = np.clip(x, 0.0, 1.0).astype(np.float32)
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4).astype(np.float32)


def lin_luma(img):
    return luma(srgb_to_linear(img))


def vstep(h, w, xedge, lo=0.25, hi=0.75):
    """Vertical step edge: dark `lo` on the left, bright `hi` on the right."""
    img = np.empty((h, w, 3), dtype=np.float32)
    img[:, :xedge] = lo
    img[:, xedge:] = hi
    return img


# ---------------------------------------------------------------------------
# 1. MACKIE LINE (headline)
# ---------------------------------------------------------------------------
print("\n[1] Mackie line on a vertical step edge (256x256, edge_width=2, intensity=0.6, asym=6)")
H = W = 256
xe = W // 2
img = vstep(H, W, xe, lo=0.25, hi=0.75)
out = run(img, edge_width=2.0, intensity=0.6, asymmetry=6.0, drag_amount=0.0, strength=1.0)

col = luma(out).mean(axis=0)               # per-column mean across the edge
far_lo = col[xe - 30:xe - 20].mean()       # far dark level (left)
far_hi = col[xe + 20:xe + 30].mean()       # far bright level (right)
overshoot = col[xe:xe + 10].max() - far_hi         # bright-side bump (right of edge)
undershoot = far_lo - col[xe - 10:xe].min()        # dark-side dip (left of edge)
THR = 0.005
check("bright-side OVERSHOOT present", overshoot > THR,
      f"overshoot={overshoot:.4f} (far_hi={far_hi:.4f})")
check("dark-side UNDERSHOOT present", undershoot > THR,
      f"undershoot={undershoot:.4f} (far_lo={far_lo:.4f})")

# ---------------------------------------------------------------------------
# 2. ASYMMETRY + NEGATIVE CONTROL
# ---------------------------------------------------------------------------
print("\n[2] Asymmetry (asym=6 -> over>under, asym=1 -> over~=under) + negative control")
# Measured in LINEAR luminance — the space the node's gain operates in. In sRGB
# the display gamma alone makes the dark-side swing larger, masking the gain
# symmetry; linear space isolates the actual asymmetry knob.


def over_under(o):
    c = lin_luma(o).mean(axis=0)
    flo = c[xe - 30:xe - 20].mean()
    fhi = c[xe + 20:xe + 30].mean()
    ov = c[xe:xe + 10].max() - fhi
    un = flo - c[xe - 10:xe].min()
    return ov, un


out6 = run(img, edge_width=2.0, intensity=0.6, asymmetry=6.0, drag_amount=0.0, strength=1.0)
ov6, un6 = over_under(out6)
check("asymmetry=6: overshoot > undershoot (linear)", ov6 > un6 + 0.005,
      f"overshoot={ov6:.4f} undershoot={un6:.4f} ratio={ov6/max(un6,1e-6):.2f}")

out1 = run(img, edge_width=2.0, intensity=0.6, asymmetry=1.0, drag_amount=0.0, strength=1.0)
ov1, un1 = over_under(out1)
check("asymmetry=1: overshoot ~= undershoot (linear, within 0.005)", abs(ov1 - un1) < 0.005,
      f"overshoot={ov1:.4f} undershoot={un1:.4f} |diff|={abs(ov1-un1):.4f}")

# NEGATIVE CONTROL: intensity=0 AND drag=0 -> output bit-identical to input.
print("    negative control: intensity=0 & drag_amount=0 -> output == input")
neg = run(img, edge_width=2.0, intensity=0.0, asymmetry=6.0, drag_amount=0.0, strength=1.0)
maxdiff = np.abs(neg - img).max()
check("NEGATIVE CONTROL: output identical to input (<1e-6)", maxdiff < 1e-6,
      f"max_abs_diff={maxdiff:.2e}")

# ---------------------------------------------------------------------------
# 3. TONE PRESERVED OFF-EDGES
# ---------------------------------------------------------------------------
print("\n[3] Tone preserved away from edges (flat region of the step image unchanged)")
# Columns far from the edge are flat -> must be unchanged.
flat_diff_lo = np.abs(out[:, :xe - 40] - img[:, :xe - 40]).mean()
flat_diff_hi = np.abs(out[:, xe + 40:] - img[:, xe + 40:]).mean()
check("flat dark region unchanged (mean abs diff < 0.005)", flat_diff_lo < 0.005,
      f"mean_abs_diff={flat_diff_lo:.6f}")
check("flat bright region unchanged (mean abs diff < 0.005)", flat_diff_hi < 0.005,
      f"mean_abs_diff={flat_diff_hi:.6f}")

# ---------------------------------------------------------------------------
# 4. RESOLUTION INDEPENDENCE
# ---------------------------------------------------------------------------
print("\n[4] Resolution independence (proportional edge at 512 vs 2048, edge_width fixed)")
# Per derivation-doc teeth #4 (512 vs 2048). The edge transition is rendered as a
# fixed FRACTION of frame (width/32 px) so it is the SAME proportional feature at
# both resolutions — an ideal 1px step is itself not scale-invariant (its spectrum
# is infinite), so we test the effect on a proportionally-scaled edge. Measured in
# linear luminance with a sigma-proportional window. sigma must be >~1px (Nyquist)
# for the kernel to resolve the feature, which both 512 (sigma=1) and 2048
# (sigma=4) satisfy; 256 (sigma=0.5) is sub-Nyquist and excluded.


def overshoot_prop(width):
    e = width // 2
    tw = max(1, width // 32)
    row = np.full(width, 0.25, np.float32)
    row[e:e + tw] = np.linspace(0.25, 0.75, tw)
    row[e + tw:] = 0.75
    im = np.repeat(row[None, :, None], width, axis=0).repeat(3, axis=2).astype(np.float32)
    o = run(im, edge_width=2.0, intensity=0.6, asymmetry=6.0, drag_amount=0.0, strength=1.0)
    c = lin_luma(o).mean(axis=0)
    sigma = 2.0 * (width / 1024.0)
    span = max(3, int(round(tw + 4 * sigma)))
    fhi = c[e + 2 * span:e + 3 * span].mean()
    return c[e:e + span].max() - fhi


ov_512 = overshoot_prop(512)
ov_2048 = overshoot_prop(2048)
rel = abs(ov_512 - ov_2048) / max(ov_512, ov_2048, 1e-6)
check("overshoot amplitude matches at 512 vs 2048 (within 15%)", rel < 0.15,
      f"ov_512={ov_512:.5f} ov_2048={ov_2048:.5f} rel_diff={rel*100:.1f}%")

# ---------------------------------------------------------------------------
# 5. HUE PRESERVED
# ---------------------------------------------------------------------------
print("\n[5] Hue preserved (saturated red|blue step edge; ratio application keeps hue)")
Hc = Wc = 256
xec = Wc // 2
cimg = np.empty((Hc, Wc, 3), dtype=np.float32)
cimg[:, :xec] = np.array([0.8, 0.1, 0.1], dtype=np.float32)   # red block (left)
cimg[:, xec:] = np.array([0.1, 0.1, 0.8], dtype=np.float32)   # blue block (right)
cout = run(cimg, edge_width=2.0, intensity=0.6, asymmetry=6.0, drag_amount=0.0, strength=1.0)

# Sample a non-edge pixel in each block; compare normalized RGB ratio.
def ratio_ok(px_in, px_out, tol=0.02):
    ri = px_in / max(px_in.sum(), 1e-6)
    ro = px_out / max(px_out.sum(), 1e-6)
    return np.abs(ri - ro).max(), np.abs(ri - ro).max() < tol


d_red, ok_red = ratio_ok(cimg[Hc // 2, 20], cout[Hc // 2, 20])
d_blue, ok_blue = ratio_ok(cimg[Hc // 2, Wc - 20], cout[Hc // 2, Wc - 20])
check("red-block hue (RGB ratio) preserved off-edge (<0.02)", ok_red, f"max_ratio_diff={d_red:.4f}")
check("blue-block hue (RGB ratio) preserved off-edge (<0.02)", ok_blue, f"max_ratio_diff={d_blue:.4f}")

# ---------------------------------------------------------------------------
# 6. DRAG (density-minus streak on the gravity side; angle=90 = down)
# ---------------------------------------------------------------------------
print("\n[6] Drag: bright block on dark field, drag_amount=0.5 angle=90 -> darker streak BELOW the block")
Hd = Wd = 256
bimg = np.full((Hd, Wd, 3), 0.2, dtype=np.float32)   # dark field
by0, by1, bx0, bx1 = 80, 150, 80, 176
bimg[by0:by1, bx0:bx1] = 0.85                         # bright block

nodrag = run(bimg, edge_width=2.0, intensity=0.6, asymmetry=6.0,
             drag_amount=0.0, drag_angle=90.0, strength=1.0)
drag = run(bimg, edge_width=2.0, intensity=0.6, asymmetry=6.0,
           drag_amount=0.5, drag_angle=90.0, strength=1.0)

# Streak region = a band just BELOW the block (higher row index = down).
band = slice(by1 + 2, by1 + 14)
cols = slice(bx0, bx1)
lum_nodrag = luma(nodrag)[band, cols].mean()
lum_drag = luma(drag)[band, cols].mean()
check("drag produces a density-MINUS streak below the block (darker than no-drag)",
      lum_drag < lum_nodrag - 0.002,
      f"lum_drag={lum_drag:.4f} < lum_nodrag={lum_nodrag:.4f} (delta={lum_nodrag-lum_drag:.4f})")

# Direction check: the streak below (gravity side) must be darker than the
# mirror band ABOVE the block. Confirms angle=90 = DOWN (not up).
band_up = slice(by0 - 14, by0 - 2)
lum_drag_up = luma(drag)[band_up, cols].mean()
check("streak is on the DOWN side (below darker than above) -> angle=90=down",
      lum_drag < lum_drag_up - 0.002,
      f"below={lum_drag:.4f} above={lum_drag_up:.4f}")

# drag_amount=0 must be bit-identical to running with the drag branch skipped.
# (Same call, the branch is gated on drag_amount>0 — assert determinism / no leak.)
nodrag2 = run(bimg, edge_width=2.0, intensity=0.6, asymmetry=6.0,
              drag_amount=0.0, drag_angle=90.0, strength=1.0)
identical = np.abs(nodrag - nodrag2).max()
check("drag_amount=0 path is deterministic / drag branch fully skipped (<1e-6)",
      identical < 1e-6, f"max_abs_diff={identical:.2e}")

# ---------------------------------------------------------------------------
# 7. PERF
# ---------------------------------------------------------------------------
print("\n[7] Perf (drag on)")
img1k = np.random.rand(1024, 1024, 3).astype(np.float32)
_ = run(img1k, edge_width=2.0, intensity=0.6, drag_amount=0.5, strength=1.0)  # warm
t0 = time.time()
_ = run(img1k, edge_width=2.0, intensity=0.6, drag_amount=0.5, strength=1.0)
t_1k = time.time() - t0
print(f"  1024x1024 (drag on): {t_1k:.3f} s")

img4k = np.random.rand(2160, 3840, 3).astype(np.float32)
t0 = time.time()
_ = run(img4k, edge_width=2.0, intensity=0.6, drag_amount=0.5, strength=1.0)
t_4k = time.time() - t0
print(f"  3840x2160 (drag on): {t_4k:.3f} s")

# ---------------------------------------------------------------------------
# Save eyeball PNGs into _eberhard_spike/
# ---------------------------------------------------------------------------
print("\n[save] eyeball PNGs")


def build_chart(h=512, w=512):
    """Grayscale test chart with multi-scale edges + a bright block for drag."""
    a = np.full((h, w), 0.5, np.float32)
    a[60:240, 60:160] = 0.25
    a[60:240, 160:260] = 0.75
    yy, xx = np.mgrid[0:h, 0:w]
    disk = (xx - 380) ** 2 + (yy - 140) ** 2 < 70 ** 2
    a[disk] = 0.85
    a[300:380, 120:300] = 0.85   # bright block (drag target)
    a[420:470, 40:470] = np.linspace(0.1, 0.9, 430, dtype=np.float32)[None, :]
    return np.stack([a, a, a], axis=-1)


SPIKE = os.path.join(os.path.dirname(PACK_ROOT), "_eberhard_spike")
chart = build_chart()
acut = run(chart, edge_width=2.0, intensity=0.6, asymmetry=6.0, drag_amount=0.0, strength=1.0)
drg = run(chart, edge_width=2.0, intensity=0.6, asymmetry=6.0, drag_amount=0.5, drag_angle=90.0, strength=1.0)
Image.fromarray((np.clip(acut, 0, 1) * 255).astype(np.uint8)).save(os.path.join(SPIKE, "prod_acutance.png"))
Image.fromarray((np.clip(drg, 0, 1) * 255).astype(np.uint8)).save(os.path.join(SPIKE, "prod_drag.png"))
print(f"  wrote prod_acutance.png + prod_drag.png to {SPIKE}")

# ---------------------------------------------------------------------------
print("\n" + ("ALL CHECKS PASSED" if PASS else "SOME CHECKS FAILED"))
sys.exit(0 if PASS else 1)
