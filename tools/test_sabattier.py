"""
Offline validation for the Sabattier (Solarization) node — "teeth before
trust". Run with the ComfyUI embedded python. No pytest; prints PASS/FAIL per
check and exits nonzero if any check fails.

Loads the node via the namespace-shim pattern (cf. tools/test_eberhard.py) so
the node's `from ..utils...` relative imports resolve WITHOUT triggering the
top-level package __init__ (server_routes / ComfyUI runtime).

Teeth (per docs/sabattier-derivation.md §7):
  1. FOLD / direction: ramp -> argmax interior & highlights reversed.
     NEGATIVE CONTROL: re_exposure=0 -> identity LUT -> argmax at last index.
  2. Deepest black preserved: x_out[0] < 0.02.
  3. Identity: strength=0 bit-identical; re_exposure=0 & mackie_intensity=0 ~identity.
  4. Renormalization / not-muddy: max output luminance > 0.9 for tones covering the ridge.
     NEGATIVE CONTROL: normalization disabled -> max stays low (< 0.5).
  5. Mackie line present: step edge + mackie_intensity>0 -> bright overshoot on bright side.
     NEGATIVE CONTROL: mackie_intensity=0 -> no overshoot.
  6. Resolution independence: Mackie scale proportional to long edge (512 vs 2048).
  7. Hue preserved: colored patch keeps hue angle through ratio recombine.
  8. Perf gate: 1024^2 sub-second, ~4K a couple seconds.
"""

import os
import sys
import time
import types
import importlib.util

import numpy as np
import torch

PACK_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ---------------------------------------------------------------------------
# Synthetic package shim (mirrors test_eberhard.py).
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

# Load color utils first (sabattier imports them)
spec_color = importlib.util.spec_from_file_location(
    PKG + ".utils.color", os.path.join(PACK_ROOT, "utils", "color.py")
)
color_mod = importlib.util.module_from_spec(spec_color)
sys.modules[PKG + ".utils.color"] = color_mod
spec_color.loader.exec_module(color_mod)

# Load image utils
spec_image = importlib.util.spec_from_file_location(
    PKG + ".utils.image", os.path.join(PACK_ROOT, "utils", "image.py")
)
image_mod = importlib.util.module_from_spec(spec_image)
sys.modules[PKG + ".utils.image"] = image_mod
spec_image.loader.exec_module(image_mod)

# Load the node
spec = importlib.util.spec_from_file_location(
    PKG + ".nodes.sabattier", os.path.join(PACK_ROOT, "nodes", "sabattier.py")
)
sabattier_mod = importlib.util.module_from_spec(spec)
sys.modules[PKG + ".nodes.sabattier"] = sabattier_mod
spec.loader.exec_module(sabattier_mod)

Sabattier = sabattier_mod.Sabattier
solarize_lut = sabattier_mod.solarize_lut

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
    """(H, W, C) float32 -> (1, H, W, C) tensor."""
    return torch.from_numpy(np.ascontiguousarray(arr, dtype=np.float32)).unsqueeze(0)


def from_tensor(t):
    """(1, H, W, C) tensor -> (H, W, C) float32."""
    return t[0].cpu().numpy().astype(np.float32)


def run(img_np, **kw):
    node = Sabattier()
    out = node.execute(to_tensor(img_np), **kw)[0]
    return from_tensor(out)


def luma(img):
    """sRGB display-space luminance."""
    return 0.2126 * img[..., 0] + 0.7152 * img[..., 1] + 0.0722 * img[..., 2]


def srgb_to_linear(x):
    x = np.clip(x, 0.0, 1.0).astype(np.float32)
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4).astype(np.float32)


def lin_luma(img):
    return luma(srgb_to_linear(img))


# ---------------------------------------------------------------------------
# 1. FOLD / DIRECTION (headline) — including NEGATIVE CONTROL
# ---------------------------------------------------------------------------
print("\n[1] Fold / direction: 0->1 luminance ramp, re_exposure=1.0 -> interior peak + reversed highlights")
print("    NEGATIVE CONTROL: re_exposure=0 -> identity -> argmax at last index")

N = 64
# Build a ramp image: each row is a constant gray value 0..1, arranged as a
# single-column image so we can test the LUT numerically.
ramp_vals = np.linspace(0.0, 1.0, N, dtype=np.float32)

# Test the LUT directly (same function the node uses internally)
xs = np.linspace(0.0, 1.0, 256, dtype=np.float32)
ys = solarize_lut(xs, beta=1.0, k=1.0, normalize=True)

peak_i = int(np.argmax(ys))
check(
    "argmax is strictly INTERIOR (not at last index)",
    peak_i < len(ys) - 1,
    f"argmax_index={peak_i} (max_index={len(ys)-1})"
)
check(
    "highlights reversed: ys[-1] < max(ys)",
    ys[-1] < ys[peak_i] - 1e-3,
    f"ys[-1]={ys[-1]:.4f} peak={ys[peak_i]:.4f} at x={xs[peak_i]:.3f}"
)

# NEGATIVE CONTROL: re_exposure=0 -> LUT is identity (solarize_lut not called)
# We verify via the node's early-return path AND the LUT directly.
# With beta=0: x * 10^(0) = x -> monotonic -> argmax at last index.
ys_zero = solarize_lut(xs, beta=0.0, k=1.0, normalize=True)
peak_zero = int(np.argmax(ys_zero))
check(
    "NEGATIVE CONTROL fires: re_exposure=0 -> argmax at last index (identity LUT)",
    peak_zero == len(ys_zero) - 1,
    f"argmax_index={peak_zero} (last={len(ys_zero)-1})"
)

# ---------------------------------------------------------------------------
# 2. DEEPEST BLACK PRESERVED
# ---------------------------------------------------------------------------
print("\n[2] Deepest black preserved: x_out[0] < 0.02")

# eps clips to 1e-4 in the LUT; at eps: eps * 10^(-beta * eps^k) ~ eps -> near 0
# With normalization the near-zero value stays very small.
ys_check = solarize_lut(xs, beta=1.0, k=1.0, normalize=True)
check(
    "deepest black (x~0) stays near-black: ys[0] < 0.02",
    float(ys_check[0]) < 0.02,
    f"ys[0]={ys_check[0]:.5f}"
)

# ---------------------------------------------------------------------------
# 3. IDENTITY PATHS
# ---------------------------------------------------------------------------
print("\n[3] Identity paths: strength=0 and (re_exposure=0 & mackie_intensity=0)")

img_gray = np.full((64, 64, 3), 0.5, dtype=np.float32)

# strength=0 -> early-return identity, bit-identical
out_s0 = run(img_gray, re_exposure=1.0, mackie_intensity=0.5, strength=0.0)
diff_s0 = np.abs(out_s0 - img_gray).max()
check(
    "strength=0 -> bit-identical output",
    diff_s0 == 0.0,
    f"max_abs_diff={diff_s0:.2e}"
)

# re_exposure=0 & mackie_intensity=0 -> early-return identity (fp)
out_both0 = run(img_gray, re_exposure=0.0, mackie_intensity=0.0, strength=1.0)
diff_both0 = np.abs(out_both0 - img_gray).max()
check(
    "re_exposure=0 & mackie_intensity=0 -> ~identity (early-return, bit-identical)",
    diff_both0 == 0.0,
    f"max_abs_diff={diff_both0:.2e}"
)

# ---------------------------------------------------------------------------
# 4. RENORMALIZATION / NOT-MUDDY (key spike finding) + NEGATIVE CONTROL
# ---------------------------------------------------------------------------
print("\n[4] Renormalization: max output luminance > 0.9 for full-range image")
print("    NEGATIVE CONTROL: normalization OFF -> max stays low (< 0.5)")

# Build a full-range image (covers 0..1 in luminance, ensuring ridge coverage)
# Use a gradient image with all tones present.
ramp_img = np.zeros((64, 256, 3), dtype=np.float32)
ramp_img[:, :, :] = np.linspace(0.0, 1.0, 256, dtype=np.float32)[None, :, None]
# Convert from linear to sRGB for input (the node expects sRGB input)
ramp_srgb = np.where(
    ramp_img <= 0.0031308, ramp_img * 12.92, 1.055 * np.power(np.clip(ramp_img, 0, 1), 1.0 / 2.4) - 0.055
).astype(np.float32)

out_norm = run(ramp_srgb, re_exposure=1.0, shield=1.0, mackie_intensity=0.0, strength=1.0)
max_lum_norm = float(lin_luma(out_norm).max())
check(
    "with normalization: max output luminance > 0.9",
    max_lum_norm > 0.9,
    f"max_lin_lum={max_lum_norm:.4f}"
)

# NEGATIVE CONTROL: call solarize_lut directly with normalize=False
ys_no_norm = solarize_lut(xs, beta=1.0, k=1.0, normalize=False)
max_no_norm = float(ys_no_norm.max())
check(
    "NEGATIVE CONTROL fires: normalization OFF -> peak < 0.5 (muddy)",
    max_no_norm < 0.5,
    f"unnormalized_peak={max_no_norm:.4f}"
)

# ---------------------------------------------------------------------------
# 5. MACKIE LINE PRESENT + NEGATIVE CONTROL
# ---------------------------------------------------------------------------
print("\n[5] Mackie line: step edge + mackie_intensity>0 -> bright overshoot on bright side")
print("    NEGATIVE CONTROL: mackie_intensity=0 -> no overshoot")

H = W = 256
xe = W // 2
# Vertical step edge (all three channels equal -> grayscale)
step = np.empty((H, W, 3), dtype=np.float32)
step[:, :xe] = 0.25   # dark left
step[:, xe:] = 0.75   # bright right

out_mackie = run(step, re_exposure=1.0, mackie_intensity=0.8, edge_width=2.0, strength=1.0)
out_no_mackie = run(step, re_exposure=1.0, mackie_intensity=0.0, edge_width=2.0, strength=1.0)

# Measure bright-side overshoot in linear luminance (columns near the edge vs far right)
def bright_overshoot(out, xedge, w):
    c = lin_luma(out).mean(axis=0)
    far_hi = c[xedge + 20:xedge + 30].mean()
    near_hi = c[xedge:xedge + 10].max()
    return float(near_hi - far_hi)


ov_with = bright_overshoot(out_mackie, xe, W)
ov_without = bright_overshoot(out_no_mackie, xe, W)

THR = 0.005
check(
    "mackie_intensity=0.8: bright-side overshoot present (REE > 0.005)",
    ov_with > THR,
    f"overshoot={ov_with:.5f}"
)
check(
    "NEGATIVE CONTROL fires: mackie_intensity=0 -> no meaningful overshoot (<= THR)",
    ov_without <= THR,
    f"overshoot_no_mackie={ov_without:.5f}"
)

# ---------------------------------------------------------------------------
# 6. RESOLUTION INDEPENDENCE
# ---------------------------------------------------------------------------
print("\n[6] Resolution independence: same Mackie overshoot at 512 vs 2048 (within 15%)")


def overshoot_at_width(width):
    """Proportional step edge at given width, measure Mackie overshoot."""
    e = width // 2
    tw = max(1, width // 32)
    row = np.full(width, 0.25, np.float32)
    row[e:e + tw] = np.linspace(0.25, 0.75, tw)
    row[e + tw:] = 0.75
    im = np.repeat(row[None, :, None], width, axis=0).repeat(3, axis=2).astype(np.float32)
    o = run(im, re_exposure=1.0, mackie_intensity=0.6, edge_width=2.0, strength=1.0)
    c = lin_luma(o).mean(axis=0)
    sigma = 2.0 * (width / 1024.0)
    span = max(3, int(round(tw + 4 * sigma)))
    fhi = c[e + 2 * span:e + 3 * span].mean()
    return float(c[e:e + span].max() - fhi)


ov_512 = overshoot_at_width(512)
ov_2048 = overshoot_at_width(2048)
rel = abs(ov_512 - ov_2048) / max(ov_512, ov_2048, 1e-6)
check(
    "overshoot amplitude matches at 512 vs 2048 (within 15%)",
    rel < 0.15,
    f"ov_512={ov_512:.5f} ov_2048={ov_2048:.5f} rel_diff={rel*100:.1f}%"
)

# ---------------------------------------------------------------------------
# 7. HUE PRESERVED
# ---------------------------------------------------------------------------
print("\n[7] Hue preserved: colored patch keeps its hue angle through ratio recombine")
# The ratio recombine (out_lin = lin * scalar) multiplies all channels by the
# same luminance ratio, so hue is EXACTLY preserved in LINEAR space. We check
# in linear space to be definitive. A note on what "hue preserved" means here:
# the sRGB → gamma transform is NOT linear, so sRGB channel ratios are NOT
# preserved — only linear channel ratios are. This is the same guarantee
# Eberhard makes, and it is sufficient for hue-angle preservation.
#
# We use mid-tone saturated colors (neither very dark nor very bright) to
# avoid the corner case where the luminance boost clips a channel to 1.0
# (which WOULD break linear hue for that pixel — an expected artifact of the
# luminance-ratio approach when the input has very high dynamic range).

Hc = Wc = 128
cimg = np.empty((Hc, Wc, 3), dtype=np.float32)
# Choose saturated-but-dim colors whose linear luminance is low enough that
# the solarization ratio boost (~6x) does NOT clip any channel to 1.0.
# [0.3, 0.05, 0.05] sRGB -> lin_R~0.073 -> 0.073*6.0 = 0.44 < 1.0 (safe).
cimg[:, :Wc // 2] = np.array([0.30, 0.05, 0.05], dtype=np.float32)   # dim red
cimg[:, Wc // 2:] = np.array([0.05, 0.05, 0.30], dtype=np.float32)   # dim blue

cout = run(cimg, re_exposure=1.0, mackie_intensity=0.3, strength=1.0)


def lin_from_srgb(px):
    px = np.clip(px, 0.0, 1.0).astype(np.float32)
    return np.where(px <= 0.04045, px / 12.92, ((px + 0.055) / 1.055) ** 2.4).astype(np.float32)


def hue_ok_linear(px_in_srgb, px_out_srgb, tol=0.005):
    """Hue check in LINEAR space: ratio recombine preserves linear channel ratios exactly."""
    li = lin_from_srgb(px_in_srgb)
    lo = lin_from_srgb(px_out_srgb)
    # Verify no channel clipped (if it did, hue cannot be preserved by any method)
    if lo.max() >= 0.9999:
        return False, 999.0
    ri = li / max(float(li.sum()), 1e-6)
    ro = lo / max(float(lo.sum()), 1e-6)
    diff = float(np.abs(ri - ro).max())
    return diff < tol, diff


ok_red, diff_red = hue_ok_linear(cimg[Hc // 2, 10], cout[Hc // 2, 10])
ok_blue, diff_blue = hue_ok_linear(cimg[Hc // 2, Wc - 10], cout[Hc // 2, Wc - 10])
check("red-block hue (linear RGB ratio) preserved off-edge (<0.005)", ok_red,
      f"linear_ratio_diff={diff_red:.5f}")
check("blue-block hue (linear RGB ratio) preserved off-edge (<0.005)", ok_blue,
      f"linear_ratio_diff={diff_blue:.5f}")

# ---------------------------------------------------------------------------
# 8. PERF GATE
# ---------------------------------------------------------------------------
print("\n[8] Perf gate (pointwise LUT + 1 gaussian blur)")

img1k = np.random.rand(1024, 1024, 3).astype(np.float32)
# warm-up
_ = run(img1k, re_exposure=1.0, mackie_intensity=0.5, strength=1.0)
t0 = time.time()
_ = run(img1k, re_exposure=1.0, mackie_intensity=0.5, strength=1.0)
t_1k = time.time() - t0
print(f"  1024x1024: {t_1k:.3f} s")
check("1024^2 under 1 second", t_1k < 1.0, f"{t_1k:.3f}s")

img4k = np.random.rand(2160, 3840, 3).astype(np.float32)
t0 = time.time()
_ = run(img4k, re_exposure=1.0, mackie_intensity=0.5, strength=1.0)
t_4k = time.time() - t0
print(f"  3840x2160: {t_4k:.3f} s")
check("~4K under 10 seconds", t_4k < 10.0, f"{t_4k:.3f}s")

# ---------------------------------------------------------------------------
print("\n" + ("ALL CHECKS PASSED" if PASS else "SOME CHECKS FAILED"))
sys.exit(0 if PASS else 1)
