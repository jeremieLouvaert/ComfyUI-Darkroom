"""
Offline validation for the Film Rebate node -- "teeth before trust". Run with
the ComfyUI embedded python. No pytest; prints PASS/FAIL per check and exits
nonzero if any check fails.

Loads the node via the namespace-shim pattern (cf. tools/test_halation.py) so
the node's `from ..utils...` relative imports resolve WITHOUT triggering the
top-level package __init__ (server_routes / ComfyUI runtime).

Teeth (plan lovely-coalescing-rain.md "Teeth" section):
  1. Perf pitch oracle (135): hole-center spacing == 4.75/36 * aperture_w_px,
     count == 8 per frame. NEGATIVE: a +10%-perturbed pitch fails the check.
  2. Canvas dims formula oracle per format (independently re-derived from the
     mm spec) +-1px; MASK area == composed image placement exactly.
  3. Rebate polarity: mean rebate color per film_type matches the table
     +-2/255; c41 blue > red (cool cast). NEGATIVE: swapped table fails.
  4. Filed-carrier: perimeter variance > 0 (default roughness); same seed ->
     bit-exact; different seed -> differs; roughness=0 -> variance ~= 0.
  5. Notch code: 4x5 top edge contains notch-shaped cuts; stock_name change
     -> different pattern; same name -> same pattern.
  6. Frame numbering: batch of 3 with increment -> edge-print region differs
     frame to frame; increment off -> identical.
  7. Resolution independence: 512 vs 2048 input -> all mm-derived dims scale
     ~4x +-1%.
  8. Determinism + perf budgets (1024^2, 4K -- printed, expect well under 1s).
  9. EYE renders (all 7 formats + polarity variants) -> ../_film_rebate_spike/
     for Jeremie's taste pass (not a physics gate).
"""

import hashlib
import os
import sys
import time
import types
import importlib.util
import importlib

import numpy as np
import torch
from PIL import Image

PACK_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ---------------------------------------------------------------------------
# Synthetic package shim (cf. test_halation.py).
# ---------------------------------------------------------------------------
PKG = "dr_pkg_film_rebate"

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
    PKG + ".nodes.film_rebate", os.path.join(PACK_ROOT, "nodes", "film_rebate.py")
)
fr_mod = importlib.util.module_from_spec(spec)
sys.modules[PKG + ".nodes.film_rebate"] = fr_mod
spec.loader.exec_module(fr_mod)

DarkroomFilmRebate = fr_mod.DarkroomFilmRebate
REBATE_COLORS = fr_mod.REBATE_COLORS
NOTCH_PATTERNS = fr_mod.NOTCH_PATTERNS
_hash_index = fr_mod._hash_index

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


def batch_tensor(arrs):
    return torch.stack([torch.from_numpy(np.ascontiguousarray(a, dtype=np.float32)) for a in arrs], dim=0)


def run(img_np, **kw):
    node = DarkroomFilmRebate()
    img_out, mask_out = node.execute(to_tensor(img_np), **kw)
    return img_out[0].cpu().numpy(), mask_out[0].cpu().numpy()


def run_batch(imgs_np, **kw):
    node = DarkroomFilmRebate()
    t = batch_tensor(imgs_np)
    img_out, mask_out = node.execute(t, **kw)
    return [img_out[i].cpu().numpy() for i in range(img_out.shape[0])], \
           [mask_out[i].cpu().numpy() for i in range(mask_out.shape[0])]


def synth_image(w, h, rng=None):
    rng = rng or np.random.default_rng(1234)
    # smooth-ish gradient + noise, avoids accidental collisions with rebate
    # colors (near-black) and background/hole colors so pattern detection is
    # unambiguous.
    yy, xx = np.mgrid[0:h, 0:w]
    img = np.stack([
        0.3 + 0.4 * (xx / max(w - 1, 1)),
        0.3 + 0.4 * (yy / max(h - 1, 1)),
        0.5 * np.ones((h, w)),
    ], axis=-1).astype(np.float32)
    img += rng.normal(0, 0.02, img.shape).astype(np.float32)
    return np.clip(img, 0.05, 0.95).astype(np.float32)


# ---------------------------------------------------------------------------
# Independent geometry predictors (re-derived from the mm spec table, NOT
# imported from the node -- this is the cross-check, not a mirror).
# ---------------------------------------------------------------------------
def predict_135_full(src_w, src_h, strip_context=2.0):
    px_per_mm = max(src_w, src_h) / 36.0
    inter_gap = 2.0
    canvas_w_mm = strip_context + inter_gap + 36.0 + inter_gap + strip_context
    canvas_h_mm = 35.0
    ap_w = round(36.0 * px_per_mm)
    ap_h = round(24.0 * px_per_mm)
    ap_x0 = round((strip_context + inter_gap) * px_per_mm)
    ap_y0 = round((35.0 - 24.0) / 2.0 * px_per_mm)
    return dict(px_per_mm=px_per_mm, canvas_w=round(canvas_w_mm * px_per_mm),
                canvas_h=round(canvas_h_mm * px_per_mm), ap_w=ap_w, ap_h=ap_h,
                ap_x0=ap_x0, ap_y0=ap_y0)


def predict_120(src_w, src_h, six_nine):
    ap_w_mm = 84.0 if six_nine else 56.0
    ap_h_mm = 56.0
    px_per_mm = max(src_w, src_h) / max(ap_w_mm, ap_h_mm)
    canvas_w_mm, canvas_h_mm = ap_w_mm, 61.0
    return dict(px_per_mm=px_per_mm, canvas_w=round(canvas_w_mm * px_per_mm),
                canvas_h=round(canvas_h_mm * px_per_mm),
                ap_w=round(ap_w_mm * px_per_mm), ap_h=round(ap_h_mm * px_per_mm),
                ap_x0=0, ap_y0=round((canvas_h_mm - ap_h_mm) / 2.0 * px_per_mm))


def predict_4x5(src_w, src_h):
    sheet_w_mm, sheet_h_mm, margin = 101.6, 127.0, 2.0
    ap_w_mm, ap_h_mm = sheet_w_mm - 2 * margin, sheet_h_mm - 2 * margin
    px_per_mm = max(src_w, src_h) / max(ap_w_mm, ap_h_mm)
    return dict(px_per_mm=px_per_mm, canvas_w=round(sheet_w_mm * px_per_mm),
                canvas_h=round(sheet_h_mm * px_per_mm),
                ap_w=round(ap_w_mm * px_per_mm), ap_h=round(ap_h_mm * px_per_mm),
                ap_x0=round(margin * px_per_mm), ap_y0=round(margin * px_per_mm))


def predict_polaroid(src_w, src_h):
    ap_w_mm, ap_h_mm = 79.0, 77.0
    px_per_mm = max(src_w, src_h) / max(ap_w_mm, ap_h_mm)
    print_w_mm, print_h_mm = 88.5, 107.5
    side_mm = (print_w_mm - ap_w_mm) / 2.0
    return dict(px_per_mm=px_per_mm, canvas_w=round(print_w_mm * px_per_mm),
                canvas_h=round(print_h_mm * px_per_mm),
                ap_w=round(ap_w_mm * px_per_mm), ap_h=round(ap_h_mm * px_per_mm),
                ap_x0=round(side_mm * px_per_mm), ap_y0=round(side_mm * px_per_mm))


def predict_slide(src_w, src_h):
    ap_w_mm, ap_h_mm, mount_mm = 23.0, 35.0, 50.0
    px_per_mm = max(src_w, src_h) / max(ap_w_mm, ap_h_mm)
    return dict(px_per_mm=px_per_mm, canvas_w=round(mount_mm * px_per_mm),
                canvas_h=round(mount_mm * px_per_mm),
                ap_w=round(ap_w_mm * px_per_mm), ap_h=round(ap_h_mm * px_per_mm),
                ap_x0=round((mount_mm - ap_w_mm) / 2.0 * px_per_mm),
                ap_y0=round((mount_mm - ap_h_mm) / 2.0 * px_per_mm))


def predict_filed_carrier(src_w, src_h, border_width=1.2, paper_margin_pct=6.0):
    px_per_mm = max(src_w, src_h) / 36.0
    border_px = max(1, round(border_width * px_per_mm))
    paper_px = max(1, round(paper_margin_pct / 100.0 * max(src_w, src_h)))
    return dict(px_per_mm=px_per_mm, canvas_w=src_w + 2 * border_px + 2 * paper_px,
                canvas_h=src_h + 2 * border_px + 2 * paper_px,
                ap_w=src_w, ap_h=src_h, ap_x0=border_px + paper_px, ap_y0=border_px + paper_px)


def with_margin(g):
    """Every format canvas gets a 4%-of-long-edge background margin added on
    all sides before the final image is returned."""
    margin = round(0.04 * max(g["canvas_w"], g["canvas_h"]))
    return g["canvas_w"] + 2 * margin, g["canvas_h"] + 2 * margin, margin


# ===========================================================================
print("\n[1] Perf pitch oracle (135 full frame)")
SRC_W, SRC_H = 720, 480
src = synth_image(SRC_W, SRC_H)
g135 = predict_135_full(SRC_W, SRC_H)
img_out, mask_out = run(src, format="135 full frame", film_type="Color neg", seed=1)

fw, fh, margin = with_margin(g135)
row_y = margin + round(2.0 * g135["px_per_mm"])  # top perf row inset 2mm from film edge (canvas top, not aperture top)
bg_rgb = np.array(fr_mod.BACKGROUND_COLORS["black"], dtype=np.float32) / 255.0
row = img_out[row_y, :, :]
is_hole = np.all(np.abs(row - bg_rgb) < 0.03, axis=-1)
xs = np.where(is_hole)[0]
# group contiguous runs -> centers
centers = []
if len(xs):
    runs = np.split(xs, np.where(np.diff(xs) > 1)[0] + 1)
    centers = [r.mean() for r in runs]
frame_x0 = margin + g135["ap_x0"]
frame_x1 = frame_x0 + g135["ap_w"]
in_frame = [c for c in centers if frame_x0 - 1 <= c <= frame_x1 + 1]
pitches = np.diff(sorted(in_frame)) if len(in_frame) > 1 else np.array([])
expected_pitch = (4.75 / 36.0) * g135["ap_w"]
check("8 perf holes detected in the frame x-range",
      len(in_frame) == 8, f"found {len(in_frame)}: {[round(c,1) for c in in_frame]}")
check("pitch == 4.75/36 * aperture_w_px (+-1px)",
      len(pitches) > 0 and np.allclose(pitches, expected_pitch, atol=1.0),
      f"pitches={np.round(pitches,2)} expected~{expected_pitch:.2f}")

print("    NEGATIVE CONTROL: pitch constant perturbed +10% must NOT match")
bad_expected = expected_pitch * 1.10
neg_ok = len(pitches) > 0 and np.allclose(pitches, bad_expected, atol=1.0)
check("perturbed-pitch prediction does NOT match rendered pitch (control fires)",
      not neg_ok, f"pitches={np.round(pitches,2)} bad_expected~{bad_expected:.2f}")

# ===========================================================================
print("\n[2] Canvas dims formula oracle (all formats) + MASK area exactness")
FORMAT_PREDICTORS = {
    "135 full frame": lambda w, h: predict_135_full(w, h),
    "120 6x6": lambda w, h: predict_120(w, h, six_nine=False),
    "120 6x9": lambda w, h: predict_120(w, h, six_nine=True),
    "4x5 sheet": lambda w, h: predict_4x5(w, h),
    "Polaroid": lambda w, h: predict_polaroid(w, h),
    "Slide mount": lambda w, h: predict_slide(w, h),
    "135 filed carrier": lambda w, h: predict_filed_carrier(w, h),
}

TEST_W, TEST_H = 640, 424
src2 = synth_image(TEST_W, TEST_H)
for fmt, predictor in FORMAT_PREDICTORS.items():
    g = predictor(TEST_W, TEST_H)
    pred_w, pred_h, pred_margin = with_margin(g)
    img_out, mask_out = run(src2, format=fmt, film_type="Color neg", compose="fill", seed=3)
    actual_h, actual_w = img_out.shape[0], img_out.shape[1]
    check(f"[{fmt}] canvas dims match mm-ratio prediction (+-1px)",
          abs(actual_w - pred_w) <= 1 and abs(actual_h - pred_h) <= 1,
          f"actual=({actual_w},{actual_h}) pred=({pred_w},{pred_h})")

    # MASK area exactness (fill mode -> full aperture rect)
    exp_x0 = pred_margin + g["ap_x0"]
    exp_y0 = pred_margin + g["ap_y0"]
    exp_area = g["ap_w"] * g["ap_h"]
    actual_area = int(mask_out.sum())
    ys, xs = np.where(mask_out > 0.5)
    actual_rect_ok = False
    if len(xs) and len(ys):
        actual_rect_ok = (xs.min() == exp_x0 and ys.min() == exp_y0 and
                           xs.max() == exp_x0 + g["ap_w"] - 1 and ys.max() == exp_y0 + g["ap_h"] - 1)
    check(f"[{fmt}] MASK area == aperture area exactly (fill mode)",
          actual_area == exp_area and actual_rect_ok,
          f"mask_area={actual_area} expected={exp_area} rect_ok={actual_rect_ok}")

# fit-mode mask sanity on 135 (letterboxed, sub-rect of aperture)
g_fit = predict_135_full(TEST_W, TEST_H)
img_fit, mask_fit = run(src2, format="135 full frame", compose="fit", seed=3)
scale = min(g_fit["ap_w"] / TEST_W, g_fit["ap_h"] / TEST_H)
exp_w = max(1, round(TEST_W * scale))
exp_h = max(1, round(TEST_H * scale))
check("[135 full frame] fit-mode MASK area == letterboxed sub-rect (+-4px area tol)",
      abs(int(mask_fit.sum()) - exp_w * exp_h) <= 4 * max(exp_w, exp_h),
      f"mask_area={int(mask_fit.sum())} expected~{exp_w*exp_h} ({exp_w}x{exp_h})")

# ===========================================================================
print("\n[3] Rebate polarity")
for film_type, key in fr_mod.FILM_TYPE_TO_KEY.items():
    expected = np.array(REBATE_COLORS[key], dtype=np.float32)
    img_out, _ = run(synth_image(400, 300), format="135 full frame", film_type=film_type,
                      strip_context=0.0, edge_print_intensity=0.0, seed=5)
    g = predict_135_full(400, 300, strip_context=0.0)
    _, _, margin = with_margin(g)
    # sample the outer rebate corner, well clear of perfs/aperture/text
    sample = img_out[margin + 1, margin + 1] * 255.0
    check(f"[{film_type}] rebate corner color matches table (+-2/255)",
          np.all(np.abs(sample - expected) <= 2.0),
          f"sampled={np.round(sample,1)} expected={expected}")

c41_img, _ = run(synth_image(400, 300), format="135 full frame", film_type="Color neg",
                  strip_context=0.0, edge_print_intensity=0.0, seed=5)
g0 = predict_135_full(400, 300, strip_context=0.0)
_, _, m0 = with_margin(g0)
c41_corner = c41_img[m0 + 1, m0 + 1]
check("c41 rebate: blue channel > red channel (cool cast)",
      c41_corner[2] > c41_corner[0], f"R={c41_corner[0]:.4f} B={c41_corner[2]:.4f}")

print("    NEGATIVE CONTROL: swapped table (bw_neg color asserted against c41 expectation) must fail")
bw_img, _ = run(synth_image(400, 300), format="135 full frame", film_type="B&W neg",
                 strip_context=0.0, edge_print_intensity=0.0, seed=5)
bw_corner = bw_img[m0 + 1, m0 + 1] * 255.0
c41_table = np.array(REBATE_COLORS["c41"], dtype=np.float32)
swapped_matches = np.all(np.abs(bw_corner - c41_table) <= 2.0)
check("bw_neg does NOT match the c41 table entry (control fires)",
      not swapped_matches, f"bw_corner={np.round(bw_corner,1)} c41_table={c41_table}")

# ===========================================================================
print("\n[4] Filed-carrier perimeter noise")
fc_src = synth_image(300, 220)


def border_outer_x_profile(img_out, g, margin, side="left"):
    """Scan rows through the border ring and find the paper/border transition
    x position (outer edge of the black border) -> a 1D profile. Constrained
    to [canvas paper edge, aperture edge) so the (also-dark) background
    margin outside the paper never gets picked up as the 'transition'."""
    x0 = margin + g["ap_x0"]
    y0 = margin + g["ap_y0"]
    xs = []
    for y in range(y0 + 2, y0 + g["ap_h"] - 2, 4):
        if side == "left":
            row = img_out[y, margin:x0][::-1]
        else:
            row = img_out[y, x0 + g["ap_w"]:margin + g["canvas_w"]]
        dark = np.where(row.mean(axis=-1) < 0.25)[0]
        xs.append(dark.max() if len(dark) else 0)
    return np.array(xs, dtype=np.float64)


g_fc = predict_filed_carrier(300, 220)
img_r05a, _ = run(fc_src, format="135 filed carrier", roughness=0.5, seed=11)
img_r05b, _ = run(fc_src, format="135 filed carrier", roughness=0.5, seed=11)
img_r05c, _ = run(fc_src, format="135 filed carrier", roughness=0.5, seed=99)
img_r0, _ = run(fc_src, format="135 filed carrier", roughness=0.0, seed=11)

_, _, m_fc = with_margin(g_fc)
prof_a = border_outer_x_profile(img_r05a, g_fc, m_fc, "left")
prof_c = border_outer_x_profile(img_r05c, g_fc, m_fc, "left")
prof_0 = border_outer_x_profile(img_r0, g_fc, m_fc, "left")

check("roughness=0.5: border-edge profile has variance > 0",
      prof_a.var() > 0.1, f"var={prof_a.var():.4f}")
check("same seed -> bit-exact identical border",
      np.array_equal(img_r05a, img_r05b), f"max_abs_diff={np.abs(img_r05a-img_r05b).max():.6f}")
check("different seed -> border differs",
      not np.array_equal(img_r05a, img_r05c) and prof_a.var() >= 0 and not np.array_equal(prof_a, prof_c),
      f"profiles equal: {np.array_equal(prof_a, prof_c)}")
check("roughness=0 -> straight edges (variance ~= 0)",
      prof_0.var() < 0.5, f"var={prof_0.var():.4f}")

# ===========================================================================
print("\n[5] Notch code (4x5 sheet)")


def has_notch_cuts(img_out, g, margin):
    """Scan the sheet's TOP edge (canvas/paper edge, not the aperture) near
    the top-right corner for background-color intrusions -- notches are cut
    inward from the true sheet edge (y=0 local / x=canvas_w local)."""
    zone_x0 = margin + g["canvas_w"] - round(12.0 * g["px_per_mm"])
    zone_x1 = margin + g["canvas_w"]
    band = img_out[margin:margin + round(3.0 * g["px_per_mm"]), zone_x0:zone_x1]
    bg = np.array(fr_mod.BACKGROUND_COLORS["black"], dtype=np.float32) / 255.0
    is_bg = np.all(np.abs(band - bg) < 0.03, axis=-1)
    return is_bg.sum()


g_45 = predict_4x5(500, 630)
img_n1, _ = run(synth_image(500, 630), format="4x5 sheet", stock_name="AKURATE 400", seed=1)
_, _, m45 = with_margin(g_45)
cuts_a = has_notch_cuts(img_n1, g_45, m45)
check("notch cuts present near top-right corner", cuts_a > 5, f"bg pixels in notch zone={cuts_a}")

idx_a = _hash_index("AKURATE 400", "notch", len(NOTCH_PATTERNS))
idx_same = _hash_index("AKURATE 400", "notch", len(NOTCH_PATTERNS))
found_diff = False
for alt in ["Kodak Fictional 100", "Ilford Invented HP", "Fuji Placeholder 200", "Ferrania Nonexistent"]:
    if _hash_index(alt, "notch", len(NOTCH_PATTERNS)) != idx_a:
        found_diff = True
        alt_name = alt
        break
check("stock_name hash: same name -> same pattern index",
      idx_a == idx_same, f"{idx_a} vs {idx_same}")
check("stock_name hash: a different name maps to a different pattern (exists)",
      found_diff, f"base idx={idx_a}")

if found_diff:
    img_n2, _ = run(synth_image(500, 630), format="4x5 sheet", stock_name=alt_name, seed=1)
    check("different stock_name -> visibly different notch region",
          not np.array_equal(img_n1, img_n2), f"images_equal={np.array_equal(img_n1, img_n2)}")

img_n3, _ = run(synth_image(500, 630), format="4x5 sheet", stock_name="AKURATE 400", seed=1)
check("same stock_name -> identical notch region (deterministic)",
      np.array_equal(img_n1, img_n3), f"images_equal={np.array_equal(img_n1, img_n3)}")

# ===========================================================================
print("\n[6] Frame numbering (batch increment)")


def edge_print_region(img_out, g, margin):
    x0 = margin + g["ap_x0"]
    y1 = margin + g["ap_y0"] + g["ap_h"]
    y_hole = margin + round(2.0 * g["px_per_mm"])  # top of bottom-perf zone approx
    y_bot = margin + g["canvas_h"] - round(2.0 * g["px_per_mm"])
    return img_out[y1:max(y1 + 1, y_bot), x0:x0 + g["ap_w"]]


imgs3 = [synth_image(500, 340, rng=np.random.default_rng(k)) for k in range(3)]
g_num = predict_135_full(500, 340)
_, _, m_num = with_margin(g_num)

outs_inc, _ = run_batch(imgs3, format="135 full frame", frame_number=7, increment_per_frame=True,
                         edge_print_intensity=0.9, strip_context=2.0, seed=1)
outs_flat, _ = run_batch(imgs3, format="135 full frame", frame_number=7, increment_per_frame=False,
                          edge_print_intensity=0.9, strip_context=2.0, seed=1)

r0i = edge_print_region(outs_inc[0], g_num, m_num)
r1i = edge_print_region(outs_inc[1], g_num, m_num)
r0f = edge_print_region(outs_flat[0], g_num, m_num)
r1f = edge_print_region(outs_flat[1], g_num, m_num)

# isolate the numbering column only (right-aligned text) to avoid the photo
# content itself (which differs per-frame regardless) polluting the diff --
# compare just the bottom-rebate strip pixels EXCLUDING the aperture (edge
# print region above already excludes the aperture: y1..canvas_h is rebate).
diff_inc = np.abs(r0i.astype(np.float32) - r1i.astype(np.float32)).sum()
diff_flat = np.abs(r0f.astype(np.float32) - r1f.astype(np.float32)).sum()
check("increment_per_frame=True: edge-print region differs frame 0 vs frame 1",
      diff_inc > 0.5, f"diff_sum={diff_inc:.3f}")
check("increment_per_frame=False: edge-print region identical frame 0 vs frame 1",
      diff_flat < 1e-6, f"diff_sum={diff_flat:.6f}")

# ===========================================================================
print("\n[7] Resolution independence")
g512 = predict_135_full(512, 340)
g2048 = predict_135_full(2048, 1360)
img512, _ = run(synth_image(512, 340), format="135 full frame", seed=1)
img2048, _ = run(synth_image(2048, 1360), format="135 full frame", seed=1)
ratio_w = img2048.shape[1] / img512.shape[1]
ratio_h = img2048.shape[0] / img512.shape[0]
check("canvas dims scale ~4x (512->2048, +-1%)",
      abs(ratio_w - 4.0) / 4.0 <= 0.01 and abs(ratio_h - 4.0) / 4.0 <= 0.01,
      f"ratio_w={ratio_w:.4f} ratio_h={ratio_h:.4f}")

# ===========================================================================
print("\n[8] Determinism + perf")
d1, m1_ = run(synth_image(600, 400), format="4x5 sheet", seed=77)
d2, m2_ = run(synth_image(600, 400), format="4x5 sheet", seed=77)
check("determinism: identical inputs+seed -> bit-exact image", np.array_equal(d1, d2))
check("determinism: identical inputs+seed -> bit-exact mask", np.array_equal(m1_, m2_))

img1k = np.random.rand(1024, 1024, 3).astype(np.float32)
_ = run(img1k, format="135 full frame")  # warm
t0 = time.time()
_ = run(img1k, format="135 full frame")
t_1k = time.time() - t0
warn_1k = " *** WARN: over 1s budget ***" if t_1k > 1.0 else ""
print(f"  1024x1024 (135 full frame): {t_1k:.3f} s{warn_1k}")

img4k = np.random.rand(2160, 3840, 3).astype(np.float32)
t0 = time.time()
_ = run(img4k, format="4x5 sheet")
t_4k = time.time() - t0
warn_4k = " *** WARN: over 1s budget ***" if t_4k > 1.0 else ""
print(f"  3840x2160 (4x5 sheet): {t_4k:.3f} s{warn_4k}")

# ===========================================================================
print("\n[9] EYE renders")
SPIKE = os.path.join(os.path.dirname(PACK_ROOT), "_film_rebate_spike")
os.makedirs(SPIKE, exist_ok=True)
PHOTO = os.path.join(os.path.dirname(PACK_ROOT), "_sabattier_spike", "00_original.png")


def save(img, name):
    Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8)).save(os.path.join(SPIKE, name))
    print(f"  saved {name}")


if os.path.exists(PHOTO):
    photo = np.asarray(Image.open(PHOTO).convert("RGB"), dtype=np.float32) / 255.0
    save(photo, "EYE_00_original.png")

    eye_formats = [
        ("EYE_01_135_full_frame.png", dict(format="135 full frame")),
        ("EYE_02_135_filed_carrier.png", dict(format="135 filed carrier")),
        ("EYE_03_120_6x6.png", dict(format="120 6x6")),
        ("EYE_04_120_6x9.png", dict(format="120 6x9")),
        ("EYE_05_4x5_sheet.png", dict(format="4x5 sheet")),
        ("EYE_06_polaroid.png", dict(format="Polaroid")),
        ("EYE_07_slide_mount.png", dict(format="Slide mount", date_text="MAY 74")),
    ]
    for name, kw in eye_formats:
        out, _ = run(photo, seed=42, stock_name="AKURATE 400", frame_number=7, **kw)
        save(out, name)

    polarity_variants = [
        ("EYE_08_135_full_c41.png", "Color neg"),
        ("EYE_08b_135_full_bw_neg.png", "B&W neg"),
        ("EYE_09_135_full_reversal.png", "Reversal"),
    ]
    for name, film_type in polarity_variants:
        out, _ = run(photo, format="135 full frame", film_type=film_type, seed=42,
                      stock_name="AKURATE 400", frame_number=7)
        save(out, name)
else:
    print(f"  (photo not found at {PHOTO}, skipped)")

# ---------------------------------------------------------------------------
print("\n" + ("ALL CHECKS PASSED" if PASS else "SOME CHECKS FAILED"))
sys.exit(0 if PASS else 1)
