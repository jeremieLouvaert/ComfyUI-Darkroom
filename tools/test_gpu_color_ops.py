"""
Op-level gate for utils/gpu_color.py: every torch port against the numpy
original it replaces, on inputs built to hit the edges (exact transfer-function
thresholds, 0 and 1, out-of-range values, negatives where the numpy op allows
them, hue wrap at 0/360, ties between channels in RGB->HSL).

Gate: max |torch - numpy| <= 0.5/255 for display-range outputs; hue is compared
as an angle (wrap-aware) and only where saturation is meaningful. A negative
control perturbs one constant and must fail the gate.

Run: python.exe tools/test_gpu_color_ops.py
"""

import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
PACK = os.path.dirname(HERE)
sys.path.insert(0, os.path.dirname(PACK))
import importlib.util  # noqa: E402

for name in ("utils",):
    pass
_spec = importlib.util.spec_from_file_location(
    "dr_utils", os.path.join(PACK, "utils", "__init__.py"),
    submodule_search_locations=[os.path.join(PACK, "utils")])
_u = importlib.util.module_from_spec(_spec)
sys.modules["dr_utils"] = _u
_spec.loader.exec_module(_u)
import dr_utils.color as C  # noqa: E402
import dr_utils.grading as G  # noqa: E402
import dr_utils.raw as R  # noqa: E402
import dr_utils.colorspace as CS  # noqa: E402
import dr_utils.lut as L  # noqa: E402
import dr_utils.gpu_color as T  # noqa: E402

TOL = 0.5 / 255
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PASS = FAIL = 0


def check(name, ok, detail=""):
    global PASS, FAIL
    if ok:
        PASS += 1
    else:
        FAIL += 1
        print(f"  FAIL {name} {detail}")


def t(a):
    return torch.as_tensor(np.ascontiguousarray(a, dtype=np.float32), device=DEV)


def n(x):
    return x.detach().cpu().numpy() if torch.is_tensor(x) else np.asarray(x)


def close(name, a, b, tol=TOL):
    a, b = n(a).astype(np.float64), n(b).astype(np.float64)
    if a.shape != b.shape:
        check(name, False, f"shape {a.shape} vs {b.shape}")
        return
    if not np.isfinite(b).all():
        check(name, False, "NaN/inf in torch output")
        return
    d = float(np.nanmax(np.abs(a - b))) if a.size else 0.0
    check(name, d <= tol, f"max diff {d:.3e}")


rng = np.random.default_rng(7)
special = np.array([0.0, 1.0, 0.04045, 0.0031308, 0.18, 0.5, 1e-6, 0.999999,
                    np.nextafter(np.float32(0.04045), 1), np.nextafter(np.float32(0.0031308), 0)],
                   np.float32)
img = rng.random((48, 64, 3), dtype=np.float32)
img[0, :special.size] = special[:, None]
img[1, :10] = [[0.3, 0.3, 0.1], [0.3, 0.1, 0.3], [0.1, 0.3, 0.3], [0.5, 0.5, 0.5],
               [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [0, 1, 1], [1, 0, 1]]  # ties
wide = (rng.random((48, 64, 3), dtype=np.float32) * 3.0 - 1.0)  # out of range
hdr = rng.random((48, 64, 3), dtype=np.float32) * 12.0


def run():
    # color.py
    for x in (img, wide):
        close("srgb_to_linear", C.srgb_to_linear(x), T.srgb_to_linear(t(x)))
        close("linear_to_srgb", C.linear_to_srgb(x), T.linear_to_srgb(t(x)))
    for p in [(1.0, 1.0, 1.0), (1.4, 0.7, 1.2), (0.6, 2.2, 0.9, 0.25, 0.2)]:
        close(f"characteristic_curve{p}", C.characteristic_curve(img[..., 0], *p),
              T.characteristic_curve(t(img[..., 0]), *p))
    close("apply_per_channel_curves",
          C.apply_per_channel_curves(img, (1.2, 0.8, 1.1), (1.0, 1.0, 1.0), (0.9, 1.3, 0.95)),
          T.apply_per_channel_curves(t(img), (1.2, 0.8, 1.1), (1.0, 1.0, 1.0), (0.9, 1.3, 0.95)))
    close("luminance_rec709", C.luminance_rec709(img), T.luminance_rec709(t(img)))
    for f in (0.0, 0.5, 1.0005, 1.6):
        close(f"adjust_saturation({f})", C.adjust_saturation(img, f), T.adjust_saturation(t(img), f))
    for bal in (0.2, 0.5, 0.8):
        close(f"split_tone({bal})", C.split_tone(img, (0.05, -0.02, 0.1), (-0.03, 0.04, 0.0), bal),
              T.split_tone(t(img), (0.05, -0.02, 0.1), (-0.03, 0.04, 0.0), bal))
    for s in (0.0, 0.35, 1.0):
        close(f"blend({s})", C.blend(img, wide, s), T.blend(t(img), t(wide), s))

    # grading.py
    for pts in [[(0, 0), (1, 1)], [(0, 0.05), (0.25, 0.2), (0.5, 0.55), (0.75, 0.85), (1, 0.97)],
                [(0.1, 0.0), (0.4, 0.6), (0.9, 1.0)], [(0, 0), (0.5, 0.5), (0.5001, 0.9), (1, 1)]]:
        for x in (img[..., 1], np.linspace(-0.2, 1.2, 997, dtype=np.float32)):
            close(f"cubic_spline_curve({len(pts)} pts)", G.cubic_spline_curve(x, pts),
                  T.cubic_spline_curve(t(x), pts))
    for a, b in zip(G.lgg_zone_masks(img[..., 2]), T.lgg_zone_masks(t(img[..., 2]))):
        close("lgg_zone_masks", a, b)
    for sr, hr in ((0.15, 0.85), (0.05, 0.6)):
        for a, b in zip(G.log_zone_masks(img[..., 0], sr, hr), T.log_zone_masks(t(img[..., 0]), sr, hr)):
            close(f"log_zone_masks({sr},{hr})", a, b)
    hue = rng.random((48, 64), dtype=np.float32) * 360.0
    hue[0, :4] = [0.0, 359.999, 180.0, 45.0]
    for cen, w, sft in ((0, 45, 0.5), (350, 30, 0.0), (120, 60, 1.0), (200, 0.5, 0.2)):
        close(f"hue_range_mask({cen},{w},{sft})", G.hue_range_mask(hue, cen, w, sft),
              T.hue_range_mask(t(hue), cen, w, sft))
    for args in [((0, 0, 0), (1, 1, 1), (1, 1, 1), (0, 0, 0)),
                 ((0.1, -0.05, 0.2), (1.3, 0.8, 0.005), (1.2, 0.9, 1.1), (0.02, -0.01, 0.0))]:
        close("apply_lgg", G.apply_lgg(img, *args), T.apply_lgg(t(img), *args))
    mask = img[..., 1]
    for ang, inten in ((30, 50), (200, 0.2), (300, 100)):
        close(f"apply_color_tint_to_zone({ang},{inten})", G.apply_color_tint_to_zone(img, mask, ang, inten),
              T.apply_color_tint_to_zone(t(img), t(mask), ang, inten))
    close("sat_from_rgb", G.sat_from_rgb(img), T.sat_from_rgb(t(img)))
    close("sat_range_mask", G.sat_range_mask(img[..., 0], 0.4, 0.15), T.sat_range_mask(t(img[..., 0]), 0.4, 0.15))

    # raw.py HSL
    h, s, l = R.rgb_to_hsl(img)
    th, ts, tl = T.rgb_to_hsl(t(img))
    close("rgb_to_hsl S", s, ts)
    close("rgb_to_hsl L", l, tl)
    dh = np.abs(h - n(th))
    dh = np.minimum(dh, 360.0 - dh)
    check("rgb_to_hsl H (deg, wrap-aware)", float(dh.max()) <= 0.05, f"max {dh.max():.4f} deg")
    close("hsl_to_rgb", R.hsl_to_rgb(h, s, l), T.hsl_to_rgb(t(h), t(s), t(l)))
    close("hsl round trip", R.hsl_to_rgb(*R.rgb_to_hsl(img)), T.hsl_to_rgb(*T.rgb_to_hsl(t(img))))

    # colorspace.py
    for M in (CS._SRGB_TO_ACES, CS._ACES_TO_SRGB, CS._OKLAB_M1, CS._SRGB_TO_REC2020):
        close("apply_matrix", CS._apply_matrix(wide, M), T.apply_matrix(t(wide), M), tol=1e-5)
    lab = CS.linear_srgb_to_oklab(wide)
    close("linear_srgb_to_oklab (wide, negatives)", lab, T.linear_srgb_to_oklab(t(wide)), tol=1e-5)
    close("oklab_to_linear_srgb", CS.oklab_to_linear_srgb(lab), T.oklab_to_linear_srgb(t(lab)), tol=1e-4)
    close("oklab_to_oklch", CS.oklab_to_oklch(lab), T.oklab_to_oklch(t(lab)), tol=1e-5)
    lch = CS.oklab_to_oklch(lab)
    close("oklch_to_oklab", CS.oklch_to_oklab(lch), T.oklch_to_oklab(t(lch)), tol=1e-5)
    close("linear_to_acescct", CS.linear_to_acescct(hdr), T.linear_to_acescct(t(hdr)))
    cct = CS.linear_to_acescct(hdr)
    close("acescct_to_linear", CS.acescct_to_linear(cct), T.acescct_to_linear(t(cct)), tol=1e-3)
    for src in CS.SPACE_NAMES:
        for dst in CS.SPACE_NAMES:
            close(f"convert_colorspace {src}->{dst}", CS.convert_colorspace(img, src, dst),
                  T.convert_colorspace(t(img), src, dst))
    for name, f in CS.TONEMAP_CURVES.items():
        close(f"tonemap {name}", f(hdr), T.TONEMAP_CURVES[name](t(hdr)))

    # lut.py
    for size in (2, 17, 33):
        g = np.linspace(0, 1, size, dtype=np.float32)
        r_, g_, b_ = np.meshgrid(g, g, g, indexing="ij")
        lut = np.stack([r_ ** 0.8, 0.7 * g_ + 0.3 * b_, np.sqrt(b_) * 0.9 + 0.05 * r_], -1).astype(np.float32)
        for x in (img, wide):
            close(f"apply_lut_trilinear size {size}", L.apply_lut_trilinear(x, lut, size),
                  T.apply_lut_trilinear(t(x), T.lut_to_device(lut, DEV), size))

    # batch broadcast: (B,H,W,3) in one call == per image
    batch = t(np.stack([img, np.clip(wide, 0, 1)]))
    per = torch.stack([T.srgb_to_linear(batch[0]), T.srgb_to_linear(batch[1])])
    check("batch broadcast", torch.equal(T.srgb_to_linear(batch), per))


def negative_control():
    """The gate must catch a change worth ~2/255 (sRGB exponent 2.4 -> 2.45).
    (2.41 moves outputs by <= 0.0015, under the 0.5/255 budget, so it is
    correctly NOT caught.)"""
    global PASS, FAIL
    saved = (PASS, FAIL)
    orig = T.srgb_to_linear
    T.srgb_to_linear = lambda x: torch.where(x.clamp(0, 1) <= 0.04045, x.clamp(0, 1) / 12.92,
                                             ((x.clamp(0, 1) + 0.055) / 1.055) ** 2.45)
    close("NC", C.srgb_to_linear(img), T.srgb_to_linear(t(img)))
    fired = FAIL > saved[1]
    T.srgb_to_linear = orig
    PASS, FAIL = saved
    check("negative control fires (2.4 -> 2.45 caught)", fired)


if __name__ == "__main__":
    print(f"device {DEV}, gate {TOL:.6f}")
    run()
    negative_control()
    print(f"\n{PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)
