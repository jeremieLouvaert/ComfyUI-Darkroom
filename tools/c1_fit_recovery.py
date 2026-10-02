"""
Fit the global stand-in for Capture One's (local) Shadow/Highlight Recovery used by
utils/film_luts.py (docs/film-stock-c1-derivation.md).

Model: gain = 1 + SR*S(Lb) - HR*H(Lb), applied before the baked LUT, where Lb is
luminance blurred at a fraction of the long edge and S, H are 9-knot curves shared
by all styles. Fitted with Adam (torch autograd through gpu_color's LUT) on the
reference styles' full C1 renders vs the baked LUTs; picks the blur size, then
reports leave-one-style-out error.

Usage:
  python tools/c1_fit_recovery.py <luts_65.npz> <kit_bake_dir> <full_renders_dir> <out_model.npz>

  luts_65.npz       from `c1_lut_bake.py collect`
  kit_bake_dir      the bake kit (holds bake_manifest.json); its parent holds chart.tif / photo_nb2.tif
  full_renders_dir  exports of the reference kit: <img>__style_<Name>.tif (full style incl. recovery)
"""

import importlib.util
import json
import os
import re
import sys

import numpy as np
import tifffile
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
_s = importlib.util.spec_from_file_location("gc", os.path.join(os.path.dirname(HERE), "utils", "gpu_color.py"))
G = importlib.util.module_from_spec(_s)
_s.loader.exec_module(G)
_k = importlib.util.spec_from_file_location("ref_kit", os.path.join(HERE, "c1_reference_kit.py"))
kit = importlib.util.module_from_spec(_k)
_k.loader.exec_module(kit)

FRACS = (0.0, 0.005, 0.015, 0.04, 0.1)


def rd(p):
    return tifffile.imread(p)[..., :3].astype(np.float32) / 65535


def blur(y, sigma_px):
    f = max(1, int(sigma_px // 4))
    yd = F.avg_pool2d(y[None, None], f, ceil_mode=True)[0, 0] if f > 1 else y
    s = sigma_px / f
    r = max(1, int(3 * s))
    x = torch.arange(-r, r + 1, device=y.device, dtype=y.dtype)
    k = torch.exp(-0.5 * (x / s) ** 2)
    k = k / k.sum()
    t = F.pad(yd[None, None], (r, r, r, r), mode="replicate")
    t = F.conv2d(F.conv2d(t, k.view(1, 1, 1, -1)), k.view(1, 1, -1, 1))
    return F.interpolate(t, size=y.shape, mode="bilinear", align_corners=False)[0, 0]


def interp(v, kn):
    t = v.clamp(0, 1) * 8
    i = t.floor().clamp(max=7).long()
    f = t - i
    return kn[i] * (1 - f) + kn[i + 1] * f


def model(X, y, ylf, sr, hr, th, mix):
    yb = mix * ylf + (1 - mix) * y
    return (X * (1 + sr * interp(yb, th[:9]) - hr * interp(yb, th[9:]))[..., None]).clamp(0, 1)


def main(luts_path, kitdir, full_dir, out):
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    z = np.load(luts_path)
    luts, names = z["luts"], [str(n) for n in z["names"]]
    man = json.load(open(os.path.join(kitdir, "bake_manifest.json")))
    key_of = {os.path.splitext(f)[0]: k for k, f in man["files"].items()}
    refdir = os.path.dirname(os.path.normpath(kitdir))
    rng = np.random.default_rng(0)
    imgs = {}
    for img, step in (("photo_nb2", 2), ("chart", 1)):
        src = torch.tensor(rd(os.path.join(refdir, f"{img}.tif"))[::step, ::step], device=dev)
        y = G.luminance_rec709(src)
        Lp = max(src.shape[:2])
        yl = {f: (blur(y, f * Lp) if f > 0 else y) for f in FRACS}
        idx = (torch.tensor(rng.integers(0, src.shape[0], 60000), device=dev),
               torch.tensor(rng.integers(0, src.shape[1], 60000), device=dev))
        imgs[img] = (step, idx, src[idx], y[idx], {f: v[idx] for f, v in yl.items()})
    data = []
    for short, rel in kit.REFERENCE_STYLES.items():
        st = open(os.path.join(kit.STYLES_DIR, rel), encoding="utf-8", errors="ignore").read()
        g = lambda k: float((re.search(rf'K="{k}" V="([^"]*)"', st) or [0, 0])[1])
        c1 = os.path.splitext(os.path.basename(rel))[0]
        lut = G.lut_to_device(luts[names.index(key_of[c1])][::2, ::2, ::2].copy(), dev)
        for img, (step, idx, X, y, yl) in imgs.items():
            full = torch.tensor(rd(os.path.join(full_dir, f"{img}__style_{short}.tif"))[::step, ::step], device=dev)[idx]
            data.append((short, img, g("ShadowRecovery") / 100, g("HighlightRecovery") / 100, lut, X, y, yl, full))

    def run(frac, mix, use, steps=400):
        th = torch.zeros(18, device=dev, requires_grad=True)
        opt = torch.optim.Adam([th], lr=0.05)
        for _ in range(steps):
            opt.zero_grad()
            loss = torch.stack([(G.apply_lut_trilinear(model(X, y, yl[frac], sr, hr, th, mix), lut, 33) - Y).abs().mean()
                                for (n, im, sr, hr, lut, X, y, yl, Y) in data if n in use]).mean()
            loss.backward()
            opt.step()
        return th.detach(), loss.item()

    alln = list(kit.REFERENCE_STYLES)
    best = None
    for frac in FRACS:
        for mix in ((1.0, 0.6) if frac > 0 else (1.0,)):
            th, loss = run(frac, mix, alln)
            print(f"blur {frac:5.3f} x long edge, mix {mix}: train mean |err| {loss * 255:.2f}/255", flush=True)
            if best is None or loss < best[0]:
                best = (loss, frac, mix, th)
    loss, frac, mix, th = best
    print(f"BEST blur {frac} mix {mix}")
    for n in alln:
        loo, _ = run(frac, mix, [m for m in alln if m != n], 300)
        for (nn, im, sr, hr, lut, X, y, yl, Y) in data:
            if nn != n:
                continue
            with torch.no_grad():
                b = (G.apply_lut_trilinear(X, lut, 33) - Y).abs() * 255
                e = (G.apply_lut_trilinear(model(X, y, yl[frac], sr, hr, loo, mix), lut, 33) - Y).abs() * 255
            print(f"{n:12s} {im:9s} LUT only {b.mean():5.2f} -> +recovery (held out) mean {e.mean():.2f}")
    np.savez(out, knots=th.cpu().numpy(), blur_frac=frac, mix=mix)
    print(f"saved {out}")


if __name__ == "__main__":
    if len(sys.argv) != 5:
        print(__doc__)
        sys.exit(1)
    main(*sys.argv[1:5])
