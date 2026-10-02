"""
Bake the Capture One Film Styles behind Darkroom's colour stocks into 3D LUTs
(decisions.md 2026-10-02). Uses the sidecar-injection route of
tools/c1_reference_kit.py: one Hald grid image per style, C1 renders it, the
export IS the LUT.

Only the GLOBAL tools are baked (curves, levels, colour balance, saturation,
contrast, colour editor). Left out:
  * Shadow/HighlightRecovery and Clarity: local/spatial in C1, they would
    smear across the grid. Recovery gets its own fitted global model.
  * Grain and sharpening: not part of the colour look.

Also writes "nolocal" validation variants of the reference chart/photo (same
keys as the bake), so applying a baked LUT can be checked against C1 directly.

Usage:
  python tools/c1_lut_bake.py prepare <kit_dir> <chart.tif> <photo.tif> [grid_n=65]
  python tools/c1_lut_bake.py inject  <kit_dir>
  python tools/c1_lut_bake.py collect <kit_dir> <export_dir> <out_npz>
"""

import glob
import importlib.util
import json
import os
import re
import shutil
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("ref_kit", os.path.join(HERE, "c1_reference_kit.py"))
kit = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(kit)

LOCAL_KEYS = {"ShadowRecovery", "HighlightRecovery", "HighlightRecoveryEx",
              "Clarity", "ClarityStructure", "ClarityMethod"}


def stock_map():
    """Darkroom colour-stock key -> C1 .costyle path, from the generator's tables."""
    spec = importlib.util.spec_from_file_location("gs", os.path.join(HERE, "generate_stocks.py"))
    g = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(g)
    files = {os.path.splitext(os.path.basename(f))[0]: f
             for f in glob.glob(os.path.join(kit.STYLES_DIR, "*", "*.costyle"))}
    out = {}
    for name in dir(g):
        d = getattr(g, name)
        if isinstance(d, dict):
            for c1, v in d.items():
                if isinstance(v, tuple) and v and isinstance(v[0], str) and "/" in v[0] and c1 in files:
                    out[v[0]] = files[c1]
    return dict(sorted(out.items()))


def global_keys(path):
    s = open(path, encoding="utf-8", errors="ignore").read()
    return {k: v for k, v in re.findall(r'<E K="([^"]+)" V="([^"]*)"', s)
            if k not in kit.SKIP and k not in LOCAL_KEYS}


def safe(key):
    return re.sub(r"[^A-Za-z0-9]+", "_", key).strip("_")


TILES_X = 9   # blue slices laid out as a 9-wide mosaic of n x n tiles


def hald(n):
    """Near-square mosaic: tile b (row b // 9, col b % 9) holds colours with
    blue = b/(n-1); inside a tile, pixel (y, x) = (red r, green g).
    Capture One silently skips processing very thin images (a 65 x 4225 strip
    came back untouched), so the grid must be roughly square."""
    v = np.linspace(0.0, 1.0, n)
    rows = -(-n // TILES_X)
    out = np.zeros((rows * n, TILES_X * n, 3))
    r, g = np.meshgrid(v, v, indexing="ij")
    for b in range(n):
        ty, tx = divmod(b, TILES_X)
        out[ty * n:(ty + 1) * n, tx * n:(tx + 1) * n] = np.stack([r, g, np.full_like(r, v[b])], -1)
    return out


def unhald(img, n):
    """Inverse of hald(): mosaic image -> LUT indexed [r, g, b]."""
    lut = np.zeros((n, n, n, 3), dtype=np.float32)
    for b in range(n):
        ty, tx = divmod(b, TILES_X)
        lut[:, :, b] = img[ty * n:(ty + 1) * n, tx * n:(tx + 1) * n]
    return lut


def write_tif(path, arr01):
    import tifffile
    from PIL import ImageCms
    icc = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()
    a = np.round(np.clip(arr01, 0, 1) * 65535).astype(np.uint16)
    tifffile.imwrite(path, a, photometric="rgb", extratags=[(34675, 7, len(icc), icc, True)])


def plan():
    """variant name -> keys, for every file of the bake kit."""
    out = {"hald__identity": {}}   # control: an unstyled grid must come back unchanged
    for key, path in stock_map().items():
        out[f"hald__{safe(key)}"] = global_keys(path)
    for name, rel in kit.REFERENCE_STYLES.items():
        out[f"nolocal_{name}"] = global_keys(os.path.join(kit.STYLES_DIR, rel))
    return out


def prepare(kitdir, chart, photo, n=65):
    os.makedirs(kitdir, exist_ok=True)
    grid = hald(n)
    p = plan()
    for v in p:
        if v.startswith("hald__"):
            write_tif(os.path.join(kitdir, f"{v}.tif"), grid)
        else:
            for img in (chart, photo):
                stem = os.path.splitext(os.path.basename(img))[0]
                shutil.copyfile(img, os.path.join(kitdir, f"{stem}__{v}.tif"))
    json.dump({"grid_n": n, "stocks": {safe(k): k for k in stock_map()},
               "files": {k: os.path.basename(v) for k, v in stock_map().items()}},
              open(os.path.join(kitdir, "bake_manifest.json"), "w"), indent=1)
    print(f"{len(os.listdir(kitdir)) - 1} images in {kitdir} (grid {n}^3). "
          "Open the folder in the C1 session once, quit C1, then run inject.")


def inject(kitdir):
    sdir = kit._sidecar_dir(kitdir)
    p = plan()
    done = 0
    for cos in sorted(glob.glob(os.path.join(sdir, "*.tif.cos"))):
        base = os.path.basename(cos)[:-4]
        stem = os.path.splitext(base)[0]
        v = stem if stem.startswith("hald__") else stem.split("__", 1)[1]
        if v not in p:
            continue
        s = open(cos, encoding="utf-8").read()
        m = re.search(r"(<DL>)(.*?)(\s*</DL>)", s, re.S)
        body = m.group(2)
        added = []
        for k, val in p[v].items():
            pat = rf'(<E K="{re.escape(k)}" V=")[^"]*("\s*/>)'
            if re.search(pat, body):
                body = re.sub(pat, lambda mm: mm.group(1) + val + mm.group(2), body, count=1)
            else:
                added.append("\n\t\t\t" + f'<E K="{k}" V="{val}" />')
        s = s[:m.start(2)] + body + "".join(added) + s[m.end(2):]
        if not os.path.isfile(cos + ".orig"):
            shutil.copyfile(cos, cos + ".orig")
        open(cos, "w", encoding="utf-8").write(s)
        done += 1
    print(f"injected {done} sidecars. Reopen the session, select all, export (TIFF 16-bit sRGB).")


def collect(kitdir, export_dir, out_npz):
    """Read the exported Hald grids back into (S, n, n, n, 3) LUTs indexed [r, g, b]."""
    import tifffile
    man = json.load(open(os.path.join(kitdir, "bake_manifest.json")))
    n = man["grid_n"]
    luts, names = [], []
    for sk, key in sorted(man["stocks"].items(), key=lambda kv: kv[1]):
        f = os.path.join(export_dir, f"hald__{sk}.tif")
        lut = unhald(tifffile.imread(f)[..., :3].astype(np.float32) / 65535.0, n)
        luts.append(lut)
        names.append(key)
    np.savez_compressed(out_npz, luts=np.stack(luts).astype(np.float32), names=np.array(names), grid_n=n)
    print(f"{len(names)} LUTs ({n}^3) -> {out_npz}")


def pack(kitdir, luts65, model, out):
    """65^3 float LUTs -> shipped data: 33^3 uint16 (every 2nd node of the 65 grid,
    exact), each style's recovery values, and the fitted recovery model
    (tools/c1_fit_recovery.py)."""
    z = np.load(luts65)
    names = [str(n) for n in z["names"]]
    l33 = z["luts"][:, ::2, ::2, ::2]
    man = json.load(open(os.path.join(kitdir, "bake_manifest.json")))
    files = {os.path.basename(f): f for f in glob.glob(os.path.join(kit.STYLES_DIR, "*", "*.costyle"))}
    sr, hr = [], []
    for n in names:
        st = open(files[man["files"][n]], encoding="utf-8", errors="ignore").read()
        sr.append(float((re.search(r'K="ShadowRecovery" V="([^"]*)"', st) or [0, 0])[1]))
        hr.append(float((re.search(r'K="HighlightRecovery" V="([^"]*)"', st) or [0, 0])[1]))
    rec = np.load(model)
    np.savez_compressed(out, luts=np.round(np.clip(l33, 0, 1) * 65535).astype(np.uint16),
                        names=np.array(names), shadow_recovery=np.array(sr, np.float32),
                        highlight_recovery=np.array(hr, np.float32),
                        recovery_knots=rec["knots"].astype(np.float32),
                        recovery_blur_frac=np.float32(rec["blur_frac"]))
    print(f"{len(names)} stocks -> {out}")


if __name__ == "__main__":
    a = sys.argv
    if len(a) >= 5 and a[1] == "prepare":
        prepare(a[2], a[3], a[4], int(a[5]) if len(a) > 5 else 65)
    elif len(a) == 3 and a[1] == "inject":
        inject(a[2])
    elif len(a) == 5 and a[1] == "collect":
        collect(a[2], a[3], a[4])
    elif len(a) == 6 and a[1] == "pack":
        pack(a[2], a[3], a[4], a[5])
    else:
        print(__doc__)
