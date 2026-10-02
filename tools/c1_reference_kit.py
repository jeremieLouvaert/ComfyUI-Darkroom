"""
Capture One reference kit for the film-stock rebuild (docs/film-stock-c1-derivation.md).

Capture One on Windows has no scripting API, but it stores each image's
adjustments in an XML sidecar (<folder>/CaptureOne/Settings*/<image>.cos) whose
adjustment keys are the same as a .costyle's. So:

  1. `prepare`: copy the test images once per variant into a kit folder
     (file name = variant name, so the export is self-describing).
  2. Jeremie opens the kit folder in a Capture One SESSION once and closes it;
     C1 writes a sidecar per image.
  3. `inject`: write each variant's adjustment keys into its sidecar.
  4. Jeremie reopens, selects all, exports once (16-bit TIFF, sRGB).

Variants: an unstyled baseline, each reference style in full, and for the probe
style each C1 tool on its own (the rest of the style removed), which is what
lets the derivation fit one tool at a time.

Usage:
  python tools/c1_reference_kit.py prepare <kit_dir> <image> [<image> ...]
  python tools/c1_reference_kit.py inject  <kit_dir>
"""

import glob
import os
import re
import shutil
import sys

STYLES_DIR = r"C:\Users\Jeremie\Downloads\CaptureOne_FilmStyles-master\CaptureOne_FilmStyles-master\styles"
REFERENCE_STYLES = {
    "Portra400NC": r"Kodak\Kodak Portra 400 NC.costyle",
    "Portra160VC": r"Kodak\Kodak Portra 160 VC.costyle",
    "Superia400": r"Fuji\Fuji Superia 400.costyle",
    "Velvia50": r"Fuji\Fuji Velvia 50.costyle",
    "Ektar100": r"Kodak\Kodak Ektar 100.costyle",
    "Vista400": r"Agfa\Agfa Vista 400.costyle",
}
PROBE_STYLE = "Portra400NC"

# C1 tool -> the style keys that belong to it (everything else is left at default)
TOOLS = {
    "curve": ["GradationCurve", "GradationCurveRed", "GradationCurveGreen",
              "GradationCurveBlue", "GradationCurveY"],
    "levels": ["Shadow", "Midtone", "Highlight", "TargetShadow", "TargetHighlight"],
    "balance": ["ColorBalanceShadow", "ColorBalanceMidtone", "ColorBalanceHighlight"],
    "coloreditor": ["ColorCorrections"],
    "hdr": ["ShadowRecovery", "HighlightRecovery"],
    "saturation": ["Saturation", "Contrast"],
}
# never injected: identity/metadata, grain and sharpening (not part of the stock look)
SKIP = {"Name", "UUID", "StyleSource", "FilmGrainAmount", "FilmGrainDensity",
        "FilmGrainGranularity", "FilmGrainType", "UsmAmount", "UsmRadius"}


def style_keys(rel):
    s = open(os.path.join(STYLES_DIR, rel), encoding="utf-8", errors="ignore").read()
    return {k: v for k, v in re.findall(r'<E K="([^"]+)" V="([^"]*)"', s) if k not in SKIP}


def variants():
    """name -> {key: value} for every kit variant."""
    out = {"baseline": {}}
    for name, rel in REFERENCE_STYLES.items():
        out[f"style_{name}"] = style_keys(rel)
    probe = style_keys(REFERENCE_STYLES[PROBE_STYLE])
    for tool, keys in TOOLS.items():
        sub = {k: probe[k] for k in keys if k in probe}
        if sub:
            out[f"tool_{PROBE_STYLE}_{tool}"] = sub
    return out


def prepare(kit, images):
    os.makedirs(kit, exist_ok=True)
    n = 0
    for img in images:
        stem, ext = os.path.splitext(os.path.basename(img))
        for v in variants():
            if v.startswith("tool_") and "chart" not in stem:
                continue  # tool isolation only needed on the chart
            shutil.copyfile(img, os.path.join(kit, f"{stem}__{v}{ext}"))
            n += 1
    print(f"{n} files in {kit}. Open this folder in a Capture One session once, close C1, then run inject.")


def _sidecar_dir(kit):
    dirs = sorted(glob.glob(os.path.join(kit, "CaptureOne", "Settings*")))
    if not dirs:
        sys.exit("No CaptureOne/Settings* folder yet: open the kit folder in a Capture One session first.")
    return dirs[-1]


def inject(kit):
    sdir = _sidecar_dir(kit)
    var = variants()
    done = missing = 0
    for path in sorted(glob.glob(os.path.join(kit, "*__*.*"))):
        base = os.path.basename(path)
        v = os.path.splitext(base)[0].split("__", 1)[1]
        if v not in var:
            continue  # e.g. the hand-styled control image
        cos = os.path.join(sdir, base + ".cos")
        if not os.path.isfile(cos):
            print(f"  no sidecar for {base}")
            missing += 1
            continue
        s = open(cos, encoding="utf-8").read()
        m = re.search(r'(<DL>)(.*?)(\s*</DL>)', s, re.S)
        if not m:
            print(f"  no <DL> adjustment block in {cos}")
            missing += 1
            continue
        body = m.group(2)
        added = []
        for k, val in var[v].items():
            pat = rf'(<E K="{re.escape(k)}" V=")[^"]*("\s*/>)'
            if re.search(pat, body):
                body = re.sub(pat, lambda mm: mm.group(1) + val + mm.group(2), body, count=1)
            else:
                added.append(f'\n\t\t\t<E K="{k}" V="{val}" />')
                print(f"  {base}: key {k} not in C1's defaults, added (may be ignored by this engine)")
        s = s[:m.start(2)] + body + "".join(added) + s[m.end(2):]
        if not os.path.isfile(cos + ".orig"):
            shutil.copyfile(cos, cos + ".orig")
        open(cos, "w", encoding="utf-8").write(s)
        done += 1
    print(f"injected {done} sidecars in {sdir} ({missing} missing). Reopen the session, select all, export.")


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "prepare":
        prepare(sys.argv[2], sys.argv[3:])
    elif len(sys.argv) == 3 and sys.argv[1] == "inject":
        inject(sys.argv[2])
    else:
        print(__doc__)
