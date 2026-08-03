"""
Diagnose where the residual error between our RAW pipeline and ACR lives.

Session 3 left us with mean error +0.005 but per-pixel MAE 0.047.
The 0.047 is suspicious — the XMP reveals ACR's reference had sharpening (40/+1/25),
color NR (25), and lateral CA correction ON. Those are user-space effects we
deliberately don't replicate.

This script:
  1. Renders test.RAF through our pipeline (full-size).
  2. Aligns against ACR's test.PNG by cropping 12 px per side (ACR active-area crop).
  3. Reports full-res MAE.
  4. Downsamples both to a succession of sizes to separate high-frequency (sharpening /
     NR / demosaic) error from low-frequency (calibration / curve / matrix) error.
  5. Breaks MAE down by luminance band and by edge vs flat classification.
  6. Writes an error heatmap PNG.

Run from repo root:
  F:/ComfyUI_windows_portable_nvidia/ComfyUI_windows_portable/python_embeded/python.exe \\
      ComfyUI-Darkroom/tools/diagnose_acr_match.py
"""

import os
import sys
import time

import numpy as np
from PIL import Image

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from utils.raw_loader import load_raw  # noqa: E402

TEST_RAF = os.path.join(REPO_ROOT, "test_data", "test.RAF")
ACR_PNG  = os.path.join(REPO_ROOT, "test_data", "test.PNG")
OUT_DIR  = os.path.join(REPO_ROOT, "test_data")


def load_png_float(path):
    img = Image.open(path)
    arr = np.asarray(img)
    if arr.dtype == np.uint16:
        return arr.astype(np.float32) / 65535.0
    if arr.dtype == np.uint8:
        return arr.astype(np.float32) / 255.0
    raise RuntimeError(f"Unexpected dtype {arr.dtype} for {path}")


def align_crops(ours, acr):
    """
    ACR crops 12 px from each side of the sensor's active area by default.
    rawpy returns the full sensor area. Trim ours to match.
    If dimensions still differ by a few pixels, center-crop to the shared extent.
    """
    oh, ow = ours.shape[:2]
    ah, aw = acr.shape[:2]

    # Trim 12 px per side from ours first
    ours_trim = ours[12:-12, 12:-12]
    th, tw = ours_trim.shape[:2]

    if (th, tw) == (ah, aw):
        return ours_trim, acr

    # Fall back to common center crop
    h = min(th, ah)
    w = min(tw, aw)
    def ccrop(img, h, w):
        ih, iw = img.shape[:2]
        y0 = (ih - h) // 2
        x0 = (iw - w) // 2
        return img[y0:y0+h, x0:x0+w]
    return ccrop(ours_trim, h, w), ccrop(acr, h, w)


def downsample(img, long_edge):
    h, w = img.shape[:2]
    scale = long_edge / max(h, w)
    if scale >= 1.0:
        return img
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    chans = []
    for c in range(img.shape[2]):
        pil_c = Image.fromarray(img[..., c].astype(np.float32), mode="F")
        chans.append(np.asarray(pil_c.resize((new_w, new_h), Image.LANCZOS)))
    return np.stack(chans, axis=-1).astype(np.float32)


def mae(a, b):
    return float(np.mean(np.abs(a - b)))


def per_channel_mae(a, b):
    d = np.abs(a - b)
    return [float(np.mean(d[..., c])) for c in range(3)]


def mean_signed(a, b):
    d = a - b
    return [float(np.mean(d[..., c])) for c in range(3)]


def edge_mask(img, threshold=0.03):
    """Sobel gradient magnitude > threshold on sRGB luminance."""
    y = 0.2126 * img[..., 0] + 0.7152 * img[..., 1] + 0.0722 * img[..., 2]
    gy = np.zeros_like(y)
    gy[1:-1, :] = y[2:, :] - y[:-2, :]
    gx = np.zeros_like(y)
    gx[:, 1:-1] = y[:, 2:] - y[:, :-2]
    mag = np.sqrt(gx * gx + gy * gy)
    return mag > threshold


def luminance_bands(img):
    y = 0.2126 * img[..., 0] + 0.7152 * img[..., 1] + 0.0722 * img[..., 2]
    shadow = y < 0.25
    mid    = (y >= 0.25) & (y < 0.70)
    high   = y >= 0.70
    return shadow, mid, high


def sat_bands(img):
    mx = img.max(axis=-1)
    mn = img.min(axis=-1)
    sat = np.where(mx > 1e-6, (mx - mn) / np.maximum(mx, 1e-6), 0.0)
    low = sat < 0.10
    midsat = (sat >= 0.10) & (sat < 0.35)
    high = sat >= 0.35
    return low, midsat, high


def error_heatmap(ours, acr, path):
    d = np.mean(np.abs(ours - acr), axis=-1)
    # Gamma-stretch to emphasise small differences
    vis = np.clip(d / 0.10, 0.0, 1.0)  # 0.0 → black, 0.10+ → white
    out = (vis * 255.0).astype(np.uint8)
    Image.fromarray(out).save(path)


def main():
    print(f"[diagnose] ours = {TEST_RAF}")
    print(f"[diagnose] acr  = {ACR_PNG}")

    t0 = time.time()
    print("[diagnose] rendering ours (full-size, sRGB display)...")
    ours, meta = load_raw(
        TEST_RAF,
        output_mode="sRGB display",
        baseline_exposure=0.0,
        half_size=False,
    )
    print(f"[diagnose] ours rendered in {time.time()-t0:.1f}s, shape={ours.shape}, dcp={meta.get('dcp_profile')}")

    t1 = time.time()
    print("[diagnose] loading ACR reference PNG...")
    acr = load_png_float(ACR_PNG)
    print(f"[diagnose] acr loaded in {time.time()-t1:.1f}s, shape={acr.shape}")

    ours_a, acr_a = align_crops(ours, acr)
    print(f"[diagnose] aligned: ours={ours_a.shape}, acr={acr_a.shape}")

    print()
    print("=== Full-resolution ===")
    print(f"  MAE total       = {mae(ours_a, acr_a):.4f}")
    print(f"  MAE per-channel = {per_channel_mae(ours_a, acr_a)}")
    print(f"  mean signed     = {mean_signed(ours_a, acr_a)}")

    for le in (4096, 2048, 1024, 512, 256):
        os_ = downsample(ours_a, le)
        as_ = downsample(acr_a,  le)
        if os_.shape != as_.shape:
            h = min(os_.shape[0], as_.shape[0])
            w = min(os_.shape[1], as_.shape[1])
            os_ = os_[:h, :w]
            as_ = as_[:h, :w]
        print(f"=== Downsampled to {le}px long edge ===")
        print(f"  MAE total       = {mae(os_, as_):.4f}")
        print(f"  MAE per-channel = {per_channel_mae(os_, as_)}")

    # Edge vs flat at 2048 — kills sub-pixel shift effects
    os_ = downsample(ours_a, 2048)
    as_ = downsample(acr_a,  2048)
    if os_.shape != as_.shape:
        h = min(os_.shape[0], as_.shape[0])
        w = min(os_.shape[1], as_.shape[1])
        os_ = os_[:h, :w]
        as_ = as_[:h, :w]

    em = edge_mask(as_)
    print()
    print("=== Edge vs flat (at 2048px) ===")
    print(f"  Edge pixels   : {em.mean()*100:.1f}% — MAE {np.mean(np.abs(os_[em] - as_[em])):.4f}")
    print(f"  Flat pixels   : {(~em).mean()*100:.1f}% — MAE {np.mean(np.abs(os_[~em] - as_[~em])):.4f}")

    shadow, mid, high = luminance_bands(as_)
    print()
    print("=== Luminance bands (at 2048px) ===")
    for name, m in (("shadow<0.25", shadow), ("mid 0.25-0.70", mid), ("high>=0.70", high)):
        if m.sum() == 0:
            continue
        print(f"  {name:<16} {m.mean()*100:5.1f}% — MAE {np.mean(np.abs(os_[m] - as_[m])):.4f}  signed {[float(np.mean(os_[m][...,c] - as_[m][...,c])) for c in range(3)]}")

    lo, msat, hi = sat_bands(as_)
    print()
    print("=== Saturation bands (at 2048px) ===")
    for name, m in (("sat<0.10", lo), ("sat 0.10-0.35", msat), ("sat>=0.35", hi)):
        if m.sum() == 0:
            continue
        print(f"  {name:<16} {m.mean()*100:5.1f}% — MAE {np.mean(np.abs(os_[m] - as_[m])):.4f}  signed {[float(np.mean(os_[m][...,c] - as_[m][...,c])) for c in range(3)]}")

    heat_path = os.path.join(OUT_DIR, "error_heatmap.png")
    error_heatmap(downsample(ours_a, 2048), downsample(acr_a, 2048), heat_path)
    print(f"\n[diagnose] wrote heatmap -> {heat_path}")


if __name__ == "__main__":
    main()
