"""
Extract the actual transfer function ACR applies on top of our DCP-corrected output.

Approach:
  1. Render ours at full-size through the standard pipeline (DCP + current user curve
     + sRGB encode).
  2. Inverse our user curve on the linearized render, recovering per-pixel post-DCP
     linear values.
  3. Linearize the ACR reference.
  4. Downsample both to 2048 px long edge to kill sharpening halos.
  5. Per channel, build a binned-median scatter of (post-DCP ours) -> (ACR linear).
  6. Plot the recovered curve against our current stopgap curve.
  7. Also plot residual-after-session3-curve to check symmetry.

This does NOT fit a new curve yet. It reveals the target shape first so we can decide
whether a 1D curve is sufficient or we need HSV-aware correction.
"""

import os
import sys
import time

import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.interpolate import PchipInterpolator

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from utils.raw_loader import load_raw, CAMERA_RAW_DEFAULT_CURVE  # noqa: E402
from utils.color import srgb_to_linear  # noqa: E402

TEST_RAF = os.path.join(REPO_ROOT, "test_data", "test.RAF")
ACR_PNG  = os.path.join(REPO_ROOT, "test_data", "test.PNG")
OUT_DIR  = os.path.join(REPO_ROOT, "test_data")


def load_png_float(path):
    arr = np.asarray(Image.open(path))
    if arr.dtype == np.uint16:
        return arr.astype(np.float32) / 65535.0
    if arr.dtype == np.uint8:
        return arr.astype(np.float32) / 255.0
    raise RuntimeError(f"Unexpected dtype {arr.dtype}")


def align_crops(ours, acr):
    ours_trim = ours[12:-12, 12:-12]
    th, tw = ours_trim.shape[:2]
    ah, aw = acr.shape[:2]
    if (th, tw) == (ah, aw):
        return ours_trim, acr
    h = min(th, ah)
    w = min(tw, aw)
    def ccrop(img):
        ih, iw = img.shape[:2]
        y0 = (ih - h) // 2
        x0 = (iw - w) // 2
        return img[y0:y0+h, x0:x0+w]
    return ccrop(ours_trim), ccrop(acr)


def downsample(img, long_edge):
    h, w = img.shape[:2]
    scale = long_edge / max(h, w)
    if scale >= 1.0:
        return img
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    # Float per-channel resize via PIL mode "F" — preserves precision, avoids
    # the uint16-RGB fromarray limitation.
    chans = []
    for c in range(img.shape[2]):
        pil_c = Image.fromarray(img[..., c].astype(np.float32), mode="F")
        chans.append(np.asarray(pil_c.resize((new_w, new_h), Image.LANCZOS)))
    return np.stack(chans, axis=-1).astype(np.float32)


def make_curve_and_inverse(knots):
    xs = np.array([p[0] for p in knots], dtype=np.float64)
    ys = np.array([p[1] for p in knots], dtype=np.float64)
    fwd = PchipInterpolator(xs, ys, extrapolate=False)
    # Inverse: swap x/y. Curve is monotonic so this works.
    inv = PchipInterpolator(ys, xs, extrapolate=False)
    def apply_fwd(v):
        out = fwd(np.clip(v, 0.0, 1.0))
        return np.where(np.isnan(out), np.clip(v, 0.0, 1.0), out)
    def apply_inv(v):
        out = inv(np.clip(v, 0.0, 1.0))
        return np.where(np.isnan(out), np.clip(v, 0.0, 1.0), out)
    return apply_fwd, apply_inv


def binned_medians(x_flat, y_flat, n_bins=64, min_count=200):
    """For each bin of x, compute median y."""
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.searchsorted(bin_edges, x_flat, side="right") - 1, 0, n_bins - 1)
    centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    xs_out = []
    ys_out = []
    counts = []
    for b in range(n_bins):
        mask = idx == b
        c = int(mask.sum())
        if c < min_count:
            continue
        xs_out.append(centers[b])
        ys_out.append(float(np.median(y_flat[mask])))
        counts.append(c)
    return np.array(xs_out), np.array(ys_out), np.array(counts)


def main():
    print("[extract] rendering ours (full-size)...")
    t0 = time.time()
    ours_srgb, meta = load_raw(
        TEST_RAF, output_mode="sRGB display", baseline_exposure=0.0, half_size=False,
    )
    print(f"[extract] ours rendered in {time.time()-t0:.1f}s  dcp={meta.get('dcp_profile')}")

    acr_srgb = load_png_float(ACR_PNG)

    ours_srgb, acr_srgb = align_crops(ours_srgb, acr_srgb)
    print(f"[extract] aligned: {ours_srgb.shape}")

    # Downsample BEFORE linearizing — Lanczos in sRGB space is standard,
    # and we just want to kill sharpening halos.
    ours_srgb_ds = downsample(ours_srgb, 2048)
    acr_srgb_ds  = downsample(acr_srgb,  2048)
    if ours_srgb_ds.shape != acr_srgb_ds.shape:
        h = min(ours_srgb_ds.shape[0], acr_srgb_ds.shape[0])
        w = min(ours_srgb_ds.shape[1], acr_srgb_ds.shape[1])
        ours_srgb_ds = ours_srgb_ds[:h, :w]
        acr_srgb_ds  = acr_srgb_ds[:h, :w]
    print(f"[extract] downsampled to {ours_srgb_ds.shape}")

    # Remove sRGB OETF to get linear display-referred values
    ours_lin = srgb_to_linear(ours_srgb_ds).astype(np.float64)
    acr_lin  = srgb_to_linear(acr_srgb_ds).astype(np.float64)

    # Invert the current user curve on ours_lin to recover post-DCP linear values
    fwd, inv = make_curve_and_inverse(CAMERA_RAW_DEFAULT_CURVE)
    ours_post_dcp = np.empty_like(ours_lin)
    for c in range(3):
        ours_post_dcp[..., c] = inv(ours_lin[..., c])

    # For each pixel we now have (post_DCP_ours[c], acr_linear_target[c]).
    # Extract binned-median transfer function per channel.
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharex=True, sharey=True)
    labels = ["Red", "Green", "Blue"]
    colors = ["#d62728", "#2ca02c", "#1f77b4"]

    # Dense sampling of the current curve for overlay
    x_plot = np.linspace(0, 1, 256)
    y_current = fwd(x_plot)

    results = {}
    for c in range(3):
        x_flat = ours_post_dcp[..., c].flatten()
        y_flat = acr_lin[..., c].flatten()

        # Keep only finite, in-range pairs
        m = np.isfinite(x_flat) & np.isfinite(y_flat) & (x_flat >= 0) & (x_flat <= 1) & (y_flat >= 0) & (y_flat <= 1)
        x_flat = x_flat[m]
        y_flat = y_flat[m]

        xs, ys, cnts = binned_medians(x_flat, y_flat, n_bins=80, min_count=200)
        results[labels[c]] = (xs, ys, cnts)

        ax = axes[c]
        # 2D hexbin density plot (subsample for speed)
        if len(x_flat) > 2_000_000:
            rng = np.random.default_rng(0)
            idx = rng.choice(len(x_flat), size=2_000_000, replace=False)
            ax.hexbin(x_flat[idx], y_flat[idx], gridsize=120, cmap="Greys", bins="log", mincnt=1)
        else:
            ax.hexbin(x_flat, y_flat, gridsize=120, cmap="Greys", bins="log", mincnt=1)
        ax.plot(x_plot, y_current, color="orange", lw=2, label="current stopgap (from session 3)")
        ax.plot(xs, ys, color=colors[c], lw=2.5, marker="o", ms=3, label=f"recovered {labels[c]} (median)")
        ax.plot([0, 1], [0, 1], color="gray", lw=1, ls="--", alpha=0.5, label="identity")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("post-DCP linear value (ours)")
        if c == 0:
            ax.set_ylabel("ACR linear value (target)")
        ax.set_title(f"{labels[c]} channel transfer function")
        ax.legend(loc="lower right", fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Recovered per-channel transfer function: our post-DCP linear -> ACR linear\n"
                 "(2048 px long edge, sharpening halos collapsed)")
    fig.tight_layout()
    out = os.path.join(OUT_DIR, "recovered_transfer_functions.png")
    fig.savefig(out, dpi=100)
    plt.close(fig)
    print(f"[extract] wrote plot -> {out}")

    print("\n=== Recovered per-channel curves (x=post-DCP linear, y=ACR linear) ===")
    for name, (xs, ys, cnts) in results.items():
        print(f"\n-- {name} -- ({len(xs)} bins)")
        print("   x       y       count     current   delta")
        for x, y, n in zip(xs, ys, cnts):
            cy = float(np.asarray(fwd(np.array([x]))).ravel()[0])
            print(f"  {x:.4f}  {y:.4f}  {n:8d}   {cy:.4f}   {y-cy:+.4f}")


if __name__ == "__main__":
    main()
