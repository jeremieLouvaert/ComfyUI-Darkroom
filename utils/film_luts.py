"""
Baked Capture One film looks (decisions.md 2026-10-02).

data/c1_film_luts.npz holds, for each Darkroom colour stock that comes from the
Capture One Film Styles pack (MIT), the style's GLOBAL tools (curves, levels,
colour balance, saturation, Advanced Color Editor) rendered by Capture One
itself onto a 65^3 grid and stored at 33^3 (uint16). Validated against C1's own
renders: mean 0.1-0.2/255, p99 ~1/255 on a photo (tools/c1_lut_bake.py).

What a grid cannot hold is C1's Shadow/Highlight Recovery, which is local. It
is approximated here by a luminance-dependent gain driven by a lightly blurred
luminance (0.5% of the long edge), scaled by each style's recovery values, with
shared gain curves fitted on C1 renders (held-out photo error 1.6-2.5/255 mean).

Pipeline per image: recovery (optional) -> 3D LUT, all on the device.
"""

import os

import numpy as np
import torch
import torch.nn.functional as F

from . import gpu_color as G

_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "c1_film_luts.npz")
_data = None
_device_luts = {}


def _load():
    global _data
    if _data is None:
        z = np.load(_PATH)
        names = [str(n) for n in z["names"]]
        _data = {
            "index": {n: i for i, n in enumerate(names)},
            "luts": z["luts"],
            "sr": z["shadow_recovery"],
            "hr": z["highlight_recovery"],
            "knots": z["recovery_knots"],
            "blur_frac": float(z["recovery_blur_frac"]),
        }
    return _data


_warned_missing = False


def has_baked(stock_name):
    global _warned_missing
    if not os.path.isfile(_PATH):
        if not _warned_missing:
            print(f"[Darkroom] Film Stock: {_PATH} is missing; the Capture One stocks fall back to "
                  "their old curve approximation. Reinstall the pack to restore them.")
            _warned_missing = True
        return False
    return stock_name in _load()["index"]


def _lut(stock_name, dev):
    key = (stock_name, str(dev))
    if key not in _device_luts:
        d = _load()
        lut = d["luts"][d["index"][stock_name]].astype(np.float32) / 65535.0
        _device_luts[key] = G.lut_to_device(lut, dev)
    return _device_luts[key]


def _blur(y, sigma_px):
    """Gaussian blur of (H, W) via downsample -> blur -> bilinear upsample."""
    if sigma_px < 0.5:
        return y
    f = max(1, int(sigma_px // 4))
    yd = F.avg_pool2d(y[None, None], f, ceil_mode=True)[0, 0] if f > 1 else y
    s = sigma_px / f
    r = max(1, int(3 * s))
    x = torch.arange(-r, r + 1, device=y.device, dtype=y.dtype)
    k = torch.exp(-0.5 * (x / s) ** 2)
    k = k / k.sum()
    t = F.pad(yd[None, None], (r, r, r, r), mode="replicate")   # any size, even 1 px
    t = F.conv2d(F.conv2d(t, k.view(1, 1, 1, -1)), k.view(1, 1, -1, 1))
    return F.interpolate(t, size=y.shape, mode="bilinear", align_corners=False)[0, 0]


def _interp(v, knots):
    t = v.clamp(0.0, 1.0) * (knots.numel() - 1)
    i = t.floor().clamp(max=knots.numel() - 2).long()
    f = t - i
    return knots[i] * (1.0 - f) + knots[i + 1] * f


def _recovery(img, sr, hr, d):
    """img (H, W, 3) on device -> recovered image (same shape)."""
    knots = G.const(d["knots"], img)
    n = knots.numel() // 2
    y = G.luminance_rec709(img)
    yb = _blur(y, d["blur_frac"] * max(img.shape[0], img.shape[1]))
    gain = 1.0 + sr * _interp(yb, knots[:n]) - hr * _interp(yb, knots[n:])
    return (img * gain[..., None]).clamp(0.0, 1.0)


def apply_baked(batch, stock_name, recovery=True):
    """batch (B, H, W, 3) on device -> C1 film look. Recovery is per image (local)."""
    d = _load()
    i = d["index"][stock_name]
    lut = _lut(stock_name, batch.device)
    sr, hr = float(d["sr"][i]) / 100.0, float(d["hr"][i]) / 100.0
    out = []
    for img in batch:
        if recovery and (sr != 0.0 or hr != 0.0):
            img = _recovery(img, sr, hr, d)
        out.append(G.apply_lut_trilinear(img, lut, 33))
    return torch.stack(out, 0)
