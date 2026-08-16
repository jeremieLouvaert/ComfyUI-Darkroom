"""
Chromatic Aberration node for ComfyUI-Darkroom.

Simulates lateral chromatic aberration -- the color fringing caused by a lens
failing to focus all wavelengths to the same point.

Select a real lens profile or use Custom for manual control.

GPU-accelerated via torch. No CPU roundtrip.
"""

import math
import torch
import torch.nn.functional as F

from ..data.lens_profiles import LENS_PROFILES_FLAT, LENS_PROFILE_NAMES
from ..utils.torch_ops import pixel_to_grid_coords, grid_sample_channel


PRESET_NAMES = ["Custom"] + LENS_PROFILE_NAMES


def _apply_lateral_ca(img, shift_r, shift_b):
    """
    Apply lateral CA by radially scaling R and B channels on GPU.
    Green channel stays fixed (reference). Shift in pixels at the image edge.
    img: (H, W, C) tensor on device.
    """
    h, w = img.shape[:2]
    cy, cx = h / 2.0, w / 2.0
    max_r = math.sqrt(cx * cx + cy * cy)
    device = img.device

    yy = torch.arange(h, dtype=torch.float32, device=device)
    xx = torch.arange(w, dtype=torch.float32, device=device)
    yy, xx = torch.meshgrid(yy, xx, indexing='ij')
    dy = yy - cy
    dx = xx - cx
    r = torch.sqrt(dy * dy + dx * dx) / max_r

    result = img.clone()

    for c, shift in enumerate([shift_r, 0.0, shift_b]):
        if abs(shift) < 0.01:
            continue

        scale = 1.0 + (shift / max_r) * r
        new_y = cy + dy * scale
        new_x = cx + dx * scale

        grid = pixel_to_grid_coords(new_y, new_x, h, w)
        result[..., c] = grid_sample_channel(img[..., c], grid,
                                             padding_mode='reflection')

    return result.clamp(0.0, 1.0)


def _prepare_masks(mask, batch, h, w, device, dtype):
    """
    Mask prep for the torch engine (docs/lens-mask-derivation.md §5.4/§3):
    clamp(0,1) -> pin dtype/device -> 2-D unsqueeze -> M==0 guard (treated as
    absent, console warn) -> per-frame min(i, M-1) -> resize via
    F.interpolate(bilinear, align_corners=True) + clamp.
    Returns a list of (H, W) tensors (one per batch frame), or None if the
    mask batch is empty.
    """
    m = mask.clamp(0.0, 1.0).to(device=device, dtype=dtype)
    if m.dim() == 2:
        m = m.unsqueeze(0)
    M = m.shape[0]
    if M == 0:
        print("[Darkroom] Chromatic Aberration: mask batch is empty (M=0), treating as absent")
        return None

    masks = []
    for i in range(batch):
        mi = m[min(i, M - 1)]
        if mi.shape != (h, w):
            mi = F.interpolate(mi.unsqueeze(0).unsqueeze(0), size=(h, w),
                               mode='bilinear', align_corners=True)
            mi = mi.squeeze(0).squeeze(0).clamp(0.0, 1.0)
        masks.append(mi)
    return masks


def _masked_coords_ca(h, w, shift_r, shift_b, m, device):
    """
    Declared seam (docs/lens-mask-derivation.md §5.3), per active channel.
    Every channel uses the SAME mask tensor m (§2b -- one mask, three
    channels, so the R:B displacement ratio stays lens-true). Channels below
    the shift skip threshold are absent from the dict, mirroring the
    unmasked skip.
    Returns {c: (src_y, src_x, src_y_m, src_x_m)} for c in {0 (R), 2 (B)}.
    """
    cy, cx = h / 2.0, w / 2.0
    max_r = math.sqrt(cx * cx + cy * cy)

    yy = torch.arange(h, dtype=torch.float32, device=device)
    xx = torch.arange(w, dtype=torch.float32, device=device)
    yy, xx = torch.meshgrid(yy, xx, indexing='ij')
    dy = yy - cy
    dx = xx - cx
    r = torch.sqrt(dy * dy + dx * dx) / max_r

    out = {}
    for c, shift in ((0, shift_r), (2, shift_b)):
        if abs(shift) < 0.01:
            continue

        scale = 1.0 + (shift / max_r) * r
        src_y = cy + dy * scale
        src_x = cx + dx * scale

        d_y = src_y - yy
        d_x = src_x - xx
        src_y_m = yy + m * d_y
        src_x_m = xx + m * d_x

        out[c] = (src_y, src_x, src_y_m, src_x_m)

    return out


def _apply_lateral_ca_masked(img, shift_r, shift_b, m):
    """
    Masked Chromatic Aberration (§2b): universal D-form per channel, same
    mask tensor for R and B. The §1c select is applied once, at the end,
    against the pristine input. Green is untouched (D_g = 0), same as today.
    img: (H, W, C) tensor on device. m: (H, W) tensor.
    """
    h, w = img.shape[:2]
    device = img.device

    coords = _masked_coords_ca(h, w, shift_r, shift_b, m, device)
    result = img.clone()

    for c, (_, _, src_y_m, src_x_m) in coords.items():
        grid = pixel_to_grid_coords(src_y_m, src_x_m, h, w)
        result[..., c] = grid_sample_channel(img[..., c], grid,
                                             padding_mode='reflection')

    result = result.clamp(0.0, 1.0)
    return torch.where((m == 0.0).unsqueeze(-1), img, result)


class ChromaticAberration:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "lens": (PRESET_NAMES, {
                    "default": "Custom",
                    "tooltip": "Select a real lens or 'Custom' for manual CA control"
                }),
                "strength": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 3.0, "step": 0.1,
                    "tooltip": "Overall CA intensity multiplier"
                }),
            },
            "optional": {
                "shift_r": ("FLOAT", {
                    "default": -1.0, "min": -5.0, "max": 5.0, "step": 0.1,
                    "tooltip": "Red channel shift in pixels at image edge. Negative = inward"
                }),
                "shift_b": ("FLOAT", {
                    "default": 1.0, "min": -5.0, "max": 5.0, "step": 0.1,
                    "tooltip": "Blue channel shift in pixels at image edge. Positive = outward"
                }),
                "mask": ("MASK", {
                    "tooltip": "Where the fringing applies, 0 to 1. One mask drives all channels "
                               "together so the red/blue ratio stays lens-true. A few pixels of feather "
                               "is enough at typical shifts."
                }),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "execute"
    CATEGORY = "AKURATE/Darkroom/Lens"

    def execute(self, image, lens, strength, shift_r=-1.0, shift_b=1.0, mask=None):
        if strength <= 0.0:
            return (image,)

        if lens != "Custom" and lens in LENS_PROFILES_FLAT:
            p = LENS_PROFILES_FLAT[lens]
            shift_r = p.ca_r
            shift_b = p.ca_b

        masks = None
        if mask is not None:
            b, h0, w0 = image.shape[0], image.shape[1], image.shape[2]
            masks = _prepare_masks(mask, b, h0, w0, image.device, image.dtype)

        results = []
        for i in range(image.shape[0]):
            img = image[i]
            h, w = img.shape[:2]
            scale = min(h, w) / 1024.0
            sr = shift_r * strength * scale
            sb = shift_b * strength * scale

            print(f"[Darkroom] Chromatic Aberration: {lens}, shift_r={sr:.2f}px, shift_b={sb:.2f}px")
            if masks is None:
                results.append(_apply_lateral_ca(img, sr, sb))
            else:
                results.append(_apply_lateral_ca_masked(img, sr, sb, masks[i]))

        return (torch.stack(results, dim=0),)


NODE_CLASS_MAPPINGS = {
    "DarkroomChromaticAberration": ChromaticAberration,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DarkroomChromaticAberration": "Chromatic Aberration",
}
