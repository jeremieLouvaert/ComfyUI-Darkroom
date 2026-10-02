"""
Color Qualifier node for ComfyUI-Darkroom.
DaVinci Resolve-style secondary color selection and correction.

Two workflows:
1. Pick an ACTION preset (e.g. "Warm up skin", "Cool down sky") — immediately visible
2. Pick a SELECTION preset + dial corrections manually

Matte preview output lets you see exactly what's selected.
"""

import numpy as np
import torch
import torch.nn.functional as F

from ..utils.gpu_color import blend, rgb_to_hsl, hsl_to_rgb, run_on_device


# Action presets: selection params + built-in correction
# (hue_c, hue_w, hue_soft, sat_c, sat_w, sat_soft, lum_c, lum_w, lum_soft, hue_shift, sat_adj, lum_adj)
ACTION_PRESETS = {
    "Custom (manual)": None,
    # --- Skin ---
    "Warm up skin tones": (25, 70, 0.4, 0.45, 0.70, 0.3, 0.50, 0.80, 0.3, 8, 5, 0),
    "Cool down skin tones": (25, 70, 0.4, 0.45, 0.70, 0.3, 0.50, 0.80, 0.3, -8, -5, 0),
    "Reduce skin redness": (8, 40, 0.4, 0.40, 0.80, 0.3, 0.50, 0.80, 0.3, 12, -15, 0),
    "Even out skin saturation": (25, 70, 0.4, 0.45, 0.70, 0.3, 0.50, 0.80, 0.3, 0, -20, 0),
    # --- Sky ---
    "Deepen blue sky": (215, 70, 0.4, 0.35, 0.90, 0.3, 0.55, 0.90, 0.3, 0, 25, -10),
    "Warm up sky": (215, 70, 0.4, 0.35, 0.90, 0.3, 0.55, 0.90, 0.3, -20, -10, 5),
    "Teal sky": (215, 70, 0.4, 0.35, 0.90, 0.3, 0.55, 0.90, 0.3, -35, 10, 0),
    # --- Foliage ---
    "Vivid greens": (120, 80, 0.3, 0.40, 0.90, 0.2, 0.45, 0.90, 0.2, 0, 30, 5),
    "Autumn foliage": (120, 80, 0.3, 0.40, 0.90, 0.2, 0.45, 0.90, 0.2, -40, 15, -5),
    "Muted greens": (120, 80, 0.3, 0.40, 0.90, 0.2, 0.45, 0.90, 0.2, 0, -35, -5),
    # --- General color ---
    "Pop reds": (0, 50, 0.3, 0.55, 0.90, 0.2, 0.50, 1.0, 0.2, 0, 30, 5),
    "Mute reds": (0, 50, 0.3, 0.55, 0.90, 0.2, 0.50, 1.0, 0.2, 0, -30, 0),
    "Pop yellows": (55, 40, 0.3, 0.45, 0.90, 0.2, 0.50, 1.0, 0.2, 0, 25, 5),
    "Warm highlights": (0, 360, 0.0, 0.50, 1.0, 0.0, 0.85, 0.30, 0.3, 10, 5, 0),
    "Cool shadows": (0, 360, 0.0, 0.50, 1.0, 0.0, 0.15, 0.30, 0.3, -15, 5, 0),
    "Desaturate shadows": (0, 360, 0.0, 0.50, 1.0, 0.0, 0.15, 0.30, 0.3, 0, -40, 0),
    "Boost all saturation": (0, 360, 0.0, 0.50, 1.0, 0.0, 0.50, 1.0, 0.2, 0, 25, 0),
    "Desaturate everything": (0, 360, 0.0, 0.50, 1.0, 0.0, 0.50, 1.0, 0.2, 0, -30, 0),
}

PRESET_LIST = list(ACTION_PRESETS.keys())


def _soft_range(values, center, width, softness, wrap=None):
    """
    Compute a soft selection mask for values within center+-width,
    with smooth falloff controlled by softness.
    """
    diff = torch.abs(values - center)
    if wrap is not None:
        diff = torch.minimum(diff, wrap - diff)

    half_width = width * 0.5
    soft_zone = half_width * softness

    mask = torch.ones_like(values)

    if soft_zone > 0.01:
        transition = (diff > half_width) & (diff <= half_width + soft_zone)
        t = (diff - half_width) / (soft_zone + 1e-10)
        mask = torch.where(transition, (1.0 + torch.cos(np.pi * t)) * 0.5, mask)

    return torch.where(diff > half_width + soft_zone, torch.zeros_like(mask), mask)


def _gaussian_filter(x, sigma):
    """
    scipy.ndimage.gaussian_filter on (B, H, W): truncate=4.0, mode='reflect'
    (half-sample symmetric, edge pixel repeated), separable, per image.
    """
    radius = int(4.0 * sigma + 0.5)
    if radius < 1:
        return x
    ax = torch.arange(-radius, radius + 1, dtype=torch.float64)
    k = torch.exp(-0.5 * ax * ax / (sigma * sigma))
    k = (k / k.sum()).to(device=x.device, dtype=x.dtype)

    def reflect_pad(t, dim):
        n = t.shape[dim]
        idx = torch.arange(-radius, n + radius, device=t.device) % (2 * n)
        idx = torch.where(idx >= n, 2 * n - 1 - idx, idx)
        return t.index_select(dim, idx)

    y = F.conv2d(reflect_pad(x, 1).unsqueeze(1), k.view(1, 1, -1, 1))
    y = F.conv2d(reflect_pad(y.squeeze(1), 2).unsqueeze(1), k.view(1, 1, 1, -1))
    return y.squeeze(1)


def _binary_morph(binary, iterations, erode):
    """
    scipy binary_erosion / binary_dilation with the default 4-connected cross
    structure and border_value=0, on a (B, H, W) bool tensor.
    """
    for _ in range(iterations):
        p = F.pad(binary, (1, 1, 1, 1), value=False)
        c = p[:, 1:-1, 1:-1]
        up, down = p[:, :-2, 1:-1], p[:, 2:, 1:-1]
        left, right = p[:, 1:-1, :-2], p[:, 1:-1, 2:]
        if erode:
            binary = c & up & down & left & right
        else:
            binary = c | up | down | left | right
    return binary


class ColorQualifier:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "preset": (PRESET_LIST, {
                    "default": "Custom (manual)",
                    "tooltip": (
                        "Action presets: select a color range AND apply a correction in one step. "
                        "Use 'Custom (manual)' for full control over selection + corrections"
                    )
                }),
            },
            "optional": {
                # Manual HSL selection (used with Custom, or to override preset selection)
                "hue_center": ("FLOAT", {
                    "default": 25.0, "min": 0.0, "max": 360.0, "step": 1.0,
                    "tooltip": "Center hue: 0=red, 30=orange, 60=yellow, 120=green, 180=cyan, 240=blue, 300=magenta"
                }),
                "hue_width": ("FLOAT", {
                    "default": 70.0, "min": 1.0, "max": 360.0, "step": 1.0,
                    "tooltip": "Hue range width in degrees"
                }),
                "hue_softness": ("FLOAT", {
                    "default": 0.3, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Hue edge softness (0 = hard cut, 1 = very soft)"
                }),
                "sat_center": ("FLOAT", {
                    "default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Center saturation for selection"
                }),
                "sat_width": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Saturation range width"
                }),
                "sat_softness": ("FLOAT", {
                    "default": 0.2, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Saturation edge softness"
                }),
                "lum_center": ("FLOAT", {
                    "default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Center luminance for selection"
                }),
                "lum_width": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Luminance range width"
                }),
                "lum_softness": ("FLOAT", {
                    "default": 0.2, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Luminance edge softness"
                }),
                # Matte finesse
                "matte_blur": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 20.0, "step": 0.5,
                    "tooltip": "Blur matte edges for smoother transitions"
                }),
                "matte_shrink": ("INT", {
                    "default": 0, "min": -10, "max": 10, "step": 1,
                    "tooltip": "Shrink (+) or grow (-) the selection"
                }),
                # Corrections (these add on top of preset corrections)
                "hue_shift": ("FLOAT", {
                    "default": 0.0, "min": -180.0, "max": 180.0, "step": 1.0,
                    "tooltip": "Shift hue of selected pixels (degrees). Adds to preset value"
                }),
                "saturation_adjust": ("FLOAT", {
                    "default": 0.0, "min": -100.0, "max": 100.0, "step": 1.0,
                    "tooltip": "Adjust saturation of selected pixels. Adds to preset value"
                }),
                "luminance_adjust": ("FLOAT", {
                    "default": 0.0, "min": -100.0, "max": 100.0, "step": 1.0,
                    "tooltip": "Adjust luminance of selected pixels. Adds to preset value"
                }),
                "invert_matte": (["no", "yes"], {
                    "default": "no",
                    "tooltip": "Invert — correct everything EXCEPT the selected range"
                }),
                "strength": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Blend between original (0) and corrected (1)"
                }),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE",)
    RETURN_NAMES = ("image", "matte_preview",)
    FUNCTION = "execute"
    CATEGORY = "AKURATE/Darkroom/Raw"

    def execute(self, image, preset="Custom (manual)",
                hue_center=25.0, hue_width=70.0, hue_softness=0.3,
                sat_center=0.5, sat_width=1.0, sat_softness=0.2,
                lum_center=0.5, lum_width=1.0, lum_softness=0.2,
                matte_blur=1.0, matte_shrink=0,
                hue_shift=0.0, saturation_adjust=0.0, luminance_adjust=0.0,
                invert_matte="no", strength=1.0):

        # Apply action preset: overrides selection AND adds correction values
        p = ACTION_PRESETS.get(preset)
        if p is not None:
            (hue_center, hue_width, hue_softness,
             sat_center, sat_width, sat_softness,
             lum_center, lum_width, lum_softness,
             preset_hue, preset_sat, preset_lum) = p
            # Corrections: preset baseline + user manual adjustments stack
            hue_shift = preset_hue + hue_shift
            saturation_adjust = preset_sat + saturation_adjust
            luminance_adjust = preset_lum + luminance_adjust

        # Corrections are decided once from the params, as in the per-image loop
        has_correction = (abs(hue_shift) > 0.1 or
                          abs(saturation_adjust) > 0.5 or
                          abs(luminance_adjust) > 0.5)

        def qualify(img):
            h, s, l = rgb_to_hsl(img)

            # Build qualifier matte
            hue_mask = _soft_range(h, hue_center, hue_width, hue_softness, wrap=360.0)
            sat_mask = _soft_range(s, sat_center, sat_width, sat_softness)
            lum_mask = _soft_range(l, lum_center, lum_width, lum_softness)

            matte = hue_mask * sat_mask * lum_mask

            # Matte finesse: shrink/grow
            if matte_shrink != 0:
                binary = matte > 0.5
                iterations = abs(matte_shrink)
                refined = _binary_morph(binary, iterations, erode=matte_shrink > 0)
                matte = torch.where(refined, torch.clamp(matte, min=0.5), torch.clamp(matte, max=0.5))
                matte = matte.clamp(0.0, 1.0)

            # Matte finesse: blur
            if matte_blur > 0.1:
                matte = _gaussian_filter(matte, matte_blur)

            # Invert
            if invert_matte == "yes":
                matte = 1.0 - matte

            # Apply corrections
            if has_correction and strength > 0.0:
                h_new = h
                s_new = s
                l_new = l

                if abs(hue_shift) > 0.1:
                    h_new = (h_new + matte * hue_shift) % 360.0

                if abs(saturation_adjust) > 0.5:
                    s_new = s_new * (1.0 + matte * (saturation_adjust / 100.0))
                    s_new = s_new.clamp(0.0, 1.0)

                if abs(luminance_adjust) > 0.5:
                    l_new = l_new * (1.0 + matte * (luminance_adjust / 100.0))
                    l_new = l_new.clamp(0.0, 1.0)

                result = hsl_to_rgb(h_new, s_new, l_new)
                result = result.clamp(0.0, 1.0)
                out = blend(img, result, strength)
            else:
                out = img

            # Matte preview rides along as channels 3..5 so one device pass returns both
            matte_vis = torch.stack([matte, matte, matte], dim=-1)
            return torch.cat([out, matte_vis], dim=-1)

        both = run_on_device(qualify, image)
        return (both[..., :3].contiguous(), both[..., 3:].contiguous())


NODE_CLASS_MAPPINGS = {"DarkroomColorQualifier": ColorQualifier}
NODE_DISPLAY_NAME_MAPPINGS = {"DarkroomColorQualifier": "Color Qualifier"}
