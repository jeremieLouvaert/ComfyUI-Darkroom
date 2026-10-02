"""
Lum vs Sat node for ComfyUI-Darkroom.
Adjust saturation based on luminance zones with presets and per-zone control.
"""

import torch

from ..utils import gpu_color as G
from ..data.grading_presets import LUM_VS_SAT_PRESETS, LUM_VS_SAT_PRESET_NAMES


def _parametric_tone_mask(luminance, center, sigma):
    """Torch mirror of utils.raw.parametric_tone_mask (Gaussian in luminance)."""
    return torch.exp(-0.5 * ((luminance - center) / (sigma + 1e-10)) ** 2)


# 5 luminance zones: center and sigma for Gaussian masks
LUM_ZONES = {
    "blacks":     (0.05, 0.06),
    "shadows":    (0.20, 0.10),
    "midtones":   (0.50, 0.18),
    "highlights": (0.80, 0.10),
    "whites":     (0.95, 0.06),
}


class LumVsSat:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "preset": (LUM_VS_SAT_PRESET_NAMES, {
                    "default": "Custom (manual)",
                    "tooltip": "Select a luminance-based saturation preset or use Custom"
                }),
            },
            "optional": {
                "blacks_saturation": ("FLOAT", {
                    "default": 0.0, "min": -100.0, "max": 100.0, "step": 1.0,
                    "tooltip": "Saturation of darkest tones (0-10%)"
                }),
                "shadows_saturation": ("FLOAT", {
                    "default": 0.0, "min": -100.0, "max": 100.0, "step": 1.0,
                    "tooltip": "Saturation of shadows (10-30%)"
                }),
                "midtones_saturation": ("FLOAT", {
                    "default": 0.0, "min": -100.0, "max": 100.0, "step": 1.0,
                    "tooltip": "Saturation of midtones (30-70%)"
                }),
                "highlights_saturation": ("FLOAT", {
                    "default": 0.0, "min": -100.0, "max": 100.0, "step": 1.0,
                    "tooltip": "Saturation of highlights (70-90%)"
                }),
                "whites_saturation": ("FLOAT", {
                    "default": 0.0, "min": -100.0, "max": 100.0, "step": 1.0,
                    "tooltip": "Saturation of brightest tones (90-100%)"
                }),
                "strength": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Blend between original (0) and adjusted (1)"
                }),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "execute"
    CATEGORY = "AKURATE/Darkroom/Grading"

    def execute(self, image, preset="Custom (manual)", blacks_saturation=0.0,
                shadows_saturation=0.0, midtones_saturation=0.0,
                highlights_saturation=0.0, whites_saturation=0.0, strength=1.0):

        if strength <= 0.0:
            return (image,)

        adjustments = {
            "blacks": blacks_saturation,
            "shadows": shadows_saturation,
            "midtones": midtones_saturation,
            "highlights": highlights_saturation,
            "whites": whites_saturation,
        }

        if preset != "Custom (manual)" and preset in LUM_VS_SAT_PRESETS:
            p = LUM_VS_SAT_PRESETS[preset]
            adjustments["blacks"] += p.blacks
            adjustments["shadows"] += p.shadows
            adjustments["midtones"] += p.midtones
            adjustments["highlights"] += p.highlights
            adjustments["whites"] += p.whites

        active = [(name, val) for name, val in adjustments.items() if abs(val) > 0.5]
        if not active:
            return (image,)

        print(f"[Darkroom] Lum vs Sat: preset={preset}, {len(active)} active zones, strength={strength}")

        def _pipeline(x):
            linear = G.srgb_to_linear(x)
            lum = G.luminance_rec709(linear)

            # Compute combined saturation factor per pixel
            sat_factor = torch.ones_like(lum)
            for zone_name, adj_value in active:
                center, sigma = LUM_ZONES[zone_name]
                mask = _parametric_tone_mask(lum, center, sigma)
                sat_factor = sat_factor + mask * (adj_value / 100.0)

            # Apply luminance-preserving saturation adjustment
            lum_3d = lum[..., None]
            result = lum_3d + sat_factor[..., None] * (linear - lum_3d)
            result = result.clamp(0.0, 1.0)

            result = G.linear_to_srgb(result)
            return G.blend(x, result, strength)

        return (G.run_on_device(_pipeline, image),)


NODE_CLASS_MAPPINGS = {"DarkroomLumVsSat": LumVsSat}
NODE_DISPLAY_NAME_MAPPINGS = {"DarkroomLumVsSat": "Lum vs Sat"}
