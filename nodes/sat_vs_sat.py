"""
Sat vs Sat node for ComfyUI-Darkroom.
Adjust saturation based on existing saturation level with presets and per-zone control.
"""

import torch

from ..utils import gpu_color as G
from ..data.grading_presets import SAT_VS_SAT_PRESETS, SAT_VS_SAT_PRESET_NAMES


# 4 saturation zones: center and width for Gaussian masks
SAT_ZONES = {
    "low":      (0.125, 0.12),
    "mid_low":  (0.375, 0.12),
    "mid_high": (0.625, 0.12),
    "high":     (0.875, 0.12),
}


class SatVsSat:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "preset": (SAT_VS_SAT_PRESET_NAMES, {
                    "default": "Custom (manual)",
                    "tooltip": "Select a saturation-based saturation preset or use Custom"
                }),
            },
            "optional": {
                "low_sat_adjust": ("FLOAT", {
                    "default": 0.0, "min": -100.0, "max": 100.0, "step": 1.0,
                    "tooltip": "Adjust nearly-desaturated tones (0-25% saturation)"
                }),
                "mid_low_sat_adjust": ("FLOAT", {
                    "default": 0.0, "min": -100.0, "max": 100.0, "step": 1.0,
                    "tooltip": "Adjust low-saturation tones (25-50% saturation)"
                }),
                "mid_high_sat_adjust": ("FLOAT", {
                    "default": 0.0, "min": -100.0, "max": 100.0, "step": 1.0,
                    "tooltip": "Adjust medium-saturation tones (50-75% saturation)"
                }),
                "high_sat_adjust": ("FLOAT", {
                    "default": 0.0, "min": -100.0, "max": 100.0, "step": 1.0,
                    "tooltip": "Adjust highly-saturated tones (75-100% saturation)"
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

    def execute(self, image, preset="Custom (manual)", low_sat_adjust=0.0,
                mid_low_sat_adjust=0.0, mid_high_sat_adjust=0.0,
                high_sat_adjust=0.0, strength=1.0):

        if strength <= 0.0:
            return (image,)

        adjustments = {
            "low": low_sat_adjust,
            "mid_low": mid_low_sat_adjust,
            "mid_high": mid_high_sat_adjust,
            "high": high_sat_adjust,
        }

        if preset != "Custom (manual)" and preset in SAT_VS_SAT_PRESETS:
            p = SAT_VS_SAT_PRESETS[preset]
            adjustments["low"] += p.low
            adjustments["mid_low"] += p.mid_low
            adjustments["mid_high"] += p.mid_high
            adjustments["high"] += p.high

        active = [(name, val) for name, val in adjustments.items() if abs(val) > 0.5]
        if not active:
            return (image,)

        print(f"[Darkroom] Sat vs Sat: preset={preset}, {len(active)} active zones, strength={strength}")

        def _pipeline(x):
            linear = G.srgb_to_linear(x)
            lum = G.luminance_rec709(linear)

            # Compute per-pixel saturation
            sat = G.sat_from_rgb(linear)

            # Compute combined adjustment factor
            sat_factor = torch.ones_like(sat)
            for zone_name, adj_value in active:
                center, width = SAT_ZONES[zone_name]
                mask = G.sat_range_mask(sat, center, width)
                sat_factor = sat_factor + mask * (adj_value / 100.0)

            # Apply luminance-preserving saturation scaling
            lum_3d = lum[..., None]
            result = lum_3d + sat_factor[..., None] * (linear - lum_3d)
            result = result.clamp(0.0, 1.0)

            result = G.linear_to_srgb(result)
            return G.blend(x, result, strength)

        return (G.run_on_device(_pipeline, image),)


NODE_CLASS_MAPPINGS = {"DarkroomSatVsSat": SatVsSat}
NODE_DISPLAY_NAME_MAPPINGS = {"DarkroomSatVsSat": "Sat vs Sat"}
