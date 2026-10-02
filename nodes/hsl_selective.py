"""
HSL Selective node for ComfyUI-Darkroom.
Per-hue adjustments to Hue, Saturation, and Luminance with smooth feathered transitions.
"""

import numpy as np
import torch

from dataclasses import asdict

from ..utils.gpu_color import (
    srgb_to_linear, linear_to_srgb, blend, rgb_to_hsl, hsl_to_rgb, run_on_device,
)
from ..data.ai_mitigation_presets import AI_MITIGATION_HSL, HSL_PRESET_NAMES


# Hue centers in degrees for each range
HUE_CENTERS = {
    "red": 0.0,
    "orange": 30.0,
    "yellow": 60.0,
    "green": 120.0,
    "aqua": 180.0,
    "blue": 240.0,
    "purple": 270.0,
    "magenta": 330.0,
}

HUE_WIDTH = 45.0  # Degrees, half-width of the raised cosine window


def _hue_weight(hue, center, width=HUE_WIDTH):
    """
    Raised cosine weight for a hue range.
    Smooth bell curve centered at 'center' with half-width 'width'.
    Handles wraparound at 0/360.
    """
    # Angular distance with wraparound
    diff = torch.abs(hue - center)
    diff = torch.minimum(diff, 360.0 - diff)

    # Raised cosine: cos-based smooth falloff within width
    weight = ((1.0 + torch.cos(np.pi * diff / width)) * 0.5).clamp(0.0, 1.0)
    # Zero out beyond the width
    return torch.where(diff > width, torch.zeros_like(weight), weight)


class HSLSelective:

    @classmethod
    def INPUT_TYPES(cls):
        sliders = {"image": ("IMAGE",)}
        optional = {}

        for hue_name in HUE_CENTERS:
            cap = hue_name.capitalize()
            optional[f"{hue_name}_hue"] = ("FLOAT", {
                "default": 0.0, "min": -30.0, "max": 30.0, "step": 1.0,
                "tooltip": f"Shift {cap} hues (degrees)"
            })
            optional[f"{hue_name}_saturation"] = ("FLOAT", {
                "default": 0.0, "min": -100.0, "max": 100.0, "step": 1.0,
                "tooltip": f"{cap} saturation adjustment"
            })
            optional[f"{hue_name}_luminance"] = ("FLOAT", {
                "default": 0.0, "min": -100.0, "max": 100.0, "step": 1.0,
                "tooltip": f"{cap} luminance adjustment"
            })

        optional["strength"] = ("FLOAT", {
            "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05,
            "tooltip": "Blend between original (0) and adjusted (1)"
        })
        optional["preset"] = (HSL_PRESET_NAMES, {
            "default": "Custom (manual)",
            "tooltip": "AI Mitigation presets pair with Clarity/Texture/Dehaze preset of the same tier. Manual sliders add on top"
        })

        return {"required": sliders, "optional": optional}

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "execute"
    CATEGORY = "AKURATE/Darkroom/Raw"

    def execute(self, image, strength=1.0, preset="Custom (manual)", **kwargs):
        if strength <= 0.0:
            return (image,)

        if preset != "Custom (manual)" and preset in AI_MITIGATION_HSL:
            for k, v in asdict(AI_MITIGATION_HSL[preset]).items():
                if abs(v) > 0.01:
                    kwargs[k] = kwargs.get(k, 0.0) + v

        # Collect active adjustments
        adjustments = []
        for hue_name, center in HUE_CENTERS.items():
            h_shift = kwargs.get(f"{hue_name}_hue", 0.0)
            s_adj = kwargs.get(f"{hue_name}_saturation", 0.0)
            l_adj = kwargs.get(f"{hue_name}_luminance", 0.0)
            if abs(h_shift) > 0.1 or abs(s_adj) > 0.5 or abs(l_adj) > 0.5:
                adjustments.append((center, h_shift, s_adj, l_adj))

        if not adjustments:
            return (image,)

        def grade(img):
            linear = srgb_to_linear(img)

            h, s, l = rgb_to_hsl(linear)

            for center, h_shift, s_adj, l_adj in adjustments:
                weight = _hue_weight(h, center)

                if abs(h_shift) > 0.1:
                    h = h + weight * h_shift
                    h = h % 360.0

                if abs(s_adj) > 0.5:
                    s = s * (1.0 + weight * (s_adj / 100.0))
                    s = s.clamp(0.0, 1.0)

                if abs(l_adj) > 0.5:
                    l = l * (1.0 + weight * (l_adj / 100.0))
                    l = l.clamp(0.0, 1.0)

            result = hsl_to_rgb(h, s, l)
            result = linear_to_srgb(result)
            return blend(img, result, strength)

        return (run_on_device(grade, image),)


NODE_CLASS_MAPPINGS = {"DarkroomHSLSelective": HSLSelective}
NODE_DISPLAY_NAME_MAPPINGS = {"DarkroomHSLSelective": "HSL Selective"}
