"""
White Balance node for ComfyUI-Darkroom.
Adjusts color temperature and tint in linear light.
"""

import numpy as np
import torch

from ..utils.gpu_color import run_on_device, srgb_to_linear, linear_to_srgb, blend
from ..utils.raw import kelvin_to_rgb


class WhiteBalance:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "temperature": ("INT", {
                    "default": 6500, "min": 2000, "max": 12000, "step": 100,
                    "tooltip": "Color temperature in Kelvin. Lower=warmer, higher=cooler"
                }),
                "tint": ("FLOAT", {
                    "default": 0.0, "min": -1.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Green-magenta tint. Negative=green, positive=magenta"
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
    CATEGORY = "AKURATE/Darkroom/Raw"

    def execute(self, image, temperature=6500, tint=0.0, strength=1.0):
        if strength <= 0.0 and temperature == 6500 and abs(tint) < 0.01:
            return (image,)

        # Compute channel multipliers from temperature shift
        source_rgb = np.array(kelvin_to_rgb(6500), dtype=np.float32)
        target_rgb = np.array(kelvin_to_rgb(temperature), dtype=np.float32)

        # Multipliers: ratio of source to target white point
        multipliers = source_rgb / (target_rgb + 1e-10)

        # Apply tint (green-magenta axis, orthogonal to temperature)
        multipliers[0] *= (1.0 + tint * 0.05)   # Red
        multipliers[1] *= (1.0 - tint * 0.10)   # Green (primary axis)
        multipliers[2] *= (1.0 + tint * 0.05)   # Blue

        # Normalize so max multiplier = 1.0 (prevent clipping, just shift ratios)
        multipliers /= multipliers.max()

        m_r, m_g, m_b = (float(m) for m in multipliers)

        def process(img):
            original = img
            linear = srgb_to_linear(img)

            # Apply white balance multipliers
            linear = torch.stack([linear[..., 0] * m_r,
                                  linear[..., 1] * m_g,
                                  linear[..., 2] * m_b], dim=-1)

            result = linear_to_srgb(linear.clamp(0.0, 1.0))
            return blend(original, result, strength)

        return (run_on_device(process, image),)


NODE_CLASS_MAPPINGS = {"DarkroomWhiteBalance": WhiteBalance}
NODE_DISPLAY_NAME_MAPPINGS = {"DarkroomWhiteBalance": "White Balance"}
