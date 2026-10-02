"""
Vibrance node for ComfyUI-Darkroom.
Intelligent saturation that protects already-saturated colors and skin tones.
"""

import torch

from ..utils.gpu_color import (
    run_on_device, srgb_to_linear, linear_to_srgb, luminance_rec709,
    adjust_saturation, blend, rgb_to_hsl,
)


class Vibrance:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "vibrance": ("FLOAT", {
                    "default": 0.0, "min": -100.0, "max": 100.0, "step": 1.0,
                    "tooltip": "Intelligent saturation. Protects skin tones and already-saturated colors"
                }),
            },
            "optional": {
                "saturation": ("FLOAT", {
                    "default": 0.0, "min": -100.0, "max": 100.0, "step": 1.0,
                    "tooltip": "Uniform saturation boost (no protection). Use vibrance for natural results"
                }),
                "protect_skin": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Reduce vibrance effect on skin-tone hues (oranges/warm yellows)"
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

    def execute(self, image, vibrance=0.0, saturation=0.0, protect_skin=True, strength=1.0):
        if strength <= 0.0 or (abs(vibrance) < 0.5 and abs(saturation) < 0.5):
            return (image,)

        def process(img):
            original = img
            result = srgb_to_linear(img)

            # Vibrance: saturation-aware boost
            if abs(vibrance) > 0.5:
                r, g, b = result[..., 0], result[..., 1], result[..., 2]
                lum = luminance_rec709(result)

                # Per-pixel chroma (how saturated each pixel already is)
                cmax = torch.maximum(torch.maximum(r, g), b)
                cmin = torch.minimum(torch.minimum(r, g), b)
                chroma = cmax - cmin

                # Weight: less saturated pixels get more boost
                weight = 1.0 - (chroma * 2.0).clamp(0.0, 1.0)

                # Skin tone protection
                if protect_skin:
                    hue = rgb_to_hsl(result)[0]  # same hue maths as the old _fast_hue
                    # Skin tones: ~15-45 degrees (orange/warm yellow)
                    skin_center = 30.0
                    skin_width = 30.0
                    diff = torch.abs(hue - skin_center)
                    diff = torch.minimum(diff, 360.0 - diff)
                    skin_mask = ((1.0 + torch.cos(torch.pi * diff / skin_width)) * 0.5).clamp(0.0, 1.0)
                    skin_mask = torch.where(diff > skin_width, torch.zeros_like(skin_mask), skin_mask)
                    weight = weight * (1.0 - 0.7 * skin_mask)

                # Apply vibrance
                vib_factor = 1.0 + (vibrance / 100.0) * weight
                result = lum[..., None] + vib_factor[..., None] * (result - lum[..., None])

            # Uniform saturation (no protection)
            if abs(saturation) > 0.5:
                result = adjust_saturation(result, 1.0 + saturation / 100.0)

            result = linear_to_srgb(result.clamp(0.0, 1.0))
            return blend(original, result, strength)

        return (run_on_device(process, image),)


NODE_CLASS_MAPPINGS = {"DarkroomVibrance": Vibrance}
NODE_DISPLAY_NAME_MAPPINGS = {"DarkroomVibrance": "Vibrance"}
