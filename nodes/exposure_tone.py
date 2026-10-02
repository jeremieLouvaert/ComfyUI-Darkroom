"""
Exposure & Tone node for ComfyUI-Darkroom.
Lightroom-style tonal controls with parametric luminance masks.
"""

import torch

from ..utils.gpu_color import run_on_device, srgb_to_linear, linear_to_srgb, luminance_rec709, blend


def parametric_tone_mask(luminance, center, sigma):
    """Torch twin of utils.raw.parametric_tone_mask (Gaussian bell in luminance)."""
    return torch.exp(-0.5 * ((luminance - center) / (sigma + 1e-10)) ** 2)


class ExposureTone:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "exposure": ("FLOAT", {
                    "default": 0.0, "min": -5.0, "max": 5.0, "step": 0.1,
                    "tooltip": "Exposure in EV stops. +1 = double brightness"
                }),
                "contrast": ("FLOAT", {
                    "default": 0.0, "min": -100.0, "max": 100.0, "step": 1.0,
                    "tooltip": "Midtone contrast around 18% grey"
                }),
                "highlights": ("FLOAT", {
                    "default": 0.0, "min": -100.0, "max": 100.0, "step": 1.0,
                    "tooltip": "Recover or boost highlights (top ~25% of tonal range)"
                }),
                "shadows": ("FLOAT", {
                    "default": 0.0, "min": -100.0, "max": 100.0, "step": 1.0,
                    "tooltip": "Open up or crush shadows (bottom ~25% of tonal range)"
                }),
                "whites": ("FLOAT", {
                    "default": 0.0, "min": -100.0, "max": 100.0, "step": 1.0,
                    "tooltip": "White clipping point adjustment (top ~10%)"
                }),
                "blacks": ("FLOAT", {
                    "default": 0.0, "min": -100.0, "max": 100.0, "step": 1.0,
                    "tooltip": "Black clipping point adjustment (bottom ~10%)"
                }),
            },
            "optional": {
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

    def execute(self, image, exposure=0.0, contrast=0.0, highlights=0.0,
                shadows=0.0, whites=0.0, blacks=0.0, strength=1.0):
        # Skip if all sliders at zero
        if (strength <= 0.0 or
            (abs(exposure) < 0.01 and abs(contrast) < 0.5 and
             abs(highlights) < 0.5 and abs(shadows) < 0.5 and
             abs(whites) < 0.5 and abs(blacks) < 0.5)):
            return (image,)

        def process(img):
            original = img
            linear = srgb_to_linear(img)

            # 1. Exposure: multiply by 2^EV (photometrically correct)
            if abs(exposure) > 0.01:
                linear = linear * (2.0 ** exposure)

            # 2. Contrast: S-curve around 18% grey
            if abs(contrast) > 0.5:
                pivot = 0.18
                c = contrast / 100.0
                # Power-based contrast: steepen or flatten around pivot
                ratio = linear / (pivot + 1e-10)
                power = 1.0 + c * 0.5  # contrast/100 * 0.5 gives subtle-to-strong range
                linear = pivot * torch.pow(ratio.clamp(min=1e-10), power)

            # Get luminance for parametric masks
            lum = luminance_rec709(linear.clamp(0.0, 1.0))

            # 3. Parametric tone adjustments
            # Each slider modifies the image proportionally within its tonal range
            if abs(highlights) > 0.5:
                mask = parametric_tone_mask(lum, center=0.75, sigma=0.15)
                adj = highlights / 100.0
                linear = linear * (1.0 + mask[..., None] * adj * 0.5)

            if abs(shadows) > 0.5:
                mask = parametric_tone_mask(lum, center=0.25, sigma=0.15)
                adj = shadows / 100.0
                # Shadows use additive lift to open up without color shifts
                linear = linear + mask[..., None] * adj * 0.15

            if abs(whites) > 0.5:
                mask = parametric_tone_mask(lum, center=0.90, sigma=0.08)
                adj = whites / 100.0
                linear = linear * (1.0 + mask[..., None] * adj * 0.3)

            if abs(blacks) > 0.5:
                mask = parametric_tone_mask(lum, center=0.10, sigma=0.08)
                adj = blacks / 100.0
                linear = linear + mask[..., None] * adj * 0.10

            result = linear_to_srgb(linear.clamp(0.0, 1.0))
            return blend(original, result, strength)

        return (run_on_device(process, image),)


NODE_CLASS_MAPPINGS = {"DarkroomExposureTone": ExposureTone}
NODE_DISPLAY_NAME_MAPPINGS = {"DarkroomExposureTone": "Exposure & Tone"}
