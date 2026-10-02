"""
Color Space Transform node for ComfyUI-Darkroom.
Convert between sRGB, Linear sRGB, ACEScg, ACEScct, Rec.2020, and DCI-P3.
Makes Darkroom the only ACES-aware toolset in ComfyUI.
"""

import torch

from ..utils.colorspace import SPACE_NAMES
from ..utils.gpu_color import run_on_device, blend, convert_colorspace


class ColorSpaceTransform:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "source_space": (SPACE_NAMES, {
                    "default": "sRGB",
                    "tooltip": "Color space of the input image"
                }),
                "target_space": (SPACE_NAMES, {
                    "default": "ACEScg",
                    "tooltip": "Color space for the output image"
                }),
            },
            "optional": {
                "gamut_clip": (["Clip", "Soft Compress"], {
                    "default": "Clip",
                    "tooltip": "How to handle out-of-gamut values. Clip = hard clamp to [0,1]. "
                               "Soft Compress = gently roll off values approaching gamut boundary"
                }),
                "strength": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Blend between original (0) and transformed (1)"
                }),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "execute"
    CATEGORY = "AKURATE/Darkroom/Pipeline"

    def execute(self, image, source_space="sRGB", target_space="ACEScg",
                gamut_clip="Clip", strength=1.0):

        if strength <= 0.0 or source_space == target_space:
            return (image,)

        print(f"[Darkroom] Color Space Transform: {source_space} → {target_space}")

        def process(img):
            original = img
            converted = convert_colorspace(img, source_space, target_space)

            # Gamut handling
            if gamut_clip == "Soft Compress":
                # Soft compression: smoothly map values outside [0,1] back in
                # Using a simple knee function at the boundaries
                converted = self._soft_compress(converted)
            else:
                converted = converted.clamp(0.0, 1.0)

            return blend(original, converted, strength)

        return (run_on_device(process, image),)

    @staticmethod
    def _soft_compress(img, knee=0.9):
        """
        Soft-compress values outside [0, 1] using a smooth knee.
        Values below knee/above (1-knee) pass through linearly.
        Values beyond are compressed asymptotically toward the boundary.
        """
        result = img

        # Compress highlights (values above knee toward 1.0)
        excess = result - knee
        compressed = knee + (1.0 - knee) * (1.0 - torch.exp(-excess / (1.0 - knee + 1e-10)))
        result = torch.where(result > knee, compressed, result)

        # Compress shadows (values below 1-knee toward 0.0)
        neg_knee = 1.0 - knee  # 0.1 for knee=0.9
        deficit = neg_knee - result
        compressed = neg_knee - neg_knee * (1.0 - torch.exp(-deficit / (neg_knee + 1e-10)))
        result = torch.where(result < neg_knee, compressed, result)

        return result.clamp(0.0, 1.0)


NODE_CLASS_MAPPINGS = {"DarkroomColorSpaceTransform": ColorSpaceTransform}
NODE_DISPLAY_NAME_MAPPINGS = {"DarkroomColorSpaceTransform": "Color Space Transform"}
