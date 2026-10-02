"""
OkLab Color node for ComfyUI-Darkroom.
Perceptually-uniform grading in Björn Ottosson's OkLab / OkLch.
Lightness and contrast hold hue and chroma; chroma is even across the wheel —
"expensive colorist" behavior, correct by construction (see docs/oklab-color-derivation.md).
"""

import numpy as np
import torch

from ..utils.gpu_color import (
    run_on_device,
    srgb_to_linear,
    linear_to_srgb,
    blend,
    linear_srgb_to_oklab,
    oklab_to_linear_srgb,
    oklab_to_oklch,
    oklch_to_oklab,
)


class OkLabColor:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
            },
            "optional": {
                "lightness": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01,
                    "tooltip": "Perceptual lightness multiplier (L *= lightness). Holds hue and chroma."
                }),
                "contrast": ("FLOAT", {
                    "default": 0.0, "min": -1.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Perceptual contrast around mid-grey. slope = 2**contrast; 0 = identity, -1 flatten, +1 punch."
                }),
                "chroma": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01,
                    "tooltip": "Chroma (colorfulness) multiplier, even across all hues. 1 = unchanged, 0 = greyscale."
                }),
                "hue": ("FLOAT", {
                    "default": 0.0, "min": -180.0, "max": 180.0, "step": 1.0,
                    "tooltip": "Global hue rotation in degrees. Holds lightness and chroma."
                }),
                "tint_a": ("FLOAT", {
                    "default": 0.0, "min": -0.1, "max": 0.1, "step": 0.005,
                    "tooltip": "Tint along the green↔red axis (a offset). Small range — ±0.1 is a strong cast."
                }),
                "tint_b": ("FLOAT", {
                    "default": 0.0, "min": -0.1, "max": 0.1, "step": 0.005,
                    "tooltip": "Tint along the blue↔yellow axis (b offset). Small range — ±0.1 is a strong cast."
                }),
                "strength": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Blend between original (0) and graded (1)"
                }),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "execute"
    CATEGORY = "AKURATE/Darkroom/Grading"

    def execute(self, image, lightness=1.0, contrast=0.0, chroma=1.0, hue=0.0,
                tint_a=0.0, tint_b=0.0, strength=1.0):

        if strength <= 0.0:
            return (image,)

        # Identity check — every control at its no-op value
        is_identity = (
            np.allclose(lightness, 1.0, atol=1e-4) and
            np.allclose(contrast, 0.0, atol=1e-4) and
            np.allclose(chroma, 1.0, atol=1e-4) and
            np.allclose(hue, 0.0, atol=1e-4) and
            np.allclose(tint_a, 0.0, atol=1e-4) and
            np.allclose(tint_b, 0.0, atol=1e-4)
        )
        if is_identity:
            return (image,)

        print(f"[Darkroom] OkLab Color: lightness={lightness}, contrast={contrast}, "
              f"chroma={chroma}, hue={hue}, tint_a={tint_a}, tint_b={tint_b}, strength={strength}")

        contrast_slope = 2.0 ** contrast
        hue_rad = float(np.radians(hue).astype(np.float32))

        def process(img):
            original = img
            lab = linear_srgb_to_oklab(srgb_to_linear(img))

            # tone (L only)
            L = lab[..., 0] * lightness
            L = 0.5 + (L - 0.5) * contrast_slope
            lab = torch.stack([L, lab[..., 1], lab[..., 2]], dim=-1)

            # color (C / h only)
            lch = oklab_to_oklch(lab)
            lch = torch.stack([lch[..., 0], lch[..., 1] * chroma, lch[..., 2] + hue_rad], dim=-1)
            lab = oklch_to_oklab(lch)

            # tint (a, b offset: global cast, last)
            lab = torch.stack([lab[..., 0], lab[..., 1] + tint_a, lab[..., 2] + tint_b], dim=-1)

            lin2 = oklab_to_linear_srgb(lab).clamp(0.0, 1.0)
            return blend(original, linear_to_srgb(lin2), strength)

        return (run_on_device(process, image),)


NODE_CLASS_MAPPINGS = {"DarkroomOkLabColor": OkLabColor}
NODE_DISPLAY_NAME_MAPPINGS = {"DarkroomOkLabColor": "OkLab Color"}
