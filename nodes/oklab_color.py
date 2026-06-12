"""
OkLab Color node for ComfyUI-Darkroom.
Perceptually-uniform grading in Björn Ottosson's OkLab / OkLch.
Lightness and contrast hold hue and chroma; chroma is even across the wheel —
"expensive colorist" behavior, correct by construction (see docs/oklab-color-derivation.md).
"""

import numpy as np

from ..utils.color import srgb_to_linear, linear_to_srgb, blend
from ..utils.image import tensor_to_numpy_batch, numpy_batch_to_tensor
from ..utils.colorspace import (
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
        hue_rad = np.radians(hue).astype(np.float32)

        images = tensor_to_numpy_batch(image)
        results = []

        for img in images:
            original = img.copy()
            lin = srgb_to_linear(img)
            lab = linear_srgb_to_oklab(lin)

            # tone (L only)
            lab[..., 0] = lab[..., 0] * lightness
            lab[..., 0] = 0.5 + (lab[..., 0] - 0.5) * contrast_slope

            # color (C / h only)
            lch = oklab_to_oklch(lab)
            lch[..., 1] = lch[..., 1] * chroma
            lch[..., 2] = lch[..., 2] + hue_rad
            lab = oklch_to_oklab(lch)

            # tint (a, b offset — global cast, last)
            lab[..., 1] = lab[..., 1] + tint_a
            lab[..., 2] = lab[..., 2] + tint_b

            lin2 = oklab_to_linear_srgb(lab)
            lin2 = np.clip(lin2, 0.0, 1.0)
            out = linear_to_srgb(lin2)
            results.append(blend(original, out, strength))

        return (numpy_batch_to_tensor(results),)


NODE_CLASS_MAPPINGS = {"DarkroomOkLabColor": OkLabColor}
NODE_DISPLAY_NAME_MAPPINGS = {"DarkroomOkLabColor": "OkLab Color"}
