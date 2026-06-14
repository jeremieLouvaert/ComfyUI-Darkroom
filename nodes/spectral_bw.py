"""
Spectral B&W (Ortho/Pan) node for ComfyUI-Darkroom.

Black-and-white conversion driven by a film's SPECTRAL SENSITIVITY type:
orthochromatic renders red dark and blue light (white skies, dark skin/lips),
panchromatic renders naturally, extended-red (pan+) lightens foliage/skin.

The per-type RGB->gray weights are DERIVED OFFLINE (tools/derive_spectral_bw_weights.py)
by integrating an analytic sensitivity curve S(lambda) against the Mallett & Yuksel
2019 sRGB spectral basis. Because spectral upsampling is linear, the whole spectral
computation collapses to a fixed RGB weight triple per type — a principled (spectrally
derived) channel mixer. Runtime here is a single weighted sum; no spectral lib needed.

HONEST LIMIT: a true B&W weight is integral(S*I); sRGB gives 3 numbers per pixel, not
the spectrum I(lambda). This is the spectral-upsampling approximation every spectral
renderer makes — ortho/pan TONAL CHARACTER, not a claim of reproducing what the film
did to a real-world spectrum. Pan+ is the most stylized (sRGB carries no IR).
"""

import numpy as np

from ..utils.color import srgb_to_linear, linear_to_srgb, blend
from ..utils.image import tensor_to_numpy_batch, numpy_batch_to_tensor


# Per-type RGB->gray weights, each sums to 1. DERIVED OFFLINE — see
# tools/derive_spectral_bw_weights.py (Mallett-Yuksel 2019 sRGB basis). Do not
# hand-edit; regenerate with the script.
_WEIGHTS = {
    "Blue-sensitive": (0.034845, 0.078376, 0.886779),
    "Orthochromatic": (0.020890, 0.491945, 0.487165),
    "Orthopanchromatic": (0.162394, 0.453700, 0.383906),
    "Panchromatic": (0.284952, 0.389573, 0.325476),
    "Panchromatic+": (0.382438, 0.338749, 0.278813),
}

_SENSITIVITY_NAMES = list(_WEIGHTS.keys())


class SpectralBW:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "sensitivity": (_SENSITIVITY_NAMES, {
                    "default": "Panchromatic",
                    "tooltip": "Film's spectral 'eye'. Orthochromatic darkens red / "
                               "lightens blue (white skies); Panchromatic is natural; "
                               "Panchromatic+ is pseudo-IR (reds/foliage/skin light)."
                }),
                "strength": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Blend color->B&W. 1 = full B&W, <1 = partial desaturation."
                }),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "execute"
    CATEGORY = "AKURATE/Darkroom/Film"

    def execute(self, image, sensitivity="Panchromatic", strength=1.0):
        if strength <= 0.0:
            return (image,)

        w_r, w_g, w_b = _WEIGHTS[sensitivity]
        print(f"[Darkroom] Spectral B&W: sensitivity={sensitivity}, "
              f"weights=({w_r:.4f}, {w_g:.4f}, {w_b:.4f}), strength={strength}")

        images = tensor_to_numpy_batch(image)
        results = []

        for img in images:
            original = img.copy()
            linear = srgb_to_linear(img)

            # Spectrally-derived weighted sum in linear light (clip >= 0).
            gray = w_r * linear[..., 0] + w_g * linear[..., 1] + w_b * linear[..., 2]
            gray = np.clip(gray, 0.0, None)

            out = np.stack([gray, gray, gray], axis=-1)
            out = linear_to_srgb(np.clip(out, 0.0, 1.0))
            results.append(blend(original, out, strength))

        return (numpy_batch_to_tensor(results),)


NODE_CLASS_MAPPINGS = {"DarkroomSpectralBW": SpectralBW}
NODE_DISPLAY_NAME_MAPPINGS = {"DarkroomSpectralBW": "Spectral B&W (Ortho/Pan)"}
